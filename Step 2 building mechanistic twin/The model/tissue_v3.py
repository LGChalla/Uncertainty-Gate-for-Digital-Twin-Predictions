"""
A batch of tissues, version 3. tissue.py and tissue_v2.py are unchanged.

One change from v2: fibroblast contact inhibition counts lineage neighbours
over its own disc, fib_sense_radius_um (120 um, 112 neighbours), instead of
the 60 um disc (28 neighbours) shared by the tumour, M2 and CD4 rules. Those
rules keep the 60 um disc. Both the fibroblast division rule (each hour and
in pre-equilibration) and its analytic steady state (_fib_density) use the
new neighbour count, so the tissue still starts at its steady state.
Reason and evidence: config_v3.py docstring and corrections log.

Everything below the original v2 docstring is identical to tissue_v2.py except
those lines (marked "v3").

--- v2 docstring ---

What v2 adds
------------
Every stromal and immune population now has its own source and sink
(cells_v2), CD4 T cells and B cells are present (grid_v2), and each population
starts at its analytic steady state rather than at a guessed fraction.

Why start at steady state. A population fed by recruitment R and lost at net
rate (death - division) settles at N* = R / (death - division) with time
constant 1 / (death - division). For the slower ones that is months:
macrophages with a 30-day dwell need ~90 days to reach 95%, memory B cells
~77 days, fibroblasts turning over at 0.005/day ~600 days. A 20-day burn-in
cannot get there, so each population is placed at N* when the tissue is built
and the burn-in only settles the spatial layout and the local polarisation.

The steady states (mean field, solved to a fixed point with crowding):
  succ = 1 - phi, the chance a neighbouring square is free (phi counts every
  cell and vessel); recruits and daughters land only on free squares
  macrophages   N = r_mac * n_vessels * succ / (d_mac - p_mac * succ)
                M2 share = a / (a + d_mac + k_m2_rev), a = k_m2 * f_tumour
  CD8, CD4, B   N = r * n_vessels * succ / (d * (1 - s * succ)),  s = division / death
  fibroblasts   density rho where the mean room seen by a fibroblast equals
                d / (p_bar * succ); room = max(0, 1 - c / (28 K_fib)) for c
                lineage neighbours, c ~ Binomial(28, rho) (_fib_density)
                CAF share x solves  b(1-x) - rev x + d m x / (m x + 1 - x) - d x = 0
                with b = k_caf * mean TGF-beta, m = CAF division multiplier
  mean TGF-beta = c0 + r_tgfb * (M2 per square) / lambda_tgfb

Checked against simulation (tests/test_population_v2.py, and 60-day runs on
Nov 5): every population holds within noise of these values, fibroblasts
within 2% (drift +0.06%/day). Four fixes were needed to get there: CAF
reversion, the crowding term, fibroblast movement with division > death, and
counting neighbours only with the binomial expectation (corrections log 18).

If the populations would fill more than MAX_FILL of the free squares, every
non-tumour population is scaled down by the same factor and the tissue is
flagged in `crowded_init`.

Order of events each hour
-------------------------
  fields: oxygen, IL-6 and TGF-beta, drug exposure (as v1)
  state changes: M0 -> M2, fibroblast -> CAF, CAF -> fibroblast; M2 -> M0 (0 in main analysis)
  tumour: drug kill, background and necrotic death, CD8 killing (with the net
          CD4 effect)
  stroma and immune: death, then division, then recruitment at vessels
  movement: lymphoid cells, then myeloid cells, then fibroblasts and CAFs
"""

import math

import torch

from .config_v3 import TME_V3                                    # v3
from .config import values
from . import grid_v2 as G
from . import cells as C
from . import cells_v2 as C2
from .tissue import Tissue
from .fields import implicit_step                                   # noqa: F401

MAX_FILL = 0.95
TRACKED = ("tumour", "M0", "M2", "fibroblast", "CAF", "CD8", "CD4", "B")


def _fib_density(K, target_room, n_nb, iters=60):
    """
    Lineage density rho (share of all squares) at which the mean room factor
    seen by a fibroblast equals target_room = death / (division * success).

    Room for a cell with c lineage neighbours out of n_nb is max(0, 1 - c / (n_nb K)).
    Under good mixing c ~ Binomial(n_nb, rho). The clamp at 0 matters: with only
    28 neighbours the count is lumpy, and the linear mean-field 1 - rho/K
    underestimated the mean room (0.20 against 0.33 measured, Nov 5), which put
    the predicted density ~15% too low. Mean room falls monotonically in rho,
    so bisect. target_room >= 1 means the lineage cannot persist: rho = 0.
    """
    K = K.reshape(-1, 1)
    target = target_room.reshape(-1, 1)
    c = torch.arange(n_nb + 1, dtype=torch.float64, device=K.device).reshape(1, -1)
    room_c = (1.0 - c / (n_nb * K)).clamp(min=0.0)
    logC = (torch.lgamma(torch.tensor(n_nb + 1.0)) - torch.lgamma(c + 1) - torch.lgamma(n_nb - c + 1))
    def mean_room(rho):
        r = rho.clamp(1e-12, 1 - 1e-12)
        logp = logC + c * torch.log(r) + (n_nb - c) * torch.log1p(-r)
        return (torch.exp(logp) * room_c).sum(1, keepdim=True)
    lo = torch.zeros_like(K)
    hi = torch.ones_like(K)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = mean_room(mid) > target          # still room to grow: rho* is higher
        lo = torch.where(above, mid, lo)
        hi = torch.where(above, hi, mid)
    rho = 0.5 * (lo + hi)
    return torch.where(target >= 1.0, torch.zeros_like(rho), rho).reshape(-1)


def _caf_share(b, rev, d, m, iters=60):
    """Solve the fibroblast-lineage CAF share by bisection (vectorised)."""
    lo = torch.zeros_like(b)
    hi = torch.ones_like(b)
    def f(x):
        return b * (1 - x) - rev * x + d * m * x / (m * x + 1 - x).clamp(min=1e-12) - d * x
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = f(mid) > 0
        lo = torch.where(pos, mid, lo)
        hi = torch.where(pos, hi, mid)
    return 0.5 * (lo + hi)


class TissueV3(Tissue):
    """
    f0_tumour: starting tumour fraction of free squares, scalar or (B,).
    overrides: any TME_V3 constant, as a scalar or a (B,) tensor.
    """

    def __init__(self, B, f0_tumour, seed, device="cpu", overrides=None, cfg=TME_V3):
        self.v = values(cfg)
        if overrides:
            self.v.update(overrides)
        v = self.v
        self.B, self.device = B, device
        self.ny, self.nx, self.dx = int(v["ny"]), int(v["nx"]), v["dx"]
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.seed = seed
        self.hours = 0

        self.vessels = G.make_vessels(B, self.ny, self.nx, self.dx,
                                      self.pb(v["vessel_density"]).flatten(), self.gen, device)
        r_fib = max(1, round(v["fib_sense_radius_um"] * 1e-4 / self.dx))                # v3
        self.n_nb = int(C.disc_kernel(r_fib).sum().item()) - 1   # v3: fibroblast neighbours, self excluded
        self.fractions0, self.crowded_init = self.steady_state_fractions(f0_tumour)
        self.kind = G.place_cells(self.vessels, self.fractions0, self.gen)

        f64 = dict(dtype=torch.float64, device=device)
        self.il6 = torch.full((B, self.ny, self.nx), v["c0_il6"], **f64)
        # TGF-beta starts at its mean-field level, not at background, so the
        # fibroblast lineage pre-equilibrates against the right activation rate.
        self.tgfb = torch.ones((B, self.ny, self.nx), **f64) * self.analytic["tgfb_mean"].reshape(-1, 1, 1)
        self.o2 = torch.zeros((B, self.ny, self.nx), **f64) + self.pb(v["o_vessel"])
        self.kernel = C.disc_kernel(max(1, round(v["neighbor_radius_um"] * 1e-4 / self.dx)), device)
        self.fib_kernel = C.disc_kernel(r_fib, device)                                  # v3
        self.events = {}
        self.drug = None
        self.preeq = self._preequilibrate_fibroblasts(int(v["fib_preeq_days"]), float(v["fib_preeq_speedup"]))
        self.solve_oxygen()

    def _preequilibrate_fibroblasts(self, days, speedup):
        """Run the fibroblast lineage alone, every rate x speedup, for `days`.
        Only death, division, activation and reversion of fibroblasts and CAFs;
        nothing else moves and no field is solved. Returns lineage counts and
        CAF share before and after."""
        if days <= 0:
            return {}
        v, pb, g, dt = self.v, self.pb, self.gen, self.v["dt_agent"]
        def lineage_stats(k):
            fl = ((k == G.FIB) | (k == G.CAF)).sum((1, 2)).double()
            caf = (k == G.CAF).sum((1, 2)).double()
            return fl, caf / fl.clamp(min=1)
        k = self.kind
        n0, s0 = lineage_stats(k)
        # Cap the speedup per tissue so no rate exceeds 0.1 events per step;
        # beyond that 1 - exp(-rate dt) saturates and distorts the rate ratios
        # the equilibrium depends on.
        fastest = torch.stack([
            pb(v["u_fib"]) * pb(v["d_fib"]) * pb(v["m_caf"]),
            pb(v["k_caf"]) * self.tgfb.amax((1, 2), keepdim=True),
            pb(v["k_caf_rev"]) * torch.ones_like(pb(v["d_fib"])),
            pb(v["d_fib"])]).amax(0)
        S = torch.minimum(torch.full_like(fastest, speedup), 0.1 / (fastest * dt).clamp(min=1e-30))
        self.preeq_speedup = S.flatten()
        for _ in range(24 * days):
            k, _ = C.activate_fibroblasts(k, self.tgfb, pb(v["k_caf"]) * S, dt, g)
            if _any_positive(v["k_caf_rev"]):
                k, _ = C2.revert(k, G.CAF, G.FIB, pb(v["k_caf_rev"]) * S, dt, g)
            k, _ = C2.die(k, (G.FIB, G.CAF), pb(v["d_fib"]) * S, dt, g)
            k, _ = _fibroblast_divide_logistic(k, self.fib_kernel, pb(v["K_fib"]),   # v3
                                               pb(v["u_fib"]) * pb(v["d_fib"]) * S,
                                               pb(v["m_caf"]), dt, g)
        self.kind = k
        n1, s1 = lineage_stats(k)
        return {"lineage_before": n0.tolist(), "lineage_after": n1.tolist(),
                "caf_share_before": s0.tolist(), "caf_share_after": s1.tolist(),
                "speedup": S.flatten().tolist(),
                "effective_days": (S.flatten() * days).tolist()}

    # per-tissue parameter as a (B,) tensor
    def pv(self, name):
        t = torch.as_tensor(self.v[name], dtype=torch.float64, device=self.device)
        return t.reshape(-1).expand(self.B) if t.dim() <= 1 else t.reshape(self.B)

    # ---------------- starting composition ----------------
    def steady_state_fractions(self, f0_tumour):
        v, pv = self.v, self.pv
        n_sq = float(self.ny * self.nx)
        n_v = self.vessels.view(self.B, -1).sum(1).double()
        n_free = n_sq - n_v
        f_tum = torch.as_tensor(f0_tumour, dtype=torch.float64, device=self.device).reshape(-1).expand(self.B)

        a = pv("k_m2") * f_tum
        m2_share = a / (a + pv("d_mac") + pv("k_m2_rev"))
        n_tum = f_tum * n_free

        # Crowding. A recruit or a daughter goes to one random neighbouring
        # square and fails if it is occupied, so both are scaled by the chance
        # a square is free, 1 - phi, where phi counts every cell and vessel.
        # phi depends on the populations, so iterate to a fixed point (damped).
        phi = (n_v + n_tum) / n_sq
        caf_share = torch.zeros_like(phi)
        for _ in range(40):
            succ = (1.0 - phi).clamp(min=1e-3)
            n_mac = pv("r_mac") * n_v * succ / (pv("d_mac") - pv("p_mac") * succ).clamp(min=1e-12)
            n_cd8 = pv("r_cd8") * n_v * succ / (pv("d_cd8") * (1 - pv("s_cd8") * succ)).clamp(min=1e-12)
            n_cd4 = pv("r_cd4") * n_v * succ / (pv("d_cd4") * (1 - pv("s_cd4") * succ)).clamp(min=1e-12)
            n_b = pv("r_b") * n_v * succ / (pv("d_b") * (1 - pv("s_b") * succ)).clamp(min=1e-12)
            # Logistic contact inhibition: births p_bar (1 - rho/K) succ = deaths d,
            # so rho* = K (1 - d / (p_bar succ)), with p_bar the lineage's mean
            # division rate given its CAF share. rho is a share of all squares.
            p_bar = pv("u_fib") * pv("d_fib") * (1 + (pv("m_caf") - 1) * caf_share)
            rho = _fib_density(pv("K_fib"), pv("d_fib") / (p_bar * succ), self.n_nb)
            n_fib = rho * n_sq
            tgfb = v["c0_tgfb"] + pv("r_tgfb") * (n_mac * m2_share / n_sq) / v["lam_tgfb"]
            caf_share = _caf_share(pv("k_caf") * tgfb, pv("k_caf_rev"), pv("d_fib"), pv("m_caf"))
            new_phi = (n_v + n_tum + n_mac + n_fib + n_cd8 + n_cd4 + n_b) / n_sq
            phi = 0.5 * phi + 0.5 * new_phi.clamp(max=MAX_FILL)

        stroma = torch.stack([n_mac * (1 - m2_share), n_mac * m2_share,
                              n_fib * (1 - caf_share), n_fib * caf_share,
                              n_cd8, n_cd4, n_b], 1) / n_free[:, None]
        room = (MAX_FILL - f_tum).clamp(min=0)
        total = stroma.sum(1)
        scale = torch.where(total > room, room / total.clamp(min=1e-12), torch.ones_like(total))
        stroma = stroma * scale[:, None]
        fr = torch.cat([f_tum[:, None], stroma], 1)
        self.analytic = {"n_mac": n_mac, "m2_share": m2_share, "n_cd8": n_cd8, "n_cd4": n_cd4,
                         "n_b": n_b, "n_fib": n_fib, "caf_share": caf_share, "tgfb_mean": tgfb,
                         "phi": phi, "init_scale": scale}
        return fr, (scale < 1.0)

    # ---------------- one hour ----------------
    def step_hour(self, cells=True):
        v, dt, pb = self.v, self.v["dt_agent"], self.pb
        if self.hours % int(v["o2_every_h"]) == 0:
            self.solve_oxygen()
        self.advance_cytokines(dt)
        exposure = self.drug.advance_hour() if self.drug is not None else None
        if not cells:
            self.hours += 1
            return {}
        ev, g, k = {}, self.gen, self.kind

        # state changes
        f_tum = C.local_fraction(k == G.TUMOR, self.kernel)
        k, ev["m2_new"] = C.polarize_macrophages(k, f_tum, pb(v["k_m2"]), dt, g)
        k, ev["caf_new"] = C.activate_fibroblasts(k, self.tgfb, pb(v["k_caf"]), dt, g)
        if _any_positive(v["k_m2_rev"]):
            k, ev["m2_reverted"] = C2.revert(k, G.M2, G.M0, pb(v["k_m2_rev"]), dt, g)
        if _any_positive(v["k_caf_rev"]):
            k, ev["caf_reverted"] = C2.revert(k, G.CAF, G.FIB, pb(v["k_caf_rev"]), dt, g)

        # tumour
        if exposure is not None:
            k, ev["tumour_killed_drug"] = C.drug_kill(k, exposure, pb(v["k_kill"]), g)
        k, ev["tumour_died"] = C.tumour_death(k, self.o2, pb(v["d0"]), pb(v["r_nec"]),
                                              pb(v["o_nec"]), dt, g)
        m2_local = C.local_sum(k == G.M2, self.kernel)
        cd4_local = C.local_fraction(k == G.CD4, self.kernel)
        k, ev["tumour_killed_cd8"] = C2.cd8_kill(k, m2_local, cd4_local, pb(v["k_T"]),
                                                 pb(v["K_sup"]), pb(v["gamma_cd4"]), dt, g)

        # stroma and immune: death
        k, ev["mac_died"] = C2.die(k, (G.M0, G.M2), pb(v["d_mac"]), dt, g)
        k, ev["fib_died"] = C2.die(k, (G.FIB, G.CAF), pb(v["d_fib"]), dt, g)
        k, ev["cd8_died"] = C2.die(k, (G.CD8,), pb(v["d_cd8"]), dt, g)
        k, ev["cd4_died"] = C2.die(k, (G.CD4,), pb(v["d_cd4"]), dt, g)
        k, ev["b_died"] = C2.die(k, (G.BCELL,), pb(v["d_b"]), dt, g)

        # division
        k, ev["tumour_born"] = C.tumour_divide(k, self.o2, pb(v["r_div"]), pb(v["K_o_growth"]), dt, g)
        k, ev["fib_born"] = _fibroblast_divide_logistic(k, self.fib_kernel, pb(v["K_fib"]),   # v3
                                                        pb(v["u_fib"]) * pb(v["d_fib"]),
                                                        pb(v["m_caf"]), dt, g)
        k, ev["cd8_born"] = C2.divide(k, k == G.CD8, pb(v["s_cd8"]) * pb(v["d_cd8"]), dt, g)
        k, ev["cd4_born"] = C2.divide(k, k == G.CD4, pb(v["s_cd4"]) * pb(v["d_cd4"]), dt, g)
        k, ev["b_born"] = C2.divide(k, k == G.BCELL, pb(v["s_b"]) * pb(v["d_b"]), dt, g)
        if _any_positive(v["p_mac"]):
            k, ev["mac_born"] = C2.divide(k, (k == G.M0) | (k == G.M2), pb(v["p_mac"]), dt, g)

        # recruitment at vessels, populations in a fresh random order each hour
        rec = [(G.M0, "r_mac", "mac_recruited"), (G.CD8, "r_cd8", "cd8_recruited"),
               (G.CD4, "r_cd4", "cd4_recruited"), (G.BCELL, "r_b", "b_recruited")]
        for i in torch.randperm(len(rec), generator=g, device=g.device).tolist():
            kk, rname, evn = rec[i]
            k, ev[evn] = C2.recruit(k, self.vessels, kk, pb(v[rname]), dt, g)

        # movement
        k, ev["lymphoid_moves"] = C2.move(k, G.MOTILE_LYMPHOID, int(v["move_lymphoid"]), g)
        k, ev["myeloid_moves"] = C2.move(k, G.MOTILE_MYELOID, int(v["move_myeloid"]), g)
        k, ev["fibroblast_moves"] = C2.move(k, (G.FIB, G.CAF), int(v["move_fibroblast"]), g)

        self.kind = k
        self.hours += 1
        self.events = ev
        return ev

    # ---------------- summaries ----------------
    def counts(self):
        return {name: c.cpu() for name, c in G.counts(self.kind).items()}


def _fibroblast_divide_logistic(kind, kernel, K_fib, p_rest, m_caf, dt, generator):
    """
    Fibroblasts and CAFs divide at p_rest * (1 - rho / K_fib), clamped at 0,
    where rho is the local lineage share of the neighbourhood disc; CAFs at
    m_caf times that. Logistic contact inhibition, so the steady state has a
    closed form (see steady_state_fractions).

    This replaces cells_v2.fibroblast_divide, which switches division fully on
    below the set point and fully off above it. Under that rule the lineage
    declined steadily whenever division did not far outpace death, and the
    level it settled at had no closed form, so it could not be used as a
    starting state. cells_v2.py is left as written.
    """
    lineage = (kind == G.FIB) | (kind == G.CAF)
    # Neighbours only: the cell's own square is excluded. Counting it (as
    # cells.local_fraction does) adds 1/29 = 0.034 to every fibroblast's own
    # density on the 29-square disc, against a set point near 0.1, which shut
    # division off far below the intended level (v2 test, Nov 5).
    own = lineage.double()
    ones = torch.ones_like(own, dtype=torch.bool)
    rho = (C.local_sum(lineage, kernel) - own) / (C.local_sum(ones, kernel) - 1.0).clamp(min=1.0)
    room = (1.0 - rho / K_fib).clamp(min=0.0)
    is_caf = (kind == G.CAF).double()
    rate = p_rest * room * (1.0 + (m_caf - 1.0) * is_caf)
    wants = lineage & (C.rand_like(rho, generator) < C.p_event(rate, dt))
    kind, arrived, _ = C2._place_into_neighbours(kind, wants, kind, generator)
    return kind, int(arrived.sum())


def _any_positive(x):
    t = torch.as_tensor(x)
    return bool((t > 0).any())
