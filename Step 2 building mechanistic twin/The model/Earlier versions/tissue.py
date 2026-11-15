"""
A batch of tissues stepped forward one hour at a time (Phases 1-2).

Each hour:
  1. oxygen is re-solved as a steady state (it settles in seconds; step 15),
     with fixed pO2 at vessels and Michaelis-Menten use by every cell;
  2. IL-6 and TGF-beta advance in ~5 min implicit steps, made by M2 cells,
     plus a basal background source (no clamp; step 13);
  3. the cell rules run once, using the fields from steps 1-2.

Per-tissue parameters (vessel density, k_M2, r_IL6, ...) can be given as (B,)
tensors so each tissue can be a different patient later on.
"""

import math

import torch

from .config import TME, values
from . import grid as G
from . import cells as C
from .fields import implicit_step, steady_state
from .drug import Drug


class Tissue:
    def __init__(self, B, fractions, seed, device="cpu", overrides=None, cfg=TME):
        self.v = values(cfg)
        if overrides:
            self.v.update(overrides)
        v = self.v
        self.B, self.device = B, device
        self.ny, self.nx, self.dx = int(v["ny"]), int(v["nx"]), v["dx"]
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.seed = seed
        self.hours = 0

        self.vessels = G.make_vessels(B, self.ny, self.nx, self.dx, self.pb(v["vessel_density"]).flatten(),
                                      self.gen, device)
        self.kind = G.place_cells(self.vessels, fractions, self.gen)

        f64 = dict(dtype=torch.float64, device=device)
        self.il6 = torch.full((B, self.ny, self.nx), v["c0_il6"], **f64)
        self.tgfb = torch.full((B, self.ny, self.nx), v["c0_tgfb"], **f64)
        self.o2 = torch.zeros((B, self.ny, self.nx), **f64) + self.pb(v["o_vessel"])
        self.kernel = C.disc_kernel(max(1, round(v["neighbor_radius_um"] * 1e-4 / self.dx)), device)
        self.events = {}
        self.drug = None
        self.solve_oxygen()

    # per-tissue parameter -> (B, 1, 1) tensor
    def pb(self, x):
        t = torch.as_tensor(x, dtype=torch.float64, device=self.device)
        return t.reshape(-1, 1, 1).expand(self.B, 1, 1) if t.dim() <= 1 else t

    # ---------------- fields ----------------
    def solve_oxygen(self, iters=30, tol_mmHg=0.01):
        """
        Steady state of  div(D grad o) = q(o),  q(o) = Vmax * rho * o / (Km + o),
        with o fixed at vessels. Newton's method: linearise q around the current o,
            q(o) ~ q(o_k) + q'(o_k) (o - o_k),   q' = Vmax * rho * Km / (Km + o_k)^2,
        solve the linear problem, repeat until o changes by < tol_mmHg everywhere.
        (A plain fixed-point iteration converged too slowly here; Phase 2 check B.)
        """
        v = self.v
        rho = ((self.kind != G.EMPTY) & (self.kind != G.VESSEL)).double()
        Vm, Km = self.pb(v["Vmax_o2"]), self.pb(v["Km_o2"])
        o = self.o2.clamp(min=0)
        self.o2_iters = 0
        for it in range(iters):
            q = Vm * rho * o / (Km + o)
            dq = Vm * rho * Km / (Km + o) ** 2
            rhs = dq * o - q
            o_new, _, _ = steady_state(v["D_o2"], dq, rhs, self.dx,
                                       fixed_mask=self.vessels,
                                       fixed_value=self.pb(v["o_vessel"]),
                                       x0=o, tol=1e-10)
            o_new = o_new.clamp(min=0)
            change = float((o_new - o).abs().max())
            o = o_new
            self.o2_iters = it + 1
            if change < tol_mmHg:
                break
        self.o2 = o
        self.o2_last_change = change

    def advance_cytokines(self, seconds):
        v = self.v
        m2 = (self.kind == G.M2).double()
        s_il6 = self.pb(v["r_il6"]) * m2 + v["lam_il6"] * v["c0_il6"]
        s_tgf = self.pb(v["r_tgfb"]) * m2 + v["lam_tgfb"] * v["c0_tgfb"]
        n = max(1, round(seconds / v["dt_field"]))
        dt = seconds / n
        for _ in range(n):
            self.il6, _, _ = implicit_step(self.il6, v["D_il6"], v["lam_il6"], s_il6, dt, self.dx)
            self.tgfb, _, _ = implicit_step(self.tgfb, v["D_tgfb"], v["lam_tgfb"], s_tgf, dt, self.dx)

    # ---------------- chemo ----------------
    def start_chemo(self, cycles):
        """Attach the drug from now on. cycles=0 is the no-chemo arm of a pair: it makes
        the same random draws as the chemo arm, so the two copies stay matched."""
        self.drug = Drug(self, cycles)

    # ---------------- one hour ----------------
    def step_hour(self, cells=True):
        v, dt = self.v, self.v["dt_agent"]
        if self.hours % int(v["o2_every_h"]) == 0:
            self.solve_oxygen()
        self.advance_cytokines(dt)
        exposure = self.drug.advance_hour() if self.drug is not None else None
        if not cells:
            self.hours += 1
            return {}
        ev = {}
        f_tum = C.local_fraction(self.kind == G.TUMOR, self.kernel)
        self.kind, ev["m2_new"] = C.polarize_macrophages(self.kind, f_tum, self.pb(v["k_m2"]), dt, self.gen)
        self.kind, ev["caf_new"] = C.activate_fibroblasts(self.kind, self.tgfb, self.pb(v["k_caf"]), dt, self.gen)
        if exposure is not None:
            self.kind, ev["tumour_killed_drug"] = C.drug_kill(self.kind, exposure, self.pb(v["k_kill"]), self.gen)
        self.kind, ev["tumour_died"] = C.tumour_death(self.kind, self.o2, self.pb(v["d0"]),
                                                     self.pb(v["r_nec"]), self.pb(v["o_nec"]),
                                                     dt, self.gen)
        m2_local = C.local_sum(self.kind == G.M2, self.kernel)
        self.kind, ev["tumour_killed_cd8"] = C.cd8_kill(self.kind, m2_local, self.pb(v["k_T"]),
                                                        self.pb(v["K_sup"]), dt, self.gen)
        self.kind, ev["tumour_born"] = C.tumour_divide(self.kind, self.o2, self.pb(v["r_div"]),
                                                       self.pb(v["K_o_growth"]), dt, self.gen)
        self.hours += 1
        self.events = ev
        return ev

    def run(self, hours, cells=True, every=None, callback=None):
        for h in range(hours):
            self.step_hour(cells=cells)
            if callback and every and (self.hours % every == 0):
                callback(self)
        return self

    # ---------------- summaries ----------------
    def counts(self):
        return {k: c.cpu() for k, c in G.counts(self.kind).items()}

    def hypoxic_fraction(self):
        cells = (self.kind != G.EMPTY) & (self.kind != G.VESSEL)
        hyp = cells & (self.o2 < self.pb(self.v["o_hypoxic"]))
        return (hyp.sum((1, 2)).double() / cells.sum((1, 2)).clamp(min=1).double()).cpu()

    def distance_to_vessel_um(self):
        """Distance from every square to its nearest vessel (um), via repeated dilation."""
        far = torch.full(self.vessels.shape, float("inf"), dtype=torch.float64, device=self.device)
        far[self.vessels] = 0.0
        # exact Euclidean distance would need a transform; chessboard-free approach:
        ys = torch.arange(self.ny, device=self.device, dtype=torch.float64)
        xs = torch.arange(self.nx, device=self.device, dtype=torch.float64)
        out = torch.empty_like(far)
        for b in range(self.B):
            vy, vx = torch.nonzero(self.vessels[b], as_tuple=True)
            if vy.numel() == 0:
                out[b] = float("inf")
                continue
            d = torch.full((self.ny, self.nx), float("inf"), dtype=torch.float64, device=self.device)
            for i0 in range(0, vy.numel(), 256):
                py, px = vy[i0:i0 + 256].double(), vx[i0:i0 + 256].double()
                dd = torch.sqrt((ys[None, :, None] - py[:, None, None]) ** 2 +
                                (xs[None, None, :] - px[:, None, None]) ** 2).amin(0)
                d = torch.minimum(d, dd)
            out[b] = d
        return out * self.dx * 1e4
