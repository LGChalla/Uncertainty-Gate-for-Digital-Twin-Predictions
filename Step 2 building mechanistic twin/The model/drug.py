"""
Cisplatin: plasma curve, spread through tissue, exposure (Phase 3, steps 17-19).

Plasma (step 17). Intact cisplatin falls monoexponentially (FDA label), so each
dose is a zero-order infusion into one compartment:
    during the infusion (0 <= tau < T):  C = R/CL * (1 - exp(-k tau))
    afterwards:                          C = R/CL * (1 - exp(-k T)) * exp(-k (tau - T))
with R = dose / T, k = CL / V. Doses add up. Result in uM.

Tissue (step 18). The general field equation with the vessel squares pinned to
the plasma value, uptake mu by every occupied square, and diffusivity lowered by
local CAF density:  D(x) = D0 / (1 + beta_CAF * f_CAF(x)).

Exposure (step 19). Over each hour, every square accumulates
    E = integral of  c(t) * o/(o + K_o) * 1/(1 + IL6/K_IL6)  dt     (uM s),
and the cell step kills a tumour cell with P = 1 - exp(-k_kill * E).
"""

import math

import torch

from .fields import implicit_step
from . import grid as G
from . import cells as C


def dose_times_h(v, cycles):
    """Start time (h) of every dose, counted from the first day of chemo."""
    days = v["cis_days"]
    out = []
    for c in range(int(cycles)):
        for d in days:
            out.append(((d - 1) + c * v["cis_cycle_days"]) * 24.0)
    return out


def plasma_uM(t_h, v, cycles):
    """Plasma cisplatin (uM) at time t_h hours after chemo starts. t_h: float or tensor."""
    T = v["cis_infusion_h"]
    k = v["cis_CL"] / v["cis_V"]                       # 1/h
    R = v["cis_dose"] / T                              # mg/h/m^2
    scale = R / v["cis_CL"] / v["cis_MW"] * 1000.0     # mg/L -> uM
    t = torch.as_tensor(t_h, dtype=torch.float64)
    total = torch.zeros_like(t)
    for t0 in dose_times_h(v, cycles):
        tau = t - t0
        during = scale * (1 - torch.exp(-k * tau.clamp(min=0)))
        after = scale * (1 - math.exp(-k * T)) * torch.exp(-k * (tau - T).clamp(min=0))
        total = total + torch.where(tau < 0, torch.zeros_like(t), torch.where(tau < T, during, after))
    return total


class Drug:
    """Drug field and per-hour exposure for one batch of tissues."""

    def __init__(self, tissue, cycles, threshold_uM=1e-4):
        self.T = tissue
        self.cycles = cycles
        self.t0 = tissue.hours                 # chemo starts now
        self.c = torch.zeros_like(tissue.il6)  # uM
        self.threshold = threshold_uM
        self.last_iters = 0

    def coefficients(self):
        T, v = self.T, self.T.v
        rho = ((T.kind != G.EMPTY) & (T.kind != G.VESSEL)).double()
        f_caf = C.local_fraction(T.kind == G.CAF, T.kernel)
        D = T.pb(v["D0_drug"]) / (1.0 + T.pb(v["beta_caf"]) * f_caf)
        k = T.pb(v["mu_drug"]) * rho
        f_o = T.o2 / (T.o2 + T.pb(v["K_o_kill"]))
        f_il6 = 1.0 / (1.0 + T.il6 / T.pb(v["K_il6_protect"]))
        return D, k, f_o * f_il6

    def advance_hour(self):
        """Advance the drug field one agent hour; return the hour's effective exposure (uM s)."""
        T, v = self.T, self.T.v
        t_start = float(T.hours - self.t0)
        n = max(1, round(v["dt_agent"] / v["dt_drug"]))
        dt = v["dt_agent"] / n
        ts = t_start + torch.arange(n + 1, dtype=torch.float64) * dt / 3600.0
        cp = plasma_uM(ts, v, self.cycles)
        E = torch.zeros_like(self.c)
        if float(cp.max()) < self.threshold and float(self.c.max()) < self.threshold:
            self.c.zero_()
            return E
        D, k, factor = self.coefficients()
        S = torch.zeros_like(self.c)
        for i in range(n):
            c_old = self.c
            self.c, self.last_iters, _ = implicit_step(c_old, D, k, S, dt, T.dx,
                                                       fixed_mask=T.vessels, fixed_value=float(cp[i + 1]))
            E = E + 0.5 * (c_old + self.c) * dt          # trapezoid
        return E * factor
