"""
Treatment response B_i (Phase 4, protocol steps 20-23).

burn_in        - run without drug until every cell count changes by < tol per day
paired_response - copy the burned-in state; chemo arm vs no-chemo arm with the same
                  random draws; B_i = log10(N_no chemo / N_chemo) at chosen hours.

Primary read-out (decided Oct 28): 24 h after the last dose of cycle 1 = day 9
(hour 192 after chemo starts). Measuring later lets survivors refill the space the
drug freed, which in a nearly full patch says little about residual disease.
"""

import copy

import torch

from . import grid as G

READ_HOURS = (24, 168, 192)       # after dose 1, just before dose 2, 24 h after dose 2 (primary)
PRIMARY_HOUR = 192
TRACKED = ("tumour", "M0", "M2", "fibroblast", "CAF", "CD8")


def _counts(T):
    c = G.counts(T.kind)
    return torch.stack([c[k].double() for k in TRACKED if k in c], 1)      # (B, types)


def occupancy(T):
    """Fraction of non-vessel squares holding a cell, per tissue.

    The diagnostic that matters for the regime. Near 1 the patch is
    space-limited and tumour dynamics stop leaving a signature in the data;
    well below 1 the tumour is still free to expand."""
    c = G.counts(T.kind)
    dev = T.kind.device
    cells = sum(c[k].to(dev).double() for k in TRACKED)
    free = float(T.ny * T.nx) - c["vessel"].to(dev).double()
    return (cells / free.clamp(min=1)).cpu()


def burn_in(T, tol=0.01, max_days=30, min_days=3, min_share=0.01, settle=None):
    """Advance T in place, one day at a time, until the populations that must
    settle have changed by less than tol over the last day.

    settle names those populations; None means all of TRACKED. In the residual
    regime the tumour is deliberately excluded, because sparse post-resection
    disease is still expanding when treatment starts and waiting for it to
    stop would reintroduce the very saturation the regime exists to avoid.
    The stroma must still settle, since the patient's expression profile
    describes a microenvironment at rest.

    A population is only judged once it holds at least min_share of all cells.
    An earlier flat floor of 20 cells let a nearly exhausted M0 pool (~45
    cells, so one conversion = 2.2 %/day) hold the tissue below tolerance for
    30 days while the tumour was flat to 0.02 % (Phase 4).

    Returns days run, worst daily change among judged populations, which
    population was binding, and the final occupancy."""
    watched = [i for i, n in enumerate(TRACKED) if settle is None or n in settle]
    prev = _counts(T)
    worst, binding = float("inf"), None
    for day in range(1, max_days + 1):
        T.run(24)
        now = _counts(T)
        total = now.sum(1, keepdim=True).clamp(min=1)
        judged = (now / total) >= min_share
        rel = ((now - prev).abs() / prev.clamp(min=1)) * judged
        rel = rel[:, watched]
        worst = float(rel.max())
        binding = TRACKED[watched[int(rel.max(0).values.argmax())]] if rel.numel() else None
        prev = now
        if day >= min_days and worst < tol:
            return {"days": day, "worst_daily_change": worst, "binding": binding,
                    "occupancy": float(occupancy(T).mean()), "converged": True}
    return {"days": max_days, "worst_daily_change": worst, "binding": binding,
            "occupancy": float(occupancy(T).mean()), "converged": False}


def paired_response(T, cycles=1, hours=PRIMARY_HOUR, read=READ_HOURS):
    """B_i per tissue at each read hour. T is left untouched."""
    chemo, ctrl = copy.deepcopy(T), copy.deepcopy(T)
    chemo.start_chemo(cycles)
    ctrl.start_chemo(0)
    out = {}
    for h in range(1, hours + 1):
        chemo.step_hour()
        ctrl.step_hour()
        if h in read:
            n_c = chemo.counts()["tumour"].double()
            n_n = ctrl.counts()["tumour"].double()
            out[h] = {"B_i": torch.log10(n_n / n_c.clamp(min=1)).tolist(),
                      "tumour_chemo": n_c.tolist(), "tumour_no_chemo": n_n.tolist()}
    return out
