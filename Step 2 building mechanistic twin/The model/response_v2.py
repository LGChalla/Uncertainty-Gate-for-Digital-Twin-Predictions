"""
Burn-in and treatment response for v2 tissues. response.py is unchanged.

paired_response is reused from response.py as is: it only needs step_hour,
start_chemo and counts()["tumour"], which TissueV2 provides. burn_in and
occupancy are rewritten here because they count populations, and v2 has two
more (CD4, B).

In v1 the stroma could not reach a steady state (no source, no sink), so
burn-in convergence was never achievable and a fixed 20 days was used. In v2
every population has a source and a sink and starts at its analytic steady
state, so convergence is expected and is reported for every tissue.
"""

import torch

from . import grid_v2 as G
from .response import paired_response, READ_HOURS, PRIMARY_HOUR      # noqa: F401
from .tissue_v2 import TRACKED

SETTLE_RESIDUAL = ("M0", "M2", "fibroblast", "CAF", "CD8", "CD4", "B")


def _counts(T):
    c = G.counts(T.kind)
    return torch.stack([c[k].double() for k in TRACKED], 1)


def occupancy(T):
    c = G.counts(T.kind)
    dev = T.kind.device
    cells = sum(c[k].to(dev).double() for k in TRACKED)
    free = float(T.ny * T.nx) - c["vessel"].to(dev).double()
    return (cells / free.clamp(min=1)).cpu()


def burn_in(T, tol=0.01, max_days=30, min_days=3, min_share=0.01, settle=SETTLE_RESIDUAL,
            fixed_days=None):
    """
    As response.burn_in, over the v2 populations. With fixed_days set, runs
    exactly that long and reports whether the tolerance was met anyway, so a
    fixed-length design still records convergence per tissue.

    Returns days run, worst daily change among judged populations, which was
    binding, final occupancy, and converged (per tissue, as a list).
    """
    watched = [i for i, n in enumerate(TRACKED) if n in settle]
    prev = _counts(T)
    days = fixed_days or max_days
    per_tissue = None
    worst, binding = float("inf"), None
    for day in range(1, days + 1):
        T.run(24)
        now = _counts(T)
        total = now.sum(1, keepdim=True).clamp(min=1)
        judged = (now / total) >= min_share
        rel = ((now - prev).abs() / prev.clamp(min=1)) * judged
        rel = rel[:, watched]
        per_tissue = rel.max(1).values
        worst = float(per_tissue.max())
        binding = TRACKED[watched[int(rel.max(0).values.argmax())]] if rel.numel() else None
        prev = now
        if fixed_days is None and day >= min_days and worst < tol:
            break
    conv = (per_tissue < tol).tolist() if per_tissue is not None else []
    return {"days": day, "worst_daily_change": worst, "binding": binding,
            "occupancy": float(occupancy(T).mean()), "converged": all(conv),
            "converged_per_tissue": conv,
            "worst_daily_change_per_tissue": per_tissue.tolist() if per_tissue is not None else []}
