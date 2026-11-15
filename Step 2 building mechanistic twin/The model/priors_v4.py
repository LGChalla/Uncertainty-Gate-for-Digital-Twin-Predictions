"""
Priors, version 4. priors_v3.py is unchanged and describes the v2 campaign.

One change: K_fib 0.005-0.2 (was 0.01-0.50). With the 120 um sensing disc
(config_v3) this range gives a steady-state fibroblast share of cells of
roughly 0.6%-27% across the u_fib prior, which brackets the patients' EPIC
fibroblast share (medians 0.05-0.10). Above 0.2 the share exceeds anything
seen in the four cohorts.
"""

from .priors_v3 import *                       # noqa: F401,F403
from . import priors_v3 as _P3

PRIORS = dict(_P3.PRIORS)
PRIORS["K_fib"] = (0.005, 0.2, "log", "patient",
                   "fibroblast density set point (contact inhibition, 120 um sensing disc)")


def regime_priors(regime=_P3.DEFAULT_REGIME):
    out = dict(_P3.regime_priors(regime))
    out["K_fib"] = PRIORS["K_fib"]
    return out


# priors_v3's helpers read priors_v3.PRIORS; redefine them here so every
# caller of priors_v4 sees the new K_fib.
def bounds(names=None, regime=_P3.DEFAULT_REGIME):
    import math
    pri = regime_priors(regime)
    names = list(names or _P3.SCREEN)
    b = [[math.log(pri[n][0]), math.log(pri[n][1])] if pri[n][2] == "log" else [pri[n][0], pri[n][1]]
         for n in names]
    return {"num_vars": len(names), "names": names, "bounds": b}


def to_native(row, names, regime=_P3.DEFAULT_REGIME):
    import math
    pri = regime_priors(regime)
    return {n: (math.exp(x) if pri[n][2] == "log" else float(x)) for x, n in zip(row, names)}


def by_class(cls):
    return tuple(k for k, v in PRIORS.items() if v[3] == cls)
