"""
Prior-sensitivity check (pre-registration v0.5, reviewer item 1): is "drug
potency dominates B_i" an artefact of how wide its prior is?

Writes a SUBSET of the campaign in which the shared drug parameters sit in a
narrow band around their calibrated working values, then the existing analyses
(phase6_sensitivity_v1, phase7_value_of_information_v1) are run on it unchanged.

    python scripts/phase6_subset_sims_v1.py --sims runs/phase5_v3_fib120_sims \
        --band k_kill=0.25 --band K_o_kill=0.5 --out runs/phase5_v3_fib120_sims_narrow_kkill25_Ko50

--band NAME=W keeps draws whose NAME lies within the central fraction W of its
prior range (on the log scale for log-uniform priors), centred on the working
value in config_v3 (k_kill 1.68e-5 /(uM s), from the virtual-dish fit to Riaz
2019; K_o_kill 22 mmHg, from Wu 2015). W = 0.25 keeps a quarter of the range.
No simulation is rerun; nothing about patients or outcomes is read.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tme_twin.config_v3 import TME_V3                                 # noqa: E402
from tme_twin.config import values                                    # noqa: E402
from tme_twin import priors_v4 as P                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", required=True)
    ap.add_argument("--band", action="append", required=True, help="NAME=fraction of the prior range to keep")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = dict(np.load(os.path.join(a.sims, "sims.npz"), allow_pickle=True))
    meta = json.loads(str(z["meta"]))
    names = list(z["names"])
    pri = P.regime_priors(meta["regime"])
    v = values(TME_V3)
    keep = np.ones(len(z["theta"]), bool)
    bands = {}
    for b in a.band:
        nm, w = b.split("=")
        w = float(w)
        lo, hi, sc, _, _ = pri[nm]
        x = z["theta"][:, names.index(nm)]
        c = v[nm]
        if sc == "log":
            L = math.log(hi) - math.log(lo)
            lx, lc = np.log(x), math.log(c)
            half = w * L / 2
            lc = min(max(lc, math.log(lo) + half), math.log(hi) - half)
            k = np.abs(lx - lc) <= half
            bands[nm] = [float(math.exp(lc - half)), float(math.exp(lc + half))]
        else:
            half = w * (hi - lo) / 2
            c = min(max(c, lo + half), hi - half)
            k = np.abs(x - c) <= half
            bands[nm] = [c - half, c + half]
        keep &= k
        print(f"  {nm}: working value {v[nm]:.3g}; band {bands[nm][0]:.3g}..{bands[nm][1]:.3g} "
              f"({100 * w:.0f}% of the prior range) keeps {int(k.sum())} draws")
    col = list(meta["read_hours"]).index(192)
    nbi = int(np.isfinite(z["bi"][keep, col]).sum())
    print(f"subset: {int(keep.sum())} of {len(keep)} draws, {nbi} with B_i at 192 h")
    if nbi < 300:
        print("  warning: fewer than 300 draws with B_i; widen a band")
    sub = {k: (val[keep] if isinstance(val, np.ndarray) and val.ndim >= 1 and len(val) == len(keep) else val)
           for k, val in z.items() if k != "meta"}
    meta["subset_of"] = a.sims
    meta["subset_bands"] = bands
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "sims.npz"), **sub, meta=json.dumps(meta))
    print(f"wrote {a.out}/sims.npz")


if __name__ == "__main__":
    main()
