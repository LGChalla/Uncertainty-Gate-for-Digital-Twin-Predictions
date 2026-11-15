"""
What drives the twin's chemotherapy response? A sensitivity ranking on the
campaign we already have (no new simulations, CPU only).

    python scripts/phase6_sensitivity_v1.py --sims runs/phase5_v3_fib120_sims \
        --npe runs/phase5_v3_fib120_npe_logithyp_endolinked --out runs/phase6_sensitivity_v1

Method. First-order Sobol index from given data (Plischke 2010): for each
parameter, sort the draws by it, cut them into equal-count bins, and measure
how much of the variance of B_i is explained by the bin means:
    S_i = Var(E[B_i | theta_i]) / Var(B_i)
This works on any random sample, so the 20,000-draw campaign (each parameter
drawn independently from its prior, 4,504 draws with a finite B_i at 192 h)
can be used as it is. Log-uniform parameters are binned on the log scale.
  - Bias: with finite bins, even an unrelated parameter gets S > 0. The same
    estimate on B_i shuffled against theta (--null repeats) gives that floor;
    the reported index is S_i minus the null mean, and an index is called
    clear when it exceeds the null's 99th percentile.
  - Uncertainty: bootstrap over draws (--boot), 95% interval.
  - The first-order indices add up to less than 1; the rest is interactions
    plus run-to-run chance (the same tissue run twice gives a different B_i).
    Sum of first-order indices is reported.

Then the link to patients: for each of the 12 patient parameters, the
between/within ratio from the fitting (phase5_train_npe_v5 report.json, per
cohort). A parameter matters for which patients benefit only if it both
drives B_i (S clear) and differs between patients (ratio > 1).

Writes <out>/sensitivity.csv and sensitivity.json.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tme_twin import priors_v4 as P                                   # noqa: E402

PRIMARY_HOUR = 192


def first_order(x, y, bins):
    order = np.argsort(x, kind="mergesort")
    ys = y[order]
    groups = np.array_split(ys, bins)
    mu = y.mean()
    between = sum(len(g) * (g.mean() - mu) ** 2 for g in groups) / len(y)
    return between / y.var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", required=True)
    ap.add_argument("--npe", default=None, help="fitting output, for the between/within ratios")
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--hour", type=int, default=PRIMARY_HOUR)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    z = np.load(os.path.join(a.sims, "sims.npz"), allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    names = list(z["names"])
    col = list(meta["read_hours"]).index(a.hour)
    ok = np.isfinite(z["bi"][:, col])
    th, y = z["theta"][ok], z["bi"][ok, col]
    pri = P.regime_priors(meta["regime"])
    X = np.column_stack([np.log(th[:, j]) if pri[n][2] == "log" else th[:, j] for j, n in enumerate(names)])
    print(f"{ok.sum()} draws with B_i at {a.hour} h (of {len(ok)} in the campaign; "
          f"{(~ok & np.isfinite(z['bi']).any(1)).sum()} have B_i at other hours only)")
    print(f"B_i: mean {y.mean():.3f}, SD {y.std():.3f}")

    null = np.array([first_order(X[:, 0], rng.permutation(y), a.bins) for _ in range(a.null)])
    floor, q99 = null.mean(), np.quantile(null, 0.99)
    rows = []
    for j, n in enumerate(names):
        s = first_order(X[:, j], y, a.bins)
        bs = []
        for _ in range(a.boot):
            i = rng.integers(0, len(y), len(y))
            bs.append(first_order(X[i, j], y[i], a.bins))
        bs = np.array(bs) - floor
        r = np.corrcoef(X[:, j], y)[0, 1]
        rows.append({"parameter": n, "class": P.PRIORS[n][3], "S_first_order": s - floor,
                     "ci_low": float(np.quantile(bs, 0.025)), "ci_high": float(np.quantile(bs, 0.975)),
                     "clear": bool(s > q99), "direction": "+" if r > 0 else "-", "corr": float(r)})
    df = pd.DataFrame(rows).sort_values("S_first_order", ascending=False)

    if a.npe:
        rep = json.load(open(os.path.join(a.npe, "report.json")))
        for ds, c in rep["cohorts"].items():
            spec = c.get("specificity", {})
            df[f"between_within_{ds}"] = df["parameter"].map(lambda p: spec.get(p, {}).get("between_over_within", np.nan))

    df.to_csv(os.path.join(a.out, "sensitivity.csv"), index=False)
    total = float(df.loc[df["S_first_order"] > 0, "S_first_order"].sum())
    with open(os.path.join(a.out, "sensitivity.json"), "w") as fh:
        json.dump({"n_draws": int(len(y)), "hour": a.hour, "bins": a.bins, "null_floor": float(floor),
                   "null_q99": float(q99 - floor), "sum_first_order": total,
                   "rows": df.to_dict(orient="records")}, fh, indent=2)

    spec_cols = [c for c in df.columns if c.startswith("between_within_")]
    print(f"\nfirst-order share of B_i variance explained by each parameter alone "
          f"(bias-corrected; 'clear' = above the shuffled-data 99th percentile, {q99 - floor:.4f})")
    hdr = f"{'parameter':>18} {'class':>8} {'S':>7} {'95% CI':>17} {'dir':>4}"
    if spec_cols:
        hdr += "   between/within by cohort"
    print(hdr)
    for _, r in df.iterrows():
        line = (f"{r['parameter']:>18} {r['class']:>8} {r['S_first_order']:>7.3f} "
                f"[{r['ci_low']:>6.3f}, {r['ci_high']:>6.3f}] {r['direction']:>4}{'  *' if r['clear'] else '   '}")
        if spec_cols and r["class"] == "patient":
            line += "  " + " ".join(f"{r[c]:.2f}" for c in spec_cols)
        print(line)
    print(f"\nsum of first-order indices: {total:.2f} (the rest is interactions and run-to-run chance)")
    print(f"wrote {a.out}/sensitivity.csv and sensitivity.json")


if __name__ == "__main__":
    main()
