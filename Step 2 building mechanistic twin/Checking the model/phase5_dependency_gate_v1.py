"""
Dependency gate (pre-registration v0.12): a fixed pass/fail rule on whether the twin's
features go together the way they do in patients. Outcome-free; CPU; seconds.

    python scripts/phase5_dependency_gate_v1.py --sims runs/phase5_v3_fib120_sims \
        --cohorts GSE14814 GSE68465 GSE42127 TCGA-LUAD --out runs/phase5_dependency_gate_v1_4cohorts

What the gate establishes, and what it does not
  - sign agreement      directional consistency: for feature pairs clearly correlated in
                        patients (|rho| > 0.2, p < 0.01), does the twin's correlation have
                        the same sign?
  - purity law          structural consistency: the purity slopes b_j of the dependency
                        equation (phase5_dependency_equation_v1) point the same way in twin
                        and patients (cosine of the b vectors).
  Passing means "consistent with these two empirical constraints". It is NOT evidence
  that the twin is biologically correct, and it says nothing about identifiability
  (that is the between/within check). Failing means the twin's internal structure
  disagrees with patients in a measured, reproducible way.

Rule, fixed before it is applied (v0.12): a cohort passes if sign agreement >= 0.75
AND purity-law cosine >= 0.70; the twin passes the gate if >= 3 of the 4 cohorts pass.

Positive control for the gate: --controls synthetic cohorts drawn from the twin itself
(same sizes as the real cohorts, the fitting's observation noise added). A twin is
by construction consistent with itself, so these should pass; the share that does
is the gate's sensitivity. TumorTwin is not scored: it does not generate features.

Writes <out>/gate.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tme_twin.observe_v2 import FEATURES, PLANS, DEFAULT_PLAN             # noqa: E402
from phase5_dependency_equation_v1 import fit as eq_fit                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGN_MIN, COS_MIN, COHORTS_MIN = 0.75, 0.70, 3


def sign_agreement(rt, df, clear_rho=0.2):
    use = list(df.columns)
    res = stats.spearmanr(df.to_numpy())
    rp, pp = res.statistic, res.pvalue
    clear = agree = 0
    for i in range(len(use)):
        for k in range(i + 1, len(use)):
            if abs(rp[i, k]) > clear_rho and pp[i, k] < 0.01:
                clear += 1
                agree += np.sign(rt.loc[use[i], use[k]]) == np.sign(rp[i, k])
    return (agree / clear if clear else np.nan), int(clear)


def cosine(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def score(df, rt, twin_eq):
    sa, nclear = sign_agreement(rt, df)
    e = eq_fit(df)
    common = [q for q in e["features"] if q in twin_eq["features"]]
    b_c = np.array([e["b"][e["features"].index(q)] for q in common])
    b_t = np.array([twin_eq["b"][twin_eq["features"].index(q)] for q in common])
    cs = cosine(b_c, b_t)
    return {"sign_agreement": float(sa), "clear_pairs": nclear, "purity_cosine": cs,
            "pass": bool(sa >= SIGN_MIN and cs >= COS_MIN)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", required=True)
    ap.add_argument("--cohorts", nargs="+", required=True)
    ap.add_argument("--controls", type=int, default=20)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    z = np.load(os.path.join(a.sims, "sims.npz"), allow_pickle=True)
    feats = [q for q in FEATURES if q in list(z["features"])]
    xs = pd.DataFrame(z["x"][:, [list(z["features"]).index(q) for q in feats]], columns=feats)
    xs = xs.replace([np.inf, -np.inf], np.nan).dropna()
    rt = pd.DataFrame(stats.spearmanr(xs.to_numpy()).statistic, index=feats, columns=feats)
    twin_eq = eq_fit(xs)
    plan = PLANS[DEFAULT_PLAN]
    cv = np.array([plan["direct"].get(q, plan["linked"].get(q, 0.2)) for q in feats])

    real, sizes = {}, []
    for c in a.cohorts:
        obs = pd.read_csv(os.path.join(ROOT, "data", "processed", c, "features_v2_qc.csv"), index_col=0)
        df = obs[[q for q in feats if q in obs.columns]].dropna()
        real[c] = score(df, rt, twin_eq); sizes.append(len(df))
    n_pass = sum(r["pass"] for r in real.values())
    verdict = n_pass >= COHORTS_MIN

    ctrl = []
    for i in range(a.controls):
        n = sizes[i % len(sizes)]
        s = xs.iloc[rng.choice(len(xs), n, replace=False)].to_numpy()
        s = s * np.exp(rng.normal(0, 1, s.shape) * cv)          # observation noise, multiplicative
        ctrl.append(score(pd.DataFrame(s, columns=feats), rt, twin_eq))
    cpass = float(np.mean([r["pass"] for r in ctrl]))

    print(f"rule: a cohort passes if sign agreement >= {SIGN_MIN} and purity-law cosine >= {COS_MIN}; "
          f"the twin passes if >= {COHORTS_MIN} of {len(a.cohorts)} cohorts pass\n")
    print(f"{'':>12} {'sign agreement':>15} {'clear pairs':>12} {'purity cosine':>14} {'pass':>6}")
    for c, r in real.items():
        print(f"{c:>12} {r['sign_agreement']:>15.2f} {r['clear_pairs']:>12} {r['purity_cosine']:>14.2f} {str(r['pass']):>6}")
    print(f"\ntwin: {n_pass}/{len(a.cohorts)} cohorts pass -> gate {'PASSED' if verdict else 'FAILED'}")
    sa = [r["sign_agreement"] for r in ctrl]; cs = [r["purity_cosine"] for r in ctrl]
    print(f"positive control ({a.controls} synthetic cohorts from the twin itself): {cpass:.0%} pass; "
          f"sign agreement median {np.median(sa):.2f} [{np.min(sa):.2f}-{np.max(sa):.2f}], "
          f"purity cosine median {np.median(cs):.2f} [{np.min(cs):.2f}-{np.max(cs):.2f}]")
    print("TumorTwin: not scored (it does not generate features)")
    with open(os.path.join(a.out, "gate.json"), "w") as fh:
        json.dump({"rule": {"sign_min": SIGN_MIN, "cosine_min": COS_MIN, "cohorts_min": COHORTS_MIN},
                   "cohorts": real, "twin_passes": bool(verdict), "control_pass_rate": cpass,
                   "controls": ctrl}, fh, indent=2)
    print(f"wrote {a.out}/gate.json")


if __name__ == "__main__":
    main()
