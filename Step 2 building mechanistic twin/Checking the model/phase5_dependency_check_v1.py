"""
Cell-cell dependencies: does the twin reproduce how cell types go together in
real tumours? Outcome-free; CPU; no new simulations.

    python scripts/phase5_dependency_check_v1.py --sims runs/phase5_v3_fib120_sims \
        --cohorts GSE14814 GSE68465 GSE42127 TCGA-LUAD --out runs/phase5_dependency_check_v1

The model has designed interactions (M2 suppresses CD8 killing, TGF-b from M2
activates CAFs, tumour drives M2 polarisation, vessels bring immune cells and
oxygen). Those rules produce co-occurrence between features that nobody set
directly. Here we compare that emergent structure with patients.

Spearman correlation is used because it only depends on ranks, so the
monotone links (log, logit) between twin and patient scores do not change it.
  - twin matrix: over all campaign tissues with finite features
  - patient matrix: per cohort
For each feature pair: twin rho, patient rho per cohort, and whether the sign
agrees where the patient correlation is clear (|rho| > 0.2 and p < 0.01).
Summary: share of clear patient pairs whose sign the twin matches, and the
correlation between the two matrices' upper triangles.
A pair the twin gets wrong is a dependency missing from (or wrong in) the model.

Writes <out>/pairs.csv and summary.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tme_twin.observe_v2 import FEATURES                               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", required=True)
    ap.add_argument("--cohorts", nargs="+", required=True)
    ap.add_argument("--clear-rho", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    z = np.load(os.path.join(a.sims, "sims.npz"), allow_pickle=True)
    feats = [f for f in FEATURES if f in list(z["features"])]
    xs = z["x"][:, [list(z["features"]).index(f) for f in feats]]
    xs = xs[np.isfinite(xs).all(1)]
    rt = stats.spearmanr(xs).statistic
    print(f"twin: {len(xs)} simulated tissues, {len(feats)} features")

    rows, summ = [], {}
    iu = np.triu_indices(len(feats), 1)
    for c in a.cohorts:
        obs = pd.read_csv(os.path.join(ROOT, "data", "processed", c, "features_v2_qc.csv"), index_col=0)
        use = [f for f in feats if f in obs.columns]
        idx = [feats.index(f) for f in use]
        o = obs[use].dropna().to_numpy()
        res = stats.spearmanr(o)
        rp, pp = res.statistic, res.pvalue
        clear = agree = 0
        for i in range(len(use)):
            for k in range(i + 1, len(use)):
                t = rt[idx[i], idx[k]]
                is_clear = abs(rp[i, k]) > a.clear_rho and pp[i, k] < 0.01
                same = np.sign(t) == np.sign(rp[i, k])
                clear += is_clear; agree += is_clear and same
                rows.append({"cohort": c, "feature_a": use[i], "feature_b": use[k], "rho_twin": t,
                             "rho_patients": rp[i, k], "p_patients": pp[i, k], "clear_in_patients": is_clear,
                             "sign_agrees": bool(same)})
        sub = rt[np.ix_(idx, idx)]
        ju = np.triu_indices(len(use), 1)
        mcorr = float(np.corrcoef(sub[ju], rp[ju])[0, 1])
        summ[c] = {"n_patients": int(len(o)), "clear_pairs": int(clear), "sign_agrees": int(agree),
                   "share_agree": agree / clear if clear else None, "matrix_correlation": mcorr}
        print(f"  {c}: {len(o)} patients; {agree}/{clear} clear pairs with matching sign; "
              f"matrix correlation {mcorr:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "pairs.csv"), index=False)
    clear = df[df.clear_in_patients]
    cons = clear.groupby(["feature_a", "feature_b"]).agg(n=("cohort", "size"), agree=("sign_agrees", "sum"),
                                                         rho_twin=("rho_twin", "first"),
                                                         rho_pat=("rho_patients", "mean")).reset_index()
    wrong = cons[(cons.n >= 2) & (cons.agree == 0)].sort_values("rho_pat", key=np.abs, ascending=False)
    print("\npairs clear in >= 2 cohorts where the twin has the wrong sign everywhere:")
    for _, r in wrong.iterrows():
        print(f"  {r.feature_a:>18} ~ {r.feature_b:<18} patients {r.rho_pat:+.2f}  twin {r.rho_twin:+.2f}  ({r.n} cohorts)")
    if wrong.empty:
        print("  none")
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"features": feats, "cohorts": summ, "twin_matrix": rt.tolist(),
                   "wrong_sign_consistent": wrong.to_dict(orient="records")}, fh, indent=2)
    print(f"wrote {a.out}/pairs.csv and summary.json")


if __name__ == "__main__":
    main()
