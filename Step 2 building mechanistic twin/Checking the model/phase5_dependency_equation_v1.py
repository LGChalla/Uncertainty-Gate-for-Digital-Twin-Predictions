"""
Do the cell-cell relationships in patients follow one simple equation?
Outcome-free; CPU; seconds.

    python scripts/phase5_dependency_equation_v1.py --sims runs/phase5_v3_fib120_sims \
        --cohorts GSE14814 GSE68465 GSE42127 TCGA-LUAD --out runs/phase5_dependency_equation_v1_4cohorts

The pattern in the dependency check: everything that is not tumour rises
together and falls as tumour rises. The candidate equation, for every
non-tumour feature j of patient i (features on the log scale, hypoxia logit):

    y_ij = a_j + b_j * log(1 - f_tumour_i) + c_j * I_i + e_ij

  b_j  purity: less tumour in the sample leaves room for everything else
  I_i  one shared "infiltration" level per patient (the first principal
       component of what purity leaves over), with loading c_j per feature
  e_ij what is left

Reported per cohort, and for the twin's 20,000 tissues for comparison:
  R2_purity     share of variance explained by purity alone (averaged over features)
  R2_full       purity + the shared axis
  same_sign     share of features whose c_j has the majority sign (1.0 = all
                rise together, the "hot vs cold" axis)
  loadings      b_j and c_j per feature, and how similar the c_j are across
                cohorts (cosine; near 1 = the same equation in every cohort)
If R2_full is high and the loadings agree across cohorts, the patient
relationships fit the equation; the twin's values show whether its rules
produce the same law.

Writes <out>/equation.json and loadings.csv.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tme_twin.observe_v2 import FEATURES                               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scale(df):
    out = pd.DataFrame(index=df.index)
    for f in df.columns:
        v = df[f].to_numpy(float)
        if f == "hypoxia":
            lo, hi = np.nanmin(v), np.nanmax(v)
            p = np.clip((v - lo) / (hi - lo + 1e-12), 1e-3, 1 - 1e-3) if (v.min() < 0 or v.max() > 1) else np.clip(v, 1e-3, 1 - 1e-3)
            out[f] = np.log(p / (1 - p))
        elif v.min() <= 0:                          # z-scored pathway scores: keep as is
            out[f] = v
        else:
            out[f] = np.log(np.clip(v, 1e-6, None))
    return out


def fit(df):
    ft = df["f_tumour"].to_numpy(float)
    ft = ft if ft.max() <= 1 else ft / 100
    p = np.log(np.clip(1 - ft, 1e-3, None))
    p = (p - p.mean()) / p.std()                      # b is per SD of purity, comparable across cohorts
    Y = scale(df.drop(columns="f_tumour"))
    Y = (Y - Y.mean()) / Y.std()
    X = np.column_stack([np.ones(len(p)), p])
    B, *_ = np.linalg.lstsq(X, Y.to_numpy(), rcond=None)
    R = Y.to_numpy() - X @ B
    r2_pur = 1 - (R ** 2).sum(0) / (Y.to_numpy() ** 2).sum(0)
    u, s, vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
    I = u[:, 0] * s[0]
    c = vt[0]
    if np.sum(np.sign(c)) < 0:
        c, I = -c, -I
    R2 = R - np.outer(I, c)
    r2_full = 1 - (R2 ** 2).sum(0) / (Y.to_numpy() ** 2).sum(0)
    same = float(np.mean(np.sign(c) == np.sign(np.sum(np.sign(c)) or 1)))
    return {"features": list(Y.columns), "b": B[1].tolist(), "c": c.tolist(),
            "R2_purity_by_feature": r2_pur.tolist(), "R2_full_by_feature": r2_full.tolist(),
            "R2_purity": float(r2_pur.mean()), "R2_full": float(r2_full.mean()), "same_sign": same,
            "n": int(len(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", required=True)
    ap.add_argument("--cohorts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = np.load(os.path.join(a.sims, "sims.npz"), allow_pickle=True)
    feats = [f for f in FEATURES if f in list(z["features"])]
    xs = pd.DataFrame(z["x"][:, [list(z["features"]).index(f) for f in feats]], columns=feats)
    res = {"twin": fit(xs.replace([np.inf, -np.inf], np.nan).dropna())}
    for c in a.cohorts:
        obs = pd.read_csv(os.path.join(ROOT, "data", "processed", c, "features_v2_qc.csv"), index_col=0)
        res[c] = fit(obs[[f for f in feats if f in obs.columns]].dropna())

    print(f"{'':>10} {'n':>6} {'purity R2':>10} {'+shared axis R2':>16} {'all rise together':>18}")
    for k, r in res.items():
        print(f"{k:>10} {r['n']:>6} {r['R2_purity']:>10.2f} {r['R2_full']:>16.2f} {r['same_sign']:>18.2f}")

    pats = [c for c in a.cohorts]
    common = [f for f in res["twin"]["features"] if all(f in res[c]["features"] for c in pats)]
    def vec(k, key):
        return np.array([res[k][key][res[k]["features"].index(f)] for f in common])
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    print("\nhow alike the shared-axis loadings are (cosine; 1 = same equation)")
    for i, ci in enumerate(pats):
        print(f"  {ci:>10}: " + "  ".join(f"{cj} {cos(vec(ci, 'c'), vec(cj, 'c')):+.2f}" for cj in pats[i + 1:])
              + f"  twin {cos(vec(ci, 'c'), vec('twin', 'c')):+.2f}")
    rows = []
    for f in common:
        row = {"feature": f}
        for k in ["twin"] + pats:
            j = res[k]["features"].index(f)
            row[f"b_{k}"], row[f"c_{k}"] = res[k]["b"][j], res[k]["c"][j]
            row[f"R2full_{k}"] = res[k]["R2_full_by_feature"][j]
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\npurity slope b (per SD; + = rises as tumour falls) and shared-axis loading c")
    print(df[["feature"] + [f"b_{k}" for k in ["twin"] + pats]].round(2).to_string(index=False))
    print(df[["feature"] + [f"c_{k}" for k in ["twin"] + pats]].round(2).to_string(index=False))
    df.to_csv(os.path.join(a.out, "loadings.csv"), index=False)
    with open(os.path.join(a.out, "equation.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {a.out}/equation.json and loadings.csv")


if __name__ == "__main__":
    main()
