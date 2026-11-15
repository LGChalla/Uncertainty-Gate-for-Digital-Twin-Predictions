"""
Which extra measurement would let the twin tell patients' chemo responses
apart? ("Which measurement closes the gap", LC decision Nov 8.)
Uses only the existing campaign; no new simulations; CPU.

    python scripts/phase7_value_of_information_v1.py --sims runs/phase5_v3_fib120_sims \
        --out runs/phase7_value_of_information_v1

The question, made measurable
-----------------------------
For the 4,504 simulated tissues with a paired chemo run, we know both what a
patient's data would show and the true B_i. A flexible regressor (gradient
boosting) is trained to predict B_i from the data a patient would have; its
held-out R^2 is how much of B_i that data can explain. We compare:

  rna             the 12 bulk-RNA features, with the same deconvolution noise
                  used in fitting (15% on cell fractions, 25-30% on pathway scores)
  rna + X         the same, plus one more measurement X with realistic noise

  X (each alone, then the best pairs):
    vessels       vessel density, as from CD31 immunohistochemistry or perfusion
                  imaging (20% noise)
    proliferation tumour division rate, as from Ki-67 (30% noise)
    hypoxia_img   hypoxic share of cells, as from hypoxia PET such as FMISO (20%)
    tumour_burden starting tumour fraction, as from pathology/imaging (20%)
    drug_assay    the tissue's cisplatin kill sensitivity, as from an ex vivo
                  (organoid) drug assay (30%). In the twin this parameter is
                  shared by all patients and only known as a range; a real
                  assay would measure it per patient. Included because the
                  sensitivity ranking says it drives B_i most.

  ceiling         all 36 parameters known exactly: the most any patient data
                  could explain. The rest is run-to-run chance (where cells
                  happen to sit), which no measurement can remove.

Reported: held-out R^2 for B_i and its share of the ceiling, with a 5-fold
cross-validated spread. Every number is about the twin, not about patients:
it says which measurement the MODEL needs, which is the hypothesis a
prospective study would test.

Writes <out>/value_of_information.csv and .json.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tme_twin import priors_v4 as P                                   # noqa: E402
from tme_twin.observe_v2 import PLANS, DEFAULT_PLAN                   # noqa: E402

PRIMARY_HOUR = 192
EXTRAS = {  # name: (source, key, relative noise, plain description)
    "vessels": ("theta", "vessel_density", 0.20, "vessel density (CD31 IHC / perfusion imaging)"),
    "proliferation": ("theta", "r_div", 0.30, "tumour proliferation (Ki-67)"),
    "hypoxia_img": ("state", "hypoxic_fraction", 0.20, "hypoxic share (hypoxia PET)"),
    "tumour_burden": ("theta", "f0_tumour", 0.20, "tumour burden (pathology / imaging)"),
    "drug_assay": ("theta", "k_kill", 0.30, "ex vivo cisplatin sensitivity (organoid assay)"),
}


def noisy_features(x, feats, rng):
    plan = PLANS[DEFAULT_PLAN]
    out = np.empty_like(x)
    for j, f in enumerate(feats):
        cv = plan["direct"].get(f, plan["linked"].get(f, 0.2))
        v = np.clip(x[:, j], 1e-6, None)
        out[:, j] = np.log(v) + rng.normal(0, cv, len(v))     # multiplicative noise, on the log scale
    return out


def cv_r2(X, y, seed, folds=5):
    scores = []
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        scores.append(1 - np.sum((y[te] - p) ** 2) / np.sum((y[te] - y[te].mean()) ** 2))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", required=True)
    ap.add_argument("--hour", type=int, default=PRIMARY_HOUR)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    z = np.load(os.path.join(a.sims, "sims.npz"), allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    names, feats, state_names = list(z["names"]), list(z["features"]), list(meta["state"])
    col = list(meta["read_hours"]).index(a.hour)
    ok = np.isfinite(z["bi"][:, col]) & np.isfinite(z["x"]).all(1)
    th, x, st, y = z["theta"][ok], z["x"][ok], z["state"][ok], z["bi"][ok, col]
    print(f"{len(y)} simulated tissues with features and B_i at {a.hour} h; B_i SD {y.std():.3f}")

    base = noisy_features(x, feats, rng)
    extra = {}
    for k, (src, key, cv, _) in EXTRAS.items():
        v = th[:, names.index(key)] if src == "theta" else st[:, state_names.index(key)]
        extra[k] = (np.log(np.clip(v, 1e-6, None)) + rng.normal(0, cv, len(v)))[:, None]
    pri = P.regime_priors(meta["regime"])
    theta_scaled = np.column_stack([np.log(th[:, j]) if pri[n][2] == "log" else th[:, j]
                                    for j, n in enumerate(names)])

    sets = [("rna", [])] + [(f"rna + {k}", [k]) for k in EXTRAS]
    rows = []
    ceil, ceil_sd = cv_r2(theta_scaled, y, a.seed)
    for label, ks in sets:
        X = np.hstack([base] + [extra[k] for k in ks])
        r2, sd = cv_r2(X, y, a.seed)
        rows.append({"data": label, "extra": ks, "r2": r2, "r2_sd": sd, "share_of_ceiling": r2 / ceil})
        print(f"  {label:<26} R^2 {r2:.3f} ± {sd:.3f}   ({100 * r2 / ceil:.0f}% of the ceiling)", flush=True)
    single = sorted(rows[1:], key=lambda r: -r["r2"])
    top = [r["extra"][0] for r in single[:3]]
    for i in range(len(top)):
        for k in range(i + 1, len(top)):
            ks = [top[i], top[k]]
            X = np.hstack([base] + [extra[q] for q in ks])
            r2, sd = cv_r2(X, y, a.seed)
            rows.append({"data": "rna + " + " + ".join(ks), "extra": ks, "r2": r2, "r2_sd": sd,
                         "share_of_ceiling": r2 / ceil})
            print(f"  {'rna + ' + ' + '.join(ks):<26} R^2 {r2:.3f} ± {sd:.3f}   ({100 * r2 / ceil:.0f}% of the ceiling)",
                  flush=True)
    X = np.hstack([base] + [extra[k] for k in EXTRAS])
    r2, sd = cv_r2(X, y, a.seed)
    rows.append({"data": "rna + all five", "extra": list(EXTRAS), "r2": r2, "r2_sd": sd, "share_of_ceiling": r2 / ceil})
    print(f"  {'rna + all five':<26} R^2 {r2:.3f} ± {sd:.3f}   ({100 * r2 / ceil:.0f}% of the ceiling)")
    print(f"  {'ceiling (all 36 known)':<26} R^2 {ceil:.3f} ± {ceil_sd:.3f}")
    print(f"\n  the rest, {1 - ceil:.2f} of B_i variance, is run-to-run chance plus what the regressor misses; no patient measurement removes it")

    import csv
    with open(os.path.join(a.out, "value_of_information.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["data", "r2", "r2_sd", "share_of_ceiling"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    with open(os.path.join(a.out, "value_of_information.json"), "w") as fh:
        json.dump({"n": int(len(y)), "hour": a.hour, "ceiling_r2": ceil, "ceiling_sd": ceil_sd,
                   "extras": {k: {"source": v[1], "noise": v[2], "what": v[3]} for k, v in EXTRAS.items()},
                   "rows": rows}, fh, indent=2)
    print(f"wrote {a.out}/value_of_information.csv and .json")


if __name__ == "__main__":
    main()
