"""
Coverage check v5: v4 plus --logit-features, which linked features get logit
when --transform logit. Default: hypoxia only, the choice made on Nov 6 (logit
on m2_share created a new miss in three cohorts where log had none).
phase5_coverage_check_v4.py is unchanged.

    python scripts/phase5_coverage_check_v5.py --sims runs/phase5_v3_fib120_sims_a ... \
        --transform logit --endothelial linked --out runs/phase5_v3_fib120_coverage_partial

--- v4 docstring ---

  --transform log    link = offset + scale * log(twin)       (as v2, v3)
  --transform logit  link = offset + scale * logit(twin) for the linked shares
                     bounded by 1 (hypoxia, m2_share, caf_share). The twin's
                     hypoxic fraction cannot exceed 1 and piles up near it
                     (90th percentile 0.82, Nov 6); log maps that ceiling to a
                     hard limit the patients' GSVA scores need not respect, and
                     12-15% of patients sat above it in every cohort. logit
                     stretches values near 1 instead of squashing them. Linked
                     quantities that are not shares (tgfb, il6, f_endothelial
                     when linked) keep log.

Results on the Nov 6 partial checkpoints (v3): endothelial compared within
cohort removed the endothelial miss in all four cohorts; the robust link did not
remove the hypoxia miss.

  --link moment   offset/scale from mean and SD of log(twin) (as v2)
  --link robust   offset/scale from median and IQR, which a pile-up of tissues
                  at the floor (e.g. 18% with zero hypoxic cells, Nov 6) cannot bend
  --endothelial direct   f_endothelial compared as a fraction (as v2)
  --endothelial linked   f_endothelial through the fitted link: within-cohort
                  ordering is kept, a cohort-wide offset is not (EPIC endothelial
                  reads 3x higher in GSE42127 than TCGA; bulk samples include
                  capillary-rich normal lung)

    python scripts/phase5_coverage_check_v3.py --sims runs/phase5_v2_sims_a ... \
        --link robust --endothelial linked --out runs/phase5_v2_coverage_robust_endolinked

    python scripts/phase5_coverage_check_v2.py \
        --sims runs/phase5_v2_sims_a runs/phase5_v2_sims_b runs/phase5_v2_sims_c runs/phase5_v2_sims_d \
        --out runs/phase5_v2_coverage_partial --cohorts GSE14814 GSE68465 GSE42127 TCGA-LUAD

Each --sims folder is read from sims.npz if the shard has finished, otherwise
from partial.npz (its latest checkpoint). Nothing is written into the shards.

Like-for-like comparison. Patients' features carry deconvolution error; raw
simulations do not. The same noise the training script adds (observe_v2 PLANS:
a coefficient of variation on the fractions, a relative error carried through
the link's scale on the linked features) is added to the simulations before
comparing, so a correct model is not penalised for the patients' measurement
noise. --no-noise compares raw simulations instead.

Pass line (preregistration v0.4, Section 7): at most 10% of patients lie beyond
the simulations' own 95th-percentile k-nearest-neighbour distance. About 5%
would exceed it even if patients and simulations came from one distribution.

Per feature it also reports which way patients miss: the share of patients
below the simulations' 1st percentile and above their 99th. A feature where
patients sit below the cloud means the prior reaches too high, and the reverse.
That is what decides which prior to change if the check fails.

Writes <out>/coverage.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tme_twin.observe_v2 import FEATURES, PLANS, DEFAULT_PLAN   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS_SHARE = 0.10


def load(dirs):
    xs, fl, src, meta = [], [], [], None
    for d in dirs:
        for name in ("sims.npz", "partial.npz"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                z = np.load(p, allow_pickle=True)
                if len(z["x"]) == 0:
                    break
                m = json.loads(str(z["meta"]))
                if m.get("observe") != "observe_v2":
                    raise SystemExit(f"{p} is not a v2 simulation file")
                meta = meta or m
                xs.append(z["x"]); fl.append(z["floors"])
                src.append(f"{d}/{name}: {len(z['x'])} draws")
                break
        else:
            src.append(f"{d}: nothing yet")
    if not xs:
        raise SystemExit("no draws found in any --sims folder yet")
    return np.concatenate(xs), np.concatenate(fl), meta, src


BOUNDED = ("hypoxia", "m2_share", "caf_share")


def transform(sim_linked, names, how, logit_set=BOUNDED):
    v = np.clip(sim_linked, 1e-12, None)
    out = np.log(v)
    if how == "logit":
        for j, fn in enumerate(names):
            if fn in logit_set:
                p = np.clip(v[:, j], 1e-6, 1 - 1e-6)
                out[:, j] = np.log(p) - np.log1p(-p)
    return out


def fit_link(ls, obs_linked, how="moment"):
    if how == "robust":
        iq = lambda v: np.percentile(v, 75, 0) - np.percentile(v, 25, 0)
        scale = iq(obs_linked) / np.clip(iq(ls), 1e-12, None)
        offset = np.median(obs_linked, 0) - scale * np.median(ls, 0)
    else:
        scale = obs_linked.std(0) / np.clip(ls.std(0), 1e-12, None)
        offset = obs_linked.mean(0) - scale * ls.mean(0)
    return offset, scale


def knn_mean(q, pool, k, skip_self):
    out = np.empty(len(q))
    for i in range(len(q)):
        d = np.sort(np.sqrt(((pool - q[i]) ** 2).sum(1)))
        out[i] = (d[1:k + 1] if skip_self else d[:k]).mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", nargs="+", required=True)
    ap.add_argument("--plan", default=DEFAULT_PLAN, choices=list(PLANS))
    ap.add_argument("--out", default="runs/phase5_v2_coverage")
    ap.add_argument("--cohorts", nargs="*", default=["GSE14814", "GSE68465", "GSE42127", "TCGA-LUAD"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--link", choices=["moment", "robust"], default="moment")
    ap.add_argument("--transform", choices=["log", "logit"], default="log")
    ap.add_argument("--logit-features", nargs="+", default=["hypoxia"], choices=list(BOUNDED))
    ap.add_argument("--endothelial", choices=["direct", "linked"], default="direct")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    x, floors, meta, src = load(a.sims)
    feats = list(meta["features"]) if "features" in meta else list(FEATURES)
    for k, fn in enumerate(meta["floors"]):
        j = feats.index(fn)
        x[:, j] = np.maximum(x[:, j], floors[:, k])
    x = x[np.isfinite(x).all(1)]
    print("\n".join("  " + s for s in src))
    print(f"{len(x)} usable simulations; grid {meta['size']}, priors {meta['priors_module']}, "
          f"noise {'off' if a.no_noise else 'on'}; link {a.link}; transform {a.transform} on {a.logit_features if a.transform == 'logit' else 'none'}; endothelial {a.endothelial}")

    plan = {"direct": dict(PLANS[a.plan]["direct"]), "linked": dict(PLANS[a.plan]["linked"])}
    if a.endothelial == "linked":
        plan["linked"]["f_endothelial"] = plan["direct"].pop("f_endothelial") * 2  # CV 0.15 -> 0.30 latent
    direct = [f for f in FEATURES if f in plan["direct"]]
    linked = [f for f in FEATURES if f in plan["linked"]]
    used = direct + linked
    sim_direct = x[:, [feats.index(f) for f in direct]]
    sim_linked = x[:, [feats.index(f) for f in linked]]
    rep = {"sims": src, "n_sims": int(len(x)), "noise": not a.no_noise, "pass_share": PASS_SHARE,
           "link": a.link, "transform": a.transform, "logit_features": list(a.logit_features), "endothelial": a.endothelial,
           "cohorts": {}}

    for ds in a.cohorts:
        f = os.path.join(ROOT, "data", "processed", ds, "features_v2_qc.csv")
        if not os.path.exists(f):
            print(f"\n{ds}: no features_v2_qc.csv, skipping")
            continue
        obs = pd.read_csv(f, index_col=0)
        xo = np.concatenate([obs[direct].to_numpy(), obs[linked].to_numpy()], 1)
        tl = transform(sim_linked, linked, a.transform, tuple(a.logit_features))
        off, sc = fit_link(tl, obs[linked].to_numpy(), a.link)
        xs = np.concatenate([sim_direct, off + sc * tl], 1)
        if not a.no_noise:
            sd = np.empty_like(xs)
            for j, fn in enumerate(used):
                sd[:, j] = (plan["linked"][fn] * abs(sc[linked.index(fn)]) if fn in linked
                            else np.maximum(plan["direct"][fn] * np.abs(xs[:, j]), 1e-4))
            xs = xs + rng.normal(0, 1, xs.shape) * sd

        print(f"\n=== {ds}: {len(xo)} patients ===")
        print(f"{'feature':>14} {'sim p1':>9} {'sim p50':>9} {'sim p99':>9} {'pat p50':>9} "
              f"{'below p1':>9} {'above p99':>10}")
        per = {}
        for j, fn in enumerate(used):
            s, o = xs[:, j], xo[:, j]
            p1, p50, p99 = np.percentile(s, [1, 50, 99])
            below, above = float(np.mean(o < p1)), float(np.mean(o > p99))
            per[fn] = {"sim_p1": float(p1), "sim_p50": float(p50), "sim_p99": float(p99),
                       "patient_p50": float(np.median(o)), "below_sim_p1": below, "above_sim_p99": above}
            mark = ""
            if below > 0.10:
                mark = "  <-- patients BELOW the cloud (prior reaches too high)"
            elif above > 0.10:
                mark = "  <-- patients ABOVE the cloud (prior does not reach high enough)"
            print(f"{fn:>14} {p1:>9.4f} {p50:>9.4f} {p99:>9.4f} {np.median(o):>9.4f} "
                  f"{below:>9.2f} {above:>10.2f}{mark}")

        med = np.median(xs, 0)
        iqr = np.percentile(xs, 75, 0) - np.percentile(xs, 25, 0)
        iqr = np.where(iqr > 0, iqr, 1.0)
        zs, zo = (xs - med) / iqr, (xo - med) / iqr
        ref = zs[rng.choice(len(zs), size=min(2000, len(zs)), replace=False)]
        d_sim = knn_mean(ref[:500], ref, a.k, True)
        d_obs = knn_mean(zo, ref, a.k, False)
        q95 = float(np.percentile(d_sim, 95))
        outside = float(np.mean(d_obs > q95))
        ok = outside <= PASS_SHARE
        print(f"  joint {a.k}-NN distance: simulations p95 {q95:.2f}; "
              f"patients median {np.median(d_obs):.2f}")
        print(f"  patients beyond the simulations' own p95: {outside:.1%}  "
              f"-> {'PASS' if ok else 'FAIL'} (limit {PASS_SHARE:.0%})")
        missing = {fn: ("below" if v["below_sim_p1"] > 0.10 else "above")
                   for fn, v in per.items() if v["below_sim_p1"] > 0.10 or v["above_sim_p99"] > 0.10}
        if missing:
            print(f"  features patients miss: {missing}")
        rep["cohorts"][ds] = {"n_patients": int(len(xo)), "per_feature": per,
                              "link_offset": off.tolist(), "link_scale": sc.tolist(),
                              "d_sim_p95": q95, "share_outside": outside, "pass": ok,
                              "features_missed": missing}

    with open(os.path.join(a.out, "coverage.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"\nwrote {os.path.join(a.out, 'coverage.json')}")


if __name__ == "__main__":
    main()
