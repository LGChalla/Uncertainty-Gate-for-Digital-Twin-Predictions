"""
Version 4 of phase8_outcomes (v1-v3 unchanged), Nov 11, 2026. POST HOC (added after the
pre-registered outcome analysis had been seen; docs/amendments_after_outcomes_v1.md):
  --all-histologies   use every JBR.10 patient with RNA, treatment and survival, not only
                      adenocarcinoma; histology (adenocarcinoma yes/no) is added as a
                      covariate. Randomisation holds for all histologies.
Everything else as v3.

--- v3 docstring ---
Version 3 of phase8_outcomes (v1, v2 unchanged), Nov 11, 2026 (pre-registration v0.13,
OSF registration osf.io/vzqc5). Changes, none touching the analysis itself:
  - --llm takes several folders (agent runs v3, v4, v5 are in separate folders) and
    matches files named *_<cohort>[_tokN]_vK.csv; pilot files are skipped.
  - TumorTwin is used only for cohorts where its file exists.
  - The SHA-256 of --freeze is checked against the last freeze entry in runs/runlog.jsonl;
    the run stops if they differ.
  - --osf is written into the report.

--- v2 docstring ---

Version 2 (v1 unchanged), for pre-registration v0.5:
  - twin B_i is optional per cohort (direct simulation exists for JBR.10 only);
    a cohort without bi_<cohort>.npz runs the other markers and says so.
  - PC1 orientation: where the cohort has twin B_i, PC1 is oriented to correlate
    positively with it (as v0.4). Where it has none, PC1's loading vector is
    aligned in sign with the reference cohort's oriented loadings (--pc1-ref,
    default GSE14814), so "high PC1" means the same mix of features everywhere.
  - --precondition-failed marks Exp 4 as exploratory in every printed line and
    in the report (v0.5: 0.047 between vs 0.144 within).

--- v1 docstring ---
Phase 8: does chemotherapy help more where the marker says it should?
Run ONLY after the pre-registration is frozen; this is the first script that
reads treatment and survival together.

    python scripts/phase8_outcomes_v1.py --freeze docs/preregistration_v0.5.md \
        --twin runs/phase6_bi_emulator_v1 --tumortwin runs/phase7_tumortwin_v1 \
        --llm runs/phase7_llm_v1 --out runs/phase8_outcomes_v1

Markers compared, every one through the same test
-------------------------------------------------
  twin          B_i from our twin, 50 posterior draws per patient
  tumortwin     B_TT, TumorTwin's tumour-only equation with the same inputs (50 draws)
  llm:<file>    every <provider>_<model>_<arm>_<cohort>.csv in --llm: one score per
                patient (mean of repeats). Arms: features, features+twin.
  pc1           first principal component of the 12 features, sign set so it
                correlates positively with the twin's B_i (pre-registration v0.4)

The test
--------
Cox proportional hazards for overall survival:
    h(t) = h0(t) exp(b1 chemo + b2 z + b3 chemo*z + covariates)
z is the marker standardised within the cohort (per SD). b3 < 0 means chemo
lowers the hazard more as the marker rises: the marker picks out who benefits.
Reported: exp(b3) per SD with 95% CI and p, and, for reading, the hazard
ratio of chemo in patients above vs below the cohort median of the marker.

Draws. For markers with posterior draws (twin, tumortwin) the Cox model is fit
once per draw and pooled by Rubin's rules: total variance = mean within-draw
variance + (1 + 1/M) x between-draw variance; df by Barnard-Rubin.

Cohorts. Adenocarcinoma only, treatment known, survival known.
  GSE14814 (JBR.10): chemotherapy randomised -> primary test, covariates stage
    and age, no weighting.
  GSE68465, GSE42127, TCGA-LUAD: not randomised -> supporting. Inverse
    probability of treatment weights (stabilised, from stage, age and sex),
    a 180-day landmark (patients with an event or censoring before day 180
    are removed and time starts at day 180, so a patient cannot count as
    treated by surviving long enough to be treated), robust standard errors.
  Fixed-effect pooled b3 across the four cohorts is reported as well.

Freeze check. --freeze must point to the frozen pre-registration; its SHA-256
is written into the output, so the analysis can be matched to the plan.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["GSE14814", "GSE68465", "GSE42127", "TCGA-LUAD"]
PRIMARY = "GSE14814"
LANDMARK_MONTHS = 180 / 30.4375


def stage_group(x):
    s = str(x).upper()
    if s in ("NAN", "", "NONE"):
        return np.nan
    if re.search(r"IV|\b4", s):
        return 4.0
    if re.search(r"III|\b3", s):
        return 3.0
    if re.search(r"II|\b2", s):
        return 2.0
    if re.search(r"I|\b1", s):
        return 1.0
    return np.nan


ALL_HIST = False


def load_clinical(ds):
    f = os.path.join(ROOT, "data", "processed", ds, "clinical_harmonised.csv")
    c = pd.read_csv(f, index_col=0)
    if "stage" not in c.columns:
        for alt in ("ajcc_pathologic_stage", "tumor_stage"):
            if alt in c.columns:
                c["stage"] = c[alt]; break
    c["stage_g"] = c["stage"].map(stage_group) if "stage" in c.columns else np.nan
    c["age"] = pd.to_numeric(c.get("age"), errors="coerce")
    c["male"] = c.get("sex", pd.Series(index=c.index, dtype=object)).astype(str).str.lower().str.startswith("m").astype(float)
    c["adeno_i"] = c["adeno"].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)
    keep = (((c["adeno_i"] == 1) | ALL_HIST) & c["treated"].isin([0, 1])
            & c["os_months"].notna() & c["os_event"].notna() & (c["os_months"] > 0))
    return c[keep].copy()


def match_ids(marker_ids, clin_index):
    """Exact match; for TCGA also sample barcode -> case (first 12 characters)."""
    out = {}
    clin = set(clin_index)
    for s in marker_ids:
        if s in clin:
            out[s] = s
        elif str(s).startswith("TCGA") and str(s)[:12] in clin:
            out[s] = str(s)[:12]
    return out


def covariates(d, randomised):
    cov = []
    for c in ("stage_g", "age") + (() if randomised else ("male",)) + (("adeno_i",) if ALL_HIST else ()):
        if c in d and d[c].notna().mean() >= 0.7 and d[c].nunique() > 1:
            cov.append(c)
    return cov


def iptw(d, cov):
    from sklearn.linear_model import LogisticRegression
    X = d[cov].fillna(d[cov].median()).to_numpy()
    lr = LogisticRegression(max_iter=1000).fit(X, d["treated"])
    p = lr.predict_proba(X)[:, 1]
    pt = d["treated"].mean()
    w = np.where(d["treated"] == 1, pt / p, (1 - pt) / (1 - p))
    return np.clip(w, 0, np.quantile(w, 0.99))


def cox_once(d, z, cov, weights):
    from lifelines import CoxPHFitter
    df = d[["T", "E", "treated"] + cov].copy()
    df[cov] = df[cov].fillna(df[cov].median())
    df["z"] = z
    df["treated_x_z"] = df["treated"] * df["z"]
    kw = {}
    if weights is not None:
        df["w"] = weights; kw = {"weights_col": "w", "robust": True}
    cph = CoxPHFitter(penalizer=0.0).fit(df, "T", "E", **kw)
    b, se = cph.params_["treated_x_z"], cph.standard_errors_["treated_x_z"]
    hi = df["z"] > np.median(df["z"])
    sub = {}
    for g, m in (("high", hi), ("low", ~hi)):
        dd = df[m][["T", "E", "treated"] + cov + (["w"] if weights is not None else [])]
        try:
            c2 = CoxPHFitter().fit(dd, "T", "E", **kw)
            sub[g] = float(np.exp(c2.params_["treated"]))
        except Exception:
            sub[g] = np.nan
    return b, se, sub


def rubin(bs, ses):
    bs, ses = np.asarray(bs), np.asarray(ses)
    M = len(bs)
    qbar, W = bs.mean(), np.mean(ses ** 2)
    B = bs.var(ddof=1) if M > 1 else 0.0
    T = W + (1 + 1 / M) * B
    r = (1 + 1 / M) * B / max(W, 1e-12)
    df = (M - 1) * (1 + 1 / max(r, 1e-12)) ** 2 if M > 1 and B > 0 else np.inf
    return qbar, np.sqrt(T), df


def analyse(ds, clin, marker_name, ids, values):
    """values: (n_patients, n_draws) or (n_patients,)."""
    mp = match_ids(ids, clin.index)
    rows = [i for i, s in enumerate(ids) if s in mp]
    if len(rows) < 20:
        return None
    d = clin.loc[[mp[ids[i]] for i in rows]].copy()
    v = np.asarray(values, dtype=float)[rows]
    v = v[:, None] if v.ndim == 1 else v
    d["T"], d["E"] = d["os_months"].to_numpy(), d["os_event"].to_numpy()
    randomised = ds == PRIMARY
    if not randomised:
        lm = d["T"] > LANDMARK_MONTHS
        d, v = d[lm.to_numpy()].copy(), v[lm.to_numpy()]
        d["T"] = d["T"] - LANDMARK_MONTHS
    cov = covariates(d, randomised)
    w = None if randomised else iptw(d, cov)
    bs, ses, highs, lows = [], [], [], []
    for m in range(v.shape[1]):
        col = v[:, m]
        ok = np.isfinite(col)
        if ok.mean() < 0.9:
            continue
        col = np.where(ok, col, np.nanmedian(col))
        z = (col - col.mean()) / max(col.std(), 1e-12)
        b, se, sub = cox_once(d, z, cov, w)
        bs.append(b); ses.append(se); highs.append(sub["high"]); lows.append(sub["low"])
    if not bs:
        return None
    b, se, df = rubin(bs, ses)
    tcrit = stats.t.ppf(0.975, df) if np.isfinite(df) else 1.96
    p = 2 * (stats.t.sf(abs(b / se), df) if np.isfinite(df) else stats.norm.sf(abs(b / se)))
    return {"cohort": ds, "marker": marker_name, "design": "randomised" if randomised else "IPTW + 180-day landmark",
            "n": int(len(d)), "n_treated": int(d["treated"].sum()), "events": int(d["E"].sum()),
            "covariates": cov, "draws": len(bs),
            "b3": float(b), "se": float(se), "HR_interaction_per_SD": float(np.exp(b)),
            "CI_low": float(np.exp(b - tcrit * se)), "CI_high": float(np.exp(b + tcrit * se)), "p": float(p),
            "HR_chemo_marker_high": float(np.nanmedian(highs)), "HR_chemo_marker_low": float(np.nanmedian(lows))}


def pooled(rows):
    by = {}
    for r in rows:
        by.setdefault(r["marker"], []).append(r)
    out = []
    for m, rs in by.items():
        if len(rs) < 2:
            continue
        w = np.array([1 / r["se"] ** 2 for r in rs]); b = np.array([r["b3"] for r in rs])
        bp, sp = float((w * b).sum() / w.sum()), float(np.sqrt(1 / w.sum()))
        out.append({"cohort": "pooled (fixed effect)", "marker": m, "n": int(sum(r["n"] for r in rs)),
                    "b3": bp, "se": sp, "HR_interaction_per_SD": float(np.exp(bp)),
                    "CI_low": float(np.exp(bp - 1.96 * sp)), "CI_high": float(np.exp(bp + 1.96 * sp)),
                    "p": float(2 * stats.norm.sf(abs(bp / sp))), "cohorts": [r["cohort"] for r in rs]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True, help="the frozen pre-registration file")
    ap.add_argument("--twin", required=True, help="phase6_bi_emulator_v1 output (bi_<cohort>.npz)")
    ap.add_argument("--tumortwin", default=None, help="phase7_tumortwin_v1 output (btt_<cohort>.npz)")
    ap.add_argument("--llm", nargs="*", default=[], help="folders with agent score csv files")
    ap.add_argument("--osf", default=None, help="OSF registration link")
    ap.add_argument("--all-histologies", action="store_true", help="POST HOC: all histologies, histology as covariate")
    ap.add_argument("--no-pc1", action="store_true")
    ap.add_argument("--pc1-ref", default="GSE14814")
    ap.add_argument("--precondition-failed", action="store_true",
                    help="Exp 4 precondition failed (v0.5): label every result exploratory")
    ap.add_argument("--cohorts", nargs="*", default=DATASETS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    global ALL_HIST
    ALL_HIST = a.all_histologies
    if ALL_HIST:
        print("POST HOC analysis (not pre-registered): all histologies, histology as covariate")
    if not os.path.exists(a.freeze):
        raise SystemExit("--freeze file not found: freeze the pre-registration first")
    sha = hashlib.sha256(open(a.freeze, "rb").read()).hexdigest()
    os.makedirs(a.out, exist_ok=True)
    print(f"pre-registration {a.freeze} sha256 {sha[:16]}")
    log = os.path.join(ROOT, "runs", "runlog.jsonl")
    frz = [json.loads(l) for l in open(log) if '"freeze"' in l or '"amend_refreeze"' in l] if os.path.exists(log) else []
    if not frz or frz[-1].get("sha256") != sha:
        raise SystemExit("the frozen file does not match the last freeze recorded in runs/runlog.jsonl")
    print(f"  matches the last freeze in the run log ({frz[-1].get('from')}, {frz[-1].get('time')})"
          + (f"; OSF {a.osf}" if a.osf else ""))
    if a.precondition_failed:
        print("Exp 4 precondition failed (pre-registered): every result below is EXPLORATORY")

    rows = []
    ref_load = None
    order = [a.pc1_ref] + [d for d in a.cohorts if d != a.pc1_ref] if a.pc1_ref in a.cohorts else a.cohorts
    for ds in order:
        try:
            clin = load_clinical(ds)
        except FileNotFoundError:
            print(f"{ds}: no clinical_harmonised.csv, skipping"); continue
        print(f"\n=== {ds}: {len(clin)} {'patients (all histologies)' if ALL_HIST else 'adenocarcinoma patients'} with treatment and survival "
              f"({int(clin['treated'].sum())} treated) ===")
        markers = []
        twin_ids = twin_bi = None
        tf = os.path.join(a.twin, f"bi_{ds}.npz")
        if os.path.exists(tf):
            z = np.load(tf, allow_pickle=True)
            twin_ids, twin_bi = list(z["samples"]), z["bi"]
            markers.append(("twin B_i", twin_ids, twin_bi))
        else:
            print("  twin B_i: not computed for this cohort (v0.5, Exp 5)")
        if a.tumortwin and os.path.exists(os.path.join(a.tumortwin, f"btt_{ds}.npz")):
            t = np.load(os.path.join(a.tumortwin, f"btt_{ds}.npz"), allow_pickle=True)
            markers.append(("TumorTwin B_TT", list(t["samples"]), t["btt"]))
        for folder in a.llm:
            for f in sorted(glob.glob(os.path.join(folder, f"*_{ds}*.csv"))):
                if "pilot" in os.path.basename(f):
                    continue
                L = pd.read_csv(f, index_col=0)
                b = os.path.basename(f)[:-4]
                name = "LLM " + b.replace(f"_{ds}", "").replace("anthropic_", "").replace("local_", "").replace("_default", "")
                markers.append((name, list(L.index), L["llm_score"].to_numpy()))
        if not a.no_pc1:
            obs = pd.read_csv(os.path.join(ROOT, "data", "processed", ds, "features_v2_qc.csv"), index_col=0)
            X = obs.to_numpy(float); X = (X - X.mean(0)) / X.std(0).clip(1e-12)
            pc = np.linalg.svd(X, full_matrices=False)[2][0]
            if twin_bi is not None:
                bmed = pd.Series(np.nanmedian(twin_bi, 1), index=twin_ids).reindex(obs.index)
                if np.corrcoef((X @ pc)[bmed.notna().to_numpy()], bmed.dropna())[0, 1] < 0:
                    pc = -pc
                if ds == a.pc1_ref:
                    ref_load = pc
            elif ref_load is not None:
                if float(pc @ ref_load) < 0:
                    pc = -pc
            else:
                print("  PC1: no twin B_i and no reference orientation; sign left as computed")
            s = X @ pc
            markers.append(("PC1 of features", list(obs.index), s))
        for name, ids, vals in markers:
            r = analyse(ds, clin, name, ids, vals)
            if r is None:
                print(f"  {name}: too few matched patients"); continue
            if a.precondition_failed and ds == "GSE14814":
                r["status"] = "exploratory (Exp 4 precondition failed)"
            elif ds != "GSE14814":
                r["status"] = "exploratory (Exp 5)"
            rows.append(r)
            print(f"  {name:>45}: HR(chemo x marker) per SD {r['HR_interaction_per_SD']:.2f} "
                  f"[{r['CI_low']:.2f}, {r['CI_high']:.2f}], p {r['p']:.3f}; chemo HR marker-high "
                  f"{r['HR_chemo_marker_high']:.2f} vs low {r['HR_chemo_marker_low']:.2f}  (n {r['n']}, {r['design']})")
    pool = pooled(rows)
    if pool:
        print("\n=== pooled across cohorts (fixed effect) ===")
        for r in pool:
            print(f"  {r['marker']:>45}: HR per SD {r['HR_interaction_per_SD']:.2f} "
                  f"[{r['CI_low']:.2f}, {r['CI_high']:.2f}], p {r['p']:.3f}")
    pd.DataFrame(rows + pool).to_csv(os.path.join(a.out, "interaction_tests.csv"), index=False)
    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump({"freeze_file": a.freeze, "freeze_sha256": sha, "osf": a.osf, "script": "phase8_outcomes_v4", "post_hoc_all_histologies": bool(a.all_histologies), "primary_cohort": PRIMARY,
                   "exp4_precondition_failed": bool(a.precondition_failed),
                   "results": rows, "pooled": pool}, fh, indent=2)
    print(f"\nwrote {a.out}/interaction_tests.csv and report.json")


if __name__ == "__main__":
    main()
