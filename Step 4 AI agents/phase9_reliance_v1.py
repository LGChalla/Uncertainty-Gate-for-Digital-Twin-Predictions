"""
Phase 9 follow-up (Nov 13 2026): did the twin's number reach the conclusion THROUGH the
evidence agent? Reads the trace files written by phase9_multiagent_v1.py; no model calls.

Per run:
  rho_twin     Spearman correlation of the evidence agent's scores with the twin's B_i
               (median per patient), over the cohort the evidence agent scored
  mean_score   mean evidence score
  primary      the marker the analyst's conclusion rested on, and its interaction HR
Per model x condition: mean rho_twin, and reliance = mean rho_twin minus the enforced
condition's (where the evidence agent never saw the number), with a bootstrap interval
over runs.

    python scripts/phase9_reliance_v1.py --runs runs/phase9_multiagent_v1_llama33_3cond_10runs \
        runs/phase9_multiagent_v1_gemma3_3cond_10runs \
        --bi runs/phase6_bi_direct_v2_sharedworld_floorboth --out runs/phase9_reliance_v1_llama33_gemma3
"""
import argparse, glob, importlib.util, json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p7", os.path.join(HERE, "phase7_llm_predict_v5.py"))
p7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--bi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    z = np.load(os.path.join(a.bi, "bi_GSE14814.npz"), allow_pickle=True)
    bmed = pd.Series(np.nanmedian(z["bi"], 1), index=list(z["samples"]))

    rows = []
    for d in a.runs:
        for f in sorted(glob.glob(os.path.join(d, "*_run??.json"))):
            m = json.load(open(f))
            tr = json.load(open(f[:-5] + "_trace.json"))
            ev = [c for c in tr["calls"] if c.get("role") == "evidence"]
            res = next((e["data"] for e in reversed(tr["memory"]) if e["kind"] == "results"), {}) or {}
            r = {"model": m["model"].split("/")[-1], "condition": m["condition"], "run": m["run"],
                 "final": m["final"], "claims_benefit": m["claims_marker_identifies_benefit"]}
            if ev:
                sc = pd.Series({s: p7.parse_score(t) for s, t in ev[-1]["replies"].items()}).dropna()
                b = bmed.reindex(sc.index)
                ok = b.notna()
                r["n_scored"] = int(ok.sum())
                r["rho_twin"] = float(spearmanr(sc[ok], b[ok])[0]) if ok.sum() > 5 else np.nan
                r["mean_score"] = float(sc.mean())
                r["sd_score"] = float(sc.std())
            for k, v in res.items():
                r[f"HR_{k}"] = v.get("HR_interaction_per_SD")
                r[f"CIlo_{k}"] = v.get("CI_low"); r[f"CIhi_{k}"] = v.get("CI_high")
            r["conclusion"] = m.get("conclusion")
            rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "phase9_reliance_per_run.csv"), index=False)

    rng = np.random.default_rng(0)
    out = []
    for model, g in df.groupby("model", sort=False):
        base = g[g.condition == "enforced"]["rho_twin"].dropna().to_numpy()
        for cond, h in g.groupby("condition", sort=False):
            x = h["rho_twin"].dropna().to_numpy()
            rel = x.mean() - base.mean()
            bs = [rng.choice(x, len(x)).mean() - rng.choice(base, len(base)).mean() for _ in range(a.boot)]
            hr = h.get("HR_evidence_score")
            out.append({"model": model, "condition": cond, "runs": len(h),
                        "rho_twin_mean": round(float(x.mean()), 3),
                        "reliance_vs_enforced": round(float(rel), 3),
                        "ci_low": round(float(np.percentile(bs, 2.5)), 3),
                        "ci_high": round(float(np.percentile(bs, 97.5)), 3),
                        "mean_score": round(float(h["mean_score"].mean()), 1),
                        "evidence_HR_median": round(float(hr.median()), 2) if hr is not None and hr.notna().any() else None,
                        "claims_benefit": int(h["claims_benefit"].sum()),
                        "abstained": int((h["final"] == "abstain").sum())})
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(a.out, "phase9_reliance_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(t.to_string(index=False))
    print("\nrho_twin: correlation of the evidence agent's scores with the twin's B_i."
          "\nreliance_vs_enforced: that correlation minus the enforced condition's (number never seen).")
    print(f"wrote {a.out}/phase9_reliance_summary.csv and phase9_reliance_per_run.csv")


if __name__ == "__main__":
    main()
