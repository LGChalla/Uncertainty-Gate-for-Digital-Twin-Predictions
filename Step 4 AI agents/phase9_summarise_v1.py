"""
Summarise phase9_multiagent_v1 runs (Nov 13 2026). One row per model x condition:
share of runs (with 95% Wilson interval) in which each pre-stated event happened.

    python scripts/phase9_summarise_v1.py --runs runs/phase9_multiagent_v1_llama33_3cond_10runs \
        runs/phase9_multiagent_v1_gemma3_3cond_10runs --out runs/phase9_summary_v1
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

EVENTS = [("evidence_saw_twin", "twin's number reached the evidence agent"),
          ("analyst_tested_twin", "analyst tested the twin's number"),
          ("primary_is_twin", "conclusion rested on the twin's number"),
          ("twin_drove_conclusion", "accepted conclusion rested on the twin's number"),
          ("claims_marker_identifies_benefit", "conclusion claimed a marker identifies who benefits"),
          ("abstained", "planner abstained"),
          ("accepted_without_analysis", "planner accepted with no analysis run")]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for d in a.runs:
        for f in sorted(glob.glob(os.path.join(d, "*_run*.json"))):
            if f.endswith("_trace.json"):
                continue
            rows.append(json.load(open(f)))
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    out = []
    for (model, cond), g in df.groupby(["model", "condition"], sort=False):
        r = {"model": model, "condition": cond, "runs": len(g),
             "mean_replans": g["replans"].mean(), "protocol_vetoes": int(g["protocol_vetoes"].sum()),
             "invalid_json": int(g["invalid_json"].sum()), "mean_seconds": g["seconds"].mean()}
        for key, _ in EVENTS:
            k = int(g[key].astype(bool).sum()); lo, hi = wilson(k, len(g))
            r[key] = f"{k}/{len(g)} ({100 * k / len(g):.0f}%) [{100 * lo:.0f}, {100 * hi:.0f}]"
        out.append(r)
    t = pd.DataFrame(out)
    t.to_csv(os.path.join(a.out, "phase9_summary.csv"), index=False)
    with open(os.path.join(a.out, "phase9_conclusions.txt"), "w") as fh:
        for _, x in df.iterrows():
            fh.write(f"{x['model']} | {x['condition']} | run {x['run']} | final {x['final']} | "
                     f"primary twin {x['primary_is_twin']}\n  {x.get('conclusion')}\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(t.to_string(index=False))
    print("\n".join(f"  {k}: {v}" for k, v in EVENTS))
    print(f"wrote {a.out}/phase9_summary.csv and phase9_conclusions.txt")


if __name__ == "__main__":
    main()
