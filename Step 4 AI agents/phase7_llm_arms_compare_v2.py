"""
Version 2 of the outcome-free arm comparison (v1 unchanged). Takes any set of arm files,
so it serves the primary run (v3, 300 tokens), the enforced arm, the 1,000-token run and
the second model alike.

    python scripts/phase7_llm_arms_compare_v2.py --bi runs/phase6_bi_direct_v2_sharedworld_floorboth \
        --arm features=<file.jsonl> --arm features+twin=<file.jsonl> \
        --arm features+twin+guardrail=<file.jsonl> --arm features+twin+enforced=<file.jsonl> \
        --out runs/phase7_llm_arms_compare_v2_<condition>

Per arm, as v1: SD of patient scores across patients; Spearman with the twin's B_i
percentile; keyword share (same fixed list); invalid replies; reproducibility over
repeats. Added: median thinking tokens and stop reasons (when recorded).
Paired bootstrap (2,000) differences: every twin arm vs features+twin, and
enforced vs guardrail when both are given.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

KEYWORDS = ["uncertain", "cannot", "unreliable", "not reliable", "rank", "identif", "limited"]
TWIN_WORDS = ["twin", "b_i", "in silico", "in-silico"]


def reason_of(reply):
    m = re.search(r"\{.*\}", reply or "", re.S)
    try:
        return str(json.loads(m.group(0)).get("reason", "")) if m else (reply or "")
    except Exception:
        return reply or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="label=path/to/file.jsonl")
    ap.add_argument("--bi", required=True)
    ap.add_argument("--cohort", default="GSE14814")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    files = dict(x.split("=", 1) for x in a.arm)
    labels = list(files)

    rows = []
    for lab, f in files.items():
        for r in map(json.loads, open(f)):
            reason = reason_of(r["reply"]).lower()
            tt = r.get("thinking_tokens")
            if tt is None:
                tt = ((r.get("info", {}).get("usage") or {}).get("output_tokens_details") or {}).get("thinking_tokens")
            rows.append({"arm": lab, "sample": r["sample"], "score": r["score"], "model": r.get("model_reported"),
                         "keyword": any(k in reason for k in KEYWORDS), "twin": any(k in reason for k in TWIN_WORDS),
                         "thinking": tt, "stop": r.get("stop_reason") or r.get("info", {}).get("stop_reason")})
    df = pd.DataFrame(rows)
    z = np.load(os.path.join(a.bi, f"bi_{a.cohort}.npz"), allow_pickle=True)
    bi_pct = pd.Series({s: np.nanmedian(v) for s, v in zip(z["samples"], z["bi"])}).rank(pct=True) * 100
    pp = df.groupby(["sample", "arm"])["score"].mean().unstack("arm")[labels]
    pp = pp.loc[pp.index.isin(bi_pct.index)].dropna()
    bp = bi_pct.loc[pp.index].to_numpy()
    S = {k: pp[k].to_numpy() for k in labels}
    twin_arms = [k for k in labels if k != "features"]

    def summ(S, bp):
        o = {f"sd|{k}": float(np.std(v, ddof=1)) for k, v in S.items()}
        o.update({f"rho|{k}": float(stats.spearmanr(S[k], bp).statistic) for k in labels})   # features too: the baseline, since B_i is itself a function of the features
        ref = "features+twin"
        for k in twin_arms:
            if k != ref and ref in S:
                o[f"sd diff|{k} - {ref}"] = o[f"sd|{k}"] - o[f"sd|{ref}"]
                o[f"rho diff|{k} - {ref}"] = o[f"rho|{k}"] - o[f"rho|{ref}"]
        g, e = "features+twin+guardrail", "features+twin+enforced"
        if g in S and e in S:
            o[f"sd diff|{e} - {g}"] = o[f"sd|{e}"] - o[f"sd|{g}"]
            o[f"rho diff|{e} - {g}"] = o[f"rho|{e}"] - o[f"rho|{g}"]
        if "features" in S:
            for k in twin_arms:
                o[f"rho diff|{k} - features"] = o[f"rho|{k}"] - o["rho|features"]
        return o

    point = summ(S, bp)
    n = len(pp)
    B = pd.DataFrame([summ({k: v[i] for k, v in S.items()}, bp[i])
                      for i in (rng.integers(0, n, n) for _ in range(a.boot))])
    ci = {k: [float(B[k].quantile(.025)), float(B[k].quantile(.975))] for k in point}

    print(f"{n} patients with every arm\n")
    print(f"{'arm':>26} {'calls':>6} {'invalid':>8} {'mean':>6} {'SD across patients':>24} {'rho with B_i pct':>24} "
          f"{'keyword':>8} {'twin':>6} {'think med':>10}")
    per = {}
    for k in labels:
        d = df[df.arm == k]
        th = pd.to_numeric(d["thinking"], errors="coerce")
        per[k] = {"calls": int(len(d)), "invalid": int(d["score"].isna().sum()), "mean": float(pp[k].mean()),
                  "keyword": float(d["keyword"].mean()), "twin": float(d["twin"].mean()),
                  "thinking_median": float(th.median()) if th.notna().any() else None,
                  "stop_reasons": d["stop"].fillna("not recorded").value_counts().to_dict(),
                  "models": sorted(set(map(str, d["model"]))),
                  "within_patient_sd": float(d.groupby("sample")["score"].std().mean())}
        sd = f"{point['sd|' + k]:.1f} [{ci['sd|' + k][0]:.1f}, {ci['sd|' + k][1]:.1f}]"
        rho = (f"{point['rho|' + k]:+.2f} [{ci['rho|' + k][0]:+.2f}, {ci['rho|' + k][1]:+.2f}]" if f"rho|{k}" in point else "-")
        tm = f"{per[k]['thinking_median']:.0f}" if per[k]["thinking_median"] is not None else "-"
        print(f"{k:>26} {per[k]['calls']:>6} {per[k]['invalid']:>8} {per[k]['mean']:>6.1f} {sd:>24} {rho:>24} "
              f"{per[k]['keyword']:>8.2f} {per[k]['twin']:>6.2f} {tm:>10}")
    print("\ndifferences (95% paired bootstrap interval)")
    for k in point:
        if "diff" in k:
            print(f"  {k:<62} {point[k]:+.2f} [{ci[k][0]:+.2f}, {ci[k][1]:+.2f}]")
    print("\nstop reasons, models, within-patient SD")
    for k in labels:
        print(f"  {k:>26}: {per[k]['stop_reasons']}  {per[k]['models']}  {per[k]['within_patient_sd']:.2f}")
    pp.assign(bi_percentile=bp).to_csv(os.path.join(a.out, "per_patient.csv"))
    with open(os.path.join(a.out, "arms.json"), "w") as fh:
        json.dump({"files": files, "n_patients": n, "point": point, "ci95": ci, "per_arm": per}, fh, indent=2)
    print(f"\nwrote {a.out}/arms.json and per_patient.csv")


if __name__ == "__main__":
    main()
