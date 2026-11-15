"""
Phase 9 (added Nov 13 2026, after outcomes were opened on Nov 11; labelled as an added run).
Runs the multi-agent pipeline in tme_twin/pipeline_v1.py: planner, cohort, twin, evidence and
analyst agents sharing one study memory, with the uncertainty gate between the twin agent and
the rest, under three conditions (none, advisory, enforced), several runs each.

Reuses, unchanged:
  scripts/phase7_llm_predict_v5.py  evidence prompt text (SYSTEM, TWIN_NOTE, DESCRIBE), parse_score
  scripts/phase8_outcomes_v3.py     load_clinical, match_ids, analyse (the paper's Cox model)
so the pipeline's prompts and statistics are the paper's.

    # check the plumbing on the real files, no language model:
    python scripts/phase9_multiagent_v1.py --provider mock --runs 2 \
        --bi runs/phase6_bi_direct_v2_sharedworld_floorboth \
        --out runs/phase9_multiagent_v1_mock

    # real run, model served with vLLM on our GPU:
    python scripts/phase9_multiagent_v1.py --provider local --base-url http://localhost:8001/v1 \
        --model meta-llama/Llama-3.3-70B-Instruct --runs 10 \
        --bi runs/phase6_bi_direct_v2_sharedworld_floorboth \
        --out runs/phase9_multiagent_v1_llama33_3cond_10runs

Writes, per run: <out>/<tag>_<condition>_run<k>.json (metrics, conclusion, plan history) and
<tag>_<condition>_run<k>_trace.json (the full study memory and every model reply).
Appends one row per run to <out>/<tag>_summary.csv. Finished runs are skipped on rerun.
"""

import argparse
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from tme_twin.pipeline_v1 import Pipeline, SPEC_HASH       # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p7 = load_script("phase7_llm_predict_v5")
p8 = load_script("phase8_outcomes_v3")
COHORT = "GSE14814"


# --------------------------------------------------------------------------
# Tools: plain code the agents call
# --------------------------------------------------------------------------
def make_tools(root, bi_dir):
    obs = pd.read_csv(os.path.join(root, "data", "processed", COHORT, "features_v2_qc.csv"), index_col=0)
    clin_all = pd.read_csv(os.path.join(root, "data", "processed", COHORT, "clinical_harmonised.csv"), index_col=0)
    clin = p8.load_clinical(COHORT)
    z = np.load(os.path.join(bi_dir, f"bi_{COHORT}.npz"), allow_pickle=True)
    bi = {s: z["bi"][i] for i, s in enumerate(z["samples"])}

    def build_cohort(prop):
        with_rna = p8.match_ids(list(obs.index), clin_all.index)
        mp = p8.match_ids(list(obs.index), clin.index)
        ids = [s for s in obs.index if s in mp and s in bi]
        start = len(with_rna)
        return {"ids": ids, "flow": {"start": start, "excluded": start - len(ids), "final": len(ids)}}

    def twin_outputs(ids):
        keep = [s for s in ids if s in bi]
        return keep, np.vstack([bi[s] for s in keep])

    def features(ids):
        out = {}
        for s in ids:
            row = obs.loc[s]
            out[s] = [f"- {p7.DESCRIBE[f]}: {row[f]:.4g} (percentile {p7.pct(obs[f], row[f])})"
                      for f in p7.FEATURES]
        return out

    def evidence_prompt(lines, twin_line, gate_text):
        u = ["Tumour measurements (value; percentile among patients in the same dataset, 0-100):"] + lines
        if twin_line:
            u += ["", p7.TWIN_NOTE, "- " + twin_line]
        if gate_text:
            u += ["", gate_text]
        u += ["", "Return the JSON now."]
        return "\n".join(u)

    X = obs.to_numpy(float)
    X = (X - X.mean(0)) / X.std(0).clip(1e-12)
    pc = np.linalg.svd(X, full_matrices=False)[2][0]
    bmed = pd.Series({s: np.nanmedian(v) for s, v in bi.items()}).reindex(obs.index)
    ok = bmed.notna().to_numpy()
    if np.corrcoef((X @ pc)[ok], bmed[ok])[0, 1] < 0:          # same orientation as phase8_outcomes_v3
        pc = -pc
    pc1_all = pd.Series(X @ pc, index=obs.index)

    def fit(name, ids, values):
        return p8.analyse(COHORT, clin, name, list(ids), values)

    return {"build_cohort": build_cohort, "twin_outputs": twin_outputs, "features": features,
            "evidence_prompt": evidence_prompt, "evidence_system": p7.SYSTEM,
            "parse_score": p7.parse_score, "fit": fit,
            "pc1": lambda ids: pc1_all.reindex(ids).to_numpy(),
            "gate_rows": lambda: np.asarray(z["bi"], dtype=float)}


# --------------------------------------------------------------------------
# Language models
# --------------------------------------------------------------------------
def mock_llm(seed):
    rng = random.Random(seed)

    def llm(system, user):
        if "planning agent" in system:
            if "Return {\"plan\"" in user:
                return json.dumps({"plan": ["cohort", "twin", "evidence", "analyst"], "reason": "mock"})
            if "New event" in user:
                if "withheld" in user:
                    return json.dumps({"action": "replan", "plan": ["evidence", "analyst"], "reason": "mock"})
                return json.dumps({"action": "continue", "reason": "mock"})
            return json.dumps({"final": rng.choice(["accept", "accept", "abstain"]), "summary": "mock"})
        if "cohort agent" in system:
            h = "all" if rng.random() < 0.2 else "adenocarcinoma"
            return json.dumps({"histology": h, "require_treatment_known": True, "require_survival": True,
                               "reason": "mock"})
        if "analyst agent" in system:
            if "markers_to_test" in user:
                av = user.split("Markers available to test now: ")[1].split(".\n")[0].split(", ")
                return json.dumps({"markers_to_test": av, "primary_marker": av[0], "reason": "mock"})
            pm = "twin_B_i" if "twin_B_i:" in user else "evidence_score"
            return json.dumps({"primary_marker": pm, "conclusion": "mock conclusion.",
                               "marker_identifies_who_benefits": rng.random() < 0.5, "uses_twin": pm == "twin_B_i"})
        return json.dumps({"benefit_score": rng.randint(0, 100), "reason": "mock"})
    return llm


def real_llm(provider, model):
    def llm(system, user):
        for k in range(5):
            try:
                text, _ = p7.call(provider, model, system, user)
                return text
            except Exception as e:                                  # network or server hiccup
                if k == 4:
                    raise SystemExit(f"{provider} failed 5 times: {e}")
                time.sleep(2 * (k + 1))
    return llm


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["mock", "local", "anthropic"], required=True)
    ap.add_argument("--model", default="mock")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--conditions", nargs="*", default=["none", "advisory", "enforced"])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8, help="parallel evidence-agent calls")
    ap.add_argument("--bi", required=True, help="phase6 output folder with bi_GSE14814.npz")
    ap.add_argument("--root", default=ROOT, help="repo root holding data/processed (tests only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.provider == "local" and not (a.base_url and a.model != "mock"):
        raise SystemExit("--provider local needs --base-url and --model")
    if a.provider == "anthropic" and "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("export ANTHROPIC_API_KEY first")
    p7.MAX_TOKENS, p7.BASE_URL = a.max_tokens, a.base_url
    p8.ROOT = a.root
    os.makedirs(a.out, exist_ok=True)
    tools = make_tools(a.root, a.bi)
    tag = f"{a.provider}_{a.model.replace('/', '--')}_tok{a.max_tokens}"
    summ = os.path.join(a.out, f"{tag}_summary.csv")
    print(f"spec hash {SPEC_HASH}; model {a.model}; conditions {a.conditions}; {a.runs} runs each")

    for cond in a.conditions:
        for k in range(1, a.runs + 1):
            f = os.path.join(a.out, f"{tag}_{cond}_run{k:02d}.json")
            if os.path.exists(f):
                print(f"  {cond} run {k}: done, skipping"); continue
            t0 = time.time()
            llm = mock_llm(1000 * k + len(cond)) if a.provider == "mock" else real_llm(a.provider, a.model)
            pipe = Pipeline(llm, tools, cond, workers=a.workers)
            m = pipe.run()
            m.update({"model": a.model, "run": k, "seconds": round(time.time() - t0, 1)})
            json.dump(m, open(f, "w"), indent=1, default=str)
            json.dump({"memory": pipe.mem.entries, "calls": pipe.log}, open(f[:-5] + "_trace.json", "w"),
                      indent=1, default=str)
            row = {k2: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k2, v in m.items()}
            pd.DataFrame([row]).to_csv(summ, mode="a", header=not os.path.exists(summ), index=False)
            print(f"  {cond} run {k}: final {m['final']}; twin to evidence {m['evidence_saw_twin']}; "
                  f"analyst tested {m.get('analyst_requested')}; twin was primary {m['primary_is_twin']}; "
                  f"replans {m['replans']}; protocol vetoes {m['protocol_vetoes']}; {m['seconds']} s", flush=True)
    print(f"wrote {summ}")


if __name__ == "__main__":
    main()
