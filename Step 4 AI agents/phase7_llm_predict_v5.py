"""
Version 5 of phase7_llm_predict (v1-v4 unchanged). LC decision, Nov 10 (pre-registration
v0.13): adds --provider local, an open-weight model served on our own GPU through an
OpenAI-compatible endpoint (vLLM). Used for MedGemma 27B (open, medical).
  --base-url   e.g. http://localhost:8000/v1   (no API key; nothing leaves the server)
  --model      the served model name, e.g. google/medgemma-27b-text-it
Prompts, arms, caching and outputs as v4. No temperature is sent (model defaults, as for
Claude); the server's generation settings are recorded by the serving script.

    python scripts/phase7_llm_predict_v5.py --provider local --base-url http://localhost:8000/v1 \
        --model google/medgemma-27b-text-it --arm features --cohort GSE14814 --max-tokens 1000 \
        --npe runs/phase5_v3_fib120_npe_logithyp_endolinked \
        --bi runs/phase6_bi_direct_v2_sharedworld_floorboth --out runs/phase7_llm_v5_GSE14814_3rep

--- v4 docstring ---
Version 4 of phase7_llm_predict (v1-v3 unchanged). LC decisions, Nov 10:
  - Fourth arm, features+twin+enforced: the twin's check fails, so its outputs (B_i and
    parameters) are WITHHELD and the agent is told why. Tests an enforced guardrail
    against the advisory one (features+twin+guardrail).
  - --max-tokens (default 300, as v3): the reply budget, which the model also uses for
    its internal reasoning. Recorded in the file name and every row.
  - Records stop_reason and thinking tokens per call.
  - No temperature is sent: the model rejects it (deprecated); default sampling.
Prompts otherwise identical to v3.

    python scripts/phase7_llm_predict_v4.py --provider anthropic --model claude-opus-5-5 \
        --arm features+twin+enforced --cohort GSE14814 --max-tokens 1000 \
        --npe runs/phase5_v3_fib120_npe_logithyp_endolinked \
        --bi runs/phase6_bi_direct_v2_sharedworld_floorboth --max-dollars 40 \
        --out runs/phase7_llm_v4_GSE14814_3rep

--- v3 docstring ---
Version 3 of phase7_llm_predict (v1, v2 unchanged). LC decisions, Nov 8:
  - Claude (claude-opus-5-5) replaces OpenAI as the agent (pre-registration v0.8).
  - Third arm, features+twin+guardrail: the twin arm plus one sentence carrying the
    twin's own pre-registered identifiability check (between-patient SD of B_i vs
    mean within-patient SD), computed from the --bi file, not typed in.
  - Twin parameters shown in readable units (per day, per hour) instead of per second.
Everything else as v2: prompts, firewall, caching, cap (per arm), pilot.

    export ANTHROPIC_API_KEY=...
    python scripts/phase7_llm_predict_v3.py --provider anthropic --model claude-opus-5-5 \
        --arm features+twin+guardrail --cohort GSE14814 \
        --npe runs/phase5_v3_fib120_npe_logithyp_endolinked \
        --bi runs/phase6_bi_direct_v2_sharedworld_floorboth --max-dollars 50 --pilot \
        --out runs/phase7_llm_v3_claude-opus-5-5_GSE14814_3rep

--- v2 docstring ---
Version 2 of phase7_llm_predict (v1 unchanged). Adds, for a fixed lab budget:
  --reasoning-effort low|medium|high   sent to the model and recorded per row
  --max-dollars X    hard spending cap for this cache file. Every reply's real
                     token counts (input, output incl. hidden reasoning) are
                     priced from PRICES; the run stops before a call could
                     cross X. The cache keeps what was done; rerun to continue.
  --pilot            run 3 patients x 1 repeat, print the real cost per call
                     and the projected cost of the full run, then stop.
Spend so far is summed from the cache file, so it carries across reruns.

Budget plan (Nov 8, lab balance $100): gpt-6-astra, medium, JBR.10 (GSE14814)
x 2 arms x 3 repeats; gpt-6.1-sol, low, other cohorts x 2 arms x 2 repeats.

    export OPENAI_API_KEY=...
    python scripts/phase7_llm_predict_v2.py --provider openai --model gpt-6-astra --reasoning-effort medium \
        --arm features --cohort GSE14814 --pilot --out runs/phase7_llm_v2

--- v1 docstring ---
Phase 7 comparator: large language models asked to predict, per patient, how
much adjuvant cisplatin-based chemotherapy will help, with and without the
twin's outputs. LC decision, Nov 6: both OpenAI and Claude, lab credits.

    export OPENAI_API_KEY=...   ANTHROPIC_API_KEY=...
    python scripts/phase7_llm_predict_v1.py --provider openai --model gpt-6-astra --arm features \
        --cohort GSE14814 --out runs/phase7_llm_v1
    python scripts/phase7_llm_predict_v1.py --provider anthropic --model claude-opus-5-5 --arm features+twin \
        --cohort GSE14814 --npe runs/phase5_v3_fib120_npe_logithyp_endolinked \
        --bi runs/phase6_bi_emulator_v1 --out runs/phase7_llm_v1
    python scripts/phase7_llm_predict_v1.py --provider mock --arm features+twin ... --max-patients 3   # test, no API

Two arms
--------
features         the patient's 12 tumour features (the same numbers the twin
                 is fitted to), each with its percentile in the patient's cohort.
features+twin    the same, plus the twin's output for that patient: B_i (median
                 and 90% interval over posterior draws, and its cohort
                 percentile) and the fitted parameters (median and 90% interval).
                 This is the "agent fed by the twin" arm.

Outcome firewall
----------------
The prompt never contains treatment, survival, stage, age, sex, cohort name,
sample identifier or study name, so the model cannot look a patient up or
read the answer. Patients are sent in a shuffled order. The model's answer is
a score, used afterwards in the same Cox interaction test as B_i
(phase8_outcomes_v1.py). Every prompt and reply is stored.

Settings: temperature 0 where the model accepts it (retried without if not),
--repeats calls per patient (default 3), averaged. The model id and the date
of the run are written into every row; fill them into the pre-registration.
Resumable: rows already in the cache file are skipped.

Writes <out>/<provider>_<model>_<arm>_<cohort>.jsonl (every call) and
<out>/<provider>_<model>_<arm>_<cohort>.csv (mean score per patient).
"""

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tme_twin.observe_v2 import FEATURES                              # noqa: E402

# $ per million tokens (input, output), OpenAI and Anthropic pricing pages, checked Nov 8 2026
PRICES = {"gpt-6-astra": (10.0, 50.0), "gpt-6.1-sol": (2.0, 10.0), "gpt-6.1-terra": (1.0, 5.0),
          "gpt-6-luna": (0.10, 0.50), "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
          "claude-fable-5-1": (10.0, 50.0), "mock": (0.0, 0.0)}


def call_cost(model, usage):
    """Dollars for one call from the usage block the API returned."""
    pin, pout = PRICES.get(model, (None, None))
    if pin is None or not usage:
        return 0.0
    tin = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    tout = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    return (tin * pin + tout * pout) / 1e6


DESCRIBE = {
    "f_tumour": "tumour cell fraction (EPIC deconvolution, share of all cells incl. endothelium)",
    "f_macrophage": "macrophage fraction",
    "f_fibroblast": "fibroblast / cancer-associated fibroblast fraction",
    "f_cd8": "CD8 T cell fraction",
    "f_cd4": "CD4 T cell fraction",
    "f_b": "B cell fraction",
    "f_endothelial": "endothelial (blood vessel) fraction",
    "hypoxia": "hypoxia gene-set score (ssGSEA, Hallmark hypoxia)",
    "tgfb": "TGF-beta signalling score (ssGSEA, Hallmark)",
    "il6": "IL-6/JAK/STAT3 signalling score (ssGSEA, Hallmark)",
    "m2_share": "M2 macrophage polarisation score (ssGSEA, marker set)",
    "caf_share": "CAF activation score (ssGSEA, marker set)",
}
PARAM_DESCRIBE = {   # name: (description, factor from the model's native units, unit shown)
    "vessel_density": ("vessel density", 1.0, "vessels/mm^2"),
    "f0_tumour": ("starting tumour fraction of the simulated patch", 1.0, ""),
    "r_mac": ("macrophage recruitment per vessel", 86400.0, "per day"),
    "r_cd8": ("CD8 T cell recruitment per vessel", 86400.0, "per day"),
    "r_cd4": ("CD4 T cell recruitment per vessel", 86400.0, "per day"),
    "r_b": ("B cell recruitment per vessel", 86400.0, "per day"),
    "K_fib": ("fibroblast density set point", 1.0, ""),
    "k_m2": ("macrophage M2 polarisation rate", 86400.0, "per day per unit tumour fraction"),
    "k_caf": ("fibroblast-to-CAF activation rate", 86400.0, "per nM per day"),
    "r_div": ("tumour cell division rate", 86400.0, "per day"),
    "r_il6": ("IL-6 secretion per M2 macrophage", 3600.0, "nM per hour"),
    "Vmax_o2": ("oxygen consumption per cell", 1.0, "mmHg/s"),
}

SYSTEM = ("You are assisting a translational oncology research study. You will be given "
          "quantitative tumour-microenvironment measurements from the resected primary tumour "
          "of one patient with early-stage non-small-cell lung cancer. Estimate how much this "
          "patient is expected to benefit from adjuvant cisplatin-based chemotherapy after "
          "surgery, compared with observation alone. Use only the information given. "
          "Reply with JSON only: {\"benefit_score\": <integer 0-100>, \"reason\": \"<one sentence>\"}. "
          "0 means no expected benefit (or harm); 100 means the largest benefit you would expect "
          "for any patient in this setting. Score relative benefit from chemotherapy, not the "
          "patient's overall prognosis.")

TWIN_NOTE = ("A mechanistic digital twin of this tumour (an agent-based model of tumour, "
             "macrophage, fibroblast/CAF, CD8, CD4 and B cells with oxygen, cisplatin, IL-6 and "
             "TGF-beta fields, fitted to this patient's measurements) was given one cycle of "
             "cisplatin in silico. B_i = log10(tumour cells without chemo / tumour cells with "
             "chemo) at day 9; higher B_i means the drug killed more of this tumour in the twin.")


def guardrail_note(bi_map):
    """The twin's own identifiability check, computed from the B_i file."""
    m = np.array([np.nanmedian(v) for v in bi_map.values()])
    w = np.array([np.nanstd(v) for v in bi_map.values()])
    between, within = float(np.nanstd(m)), float(np.nanmean(w))
    verdict = ("cannot reliably rank patients by benefit from these data"
               if between < within else "can rank patients by benefit from these data")
    return (f"Reliability of the twin (its pre-registered check, computed before any outcome was seen): "
            f"across patients in this dataset, the spread of B_i between patients (SD {between:.3f}) is "
            f"{'smaller' if between < within else 'larger'} than the twin's uncertainty within a single "
            f"patient (mean SD {within:.3f}). By this check the twin {verdict}."), between, within


def enforced_note(bi_map):
    m = np.array([np.nanmedian(v) for v in bi_map.values()])
    w = np.array([np.nanstd(v) for v in bi_map.values()])
    between, within = float(np.nanstd(m)), float(np.nanmean(w))
    return ("A mechanistic digital twin of this tumour (an agent-based model of tumour, macrophage, "
            "fibroblast/CAF, CD8, CD4 and B cells with oxygen, cisplatin, IL-6 and TGF-beta fields) was "
            "fitted to these measurements and given cisplatin in silico. Its outputs are withheld: by its "
            "pre-registered check, computed before any outcome was seen, the spread of its predicted "
            f"response between patients (SD {between:.3f}) is smaller than its uncertainty within a single "
            f"patient (mean SD {within:.3f}), so it cannot rank patients by benefit from these data.")


def pct(series, x):
    return int(round(100.0 * (series < x).mean()))


def build_prompt(row, cohort, arm, post=None, bi=None, bi_all=None, guard=None, enforced=None):
    lines = ["Tumour measurements (value; percentile among patients in the same dataset, 0-100):"]
    for f in FEATURES:
        lines.append(f"- {DESCRIBE[f]}: {row[f]:.4g} (percentile {pct(cohort[f], row[f])})")
    if arm in ("features+twin", "features+twin+guardrail"):
        lines += ["", TWIN_NOTE,
                  f"- B_i: median {np.nanmedian(bi):.3f}, 90% interval {np.nanquantile(bi, .05):.3f} to "
                  f"{np.nanquantile(bi, .95):.3f} (percentile {pct(bi_all, np.nanmedian(bi))} among patients "
                  f"in the same dataset)",
                  "Fitted twin parameters (posterior median; 90% interval):"]
        for p, (desc, k, unit) in PARAM_DESCRIBE.items():
            if f"{p}_median" in post:
                u = f" ({unit})" if unit else ""
                lines.append(f"- {desc}{u}: {k * post[f'{p}_median']:.3g} "
                             f"({k * post[f'{p}_q05']:.3g} to {k * post[f'{p}_q95']:.3g})")
        if arm == "features+twin+guardrail":
            lines += ["", guard]
    if arm == "features+twin+enforced":
        lines += ["", enforced]
    lines += ["", "Return the JSON now."]
    return "\n".join(lines)


def _post(url, headers, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


MAX_TOKENS = 300
BASE_URL = None


def call(provider, model, system, user, temperature=True, effort=None):
    if provider == "mock":
        h = abs(hash(user)) % 101
        return json.dumps({"benefit_score": h, "reason": "mock"}), {"mock": True, "usage": {"prompt_tokens": 600, "completion_tokens": 300}}
    if provider == "openai":
        body = {"model": model, "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}}
        if temperature:
            body["temperature"] = 0
        if effort:
            body["reasoning_effort"] = effort
        r = _post("https://api.openai.com/v1/chat/completions",
                  {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                   "Content-Type": "application/json"}, body)
        return r["choices"][0]["message"]["content"], {"id": r.get("id"), "model": r.get("model"),
                                                      "usage": r.get("usage")}
    if provider == "local":
        body = {"model": model, "max_tokens": MAX_TOKENS,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        r = _post(BASE_URL.rstrip("/") + "/chat/completions", {"Content-Type": "application/json"}, body, timeout=600)
        ch = r["choices"][0]
        return ch["message"]["content"] or "", {"id": r.get("id"), "model": r.get("model"), "usage": r.get("usage"),
                                                 "stop_reason": ch.get("finish_reason")}
    if provider == "anthropic":
        body = {"model": model, "max_tokens": MAX_TOKENS, "system": system,
                "messages": [{"role": "user", "content": user}]}
        r = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}, body)
        text = "".join(b.get("text", "") for b in r.get("content", []))
        return text, {"id": r.get("id"), "model": r.get("model"), "usage": r.get("usage"),
                      "stop_reason": r.get("stop_reason")}
    raise ValueError(provider)


def parse_score(text):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            s = json.loads(m.group(0)).get("benefit_score")
            if s is not None and 0 <= float(s) <= 100:
                return float(s)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    m = re.search(r"benefit_score\"?\s*[:=]\s*(\d+(\.\d+)?)", text)
    return float(m.group(1)) if m and 0 <= float(m.group(1)) <= 100 else None


def robust_call(provider, model, system, user, retries=5, effort=None):
    temp = True
    for k in range(retries):
        try:
            return call(provider, model, system, user, temperature=temp, effort=effort)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code == 400 and "temperature" in msg and temp:
                temp = False                      # model does not accept temperature; retry without
                continue
            if e.code == 400 and "reasoning" in msg and effort:
                raise SystemExit(f"{model} rejected reasoning_effort={effort}: {msg[:300]}")
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(min(60, 2 ** k * 3)); continue
            raise SystemExit(f"{provider} HTTP {e.code}: {msg[:500]}")
        except (urllib.error.URLError, TimeoutError) as e:
            time.sleep(min(60, 2 ** k * 3))
            if k == retries - 1:
                raise SystemExit(f"{provider}: cannot reach the API ({e})")
    raise SystemExit(f"{provider}: gave up after {retries} tries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic", "local", "mock"], required=True)
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint for --provider local")
    ap.add_argument("--model", default=None, help="exact model id; recorded in every row")
    ap.add_argument("--arm", choices=["features", "features+twin", "features+twin+guardrail", "features+twin+enforced"], required=True)
    ap.add_argument("--max-tokens", type=int, default=300, help="reply budget incl. internal reasoning")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--npe", default=None, help="phase5_train_npe_v4 output (features+twin arm)")
    ap.add_argument("--bi", default=None, help="phase6_bi_emulator_v1 output (features+twin arm)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], default=None)
    ap.add_argument("--max-dollars", type=float, default=None, help="hard cap for this cache file")
    ap.add_argument("--pilot", action="store_true", help="3 patients x 1 repeat, then print projected cost")
    ap.add_argument("--max-patients", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model = a.model or {"openai": "gpt-6-astra", "anthropic": "claude-opus-5-5", "mock": "mock"}[a.provider]
    os.makedirs(a.out, exist_ok=True)
    if a.provider == "openai" and "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("set OPENAI_API_KEY first")
    if a.provider == "anthropic" and "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("set ANTHROPIC_API_KEY first")

    obs = pd.read_csv(os.path.join(ROOT, "data", "processed", a.cohort, "features_v2_qc.csv"), index_col=0)
    global MAX_TOKENS, BASE_URL
    MAX_TOKENS = a.max_tokens
    BASE_URL = a.base_url
    if a.provider == "local" and not (a.base_url and a.model):
        raise SystemExit("--provider local needs --base-url and --model")
    post = bi_map = bi_all = guard = enf = None
    if a.arm != "features":
        if not (a.npe and a.bi):
            raise SystemExit("features+twin needs --npe and --bi")
        post = pd.read_csv(os.path.join(a.npe, f"posterior_{a.cohort}.csv"), index_col=0)
        z = np.load(os.path.join(a.bi, f"bi_{a.cohort}.npz"), allow_pickle=True)
        bi_map = {s: z["bi"][i] for i, s in enumerate(z["samples"])}
        bi_all = pd.Series({s: np.nanmedian(v) for s, v in bi_map.items()})
        if a.arm == "features+twin+guardrail":
            guard, gb, gw = guardrail_note(bi_map)
            print(f"guardrail sentence (between {gb:.3f}, within {gw:.3f}):\n  {guard}")
        if a.arm == "features+twin+enforced":
            enf = enforced_note(bi_map)
            print(f"enforced note:\n  {enf}")

    eff = a.reasoning_effort
    tag = f"{a.provider}_{model.replace('/', '--')}_{eff or 'default'}_{a.arm.replace('+', '_')}_{a.cohort}_tok{a.max_tokens}_v5" + ("_pilot" if a.pilot else "")
    cache = os.path.join(a.out, tag + ".jsonl")
    done, spent, past = set(), 0.0, []
    if os.path.exists(cache):
        for line in open(cache):
            r = json.loads(line)
            done.add((r["sample"], r["repeat"]))
            spent += r.get("cost_usd", 0.0)
            past.append(r.get("cost_usd", 0.0))
    n_all = len(obs)
    ids = list(obs.index)
    random.Random(a.seed).shuffle(ids)
    if a.max_patients:
        ids = ids[:a.max_patients]
    if a.pilot:
        ids, a.repeats = ids[:3], 1
    today = dt.date.today().isoformat()
    todo = [(s, k) for s in ids for k in range(a.repeats) if (s, k) not in done]
    print(f"{tag}: {len(ids)} patients x {a.repeats} repeats; {len(todo)} calls to make; "
          f"spent so far ${spent:.2f}" + (f", cap ${a.max_dollars:.2f}" if a.max_dollars else ""), flush=True)
    per_call = list(past[-50:])          # a resumed run checks the cap from its first call

    with open(cache, "a") as fh:
        for n, (s, k) in enumerate(todo):
            if a.arm != "features" and (s not in bi_map or s not in post.index):
                continue
            user = build_prompt(obs.loc[s], obs, a.arm,
                                post.loc[s] if post is not None else None,
                                bi_map[s] if bi_map else None, bi_all, guard, enf)
            if a.max_dollars is not None and per_call:
                worst = max(per_call) * 1.5
                if spent + worst > a.max_dollars:
                    print(f"  stopping: ${spent:.2f} spent, next call could cross the ${a.max_dollars:.2f} cap", flush=True)
                    break
            text, info = robust_call(a.provider, model, SYSTEM, user, effort=None if eff in (None, "none") else eff)
            c = call_cost(model, info.get("usage"))
            per_call.append(c); spent += c
            fh.write(json.dumps({"sample": s, "repeat": k, "provider": a.provider, "model_requested": model,
                                 "model_reported": info.get("model"), "date": today, "arm": a.arm,
                                 "reasoning_effort": eff, "cost_usd": c, "max_tokens": a.max_tokens,
                                 "stop_reason": info.get("stop_reason"),
                                 "thinking_tokens": ((info.get("usage") or {}).get("output_tokens_details") or {}).get("thinking_tokens"),
                                 "cohort": a.cohort, "score": parse_score(text), "reply": text,
                                 "prompt": user, "system": SYSTEM, "info": info}) + "\n")
            fh.flush()
            if (n + 1) % 25 == 0:
                print(f"  {n + 1}/{len(todo)}  spent ${spent:.2f}", flush=True)

    if a.pilot and per_call:
        per_call = per_call[len(past[-50:]):] or per_call
        avg = sum(per_call) / len(per_call)
        print(f"\npilot: {len(per_call)} calls, ${sum(per_call):.4f} total, ${avg:.4f} per call")
        for reps in (2, 3):
            print(f"  projected full run, {n_all} patients x {reps} repeats (this arm): ${avg * n_all * reps:.2f}")
        print("  run the pilot for each arm: the twin arms have longer prompts")
        return

    rows = [json.loads(l) for l in open(cache)]
    df = pd.DataFrame(rows)
    bad = int(df["score"].isna().sum())
    summ = df.groupby("sample")["score"].agg(["mean", "std", "count"]).rename(
        columns={"mean": "llm_score", "std": "llm_score_sd", "count": "n_valid"})
    summ["model"] = model
    summ["date"] = today
    summ.to_csv(os.path.join(a.out, tag + ".csv"))
    print(f"wrote {a.out}/{tag}.csv ({len(summ)} patients; {bad} replies without a valid score; "
          f"mean SD across repeats {summ['llm_score_sd'].mean():.2f})")


if __name__ == "__main__":
    main()
