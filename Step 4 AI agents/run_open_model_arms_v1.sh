#!/usr/bin/env bash
# Run the four agent arms for one locally served open model, exactly as for MedGemma 27B
# (phase7_llm_predict_v5.py, prompt A, 1,000-token budget, 3 repeats, JBR.10), then compare arms.
# Added after the v0.13 freeze (Gemma 3 27B, Llama 3.3 70B): see docs/amendments_after_outcomes_v1.md.
# Usage:  bash scripts/run_open_model_arms_v1.sh <model id> <port> <npe dir> <bi dir> <short name>
#   e.g.  bash scripts/run_open_model_arms_v1.sh google/gemma-3-27b-it 8000 <npe dir> <bi dir> gemma-3-27b
# <npe dir> and <bi dir>: the same --npe and --bi folders used for the MedGemma run.
set -euo pipefail
MODEL=${1:?model}; PORT=${2:?port}; NPE=${3:?npe dir}; BI=${4:?bi dir}; SHORT=${5:?short name}
cd ~/tme_twin
source .venv/bin/activate
OUT=runs/phase7_llm_v5_GSE14814_3rep
# wait until the server answers
until curl -s http://127.0.0.1:$PORT/v1/models | grep -q "$MODEL"; do echo "waiting for $MODEL on port $PORT"; sleep 30; done
for ARM in features features+twin features+twin+guardrail features+twin+enforced; do
  python scripts/phase7_llm_predict_v5.py --provider local --base-url http://127.0.0.1:$PORT/v1 --model $MODEL \
      --arm $ARM --cohort GSE14814 --npe $NPE --bi $BI --max-tokens 1000 --repeats 3 --out $OUT
done
TAG=local_${MODEL//\//--}_default
python scripts/phase7_llm_arms_compare_v2.py --bi $BI --cohort GSE14814 \
    --arm features=$OUT/${TAG}_features_GSE14814_tok1000_v5.jsonl \
    --arm features+twin=$OUT/${TAG}_features_twin_GSE14814_tok1000_v5.jsonl \
    --arm features+twin+guardrail=$OUT/${TAG}_features_twin_guardrail_GSE14814_tok1000_v5.jsonl \
    --arm features+twin+enforced=$OUT/${TAG}_features_twin_enforced_GSE14814_tok1000_v5.jsonl \
    --out runs/phase7_llm_arms_compare_v2_${SHORT}_tok1000_4arms
echo "done: runs/phase7_llm_arms_compare_v2_${SHORT}_tok1000_4arms"
