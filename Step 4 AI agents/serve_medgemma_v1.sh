#!/usr/bin/env bash
# Serve MedGemma 27B (open-weight, medical) on one GPU for Exp 7c (pre-registration v0.13).
# Usage:  bash scripts/serve_medgemma_v1.sh <GPU> <fp8|bnb4> [port]
#   fp8   ~28 GB of weights, needs ~40 GB free on the GPU
#   bnb4  ~16 GB of weights, needs ~25 GB free on the GPU
# Runs in the separate environment ~/vllm_env (keeps the project .venv untouched).
# Writes the exact settings and versions to runs/phase7_medgemma_serving_<quant>.json.
set -euo pipefail
GPU=${1:?gpu index}; Q=${2:?fp8 or bnb4}; PORT=${3:-8000}
MODEL=google/medgemma-27b-text-it
case $Q in
  fp8)  QARG="--quantization fp8"; UTIL=0.28 ;;
  bnb4) QARG="--quantization bitsandbytes"; UTIL=0.18 ;;
  *) echo "quant must be fp8 or bnb4"; exit 1 ;;
esac
cd ~/tme_twin
V=$(~/vllm_env/bin/python -c "import vllm; print(vllm.__version__)")
G=$(~/vllm_env/bin/python - <<PY
import json
from huggingface_hub import hf_hub_download
try:
    print(json.dumps(json.load(open(hf_hub_download("$MODEL", "generation_config.json")))))
except Exception as e:
    print(json.dumps({"error": str(e)}))
PY
)
echo "{\"model\": \"$MODEL\", \"quantization\": \"$Q\", \"vllm\": \"$V\", \"gpu\": $GPU, \"max_model_len\": 4096, \"gpu_memory_utilization\": $UTIL, \"generation_config\": $G, \"temperature_sent\": false, \"time\": \"$(date -Is)\"}" > runs/phase7_medgemma_serving_$Q.json
cat runs/phase7_medgemma_serving_$Q.json
CUDA_VISIBLE_DEVICES=$GPU exec ~/vllm_env/bin/vllm serve $MODEL $QARG --max-model-len 4096 \
    --gpu-memory-utilization $UTIL --port $PORT --host 127.0.0.1
