#!/usr/bin/env bash
# Serve Gemma 3 27B (open-weight, general purpose; the base model MedGemma 27B was tuned from) on one GPU.
# Added after the v0.13 freeze: label as an added robustness run (see amendments file).
# Usage:  bash scripts/serve_gemma3_v2.sh <GPU> <fp8|bnb4> [port]
#   fp8   ~28 GB of weights, needs ~40 GB free on the GPU
#   bnb4  ~16 GB of weights, needs ~25 GB free on the GPU
# Runs in the separate environment ~/vllm_env (keeps the project .venv untouched).
# Writes the exact settings and versions to runs/phase7_gemma3_serving_<quant>.json.
set -euo pipefail
# v2: the same fixes MedGemma needed on this server: ninja on PATH (pip-installed in ~/vllm_env),
# no FlashInfer sampler (no nvcc here), FlashAttention backend, eager mode.
export PATH=~/vllm_env/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
GPU=${1:?gpu index}; Q=${2:?fp8 or bnb4}; PORT=${3:-8000}
MODEL=google/gemma-3-27b-it
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
echo "{\"model\": \"$MODEL\", \"quantization\": \"$Q\", \"vllm\": \"$V\", \"gpu\": $GPU, \"max_model_len\": 4096, \"gpu_memory_utilization\": $UTIL, \"generation_config\": $G, \"temperature_sent\": false, \"enforce_eager\": true, \"flashinfer_sampler\": false, \"attention_backend\": \"FLASH_ATTN\", \"time\": \"$(date -Is)\"}" > runs/phase7_gemma3_serving_$Q.json
cat runs/phase7_gemma3_serving_$Q.json
CUDA_VISIBLE_DEVICES=$GPU exec ~/vllm_env/bin/vllm serve $MODEL $QARG --max-model-len 4096 --enforce-eager \
    --gpu-memory-utilization $UTIL --port $PORT --host 127.0.0.1
