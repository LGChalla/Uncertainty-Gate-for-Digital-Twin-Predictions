#!/usr/bin/env bash
# Serve Llama 3.3 70B Instruct (open-weight, general purpose, Meta) on one GPU.
# Added after the v0.13 freeze: label as an added robustness run (see amendments file).
# Usage:  bash scripts/serve_llama33_v2.sh <GPU> <fp8|bnb4> [port]
#   fp8   ~70 GB of weights, needs ~85 GB free on the GPU
#   bnb4  ~40 GB of weights, needs ~50 GB free on the GPU
# The memory share is worked out from the GPU's total memory, so it is the same
# amount in GB whatever card the job lands on.
# Runs in the separate environment ~/vllm_env (keeps the project .venv untouched).
# Writes the exact settings and versions to runs/phase7_llama33_serving_<quant>.json.
set -euo pipefail
# v2: the same fixes MedGemma needed on this server: ninja on PATH (pip-installed in ~/vllm_env),
# no FlashInfer sampler (no nvcc here), FlashAttention backend, eager mode.
export PATH=~/vllm_env/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
GPU=${1:?gpu index}; Q=${2:?fp8 or bnb4}; PORT=${3:-8001}
MODEL=meta-llama/Llama-3.3-70B-Instruct
case $Q in
  fp8)  QARG="--quantization fp8"; NEED_GB=85 ;;
  bnb4) QARG="--quantization bitsandbytes"; NEED_GB=50 ;;
  *) echo "quant must be fp8 or bnb4"; exit 1 ;;
esac
TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GPU | tr -d ' ')
FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | tr -d ' ')
if [ $((FREE_MB / 1024)) -lt $NEED_GB ]; then echo "GPU $GPU has $((FREE_MB/1024)) GB free; need $NEED_GB GB. Pick another GPU or use bnb4."; exit 1; fi
UTIL=$(python3 -c "print(round($NEED_GB*1024/$TOTAL_MB, 3))")
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
echo "{\"model\": \"$MODEL\", \"quantization\": \"$Q\", \"vllm\": \"$V\", \"gpu\": $GPU, \"max_model_len\": 4096, \"gpu_memory_utilization\": $UTIL, \"generation_config\": $G, \"temperature_sent\": false, \"enforce_eager\": true, \"flashinfer_sampler\": false, \"attention_backend\": \"FLASH_ATTN\", \"time\": \"$(date -Is)\"}" > runs/phase7_llama33_serving_$Q.json
cat runs/phase7_llama33_serving_$Q.json
CUDA_VISIBLE_DEVICES=$GPU exec ~/vllm_env/bin/vllm serve $MODEL $QARG --max-model-len 4096 --enforce-eager \
    --gpu-memory-utilization $UTIL --port $PORT --host 127.0.0.1
