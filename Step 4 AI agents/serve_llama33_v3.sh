#!/usr/bin/env bash
# Serve Llama 3.3 70B Instruct, fp8, split across TWO GPUs (tensor parallel). v2 unchanged.
# Why: no single shared GPU had the ~85 GB fp8 needs (Nov 13). Same model, same fp8 weights,
# same settings as v2; only the weights are spread over two cards (~45 GB each).
# Usage:  bash scripts/serve_llama33_v3.sh <GPU_A> <GPU_B> [port]
# Writes the settings and versions to runs/phase9_llama33_serving_fp8_tp2.json.
set -euo pipefail
export PATH=~/vllm_env/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
GA=${1:?first gpu}; GB=${2:?second gpu}; PORT=${3:-8001}
MODEL=meta-llama/Llama-3.3-70B-Instruct
NEED_GB=45
for G in $GA $GB; do
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G | tr -d ' ')
  if [ $((FREE_MB / 1024)) -lt $NEED_GB ]; then echo "GPU $G has $((FREE_MB/1024)) GB free; need $NEED_GB GB."; exit 1; fi
done
TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $GA | tr -d ' ')
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
echo "{\"model\": \"$MODEL\", \"quantization\": \"fp8\", \"tensor_parallel\": 2, \"vllm\": \"$V\", \"gpus\": \"$GA,$GB\", \"max_model_len\": 4096, \"gpu_memory_utilization\": $UTIL, \"generation_config\": $G, \"temperature_sent\": false, \"enforce_eager\": true, \"flashinfer_sampler\": false, \"attention_backend\": \"FLASH_ATTN\", \"time\": \"$(date -Is)\"}" > runs/phase9_llama33_serving_fp8_tp2.json
cat runs/phase9_llama33_serving_fp8_tp2.json
CUDA_VISIBLE_DEVICES=$GA,$GB exec ~/vllm_env/bin/vllm serve $MODEL --quantization fp8 --tensor-parallel-size 2 \
    --max-model-len 4096 --enforce-eager --gpu-memory-utilization $UTIL --port $PORT --host 127.0.0.1
