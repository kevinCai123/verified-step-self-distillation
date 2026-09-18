#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime_env.sh"
cd "$SEARCH_ROOT"
MODEL_PATH="$(.venv-data/bin/python -c 'import json; print(json.load(open("config/local.json"))["model_path"])')"
export HF_HUB_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0
LORA_ARGS=(--enable-lora --max-lora-rank 16 --max-loras 1)
if [[ -n "${1:-}" ]]; then
  LORA_ARGS+=(--lora-modules "${2:-current}=$1")
fi
printf '%s\n' "$$" > runs/model-server.pid
exec .venv-rollout/bin/vllm serve "$MODEL_PATH" --served-model-name Qwen/Qwen3.5-9B --host 127.0.0.1 --port 8093 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.80 --max-num-batched-tokens 4096 --safetensors-load-strategy prefetch --enforce-eager --attention-backend TRITON_ATTN --language-model-only "${LORA_ARGS[@]}"
