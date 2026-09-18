#!/usr/bin/env bash
set -euo pipefail
SEARCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$SEARCH_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="$SEARCH_ROOT/.cache"
export VLLM_CACHE_ROOT="$SEARCH_ROOT/.cache/vllm"
export VLLM_CONFIG_ROOT="$SEARCH_ROOT/.cache/vllm-config"
export TRITON_CACHE_DIR="$SEARCH_ROOT/.cache/triton"
export CUDA_CACHE_PATH="$SEARCH_ROOT/.cache/cuda"
export FLASHINFER_WORKSPACE_DIR="$SEARCH_ROOT/.cache/flashinfer"
mkdir -p "$SEARCH_ROOT/.cache" "$CUDA_CACHE_PATH"
if [[ -f /usr/lib/wsl/lib/libcuda.so.1.1 ]]; then
  SEARCH_DRIVER=""
  for entry in /usr/lib/wsl/drivers/*; do
    if [[ -f "$entry/libcuda_loader.so" && -f "$entry/libnvidia-ptxjitcompiler.so.1" ]] && cmp -s /usr/lib/wsl/lib/libcuda.so.1.1 "$entry/libcuda_loader.so"; then SEARCH_DRIVER="$entry"; break; fi
  done
  [[ -n "$SEARCH_DRIVER" ]] || { echo 'Matching WSL CUDA libraries not found' >&2; exit 1; }
  export LD_LIBRARY_PATH="$SEARCH_DRIVER:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
