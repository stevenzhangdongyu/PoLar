#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATHS_FILE=${1:-$SCRIPT_DIR/../data/path_batching/synthetic_paths_128.json}
if [[ $# -gt 0 ]]; then shift; fi
exec "${PYTHON:-python}" -u "$SCRIPT_DIR/debug_vllm_paths.py" "$PATHS_FILE" \
  --profile --yes --concurrency 8 --warmups 20 --input-len 512 --max-tokens 256 \
  --profile-iterations "${PROFILE_ITERATIONS:-0}" \
  --kv-cache-mode "${KV_CACHE_MODE:-execution}" \
  --strategy "${STRATEGY:-dynamic_layer}" --gpu "${GPU_ID:-0}" \
  --model "${MODEL_PATH:-/home/z00894149/Qwen2.5-3B-Instruct}" \
  --vllm-root "${VLLM_ROOT:-/home/z00894149/polar_dev/vllm}" "$@"
