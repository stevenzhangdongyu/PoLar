#!/usr/bin/env bash
# Run inside the server's zdy_vllm_polar environment. No install or build is done.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# All settings can be overridden with environment variables.
exec "${PYTHON:-python}" "$SCRIPT_DIR/vllm_aligned_benchmark.py" \
  --vllm-root "${VLLM_ROOT:-/home/z00894149/polar_dev/vllm}" \
  --model "${MODEL_PATH:-/home/z00894149/Qwen2.5-3B-Instruct}" \
  --paths "${PATHS_JSON:-$SCRIPT_DIR/../data/path_batching/synthetic_paths_128.json}" \
  --output "${OUTPUT_DIR:-$SCRIPT_DIR/../outputs/vllm_aligned_$(date +%Y%m%d_%H%M%S)}" \
  --gpu "${GPU_ID:-0}" --port "${PORT:-8000}" \
  --input-len "${INPUT_LEN:-512}" --output-len "${OUTPUT_LEN:-256}" \
  --requests "${NUM_REQUESTS:-512}" --warmups "${NUM_WARMUPS:-20}" \
  --concurrency "${CONCURRENCIES:-1 2 4 8 16 32 64 96}" \
  --repeats "${REPEATS:-1}" --seed "${SEED:-42}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --max-num-seqs "${MAX_NUM_SEQS:-256}" "$@"
