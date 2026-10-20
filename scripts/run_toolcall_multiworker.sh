#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/run_toolcall_multiworker.sh model predictor input_json output_dir
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARGS=(
  --task_type tool
  --model_path "${1:?model path required}"
  --checkpoint_path "${2:?predictor checkpoint required}"
  --input_json "${3:?tool JSON/JSONL required}"
  --output_dir "${4:?output dir required}"
  --gpus "${GPUS:-0 1 2 3 4 5 6 7}"
  --workers_per_gpu "${WORKERS_PER_GPU:-1}"
  --max_total_workers "${MAX_TOTAL_WORKERS:-0}"
  --limit "${LIMIT:-0}"
  --max_new_tokens "${MAX_NEW_TOKENS:-512}"
  --beam_size "${BEAM_SIZE:-10}"
  --top_k_ops "${TOP_K_OPS:-2}"
  --top_k_paths "${TOP_K_PATHS:-5}"
  --seg_threshold "${SEG_THRESHOLD:-0.5}"
  --max_pack "${MAX_PACK:-4}"
  --embedding_model_name "${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
  --polar_d_model "${POLAR_D_MODEL:-256}"
  --polar_heads "${POLAR_HEADS:-4}"
  --polar_layers "${POLAR_LAYERS:-2}"
  --seed "${SEED:-42}"
)
if [[ "${RESUME:-0}" == "1" ]]; then ARGS+=(--resume); fi
if [[ "${MERGE_ONLY:-0}" == "1" ]]; then ARGS+=(--merge_only); fi
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python3 -u "$ROOT/scripts/run_polar_task_multiworker.py" "${ARGS[@]}"
