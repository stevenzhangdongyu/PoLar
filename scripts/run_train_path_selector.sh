#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/run_train_path_selector.sh model_path save_dir merged_results.json [more.json ...]
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?model path required}"
SAVE_DIR="${2:?save dir required}"
shift 2
if [[ $# -lt 1 ]]; then
  echo "At least one merged_results.json is required." >&2
  exit 1
fi
ARGS=(--model_path "$MODEL" --save_dir "$SAVE_DIR")
for f in "$@"; do ARGS+=(--eval_json "$f"); done
ARGS+=(
  --text_encoder_name "${TEXT_ENCODER_NAME:-Qwen/Qwen3-Embedding-0.6B}"
  --num_epochs "${NUM_EPOCHS:-5}"
  --batch_size "${BATCH_SIZE:-32}"
  --learning_rate "${LEARNING_RATE:-1e-4}"
  --hidden_dim "${HIDDEN_DIM:-256}"
  --dropout "${DROPOUT:-0.1}"
  --max_polar_paths "${MAX_POLAR_PATHS:-5}"
  --val_ratio "${VAL_RATIO:-0.1}"
  --test_ratio "${TEST_RATIO:-0.1}"
  --seed "${SEED:-42}"
  --max_examples "${MAX_EXAMPLES:-0}"
  --train_mode "${TRAIN_MODE:-standard}"
)
if [[ "${SKIP_ALL_WRONG:-0}" == "1" ]]; then ARGS+=(--skip_all_wrong); fi
if [[ "${HF_LOCAL_FILES_ONLY:-0}" == "1" ]]; then ARGS+=(--local_files_only); fi
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python3 -u "$ROOT/scripts/train_path_selector.py" "${ARGS[@]}"
