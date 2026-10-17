#!/usr/bin/env bash
set -euo pipefail
# Usage:
# bash scripts/run_math_selector_pipeline.sh model predictor_ckpt math_json work_dir test_eval_json...
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?model path required}"
PRED="${2:?POLAR predictor checkpoint required}"
MATH_JSON="${3:?math JSON/JSONL required}"
WORK_DIR="${4:?work dir required}"
shift 4
if [[ $# -lt 1 ]]; then
  echo "Need at least one held-out eval JSON for generalization test." >&2
  exit 1
fi
MATH_OUT="${WORK_DIR}/math_selector_data"
SEL_OUT="${WORK_DIR}/selector_mlp_math"

echo "[1/3] build math selector dataset -> ${MATH_OUT}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}" \
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}" \
MAX_TOTAL_WORKERS="${MAX_TOTAL_WORKERS:-0}" \
LIMIT="${MATH_LIMIT:-0}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}" \
BEAM_SIZE="${BEAM_SIZE:-10}" \
TOP_K_PATHS="${TOP_K_PATHS:-5}" \
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
bash "${ROOT}/scripts/run_math_selector_data_multiworker.sh" \
  "$MODEL" "$PRED" "$MATH_JSON" "$MATH_OUT"

echo "[2/3] train selector on math data -> ${SEL_OUT}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
NUM_EPOCHS="${NUM_EPOCHS:-5}" BATCH_SIZE="${BATCH_SIZE:-32}" LEARNING_RATE="${LEARNING_RATE:-1e-4}" \
bash "${ROOT}/scripts/run_train_path_selector.sh" \
  "$MODEL" "$SEL_OUT" "${MATH_OUT}/merged_results.json"

echo "[3/3] zero-shot generalization eval"
for eval_json in "$@"; do
  name="$(basename "$(dirname "$eval_json")")"
  out="${SEL_OUT}/eval_${name}.json"
  echo "[eval] $eval_json -> $out"
  HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  python3 "${ROOT}/scripts/eval_path_selector.py" \
    --checkpoint "${SEL_OUT}/selector_mlp_best.pt" \
    --eval_json "$eval_json" \
    --output_json "$out" \
    --local_files_only
done

echo "[done] ${WORK_DIR}"
