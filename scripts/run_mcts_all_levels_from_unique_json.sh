#!/usr/bin/env bash
set -euo pipefail

# Run MCTS data generation for difficulties 1-5 from pre-exported unique JSONs.
#
# Expected input files:
#   ${INPUT_DIR}/dart_math_diff_1_unique_limit${LIMIT_PER_DIFF}.json
#   ...
#   ${INPUT_DIR}/dart_math_diff_5_unique_limit${LIMIT_PER_DIFF}.json
#
# Example:
#   GPUS="0 1 2 3 4 5 6 7" WORKERS_PER_GPU=4 MAX_NEW_TOKENS=128 \
#   bash scripts/run_mcts_all_levels_from_unique_json.sh
#
# Optional env vars:
#   MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
#   INPUT_DIR=./data/unique_inputs
#   DATA_ROOT=./data/Qwen/Qwen2.5-3B-Instruct
#   OUT_SUBDIR=multiworker_unique_from_json
#   LIMIT_PER_DIFF=2000
#   SIMULATIONS=100
#   PARALLEL_LEVELS=0  # 1 means launch diff1-5 concurrently; 0 means sequential.

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
INPUT_DIR="${INPUT_DIR:-./data/unique_inputs}"
DATA_ROOT="${DATA_ROOT:-./data/Qwen/Qwen2.5-3B-Instruct}"
OUT_SUBDIR="${OUT_SUBDIR:-multiworker_unique_from_json}"
LIMIT_PER_DIFF="${LIMIT_PER_DIFF:-2000}"
SIMULATIONS="${SIMULATIONS:-100}"
PARALLEL_LEVELS="${PARALLEL_LEVELS:-0}"

json_count() {
  python3 - "$1" <<'PY'
import json, sys
p=sys.argv[1]
with open(p) as f:
    d=json.load(f)
s=d["samples"] if isinstance(d,dict) and "samples" in d else (list(d.values()) if isinstance(d,dict) else d)
print(len(s))
PY
}

PIDS=()

for DIFF in 1 2 3 4 5; do
  INPUT_JSON="${INPUT_DIR}/dart_math_diff_${DIFF}_unique_limit${LIMIT_PER_DIFF}.json"
  OUTPUT_DIR="${DATA_ROOT}/dart-math-diff-${DIFF}/${OUT_SUBDIR}"
  if [[ ! -f "${INPUT_JSON}" ]]; then
    echo "Missing input JSON for diff-${DIFF}: ${INPUT_JSON}" >&2
    exit 1
  fi
  ACTUAL_LIMIT="$(json_count "${INPUT_JSON}")"
  if [[ "${ACTUAL_LIMIT}" -gt "${LIMIT_PER_DIFF}" ]]; then
    ACTUAL_LIMIT="${LIMIT_PER_DIFF}"
  fi
  echo "[all-levels] diff-${DIFF}: input=${INPUT_JSON} limit=${ACTUAL_LIMIT} output=${OUTPUT_DIR}"
  if [[ "${PARALLEL_LEVELS}" == "1" || "${PARALLEL_LEVELS,,}" == "true" ]]; then
    bash scripts/run_mcts_multiworker_input_json.sh \
      "${MODEL_PATH}" \
      "${INPUT_JSON}" \
      "${ACTUAL_LIMIT}" \
      "${SIMULATIONS}" \
      "${OUTPUT_DIR}" &
    PIDS+=("$!")
  else
    bash scripts/run_mcts_multiworker_input_json.sh \
    "${MODEL_PATH}" \
    "${INPUT_JSON}" \
    "${ACTUAL_LIMIT}" \
    "${SIMULATIONS}" \
    "${OUTPUT_DIR}"
  fi
done

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  FAILED=0
  for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
      FAILED=1
    fi
  done
  if [[ "${FAILED}" -ne 0 ]]; then
    echo "[done] at least one difficulty failed." >&2
    exit 1
  fi
fi

echo "[done] generated all levels under ${DATA_ROOT}/dart-math-diff-*/${OUT_SUBDIR}"
