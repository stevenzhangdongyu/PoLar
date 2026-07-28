#!/usr/bin/env bash
set -euo pipefail

# Launch many independent MCTS workers over a local input JSON/JSONL.
#
# The input JSON should contain records with at least:
#   {"question": "...", "gt_ans": "..."}
#
# Example:
#   GPUS="0 1 2 3 4 5 6 7" WORKERS_PER_GPU=4 MAX_NEW_TOKENS=128 \
#   bash scripts/run_mcts_multiworker_input_json.sh \
#     Qwen/Qwen2.5-3B-Instruct \
#     ./data/unique_inputs/dart_math_diff_1_unique_limit2000.json \
#     2000 \
#     100 \
#     ./data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-1/multiworker_unique_from_json
#
# Arguments:
#   1 model_path
#   2 input_json
#   3 total_limit
#   4 simulations
#   5 output_dir
#
# Optional env vars:
#   GPUS="0 1 2 3 4 5 6 7"
#   WORKERS_PER_GPU=4
#   MAX_NEW_TOKENS=128
#   MAX_BLOCK=4
#   MAX_REPEAT=1
#   MAX_CHILDREN_PER_EXPAND=64
#   MAX_PATH_LEN_RATIO=1.15
#   LENGTH_PENALTY=0.05
#   SEED=42
#   QUESTION_FIELD=question
#   ANSWER_FIELD=gt_ans
#   HF_LOCAL_FILES_ONLY=1
#   MAX_TOTAL_WORKERS=0   # 0 means no cap; otherwise cap workers to at most this number.

MODEL_PATH="${1:?model_path is required}"
INPUT_JSON="${2:?input_json is required}"
TOTAL_LIMIT="${3:?total_limit is required}"
SIMULATIONS="${4:?simulations is required}"
OUTPUT_DIR="${5:?output_dir is required}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MAX_BLOCK="${MAX_BLOCK:-4}"
MAX_REPEAT="${MAX_REPEAT:-1}"
MAX_CHILDREN_PER_EXPAND="${MAX_CHILDREN_PER_EXPAND:-64}"
MAX_PATH_LEN_RATIO="${MAX_PATH_LEN_RATIO:-1.15}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.05}"
SEED="${SEED:-42}"
QUESTION_FIELD="${QUESTION_FIELD:-question}"
ANSWER_FIELD="${ANSWER_FIELD:-gt_ans}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
MAX_TOTAL_WORKERS="${MAX_TOTAL_WORKERS:-0}"

if [[ ! -f "${INPUT_JSON}" ]]; then
  echo "Missing input_json: ${INPUT_JSON}" >&2
  exit 1
fi

INPUT_COUNT="$(python3 - "${INPUT_JSON}" <<'PY'
import json, sys
p=sys.argv[1]
with open(p) as f:
    d=json.load(f)
s=d["samples"] if isinstance(d,dict) and "samples" in d else (list(d.values()) if isinstance(d,dict) else d)
print(len(s))
PY
)"
if [[ "${INPUT_COUNT}" -lt 1 ]]; then
  echo "No rows found in input_json: ${INPUT_JSON}" >&2
  exit 1
fi
if [[ "${TOTAL_LIMIT}" -gt "${INPUT_COUNT}" ]]; then
  echo "[launch-json] requested total_limit=${TOTAL_LIMIT}, but input has only ${INPUT_COUNT}; using ${INPUT_COUNT}"
  TOTAL_LIMIT="${INPUT_COUNT}"
fi

mkdir -p "${OUTPUT_DIR}"

read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "No GPU id found in GPUS='${GPUS}'" >&2
  exit 1
fi
if [[ "${WORKERS_PER_GPU}" -lt 1 ]]; then
  echo "WORKERS_PER_GPU must be >= 1" >&2
  exit 1
fi

TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
if [[ "${MAX_TOTAL_WORKERS}" -gt 0 && "${TOTAL_WORKERS}" -gt "${MAX_TOTAL_WORKERS}" ]]; then
  TOTAL_WORKERS="${MAX_TOTAL_WORKERS}"
fi
if [[ "${TOTAL_WORKERS}" -gt "${TOTAL_LIMIT}" ]]; then
  TOTAL_WORKERS="${TOTAL_LIMIT}"
fi
BASE_LIMIT=$((TOTAL_LIMIT / TOTAL_WORKERS))
REMAINDER=$((TOTAL_LIMIT % TOTAL_WORKERS))

echo "[launch-json] model=${MODEL_PATH}"
echo "[launch-json] input_json=${INPUT_JSON}"
echo "[launch-json] input_count=${INPUT_COUNT} total_limit=${TOTAL_LIMIT} simulations=${SIMULATIONS}"
echo "[launch-json] gpus=${GPUS} workers_per_gpu=${WORKERS_PER_GPU} total_workers=${TOTAL_WORKERS}"
echo "[launch-json] output_dir=${OUTPUT_DIR}"

START=0
GLOBAL_RANK=0
PIDS=()

for GPU_RANK in "${!GPU_ARRAY[@]}"; do
  GPU="${GPU_ARRAY[$GPU_RANK]}"
  for LOCAL_WORKER in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    if [[ "${GLOBAL_RANK}" -ge "${TOTAL_WORKERS}" ]]; then
      break 2
    fi
    SHARD_LIMIT="${BASE_LIMIT}"
    if [[ "${GLOBAL_RANK}" -lt "${REMAINDER}" ]]; then
      SHARD_LIMIT=$((SHARD_LIMIT + 1))
    fi

    if [[ "${SHARD_LIMIT}" -le 0 ]]; then
      GLOBAL_RANK=$((GLOBAL_RANK + 1))
      continue
    fi

    SHARD_OUTPUT="${OUTPUT_DIR}/shard_${GLOBAL_RANK}_gpu_${GPU}_worker_${LOCAL_WORKER}_start_${START}_limit_${SHARD_LIMIT}.json"
    LOG_FILE="${OUTPUT_DIR}/shard_${GLOBAL_RANK}_gpu_${GPU}_worker_${LOCAL_WORKER}.log"
    WORKER_SEED=$((SEED + GLOBAL_RANK))

    echo "[launch-json] rank=${GLOBAL_RANK} gpu=${GPU} local_worker=${LOCAL_WORKER} start_idx=${START} limit=${SHARD_LIMIT}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
    TOKENIZERS_PARALLELISM=false \
    HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY}" \
    HF_HUB_OFFLINE="${HF_LOCAL_FILES_ONLY}" \
    TRANSFORMERS_OFFLINE="${HF_LOCAL_FILES_ONLY}" \
    HF_DATASETS_OFFLINE="${HF_LOCAL_FILES_ONLY}" \
    python3 scripts/generate_mcts_samples.py \
      --model_path "${MODEL_PATH}" \
      --input_json "${INPUT_JSON}" \
      --question_field "${QUESTION_FIELD}" \
      --answer_field "${ANSWER_FIELD}" \
      --output "${SHARD_OUTPUT}" \
      --start_idx "${START}" \
      --limit "${SHARD_LIMIT}" \
      --simulations "${SIMULATIONS}" \
      --max_block "${MAX_BLOCK}" \
      --max_repeat "${MAX_REPEAT}" \
      --max_children_per_expand "${MAX_CHILDREN_PER_EXPAND}" \
      --max_path_len_ratio "${MAX_PATH_LEN_RATIO}" \
      --length_penalty "${LENGTH_PENALTY}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --device cuda \
      --seed "${WORKER_SEED}" \
      --resume \
      > "${LOG_FILE}" 2>&1 &

    PIDS+=("$!")
    START=$((START + SHARD_LIMIT))
    GLOBAL_RANK=$((GLOBAL_RANK + 1))
  done
done

echo "[launch-json] started ${#PIDS[@]} worker(s). Logs are in ${OUTPUT_DIR}/shard_*.log"

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=1
  fi
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "[done] at least one shard failed. Check ${OUTPUT_DIR}/shard_*.log" >&2
  exit 1
fi

MERGED_OUTPUT="${OUTPUT_DIR}/merged_mcts_samples.json"
python3 scripts/merge_mcts_shards.py \
  --inputs "${OUTPUT_DIR}"/shard_*.json \
  --output "${MERGED_OUTPUT}"

echo "[done] merged output: ${MERGED_OUTPUT}"
