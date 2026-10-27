#!/usr/bin/env bash
set -euo pipefail

# Resume many independent MCTS workers over a local input JSON/JSONL.
#
# This is a resume-safe wrapper around scripts/generate_mcts_samples.py:
#   - it uses deterministic shard filenames based on rank/start/limit;
#   - each worker passes --resume and skips questions already present in its shard;
#   - it never deletes existing shard files or logs;
#   - after all workers finish, it merges all shard_*.json files.
#
# IMPORTANT: resume with the same input_json, total_limit, GPUS,
# WORKERS_PER_GPU/MAX_TOTAL_WORKERS, and output_dir used in the interrupted run.
# Otherwise shard start/limit ranges may change and work may be duplicated.
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
#   WORKERS_PER_GPU=12
#   MAX_TOTAL_WORKERS=96
#   MAX_NEW_TOKENS=1024
#   MAX_BLOCK=4
#   MAX_REPEAT=1
#   MAX_CHILDREN_PER_EXPAND=64
#   MAX_PATH_LEN_RATIO=1.15
#   LENGTH_PENALTY=0.05
#   SEED=42
#   QUESTION_FIELD=question
#   ANSWER_FIELD=gt_ans
#   HF_LOCAL_FILES_ONLY=1
#   POLAR_KV_CACHE_MODE=layer    # layer = legacy, execution = separate KV slot per path execution step.
#   DEDUP_MERGE=1

MODEL_PATH="${1:?model_path is required}"
INPUT_JSON="${2:?input_json is required}"
TOTAL_LIMIT="${3:?total_limit is required}"
SIMULATIONS="${4:?simulations is required}"
OUTPUT_DIR="${5:?output_dir is required}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-12}"
MAX_TOTAL_WORKERS="${MAX_TOTAL_WORKERS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_BLOCK="${MAX_BLOCK:-4}"
MAX_REPEAT="${MAX_REPEAT:-1}"
MAX_CHILDREN_PER_EXPAND="${MAX_CHILDREN_PER_EXPAND:-64}"
MAX_PATH_LEN_RATIO="${MAX_PATH_LEN_RATIO:-1.15}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.05}"
SEED="${SEED:-42}"
QUESTION_FIELD="${QUESTION_FIELD:-question}"
ANSWER_FIELD="${ANSWER_FIELD:-gt_ans}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
POLAR_KV_CACHE_MODE="${POLAR_KV_CACHE_MODE:-layer}"
DEDUP_MERGE="${DEDUP_MERGE:-1}"

if [[ ! -f "${INPUT_JSON}" ]]; then
  echo "Missing input_json: ${INPUT_JSON}" >&2
  exit 1
fi

INPUT_COUNT="$(python3 - "${INPUT_JSON}" <<'PY'
import json, sys
p=sys.argv[1]
with open(p) as f:
    d=json.load(f)
if isinstance(d, dict) and "samples" in d:
    rows=d["samples"]
elif isinstance(d, dict):
    rows=list(d.values())
else:
    rows=d
print(len(rows))
PY
)"

if [[ "${INPUT_COUNT}" -lt 1 ]]; then
  echo "No rows found in input_json: ${INPUT_JSON}" >&2
  exit 1
fi
if [[ "${TOTAL_LIMIT}" -gt "${INPUT_COUNT}" ]]; then
  echo "[resume-json] requested total_limit=${TOTAL_LIMIT}, but input has only ${INPUT_COUNT}; using ${INPUT_COUNT}"
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

echo "[resume-json] model=${MODEL_PATH}"
echo "[resume-json] input_json=${INPUT_JSON}"
echo "[resume-json] input_count=${INPUT_COUNT} total_limit=${TOTAL_LIMIT} simulations=${SIMULATIONS}"
echo "[resume-json] gpus=${GPUS} workers_per_gpu=${WORKERS_PER_GPU} total_workers=${TOTAL_WORKERS}"
echo "[resume-json] output_dir=${OUTPUT_DIR}"
echo "[resume-json] max_new_tokens=${MAX_NEW_TOKENS} max_block=${MAX_BLOCK} max_repeat=${MAX_REPEAT}"
echo "[resume-json] polar_kv_cache_mode=${POLAR_KV_CACHE_MODE}"
echo "[resume-json] NOTE: do not change worker count/output_dir between interrupted and resumed runs."

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
    LOG_FILE="${OUTPUT_DIR}/shard_${GLOBAL_RANK}_gpu_${GPU}_worker_${LOCAL_WORKER}.resume.log"
    WORKER_SEED=$((SEED + GLOBAL_RANK))

    if [[ -f "${SHARD_OUTPUT}" ]]; then
      DONE_COUNT="$(python3 - "${SHARD_OUTPUT}" <<'PY'
import json, sys, os
p=sys.argv[1]
if not os.path.exists(p):
    print(0); raise SystemExit
with open(p) as f:
    d=json.load(f)
rows=d.get("samples", d if isinstance(d, list) else [])
print(len(rows))
PY
)"
    else
      DONE_COUNT=0
    fi

    echo "[resume-json] rank=${GLOBAL_RANK} gpu=${GPU} local_worker=${LOCAL_WORKER} start_idx=${START} limit=${SHARD_LIMIT} existing=${DONE_COUNT}"

    CUDA_VISIBLE_DEVICES="${GPU}" \
    TOKENIZERS_PARALLELISM=false \
    HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY}" \
    HF_HUB_OFFLINE="${HF_LOCAL_FILES_ONLY}" \
    TRANSFORMERS_OFFLINE="${HF_LOCAL_FILES_ONLY}" \
    HF_DATASETS_OFFLINE="${HF_LOCAL_FILES_ONLY}" \
    POLAR_KV_CACHE_MODE="${POLAR_KV_CACHE_MODE}" \
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
      --polar_kv_cache_mode "${POLAR_KV_CACHE_MODE}" \
      --device cuda \
      --seed "${WORKER_SEED}" \
      --resume \
      > "${LOG_FILE}" 2>&1 &

    PIDS+=("$!")
    START=$((START + SHARD_LIMIT))
    GLOBAL_RANK=$((GLOBAL_RANK + 1))
  done
done

echo "[resume-json] started ${#PIDS[@]} worker(s). Resume logs are in ${OUTPUT_DIR}/shard_*.resume.log"

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=1
  fi
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "[done] at least one shard failed. Check ${OUTPUT_DIR}/shard_*.resume.log" >&2
  exit 1
fi

MERGED_OUTPUT="${OUTPUT_DIR}/merged_mcts_samples.json"
MERGE_ARGS=(--inputs "${OUTPUT_DIR}"/shard_*.json --output "${MERGED_OUTPUT}")
if [[ "${DEDUP_MERGE}" == "1" || "${DEDUP_MERGE}" == "true" ]]; then
  MERGE_ARGS+=(--dedup_by_question)
fi
python3 scripts/merge_mcts_shards.py "${MERGE_ARGS[@]}"

echo "[done] merged output: ${MERGED_OUTPUT}"
