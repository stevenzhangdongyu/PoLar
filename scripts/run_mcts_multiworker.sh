#!/usr/bin/env bash
set -euo pipefail

# Launch many independent MCTS data-generation workers.
#
# This is designed for large-memory GPUs such as A100 80GB. Instead of launching
# only one process per GPU, it launches WORKERS_PER_GPU processes on each visible
# GPU. Every worker receives a different start_idx/limit slice and writes a
# separate shard JSON. After all workers finish, shards are merged.
#
# Example, difficulty 1, 2000 samples, 8 GPUs, 4 workers per GPU:
#
#   GPUS="0 1 2 3 4 5 6 7" WORKERS_PER_GPU=4 bash scripts/run_mcts_multiworker.sh \
#     meta-llama/Llama-3.2-3B-Instruct \
#     hkust-nlp/dart-math-pool-math \
#     train \
#     1 \
#     2000 \
#     64 \
#     ./data/meta-llama/Llama-3.2-3B-Instruct/dart-math-diff-1/multiworker
#
# Arguments:
#   1 model_path
#   2 hf_dataset
#   3 hf_split
#   4 difficulty
#   5 total_limit
#   6 simulations
#   7 output_dir
#
# Optional env vars:
#   GPUS="0 1 2 3 4 5 6 7"
#   WORKERS_PER_GPU=4
#   MAX_NEW_TOKENS=50
#   MAX_BLOCK=4
#   MAX_REPEAT=1
#   MAX_CHILDREN_PER_EXPAND=64
#   MAX_PATH_LEN_RATIO=1.15
#   LENGTH_PENALTY=0.05
#   SEED=42
#   DEDUP_QUESTIONS=1

MODEL_PATH="${1:?model_path is required}"
HF_DATASET="${2:?hf_dataset is required}"
HF_SPLIT="${3:?hf_split is required}"
DIFFICULTY="${4:?difficulty is required}"
TOTAL_LIMIT="${5:?total_limit is required}"
SIMULATIONS="${6:?simulations is required}"
OUTPUT_DIR="${7:?output_dir is required}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"
MAX_BLOCK="${MAX_BLOCK:-4}"
MAX_REPEAT="${MAX_REPEAT:-1}"
MAX_CHILDREN_PER_EXPAND="${MAX_CHILDREN_PER_EXPAND:-64}"
MAX_PATH_LEN_RATIO="${MAX_PATH_LEN_RATIO:-1.15}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.05}"
SEED="${SEED:-42}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
DEDUP_QUESTIONS="${DEDUP_QUESTIONS:-1}"

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
BASE_LIMIT=$((TOTAL_LIMIT / TOTAL_WORKERS))
REMAINDER=$((TOTAL_LIMIT % TOTAL_WORKERS))

echo "[launch] model=${MODEL_PATH}"
echo "[launch] dataset=${HF_DATASET} split=${HF_SPLIT} difficulty=${DIFFICULTY}"
echo "[launch] total_limit=${TOTAL_LIMIT} simulations=${SIMULATIONS}"
echo "[launch] gpus=${GPUS} workers_per_gpu=${WORKERS_PER_GPU} total_workers=${TOTAL_WORKERS}"
echo "[launch] output_dir=${OUTPUT_DIR}"

START=0
GLOBAL_RANK=0
PIDS=()

for GPU_RANK in "${!GPU_ARRAY[@]}"; do
  GPU="${GPU_ARRAY[$GPU_RANK]}"
  for LOCAL_WORKER in $(seq 0 $((WORKERS_PER_GPU - 1))); do
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

    echo "[launch] rank=${GLOBAL_RANK} gpu=${GPU} local_worker=${LOCAL_WORKER} start_idx=${START} limit=${SHARD_LIMIT}"
    EXTRA_ARGS=()
    if [[ "${DEDUP_QUESTIONS}" == "1" || "${DEDUP_QUESTIONS,,}" == "true" ]]; then
      EXTRA_ARGS+=(--dedup_questions)
    fi

    CUDA_VISIBLE_DEVICES="${GPU}" TOKENIZERS_PARALLELISM=false HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY}" HF_HUB_OFFLINE="${HF_LOCAL_FILES_ONLY}" TRANSFORMERS_OFFLINE="${HF_LOCAL_FILES_ONLY}" HF_DATASETS_OFFLINE="${HF_LOCAL_FILES_ONLY}" python3 scripts/generate_mcts_samples.py \
      --model_path "${MODEL_PATH}" \
      --hf_dataset "${HF_DATASET}" \
      --hf_split "${HF_SPLIT}" \
      --difficulty "${DIFFICULTY}" \
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
      "${EXTRA_ARGS[@]}" \
      > "${LOG_FILE}" 2>&1 &

    PIDS+=("$!")
    START=$((START + SHARD_LIMIT))
    GLOBAL_RANK=$((GLOBAL_RANK + 1))
  done
done

echo "[launch] started ${#PIDS[@]} worker(s). Logs are in ${OUTPUT_DIR}/shard_*.log"

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
  --output "${MERGED_OUTPUT}" \
  --dedup_by_question

echo "[done] merged output: ${MERGED_OUTPUT}"
