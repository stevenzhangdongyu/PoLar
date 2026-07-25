#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash scripts/run_mcts_multigpu.sh \
#     meta-llama/Llama-3.2-3B-Instruct \
#     hkust-nlp/dart-math-pool-math \
#     train \
#     1 \
#     800 \
#     64 \
#     ./data/meta-llama/Llama-3.2-3B-Instruct/dart-math-diff-1/multigpu
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
#   MAX_NEW_TOKENS=50
#   MAX_BLOCK=4
#   MAX_REPEAT=1
#   MAX_CHILDREN_PER_EXPAND=64
#   MAX_PATH_LEN_RATIO=1.15
#   LENGTH_PENALTY=0.05

MODEL_PATH="${1:?model_path is required}"
HF_DATASET="${2:?hf_dataset is required}"
HF_SPLIT="${3:?hf_split is required}"
DIFFICULTY="${4:?difficulty is required}"
TOTAL_LIMIT="${5:?total_limit is required}"
SIMULATIONS="${6:?simulations is required}"
OUTPUT_DIR="${7:?output_dir is required}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"
MAX_BLOCK="${MAX_BLOCK:-4}"
MAX_REPEAT="${MAX_REPEAT:-1}"
MAX_CHILDREN_PER_EXPAND="${MAX_CHILDREN_PER_EXPAND:-64}"
MAX_PATH_LEN_RATIO="${MAX_PATH_LEN_RATIO:-1.15}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.05}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"

mkdir -p "${OUTPUT_DIR}"

read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "No GPU id found in GPUS='${GPUS}'" >&2
  exit 1
fi

BASE_LIMIT=$((TOTAL_LIMIT / NUM_GPUS))
REMAINDER=$((TOTAL_LIMIT % NUM_GPUS))

echo "[launch] model=${MODEL_PATH}"
echo "[launch] dataset=${HF_DATASET} split=${HF_SPLIT} difficulty=${DIFFICULTY}"
echo "[launch] total_limit=${TOTAL_LIMIT} simulations=${SIMULATIONS} gpus=${GPUS}"

START=0
PIDS=()
for RANK in "${!GPU_ARRAY[@]}"; do
  GPU="${GPU_ARRAY[$RANK]}"
  SHARD_LIMIT="${BASE_LIMIT}"
  if [[ "${RANK}" -lt "${REMAINDER}" ]]; then
    SHARD_LIMIT=$((SHARD_LIMIT + 1))
  fi

  if [[ "${SHARD_LIMIT}" -le 0 ]]; then
    continue
  fi

  SHARD_OUTPUT="${OUTPUT_DIR}/shard_${RANK}_start_${START}_limit_${SHARD_LIMIT}.json"
  LOG_FILE="${OUTPUT_DIR}/shard_${RANK}.log"

  echo "[launch] rank=${RANK} gpu=${GPU} start_idx=${START} limit=${SHARD_LIMIT}"
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
    --resume \
    > "${LOG_FILE}" 2>&1 &

  PIDS+=("$!")
  START=$((START + SHARD_LIMIT))
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
