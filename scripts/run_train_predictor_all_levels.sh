#!/usr/bin/env bash
set -euo pipefail

# Train one POLAR predictor on difficulty levels 1-5 together.
#
# Expected supervision files by default:
#   ${DATA_ROOT}/dart-math-diff-1/${MCTS_SUBDIR}/merged_mcts_samples.json
#   ...
#   ${DATA_ROOT}/dart-math-diff-5/${MCTS_SUBDIR}/merged_mcts_samples.json
#
# Example:
#   bash scripts/run_train_predictor_all_levels.sh
#
# Common overrides:
#   MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
#   DATA_ROOT=./data/Qwen/Qwen2.5-3B-Instruct
#   MCTS_SUBDIR=multiworker_unique
#   SAVE_DIR=./outputs
#   NUM_EPOCHS=20
#   BATCH_SIZE=16
#   LEARNING_RATE=1e-4
#   EMBEDDING_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
#
# If your merged files are in `multiworker` instead of `multiworker_unique`:
#   MCTS_SUBDIR=multiworker bash scripts/run_train_predictor_all_levels.sh

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
DATA_ROOT="${DATA_ROOT:-./data/Qwen/Qwen2.5-3B-Instruct}"
MCTS_SUBDIR="${MCTS_SUBDIR:-multiworker_unique}"
SAVE_DIR="${SAVE_DIR:-./outputs}"

MERGED_JSONS=()
for DIFF in 1 2 3 4 5; do
  P="${DATA_ROOT}/dart-math-diff-${DIFF}/${MCTS_SUBDIR}/merged_mcts_samples.json"
  if [[ ! -f "${P}" ]]; then
    echo "Missing diff-${DIFF} merged file: ${P}" >&2
    echo "Set MCTS_SUBDIR correctly, or generate/merge this difficulty first." >&2
    exit 1
  fi
  MERGED_JSONS+=("${P}")
done

echo "[train-all] model_path=${MODEL_PATH}"
echo "[train-all] data_root=${DATA_ROOT}"
echo "[train-all] mcts_subdir=${MCTS_SUBDIR}"
echo "[train-all] save_dir=${SAVE_DIR}"
for DIFF in 1 2 3 4 5; do
  echo "[train-all] diff-${DIFF}: ${MERGED_JSONS[$((DIFF - 1))]}"
done

FORCE_LINK_DATA="${FORCE_LINK_DATA:-1}" \
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" \
bash scripts/train_predictor.sh \
  "${MODEL_PATH}" \
  "${DATA_ROOT}" \
  all \
  "${SAVE_DIR}" \
  "${MERGED_JSONS[@]}"
