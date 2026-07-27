#!/usr/bin/env bash
set -euo pipefail

# Train the POLAR predictor from prepared MCTS supervision files.
#
# Example (single difficulty, diff1):
#   bash scripts/train_predictor.sh \
#     Qwen/Qwen2.5-3B-Instruct \
#     ./data/Qwen/Qwen2.5-3B-Instruct \
#     1 \
#     ./outputs \
#     ./data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-1/multiworker/merged_mcts_samples.json
#
# Example (all difficulties 1-5):
#   bash scripts/train_predictor.sh \
#     Qwen/Qwen2.5-3B-Instruct \
#     ./data/Qwen/Qwen2.5-3B-Instruct \
#     all \
#     ./outputs \
#     ./data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-1/multiworker/merged_mcts_samples.json \
#     ./data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-2/multiworker/merged_mcts_samples.json \
#     ./data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-3/multiworker/merged_mcts_samples.json \
#     ./data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-4/multiworker/merged_mcts_samples.json \
#     ./data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-5/multiworker/merged_mcts_samples.json
#
# Arguments:
#   1 model_path
#   2 data_root
#   3 diff_or_all        (1-5 or "all")
#   4 save_dir
#   5+ merged_json paths  (one for single diff, five for all; linked into data_root before training)
#
# Optional env vars:
#   NUM_EPOCHS=20
#   BATCH_SIZE=16
#   LEARNING_RATE=1e-4
#   MAX_PATHS_PER_SAMPLE=50
#   MAX_TOTAL_EXAMPLES=
#   USE_AMP=1
#   GRAD_CLIP_NORM=0.0
#   DROPOUT_ORIG=0
#   REWEIGHT_ORIG=0
#   KEEP_ORIG_PROB=0.0
#   ORIGINAL_PATH_WEIGHT=1.0
#   ANTI_ORIGINAL_LAMBDA=0.0
#   PER_SAMPLE_WEIGHT_NORMALIZE=0
#   POLAR_D_MODEL=256
#   POLAR_HEADS=4
#   POLAR_LAYERS=2
#   EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
#   LR_SCHEDULER=none
#   WARMUP_STEPS=0
#   HF_LOCAL_FILES_ONLY=1

MODEL_PATH="${1:?model_path is required}"
DATA_ROOT="${2:?data_root is required}"
DIFF_OR_ALL="${3:?diff_or_all is required}"
SAVE_DIR="${4:?save_dir is required}"
shift 4

if [[ "$#" -lt 1 ]]; then
  echo "At least one merged_json path is required." >&2
  exit 1
fi

MERGED_JSONS=("$@")

NUM_EPOCHS="${NUM_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_PATHS_PER_SAMPLE="${MAX_PATHS_PER_SAMPLE:-50}"
MAX_TOTAL_EXAMPLES="${MAX_TOTAL_EXAMPLES:-}"
USE_AMP="${USE_AMP:-1}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.0}"
DROPOUT_ORIG="${DROPOUT_ORIG:-0}"
REWEIGHT_ORIG="${REWEIGHT_ORIG:-0}"
KEEP_ORIG_PROB="${KEEP_ORIG_PROB:-0.0}"
ORIGINAL_PATH_WEIGHT="${ORIGINAL_PATH_WEIGHT:-1.0}"
ANTI_ORIGINAL_LAMBDA="${ANTI_ORIGINAL_LAMBDA:-0.0}"
PER_SAMPLE_WEIGHT_NORMALIZE="${PER_SAMPLE_WEIGHT_NORMALIZE:-0}"
POLAR_D_MODEL="${POLAR_D_MODEL:-256}"
POLAR_HEADS="${POLAR_HEADS:-4}"
POLAR_LAYERS="${POLAR_LAYERS:-2}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
LR_SCHEDULER="${LR_SCHEDULER:-none}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"
TARGET_DIFF="${TARGET_DIFF:-}"

mkdir -p "${SAVE_DIR}"

link_merged_json() {
  local diff="$1"
  local src="$2"
  local dst_dir="${DATA_ROOT}/dart-math-diff-${diff}"
  local dst="${dst_dir}/merged_mcts_samples.json"

  if [[ ! -f "${src}" ]]; then
    echo "Missing merged_json for diff-${diff}: ${src}" >&2
    exit 1
  fi

  mkdir -p "${dst_dir}"
  local abs_src
  abs_src="$(cd "$(dirname "${src}")" && pwd)/$(basename "${src}")"

  if [[ -e "${dst}" ]]; then
    if [[ "$(realpath "${dst}")" == "${abs_src}" ]]; then
      echo "[data] diff-${diff}: ${dst} already points to ${abs_src}"
      return
    fi
    if [[ "${FORCE_LINK_DATA:-0}" == "1" ]]; then
      echo "[data] diff-${diff}: replacing ${dst} -> ${abs_src}"
      rm -f "${dst}"
    else
      echo "Refusing to overwrite existing ${dst}" >&2
      echo "It points to: $(realpath "${dst}")" >&2
      echo "New source:   ${abs_src}" >&2
      echo "Set FORCE_LINK_DATA=1 if you want this script to replace it." >&2
      exit 1
    fi
  fi

  ln -s "${abs_src}" "${dst}"
  echo "[data] diff-${diff}: linked ${dst} -> ${abs_src}"
}

if [[ "${DIFF_OR_ALL}" == "all" ]]; then
  if [[ "${#MERGED_JSONS[@]}" -ne 5 ]]; then
    echo "Mode 'all' expects exactly 5 merged_json paths, one for each diff 1-5." >&2
    exit 1
  fi
  for i in 0 1 2 3 4; do
    link_merged_json "$((i + 1))" "${MERGED_JSONS[$i]}"
  done
else
  if [[ "${#MERGED_JSONS[@]}" -ne 1 ]]; then
    echo "Single-diff mode expects exactly 1 merged_json path." >&2
    exit 1
  fi
  link_merged_json "${DIFF_OR_ALL}" "${MERGED_JSONS[0]}"
fi

COMMON_ARGS=(
  --model_path "${MODEL_PATH}"
  --data_root "${DATA_ROOT}"
  --save_dir "${SAVE_DIR}"
  --num_epochs "${NUM_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --max_paths_per_sample "${MAX_PATHS_PER_SAMPLE}"
  --polar_d_model "${POLAR_D_MODEL}"
  --polar_heads "${POLAR_HEADS}"
  --polar_layers "${POLAR_LAYERS}"
  --embedding_model_name "${EMBEDDING_MODEL_NAME}"
  --lr_scheduler "${LR_SCHEDULER}"
  --warmup_steps "${WARMUP_STEPS}"
  --hf_cache_dir "${HF_HOME:-${TRANSFORMERS_CACHE:-}}"
  --seed "${SEED:-42}"
  --run_tag "${RUN_TAG:-}"
)

if [[ -n "${MAX_TOTAL_EXAMPLES}" ]]; then
  COMMON_ARGS+=(--max_total_examples "${MAX_TOTAL_EXAMPLES}")
fi
if [[ "${USE_AMP}" == "1" || "${USE_AMP,,}" == "true" ]]; then
  COMMON_ARGS+=(--use_amp)
fi
if [[ "${GRAD_CLIP_NORM}" != "0" && "${GRAD_CLIP_NORM}" != "0.0" ]]; then
  COMMON_ARGS+=(--grad_clip_norm "${GRAD_CLIP_NORM}")
fi
if [[ "${DROPOUT_ORIG}" == "1" || "${DROPOUT_ORIG,,}" == "true" ]]; then
  COMMON_ARGS+=(--drop_original_path_if_shorter_valid)
fi
if [[ "${REWEIGHT_ORIG}" == "1" || "${REWEIGHT_ORIG,,}" == "true" ]]; then
  COMMON_ARGS+=(--reweight_original_path_if_shorter_valid)
fi
if [[ "${KEEP_ORIG_PROB}" != "0" && "${KEEP_ORIG_PROB}" != "0.0" ]]; then
  COMMON_ARGS+=(--keep_original_prob "${KEEP_ORIG_PROB}")
fi
if [[ "${ORIGINAL_PATH_WEIGHT}" != "1" && "${ORIGINAL_PATH_WEIGHT}" != "1.0" ]]; then
  COMMON_ARGS+=(--original_path_weight "${ORIGINAL_PATH_WEIGHT}")
fi
if [[ "${ANTI_ORIGINAL_LAMBDA}" != "0" && "${ANTI_ORIGINAL_LAMBDA}" != "0.0" ]]; then
  COMMON_ARGS+=(--anti_original_lambda "${ANTI_ORIGINAL_LAMBDA}")
fi
if [[ "${PER_SAMPLE_WEIGHT_NORMALIZE}" == "1" || "${PER_SAMPLE_WEIGHT_NORMALIZE,,}" == "true" ]]; then
  COMMON_ARGS+=(--per_sample_weight_normalize)
fi
if [[ -n "${TARGET_DIFF}" ]]; then
  COMMON_ARGS+=(--target_diff "${TARGET_DIFF}")
elif [[ "${DIFF_OR_ALL}" != "all" ]]; then
  COMMON_ARGS+=(--target_diff "${DIFF_OR_ALL}")
else
  COMMON_ARGS+=(--eval_all_diffs)
fi

echo "[launch] model_path=${MODEL_PATH}"
echo "[launch] data_root=${DATA_ROOT}"
echo "[launch] save_dir=${SAVE_DIR}"
echo "[launch] mode=${DIFF_OR_ALL}"
echo "[launch] env: HF_LOCAL_FILES_ONLY=${HF_LOCAL_FILES_ONLY}"

HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY}" HF_HUB_OFFLINE="${HF_LOCAL_FILES_ONLY}" TRANSFORMERS_OFFLINE="${HF_LOCAL_FILES_ONLY}" HF_DATASETS_OFFLINE="${HF_LOCAL_FILES_ONLY}" \
  python3 run_polar.py "${COMMON_ARGS[@]}"
