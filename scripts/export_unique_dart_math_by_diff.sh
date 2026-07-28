#!/usr/bin/env bash
set -euo pipefail

# Export unique DART-Math questions for difficulties 1-5.
#
# Default output:
#   ./data/unique_inputs/dart_math_diff_1_unique_limit2000.json
#   ...
#   ./data/unique_inputs/dart_math_diff_5_unique_limit2000.json
#
# Optional env vars:
#   HF_DATASET=hkust-nlp/dart-math-pool-math
#   HF_SPLIT=train
#   OUTPUT_DIR=./data/unique_inputs
#   LIMIT_PER_DIFF=2000
#   SHUFFLE=0
#   KEEP_SOURCE_ROW=0
#   HF_LOCAL_FILES_ONLY=1

HF_DATASET="${HF_DATASET:-hkust-nlp/dart-math-pool-math}"
HF_SPLIT="${HF_SPLIT:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/unique_inputs}"
LIMIT_PER_DIFF="${LIMIT_PER_DIFF:-2000}"
SHUFFLE="${SHUFFLE:-0}"
KEEP_SOURCE_ROW="${KEEP_SOURCE_ROW:-0}"
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}"

EXTRA_ARGS=(--overwrite)
if [[ "${SHUFFLE}" == "1" || "${SHUFFLE,,}" == "true" ]]; then
  EXTRA_ARGS+=(--shuffle)
fi
if [[ "${KEEP_SOURCE_ROW}" == "1" || "${KEEP_SOURCE_ROW,,}" == "true" ]]; then
  EXTRA_ARGS+=(--keep_source_row)
fi

TOKENIZERS_PARALLELISM=false \
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY}" \
HF_HUB_OFFLINE="${HF_LOCAL_FILES_ONLY}" \
HF_DATASETS_OFFLINE="${HF_LOCAL_FILES_ONLY}" \
python3 scripts/export_unique_dart_math_by_diff.py \
  --hf_dataset "${HF_DATASET}" \
  --hf_split "${HF_SPLIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --limit_per_diff "${LIMIT_PER_DIFF}" \
  --difficulties 1 2 3 4 5 \
  "${EXTRA_ARGS[@]}"
