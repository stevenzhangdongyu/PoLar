#!/usr/bin/env bash
set -euo pipefail

# Merge every readable shard_*.json from one or more MCTS output directories,
# deduplicate by question, print a short summary, and train one POLAR predictor.
#
# Arguments:
#   1 model_path
#   2 model_data_root   e.g. /data/Qwen/Qwen2.5-3B-Instruct
#   3 difficulty       1-5
#   4 merged_output
#   5 save_dir
#   6+ shard directories
#
# Example:
#   NUM_EPOCHS=20 BATCH_SIZE=16 USE_AMP=1 \
#   bash scripts/merge_successful_and_train_predictor.sh \
#     /models/Qwen2.5-3B-Instruct \
#     /data/Qwen/Qwen2.5-3B-Instruct \
#     5 \
#     /data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-5/merged_successful/merged_mcts_samples.json \
#     ./outputs \
#     /data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-5/mcts_old \
#     /data/Qwen/Qwen2.5-3B-Instruct/dart-math-diff-5/mcts_resume

MODEL_PATH="${1:?model_path is required}"
DATA_ROOT="${2:?model_data_root is required}"
DIFF="${3:?difficulty is required}"
MERGED_OUTPUT="${4:?merged_output is required}"
SAVE_DIR="${5:?save_dir is required}"
shift 5

if [[ "$#" -lt 1 ]]; then
  echo "At least one MCTS shard directory is required." >&2
  exit 1
fi

SHARD_INPUTS=()
for shard_dir in "$@"; do
  if [[ ! -d "${shard_dir}" ]]; then
    echo "[merge-train] warning: missing shard directory: ${shard_dir}" >&2
    continue
  fi
  SHARD_INPUTS+=("${shard_dir}/shard_*.json")
done

if [[ "${#SHARD_INPUTS[@]}" -eq 0 ]]; then
  echo "No existing shard directories were supplied." >&2
  exit 1
fi

python3 scripts/merge_mcts_shards.py \
  --inputs "${SHARD_INPUTS[@]}" \
  --output "${MERGED_OUTPUT}" \
  --dedup_by_question \
  --skip_invalid

python3 - "${MERGED_OUTPUT}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    payload = json.load(f)
rows = payload["samples"] if isinstance(payload, dict) else payload

questions = {
    str(row.get("question", "")).strip()
    for row in rows
    if isinstance(row, dict) and str(row.get("question", "")).strip()
}
with_valid = 0
total_valid_paths = 0
for row in rows:
    if not isinstance(row, dict):
        continue
    paths = (
        row.get("final_valid_transitions")
        or row.get("valid_paths")
        or row.get("valid_transitions")
        or []
    )
    if paths:
        with_valid += 1
        total_valid_paths += len(paths)

metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
print(f"[merge-train] loaded shards: {metadata.get('num_loaded_shards', 'unknown')}")
print(f"[merge-train] skipped shards: {metadata.get('num_skipped_shards', 'unknown')}")
print(f"[merge-train] merged samples: {len(rows)}")
print(f"[merge-train] unique non-empty questions: {len(questions)}")
print(f"[merge-train] samples with valid paths: {with_valid}")
print(f"[merge-train] total valid paths: {total_valid_paths}")
PY

FORCE_LINK_DATA="${FORCE_LINK_DATA:-1}" \
RUN_TAG="${RUN_TAG:-diff${DIFF}_successful_shards}" \
HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-1}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" \
bash scripts/train_predictor.sh \
  "${MODEL_PATH}" \
  "${DATA_ROOT}" \
  "${DIFF}" \
  "${SAVE_DIR}" \
  "${MERGED_OUTPUT}"
