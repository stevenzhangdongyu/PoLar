#!/usr/bin/env bash
set -euo pipefail

# Resume MCTS after changing the worker count.
#
# Unlike --resume on a fixed shard layout, this script:
#   1. scans all old shard_*.json files;
#   2. collects completed questions;
#   3. writes a temporary input JSON containing only unfinished questions;
#   4. launches the normal multi-worker JSON runner with the NEW worker count;
#   5. merges old and new shards with question-level deduplication.
#
# Arguments:
#   1 model_path
#   2 original_input_json
#   3 old_output_dir       (directory from the interrupted run)
#   4 new_output_dir       (directory for newly repartitioned workers)
#   5 simulations
#
# Example:
#   GPUS="0 1 2 3 4 5 6 7" WORKERS_PER_GPU=10 MAX_TOTAL_WORKERS=80 \
#   MAX_NEW_TOKENS=1024 HF_LOCAL_FILES_ONLY=1 \
#   bash scripts/resume_mcts_repartition_input_json.sh \
#     Qwen/Qwen2.5-3B-Instruct \
#     ./data/unique_inputs_diff5.json \
#     ./data/.../diff5/mcts_old_w5 \
#     ./data/.../diff5/mcts_resume_w10 \
#     100

MODEL_PATH="${1:?model_path is required}"
INPUT_JSON="${2:?original input_json is required}"
OLD_OUTPUT_DIR="${3:?old output directory is required}"
NEW_OUTPUT_DIR="${4:?new output directory is required}"
SIMULATIONS="${5:?simulations is required}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-10}"
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
DEDUP_MERGE="${DEDUP_MERGE:-1}"

if [[ ! -f "${INPUT_JSON}" ]]; then
  echo "Missing original input_json: ${INPUT_JSON}" >&2
  exit 1
fi
if [[ ! -d "${OLD_OUTPUT_DIR}" ]]; then
  echo "Missing old output directory: ${OLD_OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${NEW_OUTPUT_DIR}"
REMAINING_JSON="${NEW_OUTPUT_DIR}/.remaining_questions.json"
FINAL_OUTPUT="${NEW_OUTPUT_DIR}/merged_mcts_samples.json"

python3 - "${INPUT_JSON}" "${OLD_OUTPUT_DIR}" "${REMAINING_JSON}" "${QUESTION_FIELD}" <<'PY'
import glob
import json
import os
import sys

input_path, old_dir, remaining_path, question_field = sys.argv[1:]

def load_payload(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict) and "samples" in d:
        return d["samples"]
    if isinstance(d, dict):
        return list(d.values())
    return d

def get_value(row, field):
    cur = row
    for part in field.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return ""
    return cur

def q_of(row):
    # Completed MCTS records normally use `question`; support nested fields too.
    return str(
        row.get("question", "")
        or get_value(row, question_field)
        or row.get("query", "")
    ).strip()

completed = set()
old_files = sorted(glob.glob(os.path.join(old_dir, "shard_*.json")))
for path in old_files:
    try:
        rows = load_payload(path)
    except Exception as exc:
        print(f"[repartition] warning: cannot read {path}: {exc}", file=sys.stderr)
        continue
    if not isinstance(rows, list):
        continue
    for row in rows:
        if isinstance(row, dict):
            q = q_of(row)
            if q:
                completed.add(q)

rows = load_payload(input_path)
if not isinstance(rows, list):
    raise ValueError(f"Unsupported input shape: {type(rows)}")

seen_input = set()
remaining = []
duplicate_input = 0
already_done = 0
missing_question = 0
for row in rows:
    if not isinstance(row, dict):
        continue
    q = q_of(row)
    if not q:
        missing_question += 1
        continue
    if q in seen_input:
        duplicate_input += 1
        continue
    seen_input.add(q)
    if q in completed:
        already_done += 1
        continue
    remaining.append(row)

with open(remaining_path, "w", encoding="utf-8") as f:
    json.dump(remaining, f, ensure_ascii=False, indent=2)

print(f"[repartition] old shard files: {len(old_files)}")
print(f"[repartition] completed questions found: {len(completed)}")
print(f"[repartition] original input rows: {len(rows)}")
print(f"[repartition] duplicate input rows skipped: {duplicate_input}")
print(f"[repartition] already completed skipped: {already_done}")
print(f"[repartition] missing question rows skipped: {missing_question}")
print(f"[repartition] remaining questions: {len(remaining)}")
PY

REMAINING_COUNT="$(python3 - "${REMAINING_JSON}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    rows = json.load(f)
print(len(rows))
PY
)"

if [[ "${REMAINING_COUNT}" -gt 0 ]]; then
  GPUS="${GPUS}" \
  WORKERS_PER_GPU="${WORKERS_PER_GPU}" \
  MAX_TOTAL_WORKERS="${MAX_TOTAL_WORKERS}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  MAX_BLOCK="${MAX_BLOCK}" \
  MAX_REPEAT="${MAX_REPEAT}" \
  MAX_CHILDREN_PER_EXPAND="${MAX_CHILDREN_PER_EXPAND}" \
  MAX_PATH_LEN_RATIO="${MAX_PATH_LEN_RATIO}" \
  LENGTH_PENALTY="${LENGTH_PENALTY}" \
  SEED="${SEED}" \
  QUESTION_FIELD="${QUESTION_FIELD}" \
  ANSWER_FIELD="${ANSWER_FIELD}" \
  HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY}" \
  bash scripts/run_mcts_multiworker_input_json.sh \
    "${MODEL_PATH}" \
    "${REMAINING_JSON}" \
    "${REMAINING_COUNT}" \
    "${SIMULATIONS}" \
    "${NEW_OUTPUT_DIR}"
else
  echo "[repartition] no unfinished questions; skipping new MCTS workers"
fi

MERGE_ARGS=(
  --inputs
  "${OLD_OUTPUT_DIR}"/shard_*.json
  "${NEW_OUTPUT_DIR}"/shard_*.json
  --output "${FINAL_OUTPUT}"
)
if [[ "${DEDUP_MERGE}" == "1" || "${DEDUP_MERGE}" == "true" ]]; then
  MERGE_ARGS+=(--dedup_by_question)
fi
python3 scripts/merge_mcts_shards.py "${MERGE_ARGS[@]}"
echo "[done] merged old + resumed results: ${FINAL_OUTPUT}"
