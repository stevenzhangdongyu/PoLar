#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
DATA_ROOT="${DATA_ROOT:-./data/Qwen/Qwen2.5-3B-Instruct}"
WORK_ROOT="${WORK_ROOT:-./outputs/mcts_predictor_pipeline}"
SAVE_DIR="${SAVE_DIR:-./outputs/predictor_pipeline}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
LIMIT_DEFAULT="${PROBLEMS_PER_DIFF:-2000}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
TEST_RATIO="${TEST_RATIO:-0.1}"
VAL_RATIO="${VAL_RATIO:-0.1}"
KV_MODE="${POLAR_KV_CACHE_MODE:-execution}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
MAX_TOTAL_WORKERS="${MAX_TOTAL_WORKERS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
SIMULATIONS="${SIMULATIONS:-100}"
SEED="${SEED:-42}"
RESUME_RUN="${RESUME_RUN:-0}"
if [[ "$RESUME_RUN" == "1" && -z "${RUN_TAG:-}" && -f "${WORK_ROOT}/latest_run" ]]; then
  RUN_TAG="$(<"${WORK_ROOT}/latest_run")"
fi
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${WORK_ROOT}/${RUN_TAG}"
INPUT_DIR="${INPUT_DIR:-${RUN_DIR}/unique_splits}"
OFFLINE="${HF_LOCAL_FILES_ONLY:-1}"

MAX_BLOCK="${MAX_BLOCK:-4}"
MAX_REPEAT="${MAX_REPEAT:-1}"
MAX_CHILDREN_PER_EXPAND="${MAX_CHILDREN_PER_EXPAND:-64}"
MAX_PATH_LEN_RATIO="${MAX_PATH_LEN_RATIO:-1.15}"
LENGTH_PENALTY="${LENGTH_PENALTY:-0.05}"
QUESTION_FIELD="${QUESTION_FIELD:-question}"
ANSWER_FIELD="${ANSWER_FIELD:-gt_ans}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_PATHS_PER_SAMPLE="${MAX_PATHS_PER_SAMPLE:-50}"
POLAR_D_MODEL="${POLAR_D_MODEL:-256}"
POLAR_HEADS="${POLAR_HEADS:-4}"
POLAR_LAYERS="${POLAR_LAYERS:-2}"
PREDICTOR_RUN_TAG="${PREDICTOR_RUN_TAG:-${RUN_TAG}_kv_${KV_MODE}}"

if [[ "$KV_MODE" != "layer" && "$KV_MODE" != "execution" ]]; then
  echo "POLAR_KV_CACHE_MODE must be layer or execution." >&2
  exit 1
fi
if [[ "$WORKERS_PER_GPU" -lt 1 || "$MAX_TOTAL_WORKERS" -lt 0 ]]; then
  echo "WORKERS_PER_GPU must be >= 1 and MAX_TOTAL_WORKERS must be >= 0." >&2
  exit 1
fi
if [[ -e "$RUN_DIR" && "$RESUME_RUN" != "1" ]]; then
  echo "Run directory exists: $RUN_DIR. Set RESUME_RUN=1 and reuse the same RUN_TAG to resume." >&2
  exit 1
fi
if [[ ! -e "$RUN_DIR" && "$RESUME_RUN" == "1" ]]; then
  echo "Cannot resume; run directory does not exist: $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
mkdir -p "$WORK_ROOT"
printf '%s\n' "$RUN_TAG" > "$WORK_ROOT/latest_run"

PREPARE_ARGS=(--output_dir "$INPUT_DIR" --train_ratio "$TRAIN_RATIO"
  --test_ratio "$TEST_RATIO" --val_ratio "$VAL_RATIO" --seed "$SEED")
for DIFF in 1 2 3 4 5; do
  VAR="PROBLEMS_DIFF${DIFF}"
  CAP="${!VAR:-$LIMIT_DEFAULT}"
  PREPARE_ARGS+=("--max_unique_diff${DIFF}" "$CAP")
done

DIFF_INPUT_VARS=(INPUT_JSON_DIFF1 INPUT_JSON_DIFF2 INPUT_JSON_DIFF3 INPUT_JSON_DIFF4 INPUT_JSON_DIFF5)
DIFF_INPUT_COUNT=0
for VAR in "${DIFF_INPUT_VARS[@]}"; do
  [[ -n "${!VAR:-}" ]] && DIFF_INPUT_COUNT=$((DIFF_INPUT_COUNT + 1))
done
if [[ "$DIFF_INPUT_COUNT" -gt 0 ]]; then
  if [[ "$DIFF_INPUT_COUNT" -ne 5 || -n "${INPUT_JSON:-}" ]]; then
    echo "Set all INPUT_JSON_DIFF1..INPUT_JSON_DIFF5, or use INPUT_JSON, or use the default HF dataset." >&2
    exit 1
  fi
  PREPARE_ARGS+=(--input_jsons_by_diff \
    "$INPUT_JSON_DIFF1" "$INPUT_JSON_DIFF2" "$INPUT_JSON_DIFF3" \
    "$INPUT_JSON_DIFF4" "$INPUT_JSON_DIFF5")
elif [[ -n "${INPUT_JSON:-}" ]]; then
  PREPARE_ARGS+=(--input_json "$INPUT_JSON")
else
  PREPARE_ARGS+=(--hf_dataset "${HF_DATASET:-hkust-nlp/dart-math-pool-math}"
    --hf_split "${HF_SPLIT:-train}")
fi

CONFIG_TMP="$RUN_DIR/pipeline_config.txt.tmp"
CONFIG_FILE="$RUN_DIR/pipeline_config.txt"
{
  printf '%s\n' "$MODEL_PATH" "$DATA_ROOT" "$INPUT_DIR" "$WORK_ROOT" "$SAVE_DIR" "$GPUS" \
    "$WORKERS_PER_GPU" "$MAX_TOTAL_WORKERS" "$MAX_NEW_TOKENS" \
    "$SIMULATIONS" "$KV_MODE" "$SEED" "$TRAIN_RATIO" "$TEST_RATIO" "$VAL_RATIO"
  printf '%s\n' "${MAX_BLOCK:-4}" "${MAX_REPEAT:-1}" \
    "${MAX_CHILDREN_PER_EXPAND:-64}" "${MAX_PATH_LEN_RATIO:-1.15}" \
    "${LENGTH_PENALTY:-0.05}" "${QUESTION_FIELD:-question}" "${ANSWER_FIELD:-gt_ans}" \
    "$NUM_EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" \
    "$MAX_PATHS_PER_SAMPLE" "$POLAR_D_MODEL" \
    "$POLAR_HEADS" "$POLAR_LAYERS" \
    "$EMBEDDING_MODEL_NAME" \
    "$PREDICTOR_RUN_TAG"
  for DIFF in 1 2 3 4 5; do
    VAR="PROBLEMS_DIFF${DIFF}"
    printf '%s\n' "${!VAR:-$LIMIT_DEFAULT}"
  done
  for VAR in "${DIFF_INPUT_VARS[@]}"; do
    SOURCE_PATH="${!VAR:-}"
    if [[ -n "$SOURCE_PATH" ]]; then
      if [[ ! -f "$SOURCE_PATH" ]]; then
        echo "Missing source file: $SOURCE_PATH" >&2
        exit 1
      fi
      sha256sum "$SOURCE_PATH"
    fi
  done
  if [[ -n "${INPUT_JSON:-}" ]]; then
    sha256sum "$INPUT_JSON"
  else
    printf '%s\n' "${HF_DATASET:-hkust-nlp/dart-math-pool-math}" "${HF_SPLIT:-train}"
  fi
} > "$CONFIG_TMP"
if [[ "$RESUME_RUN" == "1" ]]; then
  if [[ ! -f "$CONFIG_FILE" ]] || ! cmp -s "$CONFIG_TMP" "$CONFIG_FILE"; then
    rm -f "$CONFIG_TMP"
    echo "Resume settings or source-data checksum differ from this run. Reuse the original configuration." >&2
    exit 1
  fi
  rm -f "$CONFIG_TMP"
else
  mv "$CONFIG_TMP" "$CONFIG_FILE"
fi
if [[ -f "$RUN_DIR/pipeline_complete" && "$RESUME_RUN" == "1" ]]; then
  echo "[resume] pipeline already completed: $RUN_DIR"
  exit 0
fi

echo "[pipeline] export and split unique problems"
SPLITS_READY=1
for DIFF in 1 2 3 4 5; do
  for SPLIT in train test val; do
    [[ -f "$INPUT_DIR/diff${DIFF}/${SPLIT}.json" ]] || SPLITS_READY=0
  done
done
if [[ "$RESUME_RUN" == "1" && "$SPLITS_READY" == "1" && -f "$INPUT_DIR/split_summary.json" ]]; then
  echo "[resume] reusing existing unique-question splits in $INPUT_DIR"
else
  TOKENIZERS_PARALLELISM=false HF_LOCAL_FILES_ONLY="$OFFLINE" \
  HF_HUB_OFFLINE="$OFFLINE" HF_DATASETS_OFFLINE="$OFFLINE" \
  python3 "$ROOT/scripts/prepare_dart_math_splits.py" "${PREPARE_ARGS[@]}"
fi

for DIFF in 1 2 3 4 5; do
  for SPLIT in train test val; do
    INPUT="${INPUT_DIR}/diff${DIFF}/${SPLIT}.json"
    COUNT="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$INPUT")"
    OUT="${RUN_DIR}/mcts/diff${DIFF}/${SPLIT}"
    if [[ "$COUNT" -eq 0 ]]; then
      echo "[pipeline] empty diff${DIFF}/${SPLIT}; skipping MCTS"
      mkdir -p "$OUT"
      printf '{"metadata":{"num_samples":0},"samples":[]}\n' > "$OUT/merged_mcts_samples.json"
      continue
    fi
    MCTS_CONFIG="$RUN_DIR/config/mcts_diff${DIFF}_${SPLIT}.json"
    mkdir -p "$(dirname "$MCTS_CONFIG")"
    python3 - "$MCTS_CONFIG" "$MODEL_PATH" "$INPUT" "$COUNT" "$SIMULATIONS" \
      "$GPUS" "$WORKERS_PER_GPU" "$MAX_TOTAL_WORKERS" "$MAX_NEW_TOKENS" \
      "$MAX_BLOCK" "$MAX_REPEAT" "$MAX_CHILDREN_PER_EXPAND" \
      "$MAX_PATH_LEN_RATIO" "$LENGTH_PENALTY" "$SEED" "$QUESTION_FIELD" \
      "$ANSWER_FIELD" "$KV_MODE" "$OFFLINE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(output, model, input_path, count, simulations, gpus, workers_per_gpu,
 max_workers, max_new_tokens, max_block, max_repeat, max_children,
 max_path_ratio, length_penalty, seed, question_field, answer_field,
 kv_mode, offline) = sys.argv[1:]
count, workers_per_gpu, max_workers = int(count), int(workers_per_gpu), int(max_workers)
gpu_ids = gpus.split()
worker_count = min(len(gpu_ids) * workers_per_gpu, max_workers or count, count)
base, remainder = divmod(count, worker_count)
workers = []
start = 0
for rank in range(worker_count):
    limit = base + (1 if rank < remainder else 0)
    workers.append({"rank": rank, "gpu": gpu_ids[rank // workers_per_gpu],
                    "local_worker": rank % workers_per_gpu, "start_idx": start,
                    "limit": limit, "seed": int(seed) + rank})
    start += limit
digest = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
config = {
    "stage": "mcts_search", "model_path": model, "input_json": input_path,
    "input_sha256": digest, "question_count": count,
    "question_field": question_field, "answer_field": answer_field,
    "simulations_per_question": int(simulations), "max_new_tokens": int(max_new_tokens),
    "search": {"max_block": int(max_block), "max_repeat": int(max_repeat),
               "max_children_per_expand": int(max_children),
               "max_path_len_ratio": float(max_path_ratio),
               "length_penalty": float(length_penalty)},
    "kv_cache_mode": kv_mode, "seed": int(seed), "offline_mode": offline,
    "requested_gpus": gpu_ids, "workers_per_gpu": workers_per_gpu,
    "max_total_workers": max_workers, "effective_workers": worker_count,
    "worker_shards": workers,
}
Path(output).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
    echo "[pipeline] MCTS config: $MCTS_CONFIG"
    cat "$MCTS_CONFIG"
    echo "[pipeline] MCTS diff=${DIFF} split=${SPLIT} problems=${COUNT} max_new_tokens=${MAX_NEW_TOKENS} kv=${KV_MODE}"
    GPUS="$GPUS" WORKERS_PER_GPU="$WORKERS_PER_GPU" MAX_TOTAL_WORKERS="$MAX_TOTAL_WORKERS" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" SIMULATIONS="$SIMULATIONS" SEED="$SEED" \
    RESUME_RUN="$RESUME_RUN" \
    MAX_BLOCK="$MAX_BLOCK" MAX_REPEAT="$MAX_REPEAT" \
    MAX_CHILDREN_PER_EXPAND="$MAX_CHILDREN_PER_EXPAND" \
    MAX_PATH_LEN_RATIO="$MAX_PATH_LEN_RATIO" LENGTH_PENALTY="$LENGTH_PENALTY" \
    QUESTION_FIELD="$QUESTION_FIELD" ANSWER_FIELD="$ANSWER_FIELD" \
    POLAR_KV_CACHE_MODE="$KV_MODE" HF_LOCAL_FILES_ONLY="$OFFLINE" \
    HF_HUB_OFFLINE="$OFFLINE" TRANSFORMERS_OFFLINE="$OFFLINE" HF_DATASETS_OFFLINE="$OFFLINE" \
    bash "$ROOT/scripts/run_mcts_multiworker_input_json.sh" \
      "$MODEL_PATH" "$INPUT" "$COUNT" "$SIMULATIONS" "$OUT"
  done
done

TRAIN_FILES=()
VAL_FILES=()
TEST_FILES=()
for DIFF in 1 2 3 4 5; do
  for SPLIT in train test val; do
    SRC="${RUN_DIR}/mcts/diff${DIFF}/${SPLIT}/merged_mcts_samples.json"
    case "$SPLIT" in
      train) TRAIN_FILES+=("$SRC") ;;
      val) VAL_FILES+=("$SRC") ;;
      test) TEST_FILES+=("$SRC") ;;
    esac
  done
done

echo "[pipeline] train predictor on train splits; checkpoint selected by validation loss"
echo "[pipeline] embedding_model_name=$EMBEDDING_MODEL_NAME"
PREDICTOR_CONFIG="$RUN_DIR/config/predictor_train.json"
mkdir -p "$(dirname "$PREDICTOR_CONFIG")"
python3 - "$PREDICTOR_CONFIG" "$MODEL_PATH" "$DATA_ROOT" "$SAVE_DIR" \
  "$EMBEDDING_MODEL_NAME" "$NUM_EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" \
  "$SEED" "$MAX_PATHS_PER_SAMPLE" "$POLAR_D_MODEL" "$POLAR_HEADS" \
  "$POLAR_LAYERS" "$PREDICTOR_RUN_TAG" "${TRAIN_FILES[@]}" -- \
  "${VAL_FILES[@]}" -- "${TEST_FILES[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(output, model, data_root, save_dir, embedding, epochs, batch_size, lr,
 seed, max_paths, d_model, heads, layers, run_tag, *rest) = sys.argv[1:]
sep = rest.index("--")
train_files = rest[:sep]
rest = rest[sep + 1:]
sep = rest.index("--")
val_files, test_files = rest[:sep], rest[sep + 1:]

def describe(paths):
    result = []
    for name in paths:
        path = Path(name)
        data = json.loads(path.read_text(encoding="utf-8"))
        samples = data.get("samples", []) if isinstance(data, dict) else data
        result.append({"path": str(path), "samples": len(samples),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return result

config = {
    "stage": "predictor_training", "model_path": model, "data_root": data_root,
    "save_dir": save_dir, "embedding_model_name": embedding,
    "train_files": describe(train_files), "validation_files": describe(val_files),
    "held_out_test_files_not_used_for_training": describe(test_files),
    "training": {"difficulty_levels": [1, 2, 3, 4, 5], "num_epochs": int(epochs),
                 "batch_size": int(batch_size), "learning_rate": float(lr),
                 "seed": int(seed), "max_paths_per_sample": int(max_paths),
                 "per_sample_weight_normalize": True, "use_amp": True,
                 "max_total_examples": None, "max_validation_examples": 20000,
                 "optimizer": "AdamW", "optimizer_eps": 1e-8,
                 "weight_decay": 0.0, "lr_scheduler": "none",
                 "gradient_clip_norm": 0.0, "nan_guard": False,
                 "validation_selects_best_checkpoint": True, "resume_training": True,
                 "run_tag": run_tag},
    "predictor_architecture": {"d_model": int(d_model), "heads": int(heads),
                               "transformer_layers": int(layers),
                               "embedding_model_frozen": True},
}
Path(output).write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
echo "[pipeline] predictor training config: $PREDICTOR_CONFIG"
cat "$PREDICTOR_CONFIG"
HF_LOCAL_FILES_ONLY="$OFFLINE" HF_HUB_OFFLINE="$OFFLINE" \
TRANSFORMERS_OFFLINE="$OFFLINE" HF_DATASETS_OFFLINE="$OFFLINE" \
python3 "$ROOT/run_polar.py" \
  --model_path "$MODEL_PATH" --data_root "$DATA_ROOT" --save_dir "$SAVE_DIR" \
  --eval_all_diffs --train_data_jsons "${TRAIN_FILES[@]}" \
  --val_data_jsons "${VAL_FILES[@]}" --train_only \
  --num_epochs "$NUM_EPOCHS" --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --max_paths_per_sample "$MAX_PATHS_PER_SAMPLE" \
  --polar_d_model "$POLAR_D_MODEL" --polar_heads "$POLAR_HEADS" \
  --polar_layers "$POLAR_LAYERS" \
  --embedding_model_name "$EMBEDDING_MODEL_NAME" \
  --seed "$SEED" --run_tag "$PREDICTOR_RUN_TAG" \
  --per_sample_weight_normalize --use_amp --resume_training

touch "$RUN_DIR/pipeline_complete"

echo "[done] run_dir=$RUN_DIR"
echo "[done] held-out test merged files: ${TEST_FILES[*]}"
echo "[done] predictor save_dir=$SAVE_DIR"
