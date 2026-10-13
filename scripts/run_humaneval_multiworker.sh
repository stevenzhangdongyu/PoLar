#!/usr/bin/env bash
set -euo pipefail

# Arguments: local_model checkpoint humaneval_json output_dir
# Run only in an isolated environment: generated Python is executed.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?local model path required}"
CHECKPOINT="${2:?predictor checkpoint required}"
INPUT="${3:?HumanEval JSON/JSONL required}"
OUTPUT="${4:?output directory required}"
ARGS=(
  --model_path "$MODEL"
  --checkpoint_path "$CHECKPOINT"
  --input_json "$INPUT"
  --output_dir "$OUTPUT"
  --gpus "${GPUS:-0 1 2 3 4 5 6 7}"
  --workers_per_gpu "${WORKERS_PER_GPU:-1}"
  --max_total_workers "${MAX_TOTAL_WORKERS:-0}"
  --limit "${LIMIT:-0}"
  --max_new_tokens "${MAX_NEW_TOKENS:-1024}"
  --beam_size "${BEAM_SIZE:-5}"
  --top_k_ops "${TOP_K_OPS:-2}"
  --top_k_paths "${TOP_K_PATHS:-5}"
  --test_timeout "${TEST_TIMEOUT:-10}"
  --embedding_model_name "${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
  --polar_d_model "${POLAR_D_MODEL:-256}"
  --polar_heads "${POLAR_HEADS:-4}"
  --polar_layers "${POLAR_LAYERS:-2}"
  --seg_threshold "${SEG_THRESHOLD:-0.5}"
  --max_pack "${MAX_PACK:-4}"
  --seed "${SEED:-42}"
)
if [[ "${RESUME:-0}" == "1" ]]; then
  ARGS+=(--resume)
fi
if [[ "${MERGE_ONLY:-0}" == "1" ]]; then
  ARGS+=(--merge_only)
elif [[ "${ALLOW_CODE_EXECUTION:-0}" == "1" ]]; then
  ARGS+=(--allow_code_execution)
else
  echo "HumanEval executes generated code. Use an isolated environment and set ALLOW_CODE_EXECUTION=1." >&2
  exit 1
fi
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python3 -u "${ROOT}/scripts/run_humaneval_multiworker.py" "${ARGS[@]}"
