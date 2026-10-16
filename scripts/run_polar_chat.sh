#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/run_polar_chat.sh model checkpoint [new_session.json]
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?model path required}"
CHECKPOINT="${2:?checkpoint path required}"
SESSION="${3:-${ROOT}/outputs/chat/session_$(date +%Y%m%d_%H%M%S)_$$.json}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
ARGS=(
  --model_path "$MODEL" --checkpoint_path "$CHECKPOINT" --session_json "$SESSION"
  --embedding_model_name "${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
  --max_new_tokens "${MAX_NEW_TOKENS:-1024}"
  --top_k_paths "${K:-5}" --beam_size "${BEAM_SIZE:-10}"
  --top_k_ops "${TOP_K_OPS:-2}" --max_pack "${MAX_PACK:-4}"
  --seg_threshold "${SEG_THRESHOLD:-0.5}"
  --polar_d_model "${POLAR_D_MODEL:-256}" --polar_heads "${POLAR_HEADS:-4}"
  --polar_layers "${POLAR_LAYERS:-2}"
  --predictor_input "${PREDICTOR_INPUT:-question}"
  --selection_strategy "${SELECTION_STRATEGY:-nonbase_top1}"
  --system_prompt "${SYSTEM_PROMPT:-}"
  --seed "${SEED:-42}"
)
if [[ "${STREAM:-1}" == "0" ]]; then ARGS+=(--no_stream); fi
python3 -u "$ROOT/scripts/chat_polar.py" "${ARGS[@]}"
