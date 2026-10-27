#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_kv_cache_generalization.sh qa|code|instruction model checkpoint input output_root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOMAIN="${1:?qa, code, or instruction required}"
MODEL_PATH="${2:?model path required}"
CHECKPOINT_PATH="${3:?checkpoint path required}"
INPUT_JSON="${4:?input JSON/JSONL required}"
OUTPUT_ROOT="${5:?output root required}"

case "$DOMAIN" in
  qa) RUNNER="$ROOT/scripts/run_qa_multiworker.sh" ;;
  code) RUNNER="$ROOT/scripts/run_humaneval_multiworker.sh" ;;
  instruction) RUNNER="$ROOT/scripts/run_instruction_multiworker.sh" ;;
  *) echo "Domain must be qa, code, or instruction." >&2; exit 1 ;;
esac

for MODE in layer execution; do
  echo "[kv-generalization] domain=$DOMAIN mode=$MODE output=$OUTPUT_ROOT/$MODE"
  POLAR_KV_CACHE_MODE="$MODE" bash "$RUNNER" \
    "$MODEL_PATH" "$CHECKPOINT_PATH" "$INPUT_JSON" "$OUTPUT_ROOT/$MODE"
done

python3 "$ROOT/scripts/compare_kv_cache_eval.py" \
  --domain "$DOMAIN" \
  --layer "$OUTPUT_ROOT/layer/merged_results.json" \
  --execution "$OUTPUT_ROOT/execution/merged_results.json" \
  > "$OUTPUT_ROOT/kv_cache_comparison.json"
echo "[kv-generalization] comparison=$OUTPUT_ROOT/kv_cache_comparison.json"
