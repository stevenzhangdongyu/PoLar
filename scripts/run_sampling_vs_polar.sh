#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/run_sampling_vs_polar.sh [--limit N] qa|code model checkpoint input output
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
        echo "--limit requires a nonnegative integer (0 means all)." >&2
        exit 1
      fi
      export LIMIT="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: bash scripts/run_sampling_vs_polar.sh [--limit N] qa|code model checkpoint input output"
      echo "--limit N: first N tasks in total across all workers; 0 means all."
      echo "LIMIT environment variable is also supported; --limit takes precedence."
      exit 0
      ;;
    --*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
if [[ ${#POSITIONAL[@]} -ne 5 ]]; then
  echo "Expected: qa|code model checkpoint input output (optional --limit N)." >&2
  exit 1
fi
set -- "${POSITIONAL[@]}"
DOMAIN="${1:?qa or code required}"
shift
export TOP_K_PATHS="${K:-5}"
export BASE_SAMPLING_K="$TOP_K_PATHS"
export BEAM_SIZE="${BEAM_SIZE:-$TOP_K_PATHS}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.95}"
if (( BEAM_SIZE < TOP_K_PATHS )); then
  echo "BEAM_SIZE must be >= K (distinct path count may still be smaller)." >&2
  exit 1
fi
case "$DOMAIN" in
  qa) bash "$ROOT/scripts/run_qa_multiworker.sh" "$@" ;;
  code) bash "$ROOT/scripts/run_humaneval_multiworker.sh" "$@" ;;
  *) echo "Use qa or code." >&2; exit 1 ;;
esac
