#!/usr/bin/env python3
"""Locate failed or incomplete MCTS JSON shards from worker logs.

Examples:
  python scripts/diagnose_mcts_shards.py ./outputs/.../diff4/train
  python scripts/diagnose_mcts_shards.py ./outputs/.../diff4/train --input-json data.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SHARD_RE = re.compile(
    r"^shard_(?P<rank>\d+)_gpu_(?P<gpu>[^_]+)_worker_(?P<worker>\d+)"
    r"(?:_start_(?P<start>\d+)_limit_(?P<limit>\d+))?"
)
ERROR_RE = re.compile(
    r"(Traceback \(most recent call last\)|CUDA out of memory|out of memory|"
    r"RuntimeError:|ValueError:|AssertionError:|Killed|SIGTERM|SIGKILL|"
    r"NCCL|device-side assert|No usable samples found|ERROR|Error)",
    re.IGNORECASE,
)


def shard_info(path: Path) -> dict:
    match = SHARD_RE.match(path.name)
    return match.groupdict() if match else {}


def json_count(path: Path) -> tuple[int | None, str | None]:
    if not path.is_file():
        return None, "missing output JSON"
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        rows = data.get("samples", []) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            return None, "JSON has no samples list"
        return len(rows), None
    except Exception as exc:  # diagnostic tool must report malformed partial files
        return None, f"unreadable JSON: {type(exc).__name__}: {exc}"


def log_errors(path: Path, tail_lines: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = [line.strip() for line in lines if ERROR_RE.search(line)]
    return hits[-tail_lines:]


def expected_questions(input_json: Path, start: int, limit: int, field: str) -> set[str]:
    with input_json.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, dict):
        data = data.get("samples", list(data.values()))
    return {
        str(row.get(field, row.get("question", ""))).strip()
        for row in data[start : start + limit]
        if isinstance(row, dict)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--tail-lines", type=int, default=8)
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        parser.error(f"output directory does not exist: {args.output_dir}")
    logs = sorted(args.output_dir.glob("shard_*.log"))
    logs += sorted(args.output_dir.glob("shard_*.resume.log"))
    logs = sorted(set(logs))
    if not logs:
        print(f"[error] no shard logs found under {args.output_dir}")
        return 2

    print(f"[scan] directory={args.output_dir}")
    print(f"[scan] logs={len(logs)}")
    failures = 0
    incomplete = 0
    for log in logs:
        info = shard_info(log)
        # The launcher uses the same stem for the output JSON, regardless of
        # whether the log is .log or .resume.log.
        output = args.output_dir / (log.name.removesuffix(".resume.log").removesuffix(".log") + ".json")
        count, json_error = json_count(output)
        errors = log_errors(log, args.tail_lines)
        start = int(info["start"]) if info.get("start") else None
        limit = int(info["limit"]) if info.get("limit") else None
        expected = None
        if args.input_json and start is not None and limit is not None:
            expected = len(expected_questions(args.input_json, start, limit, args.question_field))
        complete = count is not None and (expected is None or count >= expected)
        if not complete or errors:
            failures += 1 if errors else 0
            incomplete += 1 if not complete else 0
            label = "FAILED" if errors else "INCOMPLETE"
            location = (
                f"rank={info.get('rank','?')} gpu={info.get('gpu','?')} "
                f"worker={info.get('worker','?')}"
            )
            print(f"\n[{label}] {location} log={log.name}")
            print(f"  output={output.name} samples={count if count is not None else 'N/A'}"
                  + (f" expected={expected}" if expected is not None else ""))
            if json_error:
                print(f"  {json_error}")
            if errors:
                print("  log evidence:")
                for line in errors:
                    print(f"    {line}")

    print(f"\n[summary] failed_logs={failures} incomplete_shards={incomplete} scanned_logs={len(logs)}")
    if failures or incomplete:
        print("[next] Rerun only these ranges with the resume launcher; existing complete JSON shards can be kept.")
        return 1
    print("[ok] all discovered shard logs have a matching complete JSON output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
