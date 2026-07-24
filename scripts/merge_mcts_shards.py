#!/usr/bin/env python3
"""
Merge multiple MCTS shard JSON files produced by scripts/generate_mcts_samples.py.

Each shard can be either:
  {"metadata": ..., "samples": [...]}
or a plain list of sample dictionaries.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List


def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "samples" in payload:
        return payload["samples"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported JSON shape in {path}: {type(payload)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge MCTS shard JSON files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Shard files or glob patterns.")
    parser.add_argument("--output", required=True, help="Merged output JSON path.")
    parser.add_argument("--dedup_by_question", action="store_true", help="Drop duplicate questions, keeping the first occurrence.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_paths: List[str] = []
    for pattern in args.inputs:
        matches = sorted(glob.glob(pattern))
        input_paths.extend(matches if matches else [pattern])

    merged: List[Dict[str, Any]] = []
    seen_questions = set()
    for path in input_paths:
        if not os.path.exists(path):
            print(f"[skip] missing shard: {path}")
            continue
        samples = load_samples(path)
        kept = 0
        for sample in samples:
            question = str(sample.get("question", ""))
            if args.dedup_by_question and question in seen_questions:
                continue
            seen_questions.add(question)
            merged.append(sample)
            kept += 1
        print(f"[merge] {path}: loaded={len(samples)} kept={kept}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    payload = {
        "metadata": {
            "generator": "scripts/merge_mcts_shards.py",
            "num_shards": len(input_paths),
            "num_samples": len(merged),
        },
        "samples": merged,
    }
    tmp = f"{args.output}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, args.output)
    print(f"[done] wrote {len(merged)} samples to {args.output}")


if __name__ == "__main__":
    main()
