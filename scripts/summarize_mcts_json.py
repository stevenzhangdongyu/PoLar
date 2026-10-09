#!/usr/bin/env python3
"""
Summarize valid/invalid path counts in an MCTS JSON file.

Works with either:
  - {"metadata": ..., "samples": [...]}
  - a plain list of sample dictionaries
  - a dict keyed by ids/questions whose values are sample dictionaries
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from typing import Any, Dict, List


def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "samples" in payload:
        payload = payload["samples"]
    elif isinstance(payload, dict):
        payload = list(payload.values())
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported JSON shape in {path}: {type(payload)}")
    return [x for x in payload if isinstance(x, dict)]


def get_question(sample: Dict[str, Any]) -> str:
    return str(
        sample.get("question")
        or sample.get("query")
        or sample.get("sample_info", {}).get("question")
        or ""
    ).strip()


def path_list(sample: Dict[str, Any], key: str) -> List[Any]:
    value = sample.get(key, []) or []
    return value if isinstance(value, list) else []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize MCTS valid path counts per question.")
    parser.add_argument("--json", required=True, help="Path to merged_mcts_samples.json or a shard JSON.")
    parser.add_argument("--top", type=int, default=20, help="Print top-N questions by valid path count.")
    parser.add_argument("--show_zero", type=int, default=20, help="Print up to N questions with zero valid paths.")
    parser.add_argument("--csv", default=None, help="Optional CSV output with one row per sample/question.")
    parser.add_argument("--question_preview_chars", type=int, default=120)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = load_samples(args.json)

    rows = []
    question_counter = Counter()
    for idx, sample in enumerate(samples):
        question = get_question(sample)
        question_counter[question] += 1
        valid = path_list(sample, "final_valid_transitions")
        invalid = path_list(sample, "final_invalid_transitions")
        initial_score = sample.get("initial_score")
        rows.append(
            {
                "idx": idx,
                "question": question,
                "valid_paths": len(valid),
                "invalid_paths": len(invalid),
                "total_paths": len(valid) + len(invalid),
                "initial_score": initial_score,
            }
        )

    n = len(rows)
    valid_counts = [r["valid_paths"] for r in rows]
    invalid_counts = [r["invalid_paths"] for r in rows]
    zero_valid = [r for r in rows if r["valid_paths"] == 0]
    nonzero_valid = [r for r in rows if r["valid_paths"] > 0]
    duplicate_questions = sum(1 for _q, c in question_counter.items() if c > 1)

    print(f"[file] {args.json}")
    print(f"[samples] {n}")
    print(f"[unique_questions] {len(question_counter)}")
    print(f"[duplicate_question_groups] {duplicate_questions}")
    if n:
        print(f"[valid_path_sum] {sum(valid_counts)}")
        print(f"[invalid_path_sum] {sum(invalid_counts)}")
        print(f"[questions_with_valid_path] {len(nonzero_valid)}/{n} = {len(nonzero_valid) / n:.4f}")
        print(f"[questions_with_zero_valid_path] {len(zero_valid)}/{n} = {len(zero_valid) / n:.4f}")
        print(f"[valid_paths_per_question] min={min(valid_counts)} max={max(valid_counts)} avg={sum(valid_counts) / n:.2f}")
        print(f"[invalid_paths_per_question] min={min(invalid_counts)} max={max(invalid_counts)} avg={sum(invalid_counts) / n:.2f}")

    hist = Counter(valid_counts)
    print("\n[histogram: valid_paths -> num_questions]")
    for valid_count in sorted(hist):
        print(f"  {valid_count}: {hist[valid_count]}")

    preview_chars = max(20, int(args.question_preview_chars))
    print(f"\n[top {args.top} by valid_paths]")
    for r in sorted(rows, key=lambda x: (-x["valid_paths"], x["idx"]))[: args.top]:
        q = r["question"].replace("\n", " ")[:preview_chars]
        print(
            f"  idx={r['idx']} valid={r['valid_paths']} invalid={r['invalid_paths']} "
            f"initial_score={r['initial_score']} question={q!r}"
        )

    if args.show_zero > 0:
        print(f"\n[first {args.show_zero} zero-valid questions]")
        for r in zero_valid[: args.show_zero]:
            q = r["question"].replace("\n", " ")[:preview_chars]
            print(
                f"  idx={r['idx']} invalid={r['invalid_paths']} "
                f"initial_score={r['initial_score']} question={q!r}"
            )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "idx",
                    "valid_paths",
                    "invalid_paths",
                    "total_paths",
                    "initial_score",
                    "question",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[csv] wrote {args.csv}")


if __name__ == "__main__":
    main()
