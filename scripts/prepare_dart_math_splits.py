#!/usr/bin/env python3
"""Deduplicate DART-Math by difficulty, cap unique questions, and split them."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from export_unique_dart_math_by_diff import get_first_field, load_hf_dataset, read_json_or_jsonl


def split_counts(n, ratios):
    raw = [n * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    for idx in range(n - sum(counts)):
        counts[max(range(3), key=lambda j: raw[j] - counts[j])] += 1
    if n >= 3:
        for idx in range(3):
            if counts[idx] == 0:
                donor = max(range(3), key=lambda j: counts[j])
                counts[donor] -= 1
                counts[idx] = 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf_dataset", default=None)
    source.add_argument("--input_json", default=None)
    source.add_argument("--input_jsons_by_diff", nargs=5, metavar=("DIFF1", "DIFF2", "DIFF3", "DIFF4", "DIFF5"))
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_unique_per_diff", type=int, default=2000,
                        help="Maximum unique problems per difficulty; <=0 means all available.")
    for difficulty in range(1, 6):
        parser.add_argument(f"--max_unique_diff{difficulty}", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--question_field", default="query")
    parser.add_argument("--answer_field", default="gt_ans")
    parser.add_argument("--difficulty_field", default="query_metadata.level")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ratios = [args.train_ratio, args.test_ratio, args.val_ratio]
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("train/test/val ratios must be positive and sum to 1.")

    if args.hf_dataset:
        rows = load_hf_dataset(args.hf_dataset, args.hf_split)
        rows_by_diff = {d: rows for d in range(1, 6)}
        source_description = f"{args.hf_dataset}:{args.hf_split}"
    elif args.input_jsons_by_diff:
        rows_by_diff = {d: read_json_or_jsonl(args.input_jsons_by_diff[d - 1]) for d in range(1, 6)}
        source_description = {str(d): args.input_jsons_by_diff[d - 1] for d in range(1, 6)}
    else:
        rows = read_json_or_jsonl(args.input_json)
        rows_by_diff = {d: rows for d in range(1, 6)}
        source_description = args.input_json

    question_fields = [args.question_field, "question", "problem", "query", "sample_info.question"]
    answer_fields = [args.answer_field, "gt_ans", "ground_truth", "answer", "ref_ans", "sample_info.answer"]
    output = Path(args.output_dir)
    summary = {"source": source_description, "ratios": dict(zip(
        ("train", "test", "val"), ratios)), "seed": args.seed, "difficulties": {}}

    for difficulty in range(1, 6):
        unique, seen = [], set()
        for row in rows_by_diff[difficulty]:
            if not args.input_jsons_by_diff:
                try:
                    level = int(get_first_field(row, [args.difficulty_field]))
                except (TypeError, ValueError):
                    continue
                if level != difficulty:
                    continue
            question = get_first_field(row, question_fields)
            answer = get_first_field(row, answer_fields)
            if not question or not answer or question in seen:
                continue
            seen.add(question)
            unique.append({"question": question, "gt_ans": answer, "difficulty": difficulty})
        random.Random(args.seed + difficulty).shuffle(unique)
        level_cap = getattr(args, f"max_unique_diff{difficulty}")
        if level_cap is None:
            level_cap = args.max_unique_per_diff
        if level_cap > 0:
            unique = unique[:level_cap]
        counts = split_counts(len(unique), ratios)
        cursor = 0
        paths = {}
        for split, count in zip(("train", "test", "val"), counts):
            split_dir = output / f"diff{difficulty}"
            split_dir.mkdir(parents=True, exist_ok=True)
            path = split_dir / f"{split}.json"
            subset = unique[cursor:cursor + count]
            cursor += count
            path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
            paths[split] = {"count": len(subset), "path": str(path)}
        summary["difficulties"][str(difficulty)] = {"available_unique": len(seen),
                                                      "selected_unique": len(unique),
                                                      "splits": paths}
        print(f"[split] diff={difficulty} unique={len(unique)} "
              f"train/test/val={counts[0]}/{counts[1]}/{counts[2]}")
    (output / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] split manifest: {output / 'split_summary.json'}")


if __name__ == "__main__":
    main()
