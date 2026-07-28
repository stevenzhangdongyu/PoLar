#!/usr/bin/env python3
"""
Export unique-question DART-Math subsets by difficulty.

This script is meant to prepare clean input JSON files before running MCTS data
generation. For each requested difficulty level, it:
  1. filters rows by query_metadata.level (configurable),
  2. deduplicates by question text before slicing,
  3. keeps at most --limit_per_diff unique questions,
  4. writes one JSON file per difficulty.

Output files contain a plain list of records:
  [{"question": "...", "gt_ans": "...", "difficulty": 1, "source_row": {...}}, ...]

These files can be used by scripts/generate_mcts_samples.py via --input_json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional


def get_nested(row: Dict[str, Any], field: str) -> Any:
    cur: Any = row
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def get_first_field(row: Dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        value = get_nested(row, name)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def load_hf_dataset(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    try:
        import datasets
    except ImportError as exc:
        raise ImportError("Install `datasets`, or use --input_json.") from exc
    ds = datasets.load_dataset(
        dataset_name,
        split=split,
        download_mode="reuse_dataset_if_exists",
        verification_mode="no_checks",
        trust_remote_code=False,
    )
    return list(ds)


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "samples" in payload:
        payload = payload["samples"]
    elif isinstance(payload, dict):
        payload = list(payload.values())
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported JSON shape from {path}: {type(payload)}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export unique DART-Math questions by difficulty.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf_dataset", help="HuggingFace dataset name, e.g. hkust-nlp/dart-math-pool-math")
    src.add_argument("--input_json", help="Local JSON/JSONL with DART-Math-like rows")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--difficulties", nargs="+", type=int, default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5])
    parser.add_argument("--limit_per_diff", type=int, default=2000)
    parser.add_argument("--question_field", default="query")
    parser.add_argument("--answer_field", default="gt_ans")
    parser.add_argument("--difficulty_field", default="query_metadata.level")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rows after filtering and before deduplication.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_source_row", action="store_true", help="Keep the full original row under source_row.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)

    if args.hf_dataset:
        rows = load_hf_dataset(args.hf_dataset, args.hf_split)
    else:
        rows = read_json_or_jsonl(args.input_json)

    os.makedirs(args.output_dir, exist_ok=True)
    summary: Dict[str, Any] = {
        "source": args.hf_dataset or args.input_json,
        "split": args.hf_split if args.hf_dataset else None,
        "question_field": args.question_field,
        "answer_field": args.answer_field,
        "difficulty_field": args.difficulty_field,
        "limit_per_diff": args.limit_per_diff,
        "difficulties": {},
    }

    q_names = [args.question_field, "question", "problem", "query", "sample_info.question"]
    a_names = [args.answer_field, "gt_ans", "ground_truth", "answer", "ref_ans", "sample_info.answer"]

    for diff in args.difficulties:
        filtered = []
        for row in rows:
            try:
                level = int(get_nested(row, args.difficulty_field))
            except (TypeError, ValueError):
                continue
            if level == int(diff):
                filtered.append(row)

        if args.shuffle:
            rng.shuffle(filtered)

        seen_questions = set()
        exported = []
        missing_question_or_answer = 0
        duplicate_rows = 0
        for row in filtered:
            question = get_first_field(row, q_names)
            answer = get_first_field(row, a_names)
            if not question or not answer:
                missing_question_or_answer += 1
                continue
            if question in seen_questions:
                duplicate_rows += 1
                continue
            seen_questions.add(question)

            item: Dict[str, Any] = {
                "question": question,
                "gt_ans": answer,
                "difficulty": int(diff),
            }
            if args.keep_source_row:
                item["source_row"] = row
            else:
                # Keep a compact copy of the most useful metadata when present.
                for field in ["query_metadata", "source", "dataset", "type"]:
                    value = get_nested(row, field)
                    if value is not None:
                        item[field] = value
            exported.append(item)
            if args.limit_per_diff > 0 and len(exported) >= args.limit_per_diff:
                break

        out_path = os.path.join(args.output_dir, f"dart_math_diff_{diff}_unique_limit{args.limit_per_diff}.json")
        if os.path.exists(out_path) and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {out_path}. Pass --overwrite to replace it.")
        with open(out_path, "w") as f:
            json.dump(exported, f, indent=2, ensure_ascii=False)

        summary["difficulties"][str(diff)] = {
            "filtered_rows": len(filtered),
            "unique_exported": len(exported),
            "duplicate_rows_skipped_before_limit": duplicate_rows,
            "missing_question_or_answer": missing_question_or_answer,
            "output": out_path,
        }
        print(
            f"[diff {diff}] filtered_rows={len(filtered)} "
            f"unique_exported={len(exported)} duplicates_skipped={duplicate_rows} "
            f"output={out_path}"
        )

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
