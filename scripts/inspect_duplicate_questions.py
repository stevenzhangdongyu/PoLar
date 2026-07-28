#!/usr/bin/env python3
"""
Inspect duplicate questions in a dataset or merged JSON.

Useful for understanding whether duplicate `query` values are exact duplicate
rows, or the same question paired with different answers/metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional


def get_nested(row: Dict[str, Any], field: str) -> Any:
    cur: Any = row
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def stable_json_hash(obj: Any) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def normalize_question(q: Any) -> str:
    return str(q or "").strip()


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
        raise ValueError(f"Unsupported JSON shape: {type(payload)}")
    return payload


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


def format_value(v: Any, max_chars: int = 240) -> str:
    if isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=False, sort_keys=True)
    else:
        s = str(v)
    s = s.replace("\n", "\\n")
    if len(s) > max_chars:
        return s[:max_chars] + "..."
    return s


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect duplicate questions.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf_dataset")
    src.add_argument("--input_json")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--question_field", default="query")
    parser.add_argument("--answer_field", default="gt_ans")
    parser.add_argument("--difficulty_field", default="query_metadata.level")
    parser.add_argument("--difficulty", type=int, choices=[1, 2, 3, 4, 5], default=None)
    parser.add_argument("--limit_rows", type=int, default=None, help="Only inspect the first N rows after difficulty filtering.")
    parser.add_argument("--top_groups", type=int, default=10, help="Number of duplicate question groups to print.")
    parser.add_argument("--examples_per_group", type=int, default=5)
    parser.add_argument(
        "--compare_fields",
        nargs="*",
        default=["gt_ans", "query_metadata", "source", "dataset", "type"],
        help="Extra fields to compare within duplicate groups.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.hf_dataset:
        rows = load_hf_dataset(args.hf_dataset, args.hf_split)
    else:
        rows = read_json_or_jsonl(args.input_json)

    raw_n = len(rows)
    if args.difficulty is not None:
        kept = []
        for row in rows:
            try:
                level = int(get_nested(row, args.difficulty_field))
            except (TypeError, ValueError):
                continue
            if level == args.difficulty:
                kept.append(row)
        rows = kept
        print(f"[filter] difficulty={args.difficulty}: {len(rows)}/{raw_n} rows kept")

    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    missing_q = 0
    for row in rows:
        q = normalize_question(get_nested(row, args.question_field))
        if not q:
            missing_q += 1
            continue
        groups[q].append(row)

    sizes = Counter(len(v) for v in groups.values())
    duplicate_groups = [(q, rs) for q, rs in groups.items() if len(rs) > 1]
    duplicate_rows = sum(len(rs) for _q, rs in duplicate_groups)
    exact_duplicate_groups = 0
    for _q, rs in duplicate_groups:
        if len({stable_json_hash(r) for r in rs}) == 1:
            exact_duplicate_groups += 1

    print("\n[summary]")
    print(f"rows inspected:          {len(rows)}")
    print(f"missing question rows:   {missing_q}")
    print(f"unique questions:        {len(groups)}")
    print(f"duplicate q groups:      {len(duplicate_groups)}")
    print(f"rows in duplicate groups:{duplicate_rows}")
    print(f"exact duplicate groups:  {exact_duplicate_groups}")
    print("group size histogram:")
    for size, count in sorted(sizes.items()):
        print(f"  size={size}: {count}")

    duplicate_groups.sort(key=lambda item: len(item[1]), reverse=True)
    print(f"\n[top {min(args.top_groups, len(duplicate_groups))} duplicate groups]")
    for gidx, (q, rs) in enumerate(duplicate_groups[: args.top_groups], start=1):
        row_hashes = [stable_json_hash(r) for r in rs]
        answer_values = [get_nested(r, args.answer_field) for r in rs]
        answer_set = {format_value(v, 160) for v in answer_values}
        print("\n" + "=" * 80)
        print(f"[group {gidx}] count={len(rs)} exact_row_hashes={len(set(row_hashes))}")
        print(f"question: {format_value(q, 600)}")
        print(f"answers unique={len(answer_set)}:")
        for ans in sorted(answer_set)[:10]:
            print(f"  - {ans}")

        for field in args.compare_fields:
            vals = [format_value(get_nested(r, field), 200) for r in rs]
            uniq = sorted(set(vals))
            print(f"field {field}: unique={len(uniq)}")
            for v in uniq[:5]:
                print(f"  - {v}")

        print(f"examples, first {min(args.examples_per_group, len(rs))}:")
        for ridx, row in enumerate(rs[: args.examples_per_group]):
            print(f"  [row {ridx}] hash={row_hashes[ridx]}")
            print(f"    answer={format_value(get_nested(row, args.answer_field), 200)}")
            print(f"    difficulty={format_value(get_nested(row, args.difficulty_field), 80)}")
            for field in args.compare_fields:
                if field in {args.answer_field, args.difficulty_field}:
                    continue
                print(f"    {field}={format_value(get_nested(row, field), 240)}")


if __name__ == "__main__":
    main()
