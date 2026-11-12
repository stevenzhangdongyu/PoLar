#!/usr/bin/env python3
"""Create a deterministic synthetic workload for path-scheduler testing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def make_synthetic_workload(depth: int = 36, requests_per_group: int = 32, seed: int = 42):
    if depth < 16:
        raise ValueError("synthetic workload requires depth >= 16")
    rng = random.Random(seed)
    base = list(range(depth))
    rows = []

    def add(group: str, path: list[int]) -> None:
        rows.append({"request_id": f"req-{len(rows):03d}", "group": group, "path": path})

    # Cohort 1: exact duplicates, including full-depth and skip/loop templates.
    exact_templates = [
        base,
        [layer for layer in base if layer not in {8, 9, 10, 22, 23, 24}],
        base[:12] + [11] + base[12:],
        base[:25] + [24] + base[25:],
    ]
    for i in range(requests_per_group):
        add("exact_duplicate", exact_templates[i % len(exact_templates)])

    # Cohort 2: share a prefix and suffix while differing in the middle segment.
    prefix = list(range(8))
    suffix = list(range(depth - 8, depth))
    middle_pool = list(range(8, depth - 8))
    for i in range(requests_per_group):
        middle = middle_pool.copy()
        rng.shuffle(middle)
        middle = middle[: rng.randint(10, len(middle))]
        add("shared_prefix_suffix", prefix + middle + suffix)

    # Cohort 3: paths differ by one or two repeated visits at varied positions.
    for i in range(requests_per_group):
        path = base.copy()
        repeat_count = 1 + (i % 2)
        positions = sorted(rng.sample(range(2, depth - 2), repeat_count), reverse=True)
        for position in positions:
            path.insert(position, path[position])
        add("loop_variants", path)

    # Cohort 4: high-diversity stress paths with variable lengths and order.
    seen = set()
    while len(seen) < requests_per_group:
        length = rng.randint(max(8, depth // 2), depth)
        path = rng.sample(range(depth), length)
        path_key = tuple(path)
        if path_key not in seen:
            seen.add(path_key)
            add("high_diversity", path)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=36)
    parser.add_argument("--requests-per-group", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/path_batching/synthetic_paths_128.json"),
    )
    args = parser.parse_args()
    if args.requests_per_group < 1:
        parser.error("--requests-per-group must be positive")
    rows = make_synthetic_workload(args.depth, args.requests_per_group, args.seed)
    payload = {
        "metadata": {
            "source": "synthetic_scheduler_stress_test",
            "warning": "Not sampled from MCTS/predictor; paths are not correctness-validated.",
            "model_depth": args.depth,
            "seed": args.seed,
            "group_counts": {
                group: sum(row["group"] == group for row in rows)
                for group in sorted({row["group"] for row in rows})
            },
            "request_count": len(rows),
        },
        "paths": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {len(rows)} paths to {args.output}")
    print(f"[groups] {payload['metadata']['group_counts']}")
    print(f"[unique_paths] {len({tuple(row['path']) for row in rows})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
