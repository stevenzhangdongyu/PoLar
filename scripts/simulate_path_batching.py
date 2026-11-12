#!/usr/bin/env python3
"""Compare serial, exact-path, and dynamic-layer batching for path workloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polar.path_batch_scheduler import build_schedules, validate_paths


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def extract_paths(data: Any, field: str) -> list[list[int]]:
    if isinstance(data, list):
        if data and all(isinstance(item, int) for item in data):
            return [data]
        paths = []
        for row in data:
            if isinstance(row, list):
                paths.append(row)
            elif isinstance(row, dict):
                value = row.get(field)
                if isinstance(value, list) and value and all(type(x) is int for x in value):
                    paths.append(value)
                elif isinstance(value, list):
                    paths.extend(p for p in value if isinstance(p, list))
                elif isinstance(row.get("path"), list):
                    paths.append(row["path"])
        return paths
    if isinstance(data, dict):
        if "samples" in data:
            return extract_paths(data["samples"], field)
        for key in ("paths", "transitions", field):
            if key in data:
                return extract_paths(data[key], field)
        raise ValueError("JSON object must contain paths, transitions, or the selected path field")
    raise ValueError("input JSON must be an array or object")


def latency_lookup(cost_data: dict | None):
    if not cost_data:
        # Unit cost per layer batch: useful for comparing schedule shape only.
        return lambda batch_size: 1.0, "normalized_batch_units"
    raw = cost_data.get("batch_latency_ms", cost_data)
    points = sorted((int(k), float(v)) for k, v in raw.items())
    if not points or any(size < 1 or ms <= 0 for size, ms in points):
        raise ValueError("cost JSON must map positive batch sizes to positive milliseconds")

    def lookup(size: int) -> float:
        if size <= points[0][0]:
            return points[0][1]
        for (left_n, left_ms), (right_n, right_ms) in zip(points, points[1:]):
            if size <= right_n:
                ratio = (size - left_n) / (right_n - left_n)
                return left_ms + ratio * (right_ms - left_ms)
        if len(points) == 1:
            return points[0][1] * size / points[0][0]
        left_n, left_ms = points[-2]
        right_n, right_ms = points[-1]
        return right_ms + (size - right_n) * (right_ms - left_ms) / (right_n - left_n)

    return lookup, "estimated_ms_from_calibration"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths_json", type=Path, help="JSON array of paths, or object with a paths field")
    parser.add_argument("--path-field", default="final_valid_transitions")
    parser.add_argument("--depth", type=int, default=None, help="Optional model depth for range checking")
    parser.add_argument("--batch-cost-json", type=Path, help="Measured per-layer batch latency by batch size")
    parser.add_argument("--output", type=Path, help="Write full schedule and metrics JSON")
    args = parser.parse_args()

    raw_paths = extract_paths(load_json(args.paths_json), args.path_field)
    paths = validate_paths(raw_paths, args.depth)
    costs = load_json(args.batch_cost_json) if args.batch_cost_json else None
    latency, cost_units = latency_lookup(costs)
    schedules = build_schedules(paths)
    metrics = [schedule.metrics(latency) for schedule in schedules]
    serial_time = metrics[0]["estimated_duration_ms"]
    best_time = min(item["estimated_duration_ms"] for item in metrics)
    summary = {
        "request_count": len(paths),
        "unique_path_count": len(set(paths)),
        "total_layer_executions": sum(map(len, paths)),
        "cost_units": cost_units,
        "lower_bound_duration": best_time,
        "upper_bound_duration": serial_time,
        "duration_unit": "ms" if costs else "normalized batch units",
        "strategies": metrics,
    }

    print(f"[paths] requests={len(paths)} unique_paths={len(set(paths))} layer_executions={summary['total_layer_executions']}")
    for item in metrics:
        rate_label = "req/s" if costs else "req/normalized-unit"
        throughput = item["requests_per_cost_unit"] * (1000 if costs else 1)
        print(
            f"[{item['strategy']}] batches={item['layer_batch_invocations']} "
            f"mean_batch={item['mean_batch_size']:.2f} max_batch={item['max_batch_size']} "
            f"duration={item['estimated_duration_ms']:.3f} {summary['duration_unit']} "
            f"throughput={throughput:.3f} {rate_label}"
        )
    if costs:
        print(f"[bounds] best among tested strategies={best_time:.3f} ms, serial baseline={serial_time:.3f} ms")
    else:
        print("[bounds] normalized schedule-work bounds only; pass --batch-cost-json for calibrated time estimates")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[saved] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
