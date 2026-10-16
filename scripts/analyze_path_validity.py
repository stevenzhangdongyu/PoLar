#!/usr/bin/env python3
"""Analyze which POLAR/MCTS path structures are more often valid/correct."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_payload(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rows_from_file(path: str) -> Iterable[Dict[str, Any]]:
    payload = load_payload(path)
    if isinstance(payload, dict) and "results" in payload:
        for sample_idx, row in enumerate(payload["results"]):
            base = row.get("base", {})
            base_ok = bool(base.get("exact_match") or base.get("passed"))
            for rank, cand in enumerate(row.get("candidates", []) or [], 1):
                ok = bool(cand.get("exact_match") or cand.get("passed"))
                yield {"sample_idx": sample_idx, "source": "eval", "rank": rank,
                       "path": cand.get("path"), "score": cand.get("score"),
                       "label": ok, "base_ok": base_ok,
                       "task_id": row.get("task_id"), "sample": row, "candidate": cand}
    elif isinstance(payload, dict) and "samples" in payload:
        yield from rows_from_mcts_samples(payload["samples"])
    elif isinstance(payload, list):
        # Heuristic: eval result rows have "candidates"; MCTS samples have final_* transitions.
        if payload and isinstance(payload[0], dict) and "candidates" in payload[0]:
            for sample_idx, row in enumerate(payload):
                base = row.get("base", {})
                base_ok = bool(base.get("exact_match") or base.get("passed"))
                for rank, cand in enumerate(row.get("candidates", []) or [], 1):
                    ok = bool(cand.get("exact_match") or cand.get("passed"))
                    yield {"sample_idx": sample_idx, "source": "eval", "rank": rank,
                           "path": cand.get("path"), "score": cand.get("score"),
                           "label": ok, "base_ok": base_ok,
                           "task_id": row.get("task_id"), "sample": row, "candidate": cand}
        else:
            yield from rows_from_mcts_samples(payload)
    elif isinstance(payload, dict):
        yield from rows_from_mcts_samples(list(payload.values()))
    else:
        raise ValueError(f"Unsupported JSON shape: {type(payload)}")


def rows_from_mcts_samples(samples: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for sample_idx, sample in enumerate(samples):
        for label, key in ((True, "final_valid_transitions"), (False, "final_invalid_transitions")):
            for rank, path in enumerate(sample.get(key, []) or [], 1):
                yield {"sample_idx": sample_idx, "source": "mcts", "rank": rank,
                       "path": path, "score": None, "label": label, "base_ok": None,
                       "task_id": sample.get("task_id"), "sample": sample, "candidate": {}}


def valid_path(path: Any) -> List[int] | None:
    if not isinstance(path, list) or not path:
        return None
    out = []
    for x in path:
        if not isinstance(x, int) or x < 0:
            return None
        out.append(x)
    return out


def infer_depth(paths: Iterable[List[int]], fallback: int | None) -> int:
    if fallback:
        return fallback
    max_layer = max((max(p) for p in paths if p), default=-1)
    if max_layer < 0:
        raise ValueError("Cannot infer depth; pass --depth.")
    return max_layer + 1


def contiguous_runs(indices: List[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    runs, start, prev = [], indices[0], indices[0]
    for x in indices[1:]:
        if x == prev + 1:
            prev = x
        else:
            runs.append((start, prev - start + 1))
            start = prev = x
    runs.append((start, prev - start + 1))
    return runs


def path_features(path: List[int], depth: int) -> Dict[str, Any]:
    counts = Counter(path)
    base = list(range(depth))
    skipped = [i for i in range(depth) if counts[i] == 0]
    repeated = [i for i in range(depth) if counts[i] > 1]
    runs = contiguous_runs(repeated)
    skip_runs = contiguous_runs(skipped)
    nonmono = sum(1 for a, b in zip(path, path[1:]) if b < a)
    jumps = sum(1 for a, b in zip(path, path[1:]) if b != a + 1)
    extra = len(path) - depth
    repeat_count = sum(max(0, c - 1) for c in counts.values())
    max_visit = max(counts.values(), default=0)
    return {
        "is_base": path == base,
        "total_layer_executions": len(path),
        "unique_depth": len(counts),
        "length_ratio_bucket": bucket(len(path) / depth, [0.75, 0.9, 1.0, 1.05, 1.15, 1.3]),
        "extra_exec_bucket": bucket(extra, [-12, -6, -1, 0, 1, 4, 8]),
        "skip_count": len(skipped),
        "repeat_layer_count": len(repeated),
        "repeat_exec_count": repeat_count,
        "max_visit_count": max_visit,
        "nonmonotonic_edges": nonmono,
        "jump_edges": jumps,
        "has_skip": bool(skipped),
        "has_repeat": bool(repeated),
        "has_loop": repeat_count > 0 or nonmono > 0,
        "skip_run_count": len(skip_runs),
        "max_skip_run_len": max((n for _, n in skip_runs), default=0),
        "repeat_run_count": len(runs),
        "max_repeat_run_len": max((n for _, n in runs), default=0),
        "repeat_run_lens": [n for _, n in runs],
        "repeat_run_starts": [s for s, _ in runs],
        "repeated_layers": repeated,
        "skipped_layers": skipped,
    }


def bucket(value: float, edges: List[float]) -> str:
    for edge in edges:
        if value <= edge:
            return f"<= {edge:g}"
    return f"> {edges[-1]:g}"


def path_signature(path: List[int], depth: int) -> str:
    f = path_features(path, depth)
    if f["is_base"]:
        return "base"
    parts = []
    if f["has_skip"]:
        parts.append(f"skip{f['skip_count']}")
    if f["has_repeat"]:
        parts.append(f"repeat_layers{f['repeat_layer_count']}_exec{f['repeat_exec_count']}")
    if f["nonmonotonic_edges"]:
        parts.append(f"nonmono{f['nonmonotonic_edges']}")
    return "+".join(parts) or "other"


class Stats:
    def __init__(self) -> None:
        self.n = 0
        self.pos = 0
        self.sum_len = 0.0
        self.sum_unique = 0.0
        self.examples: List[Tuple[int, List[int]]] = []

    def add(self, row: Dict[str, Any], f: Dict[str, Any]) -> None:
        self.n += 1
        self.pos += int(row["label"])
        self.sum_len += f["total_layer_executions"]
        self.sum_unique += f["unique_depth"]
        if len(self.examples) < 3:
            self.examples.append((row["sample_idx"], row["path"]))

    def out(self, group: str, key: str) -> Dict[str, Any]:
        return {
            "group": group, "key": str(key), "n": self.n, "positive": self.pos,
            "rate": self.pos / self.n if self.n else None,
            "avg_total_layer_executions": self.sum_len / self.n if self.n else None,
            "avg_unique_depth": self.sum_unique / self.n if self.n else None,
            "examples": json.dumps(self.examples, ensure_ascii=False),
        }


def add_group(groups: Dict[Tuple[str, str], Stats], name: str, key: Any,
              row: Dict[str, Any], f: Dict[str, Any]) -> None:
    groups[(name, str(key))].add(row, f)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="append", required=True,
                   help="Eval merged_results.json or MCTS merged_mcts_samples.json; repeatable.")
    p.add_argument("--depth", type=int, default=0, help="Model layer count; inferred if omitted.")
    p.add_argument("--min_n", type=int, default=20, help="Minimum group size in printed tables.")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--out_prefix", default=None,
                   help="Write <prefix>_groups.csv, _paths.csv and _summary.json.")
    p.add_argument("--max_rows", type=int, default=0, help="Debug: cap number of path rows.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    raw_rows, paths = [], []
    for source in args.json:
        for row in rows_from_file(source):
            path = valid_path(row.get("path"))
            if path is None:
                continue
            row["path"] = path
            row["file"] = source
            raw_rows.append(row)
            paths.append(path)
            if args.max_rows and len(raw_rows) >= args.max_rows:
                break
    depth = infer_depth(paths, args.depth or None)
    groups: Dict[Tuple[str, str], Stats] = defaultdict(Stats)
    per_path = []
    per_sample = defaultdict(lambda: {"n": 0, "pos": 0})
    for row in raw_rows:
        f = path_features(row["path"], depth)
        sig = path_signature(row["path"], depth)
        per_sample[row["sample_idx"]]["n"] += 1
        per_sample[row["sample_idx"]]["pos"] += int(row["label"])
        per_path.append({k: v for k, v in {
            "file": row["file"], "source": row["source"], "sample_idx": row["sample_idx"],
            "task_id": row.get("task_id"), "rank": row["rank"], "score": row["score"],
            "label": int(row["label"]), "signature": sig,
            **{kk: vv for kk, vv in f.items() if not isinstance(vv, list)},
            "repeated_layers": " ".join(map(str, f["repeated_layers"])),
            "skipped_layers": " ".join(map(str, f["skipped_layers"])),
        }.items()})
        for name, key in [
            ("rank", row["rank"]), ("signature", sig), ("is_base", f["is_base"]),
            ("has_skip", f["has_skip"]), ("has_repeat", f["has_repeat"]),
            ("has_loop", f["has_loop"]), ("length_ratio_bucket", f["length_ratio_bucket"]),
            ("extra_exec_bucket", f["extra_exec_bucket"]),
            ("repeat_layer_count", f["repeat_layer_count"]),
            ("repeat_exec_count", f["repeat_exec_count"]),
            ("max_visit_count", f["max_visit_count"]),
            ("max_repeat_run_len", f["max_repeat_run_len"]),
            ("max_skip_run_len", f["max_skip_run_len"]),
            ("skip_count", f["skip_count"]),
            ("unique_depth", f["unique_depth"]),
        ]:
            add_group(groups, name, key, row, f)
        for layer in f["repeated_layers"]:
            add_group(groups, "repeated_layer", layer, row, f)
        for layer in f["skipped_layers"]:
            add_group(groups, "skipped_layer", layer, row, f)
        for n in f["repeat_run_lens"]:
            add_group(groups, "repeat_run_len", n, row, f)
        for s in f["repeat_run_starts"]:
            add_group(groups, "repeat_run_start", s, row, f)

    rows = [st.out(g, k) for (g, k), st in groups.items()]
    rows.sort(key=lambda r: (r["group"], -r["n"], str(r["key"])))
    overall_n = len(per_path)
    overall_pos = sum(r["label"] for r in per_path)
    sample_count = len(per_sample)
    samples_with_pos = sum(v["pos"] > 0 for v in per_sample.values())
    summary = {
        "files": args.json, "depth": depth, "path_rows": overall_n,
        "positive_paths": overall_pos,
        "path_positive_rate": overall_pos / overall_n if overall_n else None,
        "samples": sample_count,
        "samples_with_positive_path": samples_with_pos,
        "sample_positive_density": samples_with_pos / sample_count if sample_count else None,
        "avg_positive_paths_per_sample": (
            sum(v["pos"] for v in per_sample.values()) / sample_count if sample_count else None),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for group in ("rank", "signature", "length_ratio_bucket", "repeat_run_len",
                  "repeat_layer_count", "repeat_exec_count", "max_repeat_run_len",
                  "repeated_layer", "skipped_layer"):
        top = [r for r in rows if r["group"] == group and r["n"] >= args.min_n]
        top.sort(key=lambda r: (-r["rate"], -r["n"], r["key"]))
        print(f"\n[{group}] top by positive rate, n >= {args.min_n}")
        for r in top[: args.top]:
            print(f"  {r['key']}: n={r['n']} pos={r['positive']} "
                  f"rate={r['rate']:.4f} avg_len={r['avg_total_layer_executions']:.2f} "
                  f"avg_unique={r['avg_unique_depth']:.2f}")

    if args.out_prefix:
        prefix = Path(args.out_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        write_csv(prefix.with_name(prefix.name + "_groups.csv"), rows)
        write_csv(prefix.with_name(prefix.name + "_paths.csv"), per_path)
        prefix.with_name(prefix.name + "_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[wrote] {prefix}_groups.csv / _paths.csv / _summary.json")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
