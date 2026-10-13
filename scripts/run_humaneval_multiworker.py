#!/usr/bin/env python3
"""Shard a local HumanEval JSON/JSONL, run GPU workers, merge task results.

The launcher itself needs only the Python standard library. Each worker loads
its own base LLM and predictor. Generated code runs in subprocesses, NOT a
security sandbox: run the entire job in an isolated environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def read_tasks(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with path.open(encoding="utf-8") as f:
            rows = json.load(f)
        if isinstance(rows, dict):
            rows = rows.get("tasks", rows.get("samples", list(rows.values())))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty task list.")
    ids = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or any(not row.get(k) for k in ("prompt", "test", "entry_point")):
            raise ValueError(f"Task {i} lacks prompt/test/entry_point.")
        row.setdefault("task_id", f"local/{i}")
        key = str(row["task_id"])
        if key in ids:
            raise ValueError(f"Duplicate task_id: {key}")
        ids.add(key)
    return rows


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def merge_results(output_dir, manifest):
    expected = {str(x) for x in manifest["task_ids"]}
    rows_by_id = {}
    incomplete_workers = []
    unreadable = []
    for shard in manifest["shards"]:
        path = Path(output_dir) / shard["result_file"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                incomplete_workers.append(shard["rank"])
            shard_ids = set(shard["task_ids"])
            for row in payload["results"]:
                key = str(row["task_id"])
                if key not in shard_ids:
                    raise ValueError(f"Unexpected task_id in {path}: {key}")
                if key in rows_by_id:
                    raise ValueError(f"Duplicate evaluated task: {key}")
                rows_by_id[key] = row
        except (OSError, ValueError, KeyError, TypeError) as exc:
            unreadable.append({"rank": shard["rank"], "error": str(exc)})
    rows = [rows_by_id[x] for x in manifest["task_ids"] if x in rows_by_id]
    missing = sorted(expected - set(rows_by_id))
    n = len(rows)
    base = sum(bool(x["base"]["passed"]) for x in rows)
    top1 = sum(bool(x["polar_top1"]) for x in rows)
    topk = sum(bool(x["polar_topk"]) for x in rows)
    summary = {
        "expected_tasks": len(expected), "completed_tasks": n,
        "complete": not missing and not unreadable,
        "missing_task_ids": missing, "incomplete_workers": incomplete_workers,
        "unreadable_workers": unreadable,
        "base_correct": base, "polar_top1_correct": top1, "polar_topk_correct": topk,
        "base_accuracy_on_completed": base / n if n else None,
        "polar_top1_accuracy_on_completed": top1 / n if n else None,
        "polar_topk_accuracy_on_completed": topk / n if n else None,
        "top_k_paths": manifest["configuration"]["top_k_paths"],
        "metric_note": "POLAR top-k = any candidate path passes; not unbiased stochastic pass@k.",
    }
    from candidate_selection import selection_summary, sampling_summary
    summary["selector"] = selection_summary(rows, lambda c: bool(c["passed"]))
    summary["sampling_comparison"] = sampling_summary(rows, lambda c: bool(c["passed"]))
    for label, candidates in (
        ("polar_top1", [x["candidates"][0] for x in rows if x["candidates"]]),
        ("polar_all_candidates", [c for x in rows for c in x["candidates"]]),
    ):
        summary[label + "_mean_layer_executions"] = (
            sum(c["total_layer_executions"] for c in candidates) / len(candidates)
            if candidates else None)
        summary[label + "_mean_unique_depth"] = (
            sum(c["unique_depth"] for c in candidates) / len(candidates)
            if candidates else None)
    write_json(Path(output_dir) / "merged_results.json", {"summary": summary, "results": rows})
    write_json(Path(output_dir) / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return not missing and not unreadable


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpus", default="0 1 2 3 4 5 6 7")
    p.add_argument("--workers_per_gpu", type=int, default=1)
    p.add_argument("--max_total_workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="Maximum tasks; <=0 means all.")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--top_k_ops", type=int, default=2)
    p.add_argument("--top_k_paths", type=int, default=5)
    p.add_argument("--seg_threshold", type=float, default=0.5)
    p.add_argument("--max_pack", type=int, default=4)
    p.add_argument("--test_timeout", type=float, default=10.0)
    p.add_argument("--embedding_model_name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--polar_d_model", type=int, default=256)
    p.add_argument("--polar_heads", type=int, default=4)
    p.add_argument("--polar_layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selection_strategy", choices=("top1", "nonbase_top1", "majority"), default="top1")
    p.add_argument("--base_sampling_k", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--resume", action="store_true", help="Same configuration/output directory only.")
    p.add_argument("--merge_only", action="store_true")
    p.add_argument("--allow_code_execution", action="store_true",
                   help="Acknowledge generated Python execution; use an isolated environment.")
    return p


def main():
    args = build_parser().parse_args()
    if args.base_sampling_k < 0 or (args.base_sampling_k and args.base_sampling_k != args.top_k_paths):
        raise ValueError("base_sampling_k must be 0 or equal top_k_paths.")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature > 0 and 0 < top_p <= 1 required.")
    out = Path(args.output_dir).resolve()
    manifest_path = out / "manifest.json"
    if args.merge_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        merge_results(out, manifest)
        return
    if not args.allow_code_execution:
        raise ValueError("Use --allow_code_execution only inside an isolated environment.")
    gpus = args.gpus.replace(",", " ").split()
    if not gpus or len(gpus) != len(set(gpus)) or args.workers_per_gpu < 1:
        raise ValueError("Supply distinct GPU IDs and workers_per_gpu >= 1.")
    tasks = read_tasks(args.input_json)
    if args.limit > 0:
        tasks = tasks[:args.limit]
    workers = min(len(tasks), len(gpus) * args.workers_per_gpu)
    if args.max_total_workers > 0:
        workers = min(workers, args.max_total_workers)
    configuration = {
        k: v for k, v in vars(args).items()
        if k not in ("resume", "merge_only", "allow_code_execution")
    }
    configuration["input_sha256"] = hashlib.sha256(
        json.dumps(tasks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    shards = []
    for rank in range(workers):
        start = rank * len(tasks) // workers
        end = (rank + 1) * len(tasks) // workers
        gpu = gpus[rank % len(gpus)]
        shards.append({"rank": rank, "gpu": gpu, "start_idx": start,
                       "num_samples": end - start,
                       "task_ids": [str(x["task_id"]) for x in tasks[start:end]],
                       "result_file": f"shard_{rank:03d}_gpu_{gpu}.json"})
    manifest = {"configuration": configuration,
                "task_ids": [str(x["task_id"]) for x in tasks], "shards": shards}
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Output already has a manifest. Use --resume or a fresh output_dir.")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Resume configuration/input changed. Use the original settings or a fresh output_dir.")
    elif args.resume:
        raise ValueError("No previous manifest exists in output_dir.")
    write_json(manifest_path, manifest)
    # Stable IDs for records without task_id; workers all read this same snapshot.
    input_snapshot = out / "input_tasks.json"
    write_json(input_snapshot, tasks)
    evaluator = Path(__file__).resolve().with_name("eval_polar_humaneval.py")
    root = evaluator.parent.parent
    forwarded = ("model_path", "checkpoint_path", "max_new_tokens", "beam_size",
                 "top_k_ops", "top_k_paths", "seg_threshold", "max_pack",
                 "test_timeout", "embedding_model_name", "polar_d_model",
                 "polar_heads", "polar_layers", "seed", "selection_strategy",
                 "base_sampling_k", "temperature", "top_p")
    running = []
    print(f"[launch] tasks={len(tasks)} workers={workers} output={out}", flush=True)
    try:
        for shard in shards:
            cmd = [sys.executable, "-u", str(evaluator),
                   "--input_json", str(input_snapshot),
                   "--start_idx", str(shard["start_idx"]),
                   "--num_samples", str(shard["num_samples"]),
                   "--save_json", str(out / shard["result_file"])]
            for key in forwarded:
                cmd += ["--" + key, str(getattr(args, key))]
            if args.resume:
                cmd += ["--resume_existing"]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = shard["gpu"]
            env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
            log = (out / shard["result_file"]).with_suffix(".log").open(
                "a" if args.resume else "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=log)
            running.append((shard["rank"], proc, log))
            print(f"[launch] rank={shard['rank']} gpu={shard['gpu']} "
                  f"start={shard['start_idx']} count={shard['num_samples']} pid={proc.pid}",
                  flush=True)
        failures = []
        for rank, proc, log in running:
            rc = proc.wait()
            log.close()
            if rc:
                failures.append({"rank": rank, "returncode": rc})
        complete = merge_results(out, manifest)
        if failures or not complete:
            print(f"[incomplete] failed workers={failures}; see shard logs. "
                  "Rerun with identical settings plus --resume.", file=sys.stderr)
            sys.exit(1)
    except BaseException:
        for _, proc, log in running:
            if proc.poll() is None:
                proc.terminate()
            log.close()
        raise


if __name__ == "__main__":
    main()
