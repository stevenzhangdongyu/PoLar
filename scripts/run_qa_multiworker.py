#!/usr/bin/env python3
"""Parallel local QA evaluation, with balanced GPU shards and auditable merging."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from qa_eval_utils import read_tasks, summarize, write_json
from eval_polar_qa import build_parser as worker_parser


def merge_results(out, manifest):
    rows_by_id, issues = {}, []
    for shard in manifest["shards"]:
        path = Path(out) / shard["result_file"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["results"]
            ids = [r["task_id"] for r in rows]
            if len(set(ids)) != len(ids) or not set(ids) <= set(shard["task_ids"]):
                raise ValueError("Unexpected or duplicate task IDs.")
            if set(ids) & set(rows_by_id):
                raise ValueError("Overlapping shards.")
            # Validate structure before including any rows from this file.
            summarize(rows)
            rows_by_id.update({r["task_id"]: r for r in rows})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append({"rank": shard["rank"], "error": str(exc)})
    rows = [rows_by_id[i] for i in manifest["task_ids"] if i in rows_by_id]
    missing = [i for i in manifest["task_ids"] if i not in rows_by_id]
    summary = summarize(rows)
    summary.update({
        "expected_tasks": len(manifest["task_ids"]),
        "complete": not missing and not issues,
        "missing_task_ids": missing, "unreadable_workers": issues,
        "metrics_denominator": "completed_tasks; partial runs are not full-test scores",
        "configuration": manifest["configuration"],
    })
    write_json(Path(out) / "merged_results.json", {"summary": summary, "results": rows})
    write_json(Path(out) / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary["complete"]


def build_parser():
    # Share model/decode/metric arguments so the launcher cannot silently omit one.
    p = worker_parser(multiworker=True)
    p.description = __doc__
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpus", default="0 1 2 3 4 5 6 7")
    p.add_argument("--workers_per_gpu", type=int, default=1)
    p.add_argument("--max_total_workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="<=0 means all tasks.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--merge_only", action="store_true")
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
        if not merge_results(out, manifest):
            sys.exit(1)
        return
    gpus = args.gpus.replace(",", " ").split()
    if not gpus or len(set(gpus)) != len(gpus) or args.workers_per_gpu < 1:
        raise ValueError("Supply distinct GPU IDs and workers_per_gpu >= 1.")
    tasks = read_tasks(args.input_json, args.mode)
    if args.limit > 0:
        tasks = tasks[:args.limit]
    workers = min(len(tasks), len(gpus) * args.workers_per_gpu)
    if args.max_total_workers > 0:
        workers = min(workers, args.max_total_workers)
    configuration = {k: v for k, v in vars(args).items() if k not in ("resume", "merge_only")}
    configuration["input_sha256"] = hashlib.sha256(
        json.dumps(tasks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    shards = []
    for rank in range(workers):
        start, end = rank * len(tasks) // workers, (rank + 1) * len(tasks) // workers
        gpu = gpus[rank % len(gpus)]
        shards.append({"rank": rank, "gpu": gpu, "start_idx": start,
                       "num_samples": end - start,
                       "task_ids": [t["task_id"] for t in tasks[start:end]],
                       "result_file": f"shard_{rank:03d}_gpu_{gpu}.json"})
    manifest = {"configuration": configuration, "shards": shards,
                "task_ids": [t["task_id"] for t in tasks]}
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Output exists; use --resume or a new output_dir.")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume input/configuration changed; retain original parameters.")
    elif args.resume:
        raise ValueError("No previous manifest in output_dir.")
    write_json(manifest_path, manifest)
    snapshot = out / "input_tasks.json"
    write_json(snapshot, tasks)
    evaluator = Path(__file__).with_name("eval_polar_qa.py").resolve()
    root = evaluator.parent.parent
    forwarded = ("model_path", "checkpoint_path", "mode", "predictor_input",
                 "max_new_tokens", "beam_size", "top_k_ops", "top_k_paths",
                 "seg_threshold", "max_pack", "embedding_model_name",
                 "polar_d_model", "polar_heads", "polar_layers", "seed",
                 "selection_strategy", "base_sampling_k", "temperature", "top_p")
    running, failures = [], []
    print(f"[launch] tasks={len(tasks)} workers={workers} output={out}", flush=True)
    try:
        for shard in shards:
            cmd = [sys.executable, "-u", str(evaluator),
                   "--input_json", str(snapshot), "--save_json", str(out / shard["result_file"]),
                   "--start_idx", str(shard["start_idx"]), "--num_samples", str(shard["num_samples"])]
            for key in forwarded:
                cmd += ["--" + key, str(getattr(args, key))]
            if args.resume:
                cmd += ["--resume_existing"]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = shard["gpu"]
            env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
            log = (out / shard["result_file"]).with_suffix(".log").open(
                "a" if args.resume else "w", encoding="utf-8")
            try:
                proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=log)
            except BaseException:
                log.close()
                raise
            running.append((shard["rank"], proc, log))
            print(f"[launch] rank={shard['rank']} gpu={shard['gpu']} "
                  f"start={shard['start_idx']} count={shard['num_samples']} pid={proc.pid}", flush=True)
        for rank, proc, log in running:
            rc = proc.wait()
            log.close()
            if rc:
                failures.append({"rank": rank, "returncode": rc})
        complete = merge_results(out, manifest)
        if failures or not complete:
            print(f"[incomplete] failed workers={failures}; rerun same command with --resume.",
                  file=sys.stderr)
            sys.exit(1)
    finally:
        for _, proc, log in running:
            if proc.poll() is None:
                proc.terminate()
            log.close()


if __name__ == "__main__":
    main()
