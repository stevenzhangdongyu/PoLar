#!/usr/bin/env python3
"""Generic multiworker launcher for instruction/tool POLAR evaluators."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from polar_task_eval_utils import write_json
from eval_polar_instruction import load_tasks as load_instruction_tasks, summarize as summarize_instruction
from eval_polar_toolcall import load_tasks as load_tool_tasks, summarize as summarize_tool


TASKS = {
    "instruction": ("eval_polar_instruction.py", load_instruction_tasks, summarize_instruction),
    "tool": ("eval_polar_toolcall.py", load_tool_tasks, summarize_tool),
}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task_type", choices=TASKS, required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpus", default="0 1 2 3 4 5 6 7")
    p.add_argument("--workers_per_gpu", type=int, default=1)
    p.add_argument("--max_total_workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--polar_kv_cache_mode", choices=("layer", "execution"), default="layer")
    p.add_argument("--beam_size", type=int, default=10)
    p.add_argument("--top_k_ops", type=int, default=2)
    p.add_argument("--top_k_paths", type=int, default=5)
    p.add_argument("--seg_threshold", type=float, default=0.5)
    p.add_argument("--max_pack", type=int, default=4)
    p.add_argument("--embedding_model_name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--polar_d_model", type=int, default=256)
    p.add_argument("--polar_heads", type=int, default=4)
    p.add_argument("--polar_layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--merge_only", action="store_true")
    return p


def merge(out, manifest, summarize):
    rows_by_id, issues = {}, []
    for shard in manifest["shards"]:
        path = Path(out) / shard["result_file"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["results"]
            ids = [r["task_id"] for r in rows]
            if len(ids) != len(set(ids)) or not set(ids) <= set(shard["task_ids"]):
                raise ValueError("Unexpected/duplicate task IDs.")
            if set(ids) & set(rows_by_id):
                raise ValueError("Overlapping shard IDs.")
            rows_by_id.update({r["task_id"]: r for r in rows})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append({"rank": shard["rank"], "error": str(exc)})
    rows = [rows_by_id[i] for i in manifest["task_ids"] if i in rows_by_id]
    missing = [i for i in manifest["task_ids"] if i not in rows_by_id]
    summary = summarize(rows)
    summary.update({"expected_tasks": len(manifest["task_ids"]),
                    "complete": not missing and not issues,
                    "missing_task_ids": missing, "unreadable_workers": issues,
                    "configuration": manifest["configuration"]})
    write_json(Path(out) / "merged_results.json", {"summary": summary, "results": rows})
    write_json(Path(out) / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary["complete"]


def main():
    args = build_parser().parse_args()
    evaluator_name, load_tasks, summarize = TASKS[args.task_type]
    out = Path(args.output_dir).resolve()
    manifest_path = out / "manifest.json"
    if args.merge_only:
        if not merge(out, json.loads(manifest_path.read_text(encoding="utf-8")), summarize):
            sys.exit(1)
        return
    tasks = load_tasks(args.input_json)
    if args.limit > 0:
        tasks = tasks[:args.limit]
    gpus = args.gpus.replace(",", " ").split()
    if not gpus or len(set(gpus)) != len(gpus) or args.workers_per_gpu < 1:
        raise ValueError("Supply distinct GPU IDs and workers_per_gpu >= 1.")
    workers = min(len(tasks), len(gpus) * args.workers_per_gpu)
    if args.max_total_workers > 0:
        workers = min(workers, args.max_total_workers)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "input_tasks.json"
    write_json(snapshot, tasks)
    config = {k: v for k, v in vars(args).items() if k not in ("resume", "merge_only")}
    config["input_sha256"] = hashlib.sha256(
        json.dumps(tasks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    shards = []
    for rank in range(workers):
        start, end = rank * len(tasks) // workers, (rank + 1) * len(tasks) // workers
        gpu = gpus[rank % len(gpus)]
        shards.append({"rank": rank, "gpu": gpu, "start_idx": start, "num_samples": end - start,
                       "task_ids": [t["task_id"] for t in tasks[start:end]],
                       "result_file": f"shard_{rank:03d}_gpu_{gpu}.json"})
    manifest = {"configuration": config, "shards": shards, "task_ids": [t["task_id"] for t in tasks]}
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Output exists; use --resume or a new output_dir.")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume configuration changed.")
    write_json(manifest_path, manifest)
    evaluator = Path(__file__).with_name(evaluator_name).resolve()
    root = evaluator.parent.parent
    forwarded = ("model_path", "checkpoint_path", "max_new_tokens", "beam_size",
                 "top_k_ops", "top_k_paths", "seg_threshold", "max_pack",
                 "embedding_model_name", "polar_d_model", "polar_heads", "polar_layers", "seed")
    if args.task_type == "instruction":
        forwarded += ("polar_kv_cache_mode",)
    elif args.polar_kv_cache_mode != "layer":
        raise ValueError("polar_kv_cache_mode=execution is available for instruction tasks only.")
    running, failures = [], []
    print(f"[launch] {args.task_type} tasks={len(tasks)} workers={workers} output={out}", flush=True)
    try:
        for shard in shards:
            cmd = [sys.executable, "-u", str(evaluator), "--input_json", str(snapshot),
                   "--save_json", str(out / shard["result_file"]),
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
            proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=log)
            running.append((shard["rank"], proc, log))
            print(f"[launch] rank={shard['rank']} gpu={shard['gpu']} "
                  f"start={shard['start_idx']} count={shard['num_samples']} pid={proc.pid}", flush=True)
        for rank, proc, log in running:
            rc = proc.wait()
            log.close()
            if rc:
                failures.append({"rank": rank, "returncode": rc})
        complete = merge(out, manifest, summarize)
        if failures or not complete:
            print(f"[incomplete] failures={failures}; rerun with --resume", file=sys.stderr)
            sys.exit(1)
    finally:
        for _, proc, log in running:
            if proc.poll() is None:
                proc.terminate()
            log.close()


if __name__ == "__main__":
    main()
