#!/usr/bin/env python3
"""Real single-GPU Base / PoLar serving comparison, with fixed token budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def capture(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def prepare(args, tokenizer, depth):
    payload = json.loads(args.paths.read_text())
    paths = payload.get("paths") if isinstance(payload, dict) else payload
    if not paths:
        raise ValueError("Empty path workload")
    paths = [row["path"] if isinstance(row, dict) else row for row in paths]
    for path in paths:
        if not path or any(type(x) is not int or not 0 <= x < depth for x in path):
            raise ValueError(f"Invalid layer path: {path}")
    # Random token workload, not an accuracy dataset. Exclude special tokens.
    special = set(tokenizer.all_special_ids)
    vocabulary = [i for i in tokenizer.get_vocab().values() if i not in special]
    vocabulary = sorted(set(vocabulary))
    rng = random.Random(args.seed)
    assignments = [i % len(paths) for i in range(args.requests)]
    rng.shuffle(assignments)
    rows, entries = [], {}
    for index, assignment in enumerate(assignments):
        ids = rng.choices(vocabulary, k=args.input_len)
        key = digest(ids)
        if key in entries:
            raise ValueError("Prompt collision; change seed or increase input length")
        request_id = f"req-{index:06d}"
        row = {"request_id": request_id, "token_ids": ids,
               "source_path_index": assignment, "path": paths[assignment]}
        rows.append(row)
        entries[key] = {"request_id": request_id, "path": row["path"]}
    dump(args.output / "path_manifest.json", {
        "schema_version": 1, "request_count": len(rows), "entries": entries})
    with (args.output / "prompts.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return rows


def request_one(url, row, output_len):
    payload = {"model": "polar-qwen", "prompt": row["token_ids"],
               "temperature": 0, "max_tokens": output_len, "ignore_eos": True,
               "stream": True, "stream_options": {"include_usage": True}}
    request = urllib.request.Request(
        url + "/v1/completions", json.dumps(payload).encode(),
        {"Content-Type": "application/json"})
    started = time.perf_counter()
    result = {"request_id": row["request_id"], "error": None}
    text, arrivals, usage, finished, done = [], [], None, None, False
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"]))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice.get("text"):
                        text.append(choice["text"])
                        arrivals.append(time.perf_counter() - started)
                    if choice.get("finish_reason"):
                        finished = choice["finish_reason"]
        if not done or not usage or not arrivals:
            raise RuntimeError("Incomplete SSE stream, missing usage, or no text")
        if (usage["prompt_tokens"] != len(row["token_ids"])
                or usage["completion_tokens"] != output_len):
            raise RuntimeError(f"Token budget mismatch: {usage}")
        result.update(
            text="".join(text), finish_reason=finished,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            ttft_seconds=arrivals[0],
            tpot_seconds=((arrivals[-1] - arrivals[0]) / (output_len - 1)
                          if output_len > 1 else None),
            text_chunk_count=len(arrivals))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["latency_seconds"] = time.perf_counter() - started
    return result


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    return {"mean": statistics.mean(values), **{
        f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)]
        for p in (50, 95, 99)}}


def replay(args, rows, concurrency, directory):
    directory.mkdir()
    started = time.perf_counter()
    results = []
    with (directory / "results.jsonl").open("w") as stream:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(request_one, args.url, row, args.output_len)
                       for row in rows]
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
    wall = time.perf_counter() - started
    good = [row for row in results if row["error"] is None]
    prompts = sum(row["prompt_tokens"] for row in good)
    completions = sum(row["completion_tokens"] for row in good)
    summary = {
        "expected_requests": len(rows), "completed_requests": len(good),
        "failed_requests": len(rows) - len(good), "wall_seconds": wall,
        "concurrency": concurrency, "prompt_tokens": prompts,
        "completion_tokens": completions,
        "requests_per_second": len(good) / wall,
        "input_tokens_per_second": prompts / wall,
        "output_tokens_per_second": completions / wall,
        "total_tokens_per_second": (prompts + completions) / wall,
        **{field: distribution([r[field] for r in good if r[field] is not None])
           for field in ("ttft_seconds", "tpot_seconds", "latency_seconds")}}
    dump(directory / "summary.json", summary)
    if len(good) != len(rows):
        raise RuntimeError(f"Failed requests: inspect {directory / 'results.jsonl'}")
    return summary, {r["request_id"]: r["text"] for r in results}


def stop_server(process):
    # Only the process group created by this script is signalled.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run(args):
    from transformers import AutoConfig, AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    args.url = f"http://127.0.0.1:{args.port}"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    config = AutoConfig.from_pretrained(str(args.model), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    rows = prepare(args, tokenizer, config.num_hidden_layers)
    metadata = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    metadata.update(
        python=sys.executable, workload_note=(
            "Seeded random non-special token IDs; source paths repeated then shuffled. "
            "Synthetic scheduling stress test, not an accuracy evaluation."),
        metric_note=(
            "Client SSE TTFT = first nonempty text chunk; TPOT = "
            "(last-first text arrival)/(output tokens-1). Chunks may contain multiple "
            "tokens; no exact token-level ITL. Request rate is unlimited within C. "
            "Input/output throughput denominators are whole benchmark wall time."),
        comparison_note=(
            "Fresh Base uses native Qwen2; all modes V1 eager, prefix cache off, "
            "same explicit server budgets. Historical server settings were incomplete; "
            "historical numbers are not a controlled same-stack comparison. "
            "serial serializes layer subbatches within a forward, not full requests. "
            "Measured policies are not theoretical bounds."),
        input_sha256=hashlib.sha256((args.output / "prompts.jsonl").read_bytes()).hexdigest(),
        source_paths_sha256=hashlib.sha256(args.paths.read_bytes()).hexdigest(),
        vllm_git=capture(["git", "rev-parse", "HEAD"], args.vllm_root),
        vllm_dirty=capture(["git", "status", "--short"], args.vllm_root),
        packages=capture([sys.executable, "-m", "pip", "freeze"]),
        gpu_info=capture(["nvidia-smi"]),
        physical_depth=config.num_hidden_layers,
        max_path_executions=max(len(r["path"]) for r in rows))
    dump(args.output / "config.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    common = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
              "--model", str(args.model), "--served-model-name", "polar-qwen",
              "--host", "127.0.0.1", "--port", str(args.port),
              "--tensor-parallel-size", "1", "--pipeline-parallel-size", "1",
              "--enforce-eager", "--no-enable-prefix-caching",
              "--max-model-len", str(args.max_model_len),
              "--max-num-batched-tokens", str(args.max_num_batched_tokens),
              "--max-num-seqs", str(args.max_num_seqs),
              "--attention-backend", "FLASH_ATTN",
              "--gpu-memory-utilization", str(args.gpu_memory_utilization),
              "--seed", str(args.seed), "--dtype", "bfloat16"]
    summaries, references, checks = [], {}, []
    for repeat in range(args.repeats):
        # Rotate policy order across repeats to reduce systematic ordering effects.
        modes = ["base", "serial", "identical_path", "dynamic_layer"]
        modes = modes[repeat % 4:] + modes[:repeat % 4]
        for mode in modes:
            directory = args.output / f"repeat_{repeat}" / mode
            directory.mkdir(parents=True)
            env = os.environ.copy()
            for key in list(env):
                if key.startswith("VLLM_POLAR_"):
                    del env[key]
            env.update(CUDA_VISIBLE_DEVICES=args.gpu, VLLM_USE_V2_MODEL_RUNNER="0",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONPATH=str(args.vllm_root) + os.pathsep
                       + env.get("PYTHONPATH", ""))
            command = list(common)
            if mode != "base":
                env.update(VLLM_POLAR_SCHEDULER=mode,
                           VLLM_POLAR_PATH_MANIFEST=str(args.output / "path_manifest.json"))
                command += ["--hf-overrides", '{"architectures":["PolarQwen2ForCausalLM"]}']
            dump(directory / "launch.json", {"argv": command,
                 "environment": {k: v for k, v in env.items() if
                                 k.startswith(("VLLM_", "CUDA_", "HF_", "TRANSFORMERS_"))}})
            print(f"[start] repeat={repeat} mode={mode}", flush=True)
            with (directory / "server.log").open("w") as log:
                process = subprocess.Popen(command, cwd=args.vllm_root, env=env,
                                           stdout=log, stderr=log, start_new_session=True)
                try:
                    deadline = time.monotonic() + args.ready_timeout
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"Server exited: {directory / 'server.log'}")
                        try:
                            with urllib.request.urlopen(args.url + "/health", timeout=2):
                                break
                        except OSError:
                            if time.monotonic() > deadline:
                                raise RuntimeError(f"Readiness timeout: {directory}")
                            time.sleep(1)
                    for concurrency in args.concurrency:
                        print(f"[run] {mode} C={concurrency} warmup={args.warmups}", flush=True)
                        if args.warmups:
                            warm_rows = [rows[i % len(rows)] for i in range(args.warmups)]
                            replay(args, warm_rows, concurrency, directory / f"c{concurrency}_warmup")
                        summary, outputs = replay(args, rows, concurrency,
                                                  directory / f"c{concurrency}")
                        summary.update(mode=mode, repeat=repeat)
                        summaries.append(summary)
                        print(json.dumps(summary), flush=True)
                        # First completed policy is the per-family reference.
                        family = "base" if mode == "base" else "mixed"
                        if family not in references:
                            references[family] = outputs
                        different = [key for key in outputs
                                     if outputs[key] != references[family][key]]
                        checks.append({"mode": mode, "concurrency": concurrency,
                                       "repeat": repeat, "family": family,
                                       "different_text_count": len(different),
                                       "different_request_ids": different})
                        dump(args.output / "comparison.json", {
                            "runs": summaries, "output_consistency": checks})
                    server_text = (directory / "server.log").read_text()
                    if "Using V2 Model Runner" in server_text:
                        raise RuntimeError("V2 runner selected unexpectedly")
                    if mode != "base" and "PoLar heterogeneous path execution is experimental" not in server_text:
                        raise RuntimeError("PoLar hook not observed; throughput invalid")
                    if "Traceback (most recent call last)" in server_text:
                        raise RuntimeError(f"Server traceback: {directory / 'server.log'}")
                finally:
                    stop_server(process)
            write_report(args.output, summaries, checks)
    print(f"[done] {args.output / 'report.md'}", flush=True)


def write_report(root, summaries, checks):
    lines = ["# vLLM aligned real serving benchmark", "",
             "Configuration and limitations: see config.json. All latencies below are ms.", "",
             "|Repeat|Mode|C|Output tok/s|Total tok/s|TTFT mean|TTFT P99|TPOT mean|E2E P99|",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        tpot = r["tpot_seconds"]
        tpot_mean = f'{tpot["mean"] * 1000:.2f}' if tpot else "N/A"
        lines.append(
            f'|{r["repeat"]}|{r["mode"]}|{r["concurrency"]}|'
            f'{r["output_tokens_per_second"]:.2f}|{r["total_tokens_per_second"]:.2f}|'
            f'{r["ttft_seconds"]["mean"] * 1000:.2f}|'
            f'{r["ttft_seconds"]["p99"] * 1000:.2f}|{tpot_mean}|'
            f'{r["latency_seconds"]["p99"] * 1000:.2f}|')
    mismatches = sum(c["different_text_count"] for c in checks)
    lines += ["", f"Output consistency mismatches against family reference: {mismatches}.",
              "Inspect comparison.json for request IDs. Differences need investigation; "
              "matching text is not proof of correct KV semantics or accuracy.", "",
              "Base is a same-stack reference with different layer work, not an upper bound. "
              "The old 2026-11-12 benchmark used a different client and incompletely recorded "
              "server settings. Do not interpret its gap as PoLar scheduling overhead alone."]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vllm-root", "model", "paths", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--concurrency", default="1 2 4 8 16 32 64 96")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ready-timeout", type=int, default=600)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    args.concurrency = [int(c) for c in args.concurrency.replace(",", " ").split()]
    if (not args.concurrency or min(args.concurrency) < 1
            or len(set(args.concurrency)) != len(args.concurrency)
            or min(args.input_len, args.output_len, args.requests, args.repeats) < 1
            or args.warmups < 0):
        parser.error("Invalid lengths, counts, or concurrency list")
    if args.input_len + args.output_len > args.max_model_len:
        parser.error("Input + output exceeds --max-model-len")
    if max(args.concurrency) > args.max_num_seqs:
        parser.error("Concurrency exceeds --max-num-seqs")
    for name in ("vllm_root", "model", "paths", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if not (args.vllm_root / "vllm/model_executor/models/polar_qwen2.py").is_file():
        parser.error("--vllm-root must contain the modified PoLar vLLM source")
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(args)
    except KeyboardInterrupt:
        print("[stopped] Owned server stopped; completed results preserved.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
