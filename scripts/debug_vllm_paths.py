#!/usr/bin/env python3
"""Start a real PoLar server and replay each supplied path once for debugging."""

import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from vllm_aligned_benchmark import digest, dump, request_one, stop_server


def prepare_rows(payload, tokenizer, depth, default_prompt):
    if isinstance(payload, dict):
        payload = payload.get("requests", payload.get("paths"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Expected a nonempty array, or {'requests': [...]} / {'paths': [...]}")
    if all(type(value) is int for value in payload):
        payload = [payload]
    rows, entries, identifiers = [], {}, set()
    for index, value in enumerate(payload):
        value = {"path": value} if isinstance(value, list) else value
        if not isinstance(value, dict):
            raise ValueError(f"Request {index} must be a path array or object")
        path = value.get("path")
        if (not isinstance(path, list) or not path
                or any(type(layer) is not int or not 0 <= layer < depth for layer in path)):
            raise ValueError(f"Request {index}: path must contain layer indices 0..{depth-1}")
        identifier = str(value.get("request_id", f"debug-{index:04d}"))
        if identifier in identifiers:
            raise ValueError(f"Duplicate request_id: {identifier}")
        identifiers.add(identifier)
        # Routing currently uses exact prompt token hashes: make every prompt unique.
        question = value.get("prompt", default_prompt)
        messages = [{"role": "user", "content":
                     f"Request identifier: {identifier}.\n{question}"}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=True,
                                            return_dict=False)
        # Transformers versions differ in their chat-template return type.
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
            ids = ids[0]
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
            raise ValueError(f"{identifier}: tokenizer must return a nonempty integer token list")
        key = digest(ids)
        if key in entries:
            raise ValueError(f"{identifier}: tokenized prompt collides with "
                             f"{entries[key]['request_id']}; check tokenizer/template")
        row = {"request_id": identifier, "path": path, "messages": messages,
               "token_ids": ids}
        rows.append(row)
        entries[key] = {"request_id": identifier, "path": path}
    return rows, {"schema_version": 1, "request_count": len(rows), "entries": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path)
    parser.add_argument("--model", default="/home/z00894149/Qwen2.5-3B-Instruct")
    parser.add_argument("--vllm-root", type=Path,
                        default=Path("/home/z00894149/polar_dev/vllm"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--kv-cache-mode", choices=["layer", "execution"],
                        default="execution")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--strategy", choices=["serial", "identical_path", "dynamic_layer"],
                        default="dynamic_layer")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prompt", default="What is the capital of Peru? Answer briefly.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ready-timeout", type=int, default=600)
    parser.add_argument("--yes", action="store_true", help="Send immediately after readiness")
    args = parser.parse_args()
    if min(args.concurrency, args.max_tokens, args.ready_timeout) < 1:
        parser.error("Concurrency, max-tokens and ready-timeout must be positive")
    args.vllm_root = args.vllm_root.resolve()
    if not (args.vllm_root / "vllm/model_executor/models/polar_qwen2.py").is_file():
        parser.error("--vllm-root must contain the PoLar-adapted vLLM source")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows, manifest = prepare_rows(json.loads(args.paths.read_text()), tokenizer,
                                  config.num_hidden_layers, args.prompt)
    if max(len(row["token_ids"]) for row in rows) + args.max_tokens > args.max_model_len:
        parser.error("Prompt + generation exceeds --max-model-len")
    output = (args.output or Path("outputs") /
              f"debug_paths_{time.time_ns()}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "path_manifest.json", manifest)
    with (output / "prompts.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
            print(f'[request] {row["request_id"]} tokens={len(row["token_ids"])} '
                  f'path={row["path"]}', flush=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("VLLM_POLAR_"):
            del env[key]
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, VLLM_USE_V2_MODEL_RUNNER="0",
               VLLM_PLUGINS="", PYTHONUNBUFFERED="1",
               VLLM_POLAR_SCHEDULER=args.strategy,
               VLLM_POLAR_KV_CACHE_MODE=args.kv_cache_mode,
               VLLM_POLAR_PATH_MANIFEST=str(output / "path_manifest.json"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONPATH=str(args.vllm_root) + os.pathsep + env.get("PYTHONPATH", ""))
    command = [sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
               "--model", args.model, "--served-model-name", "polar-qwen",
               "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", "1", "--pipeline-parallel-size", "1",
               "--enforce-eager", "--no-enable-prefix-caching",
               "--attention-backend", "FLASH_ATTN", "--dtype", "bfloat16",
               "--max-model-len", str(args.max_model_len),
               "--max-num-seqs", str(max(8, args.concurrency)),
               "--hf-overrides", '{"architectures":["PolarQwen2ForCausalLM"]}']
    dump(output / "launch.json", {"args": {k: str(v) if isinstance(v, Path) else v
                                           for k, v in vars(args).items()},
                                 "argv": command,
                                 "environment": {k: v for k, v in env.items()
                                                 if k.startswith(("VLLM_", "CUDA_"))}})
    print(f"[start] output={output}\n[start] argv={command}", flush=True)
    process = subprocess.Popen(command, cwd=args.vllm_root, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True)

    def relay():
        with (output / "server.log").open("w") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

    thread = threading.Thread(target=relay, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{args.port}"
    results = []
    try:
        deadline = time.monotonic() + args.ready_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited; inspect {output / 'server.log'}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("Server readiness timeout")
                time.sleep(1)
        if not args.yes:
            input("[ready] Press Enter to send requests (Ctrl+C to stop): ")
        print(f"[send] requests={len(rows)} concurrency={args.concurrency}; no warmup", flush=True)
        with (output / "results.jsonl").open("w") as stream:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(request_one, url, row, args.max_tokens) for row in rows]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    print("[result] " + json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        stop_server(process)
        thread.join(timeout=5)
    print(f"[done] results={output / 'results.jsonl'}", flush=True)
    return int(any(row["error"] for row in results))


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[stopped] Debug server stopped.", flush=True)
        raise SystemExit(130)
