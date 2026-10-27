#!/usr/bin/env python3
"""
Evaluate a trained POLAR predictor on HumanEval-style code-generation tasks.

The script intentionally lives outside polar/eval.py because that module is
math-specific (boxed answers and DART-Math verification).

Supported input:
  1. Hugging Face dataset: --hf_dataset openai_humaneval
  2. Local JSON/JSONL containing records with:
       task_id, prompt, test, entry_point

For every task, the script reports:
  - Base greedy: full model path [0, 1, ..., D-1]
  - POLAR top-1: predictor's highest-scoring path
  - POLAR pass@k-like: at least one of top_k_paths passes HumanEval tests

Warning: generated code is executed in subprocesses. Use this only in a
trusted/isolated environment.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
import hashlib
from candidate_selection import select_candidate, selection_summary, sampling_summary
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from llm_depth_router.model import get_model, get_tokenizer, setup_custom_path
from polar.model import PolarPredictor, decode_polar_to_actions
from polar.data import actions_to_path, estimate_path_length_from_actions


def read_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        with p.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    with p.open() as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "tasks" in payload:
        payload = payload["tasks"]
    elif isinstance(payload, dict) and "samples" in payload:
        payload = payload["samples"]
    elif isinstance(payload, dict):
        payload = list(payload.values())
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported local dataset shape: {type(payload)}")
    return [x for x in payload if isinstance(x, dict)]


def load_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.input_json:
        tasks = read_records(args.input_json)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Install `datasets` or provide --input_json."
            ) from exc
        ds = load_dataset(
            args.hf_dataset,
            split=args.hf_split,
            download_mode="reuse_dataset_if_exists",
            verification_mode="no_checks",
        )
        tasks = [dict(x) for x in ds]

    required = ("prompt", "test", "entry_point")
    bad = [i for i, x in enumerate(tasks) if any(not x.get(k) for k in required)]
    if bad:
        raise ValueError(
            f"{len(bad)} records are missing one of {required}; first bad index={bad[0]}"
        )
    start = max(0, int(args.start_idx))
    end = len(tasks) if args.num_samples <= 0 else min(len(tasks), start + args.num_samples)
    return tasks[start:end]


def model_prompt(task: Dict[str, Any]) -> str:
    entry = task["entry_point"]
    return (
        "Complete the following Python function. Return ONLY the function "
        "implementation, with no markdown fences, explanation, or tests. "
        f"The function name is `{entry}`.\n\n"
        f"{task['prompt']}"
    )


def apply_chat_template(tokenizer, model_path: str, prompt: str) -> str:
    name = str(model_path).lower()
    if "qwen3" in name and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    if hasattr(tokenizer, "apply_chat_template") and (
        "qwen" in name or "llama" in name
    ):
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


@torch.no_grad()
def generate_completion(
    model,
    tokenizer,
    model_path: str,
    path: Sequence[int],
    task: Dict[str, Any],
    max_new_tokens: int,
    *, sample=False, temperature=0.7, top_p=0.95, kv_cache_mode="layer",
) -> str:
    setup_custom_path(model, list(path), cache_mode=kv_cache_mode)
    rendered = apply_chat_template(tokenizer, model_path, model_prompt(task))
    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=sample,
        use_cache=True,
        **({"temperature": temperature, "top_p": top_p} if sample else {}),
    )
    new_ids = output_ids[0][len(inputs.input_ids[0]) :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


_PYTHON_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_code(text: str, entry_point: str) -> str:
    """Remove markdown/explanations while preserving a runnable function."""
    match = _PYTHON_FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    # Preserve valid imports and helper functions before the entry point.
    try:
        ast.parse(text)
        return text.strip()
    except SyntaxError:
        pass
    # If the model echoed prose before the function, start at the definition.
    positions = [text.find(f"def {entry_point}("), text.find(f"async def {entry_point}(")]
    positions = [p for p in positions if p >= 0]
    if positions:
        text = text[min(positions) :]
    return text.strip()


def run_humaneval_test(
    task: Dict[str, Any],
    completion: str,
    timeout_s: float,
) -> Tuple[bool, str]:
    """Run the official HumanEval test body in a fresh Python subprocess."""
    code = extract_code(completion, str(task["entry_point"]))
    prompt = str(task["prompt"])
    # HumanEval prompts already contain the function signature. If the model
    # emits a complete `def`, keep imports/helper code before the reference
    # signature and replace the reference function with the completion.
    def_marker = f"def {task['entry_point']}("
    if def_marker in code or f"async def {task['entry_point']}(" in code:
        marker_pos = prompt.find(def_marker)
        prefix = prompt[:marker_pos] if marker_pos >= 0 else ""
        source = prefix + code
    else:
        source = prompt + "\n" + code
    program = (
        "import json\n"
        f"{source}\n"
        f"{task['test']}\n"
        f"check({task['entry_point']})\n"
    )
    with tempfile.TemporaryDirectory(prefix="polar_humaneval_") as td:
        path = Path(td) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=float(timeout_s),
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if proc.returncode == 0:
            return True, ""
        err = (proc.stderr or proc.stdout or "").strip().replace("\n", " ")
        return False, err[-1000:]


def describe_path(path: Sequence[int], depth: int) -> Dict[str, Any]:
    """Make the layer decisions human-readable for saved intermediate results."""
    visits = [int(x) for x in path]
    counts: Dict[int, int] = {}
    for layer in visits:
        counts[layer] = counts.get(layer, 0) + 1
    operations = []
    seen = set()
    for pos, layer in enumerate(visits):
        operations.append({
            "visit_index": pos,
            "layer": layer,
            "operation": "repeat" if layer in seen else "execute",
        })
        seen.add(layer)
    return {
        "operations": operations,
        "skipped_layers": [i for i in range(depth) if i not in counts],
        "repeated_layers": sorted([i for i, n in counts.items() if n > 1]),
        "layer_visit_counts": {str(i): n for i, n in sorted(counts.items())},
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate POLAR on HumanEval.")
    p.add_argument("--model_path", required=True)
    p.add_argument("--checkpoint_path", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_json")
    src.add_argument("--hf_dataset", default="openai_humaneval")
    p.add_argument("--hf_split", default="test")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=0, help="0 means all selected tasks")
    p.add_argument("--save_json", required=True)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--polar_kv_cache_mode", choices=("layer", "execution"), default="layer")
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
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--resume_existing", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.base_sampling_k < 0 or (args.base_sampling_k and args.base_sampling_k != args.top_k_paths):
        raise ValueError("base_sampling_k must be 0 or equal top_k_paths.")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature > 0 and 0 < top_p <= 1 required.")
    torch.manual_seed(args.seed)
    tasks = load_tasks(args)

    # Loading the base model also applies the repository's custom-path patch.
    model = get_model(args.model_path, device="cuda")
    tokenizer = get_tokenizer(args.model_path)
    model.eval()
    depth = len(model.model.layers)

    predictor = PolarPredictor(
        num_layers=depth,
        embedding_model_name=args.embedding_model_name,
        d_model=args.polar_d_model,
        nheads=args.polar_heads,
        n_layer_blocks=args.polar_layers,
    ).cuda()
    state = torch.load(args.checkpoint_path, map_location="cpu")
    predictor.load_state_dict(state)
    predictor.eval()

    full_path = list(range(depth))

    def clock():
        torch.cuda.synchronize()
        return time.perf_counter()

    def generate_timed(path, task, *, sample=False):
        start = clock()
        text = generate_completion(model, tokenizer, args.model_path, path, task,
                                   args.max_new_tokens, sample=sample,
                                   temperature=args.temperature, top_p=args.top_p,
                                   kv_cache_mode=args.polar_kv_cache_mode)
        return text, clock() - start

    results: List[Dict[str, Any]] = []
    completed_ids = set()
    if args.resume_existing and Path(args.save_json).is_file():
        try:
            previous = json.loads(Path(args.save_json).read_text(encoding="utf-8"))
            previous_mode = previous.get("summary", {}).get("polar_kv_cache_mode", "layer")
            if previous_mode != args.polar_kv_cache_mode:
                raise ValueError("Resume KV cache mode differs from existing results.")
            previous_results = previous.get("results", []) if isinstance(previous, dict) else []
            if isinstance(previous_results, list):
                results = [x for x in previous_results if isinstance(x, dict)]
                completed_ids = {str(x.get("task_id")) for x in results}
                print(f"[resume] loaded {len(results)} completed tasks")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[resume] ignoring unreadable partial output: {exc}", file=sys.stderr)

    def make_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        n_rows = len(rows)
        return {
            "num_tasks": n_rows,
            "base_accuracy": sum(bool(x.get("base", {}).get("passed")) for x in rows) / max(1, n_rows),
            "polar_top1_accuracy": sum(bool(x.get("polar_top1")) for x in rows) / max(1, n_rows),
            "polar_topk_accuracy": sum(bool(x.get("polar_topk")) for x in rows) / max(1, n_rows),
            "beam_size": args.beam_size,
            "top_k_ops": args.top_k_ops,
            "top_k_paths": args.top_k_paths,
            "max_new_tokens": args.max_new_tokens,
            "polar_kv_cache_mode": args.polar_kv_cache_mode,
            "model_path": args.model_path,
            "checkpoint_path": args.checkpoint_path,
            "start_idx": args.start_idx,
            "requested_num_samples": args.num_samples,
            "selector": selection_summary(rows, lambda c: bool(c["passed"])),
            "sampling_comparison": sampling_summary(rows, lambda c: bool(c["passed"])),
        }

    def save_partial(status: str = "partial") -> None:
        rows = sorted(results, key=lambda x: str(x.get("task_id")))
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps({"summary": make_summary(rows), "results": rows, "status": status},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(out_path)

    def trace(event: Dict[str, Any]) -> None:
        trace_path = Path(args.save_json).with_suffix(".trace.jsonl")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    base_correct = polar_top1_correct = polar_topk_correct = 0
    for idx, task in enumerate(tasks):
        task_key = str(task.get("task_id", args.start_idx + idx))
        if task_key in completed_ids:
            continue
        task_started = clock()
        torch.manual_seed(args.seed + int(hashlib.sha256(task_key.encode()).hexdigest()[:8], 16))
        question = model_prompt(task)
        with torch.no_grad():
            seg_logits, op_logits = predictor([question])
        beams = decode_polar_to_actions(
            seg_logits=seg_logits[0],
            op_logits=op_logits[0],
            threshold=args.seg_threshold,
            max_pack=args.max_pack,
            beam_size=args.beam_size,
            top_k_ops=args.top_k_ops,
        )
        paths: List[Tuple[List[int], float]] = []
        actions_by_path = {}
        seen = set()
        for actions, score in beams:
            path = actions_to_path(actions, depth)
            key = tuple(path)
            if key not in seen and path:
                seen.add(key)
                paths.append((path, float(score)))
                actions_by_path[key] = [
                    {"operation": op, "segment_length": size, "count": count}
                    for op, size, count in actions
                ]
        paths = paths[: max(1, args.top_k_paths)]
        predictor_seconds = clock() - task_started

        base_text, base_seconds = generate_timed(full_path, task)
        base_ok, base_err = run_humaneval_test(task, base_text, args.test_timeout)
        trace({"task_id": task_key, "stage": "base", "path": full_path,
               "completion": base_text, "passed": base_ok, "error": base_err})

        candidates = []
        for path, score in paths:
            text, generation_seconds = generate_timed(path, task)
            ok, err = run_humaneval_test(task, text, args.test_timeout)
            candidates.append(
                {
                    "path": path,
                    "generation_seconds": generation_seconds,
                    "score": score,
                    "segment_actions": actions_by_path[tuple(path)],
                    **describe_path(path, depth),
                    "unique_depth": len(set(path)),
                    "total_layer_executions": len(path),
                    "passed": ok,
                    "error": err,
                    "completion": text,
                    "extracted_code": extract_code(text, str(task["entry_point"])),
                }
            )
            trace({"task_id": task_key, "stage": "polar", "rank": len(candidates),
                   **candidates[-1]})

        base_record = {"passed": base_ok, "error": base_err, "completion": base_text,
                       "extracted_code": extract_code(base_text, str(task["entry_point"])),
                       "path": full_path, "generation_seconds": base_seconds}
        base_samples = []
        if args.base_sampling_k:
            for sample_idx in range(len(candidates)):
                text, elapsed = generate_timed(full_path, task, sample=True)
                ok, err = run_humaneval_test(task, text, args.test_timeout)
                draw = {"path": full_path, "completion": text, "passed": ok, "error": err,
                        "extracted_code": extract_code(text, str(task["entry_point"])),
                        "generation_seconds": elapsed, "do_sample": True}
                base_samples.append(draw)
                trace({"stage": "base_sampling", "task_id": task_key,
                       "rank": sample_idx + 1, **draw})
        selection_started = clock()
        selected, reason = select_candidate(base_record, candidates, args.selection_strategy)
        selection_seconds = clock() - selection_started
        polar_seconds = sum(c["generation_seconds"] for c in candidates)
        timings = {
            "base_greedy_seconds": base_seconds, "predictor_seconds": predictor_seconds,
            "polar_generation_seconds": polar_seconds,
            "base_sampling_seconds": sum(c["generation_seconds"] for c in base_samples),
            "selection_seconds": selection_seconds, "verifier_seconds": 0.0,
            "selected_pipeline_seconds": base_seconds + predictor_seconds
                + polar_seconds + selection_seconds,
            "end_to_end_seconds": clock() - task_started,
        }
        top1_ok = bool(candidates and candidates[0]["passed"])
        topk_ok = any(x["passed"] for x in candidates)
        base_correct += int(base_ok)
        polar_top1_correct += int(top1_ok)
        polar_topk_correct += int(topk_ok)
        result = {
            "task_id": task.get("task_id", args.start_idx + idx),
            "dataset_index": args.start_idx + idx,
            "entry_point": task["entry_point"],
            "original_prompt": task["prompt"],
            "predictor_input": question,
            "rendered_model_input": apply_chat_template(tokenizer, args.model_path, question),
            "segmentation_logits": seg_logits[0].detach().float().cpu().tolist(),
            "operation_logits": op_logits[0].detach().float().cpu().tolist(),
            "base": base_record, "base_samples": base_samples,
            "selected": selected, "selection_reason": reason,
            "requested_sampling_k": args.base_sampling_k,
            "selection_strategy": args.selection_strategy, "timings": timings,
            "polar_top1": top1_ok,
            "polar_topk": topk_ok,
            "candidates": candidates,
        }
        results.append(result)
        completed_ids.add(task_key)
        if len(results) % max(1, args.save_every) == 0:
            save_partial()
        print(
            f"[{idx + 1}/{len(tasks)}] {result['task_id']} "
            f"base={int(base_ok)} polar@1={int(top1_ok)} polar@{len(candidates)}={int(topk_ok)}"
        )

    summary = make_summary(results)
    save_partial("complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {args.save_json}")


if __name__ == "__main__":
    main()
