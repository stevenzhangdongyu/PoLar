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
import json
import re
import subprocess
import sys
import tempfile
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
) -> str:
    setup_custom_path(model, list(path))
    rendered = apply_chat_template(tokenizer, model_path, model_prompt(task))
    inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        use_cache=True,
    )
    new_ids = output_ids[0][len(inputs.input_ids[0]) :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


_PYTHON_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_code(text: str, entry_point: str) -> str:
    """Remove markdown/explanations while preserving a runnable function."""
    match = _PYTHON_FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
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
    return p


def main() -> None:
    args = build_parser().parse_args()
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
    results: List[Dict[str, Any]] = []
    base_correct = polar_top1_correct = polar_topk_correct = 0
    for idx, task in enumerate(tasks):
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
        seen = set()
        for actions, score in beams:
            path = actions_to_path(actions, depth)
            key = tuple(path)
            if key not in seen and path:
                seen.add(key)
                paths.append((path, float(score)))
        paths = paths[: max(1, args.top_k_paths)]

        base_text = generate_completion(
            model, tokenizer, args.model_path, full_path, task, args.max_new_tokens
        )
        base_ok, base_err = run_humaneval_test(task, base_text, args.test_timeout)

        candidates = []
        for path, score in paths:
            text = generate_completion(
                model, tokenizer, args.model_path, path, task, args.max_new_tokens
            )
            ok, err = run_humaneval_test(task, text, args.test_timeout)
            candidates.append(
                {
                    "path": path,
                    "score": score,
                    "unique_depth": len(set(path)),
                    "total_layer_executions": len(path),
                    "passed": ok,
                    "error": err,
                    "completion": text,
                }
            )

        top1_ok = bool(candidates and candidates[0]["passed"])
        topk_ok = any(x["passed"] for x in candidates)
        base_correct += int(base_ok)
        polar_top1_correct += int(top1_ok)
        polar_topk_correct += int(topk_ok)
        result = {
            "task_id": task.get("task_id", idx),
            "entry_point": task["entry_point"],
            "base": {"passed": base_ok, "error": base_err, "completion": base_text},
            "polar_top1": top1_ok,
            "polar_topk": topk_ok,
            "candidates": candidates,
        }
        results.append(result)
        print(
            f"[{idx + 1}/{len(tasks)}] {result['task_id']} "
            f"base={int(base_ok)} polar@1={int(top1_ok)} polar@{len(candidates)}={int(topk_ok)}"
        )

    n = len(tasks)
    summary = {
        "num_tasks": n,
        "base_accuracy": base_correct / max(1, n),
        "polar_top1_accuracy": polar_top1_correct / max(1, n),
        "polar_topk_accuracy": polar_topk_correct / max(1, n),
        "beam_size": args.beam_size,
        "top_k_ops": args.top_k_ops,
        "top_k_paths": args.top_k_paths,
        "max_new_tokens": args.max_new_tokens,
        "model_path": args.model_path,
        "checkpoint_path": args.checkpoint_path,
    }
    out = {"summary": summary, "results": results}
    Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {args.save_json}")


if __name__ == "__main__":
    main()
