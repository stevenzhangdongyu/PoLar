#!/usr/bin/env python3
"""Build math selector data: Base + POLAR paths + correctness labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List

import torch

from llm_depth_router.model import get_model, get_tokenizer, setup_custom_path
from polar.config import infer_original_depth, set_random_seed
from polar.data import actions_to_path
from polar.eval import (
    _batch_compare_answers,
    _is_qwen15_moe_chat_model_path,
    _is_qwen25_instruct_model_path,
    _is_qwen3_model_path,
    _qwen15_moe_apply_chat_template,
    _qwen25_apply_chat_template,
    _qwen3_apply_chat_template,
    _qwen3_split_thinking,
)
from polar.model import PolarPredictor, decode_polar_to_actions
from qa_eval_utils import describe_path, write_json


def load_samples(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        payload = [json.loads(x) for x in f if x.strip()] if p.suffix == ".jsonl" else json.load(f)
    if isinstance(payload, dict) and "samples" in payload:
        payload = payload["samples"]
    elif isinstance(payload, dict):
        payload = list(payload.values())
    if not isinstance(payload, list):
        raise ValueError("Math input must be JSON/JSONL list or {'samples': [...]}.")
    rows = []
    for i, row in enumerate(payload):
        q = str(row.get("question") or row.get("query") or "").strip()
        a = str(row.get("gt_ans") or row.get("answer") or row.get("ground_truth") or "").strip()
        if q and a:
            rows.append({"task_id": str(row.get("task_id") or row.get("id") or f"math/{i}"),
                         "question": q, "gt_ans": a})
    if not rows:
        raise ValueError("No usable math samples found.")
    return rows


def build_input_text(question: str) -> str:
    return (
        "Solve the following math problem and output ONLY the final answer directly, "
        "formatted strictly as \\boxed{ANSWER}.\n"
        "### Problem Start\n"
        f"{question}\n"
        "### Problem End\n"
        "Answer:"
    )


def render_math_prompt(tokenizer, model_path: str, input_text: str) -> str:
    if _is_qwen3_model_path(model_path):
        return _qwen3_apply_chat_template(tokenizer, input_text)
    if _is_qwen15_moe_chat_model_path(model_path):
        return _qwen15_moe_apply_chat_template(tokenizer, input_text)
    if _is_qwen25_instruct_model_path(model_path):
        return _qwen25_apply_chat_template(tokenizer, input_text)
    return input_text


@torch.no_grad()
def generate_math(model, tokenizer, model_path: str, rendered_input: str,
                  path: List[int], gt: str, max_new_tokens: int) -> Dict[str, Any]:
    started = time.perf_counter()
    setup_custom_path(model, path)
    inputs = tokenizer([rendered_input], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id, use_cache=True)
    new_ids = output[0][len(inputs.input_ids[0]):].tolist()
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    if _is_qwen3_model_path(model_path):
        _thinking, answer_part = _qwen3_split_thinking(new_ids, tokenizer)
        answer_part = answer_part.strip()
    else:
        answer_part = raw.strip()
    follows_boxed = "oxed{" in answer_part
    exact = float(bool(follows_boxed and _batch_compare_answers([answer_part], [gt])[0]))
    return {
        **describe_path(path, len(model.model.layers)),
        "completion": raw,
        "answer_part": answer_part,
        "follows_boxed_format": follows_boxed,
        "exact_match": exact,
        "generation_seconds": time.perf_counter() - started,
        "generated_tokens": len(new_ids),
        "hit_token_limit": len(new_ids) >= max_new_tokens,
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--input_json", required=True)
    p.add_argument("--save_json", required=True)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=2048)
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
    p.add_argument("--resume_existing", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    set_random_seed(args.seed)
    samples = load_samples(args.input_json)
    end = len(samples) if args.num_samples <= 0 else args.start_idx + args.num_samples
    samples = samples[args.start_idx:end]
    if not samples:
        raise ValueError("Selected subset is empty.")
    out = Path(args.save_json)
    config = {k: v for k, v in vars(args).items() if k != "resume_existing"}
    config["tasks_sha256"] = hashlib.sha256(
        json.dumps(samples, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    results = []
    if out.exists():
        if not args.resume_existing:
            raise ValueError("Output exists; use --resume_existing or a new file.")
        payload = json.loads(out.read_text(encoding="utf-8"))
        if payload.get("configuration") != config:
            raise ValueError("Resume config/input changed.")
        results = payload.get("results", [])
    done = {r["task_id"] for r in results}

    def save(status):
        ordered = sorted(results, key=lambda x: x["dataset_index"])
        summary = {
            "completed_tasks": len(ordered),
            "base_correct": sum(int(r["base"]["exact_match"]) for r in ordered),
            "polar_topk_correct": sum(int(any(c["exact_match"] for c in r["candidates"])) for r in ordered),
            "configuration": config,
        }
        write_json(out, {"configuration": config, "status": status,
                         "summary": summary, "results": ordered})

    if done == {x["task_id"] for x in samples}:
        save("complete")
        print("[resume] already complete")
        return

    model = get_model(args.model_path, device="cuda").eval()
    tokenizer = get_tokenizer(args.model_path)
    depth = len(model.model.layers)
    if depth != infer_original_depth(args.model_path):
        print(f"[warn] inferred depth mismatch? model depth={depth}")
    predictor = PolarPredictor(
        num_layers=depth, embedding_model_name=args.embedding_model_name,
        d_model=args.polar_d_model, nheads=args.polar_heads,
        n_layer_blocks=args.polar_layers,
    ).cuda()
    predictor.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))
    predictor.eval()
    full_path = list(range(depth))

    for local_idx, sample in enumerate(samples):
        if sample["task_id"] in done:
            continue
        input_text = build_input_text(sample["question"])
        rendered = render_math_prompt(tokenizer, args.model_path, input_text)
        with torch.no_grad():
            seg, op = predictor([sample["question"]])
        beams = decode_polar_to_actions(seg[0], op[0], threshold=args.seg_threshold,
                                        max_pack=args.max_pack, beam_size=args.beam_size,
                                        top_k_ops=args.top_k_ops)
        base = generate_math(model, tokenizer, args.model_path, rendered, full_path,
                             sample["gt_ans"], args.max_new_tokens)
        candidates, seen = [], set()
        for actions, score in beams:
            path = actions_to_path(actions, depth)
            key = tuple(path)
            if not path or key in seen:
                continue
            seen.add(key)
            cand = generate_math(model, tokenizer, args.model_path, rendered, path,
                                 sample["gt_ans"], args.max_new_tokens)
            cand.update({"score": float(score), "segment_actions": [
                {"operation": op_name, "segment_length": size, "count": count}
                for op_name, size, count in actions]})
            candidates.append(cand)
            if len(candidates) >= args.top_k_paths:
                break
        results.append({
            **sample,
            "dataset_index": args.start_idx + local_idx,
            "predictor_input": sample["question"],
            "rendered_model_input": rendered,
            "segmentation_logits": seg[0].detach().float().cpu().tolist(),
            "operation_logits": op[0].detach().float().cpu().tolist(),
            "base": base,
            "candidates": candidates,
            "requested_k": args.top_k_paths,
            "actual_k": len(candidates),
        })
        done.add(sample["task_id"])
        save("partial")
        print(f"[{len(done)}/{len(samples)}] {sample['task_id']} "
              f"base={int(base['exact_match'])} "
              f"polar@{len(candidates)}={int(any(c['exact_match'] for c in candidates))}",
              flush=True)
    save("complete")


if __name__ == "__main__":
    main()
