#!/usr/bin/env python3
"""Offline instruction-following evaluation/generation for IFEval-style data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from polar_task_eval_utils import read_json_or_jsonl, render_chat, path_description, mean, write_json


def load_tasks(path):
    rows, ids = [], set()
    for i, row in enumerate(read_json_or_jsonl(path)):
        prompt = row.get("prompt") or row.get("instruction") or row.get("query")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        task_id = str(row.get("task_id") or row.get("key") or row.get("id") or f"instruction/{i}")
        if task_id in ids:
            raise ValueError(f"Duplicate task_id: {task_id}")
        ids.add(task_id)
        rows.append({"task_id": task_id, "prompt": prompt.strip(),
                     "instruction_id_list": row.get("instruction_id_list", []),
                     "kwargs": row.get("kwargs", []), "raw": row})
    if not rows:
        raise ValueError("No usable instruction prompts found.")
    return rows


def build_parser(*, multiworker=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--input_json", required=True)
    if not multiworker:
        p.add_argument("--save_json", required=True)
        p.add_argument("--start_idx", type=int, default=0)
        p.add_argument("--num_samples", type=int, default=0)
        p.add_argument("--resume_existing", action="store_true")
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
    p.add_argument("--predictor_input", choices=("prompt",), default="prompt")
    p.add_argument("--seed", type=int, default=42)
    return p


def summarize(rows):
    candidates = [c for r in rows for c in r.get("candidates", [])]
    top1 = [r["candidates"][0] for r in rows if r.get("candidates")]
    return {
        "completed_tasks": len(rows),
        "scored": False,
        "note": "Instruction following requires an external checker such as official IFEval. "
                "This file stores Base/POLAR outputs and IFEval-compatible prediction records.",
        "no_candidate_tasks": sum(not r.get("candidates") for r in rows),
        "polar_top1_mean_total_layer_executions": mean([c["total_layer_executions"] for c in top1]),
        "polar_top1_mean_unique_depth": mean([c["unique_depth"] for c in top1]),
        "polar_all_mean_total_layer_executions": mean([c["total_layer_executions"] for c in candidates]),
        "polar_all_mean_unique_depth": mean([c["unique_depth"] for c in candidates]),
    }


def main():
    args = build_parser().parse_args()
    tasks = load_tasks(args.input_json)
    end = len(tasks) if args.num_samples <= 0 else args.start_idx + args.num_samples
    tasks = tasks[args.start_idx:end]
    if not tasks:
        raise ValueError("Selected subset is empty.")
    config = {k: v for k, v in vars(args).items() if k != "resume_existing"}
    config["tasks_sha256"] = hashlib.sha256(
        json.dumps(tasks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    out = Path(args.save_json)
    results = []
    if out.exists():
        if not args.resume_existing:
            raise ValueError("Output exists; use --resume_existing or a new output.")
        payload = json.loads(out.read_text(encoding="utf-8"))
        if payload.get("configuration") != config:
            raise ValueError("Resume configuration changed.")
        results = payload.get("results", [])
    done = {r["task_id"] for r in results}

    def save(status):
        ordered = sorted(results, key=lambda r: r["dataset_index"])
        write_json(out, {"configuration": config, "status": status,
                         "summary": summarize(ordered), "results": ordered})
        write_ifeval_predictions(out.with_suffix(".ifeval_predictions.jsonl"), ordered)

    if done == {t["task_id"] for t in tasks}:
        save("complete")
        return

    import torch
    from llm_depth_router.model import get_model, get_tokenizer, setup_custom_path
    from polar.model import PolarPredictor, decode_polar_to_actions
    from polar.data import actions_to_path

    torch.manual_seed(args.seed)
    model = get_model(args.model_path, device="cuda").eval()
    tokenizer = get_tokenizer(args.model_path)
    depth = len(model.model.layers)
    full_path = list(range(depth))
    predictor = PolarPredictor(depth, args.embedding_model_name, args.polar_d_model,
                               args.polar_heads, args.polar_layers).cuda()
    predictor.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))
    predictor.eval()

    def generate(path, rendered):
        started = time.perf_counter()
        setup_custom_path(model, path, cache_mode=args.polar_kv_cache_mode)
        inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                    use_cache=True, pad_token_id=tokenizer.pad_token_id)
        new_ids = output[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return {**path_description(path, depth), "completion": text,
                "generated_tokens": len(new_ids),
                "hit_token_limit": len(new_ids) >= args.max_new_tokens,
                "generation_seconds": time.perf_counter() - started}

    for local_idx, task in enumerate(tasks):
        if task["task_id"] in done:
            continue
        rendered = render_chat(tokenizer, args.model_path, task["prompt"])
        with torch.no_grad():
            seg, op = predictor([task["prompt"]])
        beams = decode_polar_to_actions(seg[0], op[0], threshold=args.seg_threshold,
                                        max_pack=args.max_pack, beam_size=args.beam_size,
                                        top_k_ops=args.top_k_ops)
        base = generate(full_path, rendered)
        candidates, seen = [], set()
        for actions, score in beams:
            path = actions_to_path(actions, depth)
            if not path or tuple(path) in seen:
                continue
            seen.add(tuple(path))
            cand = generate(path, rendered)
            cand.update({"score": float(score), "segment_actions": [
                {"operation": op_name, "segment_length": size, "count": count}
                for op_name, size, count in actions]})
            candidates.append(cand)
            if len(candidates) >= args.top_k_paths:
                break
        results.append({**task, "dataset_index": args.start_idx + local_idx,
                        "predictor_input": task["prompt"],
                        "rendered_model_input": rendered,
                        "segmentation_logits": seg[0].detach().float().cpu().tolist(),
                        "operation_logits": op[0].detach().float().cpu().tolist(),
                        "base": base, "candidates": candidates})
        done.add(task["task_id"])
        save("partial")
        print(f"[{len(done)}/{len(tasks)}] {task['task_id']} candidates={len(candidates)}", flush=True)
    save("complete")


def write_ifeval_predictions(path: Path, rows):
    """Emit one JSONL record per method/candidate for official or custom checkers."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            common = {
                "key": row["task_id"], "prompt": row["prompt"],
                "instruction_id_list": row.get("instruction_id_list", []),
                "kwargs": row.get("kwargs", []),
            }
            f.write(json.dumps({**common, "method": "base", "candidate_rank": 0,
                                "response": row["base"]["completion"]}, ensure_ascii=False) + "\n")
            for i, cand in enumerate(row.get("candidates", []), 1):
                f.write(json.dumps({**common, "method": "polar", "candidate_rank": i,
                                    "response": cand["completion"]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
