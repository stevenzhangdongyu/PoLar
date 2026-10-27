#!/usr/bin/env python3
"""Offline QA evaluation: frozen math-trained predictor, base vs POLAR paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from candidate_selection import select_candidate

from qa_eval_utils import (
    describe_path, extract_answer, model_prompt, read_tasks, score_answer,
    normalize_answer, summarize, write_json,
)


def build_parser(*, multiworker=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--input_json", required=True)
    if not multiworker:
        p.add_argument("--save_json", required=True)
    p.add_argument("--mode", choices=("short_answer", "multiple_choice"), default="short_answer")
    p.add_argument("--predictor_input", choices=("question", "prompt"), default="question")
    if not multiworker:
        p.add_argument("--start_idx", type=int, default=0)
        p.add_argument("--num_samples", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--polar_kv_cache_mode", choices=("layer", "execution"), default="layer")
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--top_k_ops", type=int, default=2)
    p.add_argument("--top_k_paths", type=int, default=5)
    p.add_argument("--seg_threshold", type=float, default=0.5)
    p.add_argument("--max_pack", type=int, default=4)
    p.add_argument("--embedding_model_name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--polar_d_model", type=int, default=256)
    p.add_argument("--polar_heads", type=int, default=4)
    p.add_argument("--polar_layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selection_strategy", choices=("top1", "nonbase_top1", "majority"), default="top1")
    p.add_argument("--base_sampling_k", type=int, default=0,
                   help="0 disables; positive k must equal top_k_paths; match actual distinct paths.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    if not multiworker:
        p.add_argument("--resume_existing", action="store_true")
    return p


def render_prompt(tokenizer, prompt, model_path):
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_path.lower():
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def main():
    args = build_parser().parse_args()
    if args.base_sampling_k < 0 or (args.base_sampling_k and args.base_sampling_k != args.top_k_paths):
        raise ValueError("base_sampling_k must be 0 or equal top_k_paths.")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature > 0 and 0 < top_p <= 1 required.")
    if args.start_idx < 0 or args.max_new_tokens < 1 or args.top_k_paths < 1:
        raise ValueError("Use start_idx >= 0, max_new_tokens/top_k_paths >= 1.")
    if args.beam_size < 1 or args.max_pack < 1 or not 1 <= args.top_k_ops <= 3:
        raise ValueError("beam_size/max_pack >= 1; top_k_ops in 1..3.")
    tasks = read_tasks(args.input_json, args.mode)
    end = len(tasks) if args.num_samples <= 0 else args.start_idx + args.num_samples
    tasks = tasks[args.start_idx:end]
    if not tasks:
        raise ValueError("Selected QA subset is empty.")
    configuration = {k: v for k, v in vars(args).items() if k != "resume_existing"}
    configuration["tasks_sha256"] = hashlib.sha256(
        json.dumps(tasks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    out_path = Path(args.save_json)
    results = []
    if out_path.exists():
        if not args.resume_existing:
            raise ValueError("Output exists; use --resume_existing or a new output.")
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        if previous.get("configuration") != configuration:
            raise ValueError("Resume parameters/input differ from previous output.")
        results = previous["results"]
    done = {r["task_id"] for r in results}
    expected = {t["task_id"] for t in tasks}
    if not done <= expected or len(done) != len(results):
        raise ValueError("Saved result IDs do not match this subset.")

    def save(status):
        ordered = sorted(results, key=lambda r: r["dataset_index"])
        write_json(out_path, {"configuration": configuration, "status": status,
                             "summary": summarize(ordered), "results": ordered})

    if done == expected:
        save("complete")
        print(f"[resume] all {len(tasks)} tasks already completed", flush=True)
        return

    # Import GPU dependencies only after validating inputs and resume metadata.
    import torch
    from llm_depth_router.model import get_model, get_tokenizer, setup_custom_path
    from polar.model import PolarPredictor, decode_polar_to_actions
    from polar.data import actions_to_path

    torch.manual_seed(args.seed)
    model = get_model(args.model_path, device="cuda").eval()
    tokenizer = get_tokenizer(args.model_path)
    depth = len(model.model.layers)
    predictor = PolarPredictor(
        num_layers=depth, embedding_model_name=args.embedding_model_name,
        d_model=args.polar_d_model, nheads=args.polar_heads,
        n_layer_blocks=args.polar_layers,
    ).cuda()
    predictor.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))
    predictor.eval()

    def trace(event):
        trace_path = out_path.with_suffix(".trace.jsonl")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def clock():
        torch.cuda.synchronize()
        return time.perf_counter()

    def generate(path, rendered, task, *, sample=False):
        started = clock()
        setup_custom_path(model, path, cache_mode=args.polar_kv_cache_mode)
        inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
        sampling = {"temperature": args.temperature, "top_p": args.top_p} if sample else {}
        with torch.no_grad():
            outputs = model.generate(
                **inputs, do_sample=sample, max_new_tokens=args.max_new_tokens,
                use_cache=True, pad_token_id=tokenizer.pad_token_id,
                **sampling,
            )
        new_ids = outputs[0][inputs.input_ids.shape[1]:]
        completion = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        answer = extract_answer(completion, args.mode, task["choices"])
        elapsed = clock() - started
        return {
            "generation_seconds": elapsed, "do_sample": sample,
            **describe_path(path, depth), "completion": completion,
            "extracted_answer": answer, "normalized_answer": answer if args.mode == "multiple_choice"
            else normalize_answer(answer),
            "generated_tokens": len(new_ids),
            "hit_token_limit": len(new_ids) >= args.max_new_tokens,
            **score_answer(answer, task["answers"], args.mode),
        }

    for idx, task in enumerate(tasks):
        if task["task_id"] in done:
            continue
        task_started = clock()
        # Stable per-task RNG regardless of sharding/resume; do not reseed each draw.
        task_seed = args.seed + int(hashlib.sha256(task["task_id"].encode()).hexdigest()[:8], 16)
        torch.manual_seed(task_seed)
        prompt = model_prompt(task, args.mode)
        predictor_input = task["question"] if args.predictor_input == "question" else prompt
        rendered = render_prompt(tokenizer, prompt, args.model_path)
        with torch.no_grad():
            seg_logits, op_logits = predictor([predictor_input])
        beams = decode_polar_to_actions(
            seg_logits[0], op_logits[0], threshold=args.seg_threshold,
            max_pack=args.max_pack, beam_size=args.beam_size, top_k_ops=args.top_k_ops)
        predictor_seconds = clock() - task_started
        trace({"stage": "start_task", **task, "predictor_input": predictor_input,
               "rendered_model_input": rendered,
               "segmentation_logits": seg_logits[0].detach().float().cpu().tolist(),
               "operation_logits": op_logits[0].detach().float().cpu().tolist()})
        base = generate(list(range(depth)), rendered, task)
        trace({"stage": "base", "task_id": task["task_id"], **base})
        candidates, seen = [], set()
        for actions, score in beams:
            path = actions_to_path(actions, depth)
            if not path or tuple(path) in seen:
                continue
            seen.add(tuple(path))
            segment_actions = [
                {"operation": op, "segment_length": size, "count": count}
                for op, size, count in actions]
            trace({"stage": "start_candidate", "task_id": task["task_id"],
                   "rank": len(candidates) + 1, "score": float(score),
                   "segment_actions": segment_actions, **describe_path(path, depth)})
            candidate = generate(path, rendered, task)
            candidate.update({
                "score": float(score),
                "segment_actions": segment_actions,
            })
            candidates.append(candidate)
            trace({"stage": "polar", "task_id": task["task_id"],
                   "rank": len(candidates), **candidate})
            if len(candidates) >= args.top_k_paths:
                break
        base_samples = []
        if args.base_sampling_k:
            for sample_idx in range(len(candidates)):
                draw = generate(list(range(depth)), rendered, task, sample=True)
                base_samples.append(draw)
                trace({"stage": "base_sampling", "task_id": task["task_id"],
                       "rank": sample_idx + 1, **draw})
        selection_started = clock()
        selected, reason = select_candidate(base, candidates, args.selection_strategy)
        selection_seconds = clock() - selection_started
        generation_seconds = sum(c["generation_seconds"] for c in candidates)
        timings = {
            "base_greedy_seconds": base["generation_seconds"],
            "predictor_seconds": predictor_seconds,
            "polar_generation_seconds": generation_seconds,
            "base_sampling_seconds": sum(c["generation_seconds"] for c in base_samples),
            "selection_seconds": selection_seconds,
            "verifier_seconds": 0.0,
            "selected_pipeline_seconds": base["generation_seconds"] + predictor_seconds
                + generation_seconds + selection_seconds,
            "end_to_end_seconds": clock() - task_started,
        }
        results.append({
            **task, "dataset_index": args.start_idx + idx,
            "predictor_input": predictor_input, "rendered_model_input": rendered,
            "segmentation_logits": seg_logits[0].detach().float().cpu().tolist(),
            "operation_logits": op_logits[0].detach().float().cpu().tolist(),
            "base": base, "candidates": candidates,
            "base_samples": base_samples, "selected": selected,
            "requested_sampling_k": args.base_sampling_k,
            "selection_strategy": args.selection_strategy, "selection_reason": reason,
            "timings": timings,
        })
        done.add(task["task_id"])
        save("partial")
        print(f"[{len(done)}/{len(tasks)}] {task['task_id']} "
              f"base_EM={base['exact_match']:.0f} "
              f"polar@1_EM={candidates[0]['exact_match'] if candidates else 0:.0f} "
              f"polar@{len(candidates)}_EM="
              f"{max((c['exact_match'] for c in candidates), default=0):.0f}", flush=True)
    save("complete")
    print(json.dumps(summarize(results), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
