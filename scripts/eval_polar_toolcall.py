#!/usr/bin/env python3
"""Offline POLAR tool/function-calling evaluation for BFCL-like JSON/JSONL data."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
import time
from typing import Any, Dict, List

from polar_task_eval_utils import read_json_or_jsonl, render_chat, path_description, mean, write_json


def normalize_functions(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return []
    funcs = []
    if isinstance(value, dict):
        value = value.get("functions") or value.get("tools") or value.get("function") or [value]
    if not isinstance(value, list):
        return []
    for f in value:
        if not isinstance(f, dict):
            continue
        fn = f.get("function") if isinstance(f.get("function"), dict) else f
        name = fn.get("name") or f.get("name")
        if name:
            funcs.append({"name": str(name), "raw": f})
    return funcs


def expected_calls(row):
    for key in ("possible_answer", "answers", "answer", "ground_truth", "gt_calls"):
        if key not in row:
            continue
        val = row[key]
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except Exception:
                try:
                    val = ast.literal_eval(val)
                except Exception:
                    continue
        calls = calls_from_obj(val)
        if calls:
            return calls
    return []


def calls_from_obj(obj):
    if isinstance(obj, dict):
        if "name" in obj or "arguments" in obj or "parameters" in obj:
            return [norm_call(obj)]
        for key in ("calls", "tool_calls", "function_calls", "answer"):
            if key in obj:
                return calls_from_obj(obj[key])
    if isinstance(obj, list):
        out = []
        for x in obj:
            out.extend(calls_from_obj(x))
        return out
    return []


def norm_call(obj):
    fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
    name = fn.get("name") or obj.get("name") or obj.get("function_name")
    args = fn.get("arguments", obj.get("arguments", obj.get("parameters", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            try:
                args = ast.literal_eval(args)
            except Exception:
                pass
    return {"name": str(name) if name is not None else "", "arguments": args if isinstance(args, dict) else {}}


def extract_calls(text):
    candidates = []
    # JSON fenced or plain JSON objects/lists.
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I):
        candidates.append(m.group(1))
    candidates.append(text)
    calls = []
    for cand in candidates:
        for snippet in json_snippets(cand):
            try:
                obj = json.loads(snippet)
            except Exception:
                try:
                    obj = ast.literal_eval(snippet)
                except Exception:
                    continue
            parsed = calls_from_obj(obj)
            if parsed:
                return parsed
    return calls


def json_snippets(text):
    text = text.strip()
    if text:
        yield text
    # Greedy but useful for generated prose wrapping a JSON object/list.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = text.find(open_c), text.rfind(close_c)
        if 0 <= s < e and not (s == 0 and e == len(text) - 1):
            yield text[s:e + 1]


def score_calls(pred, gold):
    if not gold:
        return {"has_gold": False, "name_exact": None, "args_exact": None,
                "call_exact": None, "parsed_calls": len(pred)}
    if len(pred) != len(gold):
        length_ok = False
    else:
        length_ok = True
    pred_names = [c["name"] for c in pred]
    gold_names = [c["name"] for c in gold]
    name_exact = pred_names == gold_names
    args_exact = name_exact and all(p.get("arguments", {}) == g.get("arguments", {})
                                   for p, g in zip(pred, gold))
    return {"has_gold": True, "name_exact": float(name_exact),
            "args_exact": float(args_exact), "call_exact": float(length_ok and args_exact),
            "parsed_calls": len(pred)}


def prompt_for_task(row):
    question = row.get("question") or row.get("prompt") or row.get("query") or row.get("user")
    if isinstance(question, list):
        question = "\n".join(str(x.get("content", x)) if isinstance(x, dict) else str(x) for x in question)
    funcs = normalize_functions(row.get("function") or row.get("functions") or row.get("tools"))
    tool_text = json.dumps([f["raw"] for f in funcs], ensure_ascii=False, indent=2)
    return (
        "You are a function calling model. Given the available functions and the user request, "
        "output ONLY a JSON list of function call objects. Each object must have keys "
        "`name` and `arguments`.\n\n"
        f"Available functions:\n{tool_text}\n\n"
        f"User request:\n{question}\n\n"
        "JSON function calls:"
    )


def load_tasks(path):
    rows, ids = [], set()
    for i, row in enumerate(read_json_or_jsonl(path)):
        prompt = prompt_for_task(row)
        task_id = str(row.get("id") or row.get("task_id") or row.get("idx") or f"tool/{i}")
        if task_id in ids:
            raise ValueError(f"Duplicate task_id: {task_id}")
        ids.add(task_id)
        rows.append({"task_id": task_id, "prompt": prompt, "functions": normalize_functions(
            row.get("function") or row.get("functions") or row.get("tools")),
            "expected_calls": expected_calls(row), "raw": row})
    if not rows:
        raise ValueError("No usable tool-calling tasks found.")
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
    p.add_argument("--max_new_tokens", type=int, default=512)
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
    return p


def summarize(rows):
    def avg_candidate(field, pick):
        vals = []
        for r in rows:
            c = pick(r)
            if c and c["score"].get(field) is not None:
                vals.append(c["score"][field])
        return mean(vals)
    return {
        "completed_tasks": len(rows),
        "metric_note": "Approximate built-in parser/checker; not official BFCL scoring.",
        "base_name_exact": avg_candidate("name_exact", lambda r: r["base"]),
        "base_call_exact": avg_candidate("call_exact", lambda r: r["base"]),
        "polar_top1_name_exact": avg_candidate("name_exact", lambda r: r["candidates"][0] if r["candidates"] else None),
        "polar_top1_call_exact": avg_candidate("call_exact", lambda r: r["candidates"][0] if r["candidates"] else None),
        "polar_topk_oracle_call_exact": mean([
            max((c["score"]["call_exact"] or 0.0 for c in r["candidates"]), default=0.0)
            for r in rows]),
    }


def main():
    args = build_parser().parse_args()
    tasks = load_tasks(args.input_json)
    end = len(tasks) if args.num_samples <= 0 else args.start_idx + args.num_samples
    tasks = tasks[args.start_idx:end]
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

    def generate(path, rendered, gold):
        started = time.perf_counter()
        setup_custom_path(model, path)
        inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                    use_cache=True, pad_token_id=tokenizer.pad_token_id)
        new_ids = output[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        calls = extract_calls(text)
        return {**path_description(path, depth), "completion": text,
                "parsed_calls": calls, "score": score_calls(calls, gold),
                "generated_tokens": len(new_ids), "generation_seconds": time.perf_counter() - started}

    for local_idx, task in enumerate(tasks):
        if task["task_id"] in done:
            continue
        rendered = render_chat(tokenizer, args.model_path, task["prompt"])
        with torch.no_grad():
            seg, op = predictor([task["prompt"]])
        beams = decode_polar_to_actions(seg[0], op[0], threshold=args.seg_threshold,
                                        max_pack=args.max_pack, beam_size=args.beam_size,
                                        top_k_ops=args.top_k_ops)
        base = generate(full_path, rendered, task["expected_calls"])
        candidates, seen = [], set()
        for actions, score in beams:
            path = actions_to_path(actions, depth)
            if not path or tuple(path) in seen:
                continue
            seen.add(tuple(path))
            cand = generate(path, rendered, task["expected_calls"])
            cand.update({"score_path": float(score), "segment_actions": [
                {"operation": op_name, "segment_length": size, "count": count}
                for op_name, size, count in actions]})
            candidates.append(cand)
            if len(candidates) >= args.top_k_paths:
                break
        results.append({**task, "dataset_index": args.start_idx + local_idx,
                        "predictor_input": task["prompt"], "rendered_model_input": rendered,
                        "segmentation_logits": seg[0].detach().float().cpu().tolist(),
                        "operation_logits": op[0].detach().float().cpu().tolist(),
                        "base": base, "candidates": candidates})
        done.add(task["task_id"])
        save("partial")
        print(f"[{len(done)}/{len(tasks)}] {task['task_id']} "
              f"base_call={base['score'].get('call_exact')} "
              f"polar_oracle={max((c['score'].get('call_exact') or 0 for c in candidates), default=0)}",
              flush=True)
    save("complete")


if __name__ == "__main__":
    main()
