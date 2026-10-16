#!/usr/bin/env python3
"""Interactive, streaming POLAR top-k chat; optional reference/manual best-of-k grading."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from candidate_selection import select_candidate
from polar_chat_utils import (
    chat_prompt, predictor_text, unique_paths, grade_reference, grade_manual, chat_summary,
)
from qa_eval_utils import describe_path, write_json


HELP = """
直接输入问题：Base + 最多 K 个 POLAR greedy 回复（token 实时输出）。
/multi                    多行输入，以单独一行 /end 结束
/use base 或 /use 1..K     切换最近一轮写入共同历史的回复
/reference ["答案","别名"] 给最近一轮做短答案 EM/F1 判定，不影响生成
/grade [true,false,...]    手工判定最近一轮的 K 个候选；开放问答/代码建议用此项
/summary                  显示已评分轮次的 best-of-实际K
/new                      清空对话上下文，保留日志与统计
/history                  查看当前共同历史
/help                     显示帮助
/quit                     保存退出
没有参考答案/人工判定时，不会伪造 pass@5。
"""


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--session_json", required=True, help="New session file; existing file is refused.")
    p.add_argument("--embedding_model_name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--top_k_paths", type=int, default=5)
    p.add_argument("--beam_size", type=int, default=10)
    p.add_argument("--top_k_ops", type=int, default=2)
    p.add_argument("--seg_threshold", type=float, default=0.5)
    p.add_argument("--max_pack", type=int, default=4)
    p.add_argument("--polar_d_model", type=int, default=256)
    p.add_argument("--polar_heads", type=int, default=4)
    p.add_argument("--polar_layers", type=int, default=2)
    p.add_argument("--predictor_input", choices=("question", "history"), default="question")
    p.add_argument("--selection_strategy", choices=("top1", "nonbase_top1"), default="nonbase_top1")
    p.add_argument("--system_prompt", default="")
    p.add_argument("--no_stream", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    if args.top_k_paths < 1 or args.beam_size < args.top_k_paths or args.max_new_tokens < 1:
        raise ValueError("Require beam_size >= top_k_paths >= 1 and max_new_tokens >= 1.")
    if not 1 <= args.top_k_ops <= 3 or args.max_pack < 1 or not 0 <= args.seg_threshold <= 1:
        raise ValueError("Invalid decoder configuration.")
    output = Path(args.session_json)
    trace_path = output.with_suffix(".trace.jsonl")
    if output.exists() or trace_path.exists():
        raise ValueError("Session already exists. Choose a new --session_json; no implicit resume.")
    output.parent.mkdir(parents=True, exist_ok=True)
    turns, history = [], []
    configuration = vars(args).copy()

    def save():
        write_json(output, {"configuration": configuration, "summary": chat_summary(turns),
                            "history": history, "turns": turns})

    def trace(event):
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **event},
                               ensure_ascii=False) + "\n")

    # Lazy imports allow --help/tests on a CPU-only machine.
    import torch
    from transformers import TextStreamer
    from llm_depth_router.model import get_model, get_tokenizer, setup_custom_path
    from polar.model import PolarPredictor, decode_polar_to_actions
    from polar.data import actions_to_path

    torch.manual_seed(args.seed)
    print("[load] loading model and frozen predictor...", flush=True)
    model = get_model(args.model_path, args.device).eval()
    tokenizer = get_tokenizer(args.model_path)
    depth = len(model.model.layers)
    full_path = list(range(depth))
    predictor = PolarPredictor(
        num_layers=depth, embedding_model_name=args.embedding_model_name,
        d_model=args.polar_d_model, nheads=args.polar_heads,
        n_layer_blocks=args.polar_layers,
    ).to(args.device)
    predictor.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))
    predictor.eval()
    print(HELP, flush=True)
    print("[注意] 每轮为共同历史从头 prefill；不跨路径/轮次复用 KV。"
          "循环路径沿用仓库 patch 的 KV 行为。predictor 输入最多保留其自身的 512 tokens。",
          flush=True)
    save()

    def clock():
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize(model.device)
        return time.perf_counter()

    def generate(path, rendered, turn_id, label):
        inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
        context_limit = getattr(model.config, "max_position_embeddings", None)
        if context_limit and inputs.input_ids.shape[1] + args.max_new_tokens > context_limit:
            raise ValueError(f"Input + output budget exceeds context {context_limit}; "
                             "use /new or reduce --max_new_tokens. No silent truncation.")
        trace({"stage": "generation_start", "turn_id": turn_id, "label": label,
               "path": path})
        start = clock()
        setup_custom_path(model, path)
        try:
            with torch.inference_mode():
                result = model.generate(
                    **inputs, do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                    use_cache=True, pad_token_id=tokenizer.pad_token_id,
                    streamer=None if args.no_stream else TextStreamer(
                        tokenizer, skip_prompt=True, skip_special_tokens=True),
                )
        finally:
            setup_custom_path(model, full_path)
        elapsed = clock() - start
        ids = result[0][inputs.input_ids.shape[1]:]
        completion = tokenizer.decode(ids, skip_special_tokens=True).strip()
        if args.no_stream:
            print(completion, flush=True)
        record = {**describe_path(path, depth), "completion": completion,
                  "generation_seconds": elapsed, "generated_tokens": len(ids),
                  "hit_token_limit": len(ids) >= args.max_new_tokens}
        trace({"stage": "generation_end", "turn_id": turn_id, "label": label, **record})
        return record

    try:
        while True:
            text = input("\n你> ").strip()
            if not text:
                continue
            if text in ("/quit", "/exit"):
                break
            if text == "/help":
                print(HELP)
                continue
            if text == "/new":
                history.clear()
                trace({"stage": "new_conversation"})
                save()
                print("[new] 已清空上下文，日志保留。")
                continue
            if text == "/history":
                print(json.dumps(history, ensure_ascii=False, indent=2))
                continue
            if text == "/summary":
                print(json.dumps(chat_summary(turns), ensure_ascii=False, indent=2))
                continue
            if text.startswith(("/use ", "/reference ", "/grade ")):
                try:
                    if not turns or not history:
                        raise ValueError("当前对话还没有可操作的完整轮次。")
                    last = turns[-1]
                    command, value = text.split(" ", 1)
                    if command == "/use":
                        if value != "base" and not 1 <= int(value) <= len(last["candidates"]):
                            raise ValueError("候选编号越界。")
                        selected = last["base"] if value == "base" else last["candidates"][int(value) - 1]
                        last["history_choice"] = value
                        history[-1]["content"] = selected["completion"]
                        print(f"[history] 后续对话使用 {value} 回复。")
                    else:
                        grade = grade_reference(last, json.loads(value)) if command == "/reference" \
                            else grade_manual(last, json.loads(value))
                        last["reference_grade" if command == "/reference" else "manual_grade"] = grade
                        print(json.dumps(grade, ensure_ascii=False, indent=2))
                    trace({"stage": "user_command", "turn_id": last["turn_id"], "command": text})
                    save()
                except (ValueError, IndexError, TypeError) as exc:
                    print(f"[输入错误] {exc}")
                continue
            if text == "/multi":
                print("请输入多行，单独一行 /end 结束：")
                lines = []
                while True:
                    line = input()
                    if line == "/end":
                        break
                    lines.append(line)
                text = "\n".join(lines).strip()
                if not text:
                    continue
            elif text.startswith("/"):
                print("未知命令，输入 /help。")
                continue
            turn_id = len(turns) + 1
            messages = ([{"role": "system", "content": args.system_prompt}]
                        if args.system_prompt else []) + copy.deepcopy(history) + [
                            {"role": "user", "content": text}]
            rendered = chat_prompt(tokenizer, messages, args.model_path)
            ptext = predictor_text(messages, args.predictor_input)
            try:
                started = clock()
                trace({"stage": "turn_start", "turn_id": turn_id, "messages": messages,
                       "rendered_model_input": rendered, "predictor_input": ptext})
                with torch.inference_mode():
                    seg, op = predictor([ptext])
                beams = decode_polar_to_actions(
                    seg[0], op[0], threshold=args.seg_threshold, max_pack=args.max_pack,
                    beam_size=args.beam_size, top_k_ops=args.top_k_ops)
                paths = unique_paths(beams, depth, actions_to_path, args.top_k_paths)
                predictor_seconds = clock() - started
                print("\n=== Base / 完整路径 / greedy ===", flush=True)
                base = generate(full_path, rendered, turn_id, "base")
                candidates = []
                for rank, spec in enumerate(paths, 1):
                    desc = describe_path(spec["path"], depth)
                    print(f"\n=== POLAR {rank}/{len(paths)} / score={spec['score']:.4f} "
                          f"/ 执行{len(spec['path'])}次 ===\n"
                          f"path={spec['path']}\n"
                          f"skip={desc['skipped_layers']} repeat={desc['repeated_layers']}",
                          flush=True)
                    candidates.append({**spec, **generate(
                        spec["path"], rendered, turn_id, f"polar_{rank}")})
                selection_start = clock()
                selected, reason = select_candidate(base, candidates, args.selection_strategy)
                selected_rank = next((str(i) for i, c in enumerate(candidates, 1)
                                      if c is selected), "base")
                turn = {"turn_id": turn_id, "messages": messages, "user": text,
                        "predictor_input": ptext, "rendered_model_input": rendered,
                        "segmentation_logits": seg[0].detach().float().cpu().tolist(),
                        "operation_logits": op[0].detach().float().cpu().tolist(),
                        "base": base, "candidates": candidates,
                        "requested_k": args.top_k_paths, "actual_k": len(candidates),
                        "selection_reason": reason, "history_choice": selected_rank,
                        "selection_seconds": clock() - selection_start,
                        "predictor_seconds": predictor_seconds,
                        "end_to_end_seconds": clock() - started}
                turns.append(turn)
                history.extend([{"role": "user", "content": text},
                                {"role": "assistant", "content": selected["completion"]}])
                save()
                print(f"\n[完成] 实际候选 {len(candidates)}/{args.top_k_paths}；"
                      f"共同历史使用 {selected_rank}（可 /use 修改）。"
                      "\nbest-of-k：未评分。用 /grade 或 /reference 评分。", flush=True)
            except Exception as exc:
                trace({"stage": "turn_error", "turn_id": turn_id,
                       "error": f"{type(exc).__name__}: {exc}"})
                print(f"\n[失败] {type(exc).__name__}: {exc}\n本轮未写入共同历史；已完成输出见 trace。")
                if str(args.device).startswith("cuda"):
                    torch.cuda.empty_cache()
    except (EOFError, KeyboardInterrupt):
        print("\n[退出] 保存已完成轮次。")
    finally:
        save()
        print(f"[saved] {output}\n[trace] {trace_path}")


if __name__ == "__main__":
    main()
