#!/usr/bin/env python3
"""
Re-run stored PoLar valid paths and verify that they still produce correct answers.

This is useful after generating `merged_mcts_samples.json`: it does not trust the
cache blindly, but reloads the base LLM, applies each candidate `custom_path`,
generates the final answer, and checks it with the repository's math evaluator.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List, Optional

import torch

from llm_depth_router.model import get_model, get_tokenizer
from llm_depth_router.model import setup_custom_path
from polar.eval import (
    _batch_compare_answers,
    _is_qwen15_moe_chat_model_path,
    _is_qwen25_instruct_model_path,
    _is_qwen3_model_path,
    _qwen15_moe_apply_chat_template,
    _qwen25_apply_chat_template,
    _qwen3_apply_chat_template,
    _qwen3_split_thinking,
    _online_eval_math_single,
)


def build_input_text(question: str) -> str:
    return (
        "Solve the following math problem and output ONLY the final answer directly, "
        "formatted strictly as \\boxed{ANSWER}.\n"
        "### Problem Start\n"
        f"{question}\n"
        "### Problem End\n"
        "Answer:"
    )


def format_model_input(tokenizer, model_path: str, input_text: str) -> str:
    if _is_qwen3_model_path(model_path):
        return _qwen3_apply_chat_template(tokenizer, input_text)
    if _is_qwen15_moe_chat_model_path(model_path):
        return _qwen15_moe_apply_chat_template(tokenizer, input_text)
    if _is_qwen25_instruct_model_path(model_path):
        return _qwen25_apply_chat_template(tokenizer, input_text)
    return input_text


@torch.no_grad()
def eval_math_single_with_trace(
    *,
    model,
    tokenizer,
    model_path: str,
    transition: List[int],
    question: str,
    gt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Same online check as polar.eval._online_eval_math_single, but keeps the
    actual prompt/model input/generated text so humans can inspect it.
    """
    input_text = build_input_text(question)
    rendered_input = format_model_input(tokenizer, model_path, input_text)

    setup_custom_path(model, transition)

    if _is_qwen3_model_path(model_path):
        model_inputs = tokenizer([rendered_input], return_tensors="pt").to(model.device)
        if temperature > 0:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
            )
        else:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        thinking, answer_part = _qwen3_split_thinking(output_ids, tokenizer)
        answer_part = answer_part.strip()
        raw_completion = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    else:
        model_inputs = tokenizer([rendered_input], return_tensors="pt").to(model.device)
        if temperature > 0:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=True,
                temperature=float(temperature),
            )
        else:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
            )
        new_token_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        raw_completion = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0].strip()
        thinking = ""
        if (
            _is_qwen15_moe_chat_model_path(model_path)
            or _is_qwen25_instruct_model_path(model_path)
        ):
            answer_part = raw_completion.strip()
        else:
            full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            answer_part = full_text.split("Answer:")[-1].strip()

    follows_boxed_format = "oxed{" in answer_part
    correct = bool(follows_boxed_format and _batch_compare_answers([answer_part], [gt])[0])
    return {
        "input_text": input_text,
        "rendered_model_input": rendered_input,
        "transition": transition,
        "gt": gt,
        "raw_completion": raw_completion,
        "thinking": thinking,
        "answer_part": answer_part,
        "follows_boxed_format": follows_boxed_format,
        "score": 1.0 if correct else 0.0,
    }


def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    raise ValueError(f"Unsupported JSON shape: {type(data)}")


def get_question_and_answer(sample: Dict[str, Any]) -> tuple[str, str]:
    question = sample.get("question") or sample.get("query") or sample.get("sample_info", {}).get("question") or ""
    gt = (
        sample.get("gt_ans")
        or sample.get("ground_truth")
        or sample.get("answer")
        or sample.get("ref_ans")
        or sample.get("sample_info", {}).get("ground_truth")
        or sample.get("sample_info", {}).get("answer")
        or ""
    )
    return str(question), str(gt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify final_valid_transitions by online LLM re-evaluation.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--merged_json", required=True, help="Path to merged_mcts_samples.json.")
    parser.add_argument("--sample_limit", type=int, default=10, help="Number of samples to verify.")
    parser.add_argument("--paths_per_sample", type=int, default=3, help="Max valid paths verified per sample.")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--random_samples", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include_invalid", action="store_true", help="Also verify invalid paths as a sanity check.")
    parser.add_argument("--invalid_paths_per_sample", type=int, default=1)
    parser.add_argument("--debug_io", action="store_true", help="Print model input and generated output for every checked path.")
    parser.add_argument("--debug_initial_path", action="store_true", help="Also print/save the original full-depth path [0, 1, ..., depth-1].")
    parser.add_argument("--only_initial_path", action="store_true", help="Only evaluate the original full-depth path; skip valid/invalid stored paths.")
    parser.add_argument("--debug_output_jsonl", default=None, help="Optional JSONL path to save debug traces.")
    return parser


def infer_model_depth(model: Any, sample: Dict[str, Any]) -> int:
    depth = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if depth is not None:
        return int(depth)

    all_paths = []
    all_paths.extend(sample.get("final_valid_transitions", []) or [])
    all_paths.extend(sample.get("final_invalid_transitions", []) or [])
    if all_paths:
        return max(max(int(x) for x in path) for path in all_paths if path) + 1
    raise ValueError("Cannot infer model depth from model.config or sample paths.")


def print_trace(trace: Dict[str, Any]) -> None:
    print("    --- model input begin ---")
    print(trace["rendered_model_input"])
    print("    --- model input end ---")
    print("    --- model output begin ---")
    print(trace["raw_completion"])
    print("    --- model output end ---")
    print(f"    extracted_answer={trace['answer_part']!r}")
    print(f"    ground_truth={trace['gt']!r} boxed={trace['follows_boxed_format']}")


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    samples = load_samples(args.merged_json)

    indexed = list(enumerate(samples))
    if args.random_samples:
        rng.shuffle(indexed)
        picked = indexed[: args.sample_limit]
    else:
        picked = indexed[args.sample_start : args.sample_start + args.sample_limit]

    print(f"[init] loading model {args.model_path} on {args.device}")
    model = get_model(args.model_path, device=args.device)
    tokenizer = get_tokenizer(args.model_path)
    model.eval()
    debug_f: Optional[Any] = None
    if args.debug_output_jsonl:
        debug_f = open(args.debug_output_jsonl, "w")

    checked = 0
    correct = 0
    invalid_checked = 0
    invalid_correct = 0

    for sample_idx, sample in picked:
        question, gt = get_question_and_answer(sample)
        valid_paths = sample.get("final_valid_transitions", []) or []
        invalid_paths = sample.get("final_invalid_transitions", []) or []
        print(
            f"\n[sample {sample_idx}] valid_paths={len(valid_paths)} "
            f"invalid_paths={len(invalid_paths)} question={question[:90]!r}"
        )

        if args.debug_initial_path or args.only_initial_path:
            depth = infer_model_depth(model, sample)
            initial_path = list(range(depth))
            trace = eval_math_single_with_trace(
                model=model,
                tokenizer=tokenizer,
                model_path=args.model_path,
                transition=initial_path,
                question=question,
                gt=gt,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )
            trace.update({"sample_idx": sample_idx, "path_idx": 0, "path_type": "initial"})
            if debug_f is not None:
                debug_f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                debug_f.flush()
            print_trace(trace)
            status = "PASS" if trace["score"] == 1.0 else "FAIL"
            print(f"  [initial] {status} len={len(initial_path)} path={initial_path}")

        if args.only_initial_path:
            continue

        for path_idx, path in enumerate(valid_paths[: args.paths_per_sample]):
            if args.debug_io or debug_f is not None:
                trace = eval_math_single_with_trace(
                    model=model,
                    tokenizer=tokenizer,
                    model_path=args.model_path,
                    transition=[int(x) for x in path],
                    question=question,
                    gt=gt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                )
                score = trace["score"]
                trace.update({"sample_idx": sample_idx, "path_idx": path_idx, "path_type": "valid"})
                if debug_f is not None:
                    debug_f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    debug_f.flush()
                if args.debug_io:
                    print_trace(trace)
            else:
                score = _online_eval_math_single(
                    model=model,
                    tokenizer=tokenizer,
                    model_path=args.model_path,
                    transition=[int(x) for x in path],
                    question=question,
                    gt=gt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                )
            checked += 1
            correct += int(score == 1.0)
            status = "PASS" if score == 1.0 else "FAIL"
            print(f"  [valid {path_idx}] {status} len={len(path)} path={path}")

        if args.include_invalid:
            for path_idx, path in enumerate(invalid_paths[: args.invalid_paths_per_sample]):
                if args.debug_io or debug_f is not None:
                    trace = eval_math_single_with_trace(
                        model=model,
                        tokenizer=tokenizer,
                        model_path=args.model_path,
                        transition=[int(x) for x in path],
                        question=question,
                        gt=gt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.0,
                    )
                    score = trace["score"]
                    trace.update({"sample_idx": sample_idx, "path_idx": path_idx, "path_type": "invalid"})
                    if debug_f is not None:
                        debug_f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                        debug_f.flush()
                    if args.debug_io:
                        print_trace(trace)
                else:
                    score = _online_eval_math_single(
                        model=model,
                        tokenizer=tokenizer,
                        model_path=args.model_path,
                        transition=[int(x) for x in path],
                        question=question,
                        gt=gt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.0,
                    )
                invalid_checked += 1
                invalid_correct += int(score == 1.0)
                status = "CORRECT_NOW" if score == 1.0 else "still_invalid"
                print(f"  [invalid {path_idx}] {status} len={len(path)} path={path}")

    print("\n[summary]")
    print(f"valid path verification: {correct}/{checked} passed")
    if args.include_invalid:
        print(f"invalid sanity check: {invalid_correct}/{invalid_checked} unexpectedly correct")
    if debug_f is not None:
        debug_f.close()
        print(f"debug traces saved to: {args.debug_output_jsonl}")


if __name__ == "__main__":
    main()
