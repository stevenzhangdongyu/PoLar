#!/usr/bin/env python3
"""
Generate PoLar supervision data with a lightweight MCTS-style search.

The output format matches what polar.data.PolarDataset expects:

{
  "samples": [
    {
      "question": "...",
      "gt_ans": "...",
      "initial_score": 0.0,
      "final_valid_transitions": [[0, 1, 2, ...]],
      "final_invalid_transitions": [[...]]
    }
  ]
}

This script is intentionally conservative and resumable. It is not the authors'
exact private MCTS implementation, but follows the paper's Appendix B recipe:
start from the standard path, expand skip/repeat operations over contiguous
blocks, execute each candidate path, and use answer correctness as reward.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from llm_depth_router.model import get_model, get_tokenizer
from polar.config import infer_original_depth, set_random_seed
from polar.eval import _online_eval_math_single


Path = Tuple[int, ...]


@dataclass
class Node:
    path: Path
    parent: Optional["Node"] = None
    children: Dict[Path, "Node"] = field(default_factory=dict)
    untried: List[Path] = field(default_factory=list)
    visits: int = 0
    reward_sum: float = 0.0


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    elif isinstance(data, dict):
        data = list(data.values())
    if not isinstance(data, list):
        raise ValueError(f"Expected list/dict JSON or JSONL, got {type(data)} from {path}")
    return data


def load_hf_dataset(dataset_name: str, split: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    try:
        import datasets
    except ImportError as exc:
        raise ImportError("Install `datasets` to use --hf_dataset, or pass --input_json instead.") from exc

    ds = datasets.load_dataset(dataset_name, split=split, trust_remote_code=True)
    rows = list(ds)
    return rows[:limit] if limit is not None else rows


def get_field(row: Dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        cur: Any = row
        ok = True
        for part in name.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None:
            return str(cur)
    return ""


def normalize_samples(
    rows: List[Dict[str, Any]],
    *,
    question_field: str,
    answer_field: str,
    start_idx: int,
    limit: Optional[int],
) -> List[Dict[str, str]]:
    sliced = rows[start_idx : start_idx + limit if limit is not None else None]
    out = []
    q_names = [question_field, "question", "problem", "query", "sample_info.question"]
    a_names = [answer_field, "gt_ans", "ground_truth", "answer", "ref_ans", "sample_info.answer"]
    for row in sliced:
        question = get_field(row, q_names)
        gt_ans = get_field(row, a_names)
        if question and gt_ans:
            out.append({"question": question, "gt_ans": gt_ans})
    return out


def dedup_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def generate_neighbors(
    path: Path,
    *,
    depth: int,
    max_block: int,
    max_repeat: int,
    min_path_len: int,
    max_path_len: int,
) -> List[Path]:
    candidates: List[Path] = []
    n = len(path)

    # Skip: remove a contiguous span from the current executable program.
    for start in range(n):
        for size in range(1, max_block + 1):
            end = start + size
            if end > n:
                break
            new_path = path[:start] + path[end:]
            if min_path_len <= len(new_path) <= max_path_len:
                candidates.append(new_path)

    # Repeat: duplicate a contiguous span in place.
    for start in range(n):
        for size in range(1, max_block + 1):
            end = start + size
            if end > n:
                break
            block = path[start:end]
            for rep in range(1, max_repeat + 1):
                new_path = path[:end] + block * rep + path[end:]
                if min_path_len <= len(new_path) <= max_path_len:
                    # All layer ids should stay inside the frozen base model.
                    if all(0 <= layer < depth for layer in new_path):
                        candidates.append(new_path)

    return dedup_paths(candidates)


def ucb_score(node: Node, total_visits: int, exploration: float, length_penalty: float, depth: int) -> float:
    if node.visits == 0:
        return float("inf")
    exploit = node.reward_sum / node.visits
    explore = exploration * math.sqrt(math.log(max(total_visits, 2)) / node.visits)
    length = length_penalty * (len(node.path) / max(depth, 1))
    return exploit + explore - length


@torch.no_grad()
def score_path(
    *,
    model: Any,
    tokenizer: Any,
    model_path: str,
    path: Path,
    question: str,
    gt_ans: str,
    max_new_tokens: int,
) -> float:
    if not path:
        return 0.0
    return float(
        _online_eval_math_single(
            model=model,
            tokenizer=tokenizer,
            model_path=model_path,
            transition=list(path),
            question=question,
            gt=gt_ans,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
    )


def run_search_for_sample(
    *,
    model: Any,
    tokenizer: Any,
    model_path: str,
    question: str,
    gt_ans: str,
    depth: int,
    simulations: int,
    max_block: int,
    max_repeat: int,
    max_children_per_expand: int,
    exploration: float,
    length_penalty: float,
    max_new_tokens: int,
    max_path_len_ratio: float,
    rng: random.Random,
) -> Dict[str, Any]:
    root_path: Path = tuple(range(depth))
    max_path_len = max(depth, int(math.ceil(depth * max_path_len_ratio)))
    min_path_len = 1
    cache: Dict[Path, float] = {}
    valid: List[Path] = []
    invalid: List[Path] = []

    def eval_cached(path: Path) -> float:
        if path not in cache:
            cache[path] = score_path(
                model=model,
                tokenizer=tokenizer,
                model_path=model_path,
                path=path,
                question=question,
                gt_ans=gt_ans,
                max_new_tokens=max_new_tokens,
            )
            if cache[path] > 0:
                valid.append(path)
            else:
                invalid.append(path)
        return cache[path]

    initial_score = eval_cached(root_path)
    root = Node(path=root_path)
    root.untried = generate_neighbors(
        root.path,
        depth=depth,
        max_block=max_block,
        max_repeat=max_repeat,
        min_path_len=min_path_len,
        max_path_len=max_path_len,
    )
    rng.shuffle(root.untried)
    if max_children_per_expand > 0:
        root.untried = root.untried[:max_children_per_expand]

    for _ in range(simulations):
        node = root

        # Selection.
        while not node.untried and node.children:
            total = max(1, node.visits)
            node = max(
                node.children.values(),
                key=lambda child: ucb_score(child, total, exploration, length_penalty, depth),
            )

        # Expansion.
        if node.untried:
            child_path = node.untried.pop()
            child = Node(path=child_path, parent=node)
            child.untried = generate_neighbors(
                child.path,
                depth=depth,
                max_block=max_block,
                max_repeat=max_repeat,
                min_path_len=min_path_len,
                max_path_len=max_path_len,
            )
            rng.shuffle(child.untried)
            if max_children_per_expand > 0:
                child.untried = child.untried[:max_children_per_expand]
            node.children[child_path] = child
            node = child

        # Simulation/evaluation.
        reward = eval_cached(node.path)

        # Backpropagation.
        while node is not None:
            node.visits += 1
            node.reward_sum += reward
            node = node.parent

    return {
        "question": question,
        "gt_ans": gt_ans,
        "initial_score": float(initial_score),
        "final_valid_transitions": [list(p) for p in dedup_paths(valid)],
        "final_invalid_transitions": [list(p) for p in dedup_paths(invalid)],
        "search_stats": {
            "num_evaluated_paths": len(cache),
            "num_valid_paths": len(dedup_paths(valid)),
            "num_invalid_paths": len(dedup_paths(invalid)),
            "simulations": simulations,
            "max_block": max_block,
            "max_repeat": max_repeat,
            "max_path_len_ratio": max_path_len_ratio,
        },
    }


def save_output(path: str, samples: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "metadata": {
            "generator": "scripts/generate_mcts_samples.py",
            "model_path": args.model_path,
            "original_depth": args.original_depth,
            "note": "MCTS-style supervision generated locally; not the authors' released/private traces.",
        },
        "samples": samples,
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PoLar merged_mcts_samples.json supervision.")
    parser.add_argument("--model_path", required=True, help="HF model id supported by llm_depth_router.")
    parser.add_argument("--output", required=True, help="Output merged_mcts_samples.json path.")
    parser.add_argument("--input_json", default=None, help="JSON/JSONL samples with question/answer fields.")
    parser.add_argument("--hf_dataset", default=None, help="Optional HuggingFace dataset name.")
    parser.add_argument("--hf_split", default="train", help="HuggingFace split when --hf_dataset is used.")
    parser.add_argument("--question_field", default="question")
    parser.add_argument("--answer_field", default="gt_ans")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10, help="Number of input samples to process.")
    parser.add_argument("--simulations", type=int, default=64, help="MCTS simulations per sample.")
    parser.add_argument("--max_block", type=int, default=4, help="Max contiguous block size for skip/repeat.")
    parser.add_argument("--max_repeat", type=int, default=1, help="Repeat count per action. 1 means duplicate once.")
    parser.add_argument("--max_children_per_expand", type=int, default=64, help="Subsample children per expanded node; <=0 keeps all.")
    parser.add_argument("--exploration", type=float, default=1.4)
    parser.add_argument("--length_penalty", type=float, default=0.05)
    parser.add_argument("--max_path_len_ratio", type=float, default=1.15)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1, help="Write partial output every N processed samples.")
    parser.add_argument("--resume", action="store_true", help="Append to existing output and skip already processed questions.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_random_seed(args.seed)
    rng = random.Random(args.seed)
    args.original_depth = int(infer_original_depth(args.model_path))

    if args.input_json:
        rows = read_json_or_jsonl(args.input_json)
    elif args.hf_dataset:
        rows = load_hf_dataset(args.hf_dataset, args.hf_split, args.start_idx + args.limit if args.limit else None)
    else:
        raise ValueError("Provide either --input_json or --hf_dataset.")

    samples = normalize_samples(
        rows,
        question_field=args.question_field,
        answer_field=args.answer_field,
        start_idx=args.start_idx,
        limit=args.limit,
    )
    if not samples:
        raise ValueError("No usable samples found. Check question/answer fields.")

    done: List[Dict[str, Any]] = []
    seen_questions = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r") as f:
            old = json.load(f)
        done = old.get("samples", old if isinstance(old, list) else [])
        seen_questions = {str(s.get("question", "")) for s in done}
        print(f"[resume] loaded {len(done)} existing samples")

    print(f"[init] loading model {args.model_path} on {args.device}")
    model = get_model(args.model_path, device=args.device)
    tokenizer = get_tokenizer(args.model_path)
    model.eval()

    generated = list(done)
    todo = [s for s in samples if s["question"] not in seen_questions]
    print(f"[init] processing {len(todo)} sample(s), depth={args.original_depth}")

    for idx, sample in enumerate(todo, start=1):
        print(f"[sample {idx}/{len(todo)}] searching: {sample['question'][:80]!r}")
        result = run_search_for_sample(
            model=model,
            tokenizer=tokenizer,
            model_path=args.model_path,
            question=sample["question"],
            gt_ans=sample["gt_ans"],
            depth=args.original_depth,
            simulations=args.simulations,
            max_block=args.max_block,
            max_repeat=args.max_repeat,
            max_children_per_expand=args.max_children_per_expand,
            exploration=args.exploration,
            length_penalty=args.length_penalty,
            max_new_tokens=args.max_new_tokens,
            max_path_len_ratio=args.max_path_len_ratio,
            rng=rng,
        )
        generated.append(result)
        stats = result["search_stats"]
        print(
            "[sample done] initial={:.0f} valid={} invalid={} evaluated={}".format(
                result["initial_score"],
                stats["num_valid_paths"],
                stats["num_invalid_paths"],
                stats["num_evaluated_paths"],
            )
        )
        if args.save_every > 0 and idx % args.save_every == 0:
            save_output(args.output, generated, args)

    save_output(args.output, generated, args)
    print(f"[done] wrote {len(generated)} samples to {args.output}")


if __name__ == "__main__":
    main()
