#!/usr/bin/env python3
"""Compare paired POLAR evaluations with layer and execution KV caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(path, expected_mode):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = payload["summary"]
    if not summary.get("complete"):
        raise ValueError(f"Incomplete evaluation: {path}")
    mode = summary.get("configuration", {}).get("polar_kv_cache_mode")
    if mode != expected_mode:
        raise ValueError(f"Expected {expected_mode} KV mode in {path}; found {mode!r}")
    rows = payload["results"]
    by_id = {str(row["task_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate task IDs in {path}")
    return by_id, summary["configuration"]


def correct(item, domain):
    if domain == "qa":
        return bool(item["exact_match"])
    if domain == "code":
        return bool(item["passed"])
    return None


def compare(layer_rows, execution_rows, domain):
    if set(layer_rows) != set(execution_rows):
        raise ValueError("Task IDs differ between evaluations")
    totals = {
        "tasks": len(layer_rows),
        "base_output_changes": 0,
        "candidate_output_changes": 0,
        "repeated_candidates": 0,
        "repeated_candidate_output_changes": 0,
        "nonrepeated_candidates": 0,
        "nonrepeated_candidate_output_changes": 0,
    }
    if domain != "instruction":
        totals.update({
            "layer_top1_correct": 0, "execution_top1_correct": 0,
            "layer_topk_oracle_correct": 0, "execution_topk_oracle_correct": 0,
            "layer_repeated_candidate_correct": 0,
            "execution_repeated_candidate_correct": 0,
            "repeated_candidate_improved": 0, "repeated_candidate_regressed": 0,
        })
    for task_id in layer_rows:
        old, new = layer_rows[task_id], execution_rows[task_id]
        for field in ("question", "answers", "prompt", "original_prompt"):
            if old.get(field) != new.get(field):
                raise ValueError(f"{field} differs for task {task_id}")
        old_candidates, new_candidates = old["candidates"], new["candidates"]
        if [c["path"] for c in old_candidates] != [c["path"] for c in new_candidates]:
            raise ValueError(f"Candidate paths differ for task {task_id}")
        totals["base_output_changes"] += old["base"]["completion"] != new["base"]["completion"]
        if domain != "instruction":
            totals["layer_top1_correct"] += bool(old_candidates and correct(old_candidates[0], domain))
            totals["execution_top1_correct"] += bool(new_candidates and correct(new_candidates[0], domain))
            totals["layer_topk_oracle_correct"] += any(correct(c, domain) for c in old_candidates)
            totals["execution_topk_oracle_correct"] += any(correct(c, domain) for c in new_candidates)
        for old_c, new_c in zip(old_candidates, new_candidates):
            repeated = len(old_c["path"]) != len(set(old_c["path"]))
            prefix = "repeated" if repeated else "nonrepeated"
            totals[f"{prefix}_candidates"] += 1
            changed = old_c["completion"] != new_c["completion"]
            totals["candidate_output_changes"] += changed
            totals[f"{prefix}_candidate_output_changes"] += changed
            if repeated and domain != "instruction":
                old_ok, new_ok = correct(old_c, domain), correct(new_c, domain)
                totals["layer_repeated_candidate_correct"] += old_ok
                totals["execution_repeated_candidate_correct"] += new_ok
                totals["repeated_candidate_improved"] += new_ok and not old_ok
                totals["repeated_candidate_regressed"] += old_ok and not new_ok
    if domain != "instruction":
        for label, denominator in (("top1", totals["tasks"]),
                                   ("topk_oracle", totals["tasks"]),
                                   ("repeated_candidate", totals["repeated_candidates"])):
            totals[f"layer_{label}_accuracy"] = totals[f"layer_{label}_correct"] / denominator if denominator else None
            totals[f"execution_{label}_accuracy"] = totals[f"execution_{label}_correct"] / denominator if denominator else None
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("qa", "code", "instruction"), required=True)
    parser.add_argument("--layer", required=True, help="layer-mode merged_results.json")
    parser.add_argument("--execution", required=True, help="execution-mode merged_results.json")
    args = parser.parse_args()
    layer_rows, layer_config = load_results(args.layer, "layer")
    execution_rows, execution_config = load_results(args.execution, "execution")
    for key in ("input_sha256", "model_path", "checkpoint_path", "max_new_tokens",
                "beam_size", "top_k_ops", "top_k_paths", "seg_threshold", "max_pack",
                "embedding_model_name", "polar_d_model", "polar_heads", "polar_layers", "seed"):
        if layer_config.get(key) != execution_config.get(key):
            raise ValueError(f"Evaluation configuration differs: {key}")
    print(json.dumps(compare(layer_rows, execution_rows, args.domain), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
