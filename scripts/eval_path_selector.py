#!/usr/bin/env python3
"""Evaluate a trained path selector on existing eval JSON without rerunning generation."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from polar.selector import MLPPathSelector, PathSelectorDataset, collate_selector


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval_json", action="append", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_mode", choices=("checkpoint", "standard", "repair", "repair_subset"),
                   default="checkpoint",
                   help="repair masks Base; repair_subset also keeps only Base-wrong/POLAR-correct rows.")
    p.add_argument("--local_files_only", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.local_files_only:
        import os
        os.environ["HF_LOCAL_FILES_ONLY"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    eval_mode = cfg.get("train_mode", "standard") if args.eval_mode == "checkpoint" else args.eval_mode
    dataset_mode = "repair" if eval_mode in {"repair", "repair_subset"} else "standard"
    dataset = PathSelectorDataset(
        eval_jsons=args.eval_json,
        depth=int(ckpt["depth"]),
        max_polar_paths=int(cfg.get("max_polar_paths", 5)),
        include_all_wrong=True,
        train_mode=dataset_mode,
    )
    if eval_mode == "repair_subset":
        dataset.examples = [
            ex for ex in dataset.examples
            if (not ex.meta.get("base_correct")) and ex.meta.get("polar_has_correct")
        ]
        if not dataset.examples:
            raise ValueError("No repair-subset examples in eval data.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_selector)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPPathSelector(
        feature_dim=int(ckpt["feature_dim"]),
        text_encoder_name=cfg["text_encoder_name"],
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    rows, selected_correct, base_correct, oracle_correct, selected_base = [], [], [], [], []
    with torch.no_grad():
        offset = 0
        for batch in loader:
            logits = model(batch["texts"], batch["features"].to(device), batch["mask"].to(device))
            pred = logits.argmax(dim=1).cpu()
            labels = batch["labels"]
            mask = batch["mask"]
            for i, j in enumerate(pred.tolist()):
                meta = batch["meta"][i]
                ok = bool(labels[i, j] > 0.5 and mask[i, j])
                bok = bool(labels[i, 0] > 0.5)
                ook = bool((labels[i] * mask[i].float()).max() > 0.5)
                selected_correct.append(ok)
                base_correct.append(bok)
                oracle_correct.append(ook)
                selected_base.append(j == 0)
                rows.append({"meta": meta, "selected_index": j, "selected_correct": ok,
                             "base_correct": bok, "oracle_correct": ook,
                             "scores": logits[i].detach().cpu().tolist(),
                             "labels": labels[i].tolist(),
                             "mask": mask[i].tolist()})
            offset += len(batch["texts"])
    repaired = sum((not b) and s for b, s in zip(base_correct, selected_correct))
    degraded = sum(b and (not s) for b, s in zip(base_correct, selected_correct))
    n = len(rows)
    summary = {
        "n": n,
        "selector_correct": sum(selected_correct),
        "selector_accuracy": sum(selected_correct) / n if n else None,
        "base_correct": sum(base_correct),
        "base_accuracy": sum(base_correct) / n if n else None,
        "oracle_correct": sum(oracle_correct),
        "oracle_accuracy": sum(oracle_correct) / n if n else None,
        "selected_base_count": sum(selected_base),
        "base_select_rate": sum(selected_base) / n if n else None,
        "repair_count": repaired,
        "degradation_count": degraded,
        "eval_mode": eval_mode,
        "repair_rate_given_base_wrong": repaired / max(1, sum(not x for x in base_correct)),
        "degradation_rate_given_base_correct": degraded / max(1, sum(base_correct)),
        "note": "Offline path selection over already generated candidates; no generation rerun.",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps({"summary": summary, "rows": rows},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
