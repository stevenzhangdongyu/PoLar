#!/usr/bin/env python3
"""Train a lightweight MLP selector over Base + top-5 POLAR paths."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from polar.selector import (
    MLPPathSelector, PathSelectorDataset, collate_selector,
    group_split_indices, multi_positive_listwise_loss, selector_metrics,
)
from polar.config import infer_original_depth, set_random_seed


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_json", action="append", required=True,
                   help="QA/HumanEval merged_results.json; repeat for multiple domains.")
    p.add_argument("--model_path", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--text_encoder_name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--max_polar_paths", type=int, default=5)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--max_examples", type=int, default=0)
    p.add_argument("--skip_all_wrong", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    return p


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    sums = {"selector_acc": 0.0, "base_acc": 0.0, "oracle_acc": 0.0, "all_wrong_rate": 0.0}
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["texts"], batch["features"].to(device), batch["mask"].to(device))
            loss = multi_positive_listwise_loss(logits, batch["labels"].to(device), batch["mask"].to(device))
            metrics = selector_metrics(logits, batch["labels"], batch["mask"])
            bsz = len(batch["texts"])
            total_loss += float(loss.item()) * bsz
            total_n += bsz
            for k, v in metrics.items():
                sums[k] += v * bsz
    out = {k: v / max(1, total_n) for k, v in sums.items()}
    out["loss"] = total_loss / max(1, total_n)
    out["n"] = total_n
    return out


def main():
    args = build_parser().parse_args()
    if args.local_files_only:
        import os
        os.environ["HF_LOCAL_FILES_ONLY"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    set_random_seed(args.seed)
    depth = args.depth or infer_original_depth(args.model_path)
    dataset = PathSelectorDataset(
        eval_jsons=args.eval_json,
        depth=depth,
        max_polar_paths=args.max_polar_paths,
        include_all_wrong=not args.skip_all_wrong,
        max_examples=args.max_examples,
    )
    train_idx, val_idx, test_idx = group_split_indices(dataset, args.val_ratio, args.test_ratio, args.seed)
    loaders = {}
    for name, ids, shuffle in (("train", train_idx, True), ("val", val_idx, False), ("test", test_idx, False)):
        if ids:
            loaders[name] = DataLoader(Subset(dataset, ids), batch_size=args.batch_size,
                                       shuffle=shuffle, collate_fn=collate_selector)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPPathSelector(
        feature_dim=dataset.feature_dim,
        text_encoder_name=args.text_encoder_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"depth": depth, "feature_dim": dataset.feature_dim,
                           "num_examples": len(dataset),
                           "splits": {k: len(v) for k, v in {
                               "train": train_idx, "val": val_idx, "test": test_idx}.items()}}
    (out_dir / "selector_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    best_val, best_path, history = float("inf"), out_dir / "selector_mlp_best.pt", []
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    for epoch in range(args.num_epochs):
        model.train()
        total_loss, total_n = 0.0, 0
        for step, batch in enumerate(loaders["train"], 1):
            opt.zero_grad(set_to_none=True)
            logits = model(batch["texts"], batch["features"].to(device), batch["mask"].to(device))
            loss = multi_positive_listwise_loss(logits, batch["labels"].to(device), batch["mask"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bsz = len(batch["texts"])
            total_loss += float(loss.item()) * bsz
            total_n += bsz
            if step % 50 == 0:
                print(f"[epoch {epoch}] step={step} loss={total_loss/max(1,total_n):.4f}", flush=True)
        record = {"epoch": epoch, "train_loss": total_loss / max(1, total_n)}
        for name in ("val", "test"):
            if name in loaders:
                record[name] = evaluate(model, loaders[name], device)
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        val_loss = record.get("val", record).get("loss", record["train_loss"])
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(), "config": config,
                "feature_dim": dataset.feature_dim, "depth": depth,
            }, best_path)
            print(f"[saved] {best_path}", flush=True)
    (out_dir / "selector_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
