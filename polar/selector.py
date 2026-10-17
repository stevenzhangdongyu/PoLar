"""MLP selector for choosing one path from Base + POLAR candidates."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


def load_eval_rows(path: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "results" in payload:
        rows = payload["results"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"Unsupported eval JSON shape: {type(payload)}")
    return [r for r in rows if isinstance(r, dict) and r.get("candidates")]


def is_correct(candidate: Dict[str, Any]) -> bool:
    return bool(candidate.get("exact_match") or candidate.get("passed"))


def task_text(row: Dict[str, Any]) -> str:
    # Prefer the text seen by the predictor/evaluator, but do not include references.
    return str(row.get("predictor_input") or row.get("question") or row.get("original_prompt")
               or row.get("user") or row.get("task_id") or "")


def path_features(path: List[int], depth: int, *, rank: int, score: float, is_base: bool) -> List[float]:
    counts = Counter(path)
    skipped = [i for i in range(depth) if counts[i] == 0]
    repeated = [i for i in range(depth) if counts[i] > 1]
    visits = [float(counts[i]) / max(1, max(counts.values() or [1])) for i in range(depth)]
    skip_mask = [1.0 if i in skipped else 0.0 for i in range(depth)]
    repeat_mask = [1.0 if i in repeated else 0.0 for i in range(depth)]
    nonmono = sum(1 for a, b in zip(path, path[1:]) if b < a)
    jumps = sum(1 for a, b in zip(path, path[1:]) if b != a + 1)
    scalars = [
        len(path) / depth,
        len(counts) / depth,
        len(skipped) / depth,
        len(repeated) / depth,
        sum(max(0, c - 1) for c in counts.values()) / depth,
        max(counts.values() or [0]) / 4.0,
        nonmono / depth,
        jumps / depth,
        float(score),
        rank / 6.0,
        1.0 if is_base else 0.0,
    ]
    return visits + skip_mask + repeat_mask + scalars


def build_candidates(row: Dict[str, Any], depth: int, max_polar_paths: int) -> Tuple[List[List[int]], List[float], List[float], List[int]]:
    base_path = row.get("base", {}).get("path") or list(range(depth))
    items = [(base_path, 0.0, is_correct(row.get("base", {})), 0)]
    for rank, cand in enumerate((row.get("candidates") or [])[:max_polar_paths], 1):
        path = cand.get("path")
        if not isinstance(path, list) or not path:
            continue
        items.append((path, float(cand.get("score", 0.0) or 0.0), is_correct(cand), rank))
    paths, scores, labels, ranks = [], [], [], []
    for path, score, label, rank in items:
        if all(isinstance(x, int) and 0 <= x < depth for x in path):
            paths.append(path)
            scores.append(score)
            labels.append(1.0 if label else 0.0)
            ranks.append(rank)
    return paths, scores, labels, ranks


@dataclass
class SelectorExample:
    text: str
    features: torch.Tensor  # (C,F)
    labels: torch.Tensor  # (C,)
    mask: torch.Tensor  # (C,)
    meta: Dict[str, Any]


class PathSelectorDataset(Dataset):
    def __init__(
        self,
        eval_jsons: List[str],
        depth: int,
        max_polar_paths: int = 5,
        include_all_wrong: bool = True,
        max_examples: int = 0,
    ) -> None:
        self.depth = int(depth)
        self.max_candidates = int(max_polar_paths) + 1
        examples: List[SelectorExample] = []
        for source in eval_jsons:
            for idx, row in enumerate(load_eval_rows(source)):
                paths, scores, labels, ranks = build_candidates(row, self.depth, max_polar_paths)
                if len(paths) < 2 or (not include_all_wrong and not any(labels)):
                    continue
                feats = [
                    path_features(p, self.depth, rank=r, score=s, is_base=(i == 0))
                    for i, (p, s, r) in enumerate(zip(paths, scores, ranks))
                ]
                n, fdim = len(feats), len(feats[0])
                padded = torch.zeros(self.max_candidates, fdim, dtype=torch.float32)
                label_t = torch.zeros(self.max_candidates, dtype=torch.float32)
                mask = torch.zeros(self.max_candidates, dtype=torch.bool)
                keep = min(n, self.max_candidates)
                padded[:keep] = torch.tensor(feats[:keep], dtype=torch.float32)
                label_t[:keep] = torch.tensor(labels[:keep], dtype=torch.float32)
                mask[:keep] = True
                examples.append(SelectorExample(
                    text=task_text(row),
                    features=padded,
                    labels=label_t,
                    mask=mask,
                    meta={"source": source, "idx": idx, "task_id": row.get("task_id"),
                          "base_correct": bool(labels[0]) if labels else False,
                          "num_candidates": keep},
                ))
                if max_examples and len(examples) >= max_examples:
                    break
            if max_examples and len(examples) >= max_examples:
                break
        if not examples:
            raise ValueError("No selector examples loaded.")
        self.examples = examples
        self.feature_dim = int(examples[0].features.shape[-1])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SelectorExample:
        return self.examples[idx]


def collate_selector(batch: List[SelectorExample]) -> Dict[str, Any]:
    return {
        "texts": [b.text for b in batch],
        "features": torch.stack([b.features for b in batch]),
        "labels": torch.stack([b.labels for b in batch]),
        "mask": torch.stack([b.mask for b in batch]),
        "meta": [b.meta for b in batch],
    }


class MLPPathSelector(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        text_encoder_name: str = "Qwen/Qwen3-Embedding-0.6B",
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, padding_side="left")
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        text_dim = int(self.text_encoder.config.hidden_size)
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.text_encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.text_proj(pooled)

    def forward(self, texts: List[str], features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.encode_text(texts)
        bsz, cand, _ = features.shape
        q = q.unsqueeze(1).expand(bsz, cand, q.shape[-1])
        logits = self.mlp(torch.cat([q, features.to(q.device)], dim=-1)).squeeze(-1)
        return logits.masked_fill(~mask.to(logits.device), -1e9)


def multi_positive_listwise_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    labels = labels.to(logits.device).float()
    mask = mask.to(logits.device)
    valid_logits = logits.masked_fill(~mask, -1e9)
    has_pos = (labels * mask.float()).sum(dim=1) > 0
    if not has_pos.any():
        return logits.sum() * 0.0
    log_probs = torch.log_softmax(valid_logits[has_pos], dim=1)
    pos = labels[has_pos].bool()
    return -torch.logsumexp(log_probs.masked_fill(~pos, -1e9), dim=1).mean()


def selector_metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=1).detach().cpu()
    labels_cpu = labels.detach().cpu()
    mask_cpu = mask.detach().cpu()
    selected_correct, base_correct = [], []
    oracle, all_wrong = [], []
    for i, j in enumerate(pred.tolist()):
        selected_correct.append(float(labels_cpu[i, j] > 0.5 and mask_cpu[i, j]))
        base_correct.append(float(labels_cpu[i, 0] > 0.5))
        oracle.append(float((labels_cpu[i] * mask_cpu[i].float()).max() > 0.5))
        all_wrong.append(float((labels_cpu[i] * mask_cpu[i].float()).sum() == 0))
    n = max(1, len(selected_correct))
    return {
        "selector_acc": sum(selected_correct) / n,
        "base_acc": sum(base_correct) / n,
        "oracle_acc": sum(oracle) / n,
        "all_wrong_rate": sum(all_wrong) / n,
    }
