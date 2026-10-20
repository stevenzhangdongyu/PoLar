"""Shared helpers for POLAR task evaluators."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        data = [json.loads(x) for x in f if x.strip()] if p.suffix.lower() == ".jsonl" else json.load(f)
    if isinstance(data, dict):
        data = data.get("data", data.get("examples", data.get("samples", data.get("tasks", list(data.values())))))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON/JSONL list.")
    return [x for x in data if isinstance(x, dict)]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def render_chat(tokenizer, model_path: str, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in str(model_path).lower():
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


def path_description(path: List[int], depth: int) -> Dict[str, Any]:
    counts = {i: path.count(i) for i in set(path)}
    return {
        "path": path,
        "unique_depth": len(counts),
        "total_layer_executions": len(path),
        "skipped_layers": [i for i in range(depth) if i not in counts],
        "repeated_layers": sorted(i for i, n in counts.items() if n > 1),
        "layer_visit_counts": {str(i): n for i, n in sorted(counts.items())},
    }


def mean(values):
    return sum(values) / len(values) if values else None
