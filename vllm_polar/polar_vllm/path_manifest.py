"""Stable mapping between tokenized prompts and per-request layer paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def token_ids_key(token_ids: Iterable[int]) -> str:
    """Return a stable key for the exact prompt token sequence."""
    payload = ",".join(str(int(token_id)) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_manifest(
    rows: list[dict[str, Any]], tokenizer, *, prompt_prefix: str = "Solve the following task."
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Tokenize deterministic unique prompts and map their IDs to paths."""
    entries: dict[str, Any] = {}
    prompts: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        request_id = str(row.get("request_id", f"req-{index:06d}"))
        path = row.get("path")
        if not isinstance(path, list) or not path:
            raise ValueError(f"{request_id}: path must be a non-empty list")
        if any(type(layer) is not int or layer < 0 for layer in path):
            raise ValueError(f"{request_id}: path entries must be non-negative integers")

        prompt = f"{prompt_prefix} Request identifier: {request_id}. Return a short answer."
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        key = token_ids_key(token_ids)
        if key in entries:
            raise ValueError(f"tokenized prompt collision for request {request_id}")
        entries[key] = {"request_id": request_id, "path": path}
        prompts.append(
            {
                "request_id": request_id,
                "prompt": prompt,
                "token_ids": token_ids,
                "path": path,
            }
        )

    manifest = {
        "schema_version": 1,
        "key": "sha256(comma-separated prompt token ids)",
        "request_count": len(entries),
        "entries": entries,
    }
    return manifest, prompts


def load_manifest(filename: str | Path) -> dict[str, Any]:
    with Path(filename).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("entries"), dict
    ):
        raise ValueError("unsupported or malformed PoLar path manifest")
    return manifest


def resolve_path(manifest: dict[str, Any], token_ids: Iterable[int]) -> list[int]:
    entry = manifest.get("entries", {}).get(token_ids_key(token_ids))
    if entry is None:
        raise KeyError("request prompt token IDs are missing from the PoLar path manifest")
    path = entry.get("path")
    if not isinstance(path, list) or not path:
        raise ValueError("manifest entry has an invalid path")
    return path
