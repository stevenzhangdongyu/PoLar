"""Path and cache configuration helpers, kept independent of vLLM imports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def validate_path(path: Any, depth: int) -> list[int]:
    if not isinstance(path, list) or not path:
        raise ValueError("POLAR_PATH_JSON must contain a non-empty JSON array of layer indices")
    if any(type(layer) is not int for layer in path):
        raise ValueError("Every path entry must be an integer physical layer index")
    if any(layer < 0 or layer >= depth for layer in path):
        raise ValueError(f"Path indices must be in [0, {depth})")
    return list(path)


def load_runtime_config(env: dict[str, str] | None = None) -> tuple[list[int], str]:
    env = os.environ if env is None else env
    filename = env.get("POLAR_PATH_JSON")
    if not filename:
        raise ValueError("Set POLAR_PATH_JSON to a JSON file containing a layer-index array")
    with Path(filename).open(encoding="utf-8") as stream:
        path = json.load(stream)
    mode = env.get("POLAR_KV_CACHE_MODE", "execution").strip().lower()
    aliases = {"exec": "execution", "step": "execution", "physical": "layer"}
    mode = aliases.get(mode, mode)
    if mode not in {"layer", "execution"}:
        raise ValueError("POLAR_KV_CACHE_MODE must be 'layer' or 'execution'")
    return path, mode


def execution_cache_indices(path: list[int], mode: str) -> list[int]:
    if mode == "execution":
        return list(range(len(path)))
    if mode == "layer":
        return list(path)
    raise ValueError("mode must be 'layer' or 'execution'")
