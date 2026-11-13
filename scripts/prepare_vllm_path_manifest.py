#!/usr/bin/env python3
"""Prepare unique tokenized prompts and a per-request path manifest for vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from polar_vllm.path_manifest import build_manifest


def load_rows(filename: Path) -> list[dict]:
    payload = json.loads(filename.read_text(encoding="utf-8"))
    rows = payload.get("paths", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("input must be a path array or an object containing 'paths'")
    normalized = []
    for index, row in enumerate(rows):
        if isinstance(row, list):
            normalized.append({"request_id": f"req-{index:06d}", "path": row})
        elif isinstance(row, dict):
            normalized.append(row)
        else:
            raise ValueError(f"row {index} must be a path array or object")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, help="JSON path workload")
    parser.add_argument("tokenizer", help="local model/tokenizer path")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-depth", type=int, default=36)
    args = parser.parse_args()

    rows = load_rows(args.paths)
    if not rows:
        parser.error("path workload is empty")
    for row in rows:
        path = row.get("path")
        if not isinstance(path, list) or not path:
            parser.error(f"{row.get('request_id', '?')}: missing non-empty path")
        if any(type(layer) is not int or not 0 <= layer < args.max_depth for layer in path):
            parser.error(f"{row.get('request_id', '?')}: layer index outside model depth")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    manifest, prompts = build_manifest(rows, tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "path_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "prompts.jsonl").open("w", encoding="utf-8") as stream:
        for row in prompts:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[done] requests={len(prompts)} manifest={args.output_dir / 'path_manifest.json'}")
    print(f"[done] prompts={args.output_dir / 'prompts.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
