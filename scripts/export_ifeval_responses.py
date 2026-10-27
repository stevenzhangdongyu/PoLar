#!/usr/bin/env python3
"""Export Base/POLAR responses from merged_results.json into IFEval-style JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[write] {path} rows={len(rows)}")


def response_row(row, response, method, rank):
    # Official IFEval primarily needs prompt/response. Keep extra fields for traceability;
    # most JSONL readers will ignore unknown fields.
    return {
        "key": row.get("task_id"),
        "prompt": row["prompt"],
        "response": response,
        "method": method,
        "candidate_rank": rank,
        "instruction_id_list": row.get("instruction_id_list", []),
        "kwargs": row.get("kwargs", []),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged_results", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()
    payload = json.loads(Path(args.merged_results).read_text(encoding="utf-8"))
    rows = payload["results"]
    out = Path(args.output_dir)
    base = [response_row(r, r["base"]["completion"], "base", 0) for r in rows]
    write_jsonl(out / "base.responses.jsonl", base)
    for rank in range(1, args.top_k + 1):
        selected = []
        for r in rows:
            candidates = r.get("candidates", [])
            if len(candidates) >= rank:
                selected.append(response_row(r, candidates[rank - 1]["completion"], "polar", rank))
        write_jsonl(out / f"polar_rank{rank}.responses.jsonl", selected)
    # Convenience file containing all generated responses, not directly suitable for official
    # IFEval because it has multiple responses per prompt.
    all_rows = []
    for r in rows:
        all_rows.append(response_row(r, r["base"]["completion"], "base", 0))
        for rank, c in enumerate(r.get("candidates", [])[: args.top_k], 1):
            all_rows.append(response_row(r, c["completion"], "polar", rank))
    write_jsonl(out / "all_candidates.not_for_official_eval.jsonl", all_rows)


if __name__ == "__main__":
    main()
