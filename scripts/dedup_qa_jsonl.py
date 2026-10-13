"""Remove duplicate QA IDs only when their evaluation content agrees."""
import argparse
import json
from pathlib import Path


def deduplicate(rows):
    kept, seen, duplicates, conflicts = [], {}, [], []
    for line, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {line} is not an object.")
        raw_id = row.get("task_id", row.get("id"))
        if raw_id is None or not str(raw_id).strip():
            raise ValueError(f"Row {line} has no non-empty task_id/id.")
        task_id = str(raw_id)
        answers = row.get("answers")
        if not isinstance(answers, list) or not answers or any(
                not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError(f"Row {line}: expected converted answers (non-empty string list).")
        # Ignore metadata and alias order, but require identical question/context/choices.
        content = (row.get("question"), row.get("context", ""),
                   json.dumps(row.get("choices", {}), sort_keys=True, ensure_ascii=False),
                   tuple(sorted(set(answers))))
        if task_id not in seen:
            seen[task_id] = (line, content)
            kept.append(row)
        elif seen[task_id][1] == content:
            duplicates.append({"task_id": task_id, "first_row": seen[task_id][0],
                               "duplicate_row": line})
        else:
            conflicts.append({"task_id": task_id, "first_row": seen[task_id][0],
                              "conflicting_row": line})
    return kept, duplicates, conflicts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    if source.resolve() == output.resolve():
        parser.error("Use a separate output file; do not overwrite the original.")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    with source.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    kept, duplicates, conflicts = deduplicate(rows)
    report = {"input_rows": len(rows), "unique_ids": len(kept),
              "duplicate_rows": len(duplicates), "conflict_rows": len(conflicts),
              "duplicates": duplicates, "conflicts": conflicts}
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(str(output) + ".dedup_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[audit] input={len(rows)} unique_ids={len(kept)} "
          f"duplicate_rows={len(duplicates)} conflicts={len(conflicts)}")
    print(f"[report] {report_path}")
    if conflicts:
        raise SystemExit("Conflicting content for the same ID. Inspect the report; no dataset written.")
    tmp = Path(str(output) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(output)
    print(f"[done] {len(kept)} tasks -> {output}")


if __name__ == "__main__":
    main()
