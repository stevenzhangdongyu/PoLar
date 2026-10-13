"""Standard-library-only QA inputs, metrics and result persistence."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import string
import unicodedata


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_tasks(path, mode="short_answer"):
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = ([json.loads(x) for x in f if x.strip()]
                if path.suffix.lower() == ".jsonl" else json.load(f))
    if isinstance(data, dict):
        data = data.get("tasks", data.get("samples", data.get("data", list(data.values()))))
    if not isinstance(data, list) or not data:
        raise ValueError("QA input must be a non-empty JSON/JSONL task list.")
    rows, ids = [], set()
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} is not an object.")
        question = row.get("question", row.get("query", ""))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Row {i} has no non-empty question/query.")
        task_id = str(row.get("task_id", row.get("id", f"qa/{i}")))
        if task_id in ids:
            raise ValueError(f"Duplicate task_id: {task_id}")
        ids.add(task_id)
        answers = row.get("answers", row.get("answer", row.get("gt_ans", row.get("answerKey"))))
        if isinstance(answers, dict):
            # Flat SQuAD rows and TriviaQA-style answer aliases.
            value = answers.get("value")
            aliases = answers.get("aliases", answers.get("text", []))
            answers = ([aliases] if isinstance(aliases, str) else list(aliases))
            if value is not None:
                answers.append(value)
        if isinstance(answers, (str, int, float)):
            answers = [answers]
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"{task_id}: supply non-empty answers/answer/gt_ans.")
        if any(not isinstance(a, (str, int, float)) or not str(a).strip() for a in answers):
            raise ValueError(f"{task_id}: references must be non-empty strings/numbers.")
        answers = list(dict.fromkeys(str(a) for a in answers))
        context = row.get("context", "")
        if not isinstance(context, str):
            raise ValueError(f"{task_id}: context must be text.")
        choices = row.get("choices", {})
        if mode == "multiple_choice":
            if isinstance(choices, list):
                choices = {chr(65 + j): v for j, v in enumerate(choices)}
            elif isinstance(choices, dict) and "label" in choices and "text" in choices:
                if len(choices["label"]) != len(choices["text"]):
                    raise ValueError(f"{task_id}: mismatched choice labels/text.")
                choices = dict(zip(choices["label"], choices["text"]))
            if not isinstance(choices, dict) or len(choices) < 2:
                raise ValueError(f"{task_id}: multiple_choice needs choices.")
            choices = {str(k).upper(): str(v) for k, v in choices.items()}
            if any(not re.fullmatch(r"[A-Z]", k) for k in choices):
                raise ValueError(f"{task_id}: use single-letter choice labels A-Z.")
            answers = [a.strip().upper() for a in answers]
            if any(a not in choices for a in answers):
                raise ValueError(f"{task_id}: answers must be choice labels, not choice text.")
        else:
            choices = {}
        rows.append({"task_id": task_id, "question": question.strip(),
                     "context": context, "answers": answers, "choices": choices})
    return rows


def normalize_answer(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = "".join(c for c in text if c not in string.punctuation
                   and not unicodedata.category(c).startswith("P"))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def tokens(text):
    # English words/numbers + individual CJK characters; no external tokenizer.
    return re.findall(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", normalize_answer(text), re.UNICODE)


def score_answer(prediction, answers, mode="short_answer"):
    if mode == "multiple_choice":
        em = float(prediction in answers)
        return {"exact_match": em, "f1": em, "matched_reference": prediction if em else None}
    pred_norm = normalize_answer(prediction)
    pred_tokens = tokens(prediction)
    best_em, best_f1, best_ref = 0.0, 0.0, None
    for answer in answers:
        em = float(pred_norm == normalize_answer(answer))
        gold_tokens = tokens(answer)
        overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if not pred_tokens or not gold_tokens:
            f1 = float(pred_tokens == gold_tokens)
        elif not overlap:
            f1 = 0.0
        else:
            precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        if best_ref is None or (em, f1) > (best_em, best_f1):
            best_ref = answer
        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)
    return {"exact_match": best_em, "f1": best_f1, "matched_reference": best_ref}


def extract_answer(text, mode="short_answer", choices=None):
    # This is deterministic formatting cleanup, not an LLM judge.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        return ""
    if mode == "multiple_choice":
        pattern = r"^(?:(?:final\s+answer|answer|答案|最终答案)\s*[:：]\s*)?\(?([A-Z])\)?[.。]?$"
        matches = [re.fullmatch(pattern, x, re.I) for x in lines]
        labels = {m.group(1).upper() for m in matches if m}
        if len(labels) == 1:
            label = next(iter(labels))
            return label if label in (choices or {}) else ""
        return ""  # ambiguous/unparseable answers fail, not guessed
    text = re.sub(r"^(?:final\s+answer|answer|答案|最终答案)\s*[:：]\s*", "", text, flags=re.I)
    return text.strip()


def model_prompt(task, mode="short_answer"):
    prefix = ("Answer the question with only a short answer. Do not explain."
              if mode == "short_answer" else
              "Answer the multiple-choice question with ONLY the option letter. Do not explain.")
    context = f"\n\nContext:\n{task['context']}" if task["context"] else ""
    choices = ("\n" + "\n".join(f"{k}. {v}" for k, v in task["choices"].items())
               if mode == "multiple_choice" else "")
    return f"{prefix}{context}\n\nQuestion:\n{task['question']}{choices}\n\nAnswer:"


def describe_path(path, depth):
    counts = Counter(path)
    seen, visits = set(), []
    for i, layer in enumerate(path):
        visits.append({"visit_index": i, "layer": layer,
                       "operation": "repeat" if layer in seen else "execute"})
        seen.add(layer)
    return {"path": path, "unique_depth": len(counts),
            "total_layer_executions": len(path),
            "skipped_layers": [i for i in range(depth) if i not in counts],
            "repeated_layers": sorted(i for i, n in counts.items() if n > 1),
            "layer_visit_counts": {str(i): n for i, n in sorted(counts.items())},
            "operations": visits}


def summarize(rows):
    """Top-k is oracle best-of-paths, NOT a deployable answer-selection rule."""
    summary = {"completed_tasks": len(rows)}
    for method in ("base", "polar_top1", "polar_topk_oracle"):
        selected = []
        for row in rows:
            candidates = row["candidates"]
            if method == "base":
                selected.append(row["base"])
            elif method == "polar_top1":
                selected.append(candidates[0] if candidates else {"exact_match": 0., "f1": 0.})
            else:
                selected.append({
                    "exact_match": max((c["exact_match"] for c in candidates), default=0.),
                    "f1": max((c["f1"] for c in candidates), default=0.),
                })
        summary[method] = {
            "correct": sum(int(s["exact_match"]) for s in selected),
            "exact_match": sum(s["exact_match"] for s in selected) / len(rows) if rows else None,
            "f1": sum(s["f1"] for s in selected) / len(rows) if rows else None,
        }
    summary["metric_note"] = (
        "Custom normalized EM/token F1; top-k oracle uses reference answers, "
        "not unbiased stochastic pass@k or a deployable selection policy.")
    candidates = [r["candidates"][0] for r in rows if r["candidates"]]
    for field in ("unique_depth", "total_layer_executions"):
        summary["polar_top1_mean_" + field] = (
            sum(c[field] for c in candidates) / len(candidates) if candidates else None)
    summary["no_candidate_tasks"] = sum(not r["candidates"] for r in rows)
    return summary
