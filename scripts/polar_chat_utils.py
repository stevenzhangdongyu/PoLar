"""CPU-only helpers for interactive POLAR conversations."""
from qa_eval_utils import extract_answer, score_answer


def chat_prompt(tokenizer, messages, model_path):
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Interactive chat requires a tokenizer with a chat template.")
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_path.lower():
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def predictor_text(messages, mode):
    if mode == "question":
        return messages[-1]["content"]
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def unique_paths(beams, depth, convert, k):
    rows, seen = [], set()
    for actions, score in beams:
        path = list(convert(actions, depth))
        if not path or tuple(path) in seen:
            continue
        seen.add(tuple(path))
        if any(not isinstance(i, int) or i < 0 or i >= depth for i in path):
            raise ValueError("Decoder returned invalid layer indices.")
        rows.append({"path": path, "score": float(score), "segment_actions": [
            {"operation": op, "segment_length": size, "count": count}
            for op, size, count in actions]})
        if len(rows) == k:
            break
    return rows


def grade_reference(turn, references):
    """Short-answer diagnostic only; never feeds references into model prompts."""
    if not isinstance(references, list) or not references or any(
            not isinstance(a, str) or not a.strip() for a in references):
        raise ValueError('Expected a non-empty JSON string array, e.g. ["Paris", "巴黎"].')
    def grade(candidate):
        answer = extract_answer(candidate["completion"], "short_answer", {})
        return {"extracted_answer": answer, **score_answer(answer, references)}
    base = grade(turn["base"])
    candidates = [grade(c) for c in turn["candidates"]]
    return {"method": "custom_short_answer_EM_F1", "references": references,
            "base": base, "candidates": candidates, "actual_k": len(candidates),
            "best_of_k_correct": any(c["exact_match"] for c in candidates),
            "note": "Reference-assisted diagnostic, not semantic judging or unbiased pass@k."}


def grade_manual(turn, labels):
    if not isinstance(labels, list) or len(labels) != len(turn["candidates"]) or any(
            type(x) is not bool for x in labels):
        raise ValueError("Supply one JSON true/false per POLAR candidate.")
    return {"method": "user_manual", "candidates_correct": labels,
            "actual_k": len(labels), "best_of_k_correct": any(labels)}


def chat_summary(turns):
    result = {"turns": len(turns), "metric_note":
              "Conversation turns may be correlated; best-of-k is human/reference assisted, "
              "not unbiased stochastic pass@k. Ungraded turns are excluded."}
    for key in ("reference_grade", "manual_grade"):
        graded = [t[key] for t in turns if key in t]
        result[key] = {
            "graded_turns": len(graded),
            "best_of_actual_k_rate": (
                sum(g["best_of_k_correct"] for g in graded) / len(graded) if graded else None),
            "actual_k_counts": {str(k): sum(g["actual_k"] == k for g in graded)
                                for k in sorted({g["actual_k"] for g in graded})},
        }
    return result
