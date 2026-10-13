"""Reference-free selectors and paired metrics shared by QA/code evaluation."""
from collections import Counter


def select_candidate(base, candidates, strategy="top1"):
    if strategy == "nonbase_top1":
        candidates = [c for c in candidates if c["path"] != base["path"]]
    if not candidates:
        return base, "base_fallback"
    if strategy == "majority":
        keys = [c.get("normalized_answer", c.get("extracted_code", "")) for c in candidates]
        counts = Counter(k for k in keys if k)
        if not counts:
            return base, "base_fallback"
        # Ties use original predictor ordering; never consult reference scores.
        best = max(counts.values())
        return next(c for c, k in zip(candidates, keys) if counts[k] == best), strategy
    return candidates[0], strategy


def paired_metrics(base_ok, selected_ok):
    n = len(base_ok)
    wrong = sum(not x for x in base_ok)
    correct = n - wrong
    repaired = sum(not b and s for b, s in zip(base_ok, selected_ok))
    degraded = sum(b and not s for b, s in zip(base_ok, selected_ok))
    return {
        "num_tasks": n, "base_correct": correct, "base_wrong": wrong,
        "repair_count": repaired, "degradation_count": degraded,
        "repair_rate_all": repaired / n if n else None,
        "degradation_rate_all": degraded / n if n else None,
        "repair_rate_given_base_wrong": repaired / wrong if wrong else None,
        "degradation_rate_given_base_correct": degraded / correct if correct else None,
        "final_accuracy": sum(selected_ok) / n if n else None,
        "net_accuracy_change": (repaired - degraded) / n if n else None,
    }


def selection_summary(rows, correctness):
    # Old results without a selector/timing remain readable, without invented latency.
    available = [r for r in rows if "selected" in r]
    result = paired_metrics(
        [correctness(r["base"]) for r in available],
        [correctness(r["selected"]) for r in available])
    result["fallback_count"] = sum(r.get("selection_reason") == "base_fallback" for r in available)
    result["strategies"] = sorted({r.get("selection_strategy", "unknown") for r in available})
    timed = [r["timings"] for r in rows if "timings" in r]
    result["timed_tasks"] = len(timed)
    result["timing_totals_seconds"] = {
        key: sum(t[key] for t in timed) for key in timed[0]
    } if timed else {}
    result["mean_end_to_end_seconds"] = (
        sum(t["end_to_end_seconds"] for t in timed) / len(timed) if timed else None)
    result["timing_note"] = (
        "Summed per-task worker times, not parallel wall-clock. Excludes "
        "model loading and final result writes. end_to_end includes trace IO/reference grading; "
        "selected_pipeline sums generation/predictor/selection only. No learned verifier installed.")
    return result


def sampling_summary(rows, correctness):
    requested = [r for r in rows if r.get("requested_sampling_k", 0) > 0]
    paired = [r for r in rows if r.get("base_samples") and r.get("candidates")]
    if any(len(r["base_samples"]) != len(r["candidates"]) for r in paired):
        raise ValueError("Sampling comparison requires matched actual candidate counts.")
    return {
        "num_tasks": len(paired),
        "requested_k": sorted({r["requested_sampling_k"] for r in requested}),
        "tasks_with_fewer_than_requested_k": sum(
            len(r["candidates"]) < r["requested_sampling_k"] for r in requested),
        "base_sampling_best_of_actual_k": (
            sum(any(correctness(c) for c in r["base_samples"]) for r in paired) / len(paired)
            if paired else None),
        "polar_greedy_best_of_actual_k": (
            sum(any(correctness(c) for c in r["candidates"]) for r in paired) / len(paired)
            if paired else None),
        "actual_k_counts": dict(Counter(str(len(r["candidates"])) for r in paired)),
        "note": "Reference-assisted best-of-k, not deployable selection; k counts are matched per task.",
    }
