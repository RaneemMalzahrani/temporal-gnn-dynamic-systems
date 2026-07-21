"""Global and time-aware link-prediction metrics."""

from __future__ import annotations

from typing import Iterable

from .validation import chronological_bins


def binary_metrics(labels: Iterable[int], scores: Iterable[float]) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = list(labels)
    y_score = list(scores)
    if len(set(y_true)) < 2:
        return {"average_precision": float("nan"), "roc_auc": float("nan")}
    return {
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def chronological_performance(
    positive_scores: list[float],
    negative_scores: list[float],
    n_bins: int = 4,
) -> list[dict[str, float | int]]:
    if len(positive_scores) != len(negative_scores):
        raise ValueError("positive and negative scores must align by event")
    if not positive_scores:
        raise ValueError("at least one event is required")
    effective_bins = min(n_bins, len(positive_scores))
    assignments = chronological_bins(len(positive_scores), effective_bins)
    result = []
    for bin_id in range(effective_bins):
        indices = [i for i, value in enumerate(assignments) if value == bin_id]
        pos = [positive_scores[i] for i in indices]
        neg = [negative_scores[i] for i in indices]
        values = binary_metrics([1] * len(pos) + [0] * len(neg), pos + neg)
        result.append({"time_bin": bin_id + 1, "events": len(indices), **values})
    return result


def inactivity_gap_performance(
    positive_scores: list[float],
    negative_scores: list[float],
    inactivity_gaps: list[float],
    n_bins: int = 4,
) -> list[dict[str, float | int]]:
    """Measure performance from recently active to long-inactive node pairs."""
    size = len(positive_scores)
    if len(negative_scores) != size or len(inactivity_gaps) != size:
        raise ValueError("scores and inactivity gaps must align by event")
    if not positive_scores:
        raise ValueError("at least one event is required")
    effective_bins = min(n_bins, size)
    ranked = sorted(range(size), key=inactivity_gaps.__getitem__)
    rank_bins = chronological_bins(size, effective_bins)
    assignment = {
        event_index: rank_bins[rank]
        for rank, event_index in enumerate(ranked)
    }
    result = []
    for bin_id in range(effective_bins):
        indices = [i for i in range(size) if assignment[i] == bin_id]
        pos = [positive_scores[i] for i in indices]
        neg = [negative_scores[i] for i in indices]
        gaps = [inactivity_gaps[i] for i in indices]
        values = binary_metrics([1] * len(pos) + [0] * len(neg), pos + neg)
        result.append({
            "gap_bin": bin_id + 1,
            "events": len(indices),
            "min_gap": float(min(gaps)),
            "max_gap": float(max(gaps)),
            **values,
        })
    return result
