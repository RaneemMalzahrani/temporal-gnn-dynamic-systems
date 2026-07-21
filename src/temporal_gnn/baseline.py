"""A causal recency-frequency baseline for temporal link prediction."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from .metrics import binary_metrics, chronological_performance


class RecencyFrequencyBaseline:
    """Score a link using only interactions observed before its timestamp."""

    def __init__(self) -> None:
        self.edge_count: Counter[tuple[int, int]] = Counter()
        self.dst_count: Counter[int] = Counter()
        self.edge_last_seen: dict[tuple[int, int], int] = {}

    def score(self, src: int, dst: int, timestamp: int) -> float:
        pair = (src, dst)
        frequency = math.log1p(self.edge_count[pair])
        popularity = 0.05 * math.log1p(self.dst_count[dst])
        last_seen = self.edge_last_seen.get(pair)
        recency = 0.0 if last_seen is None else 1.0 / (1.0 + timestamp - last_seen)
        return frequency + popularity + recency

    def update(self, src: int, dst: int, timestamp: int) -> None:
        pair = (src, dst)
        self.edge_count[pair] += 1
        self.dst_count[dst] += 1
        self.edge_last_seen[pair] = timestamp


def _events(data: Any):
    for src, dst, timestamp in zip(data.src.tolist(), data.dst.tolist(),
                                   data.t.tolist()):
        yield int(src), int(dst), int(timestamp)


def warm_baseline(model: RecencyFrequencyBaseline, data: Any) -> None:
    for src, dst, timestamp in _events(data):
        model.update(src, dst, timestamp)


def evaluate_baseline(model: RecencyFrequencyBaseline, data: Any, seed: int):
    rng = random.Random(seed)
    min_dst, max_dst = int(data.dst.min()), int(data.dst.max())
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for src, dst, timestamp in _events(data):
        neg_dst = rng.randint(min_dst, max_dst)
        while neg_dst == dst:
            neg_dst = rng.randint(min_dst, max_dst)
        positive_scores.append(model.score(src, dst, timestamp))
        negative_scores.append(model.score(src, neg_dst, timestamp))
        model.update(src, dst, timestamp)
    overall = binary_metrics(
        [1] * len(positive_scores) + [0] * len(negative_scores),
        positive_scores + negative_scores,
    )
    return {
        "overall": overall,
        "chronological_bins": chronological_performance(
            positive_scores, negative_scores
        ),
    }
