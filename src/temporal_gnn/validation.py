"""Correctness checks for chronological temporal-graph experiments."""

from __future__ import annotations

from collections.abc import Sequence


def assert_non_decreasing(timestamps: Sequence[int | float], name: str) -> None:
    if not timestamps:
        raise ValueError(f"{name} is empty")
    for left, right in zip(timestamps, timestamps[1:]):
        if left > right:
            raise ValueError(f"{name} is not sorted chronologically")


def assert_temporal_boundaries(
    train_t: Sequence[int | float],
    val_t: Sequence[int | float],
    test_t: Sequence[int | float],
) -> None:
    """Reject shuffled splits and future-to-past leakage."""
    for name, timestamps in (("train", train_t), ("validation", val_t),
                             ("test", test_t)):
        assert_non_decreasing(timestamps, name)
    if train_t[-1] > val_t[0]:
        raise ValueError("training events overlap the validation future")
    if val_t[-1] > test_t[0]:
        raise ValueError("validation events overlap the test future")


def chronological_bins(size: int, n_bins: int = 4) -> list[int]:
    """Assign ordered observations to near-equal chronological bins."""
    if size <= 0:
        raise ValueError("size must be positive")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    return [min(n_bins - 1, index * n_bins // size) for index in range(size)]
