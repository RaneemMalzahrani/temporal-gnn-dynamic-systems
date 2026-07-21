"""Dataset loading and chronological split validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .validation import assert_temporal_boundaries


def _timestamp_sample(values: Any) -> list[int]:
    return [int(value) for value in values.detach().cpu().tolist()]


def load_temporal_data(config: ExperimentConfig):
    from torch_geometric.datasets import JODIEDataset

    raw_path = (
        Path(config.data_root)
        / config.dataset.lower()
        / "raw"
        / f"{config.dataset.lower()}.csv"
    )
    if (
        config.dataset.lower() == "wikipedia"
        and raw_path.exists()
        and raw_path.stat().st_size < 500_000_000
    ):
        raise RuntimeError(
            f"incomplete Wikipedia download: {raw_path} is only "
            f"{raw_path.stat().st_size:,} bytes; remove it and retry"
        )
    dataset = JODIEDataset(config.data_root, name=config.dataset)
    data = dataset[0]
    train_data, val_data, test_data = data.train_val_test_split(
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    assert_temporal_boundaries(
        _timestamp_sample(train_data.t),
        _timestamp_sample(val_data.t),
        _timestamp_sample(test_data.t),
    )
    return data, train_data, val_data, test_data


def describe_data(data: Any, train_data: Any, val_data: Any,
                  test_data: Any) -> dict[str, object]:
    return {
        "num_nodes": int(data.num_nodes),
        "num_events": int(data.num_events),
        "message_features": int(data.msg.size(-1)),
        "train_events": int(train_data.num_events),
        "validation_events": int(val_data.num_events),
        "test_events": int(test_data.num_events),
        "first_timestamp": int(data.t[0]),
        "last_timestamp": int(data.t[-1]),
        "chronologically_sorted": bool((data.t[1:] >= data.t[:-1]).all()),
    }
