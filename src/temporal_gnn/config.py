"""Configuration loading with validation and path resolution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "Wikipedia"
    data_root: str = "data/raw/JODIE"
    output_dir: str = "outputs/wikipedia"
    seed: int = 42
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 200
    neighbor_size: int = 10
    memory_dim: int = 100
    time_dim: int = 100
    embedding_dim: int = 100
    model_variant: str = "baseline"
    learning_rate: float = 1e-4
    epochs: int = 20
    patience: int = 5

    def validate(self) -> None:
        if self.dataset.lower() not in {"wikipedia", "reddit"}:
            raise ValueError("dataset must be Wikipedia or Reddit")
        if self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("validation and test ratios must be positive")
        if self.val_ratio + self.test_ratio >= 1:
            raise ValueError("validation + test ratios must be below 1")
        if self.model_variant not in {"baseline", "enhanced"}:
            raise ValueError("model_variant must be baseline or enhanced")
        for name in ("batch_size", "neighbor_size", "memory_dim", "time_dim",
                     "embedding_dim", "epochs", "patience"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = ExperimentConfig(**json.load(handle))
    config.validate()

    project_root = path.parent.parent
    values = config.to_dict()
    for key in ("data_root", "output_dir"):
        candidate = Path(str(values[key]))
        if not candidate.is_absolute():
            values[key] = str((project_root / candidate).resolve())
    return ExperimentConfig(**values)
