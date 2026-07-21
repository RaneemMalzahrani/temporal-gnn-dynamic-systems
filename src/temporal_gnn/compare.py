"""Fair, reproducible comparison of baseline and enhanced TGN results."""

from __future__ import annotations

import json
from pathlib import Path


FAIRNESS_KEYS = (
    "dataset",
    "seed",
    "val_ratio",
    "test_ratio",
    "batch_size",
    "neighbor_size",
    "memory_dim",
    "time_dim",
    "embedding_dim",
    "learning_rate",
    "epochs",
    "patience",
)


def _read(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _assert_fair(baseline: dict, enhanced: dict) -> None:
    differences = [
        key
        for key in FAIRNESS_KEYS
        if baseline["config"].get(key) != enhanced["config"].get(key)
    ]
    if differences:
        raise ValueError(
            "comparison is not controlled; settings differ: "
            + ", ".join(differences)
        )
    if baseline["config"].get("model_variant") != "baseline":
        raise ValueError("the baseline result is not model_variant=baseline")
    if enhanced["config"].get("model_variant") != "enhanced":
        raise ValueError("the enhanced result is not model_variant=enhanced")


def compare_results(baseline_path: str | Path, enhanced_path: str | Path,
                    output_dir: str | Path) -> dict[str, object]:
    baseline = _read(baseline_path)
    enhanced = _read(enhanced_path)
    _assert_fair(baseline, enhanced)

    baseline_test = baseline["test"]["overall"]
    enhanced_test = enhanced["test"]["overall"]
    summary = {
        "baseline": {
            **baseline_test,
            "trainable_parameters": baseline["trainable_parameters"],
        },
        "enhanced": {
            **enhanced_test,
            "trainable_parameters": enhanced["trainable_parameters"],
        },
        "delta": {
            metric: enhanced_test[metric] - baseline_test[metric]
            for metric in ("average_precision", "roc_auc")
        },
    }
    summary["delta"]["trainable_parameters"] = (
        enhanced["trainable_parameters"] - baseline["trainable_parameters"]
    )
    result = {
        "controlled_settings": {
            key: baseline["config"][key] for key in FAIRNESS_KEYS
        },
        "summary": summary,
        "baseline_chronological_bins": baseline["test"]["chronological_bins"],
        "enhanced_chronological_bins": enhanced["test"]["chronological_bins"],
        "baseline_inactivity_bins": baseline["test"]["inactivity_gap_bins"],
        "enhanced_inactivity_bins": enhanced["test"]["inactivity_gap_bins"],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    baseline_ap = baseline_test["average_precision"]
    baseline_auc = baseline_test["roc_auc"]
    enhanced_ap = enhanced_test["average_precision"]
    enhanced_auc = enhanced_test["roc_auc"]
    delta_ap = summary["delta"]["average_precision"]
    delta_auc = summary["delta"]["roc_auc"]
    baseline_parameters = baseline["trainable_parameters"]
    enhanced_parameters = enhanced["trainable_parameters"]
    delta_parameters = summary["delta"]["trainable_parameters"]
    markdown = f"""# Baseline vs Enhanced TGN

| Model | Parameters | Test AP | Test ROC-AUC |
|---|---:|---:|---:|
| Baseline TGN | {baseline_parameters:,} | {baseline_ap:.4f} | {baseline_auc:.4f} |
| Enhanced TGN | {enhanced_parameters:,} | {enhanced_ap:.4f} | {enhanced_auc:.4f} |
| Difference | {delta_parameters:+,} | {delta_ap:+.4f} | {delta_auc:+.4f} |

The comparison uses identical data, chronological splits, negative sampling
seeds, dimensions, optimization settings, and early-stopping rules.
"""
    (output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    return result
