"""Command-line entry point for inspection, baseline, and TGN training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .baseline import RecencyFrequencyBaseline, evaluate_baseline, warm_baseline
from .compare import compare_results
from .config import load_config
from .data import describe_data, load_temporal_data


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def run_inspect(config, splits) -> None:
    data, train_data, val_data, test_data = splits
    _print(describe_data(data, train_data, val_data, test_data))


def run_reference(config, splits) -> None:
    _, train_data, val_data, test_data = splits
    model = RecencyFrequencyBaseline()
    warm_baseline(model, train_data)
    validation = evaluate_baseline(model, val_data, config.seed)
    test = evaluate_baseline(model, test_data, config.seed + 1)
    result = {"model": "recency_frequency_reference", "validation": validation,
              "test": test}
    _write_json(Path(config.output_dir) / "reference_metrics.json", result)
    _print(result)


def run_train(config, splits, smoke: bool) -> None:
    import torch

    from .experiment import TGNExperiment

    data, train_data, val_data, test_data = splits
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment = TGNExperiment(config, data, device)
    epochs = 1 if smoke else config.epochs
    best_ap = float("-inf")
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        loss = experiment.train_epoch(train_data)
        validation = experiment.evaluate(val_data, config.seed)
        val_ap = float(validation["overall"]["average_precision"])
        history.append({"epoch": epoch, "loss": loss, "validation": validation})
        print(f"epoch={epoch:03d} loss={loss:.4f} val_ap={val_ap:.4f}")
        if val_ap > best_ap:
            best_ap = val_ap
            best_state = experiment.snapshot_parameters()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    experiment.load_parameters(best_state)
    experiment.reset_state()
    experiment.replay(train_data)
    best_validation = experiment.evaluate(val_data, config.seed)
    test = experiment.evaluate(test_data, config.seed + 1)
    result = {
        "model": f"tgn_{config.model_variant}",
        "device": str(device),
        "trainable_parameters": experiment.trainable_parameter_count(),
        "best_validation": best_validation,
        "test": test,
        "history": history,
        "config": config.to_dict(),
    }
    output_dir = Path(config.output_dir)
    _write_json(output_dir / "tgn_metrics.json", result)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_dir / "best_model.pt")
    _print({"best_validation": best_validation, "test": test})


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal GNN experiments")
    parser.add_argument(
        "command", choices=("inspect", "reference", "train", "compare")
    )
    parser.add_argument("--config", default="configs/baseline_tgn.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--baseline-results",
        default="outputs/baseline_tgn/tgn_metrics.json",
    )
    parser.add_argument(
        "--enhanced-results",
        default="outputs/enhanced_tgn/tgn_metrics.json",
    )
    parser.add_argument("--out", default="outputs/comparison")
    args = parser.parse_args()

    if args.command == "compare":
        result = compare_results(
            args.baseline_results, args.enhanced_results, args.out
        )
        _print(result["summary"])
        return

    config = load_config(args.config)
    splits = load_temporal_data(config)
    if args.command == "inspect":
        run_inspect(config, splits)
    elif args.command == "reference":
        run_reference(config, splits)
    else:
        run_train(config, splits, args.smoke)


if __name__ == "__main__":
    main()
