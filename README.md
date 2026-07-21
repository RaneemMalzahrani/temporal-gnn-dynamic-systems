# Temporal Graph Neural Networks for Dynamic Systems

A from-scratch, leakage-safe study of temporal link prediction on a dynamic
interaction graph. The project implements a Temporal Graph Network (TGN) with
node memory, learned time encoding, recent-neighbor attention, and a link
predictor.

## Assignment coverage

- **Temporal dataset:** JODIE Wikipedia.
- **Temporal link prediction:** predict whether a source and destination will
  interact at a given event time.
- **Memory modules:** persistent node states updated only after prediction.
- **Time encoding:** learned relative-time features in TGN memory and graph
  attention.
- **Time-aware analysis:** metrics across chronological test segments and
  node-pair inactivity gaps.
- **Controlled comparison:** a standard Baseline TGN and a Proposed/Enhanced
  TGN trained under identical conditions.

## The two required models

### 1. Baseline Model — standard TGN

The baseline follows the standard PyTorch Geometric TGN structure:

- identity message function (concatenation without a learned message encoder),
- last-message aggregation,
- GRU node memory,
- learned relative-time encoding,
- temporal neighbor attention, and
- an MLP link predictor.

### 2. Proposed Model — enhanced TGN

The proposed model keeps the same data, split, dimensions, seed, optimizer,
negative sampling, and evaluation protocol. It changes only two components:

1. A learned nonlinear message encoder filters and compresses source memory,
   destination memory, raw event features, and time encoding before updating
   node memory.
2. A gated residual fusion layer learns how much to trust long-term node memory
   versus the recent time-aware neighborhood, followed by layer normalization.

This controlled design lets the final score difference be attributed to the
proposed architecture rather than different training conditions. A simple
recency-frequency method remains available only as an optional reference; it
is not one of the two main models.

## Why this version is reliable

The train, validation, and test sets are consecutive periods—not random rows.
The code rejects unsorted or overlapping splits, never shuffles event loaders,
scores each event before revealing it to memory, and reconstructs the temporal
state by replaying past events before final testing.

## Project structure

```text
configs/              Reproducible experiment settings
data/                 Local downloads; excluded from Git
docs/EXPERIMENT.md    Research protocol and results template
src/temporal_gnn/     Data, baseline, model, training, and analysis code
tests/                Leakage and time-analysis tests
outputs/              Metrics and checkpoints; excluded from Git
```

## Installation

Python 3.10+ is required. A GPU is helpful but not required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run the project

The first command downloads the official Wikipedia data (roughly 533MB raw).
If the Stanford server is slow, run it on Colab or another stable connection.

```bash
# Inspect the data and verify chronological boundaries.
temporal-gnn inspect --config configs/baseline_tgn.json

# Optional non-neural reference (not the main baseline model).
temporal-gnn reference --config configs/baseline_tgn.json

# Train model 1: Baseline TGN.
temporal-gnn train --config configs/baseline_tgn.json

# Train model 2: Proposed/Enhanced TGN.
temporal-gnn train --config configs/enhanced_tgn.json

# Verify that the settings match and generate comparison.json + comparison.md.
temporal-gnn compare
```

For a one-epoch integration check:

```bash
temporal-gnn train --config configs/baseline_tgn.json --smoke
temporal-gnn train --config configs/enhanced_tgn.json --smoke
```

Metrics are saved as JSON under `outputs/`. Copy the final values into the
results table in `docs/EXPERIMENT.md` and discuss both chronological and
inactivity-gap trends.

## Tests

```bash
pytest
```

The tests cover both model variants, chronological ordering, split leakage,
time bins, inactivity-gap analysis, and rejection of unfair comparisons.

## References

- [Temporal Graph Networks paper](https://arxiv.org/abs/2006.10637)
- [PyTorch Geometric TGN example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py)
- [JODIEDataset documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.JODIEDataset.html)
- [TGNMemory documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.TGNMemory.html)
- [TemporalData documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.TemporalData.html)
