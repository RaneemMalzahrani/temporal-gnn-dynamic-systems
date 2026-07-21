# Temporal Graph Neural Networks for Dynamic Systems

A from-scratch, leakage-safe study of temporal link prediction on a dynamic
interaction graph. The project implements a Temporal Graph Network (TGN) with
node memory, learned time encoding, recent-neighbor attention, and a link
predictor.

## Academic context

This project was developed for the **Graph Theory** course in **Level 3 of the
Master's program at Al-Baha University**.


## Assignment coverage

- **Temporal dataset:** JODIE Wikipedia.
- **Temporal link prediction:** predict whether a source and destination will
  interact at a given event time.
- **Memory modules:** persistent node states updated only after prediction.
- **Time encoding:** learned relative-time features in TGN memory and graph
  attention.
- **Time-aware analysis:** metrics across chronological test segments and
  endpoint inactivity gaps.
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

## Final results

Both models were trained under the same controlled setup. The simpler Baseline
TGN performed slightly better than the proposed enhancement:

| Model | Parameters | Validation AP | Test AP | Test ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline TGN | 260,301 | 0.9611 | **0.9491** | **0.9488** |
| Proposed/Enhanced TGN | 293,901 | 0.9571 | 0.9469 | 0.9423 |

The enhanced model changed Test AP by -0.0021 and ROC-AUC by -0.0065 while
using 12.9% more trainable parameters. This negative result is reported
honestly: the added message encoder and fusion gate did not improve this
dataset under the controlled protocol. Detailed time-aware results are in
`docs/RESULTS.md`.

## Project structure

```text
configs/              Reproducible experiment settings
data/                 Local downloads; excluded from Git
docs/EXPERIMENT.md    Research protocol and results template
src/temporal_gnn/     Data, baseline, model, training, and analysis code
tests/                Leakage and time-analysis tests
outputs/              Metrics and checkpoints; excluded from Git
```

## Requirements

- Python 3.10 or newer; Python 3.12 is recommended.
- About 1GB of free disk space for the raw and processed Wikipedia data.
- An internet connection for the first dataset download.
- A CUDA GPU is optional. Without CUDA, training runs on CPU and can take much
  longer. Apple Silicon currently uses CPU in this implementation.

## Quick start on the current Mac

The required ML libraries are already available in the existing Anaconda
Python 3.12 installation. Open Terminal and enter the project directory:

```bash
cd ~/Desktop/temporal-gnn-dynamic-systems
```

First, run the automated tests. This command does not need the Wikipedia data:

```bash
PYTHONPATH=src PYTHONWARNINGS=ignore \
  /opt/anaconda3/bin/python3.12 -m unittest discover -s tests -v
```

A successful setup ends with `OK`. The tests train both TGN variants on a tiny
synthetic event stream and check temporal leakage and fair comparison rules.

Display the available project commands:

```bash
PYTHONPATH=src /opt/anaconda3/bin/python3.12 \
  -m temporal_gnn.run --help
```

## Clean installation on another computer

Create an isolated Python 3.12 environment so package versions do not conflict:

```bash
conda create -n temporal-gnn python=3.12 -y
conda activate temporal-gnn
cd /path/to/temporal-gnn-dynamic-systems
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

After editable installation, use the shorter `temporal-gnn` commands shown
below. If the command is not found, use
`python -m temporal_gnn.run` in its place.

## Step 1 — download and inspect the dataset

Run this from the project directory:

```bash
temporal-gnn inspect --config configs/baseline_tgn.json
```

On the current Mac, the equivalent command without installing the project is:

```bash
PYTHONPATH=src PYTHONWARNINGS=ignore \
  /opt/anaconda3/bin/python3.12 -m temporal_gnn.run inspect \
  --config configs/baseline_tgn.json
```

PyTorch Geometric downloads the JODIE Wikipedia CSV on the first run and saves
it under `data/raw/JODIE/`. The raw download is roughly 533MB and is excluded
from Git. The command then prints event counts, feature dimensions, split
sizes, timestamp boundaries, and whether events are chronologically sorted.

If automatic downloading fails or leaves an incomplete file, download it via
HTTPS and rerun the inspect command:

```bash
mkdir -p data/raw/JODIE/wikipedia/raw
curl -L --fail --retry 5 \
  https://snap.stanford.edu/jodie/wikipedia.csv \
  --output data/raw/JODIE/wikipedia/raw/wikipedia.csv
```

Do not begin training unless `inspect` finishes successfully.

## Step 2 — optional simple reference

This is a non-neural reference only; it is not the required Baseline TGN:

```bash
temporal-gnn reference --config configs/baseline_tgn.json
```

Its metrics are written to
`outputs/baseline_tgn/reference_metrics.json`.

## Step 3 — quick smoke runs

Before full training, train each model for one epoch on a limited event subset:

```bash
temporal-gnn train --config configs/baseline_tgn.json --smoke
temporal-gnn train --config configs/enhanced_tgn.json --smoke
```

Smoke outputs are isolated under each model's `smoke/` directory, so they
cannot overwrite or be confused with final experiment results.

## Step 4 — train the Baseline Model

```bash
temporal-gnn train --config configs/baseline_tgn.json
```

This trains the standard TGN, selects the best epoch using validation Average
Precision, restores that checkpoint, replays the training and validation
history, and evaluates the untouched test period. Outputs:

- `outputs/baseline_tgn/best_model.pt`
- `outputs/baseline_tgn/tgn_metrics.json`

## Step 5 — train the Proposed/Enhanced Model

```bash
temporal-gnn train --config configs/enhanced_tgn.json
```

The enhanced model uses exactly the same experimental settings. Outputs:

- `outputs/enhanced_tgn/best_model.pt`
- `outputs/enhanced_tgn/tgn_metrics.json`

## Step 6 — compare both models

After both full training runs finish:

```bash
temporal-gnn compare
```

The command first verifies that all controlled settings match, then reports the
Baseline-to-Enhanced change in Average Precision, ROC-AUC, and parameter count.
It creates:

- `outputs/comparison/comparison.json`
- `outputs/comparison/comparison.md`

Copy the final values into the table in `docs/EXPERIMENT.md`, then discuss
chronological-bin and inactivity-gap performance.

## Complete command sequence

After installing the project and activating its environment, the full sequence
is:

```bash
cd /path/to/temporal-gnn-dynamic-systems
pytest
temporal-gnn inspect --config configs/baseline_tgn.json
temporal-gnn reference --config configs/baseline_tgn.json
temporal-gnn train --config configs/baseline_tgn.json --smoke
temporal-gnn train --config configs/enhanced_tgn.json --smoke
temporal-gnn train --config configs/baseline_tgn.json
temporal-gnn train --config configs/enhanced_tgn.json
temporal-gnn compare
```

## Common problems

### `temporal-gnn: command not found`

Activate the environment and reinstall the project:

```bash
conda activate temporal-gnn
python -m pip install -e ".[dev]"
```

Alternatively, run commands as `PYTHONPATH=src python -m temporal_gnn.run ...`.

### Incomplete Wikipedia download

The loader rejects suspiciously small files instead of silently training on
partial data. Move or delete the incomplete
`data/raw/JODIE/wikipedia/raw/wikipedia.csv`, then retry the HTTPS download.

### Training was interrupted

The current implementation saves the best model after training completes but
does not resume a partially completed run. Rerun the same training command.

### Comparison refuses to run

Both `tgn_metrics.json` files must come from full runs, and all controlled
settings must match. Do not compare smoke output against a full training run.

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
