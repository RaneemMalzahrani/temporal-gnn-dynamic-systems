# Temporal Graph Neural Networks for Dynamic Systems

A clean, reproducible implementation of a temporal graph neural network for
dynamic systems. This project is being rebuilt from scratch.

## Status

Project foundation created. The exact task, dataset, prediction target, and
evaluation protocol will be defined before model implementation.

## Design principles

- Split data chronologically to prevent future information from leaking into
  training.
- Establish a simple non-neural baseline before training a temporal GNN.
- Keep raw data immutable and outside version control.
- Make preprocessing, training, and evaluation reproducible from commands.
- Report metrics on a truly held-out time period.
- Test temporal ordering, graph construction, and negative sampling.

## Structure

```text
configs/              Experiment configuration
data/raw/             Original local data (not tracked by Git)
data/processed/       Generated datasets (not tracked by Git)
notebooks/            Exploration only
src/temporal_gnn/     Reusable project code
tests/                Automated correctness tests
```

## Next decision

Model and dependency choices will be made after confirming:

1. What one event or snapshot represents.
2. The node and edge definitions.
3. Whether time is continuous or snapshot-based.
4. The prediction target (links, nodes, graph state, or forecasting).
5. The official dataset split and evaluation metric.

