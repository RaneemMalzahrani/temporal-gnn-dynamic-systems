# Experiment protocol

## Research question

Can a Temporal Graph Network predict the next observed user-page interaction,
and how does its performance change over time and with interaction inactivity?

## Dataset

The primary dataset is the JODIE Wikipedia temporal interaction network. Each
event contains a source node, destination node, timestamp, and 172-dimensional
message vector. PyTorch Geometric downloads and represents it as a continuous
event stream.

## Models

1. **Recency-frequency baseline:** uses only edge history and destination
   popularity available before each prediction.
2. **TGN:** node memory, last-message aggregation, learned time encoding,
   attention over recent temporal neighbors, and a neural link predictor.
3. **Time-feature ablation:** the same TGN with relative-time features removed
   from graph attention. TGN memory is retained so the ablation isolates the
   graph-attention time features.

## Leakage controls

- Events remain sorted; no temporal loader is shuffled.
- The first 70% of events train the model, the next 15% validate it, and the
  final 15% form the untouched test period.
- Validation selects the checkpoint. Test metrics are computed once after
  restoring that checkpoint and replaying train and validation history.
- A prediction is scored before its ground-truth event updates memory.
- Negative destinations are generated deterministically for repeatability.

## Metrics and time-aware analysis

- Average Precision (primary)
- ROC-AUC
- Metrics in four chronological test bins
- Metrics in four node-pair inactivity-gap bins

## Results table

Fill this table from the generated JSON files after running all experiments.

| Model | Validation AP | Test AP | Test ROC-AUC |
|---|---:|---:|---:|
| Recency-frequency baseline | TBD | TBD | TBD |
| TGN | TBD | TBD | TBD |
| TGN without graph time features | TBD | TBD | TBD |

## Interpretation checklist

- Compare TGN against the causal baseline, not only against random chance.
- Check whether performance degrades in later chronological bins.
- Check whether long-inactive pairs are harder than recently active pairs.
- Quantify the change from removing graph time features.
- Report limitations: one dataset, sampled negatives, and transductive nodes.
