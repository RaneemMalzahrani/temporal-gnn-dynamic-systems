# Controlled experiment: Baseline TGN vs Enhanced TGN

## Research question

Does learning a compact temporal message representation and adaptively fusing
long-term memory with recent temporal neighbors improve link prediction over a
standard Temporal Graph Network?

## Dataset and task

The experiment uses the JODIE Wikipedia temporal interaction network. Each
event contains a source node, destination node, timestamp, and 172-dimensional
message vector. The task is temporal link prediction: score a real destination
against a sampled negative destination before the current event is revealed to
the model.

## Model 1 — Baseline TGN

- Identity message function
- Last-message aggregation
- GRU node memory
- Learned relative-time encoding
- Transformer-based temporal neighbor attention
- MLP link predictor

This is the ordinary initial model and closely follows the official PyTorch
Geometric TGN example.

## Model 2 — Proposed/Enhanced TGN

The enhanced model starts from the same TGN and introduces:

1. **Learned message encoder:** a two-layer nonlinear network transforms source
   memory, destination memory, raw event features, and time encoding into a
   compact message before the GRU memory update.
2. **Gated memory-neighborhood fusion:** a learned gate balances persistent
   long-term memory against the recent time-aware neighbor representation.
   Layer normalization stabilizes the fused embedding.

## Controlled variables

The two JSON configurations deliberately share:

- dataset and chronological 70%/15%/15% split,
- random seed and negative-sampling procedure,
- batch and temporal-neighborhood sizes,
- memory, time, and embedding dimensions,
- optimizer, learning rate, epochs, and early stopping,
- link predictor and evaluation metrics.

Only `model_variant` and `output_dir` differ. The comparison command refuses to
run if any controlled setting is inconsistent.

With the Wikipedia message dimension, the baseline has 260,301 trainable
parameters and the enhanced model has 293,901 (+12.9%). The generated
comparison reports this complexity difference beside the accuracy metrics.

## Leakage controls

- Events remain sorted; temporal loaders are never shuffled.
- Validation chooses the checkpoint; the final test period remains untouched.
- Each event is predicted before it updates node memory.
- Train and validation events are replayed chronologically before final test.
- Negative destinations use fixed seeds for reproducibility.

## Metrics and time-aware analysis

- Average Precision (primary)
- ROC-AUC
- Metrics in four chronological test bins
- Metrics in four endpoint inactivity-gap bins

## Commands

```bash
temporal-gnn train --config configs/baseline_tgn.json
temporal-gnn train --config configs/enhanced_tgn.json
temporal-gnn compare
```

## Results table

Fill this table from `outputs/comparison/comparison.md` after both runs.

| Model | Validation AP | Test AP | Test ROC-AUC |
|---|---:|---:|---:|
| Baseline TGN | 0.9611 | 0.9491 | 0.9488 |
| Proposed/Enhanced TGN | 0.9571 | 0.9469 | 0.9423 |
| Improvement | -0.0040 | -0.0021 | -0.0065 |

The optional recency-frequency reference is reported separately and must not
replace Baseline TGN in this table.

The proposed enhancement did not outperform the baseline. It added 33,600
parameters (+12.9%) while slightly reducing both test metrics. This is a valid
experimental outcome: the extra message transformation and fusion gate added
complexity without improving generalization on Wikipedia under this setup.

## Interpretation checklist

- State whether the enhanced model improves AP and ROC-AUC.
- Compare early and late chronological test bins.
- Check whether links involving long-inactive endpoints remain more difficult.
- Discuss whether gains justify the added parameters and complexity.
- Report limitations: one dataset, sampled negatives, and transductive nodes.
