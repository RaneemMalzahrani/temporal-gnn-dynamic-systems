# Experiment configurations

- `baseline_tgn.json`: the standard TGN used as the primary baseline.
- `enhanced_tgn.json`: the proposed TGN with a learned message encoder and
  gated fusion of long-term memory with the recent temporal neighborhood.

All data splits, dimensions, optimization settings, seeds, and evaluation
settings are identical. Only `model_variant` and the output directory differ.
