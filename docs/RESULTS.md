# Final results

## Overall controlled comparison

| Model | Parameters | Validation AP | Validation ROC-AUC | Test AP | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline TGN | 260,301 | 0.9611 | 0.9567 | **0.9491** | **0.9488** |
| Proposed/Enhanced TGN | 293,901 | 0.9571 | 0.9523 | 0.9469 | 0.9423 |
| Enhanced − Baseline | +33,600 | -0.0040 | -0.0045 | -0.0021 | -0.0065 |

The enhanced model did not improve the primary Average Precision metric. The
Baseline TGN remained both smaller and more accurate, so the proposed changes
are not justified by this experiment's results.

## Performance across the test timeline

| Test time bin | Baseline AP | Enhanced AP | Baseline ROC-AUC | Enhanced ROC-AUC |
|---:|---:|---:|---:|---:|
| 1 | 0.9480 | 0.9391 | 0.9470 | 0.9334 |
| 2 | 0.9482 | 0.9475 | 0.9456 | 0.9425 |
| 3 | 0.9468 | 0.9535 | 0.9514 | 0.9496 |
| 4 | 0.9544 | 0.9479 | 0.9513 | 0.9441 |

The baseline remained stable across the four chronological quarters. The
enhanced model improved after the first quarter but did not consistently exceed
the baseline.

## Performance by endpoint inactivity

| Inactivity bin | Gap range | Baseline AP | Enhanced AP | Baseline ROC-AUC | Enhanced ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 3–2,154 | 0.9724 | 0.9733 | 0.9733 | 0.9715 |
| 2 | 2,155–5,476 | 0.9591 | 0.9563 | 0.9595 | 0.9544 |
| 3 | 5,477–22,651 | 0.9328 | 0.9328 | 0.9352 | 0.9251 |
| 4 | 22,663–2,421,296 | 0.9270 | 0.9196 | 0.9232 | 0.9136 |

Both models degraded when the most recently active endpoint had been inactive
for longer. This shows that temporal distance is a meaningful source of
difficulty. The enhancement did not solve the long-inactivity problem and was
weakest in the largest-gap bin.

## Conclusion

The controlled experiment rejects the hypothesis that the proposed learned
message encoder and gated memory-neighborhood fusion improve Wikipedia
temporal link prediction under the selected hyperparameters. The standard TGN
is the preferred model because it achieves better test performance with fewer
parameters. Future work could isolate each enhancement, tune its regularization,
or evaluate inductive new-node prediction.
