# Eval — extras/checkpoints/multitask-transformer_medium-20260530-030653/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.6250 | 1.0000 | 0.0000 | 0.4399 |
| MOSFET | 0.8 | 0.6250 | 1.0000 | 0.0000 | 0.4252 |
| IGBT | 0.6 | 0.6000 | 0.7750 | 0.0000 | 0.3452 |
| IGBT | 0.8 | 0.6250 | 1.0000 | 0.0000 | 0.4662 |
| IC | 0.6 | 0.6000 | 0.9500 | 0.0000 | 0.4223 |
| IC | 0.8 | 0.4500 | 1.0000 | 0.0000 | 0.5315 |

## Anomaly detection
- n = 40, binary acc = 1.0000
- precision(valid) = 1.0000, recall(valid) = 1.0000
- TP/FP/TN/FN = 24/0/16/0
- rule attribution accuracy = 1.0000 (n_invalid=16)