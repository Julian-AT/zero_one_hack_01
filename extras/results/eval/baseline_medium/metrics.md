# Eval — extras/checkpoints/cell1-transformer_medium-compositional/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.5000 | 1.0000 | 0.0000 | 0.4014 |
| MOSFET | 0.8 | 0.5500 | 1.0000 | 0.0000 | 0.4042 |
| IGBT | 0.6 | 0.6750 | 0.9750 | 0.0000 | 0.5289 |
| IGBT | 0.8 | 0.6250 | 1.0000 | 0.0000 | 0.4504 |
| IC | 0.6 | 0.7500 | 1.0000 | 0.0000 | 0.4401 |
| IC | 0.8 | 0.5250 | 1.0000 | 0.0000 | 0.5391 |

## Anomaly detection
- n = 40, binary acc = 1.0000
- precision(valid) = 1.0000, recall(valid) = 1.0000
- TP/FP/TN/FN = 24/0/16/0
- rule attribution accuracy = 1.0000 (n_invalid=16)