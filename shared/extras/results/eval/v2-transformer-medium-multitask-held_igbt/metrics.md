# Eval — shared/extras/checkpoints/v2-transformer-medium-multitask-held_igbt/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9667 | 1.0000 | 0.0000 | 0.1950 |
| MOSFET | 0.8 | 0.5000 | 1.0000 | 0.0167 | 0.1219 |
| IGBT | 0.6 | 0.7333 | 0.9667 | 0.0000 | 0.5146 |
| IGBT | 0.8 | 0.5833 | 1.0000 | 0.0000 | 0.3000 |
| IC | 0.6 | 0.7500 | 1.0000 | 0.0000 | 0.2765 |
| IC | 0.8 | 0.4833 | 1.0000 | 0.0000 | 0.2906 |

## Anomaly detection
- n = 300, binary acc = 0.8600, AUC = 0.7162
- precision(valid) = 0.7162, recall(valid) = 1.0000
- TP/FP/TN/FN = 106/42/152/0
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **IGBT**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT ⭐ | 100 | 0.5800 | 0.1600 | 0.8800 |
| IC | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |