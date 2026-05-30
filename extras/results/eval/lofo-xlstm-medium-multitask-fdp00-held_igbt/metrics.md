# Eval — extras/checkpoints/lofo-xlstm-medium-multitask-fdp00-held_igbt/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.5400 | 0.9900 | 0.0000 | 0.4611 |
| MOSFET | 0.8 | 0.6700 | 1.0000 | 0.0000 | 0.2119 |
| IGBT | 0.6 | 0.6200 | 0.9700 | 0.0000 | 0.5239 |
| IGBT | 0.8 | 0.7000 | 0.9900 | 0.0000 | 0.3137 |
| IC | 0.6 | 0.7000 | 1.0000 | 0.0000 | 0.3323 |
| IC | 0.8 | 0.4900 | 1.0000 | 0.0000 | 0.4105 |

## Anomaly detection
- n = 300, binary acc = 0.8600, AUC = 0.7297
- precision(valid) = 0.7162, recall(valid) = 1.0000
- TP/FP/TN/FN = 106/42/152/0
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **IGBT**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT ⭐ | 100 | 0.5800 | 0.2000 | 0.8800 |
| IC | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |