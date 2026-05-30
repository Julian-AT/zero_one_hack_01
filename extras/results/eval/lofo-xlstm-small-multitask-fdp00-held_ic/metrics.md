# Eval — extras/checkpoints/lofo-xlstm-small-multitask-fdp00-held_ic/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.5800 | 1.0000 | 0.0000 | 0.2975 |
| MOSFET | 0.8 | 0.6700 | 1.0000 | 0.0000 | 0.2371 |
| IGBT | 0.6 | 0.5500 | 0.8600 | 0.0000 | 0.4489 |
| IGBT | 0.8 | 0.6200 | 1.0000 | 0.0000 | 0.3013 |
| IC | 0.6 | 0.6200 | 0.9500 | 0.0000 | 0.5082 |
| IC | 0.8 | 0.4200 | 1.0000 | 0.0000 | 0.5987 |

## Anomaly detection
- n = 300, binary acc = 1.0000, AUC = 1.0000
- precision(valid) = 1.0000, recall(valid) = 1.0000
- TP/FP/TN/FN = 148/0/152/0
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **IC**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC ⭐ | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |