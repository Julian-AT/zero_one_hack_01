# Eval — extras/checkpoints/lofo-xlstm-small-multitask-fdp02-held_igbt/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.6400 | 1.0000 | 0.0000 | 0.3051 |
| MOSFET | 0.8 | 0.6700 | 1.0000 | 0.0000 | 0.2776 |
| IGBT | 0.6 | 0.6300 | 0.9700 | 0.0000 | 0.6460 |
| IGBT | 0.8 | 0.7300 | 1.0000 | 0.0000 | 0.3696 |
| IC | 0.6 | 0.6600 | 1.0000 | 0.0000 | 0.3284 |
| IC | 0.8 | 0.4900 | 1.0000 | 0.0000 | 0.4002 |

## Anomaly detection
- n = 300, binary acc = 0.8333, AUC = 0.6622
- precision(valid) = 0.6622, recall(valid) = 1.0000
- TP/FP/TN/FN = 98/50/152/0
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **IGBT**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT ⭐ | 100 | 0.5000 | 0.0000 | 0.8800 |
| IC | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |