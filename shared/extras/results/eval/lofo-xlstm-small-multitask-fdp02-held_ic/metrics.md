# Eval — shared/extras/checkpoints/lofo-xlstm-small-multitask-fdp02-held_ic/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.6500 | 1.0000 | 0.0000 | 0.2955 |
| MOSFET | 0.8 | 0.6700 | 1.0000 | 0.0000 | 0.2367 |
| IGBT | 0.6 | 0.5700 | 1.0000 | 0.0000 | 0.4068 |
| IGBT | 0.8 | 0.6500 | 1.0000 | 0.0000 | 0.2983 |
| IC | 0.6 | 0.5800 | 0.9500 | 0.0000 | 0.5090 |
| IC | 0.8 | 0.4400 | 1.0000 | 0.0000 | 0.5970 |

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