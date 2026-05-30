# Eval — shared/extras/checkpoints/lofo-xlstm-small-multitask-fdp02-held_mosfet/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.4700 | 0.9700 | 0.0000 | 0.6439 |
| MOSFET | 0.8 | 0.6300 | 1.0000 | 0.0000 | 0.7989 |
| IGBT | 0.6 | 0.7300 | 1.0000 | 0.0000 | 0.4804 |
| IGBT | 0.8 | 0.6900 | 1.0000 | 0.0000 | 0.2922 |
| IC | 0.6 | 0.6600 | 1.0000 | 0.0000 | 0.3245 |
| IC | 0.8 | 0.4500 | 1.0000 | 0.0000 | 0.4032 |

## Anomaly detection
- n = 300, binary acc = 1.0000, AUC = 1.0000
- precision(valid) = 1.0000, recall(valid) = 1.0000
- TP/FP/TN/FN = 148/0/152/0
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **MOSFET**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET ⭐ | 100 | 1.0000 | 1.0000 | 0.8846 |