# Eval — extras/checkpoints/lofo-transformer-small-multitask-fdp02-held_mosfet/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.4700 | 0.9100 | 0.0000 | 0.5841 |
| MOSFET | 0.8 | 0.5700 | 1.0000 | 0.0000 | 0.6284 |
| IGBT | 0.6 | 0.6000 | 0.8600 | 0.0000 | 0.5911 |
| IGBT | 0.8 | 0.6900 | 1.0000 | 0.0000 | 0.5982 |
| IC | 0.6 | 0.6600 | 1.0000 | 0.0000 | 0.5744 |
| IC | 0.8 | 0.5500 | 1.0000 | 0.0000 | 0.5891 |

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