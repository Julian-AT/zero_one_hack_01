# Eval — extras/checkpoints/lofo-transformer-small-multitask-fdp00-held_mosfet/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.4700 | 0.8000 | 0.0000 | 0.5127 |
| MOSFET | 0.8 | 0.5700 | 1.0000 | 0.0000 | 0.5909 |
| IGBT | 0.6 | 0.3100 | 0.4600 | 0.0000 | 0.6392 |
| IGBT | 0.8 | 0.6900 | 1.0000 | 0.0000 | 0.4088 |
| IC | 0.6 | 0.6400 | 0.8500 | 0.0000 | 0.3660 |
| IC | 0.8 | 0.5300 | 1.0000 | 0.0000 | 0.4219 |

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