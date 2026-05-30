# Eval — shared/extras/checkpoints/lofo-transformer-small-multitask-fdp00-held_igbt/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.6400 | 1.0000 | 0.0000 | 0.3632 |
| MOSFET | 0.8 | 0.6700 | 1.0000 | 0.0000 | 0.2468 |
| IGBT | 0.6 | 0.7500 | 0.9700 | 0.0000 | 0.4905 |
| IGBT | 0.8 | 0.6300 | 1.0000 | 0.0000 | 0.5189 |
| IC | 0.6 | 0.7700 | 1.0000 | 0.0000 | 0.2923 |
| IC | 0.8 | 0.5100 | 1.0000 | 0.0000 | 0.3232 |

## Anomaly detection
- n = 300, binary acc = 1.0000, AUC = 1.0000
- precision(valid) = 1.0000, recall(valid) = 1.0000
- TP/FP/TN/FN = 148/0/152/0
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **IGBT**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT ⭐ | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |