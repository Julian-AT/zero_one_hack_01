# Eval — extras/checkpoints/v2-transformer-small-multitask-held_ic/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9667 | 1.0000 | 0.0000 | 0.1914 |
| MOSFET | 0.8 | 0.5000 | 1.0000 | 0.0167 | 0.1218 |
| IGBT | 0.6 | 0.6500 | 1.0000 | 0.0000 | 0.2493 |
| IGBT | 0.8 | 0.6667 | 1.0000 | 0.0000 | 0.2252 |
| IC | 0.6 | 0.7167 | 0.9667 | 0.0000 | 0.4714 |
| IC | 0.8 | 0.4667 | 1.0000 | 0.0000 | 0.5505 |

## Anomaly detection
- n = 300, binary acc = 0.8333, AUC = 0.6622
- precision(valid) = 0.6622, recall(valid) = 1.0000
- TP/FP/TN/FN = 98/50/152/0
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **IC**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC ⭐ | 100 | 0.5000 | 0.0000 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |