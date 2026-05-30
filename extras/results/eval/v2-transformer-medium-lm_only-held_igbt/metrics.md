# Eval — extras/checkpoints/v2-transformer-medium-lm_only-held_igbt/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.8167 | 1.0000 | 0.0000 | 0.1986 |
| MOSFET | 0.8 | 0.4167 | 1.0000 | 0.0167 | 0.1299 |
| IGBT | 0.6 | 0.6833 | 0.9667 | 0.0000 | 0.5441 |
| IGBT | 0.8 | 0.6000 | 1.0000 | 0.0000 | 0.3406 |
| IC | 0.6 | 0.7833 | 1.0000 | 0.0000 | 0.2728 |
| IC | 0.8 | 0.5333 | 1.0000 | 0.0000 | 0.2929 |

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