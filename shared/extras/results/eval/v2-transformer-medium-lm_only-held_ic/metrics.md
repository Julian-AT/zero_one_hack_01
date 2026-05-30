# Eval — shared/extras/checkpoints/v2-transformer-medium-lm_only-held_ic/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.8167 | 1.0000 | 0.0000 | 0.1961 |
| MOSFET | 0.8 | 0.4167 | 1.0000 | 0.0167 | 0.1299 |
| IGBT | 0.6 | 0.6667 | 1.0000 | 0.0000 | 0.2374 |
| IGBT | 0.8 | 0.6667 | 1.0000 | 0.0000 | 0.2238 |
| IC | 0.6 | 0.7500 | 0.9667 | 0.0000 | 0.4691 |
| IC | 0.8 | 0.4667 | 1.0000 | 0.0000 | 0.5534 |

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