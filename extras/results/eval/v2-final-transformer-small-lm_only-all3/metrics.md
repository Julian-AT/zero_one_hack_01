# Eval — extras/checkpoints/v2-final-transformer-small-lm_only-all3/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9667 | 1.0000 | 0.0000 | 0.1952 |
| MOSFET | 0.8 | 0.4667 | 1.0000 | 0.0167 | 0.1250 |
| IGBT | 0.6 | 0.6667 | 1.0000 | 0.0000 | 0.2402 |
| IGBT | 0.8 | 0.6000 | 1.0000 | 0.0000 | 0.2318 |
| IC | 0.6 | 0.7500 | 1.0000 | 0.0000 | 0.2717 |
| IC | 0.8 | 0.5500 | 1.0000 | 0.0000 | 0.2788 |

## Anomaly detection
- n = 300, binary acc = 1.0000, AUC = 1.0000
- precision(valid) = 1.0000, recall(valid) = 1.0000
- TP/FP/TN/FN = 148/0/152/0
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |