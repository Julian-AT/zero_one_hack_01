# Eval — shared/extras/checkpoints/lofo-xlstm-small-lm_only-fdp02-held_igbt/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.5600 | 1.0000 | 0.0000 | 0.2667 |
| MOSFET | 0.8 | 0.6800 | 1.0000 | 0.0000 | 0.2006 |
| IGBT | 0.6 | 0.7300 | 0.9700 | 0.0000 | 0.5105 |
| IGBT | 0.8 | 0.7200 | 1.0000 | 0.0000 | 0.4403 |
| IC | 0.6 | 0.6800 | 1.0000 | 0.0000 | 0.3298 |
| IC | 0.8 | 0.5300 | 1.0000 | 0.0000 | 0.4144 |

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