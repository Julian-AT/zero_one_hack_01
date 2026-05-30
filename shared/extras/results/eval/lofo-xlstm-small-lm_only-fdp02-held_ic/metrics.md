# Eval — shared/extras/checkpoints/lofo-xlstm-small-lm_only-fdp02-held_ic/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.5500 | 1.0000 | 0.0000 | 0.2722 |
| MOSFET | 0.8 | 0.5700 | 1.0000 | 0.0000 | 0.5461 |
| IGBT | 0.6 | 0.7100 | 0.9900 | 0.0000 | 0.2911 |
| IGBT | 0.8 | 0.6400 | 1.0000 | 0.0000 | 0.2715 |
| IC | 0.6 | 0.6300 | 0.9500 | 0.0000 | 0.4993 |
| IC | 0.8 | 0.5000 | 1.0000 | 0.0000 | 0.5941 |

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