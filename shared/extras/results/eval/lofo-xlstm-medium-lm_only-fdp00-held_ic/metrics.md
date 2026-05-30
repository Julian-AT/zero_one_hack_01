# Eval — shared/extras/checkpoints/lofo-xlstm-medium-lm_only-fdp00-held_ic/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.3400 | 0.8800 | 0.0000 | 0.5847 |
| MOSFET | 0.8 | 0.5300 | 1.0000 | 0.0000 | 0.7724 |
| IGBT | 0.6 | 0.4000 | 0.6800 | 0.0000 | 0.3623 |
| IGBT | 0.8 | 0.6700 | 1.0000 | 0.0000 | 0.3693 |
| IC | 0.6 | 0.6500 | 0.9500 | 0.0000 | 0.5097 |
| IC | 0.8 | 0.5300 | 1.0000 | 0.0000 | 0.5831 |

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