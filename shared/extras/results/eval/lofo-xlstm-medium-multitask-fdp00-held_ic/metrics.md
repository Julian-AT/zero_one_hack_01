# Eval — shared/extras/checkpoints/lofo-xlstm-medium-multitask-fdp00-held_ic/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.6400 | 1.0000 | 0.0000 | 0.2751 |
| MOSFET | 0.8 | 0.6300 | 1.0000 | 0.0000 | 0.2515 |
| IGBT | 0.6 | 0.7400 | 1.0000 | 0.0000 | 0.3321 |
| IGBT | 0.8 | 0.7000 | 1.0000 | 0.0000 | 0.2915 |
| IC | 0.6 | 0.6100 | 0.9500 | 0.0000 | 0.5071 |
| IC | 0.8 | 0.4900 | 1.0000 | 0.0000 | 0.5923 |

## Anomaly detection
- n = 300, binary acc = 0.8800, AUC = 0.7568
- precision(valid) = 0.7568, recall(valid) = 1.0000
- TP/FP/TN/FN = 112/36/152/0
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **IC**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC ⭐ | 100 | 0.6400 | 0.2800 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |