# Eval — extras/checkpoints/v2-transformer-medium-multitask-held_mosfet/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.7833 | 1.0000 | 0.0000 | 0.4937 |
| MOSFET | 0.8 | 0.5000 | 1.0000 | 0.0000 | 0.5357 |
| IGBT | 0.6 | 0.6500 | 1.0000 | 0.0000 | 0.2515 |
| IGBT | 0.8 | 0.6000 | 1.0000 | 0.0000 | 0.2322 |
| IC | 0.6 | 0.7500 | 1.0000 | 0.0000 | 0.2758 |
| IC | 0.8 | 0.5500 | 1.0000 | 0.0167 | 0.2890 |

## Anomaly detection
- n = 300, binary acc = 0.9000, AUC = 0.8041
- precision(valid) = 0.7973, recall(valid) = 1.0000
- TP/FP/TN/FN = 118/30/152/0
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **MOSFET**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET ⭐ | 100 | 0.7000 | 0.3958 | 0.8846 |