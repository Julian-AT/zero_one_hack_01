# Eval — shared/extras/checkpoints/v2-transformer-small-multitask-held_mosfet/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1@cut | Top-5@cut | ExactMatch | NED |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9167 | 1.0000 | 0.0000 | 0.2742 |
| MOSFET | 0.8 | 0.5000 | 1.0000 | 0.0000 | 0.1721 |
| IGBT | 0.6 | 0.6500 | 1.0000 | 0.0000 | 0.2504 |
| IGBT | 0.8 | 0.6000 | 1.0000 | 0.0000 | 0.2284 |
| IC | 0.6 | 0.7500 | 1.0000 | 0.0000 | 0.2758 |
| IC | 0.8 | 0.5500 | 1.0000 | 0.0167 | 0.2890 |

## Anomaly detection
- n = 300, binary acc = 0.8800, AUC = 0.7770
- precision(valid) = 0.7568, recall(valid) = 1.0000
- TP/FP/TN/FN = 112/36/152/0
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **MOSFET**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET ⭐ | 100 | 0.6400 | 0.3125 | 0.8846 |