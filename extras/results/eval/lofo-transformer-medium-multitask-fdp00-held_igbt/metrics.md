# Eval — extras/checkpoints/lofo-transformer-medium-multitask-fdp00-held_igbt/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9100 | 0.9900 | 1.0000 | 0.9508 | 0.0000 | 0.3175 | 0.2750 | 0.4394 |
| MOSFET | 0.8 | 0.6700 | 1.0000 | 1.0000 | 0.8350 | 0.0000 | 0.3310 | 0.6648 | 0.9322 |
| IGBT | 0.6 | 0.5000 | 0.8800 | 0.9000 | 0.6928 | 0.0000 | 0.4870 | 0.1138 | 0.2875 |
| IGBT | 0.8 | 0.6800 | 1.0000 | 1.0000 | 0.8367 | 0.0000 | 0.3791 | 0.2160 | 0.8626 |
| IC | 0.6 | 0.6500 | 1.0000 | 1.0000 | 0.8150 | 0.0000 | 0.4943 | 0.1559 | 0.3008 |
| IC | 0.8 | 0.4900 | 1.0000 | 1.0000 | 0.7450 | 0.0000 | 0.6402 | 0.2467 | 0.8765 |

## Anomaly detection
- n = 300, binary acc = 1.0000, AUC = 1.0000
- invalid class (Task-3 reporting): P = 1.0000, R = 1.0000, F1 = 1.0000
- valid class: P = 1.0000, R = 1.0000, F1 = 1.0000
- confusion matrix (invalid = positive):
    | | pred invalid | pred valid |
    |---|--:|--:|
    | actual invalid | 152 | 0 |
    | actual valid   | 0 | 148 |
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **IGBT**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT ⭐ | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |