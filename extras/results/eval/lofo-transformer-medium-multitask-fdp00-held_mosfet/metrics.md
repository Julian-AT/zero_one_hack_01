# Eval — extras/checkpoints/lofo-transformer-medium-multitask-fdp00-held_mosfet/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.5600 | 0.8500 | 0.8700 | 0.6850 | 0.0000 | 0.6565 | 0.1597 | 0.3750 |
| MOSFET | 0.8 | 0.4300 | 1.0000 | 1.0000 | 0.7150 | 0.0000 | 0.7778 | 0.1476 | 0.4775 |
| IGBT | 0.6 | 0.6700 | 1.0000 | 1.0000 | 0.8350 | 0.0000 | 0.4147 | 0.2125 | 0.4099 |
| IGBT | 0.8 | 0.7100 | 1.0000 | 1.0000 | 0.8483 | 0.0000 | 0.6211 | 0.2546 | 0.5054 |
| IC | 0.6 | 0.7900 | 0.9900 | 0.9900 | 0.8817 | 0.0000 | 0.3806 | 0.3381 | 0.5345 |
| IC | 0.8 | 0.4700 | 1.0000 | 1.0000 | 0.7350 | 0.0000 | 0.4821 | 0.3706 | 0.9190 |

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

### Per-family breakdown (held-out: **MOSFET**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET ⭐ | 100 | 1.0000 | 1.0000 | 0.8846 |