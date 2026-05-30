# Eval — extras/checkpoints/v4-final-transformer-medium-multitask-syn50-ood25-all3/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9000 | 1.0000 | 1.0000 | 0.9500 | 0.0000 | 0.1981 | 0.3504 | 0.5115 |
| MOSFET | 0.8 | 0.4000 | 1.0000 | 1.0000 | 0.7000 | 0.0167 | 0.1305 | 0.6763 | 0.9719 |
| IGBT | 0.6 | 0.6667 | 1.0000 | 1.0000 | 0.8333 | 0.0000 | 0.2410 | 0.3238 | 0.5140 |
| IGBT | 0.8 | 0.6000 | 1.0000 | 1.0000 | 0.7972 | 0.0000 | 0.2318 | 0.5951 | 0.8840 |
| IC | 0.6 | 0.7833 | 0.9667 | 1.0000 | 0.8778 | 0.0000 | 0.2774 | 0.3506 | 0.5702 |
| IC | 0.8 | 0.5333 | 1.0000 | 1.0000 | 0.7667 | 0.0000 | 0.3051 | 0.3555 | 0.9520 |

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

### Per-family breakdown
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |