# Eval — shared/extras/checkpoints/v3-transformer-small-multitask-ood25-held_mosfet/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.8167 | 1.0000 | 1.0000 | 0.9083 | 0.0000 | 0.2305 | 0.2673 | 0.5115 |
| MOSFET | 0.8 | 0.4167 | 1.0000 | 1.0000 | 0.7083 | 0.0000 | 0.1610 | 0.6144 | 0.9719 |
| IGBT | 0.6 | 0.6500 | 1.0000 | 1.0000 | 0.8250 | 0.0000 | 0.2404 | 0.3235 | 0.5141 |
| IGBT | 0.8 | 0.6667 | 1.0000 | 1.0000 | 0.8306 | 0.0000 | 0.2260 | 0.6006 | 0.8840 |
| IC | 0.6 | 0.7500 | 0.9667 | 1.0000 | 0.8611 | 0.0000 | 0.2674 | 0.3225 | 0.5716 |
| IC | 0.8 | 0.4667 | 1.0000 | 1.0000 | 0.7333 | 0.0000 | 0.2872 | 0.3066 | 0.9522 |

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