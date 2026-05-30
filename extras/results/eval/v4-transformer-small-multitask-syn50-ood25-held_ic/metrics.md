# Eval — extras/checkpoints/v4-transformer-small-multitask-syn50-ood25-held_ic/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9667 | 1.0000 | 1.0000 | 0.9833 | 0.0000 | 0.1875 | 0.3596 | 0.5194 |
| MOSFET | 0.8 | 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.0167 | 0.1218 | 0.6854 | 0.9719 |
| IGBT | 0.6 | 0.6500 | 1.0000 | 1.0000 | 0.8250 | 0.0000 | 0.2521 | 0.3176 | 0.5107 |
| IGBT | 0.8 | 0.6000 | 1.0000 | 1.0000 | 0.7972 | 0.0000 | 0.2322 | 0.5932 | 0.8840 |
| IC | 0.6 | 0.7167 | 0.9333 | 0.9667 | 0.8250 | 0.0000 | 0.4876 | 0.1683 | 0.4233 |
| IC | 0.8 | 0.5500 | 1.0000 | 1.0000 | 0.7750 | 0.0000 | 0.5641 | 0.1982 | 0.4413 |

## Anomaly detection
- n = 300, binary acc = 0.8333, AUC = 0.6622
- invalid class (Task-3 reporting): P = 1.0000, R = 0.7525, F1 = 0.8588
- valid class: P = 0.6622, R = 1.0000, F1 = 0.7967
- confusion matrix (invalid = positive):
    | | pred invalid | pred valid |
    |---|--:|--:|
    | actual invalid | 152 | 50 |
    | actual valid   | 0 | 98 |
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **IC**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC ⭐ | 100 | 0.5000 | 0.0000 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |