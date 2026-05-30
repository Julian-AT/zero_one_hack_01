# Eval — extras/checkpoints/v3-transformer-small-multitask-ood25-held_igbt/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9667 | 1.0000 | 1.0000 | 0.9833 | 0.0000 | 0.1875 | 0.3589 | 0.5194 |
| MOSFET | 0.8 | 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.0167 | 0.1218 | 0.6854 | 0.9719 |
| IGBT | 0.6 | 0.5833 | 0.9667 | 0.9667 | 0.7750 | 0.0000 | 0.3963 | 0.1808 | 0.5121 |
| IGBT | 0.8 | 0.6000 | 1.0000 | 1.0000 | 0.7972 | 0.0000 | 0.2839 | 0.1774 | 0.8840 |
| IC | 0.6 | 0.7500 | 0.9667 | 1.0000 | 0.8611 | 0.0000 | 0.2727 | 0.3167 | 0.5685 |
| IC | 0.8 | 0.5333 | 1.0000 | 1.0000 | 0.7667 | 0.0000 | 0.2817 | 0.3115 | 0.9524 |

## Anomaly detection
- n = 300, binary acc = 0.9900, AUC = 0.9865
- invalid class (Task-3 reporting): P = 1.0000, R = 0.9806, F1 = 0.9902
- valid class: P = 0.9797, R = 1.0000, F1 = 0.9898
- confusion matrix (invalid = positive):
    | | pred invalid | pred valid |
    |---|--:|--:|
    | actual invalid | 152 | 3 |
    | actual valid   | 0 | 145 |
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **IGBT**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT ⭐ | 100 | 0.9700 | 0.9600 | 0.8800 |
| IC | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |