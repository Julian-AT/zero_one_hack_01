# Eval — shared/extras/checkpoints/v4-transformer-small-multitask-syn50-ood25-held_igbt/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9667 | 1.0000 | 1.0000 | 0.9833 | 0.0000 | 0.1881 | 0.3586 | 0.5204 |
| MOSFET | 0.8 | 0.5167 | 1.0000 | 1.0000 | 0.7583 | 0.0167 | 0.1230 | 0.6842 | 0.9719 |
| IGBT | 0.6 | 0.6167 | 0.9667 | 0.9667 | 0.7917 | 0.0000 | 0.3870 | 0.1650 | 0.5334 |
| IGBT | 0.8 | 0.6667 | 1.0000 | 1.0000 | 0.8306 | 0.0000 | 0.2783 | 0.1880 | 0.8840 |
| IC | 0.6 | 0.7500 | 0.9667 | 1.0000 | 0.8611 | 0.0000 | 0.2788 | 0.3509 | 0.5696 |
| IC | 0.8 | 0.4667 | 1.0000 | 1.0000 | 0.7333 | 0.0000 | 0.3091 | 0.3512 | 0.9520 |

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