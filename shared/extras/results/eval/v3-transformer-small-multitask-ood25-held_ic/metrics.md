# Eval — shared/extras/checkpoints/v3-transformer-small-multitask-ood25-held_ic/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9333 | 1.0000 | 1.0000 | 0.9667 | 0.0000 | 0.1919 | 0.3390 | 0.4249 |
| MOSFET | 0.8 | 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.0167 | 0.1218 | 0.6854 | 0.9719 |
| IGBT | 0.6 | 0.6500 | 1.0000 | 1.0000 | 0.8250 | 0.0000 | 0.2502 | 0.3105 | 0.5110 |
| IGBT | 0.8 | 0.6000 | 1.0000 | 1.0000 | 0.7972 | 0.0000 | 0.2322 | 0.5932 | 0.8840 |
| IC | 0.6 | 0.7333 | 0.9333 | 0.9667 | 0.8333 | 0.0000 | 0.5219 | 0.1419 | 0.3725 |
| IC | 0.8 | 0.5500 | 1.0000 | 1.0000 | 0.7750 | 0.0000 | 0.5808 | 0.1997 | 0.4435 |

## Anomaly detection
- n = 300, binary acc = 0.9900, AUC = 0.9797
- invalid class (Task-3 reporting): P = 1.0000, R = 0.9806, F1 = 0.9902
- valid class: P = 0.9797, R = 1.0000, F1 = 0.9898
- confusion matrix (invalid = positive):
    | | pred invalid | pred valid |
    |---|--:|--:|
    | actual invalid | 152 | 3 |
    | actual valid   | 0 | 145 |
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **IC**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC ⭐ | 100 | 0.9700 | 0.9400 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |