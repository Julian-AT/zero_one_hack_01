# Eval — shared/extras/checkpoints/v4-transformer-medium-multitask-syn50-ood25-held_igbt/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9667 | 1.0000 | 1.0000 | 0.9833 | 0.0000 | 0.1979 | 0.3490 | 0.5190 |
| MOSFET | 0.8 | 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.0167 | 0.1218 | 0.6854 | 0.9719 |
| IGBT | 0.6 | 0.5667 | 0.7833 | 0.9500 | 0.7056 | 0.0000 | 0.5648 | 0.0670 | 0.2760 |
| IGBT | 0.8 | 0.6500 | 1.0000 | 1.0000 | 0.8194 | 0.0000 | 0.2860 | 0.1803 | 0.8690 |
| IC | 0.6 | 0.7833 | 0.9667 | 1.0000 | 0.8778 | 0.0000 | 0.2812 | 0.3505 | 0.5702 |
| IC | 0.8 | 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.0000 | 0.3071 | 0.3532 | 0.9520 |

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