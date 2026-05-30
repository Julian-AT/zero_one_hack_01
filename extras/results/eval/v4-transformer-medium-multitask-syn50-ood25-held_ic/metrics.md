# Eval — extras/checkpoints/v4-transformer-medium-multitask-syn50-ood25-held_ic/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9667 | 1.0000 | 1.0000 | 0.9833 | 0.0000 | 0.1875 | 0.3596 | 0.5194 |
| MOSFET | 0.8 | 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.0167 | 0.1218 | 0.6854 | 0.9719 |
| IGBT | 0.6 | 0.6833 | 1.0000 | 1.0000 | 0.8417 | 0.0000 | 0.2507 | 0.3190 | 0.5107 |
| IGBT | 0.8 | 0.6000 | 1.0000 | 1.0000 | 0.7972 | 0.0000 | 0.2322 | 0.5932 | 0.8840 |
| IC | 0.6 | 0.6667 | 0.8833 | 0.9500 | 0.7908 | 0.0000 | 0.5126 | 0.1615 | 0.4452 |
| IC | 0.8 | 0.5500 | 1.0000 | 1.0000 | 0.7750 | 0.0000 | 0.5464 | 0.1997 | 0.4435 |

## Anomaly detection
- n = 300, binary acc = 0.9967, AUC = 1.0000
- invalid class (Task-3 reporting): P = 1.0000, R = 0.9935, F1 = 0.9967
- valid class: P = 0.9932, R = 1.0000, F1 = 0.9966
- confusion matrix (invalid = positive):
    | | pred invalid | pred valid |
    |---|--:|--:|
    | actual invalid | 152 | 1 |
    | actual valid   | 0 | 147 |
- rule attribution accuracy = 0.9079 (n_invalid=152)

### Per-family breakdown (held-out: **IC**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC ⭐ | 100 | 0.9900 | 1.0000 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |