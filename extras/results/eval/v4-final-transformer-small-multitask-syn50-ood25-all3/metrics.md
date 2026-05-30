# Eval — extras/checkpoints/v4-final-transformer-small-multitask-syn50-ood25-all3/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9667 | 1.0000 | 1.0000 | 0.9833 | 0.0000 | 0.1917 | 0.3570 | 0.5157 |
| MOSFET | 0.8 | 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.0167 | 0.1218 | 0.6854 | 0.9719 |
| IGBT | 0.6 | 0.6833 | 1.0000 | 1.0000 | 0.8417 | 0.0000 | 0.2518 | 0.3179 | 0.5110 |
| IGBT | 0.8 | 0.6000 | 1.0000 | 1.0000 | 0.7972 | 0.0000 | 0.2322 | 0.5932 | 0.8840 |
| IC | 0.6 | 0.7333 | 0.9667 | 1.0000 | 0.8528 | 0.0000 | 0.2771 | 0.3484 | 0.5696 |
| IC | 0.8 | 0.5500 | 1.0000 | 1.0000 | 0.7750 | 0.0000 | 0.2999 | 0.3568 | 0.9520 |

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