# Eval — extras/checkpoints/v4-transformer-small-multitask-syn50-ood25-held_mosfet/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9167 | 1.0000 | 1.0000 | 0.9583 | 0.0000 | 0.2182 | 0.2259 | 0.5469 |
| MOSFET | 0.8 | 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.0000 | 0.1537 | 0.6223 | 0.9719 |
| IGBT | 0.6 | 0.6500 | 1.0000 | 1.0000 | 0.8250 | 0.0000 | 0.2512 | 0.3184 | 0.5107 |
| IGBT | 0.8 | 0.5833 | 1.0000 | 1.0000 | 0.7889 | 0.0000 | 0.2316 | 0.5938 | 0.8840 |
| IC | 0.6 | 0.7333 | 0.9667 | 1.0000 | 0.8528 | 0.0000 | 0.2743 | 0.3211 | 0.5729 |
| IC | 0.8 | 0.5167 | 1.0000 | 1.0000 | 0.7583 | 0.0167 | 0.2886 | 0.3151 | 0.9520 |

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