# Eval — shared/extras/checkpoints/v4-transformer-medium-multitask-syn50-ood25-held_mosfet/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.9667 | 1.0000 | 1.0000 | 0.9833 | 0.0000 | 0.2185 | 0.2724 | 0.5194 |
| MOSFET | 0.8 | 0.5000 | 1.0000 | 1.0000 | 0.7500 | 0.0000 | 0.1537 | 0.6223 | 0.9719 |
| IGBT | 0.6 | 0.6667 | 1.0000 | 1.0000 | 0.8333 | 0.0000 | 0.2518 | 0.3179 | 0.5107 |
| IGBT | 0.8 | 0.6000 | 1.0000 | 1.0000 | 0.7972 | 0.0000 | 0.2322 | 0.5932 | 0.8840 |
| IC | 0.6 | 0.7333 | 0.9667 | 1.0000 | 0.8333 | 0.0000 | 0.2801 | 0.3510 | 0.5718 |
| IC | 0.8 | 0.5500 | 1.0000 | 1.0000 | 0.7750 | 0.0000 | 0.3007 | 0.3603 | 0.9520 |

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