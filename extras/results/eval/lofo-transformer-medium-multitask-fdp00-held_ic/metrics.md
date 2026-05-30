# Eval — extras/checkpoints/lofo-transformer-medium-multitask-fdp00-held_ic/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.5800 | 1.0000 | 1.0000 | 0.7900 | 0.0000 | 0.3130 | 0.2043 | 0.3487 |
| MOSFET | 0.8 | 0.6700 | 1.0000 | 1.0000 | 0.8350 | 0.0000 | 0.3234 | 0.6015 | 0.9455 |
| IGBT | 0.6 | 0.5800 | 0.9300 | 0.9300 | 0.7533 | 0.0000 | 0.6793 | 0.0702 | 0.2096 |
| IGBT | 0.8 | 0.7100 | 1.0000 | 1.0000 | 0.8517 | 0.0000 | 0.4659 | 0.5075 | 0.8587 |
| IC | 0.6 | 0.6100 | 0.8900 | 0.8900 | 0.7417 | 0.0000 | 0.5891 | 0.1498 | 0.3188 |
| IC | 0.8 | 0.4300 | 0.9800 | 0.9800 | 0.7033 | 0.0000 | 0.7227 | 0.1725 | 0.3647 |

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

### Per-family breakdown (held-out: **IC**)
| family | n | acc | AUC | rule_attrib |
|---|--:|--:|--:|--:|
| IGBT | 100 | 1.0000 | 1.0000 | 0.8800 |
| IC ⭐ | 100 | 1.0000 | 1.0000 | 0.9600 |
| MOSFET | 100 | 1.0000 | 1.0000 | 0.8846 |