# Eval — shared/extras/checkpoints/lofo-transformer-medium-multitask-fdp02-held_mosfet/final.pt

## Next-step + completion (held-out per family)
| family | frac | Top-1 | Top-3 | Top-5 | MRR | ExactMatch | NED | TokenAcc | BlockAcc |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.4700 | 0.9900 | 0.9900 | 0.7300 | 0.0000 | 0.5967 | 0.1280 | 0.2880 |
| MOSFET | 0.8 | 0.4300 | 1.0000 | 1.0000 | 0.7150 | 0.0000 | 0.7489 | 0.1336 | 0.3814 |
| IGBT | 0.6 | 0.5400 | 0.8500 | 0.9800 | 0.7108 | 0.0000 | 0.4759 | 0.1315 | 0.2429 |
| IGBT | 0.8 | 0.7000 | 1.0000 | 1.0000 | 0.8433 | 0.0000 | 0.3986 | 0.5575 | 0.8669 |
| IC | 0.6 | 0.7900 | 0.9900 | 0.9900 | 0.8817 | 0.0000 | 0.3968 | 0.3115 | 0.5085 |
| IC | 0.8 | 0.4600 | 1.0000 | 1.0000 | 0.7300 | 0.0000 | 0.5182 | 0.3118 | 0.9328 |

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