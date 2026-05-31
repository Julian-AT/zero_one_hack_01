### Task 1 — Next-step prediction

| Model | Top-1 (ID) | Top-3 (ID) | Top-5 (ID) | MRR (ID) | Top-1 (LoFO macro) | **ID→OOD drop** |
|---|--:|--:|--:|--:|--:|--:|
| Transformer-xLSTM | 0.779 | 0.996 | 1.000 | 0.888 | 0.704 | 0.075 |
| SSL-Hybrid | 0.765 | 1.000 | 1.000 | 0.883 | 0.721 | 0.045 |
| Neurosymbolic | 0.761 | 0.996 | 1.000 | 0.879 | 0.660 | 0.101 |
| Grammar baseline | 0.721 | 0.996 | 1.000 | 0.860 | 0.653 | 0.068 |
| Trigram baseline | 0.721 | 0.982 | 1.000 | 0.856 | 0.653 | 0.068 |

### Task 2 — Sequence completion (NED lower = better; BlockAcc / %rule-clean higher = better)

> %rule-clean = fraction of completions that introduce **no new** process-rule violation beyond the (truncated) partial — isolates the model from truncation artifacts.

| Model | NED (ID) | ExactMatch (ID) | BlockAcc (ID) | %rule-clean (ID) | NED (LoFO) | %rule-clean (LoFO) |
|---|--:|--:|--:|--:|--:|--:|
| Transformer-xLSTM | 0.242 | 0.000 | 0.700 | 1.000 | 0.368 | 1.000 |
| SSL-Hybrid | 0.318 | 0.000 | 0.645 | 1.000 | 0.384 | 0.833 |
| Neurosymbolic | 0.706 | 0.000 | 0.576 | 1.000 | 0.748 | 1.000 |
| Grammar baseline | 0.563 | 0.000 | 0.510 | 1.000 | 0.581 | 1.000 |
| Trigram baseline | 0.629 | 0.000 | 0.540 | 0.500 | 0.648 | 0.511 |

### Task 3 — Anomaly detection

| Model | F1 (ID) | Precision | Recall | ROC-AUC | RuleAttr | BalancedAcc | F1 (LoFO macro) |
|---|--:|--:|--:|--:|--:|--:|--:|
| Transformer-xLSTM | 1.000 | 1.000 | 1.000 | 1.000 | 0.980 | 1.000 | 1.000 |
| SSL-Hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 0.980 | 1.000 | 1.000 |
| Neurosymbolic | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
