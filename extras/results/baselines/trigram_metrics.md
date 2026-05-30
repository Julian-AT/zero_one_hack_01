# Trigram-with-backoff baseline

Seed: 42.  Test fraction: 0.2.

## EDA reproduction (memorization upper bound)

Train and eval on all 3000 sequences. Reproduces the EDA number.

- Top-1 = **0.7220**
- Top-3 = **0.9679**
- Top-5 = **0.9930**
- MRR   = **0.8441**

## In-distribution held-out (honest baseline)

Train on 80% per family, eval on 20% held-out per family.

| split | Top-1 | Top-3 | Top-5 | MRR |
|---|--:|--:|--:|--:|
| ALL  | 0.7173 | 0.9675 | 0.9931 | 0.8416 |
| MOSFET | 0.7500 | 0.9738 | 0.9973 | 0.8621 |
| IGBT | 0.7140 | 0.9625 | 0.9866 | 0.8371 |
| IC | 0.6860 | 0.9670 | 0.9969 | 0.8249 |

## LoFO (Task 4 OOD proxy)

Train on 2 families, eval on the held-out 3rd. This is the number we expect to roughly predict performance on the hidden family 4.

| held_out | Top-1 | Top-3 | Top-5 | MRR |
|---|--:|--:|--:|--:|
| MOSFET | 0.5024 | 0.6791 | 0.7279 | 0.5976 |
| IGBT | 0.4806 | 0.6604 | 0.7066 | 0.5769 |
| IC | 0.4315 | 0.6244 | 0.6439 | 0.5283 |

## Truncation @ 0.6 / 0.8 (simulates eval_input_valid.csv)

In-distribution; Task 1 (Top-K at cut) and Task 2 (full completion).

| frac | family | Top-1@cut | Top-5@cut | ExactMatch | NormEditDist |
|---|---|--:|--:|--:|--:|
| 0.6 | MOSFET | 0.8750 | 1.0000 | 0.0000 | 0.8700 |
| 0.6 | IGBT | 0.6700 | 1.0000 | 0.0000 | 0.9085 |
| 0.6 | IC | 0.7200 | 1.0000 | 0.0000 | 0.8802 |
| 0.8 | MOSFET | 0.5150 | 1.0000 | 0.0250 | 0.1260 |
| 0.8 | IGBT | 0.6650 | 1.0000 | 0.0000 | 0.2711 |
| 0.8 | IC | 0.5200 | 1.0000 | 0.0000 | 0.5011 |

## LoFO + truncation (the realistic Task 4 number)

| held_out | frac | Top-1@cut | Top-5@cut | ExactMatch | NormEditDist |
|---|---|--:|--:|--:|--:|
| MOSFET | 0.6 | 0.7990 | 1.0000 | 0.0000 | 0.8925 |
| MOSFET | 0.8 | 0.5620 | 1.0000 | 0.0000 | 0.2060 |
| IGBT | 0.6 | 0.6790 | 0.9500 | 0.0000 | 0.9045 |
| IGBT | 0.8 | 0.6960 | 1.0000 | 0.0000 | 0.2712 |
| IC | 0.6 | 0.7290 | 0.9520 | 0.0000 | 0.8866 |
| IC | 0.8 | 0.5170 | 1.0000 | 0.0000 | 0.5192 |