# LoFO ablation — held-out family performance

_64 cells crawled; 48 LoFO + 16 final all-3._

Higher Top-1@held = better OOD generalization. `top1_drop` = `top1_id_avg − top1_held` (smaller is better; close to zero means no OOD penalty).

## LoFO cells — ranked by Top-1 on held-out family
| cell_id | arch | size | heads | fdp | held | params | Top1_held | Top5_held | NED_held | Top1_id | top1_drop | anom_AUC_held |
|---|---|---|---|--:|---|--:|--:|--:|--:|--:|--:|--:|
| `lofo-transformer-small-multitask-fdp00-held_igbt` | transformer | small | multitask | 0.0 | igbt | 4372748 | 0.6900 | 0.9850 | 0.5047 | 0.6475 | -0.0425 | 1.0000 |
| `lofo-transformer-small-multitask-fdp02-held_igbt` | transformer | small | multitask | 0.2 | igbt | 4372748 | 0.6700 | 0.9850 | 0.6153 | 0.6600 | -0.0100 | 1.0000 |
| `lofo-transformer-small-lm_only-fdp02-held_igbt` | transformer | small | lm_only | 0.2 | igbt | 4238080 | 0.6150 | 0.8900 | 0.4493 | 0.6200 | 0.0050 | 1.0000 |
| `lofo-transformer-small-lm_only-fdp00-held_ic` | transformer | small | lm_only | 0.0 | ic | 4238080 | 0.6050 | 0.9750 | 0.5564 | 0.6000 | -0.0050 | 1.0000 |
| `lofo-transformer-small-multitask-fdp02-held_ic` | transformer | small | multitask | 0.2 | ic | 4372748 | 0.6050 | 0.9600 | 0.5528 | 0.5775 | -0.0275 | 1.0000 |
| `lofo-transformer-small-lm_only-fdp02-held_ic` | transformer | small | lm_only | 0.2 | ic | 4238080 | 0.6000 | 0.9750 | 0.5911 | 0.6025 | 0.0025 | 1.0000 |
| `lofo-transformer-small-multitask-fdp00-held_ic` | transformer | small | multitask | 0.0 | ic | 4372748 | 0.5750 | 0.9650 | 0.6215 | 0.5050 | -0.0700 | 1.0000 |
| `lofo-transformer-small-lm_only-fdp00-held_igbt` | transformer | small | lm_only | 0.0 | igbt | 4238080 | 0.5600 | 0.9550 | 0.5065 | 0.6275 | 0.0675 | 1.0000 |
| `lofo-transformer-small-lm_only-fdp02-held_mosfet` | transformer | small | lm_only | 0.2 | mosfet | 4238080 | 0.5300 | 1.0000 | 0.6287 | 0.5750 | 0.0450 | 1.0000 |
| `lofo-transformer-small-multitask-fdp00-held_mosfet` | transformer | small | multitask | 0.0 | mosfet | 4372748 | 0.5200 | 0.9000 | 0.5518 | 0.5425 | 0.0225 | 1.0000 |
| `lofo-transformer-small-multitask-fdp02-held_mosfet` | transformer | small | multitask | 0.2 | mosfet | 4372748 | 0.5200 | 0.9550 | 0.6063 | 0.6250 | 0.1050 | 1.0000 |
| `lofo-transformer-small-lm_only-fdp00-held_mosfet` | transformer | small | lm_only | 0.0 | mosfet | 4238080 | 0.4550 | 1.0000 | 0.5126 | 0.5975 | 0.1425 | 1.0000 |
| `lofo-transformer-medium-lm_only-fdp00-held_ic` | transformer | medium | lm_only | 0.0 | ic | 33646080 |  |  |  |  |  |  |
| `lofo-transformer-medium-lm_only-fdp00-held_igbt` | transformer | medium | lm_only | 0.0 | igbt | 33646080 |  |  |  |  |  |  |
| `lofo-transformer-medium-lm_only-fdp00-held_mosfet` | transformer | medium | lm_only | 0.0 | mosfet | 33646080 |  |  |  |  |  |  |
| `lofo-transformer-medium-lm_only-fdp02-held_ic` | transformer | medium | lm_only | 0.2 | ic | 33646080 |  |  |  |  |  |  |
| `lofo-transformer-medium-lm_only-fdp02-held_igbt` | transformer | medium | lm_only | 0.2 | igbt | 33646080 |  |  |  |  |  |  |
| `lofo-transformer-medium-lm_only-fdp02-held_mosfet` | transformer | medium | lm_only | 0.2 | mosfet | 33646080 |  |  |  |  |  |  |
| `lofo-transformer-medium-multitask-fdp00-held_ic` | transformer | medium | multitask | 0.0 | ic | 34177548 |  |  |  |  |  |  |
| `lofo-transformer-medium-multitask-fdp00-held_igbt` | transformer | medium | multitask | 0.0 | igbt | 34177548 |  |  |  |  |  |  |
| `lofo-transformer-medium-multitask-fdp00-held_mosfet` | transformer | medium | multitask | 0.0 | mosfet | 34177548 |  |  |  |  |  |  |
| `lofo-transformer-medium-multitask-fdp02-held_ic` | transformer | medium | multitask | 0.2 | ic | 34177548 |  |  |  |  |  |  |
| `lofo-transformer-medium-multitask-fdp02-held_igbt` | transformer | medium | multitask | 0.2 | igbt | 34177548 |  |  |  |  |  |  |
| `lofo-transformer-medium-multitask-fdp02-held_mosfet` | transformer | medium | multitask | 0.2 | mosfet | 34177548 |  |  |  |  |  |  |
| `lofo-xlstm-medium-lm_only-fdp00-held_ic` | xlstm | medium | lm_only | 0.0 | ic | 12051008 |  |  |  |  |  |  |
| `lofo-xlstm-medium-lm_only-fdp00-held_igbt` | xlstm | medium | lm_only | 0.0 | igbt | 12051008 |  |  |  |  |  |  |
| `lofo-xlstm-medium-lm_only-fdp00-held_mosfet` | xlstm | medium | lm_only | 0.0 | mosfet | 12051008 |  |  |  |  |  |  |
| `lofo-xlstm-medium-lm_only-fdp02-held_ic` | xlstm | medium | lm_only | 0.2 | ic | 12051008 |  |  |  |  |  |  |
| `lofo-xlstm-medium-lm_only-fdp02-held_igbt` | xlstm | medium | lm_only | 0.2 | igbt | 12051008 |  |  |  |  |  |  |
| `lofo-xlstm-medium-lm_only-fdp02-held_mosfet` | xlstm | medium | lm_only | 0.2 | mosfet | 12051008 |  |  |  |  |  |  |
| `lofo-xlstm-medium-multitask-fdp00-held_ic` | xlstm | medium | multitask | 0.0 | ic | 12582476 |  |  |  |  |  |  |
| `lofo-xlstm-medium-multitask-fdp00-held_igbt` | xlstm | medium | multitask | 0.0 | igbt | 12582476 |  |  |  |  |  |  |
| `lofo-xlstm-medium-multitask-fdp00-held_mosfet` | xlstm | medium | multitask | 0.0 | mosfet | 12582476 |  |  |  |  |  |  |
| `lofo-xlstm-medium-multitask-fdp02-held_ic` | xlstm | medium | multitask | 0.2 | ic | 12582476 |  |  |  |  |  |  |
| `lofo-xlstm-medium-multitask-fdp02-held_igbt` | xlstm | medium | multitask | 0.2 | igbt | 12582476 |  |  |  |  |  |  |
| `lofo-xlstm-medium-multitask-fdp02-held_mosfet` | xlstm | medium | multitask | 0.2 | mosfet | 12582476 |  |  |  |  |  |  |
| `lofo-xlstm-small-lm_only-fdp00-held_ic` | xlstm | small | lm_only | 0.0 | ic | 1731344 |  |  |  |  |  |  |
| `lofo-xlstm-small-lm_only-fdp00-held_igbt` | xlstm | small | lm_only | 0.0 | igbt | 1731344 |  |  |  |  |  |  |
| `lofo-xlstm-small-lm_only-fdp00-held_mosfet` | xlstm | small | lm_only | 0.0 | mosfet | 1731344 |  |  |  |  |  |  |
| `lofo-xlstm-small-lm_only-fdp02-held_ic` | xlstm | small | lm_only | 0.2 | ic | 1731344 |  |  |  |  |  |  |
| `lofo-xlstm-small-lm_only-fdp02-held_igbt` | xlstm | small | lm_only | 0.2 | igbt | 1731344 |  |  |  |  |  |  |
| `lofo-xlstm-small-lm_only-fdp02-held_mosfet` | xlstm | small | lm_only | 0.2 | mosfet | 1731344 |  |  |  |  |  |  |
| `lofo-xlstm-small-multitask-fdp00-held_ic` | xlstm | small | multitask | 0.0 | ic | 1866012 |  |  |  |  |  |  |
| `lofo-xlstm-small-multitask-fdp00-held_igbt` | xlstm | small | multitask | 0.0 | igbt | 1866012 |  |  |  |  |  |  |
| `lofo-xlstm-small-multitask-fdp00-held_mosfet` | xlstm | small | multitask | 0.0 | mosfet | 1866012 |  |  |  |  |  |  |
| `lofo-xlstm-small-multitask-fdp02-held_ic` | xlstm | small | multitask | 0.2 | ic | 1866012 |  |  |  |  |  |  |
| `lofo-xlstm-small-multitask-fdp02-held_igbt` | xlstm | small | multitask | 0.2 | igbt | 1866012 |  |  |  |  |  |  |
| `lofo-xlstm-small-multitask-fdp02-held_mosfet` | xlstm | small | multitask | 0.2 | mosfet | 1866012 |  |  |  |  |  |  |

## Best recipe per (arch, size, heads) — averaged across folds
| arch | size | heads | fdp | Top1_held_avg | top1_drop_avg | anom_AUC_held_avg |
|---|---|---|--:|--:|--:|--:|
| transformer | small | multitask | 0.2 | 0.5983 | 0.0225 | 1.0000 |
| transformer | small | multitask | 0.0 | 0.5950 | -0.0300 | 1.0000 |
| transformer | small | lm_only | 0.2 | 0.5817 | 0.0175 | 1.0000 |
| transformer | small | lm_only | 0.0 | 0.5400 | 0.0683 | 1.0000 |
| transformer | medium | lm_only | 0.0 |  |  |  |
| transformer | medium | lm_only | 0.2 |  |  |  |
| transformer | medium | multitask | 0.0 |  |  |  |
| transformer | medium | multitask | 0.2 |  |  |  |
| xlstm | medium | lm_only | 0.0 |  |  |  |
| xlstm | medium | lm_only | 0.2 |  |  |  |
| xlstm | medium | multitask | 0.0 |  |  |  |
| xlstm | medium | multitask | 0.2 |  |  |  |
| xlstm | small | lm_only | 0.0 |  |  |  |
| xlstm | small | lm_only | 0.2 |  |  |  |
| xlstm | small | multitask | 0.0 |  |  |  |
| xlstm | small | multitask | 0.2 |  |  |  |

## Final all-3 cells (no LoFO; for submission)
| cell_id | params | LM loss | wall (s) |
|---|--:|--:|--:|
| `final-transformer-medium-lm_only-fdp00-all3` | 33646080 | 0.1061 | 244.8 |
| `final-transformer-medium-lm_only-fdp02-all3` | 33646080 | 0.1061 | 247.1 |
| `final-transformer-medium-multitask-fdp00-all3` | 34177548 | 0.1069 | 342.7 |
| `final-transformer-medium-multitask-fdp02-all3` | 34177548 | 0.1070 | 340.4 |
| `final-transformer-small-lm_only-fdp00-all3` | 4238080 | 0.1061 | 65.5 |
| `final-transformer-small-lm_only-fdp02-all3` | 4238080 | 0.1062 | 83.4 |
| `final-transformer-small-multitask-fdp00-all3` | 4372748 | 0.1080 | 92.9 |
| `final-transformer-small-multitask-fdp02-all3` | 4372748 | 0.1080 | 93.5 |
| `final-xlstm-medium-lm_only-fdp00-all3` | 12051008 | 0.1098 | 517.5 |
| `final-xlstm-medium-lm_only-fdp02-all3` | 12051008 | 0.1093 | 534.0 |
| `final-xlstm-medium-multitask-fdp00-all3` | 12582476 | 0.1090 | 700.2 |
| `final-xlstm-medium-multitask-fdp02-all3` | 12582476 | 0.1112 | 707.9 |
| `final-xlstm-small-lm_only-fdp00-all3` | 1731344 | 0.1149 | 189.0 |
| `final-xlstm-small-lm_only-fdp02-all3` | 1731344 | 0.1192 | 186.9 |
| `final-xlstm-small-multitask-fdp00-all3` | 1866012 | 0.1173 | 261.5 |
| `final-xlstm-small-multitask-fdp02-all3` | 1866012 | 0.1172 | 258.4 |
