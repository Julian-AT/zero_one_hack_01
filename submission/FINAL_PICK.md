# Final pick — which submission ships

> Decision memo after running every objective evaluation we can self-compute.
> Written 2026-05-31 ~01:00.

---

## TL;DR

**Ship `main`'s submission** (`participant_files/predictions/predictions_{nextstep,completion,anomaly}.csv`).

Rationale: highest ID Top-1 (0.804), all 3 CSVs format-compliant, length-correct on Task 2 (median 51 / 27 steps matches expected 50 / 25 at frac 0.6 / 0.8), 100% rank-fill on Task 1, perfect anomaly class balance.

---

## What we ran

Apples-to-apples self-eval on the real `participant_files/eval_input_*.csv` for 4 candidates:

| Submission | Provenance |
|---|---|
| `main (SSL+reranker)` | `participant_files/predictions/predictions_*.csv` from `origin/main` |
| `neurosymbolic` | `neurosymbolic-approach/outputs/submission_task{1,2,3}.csv` from `origin/neurosymbolic-model` |
| `abb v3-medium-ood25` | `extras/results/submission_v3_real/` (our winner) |
| `abb v2-medium-padded` | `extras/results/submission_v2_real_padded/` (our fallback) |

---

## Results

### Task 1 — Next-step prediction format

| Submission | 5-rank fill | Top-1 in real vocab | Avg unique ranks |
|---|--:|--:|--:|
| main           | 100.0 % | 99.7 % | 5.00 |
| neurosymbolic  | 100.0 % | 99.7 % | 5.00 |
| abb v3         | 100.0 % | 99.7 % | 5.00 |
| abb v2-padded  | 100.0 % | 99.7 % | 5.00 |

All four tie on the format-compliance metrics we can measure without ground truth. The differentiator is the documented LoFO performance:

| Submission | ID Top-1 | OOD Top-1 | drop |
|---|--:|--:|--:|
| **main** (their report) | **0.8044** | unmeasured | — |
| neurosymbolic (their report) | 0.696 | 0.681 | +0.015 |
| **abb v3** (our benchmark) | 0.670 | **0.658** | **−0.013** |

### Task 2 — Completion length compliance (the critical finding)

The spec requires "**predict only the steps after the cut point**". For `frac=0.6` partial, expected suffix is ~50 steps; for `frac=0.8`, ~25 steps.

| Submission | median pred @ 0.6 | median pred @ 0.8 | length-correct? |
|---|--:|--:|:--:|
| main           | 51 (vs 50 expected) | 27 (vs 25 expected) | **✅** |
| **neurosymbolic** | **132** (vs 50 expected) | **100** (vs 25 expected) | **❌ 2-4× over** |
| abb v3         | 51 (vs 50 expected) | 27 (vs 25 expected) | **✅** |
| abb v2-padded  | 51 (vs 50 expected) | 27 (vs 25 expected) | **✅** |

**Neurosymbolic over-predicts by 2-4×** despite producing 100% rule-valid sequences. Likely cause: their decoder doesn't stop at the natural end (e.g. SHIP LOT), or they include too much pre-cut context. Either way, the official `eval_metrics.py` will hammer them on:
- **Exact Match** (impossible at this length divergence)
- **NED** (massive insertions = large Levenshtein distance)
- **Token Accuracy** (drops as the predicted/reference length ratio diverges from 1)

Validator-clean rate (the only Task-2 signal we can independently verify): **100 % for all four submissions**.

### Task 3 — Anomaly

| Submission | valid/invalid | unique scores | rules used | class match |
|---|--:|--:|--:|:--:|
| main           | 600 / 387 | 2 | 10/10 | ✅ |
| neurosymbolic  | 600 / 387 | 2 | 10/10 | ✅ |
| abb v3         | 600 / 387 | 2 | 10/10 | ✅ |
| abb v2-padded  | 600 / 387 | 2 | 10/10 | ✅ |

All four anomaly submissions are essentially identical because all four delegate to the organizers' `validate_sequence` as the dominant signal. The only objective differentiator (per our own measurements) is **role-induction** for OOD-renamed steps — but that only affects Task 4 (the hidden family), not the Task 3 submission we're scored on directly.

### Cross-submission rank-1 agreement on Task 1

| Pair | % agree on rank-1 |
|---|--:|
| main vs neurosymbolic | 74.5 % |
| main vs abb v3 | 72.3 % |
| neurosymbolic vs abb v3 | 74.8 % |
| abb v3 vs abb v2 | 87.5 % (same recipe family) |

Three independent approaches agree on the rank-1 prediction ~73-75% of the time. The 25-27% disagreement is where the actual scoring will diverge.

---

## Final decision matrix

| Criterion | main | neurosymbolic | abb v3 |
|---|:--:|:--:|:--:|
| Format compliance — Task 1 | ✅ | ✅ | ✅ |
| Format compliance — Task 2 length | ✅ | **❌** | ✅ |
| Format compliance — Task 3 | ✅ | ✅ | ✅ |
| Validator-clean Task 2 | ✅ | ✅ | ✅ |
| Anomaly class balance | ✅ | ✅ | ✅ |
| Highest ID Top-1 | ✅ (0.804) | (0.696) | (0.670) |
| Best documented OOD | (no LoFO) | (0.681) | **✅ −0.013 drop** |
| Polished engineering (CI, tests) | ✅ | (less polished) | (no tests added) |
| Engineering depth narrative | (transformer + reranker) | ✅ (symbolic core + role induction) | (LoFO ablation + bug-fix story) |
| Risk of shipping a broken CSV | low | **HIGH (length bug)** | low |

---

## Recommendation

### Primary: ship `main`'s submission

```
nextstep.csv  ← participant_files/predictions/predictions_nextstep.csv
completion.csv ← participant_files/predictions/predictions_completion.csv
anomaly.csv   ← participant_files/predictions/predictions_anomaly.csv
```

- Highest ID Top-1 across all candidates (0.804 — the metric most likely to dominate Task 1 scoring)
- Length-correct Task 2 predictions
- Same validator-clean + class-balance as everyone else
- Already format-verified by the team's cross-model benchmark
- Lowest risk

### Fallback: ship `abb`'s v3 submission

```
nextstep.csv  ← extras/results/submission_v3_real/nextstep.csv
completion.csv ← extras/results/submission_v3_real/completion.csv
anomaly.csv   ← extras/results/submission_v3_real/anomaly.csv
```

Defensible because:
- Best documented OOD generalization (`top1_drop = −0.013`)
- Has the "max_len bug we caught and fixed" narrative that strongly matches the rubric's "honest engineering reporting" criterion
- Same format compliance as main

The trade-off: 13pp lower ID Top-1.

### Do NOT ship neurosymbolic

Despite the strongest theoretical engineering story (symbolic core + role induction for OOD-renamed steps), the **Task 2 length bug** means their completion submission will score very poorly on the official Exact Match / NED / Token Accuracy metrics, regardless of being 100% rule-valid. **Submission format compliance ate the engineering elegance.**

### Hybrid option (if the team wants to be greedy)

| File | Source | Why |
|---|---|---|
| nextstep.csv | **main** | Highest ID Top-1 |
| completion.csv | **main** | Length-correct + 100% rule-valid |
| anomaly.csv | **any** (all identical) | Same validator-dominated output everywhere |

In practice, just take all three from main. No reason to mix.

---

## What this means for the REPORT.md narrative

Whichever submission ships, the report should still cite the work all branches contributed. Specifically:

- **main's reranker work** is the production system
- **neurosymbolic's role-induction** is the cleanest engineering story for Task-4 anomaly (even though their Task-2 submission was buggy)
- **abb's LoFO grid + max_len bug postmortem** is the strongest "honest engineering" content the rubric explicitly rewards
- The team-level synthesis is the rubric-strongest single story

`submission/TEAM_DECISION_MEMO.md` already covers this; `submission/FINAL_NARRATIVE.md` and `submission/TRAINING_INSIGHTS.md` are drop-in sections for whoever assembles the final REPORT.md.

---

*Decision rationale based on objective self-eval across all 4 candidates against the real `participant_files/eval_input_*.csv`. No model retraining or evaluation against ground truth (organizers hold it).*
