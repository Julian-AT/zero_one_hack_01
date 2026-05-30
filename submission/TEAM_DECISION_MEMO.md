# Team decision memo — which submission ships?

> Author: abb (after pulling all branches + reading every approach's results).
> This is meant to be the artifact the team aligns around before submitting.
> Pulled state as of: ~2026-05-31 ~00:30 (~9.5 hours to deadline).

---

## 1. What each branch shipped (what is actually IN the repo)

| Branch | Owner | What they built | Headline result | Where the artifacts live |
|---|---|---|---|---|
| **emil** | Emil | SSL Transformer (840 LoC), OOD-family generator (DIODE/SCHOTTKY/SIC_MOSFET), augmented-vs-original ablation | ID Top-1 **0.812** in their own held-out | `tracks/industrial-infineon/scripts/train_ssl_*.py`, `ssl_results/` |
| **julian** | Julian | SSL with hybrid tokenizer, learned contrastive reranker for Task 1 | (merged into `main`) | (merged into `main`) |
| **main** | merged | Coverage-guided SSL Transformer + learned reranker (the team's current "official" pick per `main/REPORT.md`); cross-model benchmark harness; consolidated handover | ID Top-1 **0.8044** (best ID overall) | `participant_files/predictions/predictions_*.csv` |
| **neurosymbolic-model** | (Tobias?) | Symbolic engine (10 rules + role-induction anchors) + tiny constrained neural ranker (~0.68M params). Every completion rule-valid by construction. | LoFO held-out Top-1 **0.681** (best OOD), OOD anomaly F1 **100%** with role-induction | `neurosymbolic-approach/outputs/` |
| **abb** | abb | LoFO ablation grid (96 trained checkpoints), max_len=256→768 bug fix (+19pp), Phase-3 OOD-family aug, Phase-4 synonym aug, validator-clean submissions | LoFO held-out Top-1 **0.658**, Top-1 drop **−0.013** (best generalization gap) | `extras/results/submission_v2_real/`, `submission/EVERYTHING_WE_DID.md` |
| **leonardo-prep** | abb | Leonardo deployment scaffolding | (subsumed into `main` / `abb`) | (subsumed) |

---

## 2. The metric that matters — per task, who wins

### Task 1 — Next-step prediction (Top-K, MRR)

| Approach | ID Top-1 | LoFO OOD Top-1 | OOD drop | Comment |
|---|--:|--:|--:|---|
| Trigram (baseline) | 0.717 | 0.472 | +0.246 | memorizes; collapses on OOD |
| Grammar-trigram | 0.717 | — | — | +1.3pp Top-3 over raw trigram |
| **SSL Transformer + reranker (main)** | **0.8044** | not measured | unknown | **strongest ID**, OOD untested |
| Neurosymbolic — neural ranker | 0.696 | **0.681** | **+0.015** | **flattest OOD curve** |
| Neurosymbolic — PPM (no neural) | 0.696 | 0.658 | +0.037 | symbolic only |
| **abb v3-medium-multitask** | 0.670 | 0.658 | **−0.013** | best top1_drop (negative) |

**Winner per criterion:**
- Pure ID Top-1: **SSL Transformer** (0.804)
- OOD Top-1 absolute: **Neurosymbolic neural** (0.681)
- Smallest ID→OOD drop: **abb v3-medium** (−0.013, negative drop = held-out easier than ID)

### Task 2 — Sequence completion (NED, Token-Acc, Block-Acc, EM)

| Approach | ID NED | OOD NED | Rule-valid % | Comment |
|---|--:|--:|--:|---|
| **Neurosymbolic neural** | **0.18-0.26** | **0.37-0.46** | **100% by construction** | constrained decode + symbolic repair |
| Neurosymbolic PPM | 0.48-0.52 | 0.62-0.92 | 100% by construction | |
| **abb v2-medium-multitask** | n/a | **0.19** (v3) | **100% empirically** | grammar-mask decode |
| Retrieval baseline (no params) | 0.23-0.32 | 0.42-0.76 | not enforced | |
| SSL Transformer | not measured separately | not measured | not separately scored | |

**Winner:** Neurosymbolic OR abb v2/v3 — both produce 100% rule-valid completions. Neurosymbolic has the stronger guarantee (by construction); abb has it empirically via grammar-mask. NED is essentially tied at 0.19 on ID.

### Task 3 — Anomaly detection (Binary Acc, P, R, F1, AUC, Rule-Attrib)

| Approach | ID Binary Acc | ID F1 | OOD F1 (role-rename) | OOD Rule-Attrib | Comment |
|---|--:|--:|--:|--:|---|
| **Neurosymbolic** (role-induction ON) | **1.000** | **1.000** | **1.000** | **1.000** | 0 false positives both ID and OOD |
| Neurosymbolic (role-induction OFF) | 0.677 | 0.322 | 0.322 | 0.565 | role-induction is the killer feature |
| abb v3-medium (validator + heads) | 1.000 | 1.000 | not tested with renamed steps | 0.88 (one fold) | validator-dominant ensemble |
| SSL / main validator-based | ≈1.000 | ≈1.000 | not tested | ≈1.000 ID | same validator as everyone else |

**Winner:** **Neurosymbolic, decisively.** Their role-induction anchors recover anomaly recall from 19% → 100% on OOD-renamed steps — the exact failure mode for Task 4 (hidden family with potentially renamed step strings). Nobody else has this.

### Task 4 — OOD generalization (hidden family, organizer-scored post-submission)

This is the metric the rubric explicitly highlights. The team's own benchmark notes:
> "ID Top-1 is near-saturated; the **ID→OOD drop separates the approaches** — that gap is the evidence for 'learned process logic vs. memorization'."

| Approach | OOD Top-1 drop (LoFO proxy) | OOD anomaly recall | Verdict |
|---|--:|--:|---|
| **Neurosymbolic neural** | **+0.015** | **100% with role-induction** | **strongest comprehensive OOD** |
| Neurosymbolic PPM | +0.037 | 100% w/ roles | strong, pure symbolic |
| abb v3-medium | −0.013 | 100% (ID), untested on renamed steps | strongest individual drop number |
| SSL Transformer | unknown (no LoFO run) | unknown | unknown |
| Trigram | +0.246 | n/a | collapses |

**Winner:** Neurosymbolic. Two arguments: (a) role-induction is the only system that handles renamed-step failure mode for OOD anomaly; (b) drop is comparable to abb on Task 1 but the system is more comprehensively OOD-validated.

---

## 3. Findings each branch contributed (what we learned)

This is what should appear in the final REPORT.md "What worked / didn't work" section, regardless of which submission ships:

**Cross-team finding #1: ID Top-1 is near-saturated.**
A trigram-with-backoff hits 0.993 Top-5 on ID with zero parameters. The team independently arrived at this. Any approach that competes on Top-5 ID is competing on a saturated metric. The real axis is OOD.

**Cross-team finding #2: Held-out family Top-1 drop is the discriminator.**
Trigram drops +24pp, SSL Transformer unknown (didn't run LoFO), neurosymbolic +1.5pp, abb v3 −1.3pp. The drop number is what the rubric will care about for Task 4.

**Cross-team finding #3: Sequence completions need rule-validity, not just statistical correctness.**
Both neurosymbolic (by construction) and abb (empirically via grammar-mask) produce 100% validator-clean completions. Approaches that don't enforce this (greedy SSL) produce some invalid completions.

**Cross-team finding #4: Role-induction handles the OOD step-rename failure mode.**
This is the neurosymbolic team's killer contribution. No other approach addresses what happens when family-4 has a step like "DEPOSIT GATE OXIDE 2" instead of "DEPOSIT GATE OXIDE OR DIELECTRIC". Anomaly recall goes from 19% → 100% with this trick.

**Cross-team finding #5: max_len=256 is silently fatal for compositional tokenization.**
abb branch found this: every training sequence was being truncated to ~50 of ~125+ steps. The fix (max_len=768) added +19pp Top-1 held-out. Anyone training compositional models needs `max_len >= 600`.

**Cross-team finding #6: Bigger models don't help on ID.**
abb measured this across 96 checkpoints (5M / 25M / 100M transformer + xLSTM small/medium/large). All converge to LM loss 0.106 ± 0.0001 on ID. xLSTM is 3-4× slower than transformer at the same loss.

**Cross-team finding #7: Synthetic OOD-family augmentation has a sweet spot.**
DIODE/SCHOTTKY/SIC_MOSFET aug at p=0.25 slightly hurts ID LoFO (capacity dilution); at lower probabilities it should help Task 4. The team's coverage-guided augmentation on main is a more principled version of the same idea.

**Cross-team finding #8: Submission file format compliance is non-trivial.**
- nextstep.csv needs all 5 ranks filled (abb caught 82% had <5; fixed via trigram-grammar fallback)
- anomaly.csv SCORE should vary continuously for AUC (abb had a flat-binary bug, fixed)
- completion.csv predictions should be rule-valid (everyone enforces this differently)

---

## 4. My recommendation — which submission ships, and why

### Recommended: **Neurosymbolic approach**

**For all three tasks (1, 2, 3).** Use the submission CSVs from `neurosymbolic-approach/outputs/submission_task{1,2,3}.csv` if they exist, or regenerate via `exp06_make_submission.py` against the real eval inputs in `participant_files/`.

### Why

| Criterion | Winner | Reasoning |
|---|---|---|
| Task 1 OOD Top-1 | Neurosymbolic | 0.681 OOD (highest), +0.015 drop (near-flat) |
| Task 2 rule-valid completions | Tie (NS / abb) | Both 100%, NS by construction |
| Task 3 OOD anomaly | **Neurosymbolic** | Only approach with role-induction; 100% F1 OOD |
| Task 4 (the rubric-emphasized metric) | **Neurosymbolic** | Strongest OOD generalization story end-to-end |
| Engineering depth (rubric: "no LLM wrappers") | **Neurosymbolic** | Symbolic core + 0.68M params is the most defensible |
| Documentation quality | **Neurosymbolic** | Their RESULTS.md is publication-ready |
| ID Top-1 | SSL Transformer | 0.804, but ID is saturated — only matters if Task 1 is weighted ID-heavy |

### Trade-off honestly stated

We give up the ~12pp ID Top-1 advantage of the SSL Transformer (0.804 vs 0.696). But:
- Trigram already hits 0.993 Top-5 ID — most ID gain is in the noise above that floor
- The brief explicitly emphasizes generalization to family 4
- The team's own benchmark report concludes "ID→OOD drop is the discriminating axis"

### If the team disagrees and wants higher ID

Fallback: ship the **SSL Transformer + reranker** submission from `participant_files/predictions/predictions_*.csv` (main branch). It's the team's currently-staged choice and has the highest ID Top-1.

### If we want to be greedy and hedge

Submit a **per-task best-of-each** package:
- Task 1 nextstep.csv: SSL Transformer + reranker (highest ID)  → `main/participant_files/predictions/predictions_nextstep.csv`
- Task 2 completion.csv: Neurosymbolic (constrained decode, 100% rule-valid + best NED) → `neurosymbolic-approach/outputs/submission_task2.csv`
- Task 3 anomaly.csv: Neurosymbolic with role-induction (100% F1) → `neurosymbolic-approach/outputs/submission_task3.csv`

This is the strongest theoretical submission. Verify each CSV matches the documented format (`generation_rules.md §5.3`) before bundling.

---

## 5. What still needs doing before submitting

Regardless of which approach ships:

| Item | Status | Owner |
|---|---|---|
| Pick a final submission (Tally form requires one set of CSVs) | **DECISION PENDING** | team call |
| Verify chosen CSVs format-compliant with `eval_metrics.py` | ✅ done for our v2; need to verify others | whoever ships |
| README.md updated | ✅ on `main` | done |
| REPORT.md unified | ⚠️ exists on `main` but mentions only SSL Transformer | needs update if shipping neurosymbolic |
| Demo video (≤2 min) | ❌ not recorded | **MISSING — blocks submission** |
| Slides PDF (≤10 slides) | ❌ outline exists in `abb/submission/SLIDES.md` | **MISSING — blocks submission** |
| Tally form filled (team name, repo URL, slides upload, video link) | ❌ | someone |

---

## 6. Suggested team-call agenda (5-10 min)

1. **Confirm we're shipping one submission, not three.** (Tally form takes one set.)
2. **Vote on the approach:** Neurosymbolic / SSL Transformer / per-task best-of-each / our abb. My vote: neurosymbolic.
3. **Confirm CSV files are at correct paths** (the chosen submission folder must contain `nextstep.csv`, `completion.csv`, `anomaly.csv` matching the official schema).
4. **Update REPORT.md** so the narrative matches what we ship.
5. **Split the remaining work**: who records the demo video, who makes the PDF deck.

---

*Written by abb after pulling all 6 branches and reading every results report. This is a synthesis to inform the team decision, not a unilateral call.*
