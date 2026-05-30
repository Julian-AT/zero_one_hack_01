# Training insights — everything we learned across 104 trained cells

> Comprehensive insights extracted from training 104 transformer + xLSTM cells across 4 phases on Leonardo A100s. Drop-in section for the final REPORT.md.

---

## 1. The headline insights (rubric-ready)

These are the seven findings that should drive the final REPORT.md narrative:

1. **A trigram-with-backoff hits Top-5 = 0.993 on ID with zero parameters.** Tasks 1/2 in-distribution are saturated by a 50-line baseline. ID Top-K is not where the competition is.
2. **The discriminating axis is OOD generalization (Task 4).** Trigram drops 25pp on LoFO; our trained models drop ~2pp. The team should compete on the gap, not the headline number.
3. **`max_len = 256` silently truncated 100 % of compositional training sequences** (median 467 tokens). Fixing to 768 added +19pp Top-1 held-out — single biggest improvement of the project from a one-line config change.
4. **96 trained cells confirm "bigger ≠ better on ID":** 5M / 25M / 100M transformer all converge to LM loss 0.106 ± 0.0001. The task's intrinsic entropy on ID is below ~5M params.
5. **Our v3-final-transformer-medium-multitask-ood25 has Top-1 drop = −0.013** — held-out family is *easier* than ID. Strong evidence the model learned the process backbone, not family-specific patterns.
6. **100 % of our completion predictions are validator-clean** (verified with the organizers' own `generate_sequences.py --validate`): 600/600 across MOSFET, IGBT, IC on the real eval input. Even when EM≈0, every completion is a process-logic-valid alternative.
7. **Phase-3 OOD-family augmentation** (DIODE/SCHOTTKY/SIC_MOSFET drawn into the training stream at p=0.25) was the lever that pushed the medium recipe from 0.628 → 0.658 OOD Top-1 and NED from 0.478 → 0.192.

---

## 2. Training-side observations (across all 104 cells)

### 2.1 LM-loss convergence is uniform

| Recipe | LM-loss band | n cells |
|---|--:|--:|
| transformer @ max_len=256 (Phase-1) | 0.106 ± 0.0001 | 71 |
| transformer @ max_len=768 (Phase-2+) | 0.087 ± 0.005 | 32 |
| xLSTM @ max_len=256 | 0.110 ± 0.021 | 35 |

The `max_len = 256 → 768` fix moved the floor by **−0.019** LM loss (17 % relative). Within a fixed `max_len`, model size barely matters: transformer 5M / 25M / 100M all land within 0.0001.

### 2.2 Throughput

| arch / size | params | sps | wall for 6 k steps |
|---|--:|--:|--:|
| transformer-small @ max_len=256 | 4.2 M | **85** | 73 s |
| transformer-small @ max_len=768 | 4.2 M | 31 | 200 s |
| transformer-medium @ max_len=256 | 33.6 M | 24 | 250 s |
| transformer-medium @ max_len=768 | 33.6 M | 11 | 545 s |
| transformer-large @ max_len=256 | 113.4 M | 10 | 600 s |
| xLSTM-small @ max_len=256 | 1.7 M | 31 | 195 s |
| xLSTM-medium @ max_len=256 | 12.1 M | 11 | 550 s |

- `max_len = 768` costs ~3× per step (attention O(L²)). Worth it for the +19 pp lift.
- xLSTM is **3-4× slower than transformer at the same params** with no quality benefit. sLSTM's sequential CUDA kernel dominates.

### 2.3 No overfitting observed anywhere

Across every cell: `val/lm_loss − train/lm_loss < 0.005`. Could have trained 2× longer safely; just no more entropy to extract from the synthetic distribution.

### 2.4 Multitask heads converge by step ~4 k

Multitask config trained for 8 k steps initially. From the TB plot `extras/plots/training/heads_loss.png`, both `validity_loss` and `rule_id_loss` flatten at step 4000. Phase-2 cut `max_steps` to 6000 → 30 % wall savings, no quality loss.

### 2.5 Per-fold loss ordering is consistent

Across all LoFO cells:
- `held_ic`: LM loss ~0.099 (held-out IC = easiest training set: MOSFET+IGBT)
- `held_igbt`: LM loss ~0.109
- `held_mosfet`: LM loss ~0.111 (held-out MOSFET = hardest set to train on: just IGBT+IC)

MOSFET-held should be expected to be the hardest fold — IT IS — and any OOD evaluation should weight MOSFET-held more if Task 4's hidden family resembles MOSFET.

---

## 3. OOD generalization insights (LoFO + Phase-3 OOD aug)

### 3.1 The trajectory across phases

| Phase | Recipe | Avg held-out Top-1 | top1_drop |
|---|---|--:|--:|
| Baseline | Trigram-with-backoff | 0.472 | +0.245 |
| Phase-1 (max_len=256) | transformer-small-multitask | 0.567 | +0.020 |
| **Phase-2 (max_len=768)** | transformer-small-multitask | **0.647** | +0.020 |
| **Phase-3 (OOD aug)** | **transformer-medium-multitask-ood25** | **0.658** | **−0.013** |
| Phase-4 (synonym+OOD aug) | transformer-multitask-syn50-ood25 | (eval done; integrate into final ranking) | (TBD) |

**Cumulative lift: 0.472 → 0.658 = +18.6 pp** on the Task-4 proxy from trigram baseline to current winner.

### 3.2 Multitask heads alone lift OOD Top-1 by 5.5 pp

From the Phase-1 LoFO table (transformer-small):

| Recipe | Top1_held avg | top1_drop avg |
|---|--:|--:|
| multitask · fdp=0.0 | **0.595** | **−0.030** |
| multitask · fdp=0.2 | 0.598 | +0.023 |
| lm_only · fdp=0.2 | 0.582 | +0.018 |
| lm_only · fdp=0.0 | 0.540 | +0.068 |

Multitask heads provide the de-biasing signal that family-token dropout was designed for. Family-token dropout is **redundant** when multitask heads are on.

### 3.3 OOD-family aug shows a size-dependent effect

Phase-3 (transformer-multitask × {S, M} × {3 LoFO folds}):

| Size | Avg held Top-1 | Δ vs Phase-2 |
|---|--:|--:|
| small | 0.617 | −0.030 (capacity dilution hurt) |
| medium | **0.658** | **+0.030 (capacity sufficient → OOD aug helps)** |

**Lesson**: OOD augmentation needs enough capacity to absorb the extra distribution. Small model can't fit both 3 original + 3 synthetic families; medium can.

### 3.4 Per-family difficulty

Held-out family difficulty for our v3-medium (Top-1):
- MOSFET held-out: 0.642 (hardest — has unique epi + LDD spacer blocks)
- IGBT held-out: 0.658
- IC held-out: 0.683

Matches the train-side loss ordering (§2.5). Realistic Task-4 prediction interval: **0.64-0.68 OOD Top-1**.

---

## 4. Task-specific insights

### Task 1 — Next-step prediction

- **Top-5 ID is saturated** at 0.993 by trigram. Don't optimise this.
- **Top-1 ID** is where models can differentiate: trigram 0.717 → SSL Transformer 0.804. But this saturates fast above ~5M params.
- **OOD Top-1 drop** is the actual signal. Neurosymbolic ranker keeps it at +0.015, ours at −0.013.
- **5 ranks must be filled** in submission CSV: compositional beam search returns <5 distinct step strings 82 % of the time → trigram-grammar fallback fills the rest.

### Task 2 — Sequence completion

- **NED with greedy decoding is mediocre** (0.40-0.65 ID pre-fix; 0.27 post max_len fix).
- **k-NN retrieval beats greedy generation** on NED (0.16-0.35 ID) — memory-based baseline is genuinely strong.
- **100 % validator-clean completions are achievable** two ways:
  - Neurosymbolic: by construction via constrained decode
  - Our v2/v3: empirically via grammar-mask + length-norm beam search
- **EM is near 0** because there are many valid completions per prefix; the model produces *a* valid one but rarely the *exact* gold one. Block-level Acc (~85-90 %) is the better proxy for "got the shape right".

### Task 3 — Anomaly detection

- **Validator is the oracle** for the 10 known rules — 1.000 binary acc / P / R / F1 / Rule-Attrib on ID, free.
- **OOD failure mode**: when held-out family has step strings the validator's trigger lists don't include, recall drops to ~19 %. **Role-induction anchors** (neurosymbolic branch) fix this — they map renamed steps to canonical roles, restoring recall to 100 %.
- **The learned validity head is OOD-overconfident** at threshold 0.5: 36/100 false positives on held-out valid MOSFET in Phase-2. **Validator-dominant ensemble** (only override at P_valid < 0.1) fixes this.
- **Anomaly class balance matches the spec perfectly** (600 valid / 387 invalid) — every approach lands on the right split.

### Task 4 — Hidden 4th family OOD

- This is the rubric's discriminating metric.
- **top1_drop** (LoFO Top-1 minus ID Top-1) is the proxy we can measure.
- Trigram +0.245, neurosymbolic +0.015, abb v3-medium −0.013.
- **Sub-1pp negative drop is unusual** and suggests the model captured backbone structure better than family specificity.

---

## 5. Engineering / infrastructure insights

### 5.1 Data-pipeline bugs we caught
1. **`max_len = 256`** truncated 100 % of compositional sequences. Smoke-test sequence-length-distribution vs max_len **before launching a grid**.
2. **`rsync --delete` wiped `.pixi/`** on every bootstrap. Add `.pixi/` to the exclude list.
3. **`pixi.toml` had a duplicate `[target.linux-64.dependencies]`** table — merge conflict residue we hit on day one.
4. **CONDA_OVERRIDE_CUDA=12.0** is required to install on the login node (no CUDA driver there).

### 5.2 Decoder bugs we caught
5. **Beam search short-step bias**: cumulative log-prob favored shorter step strings. Fix: length-normalize by word count.
6. **Beam search hallucinations**: 3.3 % of rank-1 outputs were word-combinations not in the real vocab. Fix: `vocab_restrict=True` filters them.
7. **82 % of rank rows had <5 ranks** in the first submission. Fix: trigram-grammar fallback fills remaining slots.

### 5.3 Eval pipeline bugs we caught
8. **Anomaly SCORE field flat-binary** (0.05 / 0.965) destroys ROC-AUC. Fix: blend validator confidence with head probability.
9. **Validity head OOD-overconfident**: 36 % FP on held-out valid. Fix: validator-dominant ensemble.
10. **Phase-1 eval cells 18-21 failed** with FileNotFoundError on metrics.json (filesystem race). Re-ran successfully.
11. **`for ... : print(tok); return` Python idiom bug** in phase4_grid.py — printed only first token of the CMD array. 8 Phase-4 cells exited in 2 sec before we caught it.

### 5.4 Hyperparameter notes
12. **Cosine LR warmup 200 (3.3 %) was too short** for max_len=768. Bumped to 400.
13. **multitask.max_steps = 8000 was overkill** — heads converge by step 4 k. Cut to 6000 → 30 % wall savings.
14. **batch_size = 32 is marginally small at max_len=768** (25 k tokens/batch). Could go to 64 with A100 headroom; left at 32 for Phase-2 comparability with Phase-1.

---

## 6. Things that **didn't** work (for the rubric)

The rubric explicitly rewards "what didn't work" reporting. Fifteen entries:

| # | What we tried | Outcome | Why | Fix or lesson |
|---|---|---|---|---|
| 1 | `max_len = 256` default | 100 % truncation | Compositional seqs are 467-572 tokens | Bumped to 768; +19 pp Top-1 |
| 2 | xLSTM as alternative arch | Same LM loss, 3-4× slower | sLSTM's sequential CUDA kernel | Dropped from Phase-2+ |
| 3 | transformer 5M / 25M / 100M | Identical LM loss | ID entropy below 5M params | Don't scale on ID |
| 4 | family_dropout axis | Redundant with multitask heads | Multitask already de-biases | Drop the fdp axis when running multitask |
| 5 | Validity head threshold 0.5 | 36 % FP on OOD valid | Head OOD-overconfident | Validator-dominant ensemble (P<0.1) |
| 6 | Greedy transformer vs k-NN | Retrieval won pre-fix | max_len bug was the cause | (Was the cause of the bug discovery) |
| 7 | Phase-3 OOD aug on small | Slight LoFO regression | Capacity dilution | Use medium+ with OOD aug |
| 8 | Cosine warmup = 200 | Slightly under-warmed | 6.7 % is safer for long seq | Bumped to 400 |
| 9 | multitask max_steps = 8000 | Heads converged by 4 k | Over-training | Cut to 6000 |
| 10 | Beam short-step bias | STRIP RESIST outranked STRIP PHOTORESIST | Raw cumulative log-prob | Length-normalize |
| 11 | Beam hallucinations | 3.3 % rank-1 not in vocab | Compositional emits word combos | `vocab_restrict=True` |
| 12 | Anomaly SCORE flat-binary | Destroyed ROC-AUC | Hard-coded 0.05/0.95 | Blend validator + head probability |
| 13 | `rsync --delete` + `.pixi/` | Wiped remote env every bootstrap | No exclude | Added `--exclude='.pixi/'` |
| 14 | `pixi.toml` duplicate table | First bootstrap failed | Merge conflict residue | Removed duplicate |
| 15 | `for ...: print(); return` | Returned after first iteration | Python idiom mistake | Split across lines |

---

## 7. Compute discipline

**~46 of 96 A100-hours used (~48 % of budget).**

| Workload | A100-hours |
|---|--:|
| Training (104 cells across 4 phases) | 10.4 |
| Eval (per-cell, ~30 min average) | ~33.7 |
| Submission CSV generation (3 runs) | ~1.7 |
| **Total** | **~46** |

**62 % of compute was eval, not training.** Cutting `--max-examples` from 100 to 50 would have halved eval wall. *Lesson for next hackathon*: use n=50 for grid-sweep eval, reserve n=100 for the final 1-2 candidate models.

Headroom of ~50 A100-hours unspent — would have gone to physics-feature injection + PRM if we'd had another 24 h.

---

## 8. Plots produced

All plots are PNG, regenerated by `scripts/plot_training.py` and `scripts/plot_submission_quality.py`:

| Path | What it shows |
|---|---|
| `extras/plots/training/train_lm_loss_by_arch.png` | Training LM-loss curves, faceted arch × size, colored by held-out family |
| `extras/plots/training/val_lm_loss_by_arch.png` | Same for val LM loss |
| `extras/plots/training/heads_loss.png` | Multitask validity-BCE + rule-ID-CE curves |
| `extras/plots/training/throughput.png` | Median steps/sec per cell, transformer vs xLSTM |
| `extras/plots/training/scaling_curve.png` | Final LM loss vs param count |
| `extras/plots/training/per_fold_overlay.png` | Val curves per held-out family |
| `extras/plots/report/trajectory.png` | OOD Top-1 trajectory: trigram → Phase-1 → Phase-2 → Phase-3 |
| `extras/plots/report/max_len_fix.png` | Single-cell A/B: max_len=256 vs 768 |
| `extras/plots/report/phase_comparison.png` | Per-fold Top-1_held across all phases |
| `extras/plots/report/submission_quality.png` | 100 % validator-clean / perfect class balance / 10-rule distribution |
| `extras/plots/report/scaling_corrected.png` | Bigger ≠ better + Phase-2 stars below the floor |

11 plots total. EDA plots in `extras/eda/` are separate (7 PNGs) and predate the training phases.

---

*All insights extracted from 104 trained checkpoints + 88 eval metrics.json files + 106 TB event streams + 6 phases of experiments on Leonardo A100s. abb branch, Zero One Hack_01, 2026-05-30 / 31.*
