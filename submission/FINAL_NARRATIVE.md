# Final narrative — `abb` branch contribution

> The full story: what we did, why we did it, how we did it. Written
> ~2026-05-31 00:30 after ~22 hours of work. Designed to be the single
> reference whoever writes the final REPORT.md can quote from directly.

---

## The reframe — why we didn't compete on the leaderboard

Before any training, we built a **trigram-with-backoff baseline in ~50 lines**. It scored:

- **Top-5 = 0.993** on in-distribution next-step
- Top-1 = 0.717, Top-3 = 0.968, MRR = 0.842

That single result reframed the hackathon. The official `eval_metrics.py` was going to score these same metrics — and an n-gram with no parameters was already at the ceiling. If we'd spent the 36 hours throwing GPU at ID Top-K, we would have moved the needle by maybe 0.1 pp from the trigram floor.

**So we asked: what isn't saturated?**

- **OOD generalization** (the brief's Task 4, scored post-submission on a hidden 4th product family). Trigram drops to Top-1 ≈ 0.47 on leave-one-family-out — a 25 pp gap. That's where the model can actually beat the n-gram.
- **Sequence completion** — trigram alone gives NED ≈ 0.99 (essentially wrong); the model can do better by maintaining structure across many steps.
- **Anomaly detection** — the organizers' `validate_sequence` is a literal oracle for the 10 documented rules. Anyone who runs it gets 100 % on ID; the question is whether your system also handles OOD-renamed steps.

**Our entire strategy followed from this reframe**: compete on OOD generalization, build the best evidence we can that the model learned *process logic* rather than memorising surface patterns.

---

## The approach — neuro-symbolic by construction, learned where it matters

We didn't pick one model — we stacked the right tool for each task:

| Layer | What it is | Used for |
|---|---|---|
| 1 | Symbolic validator (`validate_sequence` from organizers) | Anomaly detection oracle on ID; grammar mask at inference |
| 2 | Trigram-with-backoff | Top-K ID baseline; rank-fill fallback for the submission CSV |
| 3 | Grammar-constrained beam search | Task 2 — only emit tokens that don't introduce a rule violation |
| 4 | k-NN retrieval over training prefixes | Task 2 ID baseline (beats greedy generation) |
| 5 | **Compositional multitask Transformer** (RoPE + RMSNorm + SwiGLU) | The OOD lever. Word-level tokens generalise across unseen step strings |
| 6 | Multi-task heads (validity + 11-way rule-ID on `<EOS>`) | Anomaly OOD backstop; ranker calibration |
| 7 | **Synthetic OOD-family augmentation** (DIODE / SCHOTTKY / SIC_MOSFET) | Task 4 lever — forces backbone-level learning |
| 8 | Validator-dominant ensemble | Anomaly: trust the validator first, only override when learned head is *very* sure |

Each layer is independently evaluable (separate metrics in the report).

---

## The grid we ran — 104 trained cells across 4 phases

We did this incrementally, each phase informed by what the previous taught us.

### Phase 1 — the scaling story (64 cells, 5.4 A100-hours train)

A 48-cell LoFO ablation: `arch{transformer, xLSTM} × size{small, medium} × heads{LM, multitask} × family_dropout{0.0, 0.2} × fold{3 LoFO}`. Plus 16 final all-3 cells. Goal: find the recipe with the smallest ID→OOD drop.

**What we learned (and what bit us):**

- **All three transformer sizes converge to LM loss 0.106 ± 0.0001 on ID**. 5M, 25M, 100M — same loss. "Bigger isn't better" *on this task at this max_len*.
- **xLSTM converges to the same loss as transformer at the same params, 3-4× slower per step**. The sLSTM block's sequential CUDA kernel is the bottleneck. Architecture diversification didn't pay.
- **Multitask heads alone lift held-out Top-1 by +5.5 pp** vs LM-only. The 11-way rule-ID head is the de-biasing signal that pushes the model toward family-agnostic logic.
- **family_dropout is redundant with multitask heads**. Helps lm_only by 4 pp; within noise for multitask. The fdp axis was 2× over-explored.
- **MOSFET-held is the hardest LoFO fold** (Top-1 ~0.52 in Phase-1) — IGBT and IC easier (~0.59-0.60). Matches the unique MOSFET-specific epitaxy + LDD spacer blocks.
- **Anomaly ensemble achieves 100% ID** (validator oracle + heads). Anomaly is essentially solved on ID.

### Phase 2 — the bug we found in our own grid (16 cells, 2 A100-hours train)

Mid-grid, we audited the data pipeline and discovered:

> **`max_len = 256` in the train configs. Compositional sequences are 444–604 tokens (median ~467). Every training and validation sequence was being left-truncated to the *last* ~50 steps, hiding the PREFIX → CLEAN → PREP → CYCLES backbone — exactly the structural prior we wanted the model to learn.**

Fix: `max_len = 256 → 768` in all configs + 13 function-default sites. Single A/B on the same recipe (`transformer-small-multitask`, held-out MOSFET):

| Metric | Phase-1 (256) | Phase-2 (768) | Δ |
|---|--:|--:|--:|
| MOSFET held Top-1 @ frac=0.6 | 0.625 | **0.917** | **+29 pp** |
| MOSFET held Top-1 avg | 0.520 | **0.708** | **+19 pp** |
| NED held @ frac=0.6 | 0.55 | **0.27** | **−51 %** |
| Train val LM loss | 0.111 | **0.089** | −20 % |

**Single largest improvement of the entire project from a one-line config change.** Lesson learned the hard way: smoke-test sequence-length distribution against `max_len` before launching a grid.

Phase-2 also surfaced an **OOD calibration bug** in the anomaly ensemble: the better-trained validity head produced **36 false positives out of 100 held-out valid MOSFET sequences** at threshold 0.5 — AUC dropped from 1.00 (Phase-1) to 0.31 (Phase-2). Phase-1's apparent perfection was misleading because the head was undertrained. Fix: validator-dominant ensemble (only override when `P_valid < 0.1`).

### Phase 3 — Task-4 attack via OOD-family augmentation (8 cells, 1.1 A100-hours train)

Adopted teammate's prior work: **`generate_ood_families.py`** which produces DIODE / SCHOTTKY / SIC_MOSFET sequences from the existing official step vocabulary (validator-clean by construction; 150/150 generated pass `validate_sequence`).

Refactored into `models/transformer_xlstm/data/ood_generator.py` + added an `ood_family_prob` parameter to the online-generator dataloader. With probability p, draws from OOD generators and labels the family token as `<FAMILY_UNK>` — encouraging backbone-level learning rather than family-token shortcuts.

| Recipe | Avg held Top-1 | Δ vs Phase-2 |
|---|--:|--:|
| transformer-small-multitask + ood25 | 0.617 | −0.030 (capacity dilution hurt) |
| **transformer-medium-multitask + ood25** | **0.658** | **+0.030** ← winner |

**Lesson**: OOD augmentation needs enough capacity to absorb the extra distribution. Small model can't fit 3 original + 3 synthetic families; medium can.

The **v3-medium recipe is our submission winner**: held-out Top-1 = 0.658, top1_drop = **−0.013** (negative = held-out family *easier* than ID, the strongest possible OOD generalisation signal).

### Phase 4 — stacking synonym aug + OOD aug (8 cells, 1 A100-hour train)

Inverse of canonicalize: per step, with probability p, swap to a random synonym from its equivalence class (built from the existing CANONICAL dict). Teaches the model that `STRIP RESIST ≡ STRIP PHOTORESIST` and ~25 other §4-documented synonym pairs.

Stacked on top of Phase-3 OOD aug. 8 cells trained + evaluated; results integrate into final benchmark ranking.

---

## What didn't work — the rubric-rewarded section

The brief explicitly rewards honest engineering reporting. Fifteen entries we'd want in the final report:

| # | What we tried | Outcome | Lesson |
|---|---|---|---|
| 1 | `max_len = 256` default | Truncated 100 % of compositional seqs | Smoke-test seq-length-distribution vs context window before launching a grid |
| 2 | xLSTM as alternative arch | Identical LM loss, 3-4× slower | Architecture diversification didn't pay |
| 3 | Larger models (5M → 100M) | All converge to same ID loss | No parameter scaling on this task |
| 4 | family_dropout axis | Redundant with multitask heads | Drop the fdp axis when running multitask |
| 5 | Validity head threshold 0.5 | 36 % FP on OOD valid | Validator-dominant ensemble (P<0.1) |
| 6 | Greedy transformer vs k-NN | Retrieval won (pre-fix) | max_len bug was the cause |
| 7 | Phase-3 OOD aug on small | Slight LoFO regression | Use medium+ with OOD aug (capacity matters) |
| 8 | Cosine warmup = 200 (3.3 %) | Slightly under-warmed for long seq | Bumped to 400 (6.7 %) |
| 9 | multitask max_steps = 8000 | Heads converged by 4 k | Cut to 6000 → 30 % wall savings |
| 10 | Beam short-step bias | `STRIP RESIST` outranked `STRIP PHOTORESIST` | Length-normalize cumulative log-prob |
| 11 | Beam hallucinations | 3.3 % rank-1 not in real vocab | `vocab_restrict=True` in `topk_next_step` |
| 12 | Anomaly SCORE flat-binary | Destroyed ROC-AUC | Blend validator + head probability |
| 13 | `rsync --delete` + `.pixi/` | Wiped remote env every bootstrap | `--exclude='.pixi/'` |
| 14 | `pixi.toml` duplicate `[target.linux-64.dependencies]` | First bootstrap failed | Removed duplicate |
| 15 | `for ... : print(); return` | Returned after first iteration; sbatch ran `python` (no args) | Split across lines (basic Python idiom) |

---

## Compute discipline

**~46 of 96 A100-hours used (~48 % of budget).**

| Workload | A100-hours |
|---|--:|
| Training (104 cells across 4 phases) | 10.4 |
| Eval (per-cell, ~30 min average) | ~33.7 |
| Submission CSV generation (3 runs) | ~1.7 |
| **Total** | **~46** |

**62 % of compute was eval, not training.** Compositional beam-search inference is 2-3 s per example. *Lesson for next hackathon*: use `--max-examples 50` for grid-sweep eval; reserve n=100 for final 1-2 candidate models.

---

## What we shipped — concrete artifacts on `abb`

**Code:**
- `models/transformer_xlstm/data/{tokenizer, load, validator, corrupt, canonicalize, physics, ood_generator}.py`
- `models/transformer_xlstm/model/{transformer, xlstm_model, heads, registry}.py`
- `models/transformer_xlstm/train/{trainer, launch, losses, tracking}.py`
- `models/transformer_xlstm/eval/{predict, run_eval, make_submission, simulate_eval_input, validate_submission}.py`
- `models/transformer_xlstm/experiments/{lofo_grid, phase2_grid, phase3_grid, phase4_grid}.py`

**Infrastructure:**
- `shared/scripts/leonardo/{deploy, setup_env, monitor_lofo}.sh`
- `shared/scripts/slurm/{train, grid, eval, multitask, submission, submission_real, lofo_grid, lofo_eval_grid, phase{2,3,4}_grid, phase{2,3,4}_eval}.sbatch`
- `shared/scripts/{aggregate_lofo, benchmark_candidates, plot_training, plot_submission_quality, demo_compare, validate_completions}.py`

**Configs:**
- `configs/arch/{transformer, xlstm}_{small, medium, large}.yaml` (all bumped to `max_seq_len: 768`)
- `configs/train/{default, multitask}.yaml` (both `max_len: 768`)
- `configs/token/{compositional, step}.yaml`

**Artifacts:**
- 104 trained checkpoint summaries in `shared/extras/checkpoints/*/summary.json`
- 106 TB event streams in `shared/extras/logs/tb/`
- 88 eval metrics in `shared/extras/results/eval/*/metrics.{json,md}`
- 7 EDA plots in `shared/extras/eda/`
- 6 training plots in `shared/extras/plots/training/`
- 5 report plots in `shared/extras/plots/report/`
- 3 candidate submission folders: `submission_v2_real`, `submission_v2_real_padded`, `submission_v3_real` (the winner) — each with format-compliant 3-CSV bundle against the real eval inputs in `competition/participant-files/`

**Documentation (in `submission/` for the team merger):**
- `submission/EVERYTHING_WE_DID.md` — full inventory (618 lines)
- `submission/TEAM_DECISION_MEMO.md` — cross-branch synthesis + ship recommendation
- `submission/TRAINING_INSIGHTS.md` — comprehensive insights for REPORT.md
- `submission/COMPUTE_INSIGHTS.md` — GPU usage accounting
- `submission/BENCHMARK_REPORT.md` — scaling laws + timeline
- `submission/SLIDES.md` — 10-slide outline with 3-min timing
- `submission/REPORT_TEXT.md` — dress-rehearsal text for REPORT.md
- `submission/FINAL_NARRATIVE.md` — this file
- `FINDINGS.md` — running log (700+ lines across 22 sections)
- `REPORT.md` — public report (extended Phase-2 section + bug postmortem)

---

## Submission CSV status — what's actually ready to ship

| Submission | Source checkpoint | Status | 5-rank fill | Validator-clean | Class balance |
|---|---|---|--:|--:|--:|
| `submission_v2_real` | `v2-final-transformer-medium-multitask-all3` | ✅ ready | 18 % | **100 %** | ✓ 600/387 |
| **`submission_v2_real_padded`** | same, **with trigram-grammar fallback** | **✅ ready** | **100 %** | **100 %** | ✓ 600/387 |
| **`submission_v3_real`** | **`v3-final-transformer-medium-multitask-ood25-all3` ← winner** | **✅ ready** | **100 %** | **100 %** | ✓ 600/387 |

All three are bundled correctly per `generation_rules.md §5.3`. The team should pick one and rename for the Tally form upload.

---

## Why the v3 (ood-aug, medium) is our recommendation for the submission

| Criterion | v3 winner | Why |
|---|--:|---|
| Held-out Top-1 (LoFO avg) | **0.658** | best in our `benchmark_candidates.py` ranking |
| `top1_drop` (ID → OOD) | **−0.013** | negative — held-out *easier* than ID; backbone learned, not memorised |
| NED held-out average | **0.192** | lowest of all our candidates (vs 0.353 for v2) |
| Anomaly ROC-AUC ID | **1.000** | perfect on ID |
| Completion validator-clean | **100 %** | 600/600 across all 3 families per organizers' `--validate` script |
| Submission format | ✅ | 5 ranks per row, perfect class balance, all 10 rules used |
| Recipe explanation | Compositional + multitask + OOD-family aug | matches the brief's stretch goals |

---

## What we'd do with another 36 hours

- **PRM** (Process Reward Model) — train a per-prefix `P(completable to valid)` head from `validate_sequence.step_index` labels. Re-rank beam search by `LM_logit + α·PRM`. Targets Task 2 NED specifically.
- **Physics-feature injection** — parsed into `data/processed/physics_features.json`, not yet wired into the embedding. A `Linear(10, d_model) + missingness mask` would let the model place unseen step strings in physics-feature space — biggest unexplored lever for Task 4.
- **Tune Phase-3 OOD aug probability** — current 0.25 is slightly too much capacity dilution for the small model; sweep over 0.10 / 0.15 / 0.20 to find the sweet spot.
- **A 2-min demo video** using `shared/scripts/demo_compare.py` showing baseline-vs-trained on 2-3 example prefixes.

---

*This is the abb-branch narrative. Other branches (main, neurosymbolic-model, emil, julian) contributed complementary work — see `submission/TEAM_DECISION_MEMO.md` for the cross-branch synthesis. Bedtime.*
