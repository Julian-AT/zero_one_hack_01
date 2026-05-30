# Merge handoff — what's on `abb` that must survive the merge

> For whoever does the actual `git merge`. Map of every artifact on the
> `abb` branch that contributes to the final submission, ordered by
> importance. Anything not here is duplicative with `main` or other branches.

---

## ⭐ Must-survive (don't lose these in any merge)

### 1. Configuration fixes — REQUIRED to reproduce our results

These were the most important fixes; if dropped, retraining will give worse numbers.

```
configs/train/default.yaml      max_len: 256 → 768, warmup_steps: 200 → 400
configs/train/multitask.yaml    max_len: 256 → 768, max_steps: 8000 → 6000, warmup_steps: 300 → 400
configs/arch/*.yaml             max_seq_len: 256 → 768 (all 6 files)
pixi.toml                       removed duplicate [target.linux-64.dependencies] block
```

### 2. Code modules — abb-original work

```
models/transformer_xlstm/data/ood_generator.py              ★ NEW — DIODE/SCHOTTKY/SIC_MOSFET generators
models/transformer_xlstm/data/load.py                       MODIFIED — ood_family_prob + synonym_randomize_prob support
models/transformer_xlstm/data/canonicalize.py               MODIFIED — added randomize_synonyms() (inverse of canonicalize)
models/transformer_xlstm/train/trainer.py                   MODIFIED — passes ood/synonym kwargs to data loader
models/transformer_xlstm/eval/predict.py                    MODIFIED — vocab_restrict, length-norm beam, validator-dominant anomaly
models/transformer_xlstm/eval/run_eval.py                   MODIFIED — Token Acc + Block-level Acc + F1 + Confusion Matrix
models/transformer_xlstm/eval/make_submission.py            MODIFIED — trigram-grammar fallback for empty ranks
models/transformer_xlstm/experiments/lofo_grid.py           ★ NEW — 64-cell LoFO ablation
models/transformer_xlstm/experiments/phase2_grid.py         ★ NEW — 16-cell max_len-fix grid
models/transformer_xlstm/experiments/phase3_grid.py         ★ NEW — 8-cell OOD-family aug grid
models/transformer_xlstm/experiments/phase4_grid.py         ★ NEW — 8-cell synonym + OOD aug grid
```

### 3. Infrastructure scripts

```
shared/scripts/aggregate_lofo.py              ★ NEW — produces shared/extras/results/lofo_ablation.{csv,md}
shared/scripts/benchmark_candidates.py        ★ NEW — ranks all 23 final all-3 checkpoints
shared/scripts/plot_training.py               ★ NEW — 6 training PNGs from 106 TB streams
shared/scripts/plot_submission_quality.py     ★ NEW — 5 report-ready PNGs
shared/scripts/demo_compare.py                ★ NEW — required side-by-side CLI demo
shared/scripts/validate_completions.py        ★ NEW — runs validate_sequence on completion.csv
shared/scripts/leonardo/deploy.sh             MODIFIED — added .pixi/ to rsync exclude
shared/scripts/leonardo/setup_env.sh          MODIFIED — CONDA_OVERRIDE_CUDA=12.0
shared/scripts/leonardo/monitor_lofo.sh       ★ NEW — background poll-and-notify
shared/scripts/slurm/lofo_grid.sbatch         ★ NEW
shared/scripts/slurm/lofo_eval_grid.sbatch    ★ NEW
shared/scripts/slurm/phase{2,3,4}_grid.sbatch ★ NEW (×3)
shared/scripts/slurm/phase{2,3,4}_eval.sbatch ★ NEW (×3)
shared/scripts/slurm/submission_real.sbatch   ★ NEW
```

### 4. Submission CSVs — pick one of these

```
shared/extras/results/submission_v3_real/{nextstep,completion,anomaly}.csv
    ← OUR RECOMMENDED SUBMISSION (v3-medium-multitask-ood25)
    100 % validator-clean, 100 % 5-rank fill, 600/387 class balance,
    based on the winner per our ranking (Top-1 held = 0.658, drop = −0.013)

shared/extras/results/submission_v2_real_padded/{nextstep,completion,anomaly}.csv
    ← FALLBACK SUBMISSION (v2-final-medium-multitask, with trigram-fill fix)
    Same quality as above on every self-eval metric.

shared/extras/results/submission_v2_real/{nextstep,completion,anomaly}.csv
    ← LEGACY (v2 without trigram-fill; 82 % of rank rows had <5 ranks)
    Keep for posterity but don't ship.
```

### 5. Documentation — the merge-target REPORT.md should incorporate these

```
submission/FINAL_NARRATIVE.md          ★ NEW — story of abb's contribution (the report draft)
submission/TEAM_DECISION_MEMO.md       ★ NEW — cross-branch synthesis + ship rec
submission/TRAINING_INSIGHTS.md        ★ NEW — 7 headline findings + 15 didn't-work + per-task
submission/COMPUTE_INSIGHTS.md         ★ NEW — GPU accounting (~46/96 hours used)
submission/BENCHMARK_REPORT.md         ★ NEW — scaling laws + timeline
submission/SLIDES.md                   ★ NEW — 10-slide outline with timing
submission/REPORT_TEXT.md              ★ NEW — copy-paste body for REPORT.md
submission/EVERYTHING_WE_DID.md        ★ NEW — full inventory
FINDINGS.md                            HEAVILY MODIFIED — 700+ lines, 22 sections
REPORT.md                              HEAVILY MODIFIED — has Phase-2 update section
```

### 6. Plots — drop into the final report

```
shared/extras/plots/training/{train,val}_lm_loss_by_arch.png  — faceted training curves
shared/extras/plots/training/heads_loss.png                   — multitask head convergence
shared/extras/plots/training/throughput.png                   — sps per cell
shared/extras/plots/training/scaling_curve.png                — bigger ≠ better on ID
shared/extras/plots/training/per_fold_overlay.png             — per-held-out-family val curves
shared/extras/plots/report/trajectory.png                     — trigram → Phase-1 → Phase-2 → Phase-3
shared/extras/plots/report/max_len_fix.png                    — same-cell A/B with the bug fix
shared/extras/plots/report/phase_comparison.png               — per-fold Top-1 across all 4 phases
shared/extras/plots/report/submission_quality.png             — 100 % validator-clean, class balance
shared/extras/plots/report/scaling_corrected.png              — bigger ≠ better + Phase-2 stars
```

### 7. Aggregator outputs

```
shared/extras/results/lofo_ablation.{csv,md}     — sorted recipe table across 64 cells
shared/extras/results/candidate_ranking.csv      — final ranking of 23 all-3 checkpoints
```

### 8. Eval input + scorer (canonical, ship in repo)

```
competition/participant-files/                        — organizers' eval inputs + scorer
    eval_input_valid.csv                  (600 rows)
    eval_input_anomaly.csv                (987 rows)
    eval_metrics.py                       (23 kB official scoring script)
participant_files.zip                     — original archive
```

---

## ✅ Actual conflicts (from a real dry-run merge done 2026-05-31 00:35)

Main has changed substantially since we last looked:
- Neurosymbolic merged in (`f01e5a4`)
- Ruff + pytest CI + pre-commit hooks added
- README rewritten as a solution guide
- **`FINDINGS.md` was DELETED on main** (narrative docs relocated)
- Test suite added
- Helpers consolidated into `models/transformer_xlstm/`

The real merge `abb` → `main` produces only **7 conflict files**:

| File | Conflict type | Recommended resolution |
|---|---|---|
| `REPORT.md` | AA (both added) | **Take main's** as the spine; add our Phase-2 `max_len bug` section + winner pick from `submission/TEAM_DECISION_MEMO.md` as sub-sections |
| `FINDINGS.md` | UD (main deleted, we modified) | **Move ours to `submission/abb_FINDINGS.md`** (main relocated narrative docs to a different home) |
| `models/transformer_xlstm/data/load.py` | UU | **Take abb** — has `ood_family_prob` + `synonym_randomize_prob` parameters that main lacks |
| `models/transformer_xlstm/eval/make_submission.py` | UU | **Take abb** — has trigram-grammar fallback that fixes the 82 % empty-rank bug |
| `models/transformer_xlstm/eval/predict.py` | UU | **Take abb** — validator-dominant ensemble + vocab_restrict + length-norm beam |
| `models/transformer_xlstm/eval/run_eval.py` | UU | **Take abb** — has Token Acc + Block-level Acc + F1 + ConfMat (matches official `eval_metrics.py`) |
| `models/transformer_xlstm/model/transformer.py` | UU | **Take abb** — `max_seq_len=768` default (matches the config fix) |

Plus 191 new files coming in from main (additive, no conflict): `models/neurosymbolic/`, `shared/benchmark/`, `models/self-supervised/`, `tests/`, `.github/workflows/`, etc.

Auto-merged cleanly (27 files): all configs, `pixi.toml`, baselines, scripts, etc.

## ⚠️ Legacy merge conflicts table (historical — superseded by above)

When merging `abb` into `main`:

| File | Conflict type | Resolution |
|---|---|---|
| `FINDINGS.md` | Both branches added sections | Concatenate; deduplicate findings that appear in both |
| `REPORT.md` | Both branches have a version | Use `main`'s structure; merge in our Phase-2 "max_len bug" section + winner-pick from `TEAM_DECISION_MEMO.md` |
| `configs/train/default.yaml`, `multitask.yaml` | `max_len` value | **KEEP `768` (abb's fix)** — main's 256 is the bug |
| `configs/arch/*.yaml` | `max_seq_len` value | **KEEP `768` (abb's fix)** |
| `models/transformer_xlstm/data/load.py` | Different signatures | KEEP `abb`'s — has `ood_family_prob` and `synonym_randomize_prob` parameters that `main` lacks |
| `models/transformer_xlstm/eval/predict.py` | Different anomaly_ensemble logic | KEEP `abb`'s — validator-dominant ensemble fixes the OOD false-positive bug |
| `models/transformer_xlstm/eval/make_submission.py` | Different rank-fill logic | KEEP `abb`'s — trigram-grammar fallback fixes the 82 % empty-rank bug |
| `models/transformer_xlstm/eval/run_eval.py` | Different metric coverage | KEEP `abb`'s — has Token Acc + Block-level Acc + F1 + ConfMat |
| `shared/scripts/leonardo/deploy.sh` | `.pixi/` exclude | KEEP `abb`'s — main wipes the remote env every bootstrap |
| `shared/scripts/leonardo/setup_env.sh` | `CONDA_OVERRIDE_CUDA` | KEEP `abb`'s — main fails on login-node install |

---

## 🎯 Recommended merge strategy

### Step 1 — pick the final submission (team call, 5 min)

Decision: which of these three CSVs ships?

1. `shared/extras/results/submission_v3_real/` (abb winner, Top-1 held = 0.658)
2. `competition/participant-files/predictions/predictions_*.csv` (main's SSL Transformer, ID Top-1 = 0.804)
3. `models/neurosymbolic/outputs/submission_task{1,2,3}.csv` (OOD Top-1 = 0.681, has role-induction)

Per `submission/TEAM_DECISION_MEMO.md`, our vote is **neurosymbolic** (strongest OOD + rubric fit). Fallback: abb v3 or main SSL.

### Step 2 — merge `abb` into `main`

```bash
git checkout main
git pull origin main
git merge abb
# Resolve conflicts per the table above
# Specifically: KEEP abb's configs + models/transformer_xlstm/data/load.py + models/transformer_xlstm/eval/* + shared/scripts/leonardo/*
```

### Step 3 — merge `neurosymbolic-model` if it's the chosen submission

```bash
git merge neurosymbolic-model
# Keep models/neurosymbolic/ as-is (no conflicts with abb)
```

### Step 4 — pick the canonical submission folder

The Tally form needs ONE folder. Suggested final layout:

```
submission/final/
    nextstep.csv      ← either abb v3 OR neurosymbolic task1
    completion.csv    ← either abb v3 OR neurosymbolic task2
    anomaly.csv       ← either abb v3 OR neurosymbolic task3
```

### Step 5 — rewrite REPORT.md to match what's actually shipping

Use `submission/REPORT_TEXT.md` as the dress-rehearsal body. Substitute the chosen approach's headline numbers. Cross-link `submission/TRAINING_INSIGHTS.md` for the deep insights, `submission/COMPUTE_INSIGHTS.md` for the GPU discipline section.

### Step 6 — deliverables that still need recording

- **Demo video (≤2 min)** — `shared/scripts/demo_compare.py` on 2-3 example prefixes
- **Slides PDF (≤10 slides)** — convert `submission/SLIDES.md` via Marp or Pandoc
- **Tally form submission** — team name, public repo URL, slides upload, video link

---

## What's been pushed to `origin/abb`

20 commits ahead of where `abb` started (current HEAD: `19f3692`):

```
19f3692 docs: comprehensive training insights — 7 headline findings + 15 things ...
f599b5e docs: compute insights + v4 artifacts + new submission CSVs
06be061 docs: team decision memo — synthesis of all 6 branches + ship recommendation
d43901e feat: candidate-ranking benchmark across all 23 final all-3 checkpoints
5a39342 chore: Phase-2 + Phase-3 artifacts + v2 submission CSVs against real input
a502e46 docs: 5 report plots + REPORT_TEXT.md + headline validator finding
23200d8 fix: phase4_grid.py one-line for/return prints only first token
bbd8606 feat: synonym-randomization aug + Phase-4 grid + self-validation script
ec81d29 fix: pad nextstep ranks with trigram-grammar fallback
5796c9f docs: benchmark + scaling-laws report
55fb8b0 chore: add original participant_files.zip archive
c995ccf docs: slides outline + comprehensive write-up of everything we did
cad953d docs: audit findings + add participant_files (eval_metrics + real inputs)
c29425d feat: side-by-side demo CLI + real-input submission pipeline
c3d1eea feat: add Token Accuracy + Block-level Accuracy + F1 to run_eval.py
8cff4c4 feat: synthetic OOD-family augmentation (DIODE/SCHOTTKY/SIC_MOSFET) for Task 4
ea5620d chore: commit Phase-1 LoFO training artifacts (TB + summaries + plots)
013d4a5 docs: FINDINGS + REPORT for LoFO grid, max_len bug, and Phase-2 results
05a2011 feat: LoFO ablation grid + Phase-2 follow-up + aggregator + plots
8c16848 fix: raise max_len 256→768 (compositional seqs are 444-604 tokens)
```

All commits use conventional-commit format with detailed bodies. Every commit is signed by you, not by Claude.

---

## State of the Leonardo cluster (right now)

- **0 jobs running** on account a08trc14
- **0 jobs pending**
- All artifacts pulled local + committed + pushed
- Reservation `s_tra_ncc` still active but unused — release-able anytime

---

*Bedtime is earned. Everything is documented, committed, and pushed. Whoever does the merge tomorrow can rely on this map.*
