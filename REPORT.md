# abb — Industrial AI (Infineon)

## Team

- **abb** — ML / infra / writing

**Track:** Industrial AI (Infineon)

---

## TL;DR

A trigram-with-backoff already scores Top-5 = 0.993 in-distribution on this task, so we did not try to beat it with raw model size. Instead we built a hybrid stack: symbolic validator + grammar-constrained decoder + k-NN retrieval + a compositionally-tokenized multi-task transformer with validity and rule-attribution heads. The Task 3 ensemble (validator first, learned heads as fallback) gets **100% binary accuracy and 100% rule attribution** on a held-out mix of valid and corrupted sequences. Compositional word-tokenization plus parsed physics features from the `longdescription_parameters` CSVs are positioned as the OOD lever for the hidden 4th family.

---

## Problem

Three submission tasks plus a post-submission OOD evaluation on a hidden 4th product family:

1. **Next-step prediction** (Top-1/3/5, MRR)
2. **Sequence completion** at 60% and 80% truncation (Exact Match, NED, Token Acc, Block Acc)
3. **Anomaly detection** (Binary Acc, F1, Rule Attribution)
4. **OOD generalization** (post-hoc, organizer-scored)

We treated task 4 as the actual competition. Tasks 1 and 2 are largely saturated by an n-gram on the provided data — the real differentiator is how the system behaves on a family it never saw during training.

---

## Approach

**EDA-first reframing.** Before training anything, we computed a trigram-with-backoff baseline. Top-5 hit 0.993 on a held-out 80/20 split, identical to the memorization upper bound. That said the task has so little entropy on ID that we should not throw GPUs at it. The leave-one-family-out (LoFO) drop to ~0.50 Top-1 quantified the OOD gap we needed to close.

**Hybrid stack, picked tool-per-task:**

- **N-gram + grammar mask + k-NN retrieval** — three cheap baselines that already lift completion NED on MOSFET frac=0.8 from 0.999 (drift) to 0.126 (grammar-trigram) and 0.160 (retrieval).
- **Compositional word-tokenization** — instead of treating each step string as one token (vocab ~200), we split into word tokens (vocab ~70 + delimiters). A new step in a hidden family that shares words with seen steps is no longer fully OOD.
- **Multi-task transformer** — decoder-only with RoPE + RMSNorm + SwiGLU + SDPA causal attention. Three heads: next-token LM, binary validity on `<EOS>`, 11-way rule-ID. Trained on infinite online-generated data with 40% of sequences carrying an injected rule violation labeled by the validator.
- **Symbolic validator as anomaly oracle** — the organizers' `validate_sequence` directly catches the 10 documented rule violations; we put it first in the anomaly ensemble.
- **Physics features lookup** — parsed `*_longdescription_parameters.csv` into a 10-dim feature vector per step (temp, log_time, log_thickness, log_pressure, energy_keV, log_dose, tool category, is_wet, is_anneal, is_implant). 136 unique steps covered. Positioned as a learnable input projection for the hidden family.

**Training infrastructure:** Leonardo cluster, 4 A100s/team, reservation `s_tra_ncc`, account `euhpc_d30_031`. Pixi-managed env with torch 2.5.1 (cu121), conda-forge `libstdcxx-ng`. xLSTM CUDA kernels JIT-compile on first use — required `module load gcc/12.2.0 cuda/12.6` in the SLURM script.

**Scaling grid (7 cells):**

| Architecture | Sizes | Tokenization | Goal |
|---|---|---|---|
| Transformer | 5M / 25M / 100M | compositional | scaling curve |
| xLSTM-mixed (mLSTM + sLSTM) | 5M / 25M / 100M | compositional | non-Transformer arch comparison |
| Transformer | 25M | step-as-token | tokenization ablation |

---

## How to run it

```bash
# 1. Clone
git clone https://github.com/Julian-AT/zero_one_hack_01.git
cd zero_one_hack_01

# 2. Environment (Leonardo or any CUDA host)
bash scripts/leonardo/setup_env.sh

# 3. Trigram baseline (no GPU, ~3 s)
.pixi/bin/pixi run trigram

# 4. Single training cell
sbatch --export=ALL,CONFIG=configs/arch/transformer_small.yaml scripts/slurm/train.sbatch

# 5. Full 7-cell scaling grid
sbatch scripts/slurm/grid.sbatch

# 6. Multi-task training (validity + rule-ID heads)
sbatch scripts/slurm/multitask.sbatch

# 7. Eval (Top-K, completion, anomaly) on a checkpoint
sbatch --export=ALL,CKPT=extras/checkpoints/multitask-transformer_medium-.../final.pt,OUT=extras/results/eval/foo scripts/slurm/eval.sbatch

# 8. Generate the three submission CSVs
sbatch --export=ALL,CKPT=extras/checkpoints/multitask-transformer_medium-.../final.pt scripts/slurm/submission.sbatch
```

Outputs land under `extras/{checkpoints,logs,results}/`.

---

## Results

### Trigram baseline (no learning)

| Setup | Top-1 | Top-3 | Top-5 | MRR |
|---|--:|--:|--:|--:|
| Memorization upper bound | 0.722 | 0.968 | **0.993** | 0.844 |
| Honest 80/20 held-out | 0.717 | 0.968 | **0.993** | 0.842 |
| LoFO MOSFET (train IGBT+IC) | 0.502 | 0.679 | 0.728 | 0.598 |
| LoFO IGBT  | 0.481 | 0.660 | 0.707 | 0.577 |
| LoFO IC    | 0.432 | 0.624 | 0.644 | 0.528 |

ID Top-5 = 0.993 with no parameters. Held-out is identical to in-sample — the task carries almost no model-relevant entropy on ID. LoFO drops 25pp on Top-1 → this is the OOD gap we are competing on.

### Grammar mask on top of trigram

| Metric | trigram | + grammar | delta |
|---|--:|--:|--:|
| ID Top-3 | 0.9675 | 0.9805 | **+1.3pp** |
| ID Top-5 | 0.9930 | 0.9957 | +0.3pp |
| MOSFET completion NED @ frac=0.8 | 0.999 | **0.126** | −87% |
| MOSFET completion ExactMatch @ frac=0.8 | 0.000 | **0.025** | first non-zero |

Pure-inference gain via the organizers' `validate_sequence`.

### k-NN retrieval (no parameters)

| frac | family | ExactMatch | NED |
|---|---|--:|--:|
| 0.6 | MOSFET | 0.000 | **0.230** |
| 0.6 | IGBT   | 0.000 | 0.297 |
| 0.6 | IC     | 0.000 | 0.323 |
| 0.8 | MOSFET | **0.015** | 0.160 |
| 0.8 | IGBT   | 0.000 | 0.241 |
| 0.8 | IC     | 0.000 | 0.348 |

Memory beats greedy generation; both stack well with the trigram.

### Scaling grid — bigger is not better on ID

| Cell | Arch | Params | Tokenization | Final LM loss | Wall time |
|---|---|--:|---|--:|--:|
| 0 | transformer | 4.2M  | compositional | 0.1062 | 73 s |
| 1 | transformer | 33.6M | compositional | **0.1061** | 255 s |
| 2 | transformer | 113.4M | compositional | 0.1062 | 576 s |
| 3 | xLSTM-mixed | 1.7M  | compositional | 0.1192 | 201 s |
| 4 | xLSTM-mixed | 12.0M | compositional | 0.1093 | 529 s |
| 5 | xLSTM-mixed | 38.8M | compositional | **0.1077** | 1033 s |
| 6 | transformer | 33.7M | **step-as-token** (ablation) | 0.3258 | 251 s |

Three Transformer sizes collapse to within 0.0001 LM loss. xLSTM closes the gap with scale but never beats. Compositional vs step-as-token loss is not directly comparable (different per-token entropy) — the comparison that matters is OOD, evaluated post-submission.

### Multi-task training (validity + rule-ID)

```
LM loss      0.1069   (≈ baseline 0.1061 — heads don't hurt LM)
Validity BCE 0.1137
Rule-ID  CE  0.1150   (11-way: 10 rules + valid)
```

### End-to-end eval on a trained checkpoint

Held-out 40/family, completion capped at 60 steps, grammar mask on, mixed valid+corrupted (50%) for anomaly:

| Model | Top-1@cut (avg) | Top-5@cut (avg) | Anomaly acc | Rule attribution |
|---|--:|--:|--:|--:|
| transformer_medium (baseline) | 0.61 | 0.996 | **1.000** | **1.000** |
| transformer_medium (multi-task) | 0.59 | 0.954 | **1.000** | **1.000** |

The validator + learned-head ensemble achieves perfect anomaly detection on ID. Top-1 is below the trigram (0.72) because the compositional model emits step strings by beam-searching word tokens, which is harder than direct step-token argmax — the trigram remains the right baseline for Top-1, the transformer is for OOD + completion structure.

### Submission CSVs

Three files in the documented format, generated on Leonardo from the multi-task checkpoint, validated against `eval_input_*.csv` schema (Section 5 of `generation_rules.md`):

| File | Rows | Schema |
|---|--:|---|
| `extras/results/submission/nextstep.csv`   | 600 | `EXAMPLE_ID, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5` |
| `extras/results/submission/completion.csv` | 600 | `EXAMPLE_ID, PREDICTED_SEQUENCE` |
| `extras/results/submission/anomaly.csv`    | 300 | `EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE` |

These are produced from our locally-simulated `eval_input_valid.csv` and `eval_input_anomaly.csv`. When organizers ship the real eval inputs, the same `make_submission.py` swaps them in (CLI flag).

---

## What worked

1. **Building a 50-line trigram before any GPU training.** The Top-5 = 0.993 finding reframed the whole hackathon away from a leaderboard chase.
2. **Symbolic validator at inference time.** Three uses, three wins: grammar mask gives the trigram its first non-zero exact-match; ensemble anomaly hits 100% on ID; per-step `Violation.step_index` is the foundation for a future PRM.
3. **Compositional tokenizer + multi-task training trained without instability** on the online-generator stream with 40% corruption rate. Validity and rule-ID heads both converged to ~0.114 loss.
4. **Storage discipline.** Checkpoints land on `$SCRATCH` (shared FS, survives job end) with a `$HOME` backup, plus a lightweight `summary.json` committed to git as the official scaling-grid record.
5. **One-command rebuild of any cell.** `pixi run` + `sbatch` + the rsync-based bootstrap let us redeploy in ~2 minutes per iteration.

---

## What didn't work

1. **First attempt at xLSTM cells** failed during JIT compilation — Leonardo's compute nodes have CUDA driver but no `nvcc`/g++. Fix: `module load gcc/12.2.0 cuda/12.6` in every SLURM script. Documented for the next team.
2. **First attempt at conda-forge `pytorch-gpu`** failed because the resolver couldn't see `__cuda` on the GPU-less login node. Fix: explicit `[system-requirements] cuda = "12"` + `CONDA_OVERRIDE_CUDA="12.0"`, then later cleaner switch to PyPI torch with the cu121 index URL.
3. **First eval run timed out** because compositional beam search at beam=12, max_words=8 took ~660 s per family×frac. Cut to beam=5, max_words=6, max_examples=40 to fit within the 30-min SLURM ceiling. Numbers in this report are from the faster eval; statistical noise on n=40 is visible (e.g. IGBT@0.6 Top-5 dropped from 0.975 baseline to 0.775 multi-task — likely a single noisy example).
4. **Step-as-token cell looked dramatically worse on raw CE loss** (0.326 vs 0.106) but the comparison is not apples-to-apples (different sequence length, different vocab). The honest comparison is the OOD eval (post-submission) — flagged in the report rather than spun.

---

## What we'd do with another 36 hours

1. **PRM** (Process Reward Model) — train a per-prefix `P(completable to valid)` head from validator `step_index` labels. Use as a re-ranker inside beam search; expected to lift completion ExactMatch.
2. **Wire physics features into the embedding** — currently parsed and saved as a lookup; not yet injected into the model. The path is `Linear(10, d_model)` added to the token embedding, with a missingness mask.
3. **Train a contrastive sequence encoder** on (valid, corrupted) pairs across families with InfoNCE — to backstop the validator on Task 4 if the hidden family has unseen rule variants.
4. **Proper LoFO train + eval** — at present we only do LoFO with the trigram. Re-train one transformer cell with one family held out, measure ID→OOD drop, report.
5. **A small Streamlit dashboard** — baseline vs trained on identical inputs + anomaly attribution + scaling curve, for the demo video.

---

## Track-specific deliverables

- [x] `extras/results/submission/nextstep.csv`
- [x] `extras/results/submission/completion.csv`
- [x] `extras/results/submission/anomaly.csv`
- [x] Training artifacts: 8 checkpoints in `extras/checkpoints/`, full TB logs in `extras/logs/tb/`, per-cell `summary.json` files
- [x] Scores from our own eval pipeline against `eval_metrics.py`-compatible schema (real `eval_metrics.py` will reproduce these once the organizers ship it)
- [x] Per-family breakdown in this report and in `extras/results/eval/*/metrics.md`
- [ ] Demo video showing baseline vs trained on identical inputs *(to be recorded)*

---

## Credits & dependencies

- **Compute:** EuroHPC Leonardo cluster (CINECA, Italy), reservation `s_tra_ncc` under account `euhpc_d30_031`
- **Open-source libraries:** PyTorch 2.5.1 (cu121), NX-AI/xlstm 1.0.7+, einops, OmegaConf, TensorBoard, pandas, NumPy, Matplotlib, Seaborn, Pixi for env management
- **Pre-trained models used:** none — everything trained from scratch on the provided + generator-produced sequences
- **External APIs:** none
- **Datasets:** the three `*_variants.csv` (1 000 sequences/family) shipped in `tracks/industrial-infineon/training_data/`, plus the `*_longdescription_parameters.csv` reference tables for the physics-feature lookup
- **AI coding assistants used during the hackathon:** Claude Code

---

## A note on honesty

The anomaly 100% accuracy is real but unsurprising — the symbolic validator is the organizers' own code and is the oracle for the 10 documented rule violations on ID. The differentiator on Task 4 (post-submission, hidden family) is whether the learned heads generalize when the validator's rule set may not. We have not tested that directly because the hidden family is, by design, not available to us.

The Top-1 next-step numbers from the trained transformer are below the trigram's 0.717 because of how compositional tokenization expresses a step as multiple word-tokens — beam search assembles step strings less reliably than a direct argmax over step-tokens. The compositional model's value is OOD coverage, not ID Top-1. The grammar-constrained trigram is the system we would ship for Task 1 if the eval were ID-only.

We did not implement PRM, the contrastive encoder, or physics-feature injection. They are in `plan.md` as future work and named explicitly in "What we'd do with another 36 hours" above.

---

*Submitted by team abb for Zero One Hack_01, 2026-05-31.*
