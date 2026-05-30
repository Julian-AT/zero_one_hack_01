# Plan — Zero One Hack_01 / Industrial AI (Infineon) track

> Working document on branch **`abb`**. Other teammates work on their own branches; we merge to `main` only when we agree on a submission cut.
> Status: **planning + first baseline**. EDA artifacts under `extras/eda/`; trigram baseline lands at `extras/baselines/`.

---

## TL;DR

This track is judged on **five rubric criteria** (depth, infra quality, reproducibility, comparison expressiveness, presentation) — it is *not* a pure leaderboard. Our EDA found a **trigram-with-backoff baseline already scores Top-5 = 99.3% on next-step ID**, so we are not throwing GPU at Tasks 1/2 ID. We win by stacking the right tools per task:

- **Tier 0:** Trigram + grammar-constrained decoder + symbolic validator → ~ceiling on ID for free
- **Tier 1:** Compositional word-tokenization + **physics features** parsed from the `longdescription_parameters` CSVs → the OOD lever
- **Tier 2 (the chain):** Online SFT with generator + multi-task heads → PRM trained on `Violation.step_index` → PRM-guided beam search → RFT
- **Tier 3:** A compact scaling story (2 archs × 3 sizes + 1 ablation, online-generator data) — Transformer + xLSTM-mixed
- **Tier 4:** Streamlit dashboard, contrastive encoder for OOD anomaly, LoFO evaluation in the report, demo video

---

## 1. Rubric calibration — why we optimize for this, not raw scores

`Track_industrial_en.md §9` lists the criteria:

1. Technical depth and **traceable model and data decisions**
2. Quality of **training and benchmark setup on real infrastructure**
3. **Reproducibility** and clarity of evaluation
4. **Expressiveness of the comparison** between baseline, trained model, and scaling variants
5. Quality of **demo, visualization, and result presentation**

Raw Task-1/2/3 scores feed into criterion 4. They are not criterion 1, 2, 3, or 5. Honest "what didn't work" is explicitly part of the cross-track baseline. A trigram-99.3% baseline section is *more valuable in the report than chasing 99.4% with a 100M model*.

---

## 2. EDA findings (artifacts in `extras/eda/`)

Computed on the 3 provided variants CSVs (3 × 1000 sequences, ~388k step rows).

### Headline numbers

| Metric | Value | Implication |
|---|---|---|
| Total unique step strings | **198** (94 shared across all 3 families) | Tiny vocabulary; tiny models suffice |
| Per-family unique steps | MOSFET 137 / IGBT 147 / IC 130 | Family 4 will likely add ~25 new steps |
| Mean sequence length | MOSFET 125 / IGBT 148 / IC 115 (CV ≈ 2%) | Length itself fingerprints the family |
| **Trigram-backoff (no training)** | **Top-1 0.722, Top-3 0.968, Top-5 0.993** | **Tasks 1/2 ID is almost saturated by an n-gram** |
| Mean position entropy | 2.8–3.0 bits | ~7–8 plausible next steps at any position |
| LoFO bigram coverage | 0.68–0.79 | **21–32% of held-out-family bigrams are unseen** — the OOD problem |
| Exact duplicate sequences | 0 across 3000 | No leakage in provided data |

### Plots (already in `extras/eda/`)

`01_length_distribution.png`, `02_vocab_overlap.png`, `03_top30_step_frequency.png`, `04_category_over_position.png` (shows block structure clearly), `05_position_entropy.png`, `06_bigram_coverage.png`, `07_transition_heatmap.png`.

### Strategic implications

1. **Tasks 1 & 2 ID will saturate fast.** Optimize for Top-1 and block accuracy, not Top-5/MRR.
2. **The OOD problem (Task 4) is the actual competition.** Compositional tokens + physics features + LoFO discipline matter more than model size.
3. **Task 3 ID is free** via the organizers' `validate_sequence` oracle.
4. **Synonyms kill Task 2 exact match** — need a canonicalization pass.
5. **A 25M model is the sweet spot.** 100M is in the grid only for the report's scaling-curve narrative.

---

## 3. The hidden lever: physics features from `*_longdescription_parameters.csv`

Each family ships a single reference sequence with **physics-level parameters per step**:

```
"EPITAXIAL DEPOSITION","Grow N− drift epi",
  "RPCVD epi reactor; SiHCl₃ 40 sccm, H₂ 12 slm; 1100 °C; 40 Torr; thickness 60–120 µm"
"IMPLANT P BODY","P‑body implant","Boron; 80 keV; dose 5×10¹³ cm⁻²"
"DEPOSIT POLYSILICON","Poly gate deposition","LPCVD SiH₄ 200 sccm; 620 °C; thickness 400 nm"
```

We parse these into a **~10-dim physics feature vector per step**: temperature (°C), dose (cm⁻²), energy (keV), time (s), pressure (Torr), thickness (nm), dominant material (categorical), dominant tool (categorical: LPCVD/PECVD/RPCVD/ICP/RIE/PVD/CMP/...), is_wet, is_anneal.

**Why this is the differentiator:**
- **OOD lever:** if family 4 has `DEPOSIT GATE OXIDE 2` (unseen step string) but parameters say "LPCVD; 950 °C; thickness 80 nm", we place it near `DEPOSIT GATE OXIDE OR DIELECTRIC` in feature space.
- **Anomaly enrichment:** "DEPOSIT POLYSILICON at 200 °C" is physics-implausible — real value is 620 °C. We get a second anomaly signal alongside the validator.
- **Embedding init:** initialize token embeddings from clustered physics features → better inductive bias.

Concrete usage:
- At input, concatenate `[token_embedding, physics_feature_vector]` per step.
- Build a lookup table `step_string → physics_vector` from the three parameter CSVs.
- Unknown steps (family 4) get a learnable fallback or feature inference from word tokens.

---

## 4. Modeling stack

We do not pick one model — we stack the right tool per task. Each layer adds value and is independently evaluable for the ablation section in the report.

### Tier 0 — Trigram + symbolic baseline (free, no GPU, ~4h)

| Component | Purpose | Notes |
|---|---|---|
| **Trigram-with-backoff** | Task 1 + 2 ID baseline | EDA shows 99.3% Top-5. **Our first deliverable.** |
| **Symbolic validator** | Task 3 ID oracle | ~100% on the 10 known rules. Imports organizers' `validate_sequence`. |
| **Grammar-constrained beam search** | Task 2 ID | Masks logit distribution to grammar-valid steps; on top of trigram → strong Task 2 baseline. |
| **k-NN retrieval over prefixes** | Task 1+2 ID, explainability | Last-k-step prefix → nearest training prefix → vote on next step. Strong + interpretable. |
| **Synonym canonicalizer** | Task 2 exact match | Pre/post-process to canonical form (`STRIP PHOTORESIST`/`STRIP RESIST` → same canonical). |

### Tier 1 — Tokenization & feature layer (~3h)

| Component | Purpose |
|---|---|
| **Step-as-token tokenizer** | Baseline tokenization, ~198 vocab, for ablation |
| **Compositional word tokenizer** | `DEPOSIT POLYSILICON` → `[DEPOSIT, POLYSILICON]`, ~70 word-vocab. **OOD lever.** Step boundaries marked with `<STEP>` delimiter. |
| **Physics feature parser** | Parse `*_longdescription_parameters.csv` → `step_string → 10-d numeric/categorical vector` |
| **Family-token + dropout (`p=0.2`)** | Random `<FAMILY_UNK>` substitution during training → OOD-deconfounding |

### Tier 2 — The chain (Stages 1-3, the technical core, 1-2 A100s)

**Stage 1: Online SFT with generator + multi-task heads (~5h)**
- Each batch, the generator (`generate_sequence`) produces fresh valid sequences → infinite training data, no pre-baked size axis
- A corrupter module injects one of the 10 rule violations into a fraction of sequences → labeled hard negatives via `validate_sequence`
- Multi-task loss: `L = L_LM + 0.5·L_validity + 0.3·L_rule_id`
- Validity head and rule-ID head sit on the `<EOS>` representation

**Stage 2: Process Reward Model (PRM) (~4h)**
- For any prefix, the validator (`validate_sequence`) returns `Violation.step_index` — we know exactly which prefix length first introduced a violation
- Train a small classifier `prefix → P(no violation in first |prefix|+k steps)`
- Architecture: same encoder backbone as Stage 1 + a regression head; train with BCE on validator-derived labels
- This is dense per-step reward, not terminal — the cheapest, highest-quality reward signal anyone in the hackathon could have

**Stage 3: PRM-guided beam search + RFT (~3-4h)**
- *PRM-guided decoding for Task 2:* at each beam-search step, score `LM_logit + α·PRM(prefix+candidate)`. Picks completions that are *both* probable and validator-friendly. Big edit-distance / block-accuracy win expected.
- *RFT (rejection sampling fine-tune):* sample K=8 completions per training prefix, keep only `validate_sequence(generated) == []`, SFT on survivors. Safer than full RL; almost impossible to make worse than SFT starting point.

**Stage 4 (stretch) — Dense-reward GRPO**
- Only if Stages 1-3 deliver a clean checkpoint by Sat 18:00
- Per-step reward from PRM + terminal reward from validator + KL anchor to SFT policy
- Time-boxed to 4h; abandon if not converging

### Tier 3 — Scaling story (compact, overnight, multi-GPU)

Revised from earlier 18 cells → **7 cells** (online-generator subsumes the data-volume axis):

| Cell | Arch | Size | Tokenization |
|---|---|---|---|
| 1 | Transformer | 5M  | compositional |
| 2 | Transformer | 25M | compositional |
| 3 | Transformer | 100M | compositional |
| 4 | xLSTM-mixed | 5M  | compositional |
| 5 | xLSTM-mixed | 25M | compositional |
| 6 | xLSTM-mixed | 100M | compositional |
| 7 | Transformer | 25M | **step-as-token (ablation)** |

All cells train on the online generator, with multi-task heads. Stage 2/3 only run on the winning cell.

### Tier 4 — Anomaly ensemble + dashboard + demo (~10h total)

- **Contrastive sequence encoder** (~5M params, small Transformer) trained on valid-vs-corrupted pairs across all 3 families using triplet/InfoNCE loss → Task 3 OOD scoring
- **Anomaly ensemble** for Task 3 submission: `(symbolic validator) ∪ (contrastive encoder threshold) ∪ (LM perplexity z-score)` — three signals, voted
- **LoFO evaluation harness** — train on 2 families, eval on 3rd; our self-reported Task-4 proxy
- **Streamlit dashboard** — baseline-vs-trained side-by-side, anomaly attribution, scaling curves, LoFO drop chart
- **2-min demo video** — script: show one tricky prefix → trigram suggests X (locally probable but grammar-incoherent) → trained model suggests Y (process-correct) → anomaly attribution on one violating sequence → scaling-curve flash

---

## 5. Submission components

| Deliverable | Source |
|---|---|
| `extras/results/nextstep.csv` | Trained Transformer + PRM-guided beam, top-5 per example. Trigram fallback for ties. |
| `extras/results/completion.csv` | Full PRM-guided beam decode to `<EOS>`, synonym-canonicalized |
| `extras/results/anomaly.csv` | Ensemble (validator ∪ contrastive ∪ perplexity); confidence score from contrastive |
| Training artifacts | `extras/checkpoints/*.pt`, `extras/logs/tb/`, scaling-grid CSV |
| Scores | `extras/results/scoreboard.md` from organizers' `eval_metrics.py` |
| Demo video (≤2 min) | Linked in REPORT.md and submitted via Tally |
| `REPORT.md` | Filled from `submission/REPORT_TEMPLATE.md`; lead with trigram-99.3% reframing + LoFO OOD numbers |
| `README.md` (rewrite) | Setup + 3 commands to reproduce |
| `LICENSE` | MIT (already present) |
| `requirements.txt` | torch, xlstm, numpy, pandas, pyyaml, omegaconf, tensorboard, wandb (opt), streamlit, einops |

---

## 6. Repo layout (built incrementally as we go)

```
zero_one_hack_01/
├── README.md, REPORT.md, LICENSE, requirements.txt, pyproject.toml
├── plan.md                           ← this file
├── tracks/industrial-infineon/...    ← provided, untouched
├── submission/                       ← provided, untouched
├── configs/
│   ├── arch/{transformer,xlstm}_{small,medium,large}.yaml
│   ├── train/default.yaml
│   ├── token/{step,compositional}.yaml
│   └── grids/scaling.yaml            ← 7 cells
├── src/
│   ├── data/   tokenizer.py · load.py · physics.py · corrupt.py · canonicalize.py · validator.py
│   ├── model/  transformer.py · xlstm_model.py · contrastive.py · prm.py · heads.py
│   ├── train/  trainer.py · losses.py · tracking.py · launch.py · online_generator.py
│   ├── eval/   predict.py · grammar_decoder.py · prm_decoder.py · ensemble.py · score.py · lofo.py
│   ├── experiments/ run.py
│   └── utils/  paths.py · seed.py · logging.py
├── scripts/slurm/train_array.sbatch · scripts/{run_baselines,make_submission}.sh
├── app/dashboard.py
└── extras/
    ├── eda/                          ← done
    ├── baselines/                    ← starts here, single-file trigram first
    ├── checkpoints/
    ├── logs/tb/
    └── results/
```

---

## 7. 36-hour schedule (relative to plan-lock, ~00:30 Sat)

| Window | Phase | Goals |
|---|---|---|
| **00:30 → 02:00** (1.5h) | **TRIGRAM BASELINE** (first deliverable) | Self-contained `extras/baselines/trigram_baseline.py`. Builds tri/bi/uni from 80% of provided data, evals on held-out 20% at 60% + 80% truncation. Reports Top-1/3/5/MRR. Also runs LoFO (train on 2 families, eval on 3rd). |
| **02:00 → 03:30** (1.5h) | Repo scaffolding | `src/` layout, requirements.txt, configs/, tokenizers (step + compositional), validator adapter, data loader, smoke. |
| **03:30 → 04:00** (0.5h) | Launch overnight grid | Push to Leonardo, smoke one cell, sbatch the 7-cell array. |
| **04:00 → 10:00** (6h) | **Sleep** | Cluster runs scaling grid. |
| **10:00 → 13:00** (3h) | Physics features + grammar decoder + k-NN retrieval | Parse all 3 `longdescription_parameters` CSVs → step→features lookup. Build grammar-constrained decoder. Build k-NN retrieval baseline. |
| **13:00 → 16:00** (3h) | PRM training | Build per-prefix validator-derived labels. Train PRM on best Stage-1 checkpoint. |
| **16:00 → 19:00** (3h) | PRM-guided decoder + RFT | Wire PRM into beam search. Run RFT loop (sample, filter, fine-tune). Multi-task heads on the winning cell. |
| **19:00 → 22:00** (3h) | Contrastive encoder + anomaly ensemble | Train contrastive on valid+corrupted pairs. Wire 3-signal ensemble. Run LoFO eval. |
| **22:00 → 00:00** (2h) | Dashboard | Streamlit: baseline-vs-trained, anomaly attribution, scaling curves, LoFO drop. |
| **00:00 → 06:00** (6h) | **Sleep** | Optional Stage 4 GRPO if Stages 1-3 are clean. |
| **06:00 → 08:00** (2h) | Submission CSV generation | Run final ensemble on `eval_input_valid.csv` + `eval_input_anomaly.csv` once organizers ship them. Self-score with `eval_metrics.py`. |
| **08:00 → 09:30** (1.5h) | REPORT.md + slides + demo video | Fill template; lead with trigram-99.3% finding + LoFO OOD numbers. Record 2-min demo. 10-slide deck. |
| **09:30 → 10:00** (0.5h) | Submit | Public repo, Tally form. Buffer for re-submit. |

### Decision gates

- **H+2 (after trigram):** If trigram doesn't reproduce our EDA 99.3% on the held-out split → bug in our split logic. Fix before anything else.
- **H+7 (after morning scaffold):** If smoke transformer-small doesn't beat random Top-3 within 100 steps → debug pipeline.
- **H+13 (after PRM):** If PRM accuracy on a held set isn't ≥ 0.95, debug labels; PRM only helps if it's accurate.
- **H+19 (after Tier 4 build):** If LoFO OOD drop is catastrophic across all cells, drop GRPO stretch, focus on dashboard/report polish.
- **H+27 (Sun 03:30):** Hard freeze on training. Move to submission/report.

---

## 8. Risks & mitigations

1. **Symbolic validator turns out to be the eval oracle → Task 3 ID is a wash for everyone.** Mitigation: lead the report with this observation; differentiate on Task 3 OOD via the contrastive encoder.
2. **xLSTM CUDA/Triton kernels fail on Leonardo.** Mitigation: smoke install in first hour; fallback drops xLSTM rows, substitutes Mamba.
3. **Physics-feature parsing brittle** (units, ranges, synonyms in the parameter strings). Mitigation: build a robust parser early; fallback is "physics features off" config.
4. **PRM training requires careful label construction.** Mitigation: spec the label format ("for each prefix, label = 1 if `validate_sequence(prefix) == []` and a valid completion exists, else 0") and validate label distribution before training.
5. **Online generator I/O-bottlenecks training.** Mitigation: pre-generate a 5k pool, refresh asynchronously; trainer pulls from the pool.
6. **Exact-match collapses on synonyms.** Mitigation: canonicalize at pre/post-processing.
7. **Eval inputs from organizers arrive late.** Mitigation: build the inference pipeline against the documented schema in `generation_rules.md §5`. Quick adapter once real files land.

---

## 9. Team / branch workflow

- Branch named after each teammate (this one: **`abb`**).
- Each branch is forked from `main`. No direct commits to `main` mid-hackathon.
- Late Saturday we agree on which branches' contributions get merged for the actual submission.
- Submission must be public + MIT (already MIT). No secrets in commits. `extras/checkpoints/`, `.env` gitignored.

---

## 10. Leonardo cluster workflow (locked from the onboarding deck)

From the `Z10_compressed.pdf` onboarding deck (CINECA Leonardo, Italy — #10 globally per Top 500):

### Access
- 4 login nodes: `login0{1,2,5,7}-ext.leonardo.cineca.it`
- SSH only. 2FA is **disabled** for the hackathon.
- 4 A100s per node. Each team has **1 node reserved** via reservation `s_tra_ncc`. More than 1 node = drop the reservation (queue depends on cluster load).

### Compute discipline
- Compute nodes have **no internet**. Use proxy for low-bandwidth (set `HTTP_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425` in SLURM script). Download all data / model weights / pip wheels from the **login node** beforehand.
- Login node CPU time limit: **10 min**. For longer interactive work: `srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty bash`.
- Filesystems: `$HOME` (50 GB hard limit), `$SCRATCH` (large, deleted after 40 days — use this for checkpoints, generated data), `$PUBLIC` (50 GB, can share between users), `$FAST`/`$WORK` (don't use).

### Package + environment workflow
- **Pixi** is the recommended package manager (faster + reproducible). Commit `pixi.toml` + `pixi.lock` alongside `requirements.txt` for portability.
- Bootstrap on login node:
  ```
  curl -fsSL https://pixi.sh/install.sh | bash
  cd $SCRATCH/zero_one_hack_01
  pixi add python torch numpy pandas pyyaml omegaconf tensorboard tqdm einops
  pixi add --pypi xlstm wandb
  ```
- Optional: Singularity container for reproducibility (Docker → `singularity pull mycontainer.sif docker://...`).

### Standard 1-GPU training job
```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1          # up to 4 per node
#SBATCH --mem=120GB                 # fair share = 120GB × gpus-per-task
#SBATCH --cpus-per-task=8           # fair share = 8 × gpus-per-task
#SBATCH --time=02:00:00             # max 24:00:00
export HTTP_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export HTTPS_PROXY=$HTTP_PROXY
$SCRATCH/pixi/bin/pixi run --manifest-path $SCRATCH/zero_one_hack_01/pixi.toml \
    python -m src.train.launch --config configs/arch/transformer_small.yaml
```

### Compute budget math
- 4 A100s × 24 h × team = **96 A100-hours per team**
- Our 7-cell scaling grid at ~1–3 GPU-hours per cell = ~14 A100-hours
- 4 cells can run in parallel on the 4 A100s → grid finishes in ~4 wall hours
- Leaves **~80 A100-hours** for the chain (PRM + RFT), contrastive encoder, final long run, and stretch GRPO

### Coordination among our 4 teammates
- Single team account on Leonardo → coordinate `$SCRATCH` paths so we don't clobber each other (`$SCRATCH/abb/`, `$SCRATCH/<name>/`).
- One person at a time submits the scaling grid (uses all 4 GPUs in parallel).
- Other teammates work on Tier 1/Tier 4 (no GPU needed) during grid runs.

---

## 11. Open decisions for the team (not locked unilaterally)

1. **Owner split:** roughly — one person on Tier 0 baselines + report scaffolding; one on Tier 2 chain (SFT + PRM + RFT); one on Tier 3 cluster runs + automation; one on Tier 4 contrastive + dashboard + demo.
2. **Whether to add Mamba/SSM** as a third architecture (currently dropped from the grid; could re-add if xLSTM fails).
3. **Whether to use the `Longdescr` step descriptions** as text input to a small text-encoder for richer step embeddings (currently we only use the numeric parameters).
4. **Sunday morning safety buffer:** plan has H-0.5h; could push to H-1h.

---

*Plan author: branch `abb`. Trigram baseline lands next; everything after follows the chain.*
