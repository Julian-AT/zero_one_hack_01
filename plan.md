# Plan — Zero One Hack_01 / Industrial AI (Infineon) track

> Working document on branch **`abb`**. Other teammates work on their own branches; we merge to `main` only when we agree on a submission cut.
> Status: **planning** — no code merged yet, only EDA artifacts under `extras/eda/`.

---

## TL;DR

We are building a **hybrid stack**, not a single big model, because the EDA shows the task is highly structured and a trigram baseline already scores Top-5 = 99.3% on next-step prediction. The real competitions are Task 3 anomaly attribution and the post-submission Task 4 OOD generalization on a hidden 4th product family. Our differentiators: **compositional tokenization** (word-level inside step strings → real OOD lever), a **contrastive sequence encoder** for OOD anomaly, **grammar-constrained decoding** for Task 2 exact match, and a complete **scaling story** (2 architectures × 3 sizes × 3 data volumes) for the report.

---

## 1. EDA findings (full data + plots in `extras/eda/`)

Computed on the 3 provided variants CSVs (3 × 1000 sequences, ~388k total step rows).

### Headline numbers

| Metric | Value | Why it matters |
|---|---|---|
| Total unique step strings | **198** (94 shared across all 3 families) | Tiny vocabulary; tiny models suffice |
| Per-family unique steps | MOSFET 137 / IGBT 147 / IC 130 | Family-exclusive: 20 / 27 / 29 → family 4 will likely add another ~25 new steps |
| Mean sequence length | MOSFET 125 / IGBT 148 / IC 115 (CV ≈ 2%) | Length itself fingerprints the family |
| **Trigram-backoff baseline (no training)** | **Top-1 = 0.722, Top-3 = 0.968, Top-5 = 0.993** | **Tasks 1/2 ID is almost saturated by an n-gram** |
| Mean position-conditional entropy | 2.8–3.0 bits | ~7–8 plausible next steps at any position |
| LoFO bigram coverage | 0.68–0.79 | **21–32% of held-out family's bigrams are unseen** — this is the OOD problem |
| Exact duplicate sequences | 0 across all 3000 | No leakage in provided data |

### Plots produced

- `extras/eda/01_length_distribution.png` — tight per-family histograms
- `extras/eda/02_vocab_overlap.png` — 94 shared / 76 family-specific step strings
- `extras/eda/03_top30_step_frequency.png`
- `extras/eda/04_category_over_position.png` — clean block structure visible (logistics → clean → prep → litho cycles → ILD → via → metal → passivation → backside → test → ship)
- `extras/eda/05_position_entropy.png` — entropy curves per family
- `extras/eda/06_bigram_coverage.png` — A→B bigram transfer heatmap
- `extras/eda/07_transition_heatmap.png` — top-30 step transition matrix

### Strategic conclusions from EDA

1. **Tasks 1 & 2 on ID will saturate fast.** Beating 99.3% Top-5 is marginal; the model decisions should be optimized for **Top-1** and **block accuracy**, not Top-5 / MRR.
2. **The OOD problem (Task 4) is the actual competition.** 21–32% of held-out-family bigrams are unseen in training. Models that memorize step-string co-occurrences will fail there. We need representations that generalize across step strings.
3. **Task 3 ID is free** because the organizers' `validate_sequence()` is an oracle for the 10 known rules.
4. **Synonyms are the Task 2 exact-match killer**, not modeling quality. Pairs like `STRIP PHOTORESIST` / `STRIP RESIST`, `RCA CLEAN 1` / `WET CLEAN RCA1`, `DEPOSIT INTERLAYER DIELECTRIC` / `DEPOSIT INTERLEVEL DIELECTRIC` are interchangeable per the grammar. We need a canonicalization pass.
5. **A 25M-param model is likely the sweet spot.** 100M is in the plan only for the scaling-curve story in the report.

---

## 2. Modeling approach — hybrid stack

We do not bet on one giant model. We assemble a stack where most points come from cheap, interpretable components, and the neural models are surgical.

### Tier 1 — Free wins (no GPU, ~4–6 h)

| Component | Used for | Notes |
|---|---|---|
| **Trigram-with-backoff** | Tasks 1 & 2 baseline | Already 99.3% Top-5 ID. Becomes our reported floor. |
| **Symbolic validator** (organizers' `validate_sequence`) | Task 3 binary + rule attribution on ID | ~100% on the 10 known rules. Free oracle. |
| **Grammar-constrained decoder** | Task 2 completion | Limits beam to grammar-valid continuations → big edit-distance/block-acc gains. |
| **k-NN retrieval over prefixes** | Tasks 1 & 2 explainability + fallback | Last-k-step query → nearest training prefix → vote on next step. Strong, interpretable. |
| **Synonym canonicalizer** | Task 2 exact match | Preprocess + postprocess all sequences to canonical synonym form. |

### Tier 2 — Surgical neural models (1 A100, ~12 h)

| Component | Used for | Why this design |
|---|---|---|
| **Contrastive sequence encoder** (~5M params, small Transformer or BiLSTM) | Task 3 OOD anomaly + Task 4 generalization | Train on valid-vs-corrupted across all 3 families. Encoder learns "what valid looks like" structurally → transfers to family 4 even if rule set changes. |
| **Compositional autoregressive Transformer** (~25M, word-token vocab) | Tasks 1 & 2 + Task 4 OOD | Tokenize each step into words (`DEPOSIT POLYSILICON` → `[DEPOSIT, POLYSILICON]`). Vocab shrinks ~70 word-tokens. **This is the central OOD lever — almost no other team will think of it.** |

### Tier 3 — Scaling story for the report (overnight, multi-GPU)

The grid is **2 architectures × 3 model sizes × 3 data volumes = 18 cells**, plus a small tokenization-ablation row (step-as-token vs compositional at the medium size) → **20 cells total**.

| Axis | Levels |
|---|---|
| Architecture | (a) Transformer decoder, (b) xLSTM mixed (alternating sLSTM + mLSTM) |
| Model size | small ~5M / medium ~25M / large ~100M |
| Data volume | 1k / 5k / 10k sequences per family |
| Tokenization (ablation, medium only) | step-as-token vs compositional word-token |

Logging: TensorBoard always-on; Weights & Biases opportunistic (enabled if `WANDB_API_KEY` is set and import succeeds — Leonardo outbound HTTPS may be blocked, hence the fallback).

### Tier 4 — Dashboard + demo (parallel, ~6–8 h)

Streamlit app showing:
- Baseline (trigram, retrieval) vs trained model side-by-side on identical inputs
- Anomaly attribution: highlight the offending step and the rule it violates
- Scaling curves (model size × data volume → metric)
- Per-family + ID-vs-LoFO comparison plots

---

## 3. Architecture details

### Transformer (decoder-only)

Decoder-only GPT-style. RoPE positional, RMSNorm, no bias. Max sequence length 256 (covers IGBT's 155 with headroom). Sizes:

| Size | d_model | n_layers | n_heads | d_ff | params |
|---|--:|--:|--:|--:|--:|
| small  | 256 | 4  | 4  | 1024 | ~5M  |
| medium | 512 | 8  | 8  | 2048 | ~25M |
| large  | 768 | 12 | 12 | 3072 | ~100M |

### xLSTM (mixed sLSTM + mLSTM)

Using NX-AI's `xlstm` PyPI package. Alternating block pattern `[mLSTM, sLSTM, mLSTM, sLSTM, ...]`. **Why mixed:** mLSTM provides parallel-training throughput; sLSTM provides scalar-state tracking (which we expect to help for process-logic state like "which mask level are we on" or "is the surface clean"). Pure sLSTM would be too slow to train at large sizes within the overnight budget.

| Size | d_model | num_blocks | params |
|---|--:|--:|--:|
| small  | 256 | 4  | ~5M  |
| medium | 512 | 8  | ~25M |
| large  | 768 | 12 | ~100M |

Block configs and head counts mirror the Transformer sizes for fair comparison.

### Contrastive sequence encoder

Small Transformer encoder (4 layers, d=256) → mean-pool → 128-d projection. Trained with **triplet loss** + **InfoNCE**:
- Positive pair: two valid sequences from the same family (or same valid sequence with synonyms swapped)
- Hard negative: a corrupted version of the anchor (one of the 10 rule violations injected)
- Easy negative: a random other valid sequence

Inference for Task 3: encode the test sequence, compute cosine sim to nearest valid cluster centroid (per family). Below threshold → flagged as invalid. Combined with the symbolic validator: validator-flagged → invalid + use the rule it caught; only-encoder-flagged → invalid + predict no specific rule (this is where we catch family-4 violations the validator wouldn't see).

### Tokenization (the OOD lever)

We ship **both** for ablation:

1. **Step-as-token** (baseline tokenization): each unique step string → one token. ~198 vocab + specials.
2. **Compositional word-tokenization**: each step split on whitespace into word tokens. `DEPOSIT POLYSILICON` → `[DEPOSIT, POLYSILICON]`. ~70 word-tokens + specials. Step boundaries marked with a `<STEP>` delimiter token so the model can still emit "one step at a time" at inference.

We expect the compositional model to lose marginally on ID Top-1 (sequence is longer and there's more uncertainty), but win on Task 4 OOD because it can generalize to unseen step strings that share words with seen ones.

### Multi-task heads (day 2, after baseline LM proves out)

For autoregressive models we add:
- **Validity head** on the `<EOS>` representation — binary BCE (valid vs corrupted)
- **Rule-ID head** on the `<EOS>` representation — 11-way (10 rules + "valid")

Loss: `L = L_LM + λ_v · L_validity + λ_r · L_rule` with `λ_v=0.5, λ_r=0.3` to start. These heads improve Task 3 directly and act as a regularizer that pushes the LM to encode structural validity (helpful for OOD).

### OOD discipline

- **LoFO cross-val for config selection only:** train on 2 families, eval on the 3rd. Pick architecture/hparams that minimize ID→OOD drop.
- **Family-token dropout** `p=0.2` during training: randomly replace the family token with `<FAMILY_UNK>` so the model can't lean on the family ID for generation.
- **Final model trains on all 3 families** with compositional tokens + multi-task heads + family dropout.

---

## 4. Submission components

What we generate for the Tally submission, mapped to repo locations:

| Deliverable | Source |
|---|---|
| `extras/results/nextstep.csv` | Hybrid: grammar-constrained beam from compositional Transformer + retrieval fallback, top-5 ranked |
| `extras/results/completion.csv` | Same as Tasks 1, but full greedy decode to `<EOS>`, canonicalized synonyms |
| `extras/results/anomaly.csv` | Symbolic validator ∪ contrastive-encoder scorer ensemble |
| Training artifacts | `extras/checkpoints/*.pt` + `extras/logs/tb/` (TB events) + scaling grid CSV |
| Scores | `extras/results/scoreboard.md` from organizers' `eval_metrics.py` |
| Demo video (≤2 min) | Show: baseline-vs-trained side-by-side, one corrupted-sequence attribution, scaling plot |
| `REPORT.md` | Filled from `submission/REPORT_TEMPLATE.md`; lead with the trigram-99.3% finding and OOD strategy |
| `README.md` (rewrite) | Setup + run instructions for the jury |
| `LICENSE` (exists) | MIT — already present |
| `requirements.txt` | torch, xlstm, numpy, pandas, pyyaml, omegaconf, tensorboard, wandb (opt), streamlit, einops |

---

## 5. Repo layout (will be built post-plan-lock)

```
zero_one_hack_01/
├── README.md, REPORT.md, LICENSE, requirements.txt, pyproject.toml
├── plan.md                           ← this file
├── tracks/industrial-infineon/...    ← provided, untouched
├── submission/                       ← provided, untouched
├── configs/
│   ├── arch/{transformer,xlstm}_{small,medium,large}.yaml
│   ├── data/data_{1k,5k,10k}.yaml
│   ├── train/default.yaml
│   ├── token/{step,compositional}.yaml
│   └── grids/scaling_overnight.yaml  ← drives the 20-cell run
├── src/
│   ├── data/   tokenizer.py · load.py · generate.py · corrupt.py · canonicalize.py · validator.py
│   ├── model/  transformer.py · xlstm_model.py · contrastive.py · heads.py · registry.py
│   ├── train/  trainer.py · losses.py · tracking.py · launch.py
│   ├── eval/   predict.py · grammar_decoder.py · ensemble.py · score.py
│   ├── baselines/ ngram.py · retrieval.py · template.py
│   ├── experiments/ run.py (grid → SLURM array)
│   └── utils/  paths.py · seed.py · logging.py
├── scripts/slurm/train_array.sbatch · scripts/{generate_all_data,run_baselines,make_submission}.sh
├── app/dashboard.py                  ← Streamlit demo
├── notebooks/                        ← optional, for exploratory work
└── extras/
    ├── eda/                          ← done (this PR)
    ├── checkpoints/                  ← from training
    ├── logs/tb/                      ← TensorBoard events
    └── results/                      ← submission CSVs + scoreboard.md
```

---

## 6. 36-hour schedule (relative to plan-lock)

The hackathon ends Sunday 10:00 (Tally cutoff). Plan-lock target: **00:30 Saturday**. Net working time after that: ~33 h before submission, of which ~12 h should be sleep across two nights.

| Window | Phase | Goals |
|---|---|---|
| **00:30 → 02:30** (2h) | Scaffold + smoke | requirements.txt, src/ layout, data loader, both tokenizers, transformer model, training loop. Smoke: transformer-small trains 100 steps on 1k MOSFET. |
| **02:30 → 03:00** (0.5h) | Launch overnight | Push to Leonardo, smoke one cell on cluster, sbatch the 20-cell array job. |
| **03:00 → 09:00** (6h) | **Sleep** | Cluster runs the scaling grid. |
| **09:00 → 13:00** (4h) | Tier 1 free wins | n-gram baseline, validator wiring, grammar-constrained decoder, k-NN retrieval, synonym canonicalizer. All scored against `eval_input_valid.csv` + `eval_input_anomaly.csv` once organizers ship them. |
| **13:00 → 17:00** (4h) | Tier 2 contrastive | Build corrupted training set (10 rule violations injected, labeled). Train contrastive encoder. Wire ensemble for anomaly. |
| **17:00 → 21:00** (4h) | Analyze grid + multi-task | Inspect overnight results, identify best (arch, size, data) cell. Add multi-task heads to that config; relaunch with multi-task loss. |
| **21:00 → 00:00** (3h) | Dashboard | Streamlit app: baseline vs trained, anomaly attribution, scaling curves. |
| **00:00 → 06:00** (6h) | **Sleep** | Multi-task fine-tune + final-model run continue overnight. |
| **06:00 → 08:00** (2h) | Submission file gen | Run final model + symbolic validator → produce `nextstep.csv`, `completion.csv`, `anomaly.csv`. Self-score with `eval_metrics.py`. |
| **08:00 → 09:30** (1.5h) | REPORT.md + slides + demo video | Write report from template. Record 2-min demo. 10-slide deck. |
| **09:30 → 10:00** (0.5h) | Submit | Push public repo, fill Tally form. Buffer for re-submit. |

### Decision gates

- **H+5 (after smoke):** If transformer-small doesn't reach trigram-baseline Top-3 within 100 steps on 1k data, something is wrong with the pipeline — debug before launching grid.
- **H+13 (after Tier 1):** If LoFO results from overnight grid show catastrophic OOD drop across all configs, switch to aggressive family-deconfounding (heavier family-dropout, optional family-adversarial loss).
- **H+21 (after Tier 2):** If the contrastive encoder doesn't beat the symbolic validator on LoFO anomaly, fall back to using only the validator + a perplexity-based scorer from the autoregressive model.
- **H+27 (Sat midnight):** If the final multi-task model is unstable, ship the best overnight grid checkpoint as-is.

---

## 7. Risks & mitigations

1. **Symbolic validator turns out to be the eval oracle → Task 3 ID is a wash for everyone.** Mitigation: lead the report with this observation; Task 4 OOD is where we differentiate via the contrastive encoder.
2. **xLSTM CUDA/Triton kernels fail to build on Leonardo.** Mitigation: confirm install in the first hour; if blocked, drop xLSTM rows and substitute a Mamba-SSM row instead.
3. **Leonardo queue spikes overnight.** Mitigation: small models train in <30 min on one A100; develop locally on `transformer_small` and push large/xLSTM runs to cluster. Array jobs reduce queue pressure.
4. **Compositional tokenization actually hurts OOD** (we hope it helps). Mitigation: ablation cell shows both; whichever wins on LoFO is the one we ship.
5. **Exact-match metric collapses due to synonyms.** Mitigation: canonicalize at preprocessing time; also report a "canonical exact match" alongside raw.
6. **Eval inputs (`eval_input_valid.csv`, `eval_input_anomaly.csv`) arrive late or differ in format.** Mitigation: build prediction pipeline against the documented schema in `generation_rules.md` §5; quick adapter once real files land.
7. **Time pressure causes Tier-1 work to be skipped.** Mitigation: Tier 1 is the highest ROI per hour — protect that window.

---

## 8. Team / branch workflow

- Each teammate works on a branch named after them (this branch: **`abb`**).
- All branches are forked from `main`. No direct commits to `main` mid-hackathon.
- We agree late Saturday on which branches' contributions get merged to `main` for the actual submission.
- Submission must be public + MIT-licensed (already MIT, repo will be public at submission time).
- No secrets in commits. `extras/checkpoints/` is gitignored; `.env` is gitignored.

---

## 9. Open decisions deferred to teammates

These are decisions I think should involve the rest of the team rather than be locked unilaterally:

1. **Who owns which tier?** Suggest: one person on Tier 1 free wins, one on Tier 2 contrastive, one on Tier 3 cluster runs/automation, one on Tier 4 dashboard + report.
2. **Whether to add a Mamba/SSM row** to the scaling grid (currently a stretch).
3. **Whether to scrape extra structure from the `*_Longdescr.csv` and `*_longdescription_parameters.csv` files** — they contain step descriptions and realistic fab parameters that we are *not* currently using. Could feed a text encoder for richer step embeddings; uncertain ROI given time budget.
4. **Whether Sunday morning includes a "submit at H-1 hour buffer" or not.** I have it at H-0.5h; could push to H-1h for safety.

---

## 10. What this plan does *not* commit to

- Specific learning rates, batch sizes, dropout values beyond the defaults in `configs/train/default.yaml` — these will be tuned during the LoFO sweep, not pre-decided.
- A specific contrastive loss formulation (InfoNCE vs triplet vs supervised contrastive) — will be chosen based on a 30-min ablation early Saturday afternoon.
- The exact grammar-constrained decoder implementation (mask-then-renormalize vs trie-based) — we'll prototype both quickly.
- The final dashboard layout — depends on which results are most striking after Tier 3 completes.

---

*Plan author: branch `abb`. Reviewable before any source code is written.*
