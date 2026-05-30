# Findings — running log

> Append-only log of insights as the hackathon progresses. Each finding has a
> timestamp, what it tells us, and what it means for the rest of the work.
> The report draws from this directly.

---

## 2026-05-30 ~00:00 · Trigram-with-backoff baseline (no GPU)

**What:** A three-level n-gram with Katz-style fall-through backoff over the
provided 3000 sequences (1k per family).

| Setup                                                     | Top-1 | Top-3 |     Top-5 |   MRR |
| --------------------------------------------------------- | ----: | ----: | --------: | ----: |
| Train on all 3000, eval on all (memorization upper bound) | 0.722 | 0.968 | **0.993** | 0.844 |
| Honest ID held-out (80/20 per family)                     | 0.717 | 0.968 | **0.993** | 0.842 |
| LoFO MOSFET (train IGBT+IC)                               | 0.502 | 0.679 |     0.728 | 0.598 |
| LoFO IGBT (train MOSFET+IC)                               | 0.481 | 0.660 |     0.707 | 0.577 |
| LoFO IC (train MOSFET+IGBT)                               | 0.432 | 0.624 |     0.644 | 0.528 |

**Why it matters:**

1. **Tasks 1/2 ID is almost saturated by a 50-line baseline.** Top-5 = 0.993 in-distribution means our learned models have <0.7pp of headroom on the headline metric. We optimize for Top-1 and Task-2 block-accuracy instead.
2. **The held-out 80/20 result is identical to the memorization upper bound** (Top-5 = 0.9931 vs 0.9930). The data is so structured that no held-out gap exists for n-grams. Bigger models will not change this on ID.
3. **LoFO drops Top-1 by ~25pp** (0.72 → ~0.48). This is the OOD challenge stated quantitatively — and the n-gram cannot recover the family-exclusive bigrams it never saw. **This is the gap that compositional tokenization + physics features are meant to close on Task 4.**
4. **Completion exact-match is 0** (model drifts after one wrong step), but MOSFET NED at frac=0.8 is **0.126** — meaning 87% of remaining steps still match. Grammar-constrained beam search should lift exact-match into double digits.

Artefacts: `extras/results/baselines/trigram_metrics.{json,md}`, two sample submission CSVs.

---

## 2026-05-30 ~02:15 · Transformer scaling: bigger is not better on ID

**What:** Three Transformer cells of the overnight grid, same data, same training budget (6000 steps, online-generator stream), only model size varies.

| Cell                   | Params | Tokenization  | Final LM loss | Wall time |
| ---------------------- | -----: | ------------- | ------------: | --------: |
| 0 — transformer_small  |   4.2M | compositional |    **0.1062** |      73 s |
| 1 — transformer_medium |  33.6M | compositional |    **0.1061** |     255 s |
| 2 — transformer_large  | 113.4M | compositional |    **0.1062** |     576 s |

**Why it matters:**

1. **All three sizes converge to within 0.0001 LM loss of each other.** This confirms the trigram finding from a completely different angle: the task has so little inherent entropy that capacity beyond ~5M params is wasted on ID.
2. **Wall time scales linearly with params** (1× → 4× → 8× compute for ~28× → 226× params). For ID work the smallest model is strictly preferable on the throughput/quality frontier.
3. **The report's scaling-curve figure now writes itself**: an essentially flat line on ID loss vs params. This is exactly the "honest evaluation — show what worked, show what didn't" the rubric explicitly rewards. The 100M model is not the winner; it is the _demonstration that bigger isn't the answer_.
4. **Implication for compute budget:** Spend remaining Leonardo time on (a) multi-task heads, (b) the contrastive encoder for OOD anomaly, (c) PRM, (d) longer training of one model — NOT on training larger models.

---

## 2026-05-30 ~02:15 · Compositional vs step-as-token (ablation)

**What:** Cell 6 trains transformer_medium with **step-as-token** while cell 1 trains the same arch with **compositional word-tokens**. All else identical.

| Cell                   | Tokens        | Vocab | Final LM loss (per-token CE) |  Wall |
| ---------------------- | ------------- | ----: | ---------------------------: | ----: |
| 1 — transformer_medium | compositional |   162 |                   **0.1061** | 255 s |
| 6 — transformer_medium | step          |   208 |                   **0.3258** | 251 s |

**Why it matters:**

1. **Raw cross-entropies are not directly comparable** — compositional generates ~5 tokens per step whereas step-as-token generates 1. Per-step entropy would need adjusting before head-to-head comparison.
2. **What this confirms is that compositional tokenization trains correctly** on the same budget. The Task 4 (OOD) eval will be the real comparison: we expect compositional to generalize better to family-4 step strings that share words with seen steps. Cell 6 is the control for that comparison.
3. **Wall time is essentially equal** (251 s vs 255 s) — compositional is not free but the slowdown is negligible at this scale.

---

## 2026-05-30 ~02:30 · Leonardo / xLSTM JIT compile gotcha

**What:** xLSTM's sLSTM block JIT-compiles a custom CUDA kernel on first use. Leonardo compute nodes have the CUDA _runtime_ via the driver but **no `nvcc` / headers / toolkit** on `$PATH` by default. First xLSTM array attempt failed with `compilation terminated. ninja: build stopped: subcommand failed.`

**Fix:** add `module load cuda/12.6` to `scripts/slurm/{train,grid}.sbatch`. The available CUDA modules on Leonardo are `cuda/12.2` (default), `12.3`, `12.6`. We picked 12.6 because our PyTorch was conda-forge `pytorch-gpu` 2.5.1 with CUDA-12-anything compatibility.

After the fix, the slstm extension JIT-compiles successfully (~2-5 min one-time cost), is cached at `~/.cache/torch_extensions/py312_cu126/`, and subsequent runs reuse the cached build.

**Why it matters:** anyone reproducing our work needs the `module load` line, or they will hit the same wall. This goes in the README and in the report's "what didn't work / what we'd warn the next team about" section.

---

## 2026-05-30 ~02:45 · Grammar-constrained decoder lifts trigram across the board

**What:** Wrap the trigram-with-backoff with `validate_sequence`-based masking. For every Top-K candidate, run the organizers' validator on `prefix + [candidate]`; reject candidates that introduce a new violation at the candidate's position. Fall back to unfiltered ranking if all candidates are pruned.

**ID Task 1 (held-out 80/20):**

| Metric | raw trigram | grammar-trigram |       delta |
| ------ | ----------: | --------------: | ----------: |
| Top-1  |      0.7173 |          0.7173 |         0.0 |
| Top-3  |      0.9675 |      **0.9805** | **+1.30pp** |
| Top-5  |      0.9931 |      **0.9957** |     +0.26pp |
| MRR    |      0.8416 |      **0.8469** |     +0.53pp |

**Task 2 completion (ID, frac=0.8) — NED _lower is better_:**

| family | raw trigram NED | grammar-trigram NED | raw EM | grammar EM |
| ------ | --------------: | ------------------: | -----: | ---------: |
| MOSFET |          0.9987 |          **0.1260** | 0.0000 | **0.0250** |
| IGBT   |          0.9915 |          **0.2711** | 0.0000 |     0.0000 |
| IC     |          0.9908 |          **0.5011** | 0.0000 |     0.0000 |

Same fix at frac=0.6: NED drops from ~0.97 → 0.51 (IGBT) and ~0.97 → 0.87 (MOSFET/IC).

**Why it matters:**

1. **First non-zero exact match.** Grammar masking gets MOSFET completion to ExactMatch = 2.5% at frac=0.8 — small but no longer floor-zero. With a learned model behind it, double digits look plausible.
2. **NED drop is dramatic.** MOSFET goes from "essentially wrong" (NED 0.999) to "mostly right" (NED 0.126). The trigram by itself drifts; the validator stops the drift from compounding.
3. **All wins are free** — the validator is the organizers' code; using it at inference time costs O(N) per candidate (N = sequence length) and has zero training cost.
4. **The technique transfers** to the trained Transformer / xLSTM at inference time. The Transformer's logits get the same prefix-validator mask before argmax.

Artefact: `extras/results/baselines/grammar_decoder_metrics.json`.

---

## 2026-05-30 ~02:45 · k-NN retrieval is a strong Task 2 baseline

**What:** For each test partial sequence at 60% or 80% truncation, find the most similar training sequence (weighted Jaccard over the full prefix, anchored on the last-step exact match), then output its remaining steps.

**ID completion (held-out 80/20):**

| frac | family | ExactMatch |        NED |
| ---- | ------ | ---------: | ---------: |
| 0.6  | MOSFET |     0.0000 | **0.2300** |
| 0.6  | IGBT   |     0.0000 | **0.2974** |
| 0.6  | IC     |     0.0000 | **0.3231** |
| 0.8  | MOSFET | **0.0150** | **0.1598** |
| 0.8  | IGBT   |     0.0000 | **0.2405** |
| 0.8  | IC     |     0.0000 | **0.3478** |

Compare against the raw trigram which had NED ~0.96–0.99 everywhere.

**LoFO completion (Task 4 OOD proxy):** NED 0.26–0.76 depending on family + cut. Retrieval degrades more than the grammar decoder on OOD because the held-out family's full sequences aren't available to copy from.

**Why it matters:**

1. **Memory beats greedy generation on this task.** Retrieval gets NED 0.16–0.35 ID where the trigram got 0.99. Highly structured, repetitive data → nearest-neighbor copy is genuinely competitive.
2. **Retrieval is interpretable** — for any prediction we can show "we are completing this sequence because we saw this near-identical training sequence". Great for the demo.
3. **Ensemble opportunity.** For Task 2, the final submission can be: try retrieval first (fast, no params); if NED to gold-validator-acceptable region is low, ship it; otherwise fall back to the grammar-constrained Transformer beam.

Artefact: `extras/results/baselines/retrieval_metrics.json`.

---

## 2026-05-30 ~03:00 · xLSTM scaling on Leonardo

**What:** After the `module load gcc/12.2.0 + cuda/12.6` fix, the xLSTM cells trained successfully. Compositional tokens, same training budget as the transformer cells.

| Cell | Arch                      | Params | Final LM loss | Wall time |
| ---- | ------------------------- | -----: | ------------: | --------: |
| 0    | transformer_small         |   4.2M |        0.1062 |      73 s |
| 1    | transformer_medium        |  33.6M |    **0.1061** |     255 s |
| 2    | transformer_large         | 113.4M |        0.1062 |     576 s |
| 3    | xlstm_small (mLSTM+sLSTM) |    ~5M |    **0.1192** |    ~270 s |
| 4    | xlstm_medium              |   ~25M |    **0.1093** |    ~440 s |
| 5    | xlstm_large               |  ~100M |     (running) |         — |

**Why it matters:**

1. **xLSTM converges to a slightly higher loss than the same-size Transformer** (0.119 vs 0.106 at small, 0.109 vs 0.106 at medium). The gap shrinks with scale — xLSTM-medium is essentially competitive.
2. **xLSTM-small reaches 0.119 in ~270 s** vs transformer-small's 73 s — sLSTM's sequential CUDA kernels are slower per step than SDPA. Reasonable but not free.
3. **The mixed mLSTM+sLSTM stack is competitive** and produces a legitimate non-Transformer architecture for the report's "scaling across architectures" section. The briefing explicitly mentions architecture comparison as a stretch goal.
4. **Compile-once cost amortizes** — the first JIT compile of the sLSTM kernel took ~2 minutes; cached at `~/.cache/torch_extensions/py312_cu126/` for subsequent runs.

---

## 2026-05-30 ~03:10 · Physics features lookup is alive

**What:** `src/data/physics.py` parses the three `*_longdescription_parameters.csv` reference files into a `step_string → 10-d feature vector` lookup at `data/processed/physics_features.json`. 136 step strings covered.

**Coverage of each feature across the 136 steps:**

| Feature                               | % present | Range                   |
| ------------------------------------- | --------: | ----------------------- |
| `tool_idx` (categorical)              |      100% | 0..9                    |
| `is_wet` / `is_anneal` / `is_implant` |      100% | 0 / 1                   |
| `log_thickness_nm`                    |     37.5% | -0.7 .. 5.9             |
| `log_time_s`                          |     21.3% | 0.7 .. 3.6              |
| `temp_C`                              |     20.6% | 25 .. 1100              |
| `log_pressure_torr`                   |      8.1% | -2.5 .. 1.6             |
| `energy_keV`                          |      5.9% | 30 .. 150 (implants)    |
| `log_dose_per_cm2`                    |      5.9% | 13.0 .. 15.7 (implants) |

**Examples:**

```
THERMAL OXIDATION   → temp=1000°C, time≈30 min, thick=50nm,  tool=FURNACE
DEPOSIT POLYSILICON → temp=620°C,             thick=200nm,   tool=LPCVD
IMPLANT P BODY      →            energy=80keV, log_dose=13.7, tool=OTHER, implant=1
STRIP PHOTORESIST   → time≈5 min, pressure=0.2 Torr,         tool=OTHER
```

**Why it matters:**

1. **This is the OOD lever for Task 4** (hidden family). For a new step string we have never seen, we can still place it in feature space if the organizers provide parameters: `DEPOSIT GATE OXIDE 2` at `LPCVD; 950 °C; thickness 80 nm` lands near `DEPOSIT GATE OXIDE OR DIELECTRIC` even without lexical match.
2. **Almost no team will exploit these `_parameters` CSVs.** The track briefing mentions them but doesn't push parsing — it'll be a differentiator in the report.
3. **NaN-heavy:** 70%+ of cells are NaN. The model needs to handle this gracefully — we'll use a learnable "missingness" mask alongside the projected features. Implementation defers to Tier-2.5 (post-multi-task training).

Artefact: `data/processed/physics_features.json` (also lists tool taxonomy).

---

_New findings will be appended below as they happen._
