# Plan — Zero One Hack_01 / Industrial AI (Infineon) track

> Working document on branch abb. Other teammates work on their own branches; we merge to main only when we agree on a submission cut.
>
> Status: planning + first baseline. EDA artifacts live under extras/eda/; the trigram baseline lands under extras/baselines/.

---

# TL;DR

This track is judged on five rubric criteria (technical depth, infrastructure quality, reproducibility, comparison expressiveness, presentation) — not purely on leaderboard score.

Our EDA found that a trigram-with-backoff baseline already reaches Top-5 = 99.3% on next-step prediction, meaning Tasks 1 and 2 on in-distribution (ID) data are close to saturation. The actual competition is:

- OOD generalization (hidden family 4)
- Robust anomaly detection
- Clear engineering and evaluation methodology
- Strong report and presentation

Our strategy is therefore:

### Tier 0 — Free wins

- Trigram baseline
- Grammar-constrained decoder
- Symbolic validator
- k-NN retrieval baseline
- Synonym canonicalization

### Tier 1 — OOD features

- Compositional word-tokenization
- Physics features parsed from process descriptions
- Family-token dropout

### Tier 2 — Training chain

- Online SFT
- Multi-task heads
- Process Reward Model (PRM)
- PRM-guided beam search
- Rejection-sampling fine-tuning (RFT)

### Tier 3 — Scaling story

- Transformer vs xLSTM
- 5M / 25M / 100M parameter sizes
- Tokenization ablation

### Tier 4 — Evaluation and presentation

- Contrastive anomaly encoder
- LoFO evaluation
- Streamlit dashboard
- Demo video

---

# 1. Rubric calibration

The competition rubric emphasizes:

1. Technical depth and traceable decisions
2. Infrastructure and training quality
3. Reproducibility
4. Comparison quality
5. Presentation quality

Raw task scores only contribute to one of these categories.

A convincing explanation of why a trigram baseline reaches 99.3% Top-5 is more valuable than spending the entire budget chasing a marginal improvement from 99.3% to 99.4%.

---

# 2. EDA findings

Computed from the three provided process families:

- MOSFET
- IGBT
- IC

(1000 sequences each, ~388k process-step rows total)

## Headline numbers

| Metric | Value | Implication |
|----------|----------|----------|
| Total unique step strings | 198 | Tiny vocabulary |
| Shared across all families | 94 | Strong common grammar |
| Per-family unique steps | MOSFET 137 / IGBT 147 / IC 130 | Hidden family likely introduces new steps |
| Mean sequence length | 115–148 | Length fingerprints family |
| Trigram Top-1 | 72.2% | Strong baseline |
| Trigram Top-3 | 96.8% | Near saturation |
| Trigram Top-5 | 99.3% | Tasks 1/2 ID nearly solved |
| Position entropy | 2.8–3.0 bits | ~7–8 plausible next steps |
| LoFO bigram coverage | 0.68–0.79 | 21–32% unseen transitions OOD |
| Duplicate sequences | 0 | No obvious leakage |

## Key observations

### Tasks 1 & 2 are nearly saturated

The trigram baseline already reaches:

- Top-1 = 0.722
- Top-3 = 0.968
- Top-5 = 0.993

The objective is therefore not to maximize Top-5 but to improve:

- Top-1 accuracy
- Exact completion quality
- OOD robustness

### OOD is the actual challenge

Leave-One-Family-Out (LoFO) analysis shows:

> 21–32% of held-out-family bigrams are unseen during training.

This is the strongest evidence that hidden-family generalization is the key challenge.

### Task 3 ID is almost free

The organizers provide a symbolic validator that already captures the known rule set.

This becomes our anomaly oracle for known-family evaluation.

### Synonyms hurt exact-match metrics

Examples:

- STRIP PHOTORESIST ↔ STRIP RESIST
- RCA CLEAN 1 ↔ WET CLEAN RCA1
- DEPOSIT INTERLAYER DIELECTRIC ↔ DEPOSIT INTERLEVEL DIELECTRIC

We therefore canonicalize step names during preprocessing and postprocessing.

---

# 3. Physics features

Each process family includes a parameterized process description.

Example:

text DEPOSIT POLYSILICON LPCVD SiH4 200 sccm 620 °C thickness 400 nm  IMPLANT P BODY Boron 80 keV dose 5×10^13 cm^-2 

We extract approximately 10 feature dimensions:

- temperature
- pressure
- dose
- implant energy
- thickness
- duration
- dominant material
- dominant tool
- wet-process flag
- anneal flag

## Why this matters

### OOD generalization

Hidden-family steps may be unseen textually but physically similar.

Example:

text DEPOSIT GATE OXIDE 2 

may never appear during training, but its process parameters may place it close to:

text DEPOSIT GATE OXIDE 

in feature space.

### Anomaly detection

Physics can reveal implausible processes even when grammar is valid.

Example:

text DEPOSIT POLYSILICON temperature = 200°C 

is physically suspicious despite being syntactically correct.

---

# 4. Modeling stack

## Tier 0 — Baselines

| Component | Purpose |
|------------|------------|
| Trigram with backoff | Tasks 1 & 2 |
| Symbolic validator | Task 3 |
| Grammar-constrained beam search | Task 2 |
| k-NN retrieval | Explainable fallback |
| Synonym canonicalizer | Exact-match robustness |

---

## Tier 1 — Tokenization and features

### Step-as-token

Each process step becomes a token.

Vocabulary:

- ~198 tokens

Used as baseline.

### Compositional word-tokenization

Example:

text DEPOSIT POLYSILICON 

becomes

text [DEPOSIT, POLYSILICON] 

with explicit <STEP> separators.

Vocabulary:

- ~70 word tokens

### Why this is our OOD lever

A step-level tokenizer can only memorize.

A compositional tokenizer can reuse concepts.

Example:

text DEPOSIT GATE OXIDE 2 

shares words with:

text DEPOSIT GATE OXIDE 

even if the exact step string was never seen.

We expect:

- slightly worse ID performance
- better hidden-family generalization

This is one of the central hypotheses of the project.

### Family-token dropout

During training:

text P(FAMILY → <FAMILY_UNK>) = 0.2 

This prevents the model from overfitting to family identity.

---

## Tier 2 — Online SFT

Training data is generated continuously.

For each batch:

1. Generate valid sequences
2. Corrupt a subset with rule violations
3. Train autoregressively
4. Train anomaly heads simultaneously

Loss:

text L = L_LM + 0.5 L_validity + 0.3 L_rule 

---

## Tier 3 — Process Reward Model (PRM)

The validator exposes:

text Violation.step_index 

which tells us exactly where a sequence first becomes invalid.

This enables dense supervision.

The PRM learns:

text prefix -> probability of remaining valid 

rather than only terminal success/failure.

### Usage

During decoding:

text score = LM_score + alpha * PRM_score 

This encourages process-valid continuations.

---

## Tier 4 — Rejection Fine-Tuning

For each training prefix:

1. Sample K completions
2. Validate all samples
3. Keep valid completions
4. Fine-tune on survivors

This provides a simple and stable reinforcement-learning-style improvement.

---

# 5. Contrastive anomaly encoder

A separate model is trained for OOD anomaly detection.

Architecture:

- 4-layer Transformer encoder
- d_model = 256
- mean pooling
- 128-dimensional projection

Training:

### Positive pairs

- Valid sequences from same family
- Synonym-augmented variants

### Hard negatives

- Corrupted sequences
- Injected rule violations

### Easy negatives

- Random valid sequences

### Loss

Combination of:

- InfoNCE
- Triplet loss

### Inference

Compute:

text cosine(sequence, valid_centroid) 

per family.

Low similarity implies anomaly.

Final anomaly prediction combines:

- symbolic validator
- contrastive encoder
- LM perplexity

into an ensemble.

---

# 6. Scaling study

| Cell | Architecture | Size | Tokenization |
|--------|--------|--------|--------|
| 1 | Transformer | 5M | compositional |
| 2 | Transformer | 25M | compositional |
| 3 | Transformer | 100M | compositional |
| 4 | xLSTM | 5M | compositional |
| 5 | xLSTM | 25M | compositional |
| 6 | xLSTM | 100M | compositional |
| 7 | Transformer | 25M | step-as-token |

Goal:

- Compare architectures
- Compare scaling behavior
- Measure tokenization effect

This supports the report and rubric criteria.

---

# 7. Submission outputs

| Deliverable | Source |
|------------|------------|
| nextstep.csv | Transformer + PRM-guided decoding |
| completion.csv | Full completion decode |
| anomaly.csv | Validator + contrastive + perplexity ensemble |
| checkpoints | extras/checkpoints |
| TensorBoard logs | extras/logs/tb |
| scaling results | extras/results |
| REPORT.md | Final report |
| README.md | Reproduction guide |
| demo video | ≤ 2 minutes |

---

# 8. Repository layout

text zero_one_hack_01/ ├── configs/ ├── src/ │   ├── data/ │   ├── model/ │   ├── train/ │   ├── eval/ │   ├── experiments/ │   └── utils/ ├── scripts/ ├── app/ ├── extras/ │   ├── eda/ │   ├── baselines/ │   ├── checkpoints/ │   ├── logs/ │   └── results/ └── submission/ 

---

# 9. Risks and mitigations

1. Validator already solves Task 3 ID
   - Focus on OOD anomaly detection.

2. xLSTM fails on Leonardo
   - Replace with Mamba.

3. Physics parsing is brittle
   - Fallback configuration without physics features.

4. PRM labels are noisy
   - Validate label quality before training.

5. Synonyms hurt exact-match
   - Canonicalization pipeline.

6. Evaluation files arrive late
   - Build against documented schema.

---

# 10. Team workflow

- One branch per teammate
- No direct commits to main
- Merge only agreed contributions
- Public MIT repository at submission
- No secrets committed

---

# 11. Success criteria

A successful submission is not necessarily the highest leaderboard score.

A successful submission demonstrates:

- Strong baselines
- Thoughtful OOD strategy
- Reproducible infrastructure
- Clear ablations
- Convincing evaluation
- Professional presentation

The central narrative of the report should be:

> A trigram baseline nearly saturates in-distribution prediction. The real challenge is out-of-distribution generalization. We therefore focused on compositional representations, physics-informed features, reward-guided decoding, and rigorous LoFO evaluation.