# REPORT — Industrial AI (Infineon): Learning Semiconductor Process Logic

> **Front door for the jury and for colleagues writing the final report/slides.**
> This is the concise spine. The full technical write-up lives in
> [`ssl_results/README.md`](./ssl_results/README.md). Detailed sub-reports are linked inline.

---

## TL;DR

We built an end-to-end **synthetic process-logic data engine + sequence-modeling pipeline** for
semiconductor fabrication routes (MOSFET / IGBT / IC). A hybrid self-supervised Transformer learns
the valid process grammar, and we ship organizer-format predictions for all three eval tasks.
Internal held-out next-step accuracy is **~80% Top-1 / ~99% Top-3 / ~100% Top-5**; the correct next
step is almost always in the model's Top-5, so our final gains come from **reranking**, not bigger models.

The **main contribution is the data-generation, validation, and evaluation workflow** — the model is
one component in a broader process-logic learning system.

> **Caveat (read once):** the organizer eval inputs are **unlabeled**. We can produce
> official-format prediction files and internal held-out metrics, but **only the organizers can
> compute official accuracy** (see [Why official accuracy is unavailable](#why-official-accuracy-is-unavailable-locally)).

---

## Problem

Given synthetic semiconductor fabrication sequences, learn real *process logic* (not surface
memorization) and support four tasks: **next-step prediction**, **sequence completion**,
**anomaly detection**, and **rule attribution**. Sequences are ordered process steps; ~120-token
vocabulary; 10 documented forbidden patterns define validity.

## Approach (what we built)

1. **Data engine** — rule-based valid generation → **coverage-guided** valid generation (targets rare
   steps/transitions/trigrams/blocks/rule-boundaries) → **easy invalid** (obvious violations) →
   **hard invalid** near-misses → **task datasets** (next-step / completion / anomaly / rule-attribution).
2. **Models** — a compact SSL step-token Transformer, ablated across +families and +semantic features,
   culminating in the **hybrid coverage-guided** model used for submission.
3. **Eval pipeline** — official-format prediction generation, then three ranking/decoding strategies on
   top: rule-aware reranking, retrieval augmentation, and a **learned contrastive reranker**.
4. **Diagnostics** — row-count checks and prediction-distribution plots (no labels required).

## Which approach produced which result — and what is final

| # | Approach | Code | Result / report | Status |
|---|---|---|---|---|
| 1 | Original SSL | `tracks/industrial-infineon/scripts/train_ssl_process_transformer.py` | `ssl_results/original_metrics.csv` | baseline |
| 2 | Augmented SSL | same script (more families) | `ssl_results/augmented_metrics.csv` | ablation |
| 3 | Hybrid SSL (semantic features) | `tracks/industrial-infineon/scripts/train_ssl_hybrid_process_transformer.py` | `ssl_results/hybrid_augmented_metrics.csv` | ablation |
| 4 | **Coverage-guided hybrid** | hybrid script + `data/generate_coverage_guided.py` | `ssl_results/hybrid_coverage_guided_metrics.csv` | ✅ **final submission model** (`runs/ssl_hybrid_new_coverage_guided_v1`) |
| 5 | Rule-aware reranking | `participant_files/rerank_nextstep_with_rules.py` | `participant_files/predictions/rerank_nextstep_report.md` | superseded by #7 |
| 6 | Retrieval augmentation | `participant_files/retrieval_augmented_eval.py` | `participant_files/predictions/retrieval_augmented_report.md` | ⚠️ exploratory — **0% prefix match, discarded** |
| 7 | **Learned contrastive reranker** | `participant_files/train_learned_contrastive_reranker.py` | `participant_files/predictions/learned_reranker_report.md` | ✅ **applied to final next-step** |
| 8 | Validator-based anomaly | inside `participant_files/make_eval_predictions.py` | `predictions_anomaly.csv` | ✅ final anomaly output |
| 9 | Easy/hard invalid generation | `data/generate_invalid_sequences.py`, `generate_hard_invalid_sequences.py` | (generated data, gitignored) | ✅ supports anomaly/attribution/reranking |

## Results (internal held-out)

| Model | Test loss | Top-1 | Top-3 | Top-5 | MRR |
|---|---:|---:|---:|---:|---:|
| Original step-token | 0.7631 | 0.8125 | 0.9955 | 0.9999 | 0.9034 |
| Augmented step-token | 0.7606 | 0.8117 | 0.9960 | 0.9999 | 0.9029 |
| Hybrid semantic-feature augmented | 0.7607 | 0.8116 | 0.9960 | 0.9999 | 0.9029 |
| **Hybrid coverage-guided (final base model)** | 0.7829 | 0.8031 | 0.9932 | 0.9997 | 0.8976 |

**Final next-step ranking (learned contrastive reranker, internal):** Top-1 0.7993 → **0.8044**
(+0.0052), MRR 0.8947 → **0.8979** (+0.0033) on the held-out test split — consistent on validation
too, so it was adopted as the active `predictions_nextstep.csv`.

**Final submission files** (formats validated against `participant_files/eval_metrics.py`):

| File | Rows incl. header | Content |
|---|---:|---|
| `participant_files/predictions/predictions_nextstep.csv` | 601 | learned-reranked Top-5 |
| `participant_files/predictions/predictions_completion.csv` | 601 | model greedy completion |
| `participant_files/predictions/predictions_anomaly.csv` | 988 | validator-based validity + rule |

## Why official accuracy is unavailable locally

The eval inputs ship **without labels** (`NEXT_STEP`, `FULL_SEQUENCE`, `IS_VALID`, `VIOLATION_RULE`
are hidden). So locally we have **internal held-out accuracy** + **prediction distribution
diagnostics** only; **official Top-1/F1/AUC are organizer-computed**. We never report a fabricated
official score. Diagnostics: [`participant_files/eval_plots/eval_prediction_report.md`](./participant_files/eval_plots/eval_prediction_report.md).

## What worked / what was exploratory

- **Worked / used:** coverage-guided data engine (#1–4, #9), the hybrid coverage-guided model (#4),
  the learned contrastive reranker (#7, real Top-1/MRR gain), validator-based anomaly (#8).
- **Exploratory / not in deliverable:** retrieval augmentation (#6 — 0% prefix overlap with the eval
  bank, so it changed nothing and was discarded); rule-aware heuristic reranker (#5 — only ≈+0.0004
  Top-1, superseded by the learned reranker).
- **Not fully done (honest):** a true multi-task neural model with anomaly/rule-attribution *heads* —
  invalid data was used for reranking/validation features, not direct neural multi-task training.

## How to run it

Training and prediction require the **Leonardo GPU cluster** and the trained checkpoint at
`tracks/industrial-infineon/runs/ssl_hybrid_new_coverage_guided_v1/checkpoint_best.pt`
(large; gitignored). High level:

```bash
# 1. (Leonardo) generate data
python tracks/industrial-infineon/data/generate_coverage_guided.py
python tracks/industrial-infineon/data/generate_invalid_sequences.py
python tracks/industrial-infineon/data/generate_hard_invalid_sequences.py
python tracks/industrial-infineon/data/build_task_datasets.py

# 2. (Leonardo GPU) train the final hybrid model
sbatch tracks/industrial-infineon/scripts/run_train_ssl_hybrid_newdata_normal_gpu.slurm

# 3. generate official-format predictions (uses the trained checkpoint)
python participant_files/make_eval_predictions.py

# 4. (optional) apply the learned contrastive reranker to next-step
sbatch participant_files/run_learned_contrastive_reranker.slurm

# 5. local-only diagnostics (no GPU, no labels needed)
python participant_files/plot_eval_predictions.py
```

Build the submission archive (PowerShell, Windows):

```powershell
Compress-Archive -Path participant_files\predictions\predictions_nextstep.csv,participant_files\predictions\predictions_completion.csv,participant_files\predictions\predictions_anomaly.csv -DestinationPath submission_predictions.zip -Force
```

## What we'd do with another 36 hours

Multi-task Transformer with next-step + validity + rule-attribution heads; train directly on the
invalid data; pairwise ranking with harder negatives; held-out family/branch OOD evaluation; beam
search + validator pruning for completion; error-driven regeneration around failure modes.

## Credits & dependencies

Python, PyTorch, pandas, matplotlib. Compute: CINECA Leonardo (A100). Synthetic data and scoring
script (`eval_metrics.py`) provided by the organizers. See `requirements.txt` and `pyproject.toml`.

---

### Document map

- Deep technical write-up: [`ssl_results/README.md`](./ssl_results/README.md)
- Track briefing & data: [`tracks/industrial-infineon/README.md`](./tracks/industrial-infineon/README.md), [`tracks/industrial-infineon/training_data/README.md`](./tracks/industrial-infineon/training_data/README.md)
- Final predictions: [`participant_files/predictions/README.md`](./participant_files/predictions/README.md)
- Eval diagnostics: [`participant_files/eval_plots/eval_prediction_report.md`](./participant_files/eval_plots/eval_prediction_report.md)
