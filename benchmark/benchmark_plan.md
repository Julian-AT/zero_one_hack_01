# Benchmark Plan — Industrial Infineon Process-Logic Models (MVP)

**Scope: MVP, already-trained models only — no new training (no LSTM/xLSTM rows).**
Goal: put our trained model families on **one common, fair, labeled evaluation** and report
comparable scores via the organizer scorer `participant_files/eval_metrics.py`.

> **Hard caveat:** the official organizer eval inputs are **unlabeled**, so this benchmark runs on
> **internal labeled data only**. It produces internal comparative scores, never official accuracy.

---

## 1. Models under test (already trained)

| Row | Family | Source of the trained model |
|---|---|---|
| Trigram-with-backoff | n-gram baseline (memorization floor) | `extras/baselines/trigram_baseline.py` |
| Retrieval | nearest-neighbor baseline | `extras/baselines/retrieval_baseline.py` |
| **SSL Transformer (hybrid, coverage-guided)** | Transformer arch | checkpoint `tracks/industrial-infineon/runs/ssl_hybrid_new_coverage_guided_v1/checkpoint_best.pt` |
| **Neurosymbolic (PPM + constrained neural ranker)** | Neurosymbolic arch | branch `neurosymbolic-model`, `neurosymbolic-approach/nspe/` + `outputs/` |

Reference floors (trigram/retrieval) stay in the table: any deep model that cannot beat trigram has
not learned process logic.

> **Note on the "3 architectures":** Transformer and Neurosymbolic are fully covered. The third
> family (recurrent LSTM/xLSTM) is **implemented but untrained** (`src/model/xlstm_model.py`,
> `configs/arch/xlstm_*.yaml`) and is **out of MVP scope** because it needs a GPU training run.
> Its rows stay empty/"not trained" in the leaderboard and are listed under Future Work.

---

## 2. Evaluation regimes (and an honest MVP limitation)

- **ID (in-distribution):** score every model on a **common held-out labeled eval set** none of them
  trained on. This is the **primary MVP comparison** and is fully fair.
- **OOD = Leave-One-Family-Out (LoFO):** the track's hidden-4th-family proxy. **True LoFO requires
  per-fold retraining**, which MVP excludes. So:
  - LoFO is reported **only for models that already have per-fold results** — the trigram baseline
    (`extras/results/baselines/trigram_metrics.md`) and the neurosymbolic ranker
    (`neurosymbolic-approach/outputs/exp03_*.json`).
  - The SSL Transformer trained on all 3 families, so its LoFO cell is marked
    **"requires retrain — out of MVP scope"** rather than faked.

This split keeps the headline ID table fully fair while still surfacing the OOD evidence we *do* have.

---

## 3. Common evaluation set (how it is built)

One frozen, labeled eval set, generated with a **benchmark-only seed disjoint from all training
seeds**, so it is genuinely held out for every model:

```bash
# 1. fresh valid sequences (per family) with a benchmark-only seed
python tracks/industrial-infineon/training_data/generate_sequences.py --family mosfet --count 200 --seed 90001 --output benchmark/_raw/valid_mosfet.csv
python tracks/industrial-infineon/training_data/generate_sequences.py --family igbt   --count 200 --seed 90002 --output benchmark/_raw/valid_igbt.csv
python tracks/industrial-infineon/training_data/generate_sequences.py --family ic     --count 200 --seed 90003 --output benchmark/_raw/valid_ic.csv

# 2. fresh invalid sequences (easy + hard) for the anomaly task
python tracks/industrial-infineon/data/generate_invalid_sequences.py      --seed 90010 --output-dir benchmark/_raw/easy_invalid
python tracks/industrial-infineon/data/generate_hard_invalid_sequences.py --seed 90011 --output-dir benchmark/_raw/hard_invalid

# 3. convert to labeled task datasets (next-step / completion / anomaly / rule-attribution)
python tracks/industrial-infineon/data/build_task_datasets.py \
  --valid-input benchmark/_raw/valid_*.csv \
  --easy-invalid-input benchmark/_raw/easy_invalid/invalid_sequences.csv \
  --hard-invalid-input benchmark/_raw/hard_invalid/hard_invalid_sequences.csv \
  --output-dir benchmark/eval_set_v1
```

`benchmark/eval_set_v1/` then holds the labeled ground truth used as the common eval set; a thin
adapter maps its columns to the `eval_metrics.py` ground-truth format (next-step needs
`NEXT_STEP`; completion needs `FULL_SEQUENCE`; anomaly needs `IS_VALID`, `VIOLATION_RULE`).

If regeneration is undesirable, the fallback common eval set is the existing held-out test split of
`tracks/industrial-infineon/data/task_datasets_v1/` (held out for the Transformer; unseen by
trigram/retrieval/neurosymbolic) — slightly less clean but requires no generation.

---

## 4. Tasks and the scores that matter (decided)

All scores come from `participant_files/eval_metrics.py`. Metrics are tiered: **Primary** =
leaderboard, **Secondary** = context, **Diagnostic** = debugging only.

### Task 1 — Next-step prediction
- **Primary:** Top-1 accuracy.
- **Secondary:** MRR (failure mode is ranking, not candidate discovery).
- **Diagnostic:** Top-3, Top-5 (already ~saturated; report once to show saturation, then ignore).

### Task 2 — Sequence completion
- **Primary:** Normalized Edit Distance (NED, lower better).
- **Primary (track-specific):** % rule-valid completions (run the validator on each predicted
  suffix) — the real "did it learn process logic" score.
- **Secondary:** Exact Match rate.
- **Diagnostic:** token accuracy, block-level accuracy.

### Task 3 — Anomaly detection + rule attribution
- **Primary:** F1 on the invalid class; Rule-Attribution Accuracy (among correctly-flagged invalids).
- **Secondary:** ROC-AUC; Balanced Accuracy (classes imbalanced).
- **Diagnostic:** precision, recall, confusion matrix, per-rule detection rate.

### Headline cross-cutting metric
**ID→OOD drop** on Top-1 (T1) and F1 (T3): smaller drop = learned logic, not memorization. Reported
only where LoFO exists (trigram, neurosymbolic). This is the single most important number for the
track thesis, because ID is near-saturated.

### Efficiency columns (architecture comparison)
Parameters, train wall-clock/compute (from logs), inference latency per sequence.

### Aggregation
Macro-average across families (equal weight per family). MVP runs single-seed; seed noted per row.
Multi-seed CIs are Future Work.

---

## 5. Leaderboard schema (`benchmark/RESULTS.md`)

One master table:

```
Model | Arch | Params | Seed |
T1 Top-1 (ID) | T1 MRR (ID) | T1 Top-1 (LoFO) | ΔT1 (headline) |
T2 NED (ID) | T2 ExactMatch (ID) | T2 %rule-valid |
T3 F1 (ID) | T3 RuleAttr (ID) | T3 AUC (ID) | T3 F1 (LoFO) | ΔF1 (headline) |
infer-latency
```

Plus plots: (a) ID vs LoFO grouped bars for Top-1 and F1; (b) Top-1 vs prefix-fraction curves;
(c) accuracy-vs-params/latency scatter. Linked from root `REPORT.md`.

---

## 6. Where we get the data and models from

| Asset | Path | Tracked? | How to obtain |
|---|---|---|---|
| Original provided sequences | `tracks/industrial-infineon/training_data/{MOSFET,IGBT,IC}_variants.csv`, `synthetic*.csv` | ✅ committed | in repo |
| Rule generator + validator | `tracks/industrial-infineon/training_data/generate_sequences.py` | ✅ committed | in repo |
| Coverage-guided valid data | `tracks/industrial-infineon/data/coverage_guided_v1/coverage_guided_sequences.csv` | ❌ gitignored | regenerate via `data/generate_coverage_guided.py` (Leonardo) |
| Easy invalid data | `tracks/industrial-infineon/data/easy_invalid_v1/invalid_sequences.csv` | ❌ gitignored | regenerate via `data/generate_invalid_sequences.py` |
| Hard invalid data | `tracks/industrial-infineon/data/hard_invalid_v1/hard_invalid_sequences.csv` | ❌ gitignored | regenerate via `data/generate_hard_invalid_sequences.py` |
| Labeled task datasets | `tracks/industrial-infineon/data/task_datasets_v1/{next_step_prediction,sequence_completion,anomaly_detection,rule_attribution}.csv` | ❌ gitignored | regenerate via `data/build_task_datasets.py` |
| **Common benchmark eval set** | `benchmark/eval_set_v1/` | ❌ generate | §3 commands (benchmark-only seed) |
| SSL Transformer checkpoint | `tracks/industrial-infineon/runs/ssl_hybrid_new_coverage_guided_v1/checkpoint_best.pt` + `vocab.json` | ❌ gitignored | on Leonardo; produced by `scripts/run_train_ssl_hybrid_newdata_normal_gpu.slurm` |
| Neurosymbolic model + outputs | `neurosymbolic-approach/` (branch `neurosymbolic-model`): `nspe/`, `outputs/exp03_*.json`, `outputs/submission_task*.csv` | ✅ on that branch | `git checkout neurosymbolic-model` or `git worktree add` |
| Trigram / retrieval baselines | `extras/baselines/*.py`; results `extras/results/baselines/` | ✅ committed | in repo (trigram LoFO already computed) |
| Official scorer | `participant_files/eval_metrics.py` | ✅ committed | in repo |
| Organizer eval inputs (unlabeled) | `participant_files/eval_input_valid.csv`, `eval_input_anomaly.csv` | ✅ committed | reference only — cannot be scored locally |

**Cross-branch note:** the neurosymbolic model lives on `neurosymbolic-model`. To benchmark it
alongside the `main` models, use a `git worktree` (no merge needed) and import its prediction
outputs or run its `nspe` inference, then score with the same `eval_metrics.py`.

---

## 7. Harness layout (to implement after this plan is approved)

```
benchmark/
  benchmark_plan.md        # this file
  make_eval_set.py         # §3: build common labeled eval set
  adapters/                # one wrapper per model -> organizer-format predictions
  run_benchmark.py         # model × regime × task -> eval_metrics.py -> results.csv
  report.py                # results.csv -> RESULTS.md + plots
  eval_set_v1/             # generated (gitignored)
  RESULTS.md               # the deliverable leaderboard
```

Most logic is glue around `eval_metrics.py`; no bespoke metric code.

---

## 8. Execution order (MVP)

1. Build the common eval set (§3) on Leonardo.
2. Generate organizer-format predictions for each already-trained model on the common eval set
   (ID), via each model's adapter.
3. Score all three tasks with `eval_metrics.py`; collect into `benchmark/results.csv`.
4. Fill LoFO columns from existing trigram + neurosymbolic per-fold outputs; mark Transformer LoFO
   as out-of-scope.
5. Render `benchmark/RESULTS.md` + plots; link from `REPORT.md`.

## 9. Future work (explicitly out of MVP)

- Train and add **LSTM + xLSTM** rows at matched capacity (the literal third architecture).
- True LoFO for the Transformer (per-fold retraining) to complete the ID→OOD headline.
- Multi-seed runs with 95% confidence intervals.
- FLOP-accurate efficiency frontier (xLSTM linear-context advantage).
