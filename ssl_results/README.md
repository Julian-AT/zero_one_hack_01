# Self-Supervised Process Transformer — Results and Eval Pipeline

## TL;DR

We trained and compared several next-step process-sequence models for the Industrial Infineon semiconductor process-flow task.

The original three SSL models all reach roughly **81% Top-1** and **~99.6–99.99% Top-3/Top-5** on held-out in-distribution next-step prediction. We then added a new **coverage-guided valid-data run** trained on our improved synthetic data pipeline. This new run reaches:

```text
Test loss: 0.7829
Top-1:     0.8031  ≈ 80.31%
Top-3:     0.9932  ≈ 99.32%
Top-5:     0.9997  ≈ 99.97%
MRR:       0.8976
```

The new coverage-guided run is slightly lower than the previous three on plain in-distribution next-step accuracy, but it was trained on a broader, coverage-guided valid-process corpus. The important result is that the dataset pipeline is now much stronger than the original SSL baseline: it contains valid coverage-guided sequences, easy invalid near-misses, and hard invalid near-misses.

We also generated official-format eval prediction files for:

1. next-step prediction,
2. sequence completion,
3. anomaly detection.

The official eval inputs are unlabeled, so we cannot compute official accuracy locally. The organizers hold the hidden ground truth.

---

## 1. What We Built

The full pipeline now consists of:

```text
rule-based generator
→ coverage tracker
→ coverage-guided valid generation
→ easy invalid generation
→ hard invalid generation
→ task dataset builder
→ SSL next-step model
→ eval prediction generation
→ reranking / retrieval diagnostics
```

The major additions beyond the original SSL experiments are:

* coverage-guided valid data,
* controlled easy-invalid examples,
* hard invalid near-misses,
* task-specific datasets for next-step, completion, anomaly, and rule attribution,
* official-format eval prediction files,
* reranking and retrieval-augmented prediction diagnostics.

---

## 2. Model and Dataset Configurations

| # | Run                                 | Training data                                      | Input representation                     | Purpose                             |
| - | ----------------------------------- | -------------------------------------------------- | ---------------------------------------- | ----------------------------------- |
| 1 | `ssl_original_full`                 | Original 3 families: MOSFET, IGBT, IC              | step-as-token                            | Original baseline                   |
| 2 | `ssl_augmented_full`                | 6 families: original + DIODE, SCHOTTKY, SIC_MOSFET | step-as-token                            | Tests family augmentation           |
| 3 | `ssl_hybrid_augmented_full`         | 6 families                                         | step-token + semantic feature embeddings | Tests hybrid semantic-feature input |
| 4 | `ssl_hybrid_new_coverage_guided_v1` | coverage-guided valid data                         | hybrid semantic-feature Transformer      | Tests new coverage-guided dataset   |

Runs 1–3 are the original SSL comparison. Run 4 is the new model trained after building the coverage-guided augmentation pipeline.

---

## 3. Final Test Metrics

| Model                             | Test loss |  Top-1 |  Top-3 |  Top-5 |    MRR | Interpretation                                                                |
| --------------------------------- | --------: | -----: | -----: | -----: | -----: | ----------------------------------------------------------------------------- |
| Original step-token               |    0.7631 | 0.8125 | 0.9955 | 0.9999 | 0.9034 | Original 3-family baseline                                                    |
| Augmented step-token              |    0.7606 | 0.8117 | 0.9960 | 0.9999 | 0.9029 | Adding families does not materially change in-distribution next-step accuracy |
| Hybrid semantic-feature augmented |    0.7607 | 0.8116 | 0.9960 | 0.9999 | 0.9029 | Semantic features are stable but mostly inert on in-distribution SSL          |
| Hybrid coverage-guided valid data |    0.7829 | 0.8031 | 0.9932 | 0.9997 | 0.8976 | Slightly harder/more diverse valid dataset; still very strong Top-3/Top-5     |

### Reading the table

Top-1 is standard next-step accuracy. The new coverage-guided run gets **80.31% Top-1 accuracy** on its held-out internal test split.

Top-3 and Top-5 are relaxed ranking accuracies. The coverage-guided model gets the correct answer in its Top-3 predictions **99.32%** of the time and in its Top-5 predictions **99.97%** of the time.

This means the model almost always knows the correct neighborhood of the next process step. Most remaining errors are ranking errors inside a small set of plausible candidates.

---

## 4. Training Curves and Summary Charts

### Validation curves

|                                                  |                                                  |
| ------------------------------------------------ | ------------------------------------------------ |
| ![Validation loss](ssl_val_loss_comparison.png)  | ![Validation Top-1](ssl_val_top1_comparison.png) |
| ![Validation Top-3](ssl_val_top3_comparison.png) | ![Validation Top-5](ssl_val_top5_comparison.png) |

![Validation MRR](ssl_val_mrr_comparison.png)

### Final test bar charts

|                                  |                                  |
| -------------------------------- | -------------------------------- |
| ![Test loss](test_loss_bar.png)  | ![Test Top-1](test_top1_bar.png) |
| ![Test Top-3](test_top3_bar.png) | ![Test Top-5](test_top5_bar.png) |

![Test MRR](test_mrr_bar.png)

If coverage-guided comparison plots were generated, see also:

|                                                                       |                                                                       |
| --------------------------------------------------------------------- | --------------------------------------------------------------------- |
| ![Coverage-guided test loss](test_loss_bar_with_coverage_guided.png)  | ![Coverage-guided test Top-1](test_top1_bar_with_coverage_guided.png) |
| ![Coverage-guided test Top-3](test_top3_bar_with_coverage_guided.png) | ![Coverage-guided test Top-5](test_top5_bar_with_coverage_guided.png) |

![Coverage-guided test MRR](test_mrr_bar_with_coverage_guided.png)

---

## 5. Data Augmentation Pipeline

The new dataset pipeline contains three major generated sequence types.

### 5.1 Coverage-Guided Valid Data

Path:

```text
tracks/industrial-infineon/data/coverage_guided_v1/coverage_guided_sequences.csv
```

Purpose:

```text
valid process-flow learning
next-step prediction
sequence completion
```

The coverage-guided generator does not simply sample random process flows. It accepts candidates that improve coverage over:

* rare steps,
* rare transitions,
* rare trigrams,
* block transitions,
* optional branches,
* lithography levels,
* rule-boundary situations.

This produces a broader valid-process dataset than the original generator output.

### 5.2 Easy Invalid Data

Path:

```text
tracks/industrial-infineon/data/easy_invalid_v1/invalid_sequences.csv
```

Purpose:

```text
basic anomaly detection
basic rule attribution
sanity-check invalid process logic
```

It contains controlled invalid sequences, balanced over:

```text
3 product families × 10 process rules × 1000 examples
```

Total:

```text
30,000 easy invalid sequences
```

### 5.3 Hard Invalid Data

Path:

```text
tracks/industrial-infineon/data/hard_invalid_v1/hard_invalid_sequences.csv
```

Purpose:

```text
hard anomaly detection
hard rule attribution
near-miss process validation
```

The hard invalid examples perturb realistic process regions instead of inserting obviously wrong early steps.

Examples:

* make a clean step too stale before deposition,
* move electrical test before passivation cure,
* move pad opening before passivation cure,
* skip/decrease a lithography level,
* move CMP too far away from fill/deposition.

Total:

```text
30,000 hard invalid sequences
```

---

## 6. Task-Specific Datasets

The generated data is converted into task-specific CSVs.

Path:

```text
tracks/industrial-infineon/data/task_datasets_v1/
```

Files:

| File                       | Input            | Target              | Purpose                         |
| -------------------------- | ---------------- | ------------------- | ------------------------------- |
| `next_step_prediction.csv` | process prefix   | next step           | supervised next-step prediction |
| `sequence_completion.csv`  | partial sequence | suffix continuation | sequence completion             |
| `anomaly_detection.csv`    | full sequence    | valid/invalid label | binary process validity         |
| `rule_attribution.csv`     | invalid sequence | violated rule       | explainable anomaly detection   |
| `sequence_summary.csv`     | metadata         | none                | inspection/debugging            |

Approximate size:

| Dataset                      | Approx. examples |
| ---------------------------- | ---------------: |
| Valid full sequences         |            ~5.5k |
| Easy invalid full sequences  |              30k |
| Hard invalid full sequences  |              30k |
| Next-step examples           |            ~695k |
| Sequence-completion examples |           ~16.5k |
| Anomaly examples             |           ~65.5k |
| Rule-attribution examples    |             ~60k |

---

## 7. Official Eval Prediction Files

The organizers provided:

```text
participant_files/eval_input_valid.csv
participant_files/eval_input_anomaly.csv
participant_files/eval_metrics.py
```

The eval input files are unlabeled.

We generated official-format predictions:

```text
participant_files/predictions/predictions_nextstep.csv
participant_files/predictions/predictions_completion.csv
participant_files/predictions/predictions_anomaly.csv
```

Expected row counts:

| File                         | Expected rows incl. header |
| ---------------------------- | -------------------------: |
| `eval_input_valid.csv`       |                        601 |
| `predictions_nextstep.csv`   |                        601 |
| `predictions_completion.csv` |                        601 |
| `eval_input_anomaly.csv`     |                        988 |
| `predictions_anomaly.csv`    |                        988 |

These counts were verified after generation.

### Important caveat

The official eval files do not include hidden labels such as:

* `NEXT_STEP`,
* `FULL_SEQUENCE`,
* `IS_VALID`,
* violated rule labels.

Therefore, we cannot compute official eval accuracy locally. Only the organizers can score the final prediction files.

---

## 8. Eval Prediction Diagnostics

Path:

```text
participant_files/eval_plots/eval_prediction_report.md
```

This report visualizes the generated official-format prediction files.

It does not compute official accuracy. It only checks:

* row-count consistency,
* next-step prediction distribution,
* completion length distribution,
* anomaly valid/invalid prediction distribution,
* anomaly predicted rule distribution.

Generated figures include:

```text
participant_files/eval_plots/nextstep_top1_distribution.svg
participant_files/eval_plots/nextstep_top5_distribution.svg
participant_files/eval_plots/completion_length_distribution.svg
participant_files/eval_plots/completion_first_step_distribution.svg
participant_files/eval_plots/completion_last_step_distribution.svg
participant_files/eval_plots/anomaly_validity_counts.svg
participant_files/eval_plots/anomaly_rule_distribution.svg
participant_files/eval_plots/anomaly_score_distribution.svg
```

---

## 9. Reranking Experiment

Because the coverage-guided model has:

```text
Top-1 ≈ 80.31%
Top-3 ≈ 99.32%
Top-5 ≈ 99.97%
```

the correct next step is almost always already in the candidate set. This motivated a rule-aware reranker.

### Reranking method

The reranker uses:

1. original model rank,
2. valid-sequence n-gram continuation evidence,
3. invalid-mutation context penalties,
4. process-rule validator penalties.

Relevant files:

```text
participant_files/rerank_nextstep_with_rules.py
participant_files/run_rerank_nextstep.slurm
participant_files/predictions/predictions_nextstep_reranked.csv
participant_files/predictions/rerank_nextstep_report.md
```

### Official eval reranking activity

On the 600 official valid eval examples:

```text
Top-1 changed: 10 / 600 = 1.67%
Any Top-5 order changed: 140 / 600 = 23.33%
Validator-penalized candidates: 93
```

This means the reranker is conservative. It does not rewrite the model outputs globally; it only changes a small number of Top-1 decisions and adjusts ordering in about one quarter of examples.

### Internal reranker benchmark

We also evaluated the reranker on an internal labeled test split.

Relevant files:

```text
participant_files/benchmark_reranker_internal.py
participant_files/run_internal_reranker_benchmark.slurm
participant_files/predictions/internal_reranker_benchmark_report.md
ssl_results/internal_reranker_benchmark_report.md
```

The observed improvement was very small, approximately:

```text
Δ Top-1 ≈ +0.0004
```

Interpretation:

* the reranker is safe/conservative,
* it does not substantially improve next-step accuracy,
* the model-only predictions were already very strong,
* the remaining Top-1 errors are not easily fixed by simple local n-gram/rule reranking.

---

## 10. Retrieval-Augmented Experiment

As a final high-upside extension, we implemented retrieval-augmented prediction.

### Idea

Instead of relying only on the Transformer, we generate a large valid process bank and retrieve exact matching prefixes.

Pipeline:

```text
large generated valid sequence bank
→ exact prefix matching against official partial eval examples
→ retrieved next-step frequencies
→ retrieved suffix continuations
```

Relevant files:

```text
participant_files/run_generate_retrieval_bank.slurm
participant_files/retrieval_augmented_eval.py
participant_files/run_retrieval_augmented_eval.slurm
participant_files/predictions/retrieval_augmented_report.md
participant_files/predictions/predictions_nextstep_retrieval.csv
participant_files/predictions/predictions_completion_retrieval.csv
```

This can improve eval predictions if the official eval partial prefixes overlap strongly with the generated retrieval bank.

### How to interpret retrieval results

The key diagnostics are in:

```text
participant_files/predictions/retrieval_augmented_report.md
```

Look at:

```text
Exact prefix match rate
Top-1 changed vs model-only
Retrieved completion used
Completion changed vs model-only
Validator-valid completed sequences
Validator-invalid completed sequences
```

Decision rule:

| Retrieval diagnostic          | Action                                                      |
| ----------------------------- | ----------------------------------------------------------- |
| Exact prefix match rate > 25% | use retrieval next-step + retrieval completion              |
| Exact prefix match rate < 10% | keep model-only predictions                                 |
| 10–25%                        | use only if completion validator quality does not get worse |

This retrieval step is not an official accuracy evaluation because the official labels are hidden. It is a controlled prediction-improvement attempt.

---

## 11. Current Best Submission Files

The active submission files are:

```text
participant_files/predictions/predictions_nextstep.csv
participant_files/predictions/predictions_completion.csv
participant_files/predictions/predictions_anomaly.csv
```

Backups and variants may include:

```text
participant_files/predictions/predictions_nextstep_model_only.csv
participant_files/predictions/predictions_nextstep_reranked.csv
participant_files/predictions/predictions_nextstep_retrieval.csv
participant_files/predictions/predictions_completion_before_retrieval.csv
participant_files/predictions/predictions_completion_retrieval.csv
```

Only the active three prediction files should be submitted unless the submission instructions ask for otherwise.

Create a final zip from repo root:

```bash
zip -j participant_eval_predictions_final.zip \
  participant_files/predictions/predictions_nextstep.csv \
  participant_files/predictions/predictions_completion.csv \
  participant_files/predictions/predictions_anomaly.csv
```

The zip should contain exactly:

```text
predictions_nextstep.csv
predictions_completion.csv
predictions_anomaly.csv
```

---

## 12. What We Can Claim

### Strong claims

* A small self-supervised Transformer learns the synthetic semiconductor process grammar well.
* In-distribution next-step prediction reaches around **80–81% Top-1** and near-saturated **Top-3/Top-5**.
* Coverage-guided data remains highly learnable while broadening process coverage.
* The invalid-data pipeline creates balanced easy and hard rule-violation examples.
* Official-format eval predictions were generated for all three tasks.
* Validator-based anomaly prediction directly uses the explicit process rules.

### Careful claims

* The new coverage-guided dataset is likely harder/more diverse, so slightly lower Top-1 is not necessarily worse.
* Reranking had only a very small measured internal effect.
* Retrieval augmentation is a plausible final extension, but official benefit is unknown without hidden labels.
* Official eval accuracy cannot be computed locally from the provided input-only files.

### Claims we should not make

* We should not claim official eval accuracy before organizer scoring.
* We should not claim true real-world semiconductor process validity; the data is synthetic.
* We should not claim semantic features improve in-distribution SSL, because the metrics show they are mostly inert there.
* We should not claim reranking materially improves accuracy; the measured delta is tiny.

---

## 13. Recommended Final Story

The strongest final story is:

```text
We built a full synthetic process-logic learning pipeline, not just a next-step model.
The pipeline generates broad valid flows, easy invalids, and hard invalid near-misses.
The SSL Transformer learns the valid grammar with ~80% Top-1 and ~99.3% Top-3 accuracy.
The explicit validator and invalid-data generation make the system rule-aware and support anomaly/rule-attribution evaluation.
For the official hidden eval, we generated complete submission-format predictions for next-step, completion, and anomaly tasks.
```

This is a stronger story than simply saying “we trained another Transformer,” because the central contribution is the **data-generation, validation, and benchmark pipeline**.

---

## 14. Files in This Folder

Per-run metrics:

```text
original_metrics.csv
augmented_metrics.csv
hybrid_augmented_metrics.csv
hybrid_coverage_guided_metrics.csv
```

Comparison summaries:

```text
ssl_model_comparison_summary.csv
ssl_comparison_summary.csv
ssl_model_comparison_summary_with_coverage_guided.csv
```

Coverage-guided result addendum:

```text
coverage_guided_addendum.md
```

Internal reranker result:

```text
internal_reranker_benchmark_report.md
```

Figures:

```text
ssl_val_loss_comparison.png
ssl_val_top1_comparison.png
ssl_val_top3_comparison.png
ssl_val_top5_comparison.png
ssl_val_mrr_comparison.png

test_loss_bar.png
test_top1_bar.png
test_top3_bar.png
test_top5_bar.png
test_mrr_bar.png
```

Coverage-guided comparison figures, if generated:

```text
ssl_val_loss_comparison_with_coverage_guided.png
ssl_val_top1_comparison_with_coverage_guided.png
ssl_val_top3_comparison_with_coverage_guided.png
ssl_val_top5_comparison_with_coverage_guided.png
ssl_val_mrr_comparison_with_coverage_guided.png

test_loss_bar_with_coverage_guided.png
test_top1_bar_with_coverage_guided.png
test_top3_bar_with_coverage_guided.png
test_top5_bar_with_coverage_guided.png
test_mrr_bar_with_coverage_guided.png
```

---

## 15. Next Steps After Submission

If more time is available after the hackathon submission, the most useful follow-ups are:

1. train a real multi-task Transformer with shared encoder and heads for:

   * next-step prediction,
   * validity classification,
   * rule attribution;

2. evaluate on held-out product families and held-out process branches;

3. replace greedy sequence completion with beam search plus validator pruning;

4. train a learned reranker on internal labeled data instead of using heuristic reranking;

5. report anomaly/rule-attribution metrics on the generated easy/hard invalid datasets.

The most important missing research result is not another in-distribution Top-1 number. It is **generalization and rule-aware reasoning under invalid or out-of-distribution process flows**.
