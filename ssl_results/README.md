# Self-Supervised Process Transformer Results

This folder summarizes the self-supervised pretraining experiments for semiconductor process-sequence modeling. The models were trained to predict the next process step from a valid process prefix. The goal of this stage was to learn reusable process-flow structure before downstream fine-tuning for anomaly detection and rule attribution.

## Compared Runs

We compare three self-supervised models:

| Run | Training data | Representation | Notes |
|---|---|---|---|
| `ssl_original_full` | MOSFET, IGBT, IC | Full process step as token | Baseline SSL model on original families |
| `ssl_augmented_full` | MOSFET, IGBT, IC, DIODE, SCHOTTKY, SIC_MOSFET | Full process step as token | Tests whether auxiliary OOD-style families hurt or help |
| `ssl_hybrid_augmented_full` | MOSFET, IGBT, IC, DIODE, SCHOTTKY, SIC_MOSFET | Full step token + semantic feature embeddings | Adds domain features such as action, role, material, block, side, and lithography level |

The augmented models use synthetic auxiliary families to broaden the training distribution beyond the original MOSFET/IGBT/IC setting.

## Final Test Metrics

| Model | Test loss | Top-1 | Top-3 | Top-5 | MRR |
|---|---:|---:|---:|---:|---:|
| Original step-token | 0.7631 | 0.8125 | 0.9955 | 0.9999 | 0.9034 |
| Augmented step-token | 0.7606 | 0.8117 | 0.9960 | 1.0000 | 0.9029 |
| Hybrid semantic-feature augmented | 0.7607 | 0.8116 | 0.9960 | 1.0000 | 0.9029 |

## Training Curves

### Validation Loss

![Validation loss comparison](ssl_val_loss_comparison.png)

### Validation Top-1 Accuracy

![Validation Top-1 comparison](ssl_val_top1_comparison.png)

### Validation Top-3 Accuracy

![Validation Top-3 comparison](ssl_val_top3_comparison.png)

### Validation Top-5 Accuracy

![Validation Top-5 comparison](ssl_val_top5_comparison.png)

### Validation MRR

![Validation MRR comparison](ssl_val_mrr_comparison.png)

## Bar-Chart Summary

### Test Loss

![Test loss bar chart](test_loss_bar.png)

### Test Top-1 Accuracy

![Test Top-1 bar chart](test_top1_bar.png)

### Test Top-3 Accuracy

![Test Top-3 bar chart](test_top3_bar.png)

### Test Top-5 Accuracy

![Test Top-5 bar chart](test_top5_bar.png)

### Test MRR

![Test MRR bar chart](test_mrr_bar.png)

## Interpretation

All three models reach very strong in-distribution next-step prediction performance. The correct next process step is in the Top-5 predictions essentially all the time, and Top-1 accuracy stabilizes around 81%.

The original and augmented models are almost identical on the current in-distribution test metric. The augmented model has a slightly lower test loss and slightly higher Top-3/Top-5 accuracy, while the original model has a marginally higher Top-1 and MRR. These differences are very small and should not be interpreted as a decisive win for either model.

The main conclusion is that adding auxiliary semiconductor families does **not** degrade next-step prediction performance, even though the training distribution is broader and includes twice as many product families. This is useful because it suggests that broader synthetic pretraining can be used without damaging the core sequence-modeling objective.

The hybrid semantic-feature model also performs almost identically to the augmented step-token model. This means the additional domain features are stable and do not hurt training, but they do not improve ordinary in-distribution next-step prediction. Their potential value should be tested on harder evaluations, especially OOD generalization and anomaly detection.

## Key Takeaways

1. **In-distribution next-step prediction is close to saturated.**  
   Top-5 accuracy is essentially perfect for all models. This suggests that the generated process grammar strongly constrains the next-step space.

2. **Bigger or more structured models should not be judged only on ID metrics.**  
   The current validation/test split is too easy to show whether the models truly learn general process logic.

3. **OOD and anomaly evaluation are now the decisive next step.**  
   The relevant question is whether the original, augmented, and hybrid models behave differently on unseen families or invalid process flows.

## Recommended Next Experiments

The next evaluation stage should compare the checkpoints on harder tasks:

| Evaluation | Purpose |
|---|---|
| Original model on OOD families | Tests zero-shot transfer from MOSFET/IGBT/IC to DIODE/SCHOTTKY/SIC_MOSFET |
| Augmented model on original families only | Checks whether OOD-family augmentation hurts original-family performance |
| Augmented model on OOD families | Checks whether auxiliary-family training improves robustness |
| Hybrid model on OOD families | Tests whether semantic feature embeddings improve process abstraction |
| Likelihood-based anomaly detection | Uses sequence negative log-likelihood to separate valid and invalid process flows |
| Supervised fine-tuning on `sequences.csv` | Adds valid/invalid and rule-attribution heads |

## Files in This Folder

Expected files:

```text
original_metrics.csv
augmented_metrics.csv
hybrid_augmented_metrics.csv
ssl_model_comparison_summary.csv

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