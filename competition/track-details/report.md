# Self-Supervised Process Transformer: Current Plan

## Overview

We train a self-supervised Transformer on synthetic semiconductor process sequences. Each process step is treated as a discrete token, and each full wafer route is modeled as an ordered token sequence. The model learns valid process-flow structure by predicting the next process step from the previous steps.

The main motivation is to learn reusable **industrial process grammar** across semiconductor-like product families. The pretrained model is intended for:

* next-step prediction,
* sequence completion,
* process-flow representation learning,
* later fine-tuning for anomaly detection and rule attribution.

## Current Data Setup

The generated data is stored in:

```text
data/generated/
```

Important files:

```text
data/generated/valid_long.csv
data/generated/ood_valid_long.csv
data/generated/valid_long_augmented.csv
data/generated/sequences.csv
data/generated/summary.csv
```

### Original Valid Data

```text
data/generated/valid_long.csv
```

Contains valid generated process sequences for the original three families:

```text
MOSFET
IGBT
IC
```

Approximate size:

```text
150,000 valid sequences
50,000 MOSFET
50,000 IGBT
50,000 IC
```

### OOD Family Augmentation

To reduce overfitting to the original MOSFET/IGBT/IC distribution, we generated additional valid synthetic families:

```text
DIODE
SCHOTTKY
SIC_MOSFET
```

These are stored in:

```text
data/generated/ood_valid_long.csv
```

They are generated using a separate OOD-family generator while reusing the official process-rule validator from `training_data/generate_sequences.py`.

The combined SSL pretraining file is:

```text
data/generated/valid_long_augmented.csv
```

This file contains:

```text
MOSFET
IGBT
IC
DIODE
SCHOTTKY
SIC_MOSFET
```

This is now the preferred input file for broader self-supervised pretraining.

### Valid + Invalid Data

```text
data/generated/sequences.csv
```

Contains valid and invalid full sequences for anomaly detection and rule attribution. Invalid examples were created by applying controlled rule-violation corruptions to valid sequences.

Approximate size:

```text
238,000 total examples
150,000 valid
88,000 invalid
```

This file is not used for the current SSL pretraining stage. It will be used later for supervised fine-tuning.

## Model Objective

The current model is trained with a causal next-step prediction objective:

```text
<BOS>, step_1, step_2, ..., step_t  →  step_{t+1}
```

For every position in a valid process sequence, the model predicts the next process step.

Special tokens:

```text
<PAD>   padding token
<BOS>   beginning of sequence
<EOS>   end of sequence
<UNK>   unknown step
<MASK>  masked input step for light denoising
```

## Architecture

The model is a causal Transformer implemented with a Transformer encoder and a causal attention mask.

Architecture:

```text
step embedding
+ family embedding
+ positional embedding
→ Transformer layers with causal mask
→ layer norm
→ tied output projection
→ next-step logits
```

The output projection shares weights with the step embedding matrix.

Default full-scale configuration:

```text
d_model: 256
layers: 6
attention heads: 8
feedforward multiplier: 4
dropout: 0.15
max sequence length: 192
```

Smoke/medium configuration:

```text
d_model: 128
layers: 3
attention heads: 4
parameters: ~646k
```

## Regularization

The dataset is synthetic and grammar-generated, so the model could overfit to frequent transitions or family-specific shortcuts. The training script includes the following regularizers.

### Family Dropout

During training, the family embedding is randomly replaced with `<FAM_UNKNOWN>`.

Purpose:

```text
reduce over-reliance on family labels
encourage shared process-grammar learning
improve robustness to unseen product families
```

Default:

```text
family_dropout = 0.25
```

### Input Step Masking

A small fraction of input steps is replaced with `<MASK>` during training.

Purpose:

```text
increase robustness to partial/noisy process prefixes
encourage contextual understanding
support sequence-completion behavior
```

Default:

```text
input_step_mask_prob = 0.03
```

### Label Smoothing

The loss uses light label smoothing.

Purpose:

```text
avoid overconfident predictions
handle ambiguity where several next steps may be plausible
improve generalization
```

Default:

```text
label_smoothing = 0.05
```

### Class Weighting

The script supports inverse-square-root token reweighting. However, the current stable runs use:

```text
class_weight = none
```

This performed well in the smoke and medium experiments.

## Local Smoke Test

Command:

```bash
python shared/scripts/train_ssl_process_transformer.py \
  --data data/generated/valid_long.csv \
  --out-dir runs/ssl_smoke_v2 \
  --max-sequences 3000 \
  --epochs 3 \
  --batch-size 64 \
  --d-model 128 \
  --layers 3 \
  --heads 4 \
  --class-weight none
```

Result:

```text
Epoch 1: val loss 5.1851, top1 0.0398, top5 0.1231, MRR 0.0948
Epoch 2: val loss 4.9516, top1 0.0566, top5 0.2026, MRR 0.1464
Epoch 3: val loss 4.5316, top1 0.1800, top5 0.4521, MRR 0.3144

Test:    loss 4.5330, top1 0.1791, top5 0.4526, MRR 0.3139
```

This confirmed that the initialization and training loop work correctly.

## Medium Local Run

Command:

```bash
python shared/scripts/train_ssl_process_transformer.py \
  --data data/generated/valid_long.csv \
  --out-dir runs/ssl_medium \
  --max-sequences 10000 \
  --epochs 5 \
  --batch-size 128 \
  --d-model 128 \
  --layers 3 \
  --heads 4 \
  --class-weight none
```

Result:

```text
Epoch 1: val loss 5.0258, top1 0.0560, top5 0.1878, MRR 0.1354
Epoch 2: val loss 4.2819, top1 0.2986, top5 0.6088, MRR 0.4402
Epoch 3: val loss 3.2669, top1 0.5589, top5 0.8906, MRR 0.7024
Epoch 4: val loss 2.2235, top1 0.6672, top5 0.9748, MRR 0.8023
Epoch 5: val loss 1.3813, top1 0.7599, top5 0.9957, MRR 0.8693

Test:    loss 1.3801, top1 0.7591, top5 0.9958, MRR 0.8688
```

This shows that the self-supervised Transformer clearly learns the generated process grammar. After only 10,000 sequences and 5 epochs on CPU, the correct next step is in the Top-5 predictions for almost all positions.

Important limitation: this is still in-distribution evaluation on generated valid sequences. OOD generalization and anomaly detection require additional evaluation.

## Recommended Full-Scale Run

For the supercomputer run, use the augmented valid data:

```text
data/generated/valid_long_augmented.csv
```

Recommended command:

```bash
python shared/scripts/train_ssl_process_transformer.py \
  --data data/generated/valid_long_augmented.csv \
  --out-dir runs/ssl_process_transformer_augmented_full \
  --epochs 30 \
  --batch-size 512 \
  --d-model 256 \
  --layers 6 \
  --heads 8 \
  --lr 3e-4 \
  --family-dropout 0.25 \
  --input-step-mask-prob 0.03 \
  --class-weight none
```

This run trains on both the original three product families and the three synthetic OOD-style auxiliary families.

## Outputs

Each run writes:

```text
checkpoint_best.pt
checkpoint_last.pt
vocab.json
metrics.csv
```

The most important files to preserve are:

```text
runs/.../checkpoint_best.pt
runs/.../vocab.json
runs/.../metrics.csv
```

These are required for later evaluation and plotting.

## Current Metrics Available

The SSL script already logs:

```text
loss
perplexity
Top-1 accuracy
Top-3 accuracy
Top-5 accuracy
MRR
token count
```

These metrics cover next-step prediction.

## Evaluation Still Needed

The following metrics should be added after pretraining by loading the saved checkpoint:

```text
sequence completion exact match
sequence completion edit distance
token-level completion accuracy
anomaly detection via sequence likelihood
ROC-AUC / F1 for valid vs invalid classification
leave-one-family-out evaluation
```

Rule-attribution accuracy will likely require supervised fine-tuning on `data/generated/sequences.csv`.

## Planned Next Stage

After SSL pretraining, we will train or evaluate:

1. **Markov / n-gram baseline**
   Simple next-step baseline using transition statistics.

2. **Self-supervised Transformer**
   Current pretrained model.

3. **Likelihood-based anomaly detector**
   Use sequence negative log-likelihood to distinguish valid vs invalid flows.

4. **Supervised fine-tuned Transformer**
   Add classification heads for:

   * valid vs invalid,
   * violated process rule.

5. **OOD evaluation**
   Compare models on:

   * original families,
   * synthetic OOD families,
   * leave-one-family-out splits.

## Current Recommendation

Use the augmented dataset for the main full-scale SSL run:

```text
data/generated/valid_long_augmented.csv
```

This is preferred over `valid_long.csv` because it should reduce overfitting to the original MOSFET/IGBT/IC distribution and encourage learning more general semiconductor process logic.
