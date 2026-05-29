# Self-Supervised Process Transformer

## Overview

This model is a self-supervised Transformer trained on synthetic semiconductor process sequences. The goal is to learn the structure of valid industrial process flows for the three product families **MOSFET**, **IGBT**, and **IC**. Each process step is treated as a discrete token, and each full wafer process route is modeled as an ordered token sequence.

The model is trained only on **valid sequences** and learns to predict the next process step from the previous steps. This makes the pretraining objective directly aligned with the hackathon tasks of **next-step prediction** and **sequence completion**. The learned checkpoint can later be reused for supervised fine-tuning on anomaly detection and rule attribution using the generated valid/invalid dataset.

## Data

The model is trained on:

```text
data/generated/valid_long.csv
```

This file contains valid process sequences in long format:

```text
SEQUENCE_ID,FAMILY,STEP
```

Each `SEQUENCE_ID` corresponds to one complete process flow. The `FAMILY` column identifies the product family, and the `STEP` column contains the process step token.

The current generated dataset contains approximately:

```text
150,000 valid sequences
50,000 MOSFET
50,000 IGBT
50,000 IC
```

The vocabulary in the smoke test contained **203 tokens including special tokens**.

## Objective

The model is trained with a causal next-step prediction objective:

```text
<BOS>, step_1, step_2, ..., step_t  →  step_{t+1}
```

For each position in the sequence, the model predicts the next process step. This is equivalent to language modeling over semiconductor process tokens.

Special tokens:

```text
<PAD>   padding token
<BOS>   beginning of sequence
<EOS>   end of sequence
<UNK>   unknown step
<MASK>  masked input step for light denoising
```

## Architecture

The model is a small causal Transformer implemented using a Transformer encoder with a causal attention mask.

Main components:

```text
step embedding
+ family embedding
+ positional embedding
→ Transformer layers with causal mask
→ layer norm
→ tied output projection
→ next-step logits
```

The default full-scale configuration is:

```text
d_model: 256
layers: 6
attention heads: 8
feedforward multiplier: 4
dropout: 0.15
max sequence length: 192
```

The smoke-test configuration was smaller:

```text
d_model: 128
layers: 3
attention heads: 4
parameters: ~646k
```

The output head shares weights with the step embedding matrix. This weight tying is common in language models and reduces the number of parameters while improving consistency between input and output token representations.

## Regularization and OOD Motivation

The dataset is synthetic and grammar-generated, so the model could overfit to family-specific shortcuts or frequent process-step patterns. To reduce this risk, the training script includes several regularization mechanisms.

### Family Dropout

During training, the product-family label is randomly hidden with probability `family_dropout`.

Purpose:

```text
prevent the model from relying too strongly on MOSFET/IGBT/IC labels
encourage learning shared process grammar
improve robustness to hidden or unseen product families
```

Default:

```text
family_dropout = 0.25
```

### Input Step Masking

A small fraction of input steps is randomly replaced with `<MASK>` during training.

Purpose:

```text
make the model robust to partial or noisy prefixes
encourage contextual process understanding
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
improve generalization
handle ambiguity where multiple next steps may be valid
```

Default:

```text
label_smoothing = 0.05
```

### Optional Class Reweighting

The script supports inverse-square-root class weighting for step tokens. This can help rare process steps, but the current stable smoke test used:

```text
class_weight = none
```

## Training

The model is trained with:

```text
optimizer: AdamW
learning-rate schedule: warmup + cosine decay
gradient clipping: enabled
mixed precision: enabled automatically on CUDA
checkpointing: best and last checkpoint
```

Outputs are written to the run directory:

```text
checkpoint_best.pt
checkpoint_last.pt
vocab.json
metrics.csv
```

## Smoke-Test Result

A local CPU smoke test was run on 3,000 valid sequences for 3 epochs:

```bash
python scripts/train_ssl_process_transformer.py \
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

This confirms that the model is learning meaningful process-sequence structure. The initial loss is close to `log(vocab_size)`, as expected for a randomly initialized model, and decreases substantially within only three epochs.

## Full-Scale Training Command

For the full dataset, the planned training command is:

```bash
python scripts/train_ssl_process_transformer.py \
  --data data/generated/valid_long.csv \
  --out-dir runs/ssl_process_transformer_full \
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

## Intended Use

The pretrained model can be used for:

1. **Next-step prediction**
   Predict the most likely next valid semiconductor process step.

2. **Sequence completion**
   Autoregressively complete a partial process sequence.

3. **Representation learning**
   Use hidden states as learned embeddings of process prefixes or full routes.

4. **Initialization for supervised fine-tuning**
   Fine-tune the pretrained model on valid/invalid examples for anomaly detection and rule attribution.

## Next Steps

The next planned stage is supervised fine-tuning on:

```text
data/generated/sequences.csv
```

This file contains both valid and invalid sequences, including rule labels and corruption metadata. The fine-tuning model should add classification heads for:

```text
valid vs invalid
violated process rule
```

The final benchmark should compare:

```text
Markov / n-gram baseline
symbolic validator baseline
self-supervised Transformer
fine-tuned Transformer
```

Evaluation should include:

```text
Top-1 / Top-3 / Top-5 accuracy
MRR
sequence-completion accuracy
binary anomaly classification
rule-attribution accuracy
leave-one-family-out generalization
```
