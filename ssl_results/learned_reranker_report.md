# Learned Contrastive Reranker Report

This trains a small learned reranker on top of the frozen SSL Transformer candidate generator.

The reranker scores candidate next steps using:
- original model rank,
- valid n-gram continuation evidence,
- invalid mutation-context penalty features,
- process-block/family features,
- validator violation features.

## Validation Split

| Metric | Model only | Learned reranker | Delta |
|---|---:|---:|---:|
| Top-1 Accuracy | 0.7871 | 0.7909 | +0.0038 |
| Top-3 Accuracy | 0.9881 | 0.9891 | +0.0010 |
| Top-5 Accuracy | 0.9989 | 0.9989 | +0.0000 |
| MRR | 0.8875 | 0.8900 | +0.0025 |

Validation Top-1 changed: 187 / 8000 = 2.34%

## Test Split

| Metric | Model only | Learned reranker | Delta |
|---|---:|---:|---:|
| Top-1 Accuracy | 0.7993 | 0.8044 | +0.0052 |
| Top-3 Accuracy | 0.9916 | 0.9931 | +0.0015 |
| Top-5 Accuracy | 0.9994 | 0.9994 | +0.0000 |
| MRR | 0.8947 | 0.8979 | +0.0033 |

Test Top-1 changed: 293 / 12000 = 2.44%

## Official Eval Prediction Activity

| Metric | Value |
|---|---:|
| Official eval examples | 600 |
| Official Top-1 changed vs model-only | 44 |
| Official Top-1 changed % | 7.33% |
| Official any order changed | 471 |
| Official any order changed % | 78.50% |

## Output Files

| File | Path |
|---|---|
| Learned-reranked official next-step predictions | `/leonardo/home/usertrain/a08trc13/zero_one_hack_01/participant_files/predictions/predictions_nextstep_learned_reranked.csv` |
| Learned reranker checkpoint | `/leonardo/home/usertrain/a08trc13/zero_one_hack_01/participant_files/predictions/learned_reranker.pt` |
| Internal example dump | `/leonardo/home/usertrain/a08trc13/zero_one_hack_01/participant_files/predictions/learned_reranker_internal_examples.csv` |

## Decision Rule

Use learned-reranked official predictions only if validation and test Top-1 or MRR improve.
If validation improves but test drops, keep the model-only predictions.
