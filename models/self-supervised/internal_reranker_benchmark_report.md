# Internal Reranker Benchmark

Evaluated on internal labeled `test` split from:

`/leonardo/home/usertrain/a08trc13/zero_one_hack_01/competition/track-details/data/task_datasets_v1/next_step_prediction.csv`

## Summary

| Metric | Model only | Reranked | Delta |
|---|---:|---:|---:|
| Top-1 Accuracy | 0.7984 | 0.7987 | +0.0004 |
| Top-3 Accuracy | 0.9912 | 0.9922 | +0.0010 |
| Top-5 Accuracy | 0.9992 | 0.9992 | +0.0000 |
| MRR | 0.8944 | 0.8949 | +0.0005 |

## Reranking Activity

| Metric | Value |
|---|---:|
| Examples | 20000 |
| Top-1 changed | 365 |
| Top-1 changed % | 1.82% |
| Any order changed | 5455 |
| Any order changed % | 27.27% |

## Decision Rule

Use the reranked official prediction file only if internal Top-1 or MRR improves, or if it is neutral but improves process-rule plausibility.

Output examples:

`/leonardo/home/usertrain/a08trc13/zero_one_hack_01/competition/participant-files/predictions/internal_reranker_benchmark_examples.csv`
