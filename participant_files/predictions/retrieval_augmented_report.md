# Retrieval-Augmented Eval Prediction Report

This run uses a large synthetic valid-sequence retrieval bank to improve official eval predictions.

## Data Sources

| Source | Exists |
|---|---:|
| `/leonardo/home/usertrain/a08trc13/zero_one_hack_01/tracks/industrial-infineon/data/coverage_guided_v1/coverage_guided_sequences.csv` | `True` |
| `/leonardo/home/usertrain/a08trc13/zero_one_hack_01/tracks/industrial-infineon/data/retrieval_bank_v1/MOSFET_retrieval.csv` | `True` |
| `/leonardo/home/usertrain/a08trc13/zero_one_hack_01/tracks/industrial-infineon/data/retrieval_bank_v1/IGBT_retrieval.csv` | `True` |
| `/leonardo/home/usertrain/a08trc13/zero_one_hack_01/tracks/industrial-infineon/data/retrieval_bank_v1/IC_retrieval.csv` | `True` |

## Retrieval Coverage

| Metric | Value |
|---|---:|
| Eval valid examples | 600 |
| Valid sequences scanned | 155489 |
| Exact prefix matched examples | 0 |
| Exact prefix match rate | 0.00% |
| Mean exact matches per eval prefix | 0.00 |
| Max exact matches for one prefix | 0 |

## Next-step Changes

| Metric | Value |
|---|---:|
| Top-1 changed vs model-only | 0 |
| Top-1 changed % | 0.00% |
| Any Top-5 order changed | 0 |
| Any Top-5 order changed % | 0.00% |

## Completion Changes

| Metric | Value |
|---|---:|
| Retrieved completion used | 0 |
| Retrieved completion used % | 0.00% |
| Completion changed vs model-only | 0 |
| Completion changed % | 0.00% |
| Validator-valid completed sequences | 600 |
| Validator-invalid completed sequences | 0 |

## Output Files

| File | Path |
|---|---|
| Retrieval next-step predictions | `/leonardo/home/usertrain/a08trc13/zero_one_hack_01/participant_files/predictions/predictions_nextstep_retrieval.csv` |
| Retrieval completion predictions | `/leonardo/home/usertrain/a08trc13/zero_one_hack_01/participant_files/predictions/predictions_completion_retrieval.csv` |

## Decision Rule

Use retrieval-augmented predictions if exact prefix match rate is high enough to trust the retrieval bank.

Recommended:
- If exact prefix match rate is above 25%, use both retrieval next-step and retrieval completion.
- If exact prefix match rate is below 10%, keep model-only predictions.
- If completion validator-invalid count increases substantially, keep model-only completion.
