# Predictions — which files are final

**Submit only these three** (organizer format, validated against `../eval_metrics.py`):

| File | Rows incl. header | What it is |
|---|---:|---|
| `predictions_nextstep.csv` | 601 | Task 1 — Top-5 next steps. **Active = learned-reranked output** (identical to `predictions_nextstep_learned_reranked.csv`). |
| `predictions_completion.csv` | 601 | Task 2 — model greedy completion. |
| `predictions_anomaly.csv` | 988 | Task 3 — validator-based validity + predicted rule. |

## Backup / audit variants — do NOT submit

These are intermediate stages kept for traceability. They must not be zipped into the submission
unless deliberately promoted to the active file:

```text
predictions_nextstep_model_only.csv        # raw model, no rerank
predictions_nextstep_reranked.csv          # rule-aware heuristic rerank (superseded)
predictions_nextstep_retrieval.csv         # retrieval attempt (0% prefix match → no change)
predictions_nextstep_learned_reranked.csv  # == active predictions_nextstep.csv
predictions_nextstep_before_retrieval.csv
predictions_completion_before_retrieval.csv
predictions_completion_retrieval.csv
```

Reports: `learned_reranker_report.md`, `rerank_nextstep_report.md`,
`retrieval_augmented_report.md`, `internal_reranker_benchmark_report.md`.

## Build the submission zip (PowerShell)

```powershell
Compress-Archive -Path predictions_nextstep.csv,predictions_completion.csv,predictions_anomaly.csv -DestinationPath ..\..\submission_predictions.zip -Force
```

See the root [`REPORT.md`](../../REPORT.md) for the full story.
