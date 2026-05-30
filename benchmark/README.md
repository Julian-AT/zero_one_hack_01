# Benchmark Harness — how to run it

This produces the **Stage-2 (common-split) comparison** in `RESULTS.md`. It must run where the
data and checkpoints live (**Leonardo**), because the neural models need `torch` + their
checkpoints. The scoring engine itself (`score.py`, `run_benchmark.py`) is pure-Python and runs
anywhere once prediction CSVs exist.

See `benchmark_plan.md` for scope/metric decisions and `RESULTS.md` for the Stage-1 numbers
already compiled from each model's own run.

## Pipeline

```
prepare_eval_inputs.py   →  common eval set (inputs/ + ground_truth/)
  → each model runs on inputs/  →  benchmark/predictions/<model>/predictions_*.csv
run_benchmark.py         →  results.csv + RESULTS_auto.md
```

## 0. Environment (Leonardo)

```bash
source tracks/industrial-infineon/.venv/bin/activate   # or the repo pixi env
python -c "import torch; print(torch.__version__)"
```

## 1. Build the common eval set

Option A — reuse the existing labeled test split (fastest, no generation):

```bash
python benchmark/prepare_eval_inputs.py \
  --task-dir tracks/industrial-infineon/data/task_datasets_v1 \
  --split test \
  --out-dir benchmark/eval_set_v1
```

Option B — a fresh, leakage-clean held-out set (benchmark-only seed). Generate valid + invalid with
a seed disjoint from training (see `benchmark_plan.md` §3), run `data/build_task_datasets.py` into a
new dir, then point `--task-dir` at it.

This writes:
- `benchmark/eval_set_v1/inputs/{nextstep,completion,anomaly}_input.csv` — feed to every model
- `benchmark/eval_set_v1/ground_truth/{nextstep,completion,anomaly}_gt.csv` — for scoring

## 2. Produce predictions per model (organizer format)

Put each model's outputs in `benchmark/predictions/<model>/` named exactly
`predictions_nextstep.csv`, `predictions_completion.csv`, `predictions_anomaly.csv`.
Expected columns: next-step `EXAMPLE_ID,RANK_1..RANK_5`; completion `EXAMPLE_ID,PREDICTED_SEQUENCE`
(`|`-joined); anomaly `EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE`.

### SSL Transformer (coverage-guided hybrid)
Reuse `participant_files/make_eval_predictions.py`, pointing its input/output constants (lines
~38–44) at the benchmark eval set:
```
EVAL_VALID   = ROOT/"benchmark/eval_set_v1/inputs/nextstep_input.csv"     # (also used for completion)
EVAL_ANOMALY = ROOT/"benchmark/eval_set_v1/inputs/anomaly_input.csv"
OUT_DIR      = ROOT/"benchmark/predictions/ssl_transformer"
```
Then `python participant_files/make_eval_predictions.py`. (For next-step + completion the script
reads the same valid-input file; if you keep them in separate files, run it twice or feed
`completion_input.csv` for the completion pass.)

### Neurosymbolic (PPM + constrained neural ranker)
It lives on another branch — use a worktree so you don't disturb `main`:
```bash
git worktree add ../nspe-bench neurosymbolic-model
# then follow neurosymbolic-approach/RUN_ON_LEONARDO.md to run its submission generator
# (experiments/exp06_make_submission.py / nspe/official.py) on the benchmark inputs,
# producing task1/2/3 CSVs; copy/rename into benchmark/predictions/neurosymbolic/:
#   submission_task1.csv -> predictions_nextstep.csv
#   submission_task2.csv -> predictions_completion.csv
#   submission_task3.csv -> predictions_anomaly.csv
```

### Trigram / retrieval baselines
Run `extras/baselines/trigram_baseline.py` and `retrieval_baseline.py` to emit organizer-format
predictions on `benchmark/eval_set_v1/inputs/`, into `benchmark/predictions/trigram/` and
`.../retrieval/`. (These scripts already compute the baseline numbers in
`extras/results/baselines/`; point them at the benchmark inputs to get common-split predictions.)

## 3. Score everything and build the leaderboard

```bash
python benchmark/run_benchmark.py \
  --pred-root benchmark/predictions \
  --gt-dir   benchmark/eval_set_v1/ground_truth \
  --out      benchmark
```

Outputs `benchmark/results.csv` and `benchmark/RESULTS_auto.md`. Fold the auto tables into the
Stage-2 section of `RESULTS.md`.

Score a single model/task ad hoc:
```bash
python benchmark/score.py --task next-step \
  --predictions benchmark/predictions/ssl_transformer/predictions_nextstep.csv \
  --ground-truth benchmark/eval_set_v1/ground_truth/nextstep_gt.csv
```

## Notes & caveats

- **LoFO / OOD:** the Transformer trained on all 3 families, so a true LoFO row needs a per-fold
  retrain (out of MVP scope). Trigram and neurosymbolic already have LoFO numbers (see `RESULTS.md`).
- **% rule-valid completions:** `score.py` accepts an optional `validator` callable; wire in the
  validator from `tracks/industrial-infineon/training_data/generate_sequences.py` to populate it.
- **Comparability:** only numbers produced through this harness (same `eval_set_v1`) are
  apples-to-apples. The Stage-1 tables in `RESULTS.md` are from different splits — keep them labeled.
- **gitignore:** generated `eval_set_v1/` and `predictions/` are data; add them to `.gitignore` if
  large. The committed deliverables are this harness + `RESULTS.md`.
```
