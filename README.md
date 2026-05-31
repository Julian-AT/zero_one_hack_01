# Team Attention Seeker: Track 1 (Industry) Submission

We test whether a model can learn the grammar of semiconductor fab routes or only memorize them. It learns to predict the next step, complete partial routes, and flag rule violations from synthetic data. We measure generalization by holding out an entire product family at test time.

We built and compared **three approaches** on a single shared benchmark:

- **Transformer** (`models/transformer_xlstm/`): a decoder transformer (optionally xLSTM) with a
  compositional tokenizer and multitask validity/rule heads. Our most mature model.
- **SSL-Hybrid** (`models/self-supervised/`): a self-supervised transformer with semantic-feature
  and family embeddings, plus retrieval/rerank.
- **Neurosymbolic** (`models/neurosymbolic/`): a symbolic grammar and 10-rule oracle with role
  induction, ranked by a zero-parameter PPM.

The three submission tasks (next-step prediction, sequence completion, anomaly detection) are
scored with `eval_metrics.py`. Results are in
[`submission/UNIFIED_BENCHMARK.md`](submission/UNIFIED_BENCHMARK.md). The technical write-up is in
[`REPORT.md`](REPORT.md).

## Run it on Leonardo (GPU)

Everything below runs from a **login node** and submits to a GPU node. Replace `<your_account>` with your account name.

```bash
# 1. clone into your scratch space
cd "$SCRATCH"
git clone <repo-url> zero_one_hack_01
cd zero_one_hack_01

# 2. install dependencies
bash shared/scripts/leonardo/setup_env.sh

# 3. submit to GPU Node
sbatch --account=<your_account> reproduce.sbatch

# 4. log job information
squeue --me
tail -f reproduce-*.out
```

`reproduce.sbatch` requests one A100 on the `boost_usr_prod` partition, loads `cuda`/`gcc`, and runs
the full benchmark on the GPU (~15 min). It uses the environment from step 2 directly, so the compute
node needs no internet.

Outputs: `shared/benchmark/results_summary.csv` and the report at
[`submission/UNIFIED_BENCHMARK.md`](submission/UNIFIED_BENCHMARK.md). The production-scale training grids (max_len 768, the xLSTM
architecture, multiple sizes) are in `shared/scripts/slurm/`.

## Run it locally (CPU)

```bash
git clone <repo-url> && cd zero_one_hack_01
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./reproduce.sh           # build eval set, train compact models, score, write the report
```

`reproduce.sh` runs the same comparison end to end. It auto-detects CUDA (≈10 min) and otherwise runs on CPU (≈60 to 75 min).

## Repository layout

```
reproduce.sbatch / reproduce.sh   one-command reproduction
models/
  transformer_xlstm/   decoder transformer + xLSTM: tokenizers, training, eval
  self-supervised/     SSL-hybrid transformer + retrieval/reranker, metrics, plots
  neurosymbolic/       grammar, 10-rule oracle, role induction, PPM ranker
competition/
  track-details/       the brief, the rule-based sequence generator, and the validation rules
  participant-files/   the official scorer (eval_metrics.py) and our submission CSVs
shared/
  benchmark/           the unified benchmark: eval-set builder, per-model adapters, scorer, report
  scripts/             SLURM jobs + Leonardo setup
  extras/              checkpoint summaries, training logs, loss curves, baselines, raw results
configs/               OmegaConf YAML for architecture / tokenizer / training
submission/            REPORT material and the cross-model benchmark report + figures
docs/                  documentations of specific project details
tests/                 pytest suite for tokenizers, validator, metrics, I/O
```

## Deliverables

- [x] **[`REPORT.md`](REPORT.md)**: the technical report.
- [x] **[`submission/UNIFIED_BENCHMARK.md`](submission/UNIFIED_BENCHMARK.md)**: all three approaches on
  one eval set with the official metrics, in-distribution and out-of-distribution (held-out family).
- [x] **[`competition/participant-files/predictions/`](competition/participant-files/predictions/)** holds
  the submission CSVs: `predictions_nextstep.csv`, `predictions_completion.csv`, `predictions_anomaly.csv`.
- [x] **Scores from `eval_metrics.py`** on all three tasks, with a per-family breakdown, reported in
  [`submission/UNIFIED_BENCHMARK.md`](submission/UNIFIED_BENCHMARK.md). Reproduce a single task, e.g. anomaly:
  ```bash
  python competition/participant-files/eval_metrics.py --task anomaly \
      --ground-truth shared/extras/results/eval_inputs/eval_input_anomaly_truth.csv \
      --predictions  competition/participant-files/predictions/predictions_anomaly.csv
  ```
- [x] **Training artifacts**: per-run config and final loss in `shared/extras/checkpoints/*/summary.json`,
  TensorBoard loss curves in `shared/extras/logs/`.

## License

[`LICENSE`](LICENSE)
