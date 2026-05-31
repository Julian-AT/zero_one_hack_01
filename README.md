# Process-Logic

Semiconductor fab routes are long, ordered sequences of process steps whose validity depends on that
order. For the **Industrial AI (Infineon)** track we asked one question: can a model learn the
*grammar* of those routes — predict the next step, complete a partial route, and flag rule
violations — or does it only memorize? To answer it honestly we generate synthetic routes from a
rule-based simulator, train on them, and test generalization by holding out an entire product family.

We built and compared **three approaches** on a single shared benchmark:

- **Transformer** — `models/transformer_xlstm/`. A decoder transformer (optionally xLSTM) with a
  compositional tokenizer and multitask validity/rule heads. Our most mature model.
- **SSL-Hybrid** — `models/self-supervised/`. A self-supervised transformer with semantic-feature
  and family embeddings, plus retrieval/rerank.
- **Neurosymbolic** — `models/neurosymbolic/`. A symbolic grammar and 10-rule oracle with role
  induction, ranked by a zero-parameter PPM.

The three submission tasks — next-step prediction, sequence completion, anomaly detection — are
scored by the organizers' `eval_metrics.py`. The head-to-head results are in
[`submission/UNIFIED_BENCHMARK.md`](submission/UNIFIED_BENCHMARK.md); the technical write-up is in
[`REPORT.md`](REPORT.md).

## Run it on Leonardo (GPU)

Everything below runs from a **login node** (which has internet) and submits to a GPU node. Use your
own project account — nothing here is tied to a specific allocation or reservation.

```bash
# 1. clone into your scratch space
cd "$SCRATCH"
git clone <repo-url> zero_one_hack_01
cd zero_one_hack_01

# 2. one-time: build the pinned environment (pixi + CUDA PyTorch). ~10 min, login node only.
bash shared/scripts/leonardo/setup_env.sh

# 3. submit the benchmark to a GPU node — substitute YOUR account:
sbatch --account=<your_account> reproduce.sbatch

# 4. watch it
squeue --me
tail -f reproduce-*.out
```

`reproduce.sbatch` requests one A100 on the `boost_usr_prod` partition, loads `cuda`/`gcc`, and runs
the full benchmark on the GPU (~15 min). It uses the environment from step 2 directly, so the compute
node needs no internet. Add `--reservation=<name>` to the `sbatch` line only if your project uses one;
otherwise the job goes to the normal GPU queue.

Outputs: `shared/benchmark/results_summary.csv` and the report at
[`submission/UNIFIED_BENCHMARK.md`](submission/UNIFIED_BENCHMARK.md) (figures in
`submission/benchmark_assets/`). The production-scale training grids (max_len 768, the xLSTM
architecture, multiple sizes) live in `shared/scripts/slurm/`; see [`docs/leonardo.md`](docs/leonardo.md).

## Run it locally (any machine, CPU is fine)

No GPU or cluster access required:

```bash
git clone <repo-url> && cd zero_one_hack_01
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./reproduce.sh           # build eval set → train compact models → score → write the report
./reproduce.sh quick     # ~2-min smoke: baselines + neurosymbolic only, no training
```

`reproduce.sh` runs the same comparison end to end — seed-pinned and deterministic, same outputs as
the Leonardo run. It auto-detects CUDA (≈10 min) and otherwise runs on CPU (≈60–75 min; Apple MPS is
skipped — its kernels are unstable for this model).

## Repository layout

```
reproduce.sbatch / reproduce.sh   one-command reproduction (Leonardo / local) — start here
models/
  transformer_xlstm/   decoder transformer + xLSTM: tokenizers, training, eval (our primary model)
  self-supervised/     SSL-hybrid transformer + retrieval/reranker, metrics, plots
  neurosymbolic/       grammar, 10-rule oracle, role induction, PPM ranker (0 trained parameters)
competition/
  track-details/       the brief, the rule-based sequence generator, and the validation rules
  participant-files/   the official scorer (eval_metrics.py) and our submission CSVs
shared/
  benchmark/           the unified benchmark: eval-set builder, per-model adapters, scorer, report
  scripts/             SLURM jobs + Leonardo setup
  extras/              checkpoint summaries, training logs, loss curves, baselines, raw results
configs/               OmegaConf YAML for architecture / tokenizer / training (nothing hardcoded)
submission/            REPORT material and the cross-model benchmark report + figures
docs/                  full results narrative and the Leonardo operations guide
tests/                 pytest suite for tokenizers, validator, metrics, I/O
```

## Deliverables

- [x] **[`REPORT.md`](REPORT.md)** — the technical report.
- [x] **[`submission/UNIFIED_BENCHMARK.md`](submission/UNIFIED_BENCHMARK.md)** — all three approaches on
  one eval set with the official metrics, in-distribution and out-of-distribution (held-out family).
- [x] **[`competition/participant-files/predictions/`](competition/participant-files/predictions/)** —
  the submission CSVs: `predictions_nextstep.csv`, `predictions_completion.csv`, `predictions_anomaly.csv`.
- [x] Score them with the organizers' script — e.g. the anomaly task against our labeled eval set:
  ```bash
  python competition/participant-files/eval_metrics.py --task anomaly \
      --ground-truth shared/extras/results/eval_inputs/eval_input_anomaly_truth.csv \
      --predictions  competition/participant-files/predictions/predictions_anomaly.csv
  ```
- [x] Training artifacts — per-run config and final loss in `shared/extras/checkpoints/*/summary.json`,
  TensorBoard loss curves in `shared/extras/logs/`.

## Requirements & license

Python ≥ 3.10 and the packages in [`requirements.txt`](requirements.txt) (PyTorch, NumPy, pandas,
OmegaConf, matplotlib). The benchmark is CPU-only; the xLSTM architecture additionally needs CUDA
(`pip install "xlstm>=1.0.7"` on a GPU node). On Leonardo the pinned environment is managed by pixi
(`pixi.toml`). Released under the MIT License — see [`LICENSE`](LICENSE).
