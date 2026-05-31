# Process-Logic — Semiconductor Process-Sequence Modeling

Learning the *grammar* of semiconductor fabrication flows: given partial MOSFET/IGBT/IC
process sequences, predict the next step, complete the sequence, and flag sequences that
violate process rules. Built for the Industrial AI (Infineon) track of Zero One Hack_01.

The core question: does a model learn real process logic, or just memorize? The pipeline
generates its own synthetic training data from a rule-based process simulator, trains a
decoder Transformer (optionally xLSTM) on it, and emits organizer-format predictions.

## The four tasks

1. **Next-step prediction** — given a partial sequence, rank the 5 most likely next steps.
2. **Sequence completion** — given a partial sequence, generate the remaining suffix.
3. **Anomaly detection** — given a full sequence, classify valid vs. invalid.
4. **Rule attribution** — given an invalid sequence, identify which of the 10 process
   rules was violated.

## 👉 Our submission — Industrial AI (Infineon)

This repo is our team's entry for the **Industrial AI** track. Start here:

- **[`REPORT.md`](./REPORT.md)** — executive summary: approach, results, final files, how to run.
- **[`submission/UNIFIED_BENCHMARK.md`](./submission/UNIFIED_BENCHMARK.md)** — the three model approaches
  compared head-to-head on **one common eval set** with the **official metrics** (ID + OOD).
- **[`models/self-supervised/README.md`](./models/self-supervised/README.md)** — full technical write-up.
- **[`competition/participant-files/predictions/`](./competition/participant-files/predictions/)** — final submission CSVs.

## Reproduce the unified benchmark — one command, no Leonardo needed

All three approaches (Transformer, SSL-Hybrid, Neurosymbolic) + two reference baselines, scored on
the **same** held-out data with the **same** official `eval_metrics.py`, in both in-distribution and
out-of-distribution (leave-one-family-out) regimes — from a clean checkout on any machine:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./reproduce.sh            # full run: build eval set → train compact models → score → report
                          #   CPU ~60-75 min · CUDA auto-detected (~10 min) · MPS skipped (unstable)
./reproduce.sh quick      # <2 min smoke test: baselines + neurosymbolic only (no neural training)
```

Output → `shared/benchmark/results_summary.csv` and **[`submission/UNIFIED_BENCHMARK.md`](./submission/UNIFIED_BENCHMARK.md)**
(tables + figures in `submission/benchmark_assets/`). Baselines + neurosymbolic need no training and
score instantly; the transformer/SSL checkpoints are compact, trained locally for laptop
reproducibility (≈ production lm-loss; see UNIFIED_BENCHMARK.md §8).

> **Full-scale (optional, Leonardo).** The compact checkpoints above reproduce the *comparison*. For
> the production-scale numbers (max_len 768, 6k steps, the xLSTM architecture, multiple model sizes)
> use the SLURM scripts in `shared/scripts/slurm/` per [`docs/leonardo.md`](docs/leonardo.md);
> `pip install "xlstm>=1.0.7"` on the CUDA node first. This is **not required** to verify the results.

---

## Repository layout

```text
models/transformer_xlstm/                    # the library — clean, typed, tested
  data/                 #   tokenizers, validator adapter, corrupters, CSV/sequence I/O
  model/                #   decoder Transformer, xLSTM, heads, build_model registry
  train/                #   launch CLI, training loop, losses, tracking
  eval/                 #   predict, run_eval CLI, metrics, submission writer
  utils/                #   paths, seeding
configs/                # OmegaConf YAML: arch/ token/ train/ (nothing hardcoded)
tests/                  # pure-logic pytest suite (tokenizer, validator, corrupt, metrics, io)
baselines (shared/extras/)     # trigram, grammar-decoder, retrieval reference baselines
competition/participant-files/      # competition submission pipeline (hybrid model + rerankers)
competition/track-details/   # organizer data + the canonical generate_sequences.py grammar
docs/                   # results.md (full results narrative), leonardo.md (HPC guide)
```

## Setup

The environment is managed with [pixi](https://pixi.sh). It pins the platform-specific
install — conda `pytorch-gpu` on Linux (Leonardo's CUDA-12 driver), CPU/MPS torch on
macOS — and is the source of truth for the cluster environment.

```bash
pixi install
pixi run smoke        # prints torch version + CUDA availability
```

For a plain virtualenv (CPU, no cluster specifics), the abstract dependencies are also
declared in `pyproject.toml`:

```bash
pip install -e .          # runtime deps
pip install -e ".[dev]"   # + pytest, ruff
```

## Train

Training streams freshly generated sequences from the rule-based simulator (with on-the-fly
corruption); no static training CSV is loaded. Configs are merged at launch via OmegaConf.

```bash
python -m transformer_xlstm.train.launch \
    --arch-config  configs/arch/transformer_small.yaml \
    --train-config configs/train/default.yaml \
    --token-config configs/token/compositional.yaml \
    --run-name     my-run-001
```

Override any config value inline:

```bash
python -m transformer_xlstm.train.launch ... --override train.max_steps=100 train.batch_size=16
```

Checkpoints are written to `shared/extras/checkpoints/<run-name>/` (gitignored; only `summary.json`
is committed). On Leonardo, submit via `shared/scripts/slurm/train.sbatch`.

## Evaluate

Run the internal metric suite against a checkpoint (next-step top-k / MRR, completion
EM / normalized edit distance, anomaly precision/recall):

```bash
python -m transformer_xlstm.eval.run_eval \
    --checkpoint shared/extras/checkpoints/my-run-001/final.pt \
    --output-dir shared/extras/results/eval/my-run-001
```

This writes `metrics.json` + `metrics.md`. The official organizer eval inputs are
unlabeled, so final official accuracy can only be computed by the organizers — we generate
official-format prediction CSVs from them.

## Develop

```bash
pixi run lint          # ruff check .
pixi run format        # ruff format .
pixi run test          # pytest
```

## Further reading

- [`docs/results.md`](docs/results.md) — full pipeline writeup, model comparison, and an
  honest account of what the results can and cannot claim.
- [`docs/leonardo.md`](docs/leonardo.md) — authenticating, environment setup, and SLURM job
  submission on the Leonardo cluster.
- [`competition/track-details/`](competition/track-details/) — the organizer briefing,
  reference data, and `generate_sequences.py` (the canonical grammar and validator).
