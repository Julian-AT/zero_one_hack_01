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
- **[`ssl_results/README.md`](./ssl_results/README.md)** — full technical write-up.
- **[`participant_files/predictions/`](./participant_files/predictions/)** — final submission CSVs.

---

## Repository layout

```text
src/                    # the library — clean, typed, tested
  data/                 #   tokenizers, validator adapter, corrupters, CSV/sequence I/O
  model/                #   decoder Transformer, xLSTM, heads, build_model registry
  train/                #   launch CLI, training loop, losses, tracking
  eval/                 #   predict, run_eval CLI, metrics, submission writer
  utils/                #   paths, seeding
configs/                # OmegaConf YAML: arch/ token/ train/ (nothing hardcoded)
tests/                  # pure-logic pytest suite (tokenizer, validator, corrupt, metrics, io)
baselines (extras/)     # trigram, grammar-decoder, retrieval reference baselines
participant_files/      # competition submission pipeline (hybrid model + rerankers)
tracks/industrial-infineon/   # organizer data + the canonical generate_sequences.py grammar
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
python -m src.train.launch \
    --arch-config  configs/arch/transformer_small.yaml \
    --train-config configs/train/default.yaml \
    --token-config configs/token/compositional.yaml \
    --run-name     my-run-001
```

Override any config value inline:

```bash
python -m src.train.launch ... --override train.max_steps=100 train.batch_size=16
```

Checkpoints are written to `extras/checkpoints/<run-name>/` (gitignored; only `summary.json`
is committed). On Leonardo, submit via `scripts/slurm/train.sbatch`.

## Evaluate

Run the internal metric suite against a checkpoint (next-step top-k / MRR, completion
EM / normalized edit distance, anomaly precision/recall):

```bash
python -m src.eval.run_eval \
    --checkpoint extras/checkpoints/my-run-001/final.pt \
    --output-dir extras/results/eval/my-run-001
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
- [`tracks/industrial-infineon/`](tracks/industrial-infineon/) — the organizer briefing,
  reference data, and `generate_sequences.py` (the canonical grammar and validator).
