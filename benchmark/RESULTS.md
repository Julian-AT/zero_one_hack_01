# Benchmark Results — Cross-Architecture Model Comparison

> **Two-stage report.**
> - **Stage 1 (below, populated now):** best-available numbers compiled from each model's *own*
>   committed run. These come from different splits/sizes, so they are **indicative, not a
>   controlled comparison** — read the caveat in each table.
> - **Stage 2 (to be produced by the harness):** all models scored on **one common held-out eval
>   set** via `participant_files/eval_metrics.py`. Run `benchmark/run_benchmark.py` (see
>   `benchmark/README.md`) on Leonardo to fill the Stage-2 tables.
>
> **Official accuracy is not computable locally** (organizer eval is unlabeled). All numbers here are
> internal held-out / leave-one-family-out (LoFO).

Scope: MVP, already-trained models (Transformer, Neurosymbolic, n-gram/retrieval baselines).
LSTM/xLSTM are implemented but untrained → not included (see `benchmark_plan.md` §Future Work).

---

## Sources of these numbers

| Model | Source file | Eval set used (Stage 1) |
|---|---|---|
| SSL Transformer (coverage-guided hybrid) | `ssl_results/hybrid_coverage_guided_metrics.csv`, `ssl_results/learned_reranker_report.md` | coverage-guided internal test split (large) |
| Trigram-with-backoff | `extras/results/baselines/trigram_metrics.json` | 3000 base seqs, 80/20, seed 42 |
| Grammar decoder | `extras/results/baselines/grammar_decoder_metrics.json` | same base-seq split |
| Retrieval | `extras/results/baselines/retrieval_metrics.json` | same base-seq split |
| Neurosymbolic — neural ranker (~0.68M) | `neurosymbolic-approach/outputs/exp03_*.json` (branch `neurosymbolic-model`) | 80 seqs/family LoFO folds |
| Neurosymbolic — PPM (symbolic) | `neurosymbolic-approach/outputs/exp03_*.json` | same |
| Neurosymbolic — anomaly oracle | `neurosymbolic-approach/outputs/exp01.json` | 600 ID / 600 OOD |

---

## Task 1 — Next-step prediction

### In-distribution (each on its own held-out split — NOT a common set)

| Model | Top-1 | Top-3 | Top-5 | MRR | Notes |
|---|---:|---:|---:|---:|---|
| **SSL Transformer (cov-guided)** | **0.8031** | 0.9932 | 0.9997 | 0.8976 | base model |
| SSL Transformer + learned reranker | **0.8044** | 0.9931 | 0.9994 | 0.8979 | **shipped next-step** |
| Trigram backoff | 0.7173 | 0.9675 | 0.9931 | 0.8416 | memorization floor |
| Grammar decoder | 0.7173 | 0.9805 | 0.9957 | 0.8469 | |
| Neurosymbolic — neural | 0.6958 | ~0.998 | ~1.000 | 0.846 | macro over 3 folds; 80/family |
| Neurosymbolic — PPM | 0.6958 | 1.000 | 1.000 | 0.846 | macro over 3 folds |

> Caveat: the Transformer was scored on the larger, harder coverage-guided split; baselines and
> neurosymbolic on the base 3-family data. Higher Transformer Top-1 is **not** a like-for-like win
> until Stage 2.

### Out-of-distribution (Leave-One-Family-Out) — the discriminating axis

| Model | ID Top-1 (macro) | OOD Top-1 (macro) | **ID→OOD drop** | Verdict |
|---|---:|---:|---:|---|
| Trigram backoff | 0.7173 | 0.4715 | **+0.246** | collapses → memorizes |
| Neurosymbolic — PPM | 0.6958 | 0.6583 | **+0.037** | nearly flat → learns structure |
| Neurosymbolic — neural | 0.6958 | 0.6813 | **+0.015** | flattest → strongest OOD |
| SSL Transformer | 0.8031 | — | **requires retrain** | out of MVP scope (trained on all families) |

LoFO per-family detail (neurosymbolic, from `exp03_*.json`):

| Held-out | neural ID→OOD Top-1 | PPM ID→OOD Top-1 |
|---|---|---|
| MOSFET | 0.647 → 0.700 (−0.053) | 0.672 → 0.675 (−0.003) |
| IGBT | 0.688 → 0.719 (−0.031) | 0.688 → 0.675 (+0.013) |
| IC | 0.753 → 0.625 (+0.128) | 0.728 → 0.625 (+0.103) |

**Headline finding:** in-distribution Top-1 is near-saturated and rankings there are noisy; the
**ID→OOD drop separates the approaches** — the trigram baseline loses ~25 points off the held-out
family while the neurosymbolic models stay within ~1.5–3.7 points. That gap is the evidence for
"learned process logic vs. memorization." The Transformer's place on this axis is **unknown** until
a LoFO retrain (Stage 2 / Future Work).

---

## Task 2 — Sequence completion (Normalized Edit Distance, lower better)

| Model | ID NED | OOD (LoFO) NED | Rule-valid completions | Notes |
|---|---:|---:|---:|---|
| Neurosymbolic — neural | 0.18–0.26 | 0.37–0.46 | **100% by construction** | constrained decoding + repair |
| Neurosymbolic — PPM | 0.48–0.52 | 0.62–0.92 | 100% by construction | |
| Retrieval baseline | 0.23–0.32 (@0.6) | 0.42–0.76 | not enforced | nearest-neighbor |
| SSL Transformer | — | — | not separately scored | generates completions; not internally NED-scored yet |

> The neurosymbolic completions are **guaranteed rule-valid**; the Transformer's completions are not
> yet scored on a labeled internal split — **Stage 2 closes this** (it's also the SUBMISSION.md
> "completion score" gap).

---

## Task 3 — Anomaly detection + rule attribution

| Model | Setting | Acc | Precision | Recall | F1 | AUC | Rule-Attr |
|---|---|---:|---:|---:|---:|---:|---:|
| Neurosymbolic oracle | ID (all 10 rules) | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | **1.000** |
| Neurosymbolic oracle | OOD, roles ON | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 | **1.000** |
| Neurosymbolic oracle | OOD, roles OFF | 0.677 | 1.000 | 0.192 | 0.322 | 0.596 | 0.565 |
| SSL / validator-based | ID | — | — | — | ≈1.0 (validator-exact) | — | ≈1.0 | 

> The shipped `predictions_anomaly.csv` is validator-based, i.e. the same rule checker as the
> neurosymbolic ID oracle, so on ID it is essentially exact — but it has **not been formally scored
> on a labeled internal split** (Stage 2). The neurosymbolic **role-induction** result is the real
> story: it recovers detection + attribution on 9/10 rules under renamed OOD steps (0.19 → 1.00
> recall).

---

## Efficiency (where reported)

| Model | Params | Train cost | Notes |
|---|---:|---|---|
| Neurosymbolic — neural ranker | ~0.68M | ~7 s/fold (A100) | tiny; symbolic core does the work |
| Neurosymbolic — PPM | 0 (counts) | CPU only | |
| SSL Transformer (cov-guided) | ~4–8M (unverified) | 30 epochs GPU | see `ssl_results/README.md` §6/§8 |
| Trigram / retrieval | n/a | CPU | |

---

## How to produce the Stage-2 (common-split) tables

See `benchmark/README.md`. In short: `prepare_eval_inputs.py` builds one common labeled eval set +
organizer-format inputs from the held-out split; each model produces organizer-format predictions on
those inputs; `run_benchmark.py` scores all of them with `eval_metrics.py` into `results.csv` and
regenerates the Stage-2 tables. This removes the different-split caveat and adds the Transformer's
completion/anomaly/LoFO cells.
