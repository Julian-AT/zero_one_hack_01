# Neurosymbolic Process Engine (NSPE) — Results

## TL;DR

A **symbolic-first** approach to the Infineon process-logic track, deliberately the
opposite of a pure transformer. A symbolic engine (process grammar + the 10 rules +
a role ontology with **role-induction anchors**) defines, at every prefix, which next
steps are even *legal*; a small role-factored ranker only chooses *preferences inside
that legal set*. The whole point is **generalization to the hidden 4th product family**.

Measured on Leonardo (4× / 8× A100) and locally with the organizers' official scorer:

```text
Task 3 — anomaly (rule checking)
  In-distribution (3 families, all 10 rules):  Accuracy / Precision / Recall / F1 / Rule-Attribution = 1.000
  Out-of-distribution (unseen families, renamed steps):
      correct detection + attribution   roles OFF = 0.19  ->  roles ON = 1.00   (0 false positives)
      role-induction anchors recover all 9 surface-renamable rules (10th is structural)

Tasks 1 & 2 — next-step / completion, leave-one-family-out (the OOD proxy)
  mean next-step Top-1 ID->OOD DROP:
      constrained neural ranker  = +0.015   (~1.5 pp)
      PPM (pure symbolic)        = +0.037   (~3.7 pp)
      pure-neural baseline (ref) = +0.240   (~24 pp)      => ~6-16x flatter
  Top-5 OOD stays 0.96-1.00 everywhere; every completion is rule-valid by construction.
```

The headline is not in-distribution Top-1 (this is a deliberately tiny ~0.68M-param
model). It is that **the ID→OOD curve is essentially flat** where a pure-neural model
collapses — because the rules and the step *roles* are family-agnostic and are used
directly instead of being re-learned.

---

## 1. What we built

```text
process grammar + 10 rules + role ontology (role-induction anchors)
  -> validate / validate_with_roles  (exact official validator; novel steps canonicalized)
  -> valid_next_set(prefix)           (the legal support; gold is always in it)
  -> anomaly oracle  (Task 3)
  -> PPM role-factored Markov ranker  +  small constrained neural ranker (Tasks 1 & 2)
  -> constrained decoding + symbolic repair  -> rule-valid completions
  -> official-format submissions for all 3 tasks
```

The contribution is the **symbolic core**; the neural ranker is a small, constrained,
role-factored passenger. Everything lives in `neurosymbolic-approach/`.

---

## 2. Task 3 — anomaly detection (symbolic oracle + role-induction anchors)

Scored with the organizers' `eval_metrics.py`. ID = MOSFET/IGBT/IC with injected
violations of each of the 10 rules. OOD = simulated unseen families (DIODE, SCHOTTKY,
SIC_MOSFET) with the violation's trigger/anchor step **renamed** to an unseen string.

| Setting | Accuracy | Precision | Recall | F1 | Rule-attribution |
| --- | --: | --: | --: | --: | --: |
| ID (all 10 rules) | **1.000** | 1.000 | 1.000 | 1.000 | **1.000** |
| OOD, role-induction **OFF** | — | 1.000 | **0.19** | 0.32 | — |
| OOD, role-induction **ON** | — | 1.000 | **1.00** | 1.00 | **1.000** |

Role-induction anchors recover detection **and** attribution on **9 of 10 rules**
(0.00 → 1.00 per rule), with **0 false positives** in both modes. The 10th rule
(`RULE_LITHO_LEVEL_SKIP`) is structural — it keys on lithography level integers, which
have no surface-rename failure mode. On the *real* `eval_input_anomaly.csv` the oracle
returns exactly **600 valid / 387 invalid**, matching the documented ground-truth split.

| ID per-rule | OOD recovery |
| --- | --- |
| ![anomaly ID per rule](outputs/charts/anomaly_id_per_rule.png) | ![anomaly OOD recovery](outputs/charts/anomaly_ood_recovery.png) |

---

## 3. Tasks 1 & 2 — next-step / completion, leave-one-family-out

LoFO = train on two families, evaluate the held-out third (the in-house stand-in for
the hidden 4th family). The constrained neural ranker (~0.68M params, base config
`d=128, layers=3`) was trained on Leonardo A100s; the PPM ranker is pure symbolic/CPU.

**Next-step Top-1, ID → OOD drop** (lower = flatter = better):

| Held-out family | neural ID | neural OOD | neural drop | PPM ID | PPM OOD | PPM drop |
| --- | --: | --: | --: | --: | --: | --: |
| MOSFET | 0.647 | 0.700 | **−0.053** | 0.672 | 0.675 | −0.003 |
| IGBT | 0.688 | 0.719 | **−0.031** | 0.688 | 0.675 | +0.012 |
| IC | 0.753 | 0.625 | +0.128 | 0.728 | 0.625 | +0.103 |
| **mean** | | | **+0.015** | | | **+0.037** |
| *pure-neural baseline (ref)* | 0.72 | 0.48 | *+0.240* | | | |

Top-5 OOD stays **0.96–1.00** for both rankers. Every sequence completion is
**rule-valid by construction** (constrained decoding + symbolic repair); the neural
ranker also produces much better-terminating OOD completions (OOD NED ≈ 0.37–0.46 vs
PPM ≈ 0.62–0.92). Full-data (all-3-family) in-distribution Top-1: neural 0.704, PPM 0.696.

| Neural vs PPM (per held-out family) | The OOD thesis |
| --- | --- |
| ![neural vs ppm](outputs/charts/exp03_neural_vs_ppm.png) | ![ood drop](outputs/charts/ood_drop_comparison.png) |

| PPM LoFO next-step | PPM LoFO completion |
| --- | --- |
| ![ppm lofo nextstep](outputs/charts/ppm_lofo_nextstep.png) | ![ppm lofo completion](outputs/charts/ppm_lofo_completion.png) |

> Two supplementary panels were planned — the constraint-loss ablation (`exp04`)
> and the size/data scaling sweep (`exp05`). exp04's semantic-loss variant builds a
> per-position legal-step mask over the whole corpus in pure Python, which did not
> finish in practical time on this run (it held the node CPU-bound with no GPU work),
> so it was cancelled and exp05 (gated behind it) did not run. They are optional —
> the six charts above already cover the full thesis — and can be added after the
> mask precompute is optimized/cached.

---

## 4. How it was run on Leonardo

- Code/branch: `neurosymbolic-model`, env via the repo's pre-built pixi environment
  (`.pixi/envs/default`, torch 2.5.1+cu121); jobs use it directly (compute nodes have
  no internet).
- `slurm/test_debug.sbatch` (1 GPU, debug QoS): full pipeline smoke — passed
  (anomaly + tiny GPU train, `device=cuda`).
- `slurm/grid_lofo.sbatch` (4 A100, reservation `s_tra_ncc`, account `euhpc_d30_031`):
  LoFO ×3 families + constraint-loss ablation + aggregate + charts.
- `slurm/train_full.sbatch` (1 A100): full-data ranker + submissions.
- Outputs land in `$SCRATCH/nspe_outputs`; result JSONs + charts are copied back into
  `neurosymbolic-approach/outputs/`. See `RUN_ON_LEONARDO.md`.

---

## 5. What we can claim

**Strong claims**
- Rule checking generalizes to unseen families with **0 false positives**; with
  role-induction anchors it recovers detection + attribution on all 9 surface-renamable
  rules (0.19 → 1.00 recall).
- Next-step prediction transfers to held-out families nearly flat: **+0.015 (neural) /
  +0.037 (PPM)** mean Top-1 drop vs **+0.24** for a pure-neural baseline.
- Sequence completions are rule-valid by construction.
- A ~0.68M-param constrained ranker trains in seconds on one A100 and matches/beats the
  symbolic PPM on OOD.

**Careful claims**
- In-distribution Top-1 (~0.65–0.75) is intentionally not maximized — the model is tiny
  and briefly trained; the result is the *flat OOD curve*, not peak ID accuracy.
- The "4th family" here is a simulated proxy (renamed/unseen-vocab synthetic families);
  the true hidden family is only scored by the organizers.
- The pure-neural +0.24 baseline is the documented LoFO trigram reference, not a re-run.

**Avoid claiming**
- Official hidden-eval accuracy (only organizers can score it).
- Real-world fab validity — the data is synthetic.

---

## 6. Files

```text
neurosymbolic-approach/
  FINDINGS.md         high-level plan + measured-results addendum
  Implementation.md   full build spec
  RUN_ON_LEONARDO.md  copy-paste run recipe
  RESULTS.md          this file
  nspe/               symbolic core + ranker (official, roles, rules, grammar, ppm,
                      model, losses, decode, anomaly, predict, eval, ...)
  experiments/        exp01..exp06 + ood_symbolic_probe + make_charts
  slurm/              test_debug / grid_lofo / train_full / env_setup
  outputs/
    charts/           anomaly_id_per_rule, anomaly_ood_recovery, ppm_lofo_nextstep,
                      ppm_lofo_completion, ood_drop_comparison, exp03_neural_vs_ppm (+ exp04/05 when ready)
    exp01.json exp02.json exp03_{mosfet,igbt,ic,none}.json
    submission_task{1,2,3}.csv
```
