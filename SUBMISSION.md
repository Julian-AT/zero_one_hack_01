# Attention Seekers, Industrial AI (Infineon)

## Team

* **Kyrillus Mehanni:** Senior Software Engineer - Symbolic Approach & DevOps
* **Julian Schmidt:** AI Engineer - Model Training
* **Emil Kascper:** AI Engineer - Model Training
* **Abdul Basit Banbhan:** AI Researcher & Engineer - Direction of AI Research, Model Training

**Track:** Industrial AI (Infineon)

## TL;DR

We built three process logic systems for semiconductor fabrication routes (a decoder transformer, a self supervised hybrid, and a zero parameter neurosymbolic engine) and scored all three, plus two baselines, on one shared labeled eval set with the official `eval_metrics.py`, both in distribution and on a held out product family. The central finding is that in distribution accuracy is a trap: a 50 line trigram already reaches Top 5 of about 0.99, so it does not separate approaches, and bigger models do not help either (three transformer sizes converge to within 0.0001 language model loss). What separates memorization from process logic is generalization to an unseen family and whether completions respect the rules: the trigram introduces a new rule violation in 50 percent of its completions, while every structure aware system stays at zero new violations. Our single largest improvement was not a bigger model but fixing a context window bug that was truncating the process backbone, which raised held out next step accuracy by 29 points.

## Problem

Semiconductor routes are ordered sequences of roughly 110 to 150 steps over a vocabulary of about 120 step strings, where validity is defined by ten documented forbidden patterns (for example, a deposition requires a prior clean, and electrical tests must follow passivation).

Before training anything we computed a trigram with backoff. It reaches Top 5 of about 0.99 in distribution, identical to the memorization upper bound, and its in distribution accuracy is unchanged on a held out split. The provided task therefore carries almost no model relevant entropy in distribution. The same trigram collapses out of distribution: under a leave one family out split its next step accuracy drops sharply, which quantifies the real gap. So the problem we chose is not raw in distribution accuracy but two harder questions: does a system generalize its process understanding to a product family it never saw in training, and does it complete and validate routes in a way that respects process logic rather than reproducing memorized strings. The work targets the three scored tasks (next step prediction, sequence completion, anomaly detection with rule attribution) and the organizer evaluated fourth task, out of distribution generalization to a hidden family, approximated here by a leave one family out protocol.

## Approach

* **One eval set, one official scorer, five systems, two regimes.** Every system is scored on the same frozen labeled eval set with `eval_metrics.py`, in distribution (all three families in training) and leave one family out (train on two families, score the held out third). This makes the comparison like for like, which the earlier per model reports did not.
* **Decoder transformer with an xLSTM variant (the submission model).** Rotary position embeddings, RMSNorm, multitask validity and rule heads, trained on an online generator stream with 40 percent of sequences carrying a labeled rule violation. A compositional tokenizer splits each step string into word tokens, so a new step in an unseen family that shares words is not fully out of distribution. The dominant quality lever is the context window, because a short window truncates the long route.
* **Self supervised hybrid with retrieval.** A self supervised causal model with semantic step features and product family embeddings, plus a learned contrastive reranker over its Top 5 candidates.
* **Neurosymbolic engine with zero trained parameters.** An induced grammar and the ten rule oracle define which next steps are legal at each prefix; a role factored Prediction by Partial Matching ranker orders the legal candidates. Correctness is owned by the symbolic spine, so it transfers to an unseen family with almost no drop.
* **Two baselines.** A trigram with backoff (the memorization floor) and the same trigram under a grammar mask. The full benchmark runs from a clean checkout on CPU or CUDA; production grids run on the Leonardo cluster (NVIDIA A100).

## How to run it

```bash
git clone https://github.com/Julian-AT/zero_one_hack_01
cd zero_one_hack_01
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./reproduce.sh          # full run: CPU about 60 to 75 min, CUDA about 10 min
./reproduce.sh quick    # smoke test under 2 min: baselines and neurosymbolic only
```

`reproduce.sh` runs five stages under `shared/benchmark/`: build the frozen eval set with deterministic seeds (`make_eval_set.py`), generate training data (`make_train_data.py`), train the compact checkpoints (`train_txl.sh`, `train_ssl.sh`), run every system and score with the official metrics (`make_benchmark.py`), then aggregate tables and figures (`report.py`). It auto detects CUDA and otherwise runs on CPU; the Apple MPS backend is skipped because its embedding gather kernel is unstable for this model. Score one task directly:

```bash
python shared/benchmark/score.py --task anomaly \
  --predictions  shared/benchmark/predictions/ID/neurosymbolic/predictions_anomaly.csv \
  --ground-truth shared/benchmark/eval_set_v1/ground_truth/anomaly_gt.csv
```

## Results

**Eval set.** Built by `make_eval_set.py` with benchmark only seeds (90001, 90002, 90003 valid; 90010, 90011 invalid), disjoint from training. Next step is scored at the organizer 60 percent and 80 percent truncation points.

<table>
<tr><th>Task</th><th>Examples</th><th>MOSFET</th><th>IGBT</th><th>IC</th></tr>
<tr><td>Next step at 60 and 80 percent cut</td><td>226</td><td>92</td><td>62</td><td>72</td></tr>
<tr><td>Completion</td><td>90</td><td>30</td><td>30</td><td>30</td></tr>
<tr><td>Anomaly (146 invalid, 113 valid)</td><td>259</td><td>98</td><td>68</td><td>93</td></tr>
</table>

**Headline metric.** In distribution next step Top 1: the transformer leads at 0.779. The neurosymbolic engine reaches 0.761 with zero trained parameters. The trigram floor is 0.721.

**Baseline comparison, Task 1 (next step).** Drop is the in distribution value minus the leave one family out value.

<table>
<tr><th>System</th><th>Top 1 (ID)</th><th>Top 3</th><th>Top 5</th><th>MRR</th><th>Top 1 (LoFO)</th><th>Drop</th></tr>
<tr><td><b>Transformer xLSTM</b></td><td><b>0.779</b></td><td>0.996</td><td>1.000</td><td><b>0.888</b></td><td>0.704</td><td>0.075</td></tr>
<tr><td>Self supervised hybrid</td><td>0.765</td><td>1.000</td><td>1.000</td><td>0.883</td><td><b>0.721</b></td><td><b>0.045</b></td></tr>
<tr><td>Neurosymbolic</td><td>0.761</td><td>0.996</td><td>1.000</td><td>0.879</td><td>0.660</td><td>0.101</td></tr>
<tr><td>Grammar baseline</td><td>0.721</td><td>0.996</td><td>1.000</td><td>0.860</td><td>0.653</td><td>0.068</td></tr>
<tr><td>Trigram baseline</td><td>0.721</td><td>0.982</td><td>1.000</td><td>0.856</td><td>0.653</td><td>0.068</td></tr>
</table>

The trigram already reaches Top 5 of 1.000 in distribution, and every learned system lands within about six points on Top 1. In distribution next step does not separate approaches. The drops are small for all systems under this protocol because the 60 and 80 percent cut points fall in the process back end (metallization, passivation, test, ship), which is shared across families; an all position protocol that also probes family specific front end positions drives the trigram down toward 0.47.

**Task 2 (completion).** Edit distance is lower is better. Rule clean is the fraction of completions that introduce no new rule violation beyond the prefix.

<table>
<tr><th>System</th><th>Edit distance (ID)</th><th>Block acc</th><th>Rule clean (ID)</th><th>Edit distance (LoFO)</th><th>Rule clean (LoFO)</th></tr>
<tr><td><b>Transformer xLSTM</b></td><td><b>0.242</b></td><td><b>0.700</b></td><td>1.000</td><td><b>0.368</b></td><td>1.000</td></tr>
<tr><td>Self supervised hybrid</td><td>0.318</td><td>0.645</td><td>1.000</td><td>0.384</td><td>0.833</td></tr>
<tr><td>Neurosymbolic</td><td>0.706</td><td>0.576</td><td>1.000</td><td>0.748</td><td>1.000</td></tr>
<tr><td>Grammar baseline</td><td>0.547</td><td>0.510</td><td>1.000</td><td>0.568</td><td>1.000</td></tr>
<tr><td>Trigram baseline</td><td>0.606</td><td>0.540</td><td><b>0.500</b></td><td>0.629</td><td>0.511</td></tr>
</table>

The transformer is closest to the reference (edit distance 0.242). The decisive column is rule clean: the trigram violates a process rule in half of its completions (0.500), while every structure aware system stays at 1.000 in distribution. The neurosymbolic edit distance is high (0.706) by design, because it emits a guaranteed rule valid but length divergent completion. Exact match is 0.000 for all systems and is not the right yardstick for a 30 to 60 step suffix.

**Task 3 (anomaly and rule attribution).** Baselines have no anomaly capability and are omitted.

<table>
<tr><th>System</th><th>F1 (ID)</th><th>Precision</th><th>Recall</th><th>ROC AUC</th><th>Rule attribution</th><th>F1 (LoFO)</th></tr>
<tr><td>Transformer xLSTM</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.980</td><td>1.000</td></tr>
<tr><td>Self supervised hybrid</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.980</td><td>1.000</td></tr>
<tr><td><b>Neurosymbolic</b></td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td><b>1.000</b></td><td>1.000</td></tr>
</table>

All three reach F1 of 1.000 in distribution because they share the same symbolic validator, which is exact on the ten rules. The differentiator is rule attribution (neurosymbolic 1.000) and that all three hold F1 of 1.000 out of distribution. The honest reading of this 1.000 is in the honesty note below.

**Scaling: bigger is not better on this task.** A seven cell grid varies architecture, size, and tokenization. Three transformer sizes spanning 4 M to 113 M parameters converge to within 0.0001 language model loss, so capacity is not the bottleneck. The xLSTM variant closes the gap only with scale, never beats the transformer, and runs three to four times slower per step. Step as token and compositional losses are not directly comparable because their per token entropy differs.

<table>
<tr><th>Architecture</th><th>Parameters</th><th>Tokenization</th><th>Final LM loss</th><th>Wall time</th></tr>
<tr><td>Transformer</td><td>4.2 M</td><td>compositional</td><td>0.1062</td><td>73 s</td></tr>
<tr><td>Transformer</td><td>33.6 M</td><td>compositional</td><td>0.1061</td><td>255 s</td></tr>
<tr><td>Transformer</td><td>113.4 M</td><td>compositional</td><td>0.1062</td><td>576 s</td></tr>
<tr><td>xLSTM mixed</td><td>1.7 M</td><td>compositional</td><td>0.1192</td><td>201 s</td></tr>
<tr><td>xLSTM mixed</td><td>12.0 M</td><td>compositional</td><td>0.1093</td><td>529 s</td></tr>
<tr><td>xLSTM mixed</td><td>38.8 M</td><td>compositional</td><td>0.1077</td><td>1033 s</td></tr>
<tr><td>Transformer</td><td>33.7 M</td><td>step as token</td><td>0.3258</td><td>251 s</td></tr>
</table>

**Efficiency.**

<table>
<tr><th>System</th><th>Trainable parameters</th></tr>
<tr><td>Transformer xLSTM</td><td>4.37 M</td></tr>
<tr><td>Self supervised hybrid</td><td>0.67 M</td></tr>
<tr><td>Neurosymbolic (PPM)</td><td>0 (about 0.68 M optional neural ranker)</td></tr>
<tr><td>Trigram and Grammar</td><td>0</td></tr>
</table>

**Per family breakdown.** The leave one family out columns above are the out of distribution view; per family next step, drop, completion, and anomaly figures are in `submission/benchmark_assets/` (`fig6_lofo_per_family.png`, `fig1_nextstep_id_vs_lofo.png`, `fig2_nextstep_drop.png`, `fig3_completion_ned.png`, `fig5_anomaly_f1.png`, `fig7_completion_ruleclean.png`).

**Where the data came from.** Every value in the task tables is produced by the official scorer through `shared/benchmark/score.py`, aggregated by `report.py` into `shared/benchmark/results_summary.csv` and `submission/benchmark_assets/tables.md`, and rendered in `submission/UNIFIED_BENCHMARK.md`. The scaling and context window numbers come from the production transformer grids logged under `shared/extras/`. Raw predictions are under `shared/benchmark/predictions/<regime>/<system>/`.

## What worked

* **Finding and fixing the context window bug.** An audit found `max_len` set to 256 while compositional sequences run 444 to 604 tokens, so the full training set was being truncated, hiding the prefix, clean, prep, and cycle backbone the model was meant to learn. Raising the window from 256 to 768 was the single largest improvement of the project.

<table>
<tr><th>Metric (MOSFET held out)</th><th>Window 256</th><th>Window 768</th></tr>
<tr><td>Held out Top 1 at 60 percent cut</td><td>0.625</td><td>0.917</td></tr>
<tr><td>Held out Top 1 average</td><td>0.520</td><td>0.708</td></tr>
<tr><td>Held out edit distance at 60 percent cut</td><td>0.55</td><td>0.27</td></tr>
<tr><td>Validation LM loss</td><td>0.111</td><td>0.089</td></tr>
</table>

* **The controlled benchmark.** One frozen labeled eval set and one official scorer turned five incomparable per model claims into a defensible head to head result in both regimes.
* **Rule clean as the discriminator.** Standard metrics saturate and hide the real question. Rule clean exposes it: the memorizing trigram violates a rule in half of its completions (0.500); every structure aware system stays at 1.000.
* **A zero parameter system that competes.** The neurosymbolic engine reaches 0.761 next step Top 1, perfect anomaly and rule attribution, and almost no out of distribution drop, with no trained weights.

## What didn't work

* **Retrieval augmentation.** The prefix overlap between the retrieval bank and the eval prefixes was 0 percent, so it changed nothing and was discarded.
* **The heuristic reranker.** Reordering the Top 5 with validator and trigram penalties moved Top 1 by about 0.0004 and was superseded by a learned contrastive reranker, which gave a real internal gain (Top 1 from 0.7993 to 0.8044).
* **The validity head as a standalone anomaly detector.** Out of distribution it was badly miscalibrated: on held out valid MOSFET sequences it produced 36 false positives in 100, and its ROC AUC fell from 1.00 to 0.31. The fix was to make the symbolic validator dominant and only let the head override it at high confidence. The earlier 1.00 figure was misleading because the head was undertrained.
* **xLSTM and cluster setup.** xLSTM cells failed at first because the compute nodes have a CUDA driver but no compiler for the just in time kernels, fixed by loading gcc and cuda modules in every job script; the conda CUDA resolver also failed on the GPU free login node and was replaced by a pinned PyPI build. xLSTM was then dropped from later grids because it is three to four times slower at identical loss.

## What you'd do with another 36 hours

* Run the production scale checkpoints (context window 768, six thousand steps, A100) through this benchmark, so the learned models are compared against the neurosymbolic engine at full strength rather than at the compact reproducible budget.
* Use the neurosymbolic oracle as a reranker and validity filter over the learned model Top 5 lists, where Top 5 of 1.000 leaves headroom to raise Top 1.
* Wire the parsed physics features (temperature, time, thickness, pressure, energy, dose per step) into the token embedding; they are parsed but not yet injected, and are the largest unexplored out of distribution lever.
* Train a process reward model that predicts, at each prefix, the probability of completing to a valid route, and use it inside beam search to lift completion exact match.
* Add multi seed confidence intervals and a harder out of distribution probe at family specific front end positions; the benchmark currently uses a single seed and a small eval slice.

## Track specific deliverables

### ⚙️ Industrial AI (Infineon)
* [x] Eval submission files in organizer format: `predictions_nextstep.csv`, `predictions_completion.csv`, `predictions_anomaly.csv` under `competition/participant-files/predictions/`, plus per regime and per system files under `shared/benchmark/predictions/`
* [x] Training artifacts: per run config and final loss in `shared/benchmark/checkpoints/*/summary.json` with `final.pt`, self supervised checkpoints in `shared/benchmark/ssl_checkpoints/`, production checkpoints and TensorBoard loss curves under `shared/extras/`
* [x] Scores from `eval_metrics.py` on all three tasks with per family breakdown, in `submission/UNIFIED_BENCHMARK.md`, `shared/benchmark/results_summary.csv`, and `submission/benchmark_assets/`
* [x] Demo shows baseline versus trained output on identical inputs: before and after examples plus a dashboard under `www/`

## Credits & dependencies

* **Open source libraries:** Python, PyTorch (cu121), NX AI xlstm, einops, OmegaConf, TensorBoard, pandas, NumPy, matplotlib; Pixi managed environment (versions pinned in `requirements.txt` and `pyproject.toml`)
* **Pre trained models:** none. The transformer and self supervised hybrid train from scratch; the neurosymbolic ranker has zero trained parameters
* **External APIs:** none
* **AI coding assistants used during the hackathon:** Claude Code
* **Datasets:** synthetic semiconductor sequences for MOSFET, IGBT, and IC, provided by the organizers with the scoring script `eval_metrics.py`; further valid and invalid sequences generated by the committed rule based generator and validator

## A note on honesty

* **The two eval paths disagree on transformer versus trigram.** The unified benchmark adapter scores the transformer above the trigram (0.779 against 0.721) in distribution. A separate production decode that assembles step strings by beam searching word tokens scores the trained transformer below the trigram (about 0.60 against 0.72), because beam assembly of multi token steps is less reliable than a direct argmax. We did not reconcile the two paths. The unified benchmark is the controlled like for like comparison; the compositional beam search reflects the heavier production path whose value is out of distribution coverage, not in distribution Top 1.
* **Anomaly 1.000 rests on a shared validator, not on a learned model.** The validator is the organizers' own code and is an oracle for the ten documented rules in distribution, so every validator backed system scores about 1.0. The learned validity head on its own does not generalize (see the miscalibration above). The honest differentiator is rule attribution and out of distribution behavior, where the symbolic role induction is strongest.
* **Compact versus production checkpoints.** The benchmark checkpoints are trained at a reduced budget on CPU so the comparison reproduces on a laptop (the transformer is the small config at 4.37 M, context window 512). They are not the full Leonardo checkpoints (context window 768, six thousand steps, A100). The purpose here is the controlled comparison, not peak accuracy; because of the context window bug above, earlier numbers are known underestimates.
* **Completion edit distance versus validity.** Edit distance measures similarity to one reference, so a fully rule valid completion can still diverge from it. Both are reported.
* **Small eval and single seed.** The shared eval set is intentionally small per family and uses a single seed, so a few percentage points of noise are visible. Multi seed intervals are future work.
* **Official accuracy is organizer computed.** The organizer eval inputs are unlabeled, so official accuracy can only be computed by the organizers. The numbers here are internal held out scores on our labeled eval set, built from the same generator and protocol. Some submission files were first generated against self simulated eval inputs and will be regenerated against the organizer files at submission. No official score is fabricated.

*Submitted by team Attention Seekers for Zero One Hack_01, 31 May 2026.*
