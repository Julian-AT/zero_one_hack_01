he plan's strategic thesis is correct. Its execution plan is over-scoped and caries one load-bearing dependency that should be cut
entirely.

The core read on the problem is right, and I verified it against the data myself:

- ID next-step prediction is genuinely near-saturated by an n-gram. Not a leakage artifact — on a proper 80/20 held-out split the
  pooled trigram still scores Top-1 0.72 / Top-3 0.97 / Top-5 0.99. The plan is correct that beating this on ID is marginal.
- OD to a held-out family is the real contest. Leave-one-family-out trigram collapses to Top-1 ≈ 0.43–0.50, Top-5 ≈ 0.64–0.73. The
  plan's framing — memorizers die on family 4, the differentiator is transferable process logic — is the right framing.
- The symbolic validator is a perfect oracle on ID. validate_sequence() flags 0/3000 provided sequences as invalid. Task 3 on the
  known families is, as the plan says, essentially free.
- Synonyms are a real exact-match hazard. Both forms of every synonym pair appear at ~50/50 frequency in the data.

So the diagnosis is sound. The problem is the prescription. The plan tries to ship, in 36 hours one branch, two architectures ×
three sizes × three data volumes (20 cells), a contrastive encoder, multi-task heads, grammar-constrained decoding, k-NN retrieval,
synonym canonicalization, and a Streamlit dashboard — anchored on xLSTM, which my research flags as high-risk-to-build on Leonardo
and which offers zero theoretical advantage at 150-token sequences over a plain Transformer.

What I recommend instead: collapse to a focused, fully-reproducible spine (Transformer + n-gram baseline + a local eval harness + a
self-generated eval set), make the leave-one-family-out generalization curve the headline scientific result, keep the two genuinely
high-ROI hybrid pieces (validator-as-oracle for Task 3 ID, LM-perplexity for OD anomaly), and treat xLSTM, the contrastive encoder,
and the dashboard as optional add-ons owned by other branches — built only after the spine is green and submittable. This hits every
item on the track rubric (working artifact, honest reproducible eval, visible technical choices, real infra use, no LLM-wrapper)
with far less risk.

A material logistics finding the plan understates: eval*metrics.py and the eval_input*\*.csv files are not in the repo. The whole
downstream pipeline currently depends on artifacts that don't exist yet. The fix is cheap and de-risks everything: reimplement the
metrics from the documented spec and self-generate the eval set now, using the generator and validator you already have.

---

Current Plan Review

I'll separate what's right (keep it), what's wrong, and what's missing.

What the plan gets right

1. Hybrid stack over one big model. Correct for this data. Most of the points are cheap and interpretable.
2. "ID is saturated, OD is the game." Verified. This is the single most important strategic call and it's right.
3. Validator as a free Task-3-ID oracle. Verified (0/3000 false positives).
4. Synonym canonicalization for exact match. Verified necessary.
5. LoFO for config selection and family-token dropout. Both are legitimate OD-discipline levers and the right instinct.

What the plan gets wrong

1. xLSTM is the wrong bet, and it's load-bearing. It's one of the two headline architectures. Two independent problems:

- No benefit here. xLSTM's selling points are long-context and linear scaling. At ~150-step sequences and a ~200-token vocabulary,
  O(n²) attention is 22,500 ops — trivial. There is no published evidence xLSTM beats a vanilla Transformer at this scale; there's
  strong prior reason it won't.
- High build risk on Leonardo. The xlstm package compiles sLSTM CUDA kernels from source, has multiple open
  compilation/segfault/CUDA-version-mismatch issues unresolved as of 2026, no binary wheels, and a pure-PyTorch fallback documented
  only for Apple Metal — untested on Linux HPC. The plan's mitigation ("drop xLSTM, substitute Mamba") swaps one kernel-compilation
  gamble for another (mamba-ssm also needs --no-build-isolation and fails on non-PEP440 CUDA version strings common on clusters).

2. The schedule is a fantasy for one branch. 20 grid cells + contrastive encoder + multi-task heads + grammar decoder + retrieval +
   canonicalizer + dashboard, with two6-hour sleep blocks, in 33 working hours. This is 3–4 people's work. The plan defers ownership
   "to teammates" in §9 but then schedules all of it as one timeline. Classic hackathon over-scoping; the predictable failure mode is a
   sprawling half-working pile at H-0, which loses to a clean working spine on every rubric line.

3. The 100M model is pure waste of compute and clock. 100M params on ~400K total training tokens (3000 seqs × ~130 steps) is a 250:1
   parameter:token ratio — instant memorization. It's in the plan "only for the scaling-curve story," but the interesting scaling
   story is the opposite: show that a 2–8M model already saturates ID and that scaling up does not help (and may hurt) OOD. You can
   make that point with a much smaller "too big" anchor and save hours.

4. Compositional tokenization is over-claimed. The plan cals it "the central OOD lever — almost no other team will think of it" and
   states the word vocab shrinks to "~70 word-tokens." I measured it: the word vocab is 154 tokens, not ~70. Of the 76 family-exclusive
   step strings across the three LoFO folds, only 34 are fully reconstructable from words seen in the other two families; the rest are
   partial, and it makes sequences ~2.9× longer (hurting ID Top-1 as the plan admits). It's a legitimate ablation worth running, but
   it is a moderate lever, not a silver bullet. The bigger OOD lever is that 5 of the 10 rules are purely positional and transfer
   across families for free regardless of tokenization — the model architecture and LoFO training discipline matter more than the
   tokenizer.

5. The contrastive encoder is probably wasted effort. Research consensus: for control-flow/ordering anomalies, LM perplexity from
   the model you're already training covers the same OD gap for free, with per-step localization, and the marginal gain of a
   separately-trained contrastive model is small and not well-established. The plan allocates a 4-hour Tier-2 block plus multi-task
   heads to it. If you want a contrastive signal at all, the cheap version is freeze the LM encoder + train a small binary head — an
   afternoon, not a pillar.

6. Exact-match optimism on Task 2 is misplaced. I measured tail uniqueness: the last 20% of a sequence is nearly unique per family
   (IGBT 953/1000 distinct tails, IC 992/1000; top tail frequency 2–3). There are many valid completions for any prefix.
   Grammar-constrained decoding guarantees validity, not exact match — when many valid completions exist, constraining the beam won't
   make you match the specific gold sequence. Expect constrained decoding to move validity and block-accuracy a lot, and exact match
   only modestly. The plan leans on the grammar decoder for "big exact-match gains," which won't materialize.

7. The data-volume scaling axis is largely a non-lever for the real question. Generating 5k/10k more sequences from the same
   generator gives more samples from the same distribution. ID is already saturated; more same-distribution data does not add4th-family
   OD signal. The axis is fine for an ID scaling figure but should not be sold as helping the competition's actual metric. The OD
   lever is structural diversity + LoFO discipline, not raw volume.

What the plan is missing

- Local eval harness. eval_metrics.py is not in the repo. The plan assumes it. Reimplement it from §5.2.
- Self-generated eval set. You have the generator and the validator. Build your own eval_input_valid.csv / eval_input_anomaly.csv in
  the documented §5.1 format now — never be blocked on the organizers.
- The generalization curve as a first-class deliverable. Train on 1, then 2, then 3 families; plot OOD drop. This is the cleanest
  possible answer to the track's literal question ("logic vs. memorization") and the plan only implies it.
- Rule-violation sensitivity probes. Minimal pairs (valid vs. one injected violation), measure the model's NLL delta per rule.
  Directly measures which of the 10 rules the model internalized. Strong report material, cheap to build.

---

Repository / Track Findings

Repo state (branch abb, clean): Only EDA artifacts committed (extras/eda/). No src/, no configs, no models — everything in §5 of the
plan is unbuilt. This is genuinely a grenfield plan-lock moment.

Data (verified):

- \*\_variants.csv: 3 × 1000 sequences, long format SEQUENCE_ID,STEP. MOSFET 125±2.5 steps, IGBT 148±2.9, IC 115±2.5.
- 198 unique step strings total; 94 shared across all three; 20/27/29 family-exclusive (MOSFET/IGBT/IC).
- synthetic\_\*.csv: single canonical reference sequences (BOM + quoted, single STEP column — the loader must handle both formats; the
  provided read_csv_sequences already does).
- _\_Longdescr.csv / _\_longdescription_parameters.csv: per-step text descriptions + realistic fab parameters. Currently unused. ~1
  row per step, not per sequence. Note: the parameters file uses non-ASCII characters (e.g. FAB‑LEVEL with a non-breaking hyphen, µm)
  — needs UTF-8 care if ingested.

The two provided Python artifacts are the crown jewels:

- generate_sequences.py contains both generate_sequence() (the grammar) and validate_sequence() (the 10-rule oracle). You can
  generate unlimited valid data and unlimited labeled invalid data (inject a violation → validator confirms the label). This is a
  complete supervised anomaly dataset generator, for free.
- I confirmed the validator is exact on the provided data and that single-rule coruptions are cleanly caught and attributed
  (RULE_DEP_NO_CLEAN, RULE_SHIP_BEFORE_TEST, RULE_LITHO_LEVEL_SKIP all detected in my injection test).

The 10 rules split into two transfer classes (this maters for OD anomaly):

- 5 positional/global-ordering rules — LITHO_LEVEL_SKIP, SHIP_BEFORE_TEST, TEST_BEFORE_PASIVATION, BACKSIDE_BEFORE_PASSIVATION,
  PAD_OPEN_BEFORE_DEP. These depend on a handful of anchor step strings (SHIP LOT, WAFER SORT TEST, CURE PASSIVATION, ALIGN MASK LEVEL
  N) that are stable across families. They likely transfer to family 4.
- 5 local-window category rules — DEP_NO_CLEAN, ETCH_NO_MASK, METAL_ETCH_NO_LITHO, IMPLANT_NO_MASK, CMP_NO_DEP. These depend on a
  step string being a member of a category set (DEPOSITION/CLEAN/ETCH/IMPLANT/CMP). A new family-4 deposition step with a new string
  would be invisible to the literal validator. This is precisely the gap an ML model (LM perplexity) can fill.

Track rubric (what actually scores): working artifact that runs, honest reproducible eval with real numbers, visible technical
choices, genuine Leonardo use, no LLM wrappers. "Polish does not beat substance." This rewards a clean reproducible pipeline over a
sprawling demo — which argues directly against the current plan's breadth.

Track deliverables (hard requirements): nextstep.csv, completion.csv, anomaly.csv; checkpoints + loss curves; eval_metrics.py scores
with per-family breakdown; baseline-vs-trained side-by-side in the demo; REPORT.md, README.md, requirements.txt, public + MIT.

---

Research Findings

Sources are cited inline. One honesty caveat caried from the anomaly research agent: the SCAN/COGS/memorization papers were
live-verified, but a few applied citations (LogBERT, BINet) came from model knowledge and should be spot-checked before they go in
REPORT.md.

Architecture / HPC (xLSTM, Mamba)

- xLSTM on Leonardo: high risk, no upside. xlstm 2.0.5 ships a pure-Python wheel but sLSTM compiles a CUDA extension from source (C
  ≥ 8.0; A100 = 8.0, just passes), mLSTM depends on Triton kernels (mlstm_kernels). Open issues confirm build failures (#104), broken
  examples (#107), import segfaults (#100), CUDA-version mismatch on system tolkits (#115), and "binary distribution not set up" still
  open in2026 (#120). Pure-PyTorch fallback documented only for Apple Metal. Verdict: do not make it load-bearing. (Sources:
  github.com/NX-AI/xlstm + issues #100/#104/#107/#115/#120; arXiv:2405.04517; arXiv:2503.13427.)
- No theoretical benefit at this scale. xLSTM targets 2048+ token contexts and 50k+ vocab. At 150 tokens / 200 vocab its advantages
  are irrelevant.
- Fallback ladder (lowest→highest risk): decoder-only Transformer (pure PyTorch) → torch.nn.GRU/LSTM (zero deps) → mambapy (pip
  install mambapy, pure-PyTorch SSM, ~2× slower, no compilation) → mamba-ssm (needs --no-build-isolation, pre-test required) → avoid
  xlstm. A credible architecture-comparison story is Transformer vs GRU vs mambapy, three distinct families, zero kernel risk.
  (Sources: github.com/state-spaces/mamba + issue #947; github.com/alxndrTL/mamba.py; github.com/johnma2006/mamba-minimal.)

Anomaly detection & OOD

- LM perplexity/NLL is the best-validated approach for structured symbolic sequences and is a free byproduct of the LM you're
  already training. Generalizes to unseen violation types; gives per-step localization. Normalize NLL by length and fit a threshold on
  held-out valid data for calibration. (Closest analogues: DeepLog / LogBERT / LogAnomaly lines of work — spot-check URLs.)
- Contrastive encoder: not worth a dedicated model here. Marginal gain over perplexity for control-flow anomalies; if wanted, freze
  LM encoder + small binary head on rule-violating corruptions (cheap). (Tack et al., CSI, NeurIPS 2020, arXiv:2007.08176.)
- Autoencoder reconstruction: worst choice for ordering anomalies — decoders "correct" the violation during reconstruction
  (process-mining literature, Nole et al.).
- Compositional generalization is hard and tokenization alone doesn't solve it. SCAN (arXiv:1711.00350) and COGS (arXiv:2010.05465)
  show ID→OOD drops of 60–80 points for standard seq models. Word/subword tokenization helps unseen-token generalization by a moderate
  margin (Hofmann 2021 arXiv:2101.00403; Bostrom & Durett 2020 arXiv:2004.03720) — single-digit to low-double-digit points, not
  transformative. Matches my measurement (34/76 unseen steps fully word-coverable).
- "Logic vs memorization" has established probes: systematic splits + generalization curves (train on N families → OOD),
  rule-violation minimal-pair sensitivity (cf. BLiMP arXiv:1912.00582, Linzen 2016 arXiv:1611.01368), and prior-aware memorization
  metrics (don't naively count reproduced sequences). These map perfectly onto this track's scientific question.

Grammar-constrained decoding

- Off-the-shelf grammar libraries (outlines, xgrammar, llguidance, lm-format-enforcer, guidance) are CFG/regex-only. They can
  express the block structure but cannot express the 5 window/counting rules ("X within N steps of Y", sequential litho levels) —
  those are beyond context-free as the libraries support them. (Sources: xgrammar.mlc.ai docs; dottxt-ai outlines docs;
  github.com/microsoft/llguidance.)
- Right tool given you already have validate_sequence: a custom LogitsProcessor that incrementally masks any next-token that would
  make the partial sequence un-completable-validly, inside a beam search. With a 198-token vocab the per-step cost is aceptable for a
  hackathon; an incremental "valid-next-token" oracle optimizes it later. (HF LogitsProcessor docs.)
- Honest impact: constrained decoding moves validity → ~100% and block-accuracy up; exact match improves only modestly because
  multiple valid completions exist (Willard & Louf arXiv:2307.09702; Geng et al. arXiv:2305.13971). Report validity + block-accuracy
  as primary, exact match as secondary. This matches my tail-uniqueness measurement.

---

Alternative Solutions

All three assume a pure-PyTorch decoder-only Transformer as the workhorse and the provided generator/validator as data+oracle. They
differ in breadth.

Option A — Minimal / low-risk (the guaranteed floor)

Description: One Transformer (~8M), trained on all 3 families. Trigram baseline as reported floor. Task 3 = validator (ID) + LM
perplexity (OOD). Local eval_metrics.py reimplementation + self-generated eval set. One scaling axis (model size, 3 points).
Notebook demo, no dashboard. No xLSTM, no contrastive model, no grammar decoder.

- Fit to codebase: trivial; grenfield, minimal deps (torch, numpy, pandas).
- Required changes: data loader, tokenizer (step-level), tiny GPT, training loop, predict + baseline + metrics, three submission
  CSVs.
- Pros: ships with near-certainty; fully reproducible; clean; satisfies every hard deliverable.
- Cons: thin scaling story; no architecture comparison; less novelty.
- Risk: very low. Complexity: low. Testing: unit-test metrics against validator-labeled data; smoke-train; LoFO sanity.
- Chose when: time/people are tight, or as the must-finish baseline before anything else. Don't choose when: you have a reliable 3–4
  person team and want the scaling/architecture story the rubric rewards.

Option B — Balanced (recommended)

Description: Option A plus: a second zero-risk architecture (GRU via torch.nn.GRU) for a real architecture-comparison axis; 3 model
sizes (drop 100M, kep one deliberately-oversized anchor to show memorization onset); a clean step-token vs word-token ablation
evaluated specifically on LoFO OD (directly tests the plan's headline hypothesis); a custom-LogitsProcessor grammar-constrained beam
wrapping validate_sequence for Task 1/2; synonym canonicalization; and the LoFO generalization curve + rule-sensitivity probes as
the headline scientific result. Minimal Streamlit or notebook demo (baseline vs trained + anomaly attribution).

- Fit to codebase: moderate; everything is pure PyTorch + your existing validator.
- Required changes: A's spine + GRU model + tokenizer variants + grammar decoder + canonicalizer + LoFO harness + probe scripts +
  demo.
- Pros: hits every rubric item; directly answers the scientific question; defensible architecture + scaling + OOD story; no exotic
  kernels.
- Cons: still substantial for 36h — requires disciplined cuts and ideally 2–3 people. Risk: medium (mostly time). Complexity:
  medium. Testing: as A + grammar-decoder validity assertions + ablation reproducibility.
- Choose when: you have a functioning team and want a strong, honest submission. Don't choose when: you're effectively solo — then
  do A, add B pieces oportunistically.

Option C — Ambitious / ideal (≈ the current plan, de-risked)

Description: Option B plus: mambapy as a third architecture (pure-PyTorch SSM, not xLSTM); a contrastive encoder + multi-task heads;
the full 18–20 cell grid; a polished dashboard; mechanistic probing.

- Pros: maximal scaling + novelty if it all lands.
- Cons: high risk of an unfinished pile; the contrastive/multi-task work has low marginal ROI per the research; the grid's data axis
  is mostly uninformative for OD. Risk: high. Complexity: high.
- Choose when: 4 disciplined people, A+B already green by Saturday afternoon, and you specifically want a research-grade scaling
  artifact. Don't chose when: anything is behind — C is where teams drown.

---

Recommended Solution

Build Option A first as the non-negotiable floor; then layer Option B on top. Treat Option C items
(mambapy/contrastive/dashboard/full grid) as optional, owned by other branches, attempted only after A+B are submittable.

Rationale: the track rewards a working, reproducible, honestly-evaluated artifact over breadth, and explicitly penalizes
half-finished sprawl. A is guaranteed and satisfies every hard deliverable. B ads exactly the things the rubric rewards
(architecture comparison, scaling, OOD science) using only pure-PyTorch components and the validator you already have — no
kernel-compilation risk. The two highest-ROI hybrid ideas from the original plan (validator-as-oracle, LM-perplexity for OD) survive
intact; the two lowest-ROI/highest-risk ideas (xLSTM, dedicated contrastive model) are demoted.

Concrete deltas from the current plan:

1. Drop xLSTM. Architecture axis = Transformer vs GRU (+ mambapy only if there's slack).
2. Drop the dedicated contrastive encoder. Task-3 OD = LM perplexity; optional freeze-encoder + linear head if time.
3. Drop the 100M model. Sizes ≈ 2M / 8M / 30M, with one anchor sized to demonstrate memorization rather than chase it.
4. Promote the LoFO generalization curve + rule-sensitivity probes to the headline result.
5. Build the local eval harness + self-generated eval set on day one so the organizers' file timing can't block you.
6. Reframe metrics: validity + block-accuracy primary for Task 2; exact match secondary, reported with a "canonical exact match"
   (post-synonym-canonicalization) companion.

---

Improved Plan

Goal

Deliver a reproducible, end-to-end process-sequence benchmark that (a) trains from-scratch sequence models on the synthetic fab
data, (b) honestly measures baseline-vs-trained on all three submission tasks with per-family breakdown, and (c) answers the track's
core question — logic vs. memorization — via a leave-one-family-out generalization curve and per-rule sensitivity probes. Maximize
transferable process-logic, not ID leaderboard saturation.

Non-goals

- Beating 99% ID Top-5 (already saturated by an n-gram).
- A production-grade model or a large (≥100M) model.
- xLSTM, or any architecture requiring custom CUDA/Triton kernel compilation.
- A polished web app (a minimal demo suffices per the track).
- Using the long-description/parameter files (deferred; uncertain ROI).

Assumptions

- Organizers' eval*metrics.py and eval_input*\*.csv may arrive late or differ slightly → we reimplement metrics from §5.2 and
  self-generate eval inputs from §5.1 format. Adapter swap when real files land.
- Leonardo = A100s, possibly restricted outbound internet → all deps pure-PyTorch, pre-staged; W&B optional with TensorBoard
  fallback.
- Family 4 will add ~25 new step strings, most reusing sen words; ~5 rules transfer positionally, ~5 depend on category membership
  of (possibly new) strings.

Requirements

- Three submission files in exact §5.3 format.
- Local metrics reproducing §5.2 (Top-k/MRR; exact-match/NED/token-acc/block-acc; binary ac/P/R/F1/AUC/rule-attribution), with
  per-family +-cut breakdowns.
- Checkpoints + loss curves + a scaling/architecture table.
- A LoFO generalization curve (train on 1/2/3 families → OOD drop) and per-rule sensitivity scores.
- REPORT.md, README.md, requirements.txt, MIT, public, clean-checkout runable.

Architecture / design proposal

- Workhorse: decoder-only Transformer (RoPE, RMSNorm, no bias), max_len=256. Sizes ~2M / ~8M / ~30M.
- Second architecture: torch.nn.GRU LM at matched sizes (zero-dependency, real comparison point). mambapy only if slack.
- Tokenization: step-level (primary) and word-level (ablation), with a <STEP> delimiter so word-models still emit one step at a
  time. Ablation evaluated on LoFO specifically.
- Task 1/2 decoding: model logits → custom ValidatorLogitsProcessor (incremental validate_sequence masking) → beam search → synonym
  canonicalization → top-5 / full completion. k-NN-over-prefixes retrieval as fallback for unseen contexts.
- Task 3: validate_sequence (ID oracle + rule attribution) ∪ length-normalized LM perplexity (OOD detector for the 5 category rules
  the literal validator can miss on new vocab). Threshold fit on held-out valid data.
- OOD discipline: LoFO for config selection; family-token dropout p≈0.2; final model trains on all 3 families.

Module-level change map (pure-PyTorch, no exotic deps)

src/data/ load.py (handles long + BOM/quoted formats) · tokenizer.py (step|word)
generate.py (wraps provided generator) · corrupt.py (inject 10 rules → labeled anomalies)
canonicalize.py (synonym→canonical) · validator.py (re-export validate*sequence)
src/model/ transformer.py · rnn.py (GRU) · registry.py
src/train/ trainer.py · tracking.py (TB always; W&B opt)
src/eval/ metrics.py (REIMPLEMENTS eval_metrics.py from §5.2) · predict.py
grammar_decoder.py (ValidatorLogitsProcessor + beam) · perplexity.py · ensemble.py
src/baselines/ ngram.py (trigram backoff) · retrieval.py
src/probes/ lofo_curve.py · rule_sensitivity.py
configs/ arch/{transformer,gru}*{s,m,l}.yaml · token/{step,word}.yaml · train/default.yaml
scripts/ make_eval_set.py (self-generated §5.1) · run_baselines.sh · make_submission.sh
app/ demo.py (minimal Streamlit OR notebook)

Data model / schema

No DB. CSV in/out only. Internal representation: list[list[str]] per family. Eval inputs/outputs exactly per §5.1/§5.3
(pipe-separated steps; Task 2 predicts only post-cut steps).

Security / integrity

Low blast radius (local training, synthetic data). Guard: never commit checkpoints/.env (gitignore); UTF-8-safe parsing for the
parameter files; seed everything; the validator is the trusted boundary for label correctness.

Error-handling strategy

Validate at boundaries only: CSV parse (reuse the provided robust reader), eval-file schema adapter, decoder fallback to
unconstrained-then-filter if the constrained beam empties. Trust internal code; no defensive shims.

Testing strategy (see dedicated section).

Migration / rollout

Vertical slices merged to main only at agreed cut. ab owns the spine (data + Transformer + eval harness + baselines + submission).
Optional pieces (GRU/mambapy, contrastive head, dashboard) on sibling branches, merged only if green.

Observability

TensorBoard always on (loss, Top-k, val-NED, OOD drop per eval interval). W&B oportunistic. Every run writes a config snapshot + git
SHA + sed for reproducibility.

Risks → see dedicated section.

---

Implementation Phases

Dependencies in brackets. Phases 0–3 = Option A (the flor); 4–6 = Option B; 7 = optional Option C.

- Phase 0 — Spine & eval harness [none]. Repo scaffold, requirements.txt, data loader, step tokenizer, reimplement metrics.py from
  §5.2, make_eval_set.py (self-generate eval_input_valid.csv + eval_input_anomaly.csv via generator+validator), trigram baseline.
  Exit: baseline scored end-to-end against the self-made eval set; three CSVs emit in correct format. This is the de-risking phase and
  must finish first.
- Phase 1 — Transformer + smoke [0]. Minimal GPT, training loop, TB loging. Exit (decision gate): transformer-small reaches trigram
  Top-3 on held-out within a short smoke run; if not, debug pipeline before scaling.
- Phase 2 — Task 3 anomaly [0]. Wire validate_sequence (ID + attribution) + length-normalized LM perplexity (OOD); fit threshold;
  ensemble → anomaly.csv. Exit: ID anomaly ≈ oracle; perplexity AUC measured on LoFO-held family.
- Phase 3 — Task 1/2 decoding [1]. ValidatorLogitsProcessor + beam + synonym canonicalization + retrieval fallback → nextstep.csv,
  completion.csv. Exit: 100% emitted-sequence validity; block-accuracy reported. End of Option A — submittable.
- Phase 4 — Scaling + architecture [1]. Transformer ×3 sizes + GRU ×3 sizes; data axis 1k/5k/10k (framed as ID-saturation vs
  OOD-non-transfer). Exit: scaling/architecture table + curves.
- Phase 5 — Tokenization ablation + LoFO science [4]. step vs word at one size, evaluated on LoFO OOD; generalization curve (train
  1/2/3 families); per-rule sensitivity probes. Exit: the headline figures. Core scientific contribution.
- Phase 6 — Demo + report [2,3,5]. Baseline-vs-trained side-by-side + anomaly attribution; fill REPORT.md, README.md; record≤2-min
  video; slides.
- Phase 7 — Optional [6, only if green]. mambapy 3rd architecture; freze-encoder contrastive head; polished dashboard.

---

Testing & Validation Strategy

- Unit: metrics.py against hand-computed tiny cases and against the validator's labels (an all-valid set must score 100% binary ac;
  injected-violation sets must score by rule). Tokenizer round-trip (encode∘decode = identity at step boundaries). Synonym
  canonicalizer idempotence. Corruption generator: every injected violation is confirmed by validate_sequence.
- Integration: full pipeline on a 50-sequence subset emits all three CSVs in exact §5.3 schema; a schema-adapter test feds a mocked
  organizer file.
- Decoder invariant: assert 100% of constrained-beam outputs pass validate_sequence (validity is the guarantee; exact match is not).
- OD / regression: LoFO trigram numbers (Top-1 0.43–0.50, Top-5 0.64–0.73) are the regression anchors — the trained model must beat
  them on the held-out family or the OD claim fails. Track ID Top-5≥ 0.99 as a non-regression flor.
- Edge cases: sequences at max_len; prefixes whose only valid next-token is masked (fallback path); unseen step strings at inference
  (word-tokenizer path); empty/degenerate eval rows.
- Reproducibility check: clean-checkout run on a second machine reproduces a reported number within sed tolerance (the rubric's
  explicit ask).
- Manual: eyeball baseline-vs-trained on the track's worked example (RECEIVE WAFER LOT → LOT IDENTIFICATION → INITIAL WAFER
  INSPECTION → ? should yield a measurement step, not ETCH).
- Perf/security: profile the constrained decoder (198 validator calls/step/beam) on the longest IGBT sequences; confirm aceptable;
  no secrets/checkpoints committed.

---

Risks, Tradeoffs, and Mitigations

1. xLSTM/Mamba kernels fail on Leonardo. Mitigation: not on the critical path; Transformer+GRU are pure-PyTorch; mambapy
   (pure-torch) only if slack. Eliminated as a blocker by design.
2. Organizer eval files arrive late / differ. Mitigation: self-generated eval set + local metrics from spec; thin adapter when real
   files land. (Phase 0.)
3. Over-scope → unfinished pile. Mitigation: Option A is the hard floor and ships first; B/C are additive and abortable. Phase exit
   gates enforce this.
4. Exact match stays low on Task 2. Tradeoff, not a bug: many valid completions exist (tails near-unique). Mitigation: report
   validity + block-accuracy as primary, exact match + canonical-exact-match as secondary; explain why in the report (this is itself a
   finding).
5. Compositional tokenization doesn't help OOD. Mitigation: it's an ablation — whichever wins LoFO is what we ship; the negative
   result is still a reportable finding.
6. Perplexity anomaly detector is porly calibrated. Mitigation: length-normalize, fit threshold on held-out valid; fall back to
   validator-only + optional freeze-encoder head.
7. Constrained decoder too slow. Mitigation: precompute valid-next-token sets per state; cap beam width; fallback to
   unconstrained-then-filter.
8. Branch cordination (one branch can't own all tiers). Mitigation: abb owns the spine; optional components on sibling branches,
   merged only when green.

---

Open Questions

Only the ones that change the plan; each with the default I'd proceed on.

1. Team size actually committed to this track? Default if unknown: assume effectively 1–2 → execute Option A fully, add B
   oportunistically; do not start C.
2. Is Leonardo outbound internet open, and can deps be pre-staged? Default: assume restricted → pure-PyTorch only, pre-stage wheels,
   TensorBoard (W&B opt-in only if WANDB_API_KEY present and import succeeds).
3. Do the long-description/parameter files cary signal worth the time? Default: skip for the core submission; revisit only in
   Phase 7. (Uncertain ROI; they're per-step metadata, not per-sequence.)
4. Will the 4th family reuse the same 10-rule set? Default: assume yes for the 5 positional rules (validator transfers) and treat
   the 5 category rules as the perplexity detector's job; report bothID and OOD anomaly separately so the assumption is visible.

---

First Actionable Step

Build Phase 0's de-risking core: a local eval harness + a self-generated eval set, anchored on the already-validated trigram
baseline. Concretely, the smallest safe slice is:

1. src/data/load.py — load the three \*\_variants.csv (the loader logic in the provided read_csv_sequences already handles the
   formats).
2. scripts/make_eval_set.py — use the provided generate_sequence + validate_sequence to emit eval_input_valid.csv (100/family ×
   {0.6, 0.8} cuts) and eval_input_anomaly.csv (valid + injected-violation mix), in exact §5.1 format.
3. src/eval/metrics.py — reimplement the §5.2 metrics; unit-test against the validator's labels.
4. Wire the existing trigram backoff as the reference baseline and score it end-to-end.

This unblocks every downstream task, depends on nothing from the organizers, and produces a green, reproducible pipeline you can
demo on its own. It's also the cheapest possible insurance against the biggest logistics risk in the current plan.

I have not written any code or modified plan.md — this is analysis only, per your instruction.

---

Optional: Proposed plan.md Replacement

Here is a clean, tightened rewrite. I have not written it to disk. Say the word and I'll replace plan.md with it.

# Plan — Zero One Hack_01 / Industrial AI (Infineon)

> Branch **`abb`** owns the spine (data + Transformer + eval harness + baselines + submission).
> Optional components (2nd/3rd architecture, contrastive head, dashboard) live on sibling
> branches and merge to `main` only when green. Status: **planning** — only EDA committed.

## TL;DR

ID next-step prediction is already saturated by a trigram (held-out Top-5 ≈ 0.99 — verified).
The real contest is **OD generalization to a hidden 4th family**, where a trigram collapses to
Top-5 ≈ 0.64–0.73 (verified, leave-one-family-out). So we don't chase the ID leaderboard; we
build a **reproducible benchmark** whose headline result is a **leave-one-family-out
generalization curve** that distinguishes learned process-logic from memorization. The model is a
small pure-PyTorch Transformer; the hybrid wins (validator-as-oracle, LM-perplexity anomaly,
grammar-constrained decoding) are cheap and interpretable. **No xLSTM, no exotic CUDA kernels.**

## Verified EDA facts

- 198 step strings (94 shared); 20/27/29 family-exclusive (MOSFET/IGBT/IC).
- Trigram next-step (proper held-out): Top-1 0.72 / Top-3 0.97 / Top-5 0.99. **ID is saturated.**
- Trigram LoFO (train 2, test 3rd): Top-1 0.43–0.50 / Top-5 0.64–0.73. **OD is the game.**
- `validate_sequence` flags 0/3000 provided sequences → perfect Task-3 ID oracle.
- Synonym pairs appear ~50/50 → canonicalization need for exact match.
- Task-2 tails are near-unique per family (IGBT 953/1000, IC 992/1000) → exact match will be
  low regardless; report **validity + block-accuracy** as primary, exact match as secondary.
- Word-vocab = 154 tokens (not ~70); 34/76 unseen steps fully word-reconstructable →
  compositional tokenization is a _moderate_ OD ablation, not a silver bullet.

## Approach (de-risked hybrid)

| Component                                         | Task         | Notes                                                                             |
| ------------------------------------------------- | ------------ | --------------------------------------------------------------------------------- |
| Trigram backoff                                   | 1/2 baseline | Reported floor (ID + LoFO).                                                       |
| `validate_sequence`                               | 3 ID         | Oracle + rule attribution. Free.                                                  |
| LM perplexity (length-norm)                       | 3 OOD        | Catches category-rule violations on new vocab the literal validator misses.       |
| Custom `ValidatorLogitsProcessor` + beam          | 1/2          | Masks invalid next-tokens (5 rules are window/counting — beyond off-the-shelf CFG |
| decoders). Guarantees validity.                   |
| Synonym canonicalizer                             | 2            | Raw + canonical exact match.                                                      |
| Decoder-only Transformer (~2/8/30M, pure PyTorch) | 1/2/3        | Workhorse. RoPE, RMSNorm, max_len 256.                                            |
| GRU (`torch.nn.GRU`, matched sizes)               | scaling      | Zero-dep architecture comparison.                                                 |

Dropped vs prior plan: **xLSTM** (high build-risk on Leonardo, no benefit at 150 tokens),
**100M model** (250:1 param:token → pure memorization), **dedicated contrastive encoder**
(LM perplexity covers the same OD gap for free; use a freeze-encoder head only if time allows).

## Scaling / science story

- Transformer vs GRU × {2M, 8M, 30M} × {1k, 5k, 10k}. Data axis framed honestly: more
  same-distribution data saturates ID but does **not** add 4th-family OOD signal.
- **Headline:** LoFO generalization curve (train on 1→2→3 families, measure OOD drop) +
  per-rule sensitivity probes (minimal pairs, NLL delta per rule) + step-vs-word tokenization
  ablation evaluated on LoFO.

## OD discipline

LoFO for config selection; family-token dropout p≈0.2; final model trains on all 3 families.

## Submission artifacts

`extras/results/{nextstep,completion,anomaly}.csv` (exact §5.3 schema), checkpoints + TB loss
curves, local-metrics scoreboard (per-family + per-cut), ≤2-min demo (baseline vs trained +
anomaly attribution), `REPORT.md`, `README.md`, `requirements.txt` (torch, numpy, pandas,
pyyaml, tensorboard, streamlit, einops; wandb optional), MIT, public.

## Repo layout

```
src/{data,model,train,eval,baselines,probes}/  configs/{arch,token,train}/
scripts/{make_eval_set,run_baselines,make_submission}.{py,sh}  app/demo.py  extras/{eda,checkpoints,logs,results}/
```

## Phases (0–3 = guaranteed floor; 4–6 = full; 7 = optional)

0. **Spine & eval harness** — loader, reimplement `eval_metrics.py` from §5.2,
   `make_eval_set.py` (self-generated §5.1 inputs), trigram baseline. _De-risk first._
1. Transformer + smoke (gate: beat trigram Top-3 on held-out).
2. Task-3 anomaly: validator ∪ perplexity ensemble.
3. Task-1/2: constrained beam + canonicalization + retrieval fallback. **← submittable.**
4. Scaling + GRU architecture axis.
5. Tokenization ablation + LoFO curve + rule probes. **← headline science.**
6. Demo + REPORT + slides + video.
7. Optional: mambapy 3rd arch, freeze-encoder contrastive head, dashboard.

## Key risks → mitigations

- Exotic kernels fail → not on critical path (pure-PyTorch only).
- Organizer eval files late/different → self-generated eval set + local metrics + adapter.
- Over-scope → Option-A floor ships first; rest is additive and abortable behind phase gates.
- Low exact match → expected (many valid completions); report validity/block-accuracy primary.

## Open decisions for the team

Team size (drives A-only vs full); Leonardo internet/dep-staging; whether to ingest the
long-description/parameter files (default: skip for core); whether the 4th family reuses the
10-rule set (default: assume positional rules transfer, perplexity covers category rules).
