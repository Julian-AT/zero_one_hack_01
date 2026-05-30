# Glossary — abbreviations and terms across the project

> One stop reference for every abbreviation used in `submission/`,
> `FINDINGS.md`, `REPORT.md`, and the LaTeX paper.
> Alphabetical. Cross-link from any other doc.

---

## Abbreviations

| Term | Full form | Meaning in this project |
|---|---|---|
| **A100** | NVIDIA A100 Tensor Core GPU | The accelerator we trained on. Leonardo provides 4 per node. 80 GB HBM2e memory. |
| **AdamW** | Adam with decoupled Weight decay | The optimizer; defaults `\beta_1=0.9`, `\beta_2=0.95`, weight_decay=0.1. |
| **AUC** | Area Under the (ROC) Curve | A ranking metric for binary classification (anomaly here); 0.5 random, 1.0 perfect. |
| **bf16** | Brain Floating Point, 16-bit | A numerical precision format used by A100s; faster than fp32 with negligible accuracy loss. |
| **BCE** | Binary Cross-Entropy | The loss used by the validity head: invalid (0) vs valid (1). |
| **Block-level Accuracy** | — | Task 2 metric: collapse each step to one of 10 coarse blocks (LITHO / ETCH / DEPOSITION / …) and compare position-wise. |
| **BOS** | Beginning Of Sequence | Special token marking the start of a sequence. |
| **CE** | Cross-Entropy | Standard classification loss; used by the rule-ID head (11-way). |
| **CLI** | Command Line Interface | A terminal program; `scripts/demo_compare.py` is our demo CLI. |
| **CMP** | Chemical Mechanical Planarization | A semiconductor step; planarizes the wafer surface after deposition. |
| **CSV** | Comma-Separated Values | Format of eval inputs and submission outputs. |
| **CUDA** | Compute Unified Device Architecture | NVIDIA's GPU programming platform. We use `cuda/12.6`. |
| **DIODE / SCHOTTKY / SIC_MOSFET** | (semiconductor product families) | The 3 synthetic OOD families generated for training augmentation. |
| **EM** | Exact Match | Task 2 metric: fraction of completions where prediction equals gold reference exactly. Low for us by design (many valid completions per prefix). |
| **EOS** | End Of Sequence | Special token. Validity and rule-ID heads pool the model's hidden state at this position. |
| **EuroHPC** | European High-Performance Computing | The joint undertaking that hosts the Leonardo cluster. |
| **F1** | F1 score | Harmonic mean of precision and recall: F1 = 2PR/(P+R). |
| **Frac=0.6 / 0.8** | Completion fraction | The eval truncates the gold sequence at 60% or 80% and asks the model to predict the remaining 40% or 20%. |
| **fdp** (or family_dropout) | Family-token Dropout | A regularizer that replaces the family token with `<FAMILY_UNK>` with probability p, forcing the model to handle unknown families. |
| **GLIBCXX** | GNU C++ Library | The C++ standard library; `libstdcxx-ng` is the conda-forge build of it. |
| **GPU** | Graphics Processing Unit | The accelerator hardware. We use 4 A100s in parallel. |
| **HBM** | High Bandwidth Memory | The memory technology on the A100; ~2 TB/s bandwidth. |
| **ID** | In-Distribution | Evaluation on the same product families the model was trained on. |
| **IGBT** | Insulated-Gate Bipolar Transistor | One of the 3 known product families in the benchmark. |
| **IC** | Integrated Circuit | One of the 3 known product families in the benchmark. |
| **JIT** | Just-In-Time (compilation) | The xLSTM kernel compiles JIT on first use; `module load gcc/12.2.0` was required. |
| **k-NN** | k-Nearest Neighbors | A non-parametric retrieval method we use as a Task 2 baseline. |
| **LM** | Language Model | The next-step prediction objective; also the loss component `\mathcal{L}_\text{LM}`. |
| **LoFO** | Leave-One-Family-Out | Training methodology: train on 2 of 3 families, evaluate on the third. Our Task-4 proxy. |
| **LR** | Learning Rate | Cosine schedule: warmup to peak 3e-4, then cosine decay to 10% of peak. |
| **mLSTM / sLSTM** | matrix LSTM / scalar LSTM | The two block types in the xLSTM architecture. |
| **MOSFET** | Metal-Oxide-Semiconductor Field-Effect Transistor | One of the 3 known product families in the benchmark. |
| **MP4** | MPEG-4 Part 14 | The required video container format for the submission. |
| **MRR** | Mean Reciprocal Rank | Task 1 metric: 1/(rank of correct answer), averaged across examples. 1.0 if always rank-1; 0 if never in Top-5. |
| **MT** (multi-task) | Multi-task | A training setup with multiple loss heads simultaneously (LM + validity + rule-ID). |
| **NED** | Normalized Edit Distance | Task 2 metric: token-level Levenshtein distance ÷ max(`|pred|`, `|ref|`). Range [0, 1]; lower is better. |
| **NS** | Neuro-Symbolic | Approach combining symbolic rules with learned neural ranking. |
| **OOD** | Out-Of-Distribution | Evaluation on a product family the model never saw. |
| **OOM** | Out-Of-Memory | A failure mode we did not hit (the 25M-param transformer fits in 80 GB easily). |
| **PD** | Pending (SLURM state) | Job is queued, not running yet. |
| **pp** | percentage points | Used when comparing percentages: 0.65 vs 0.50 is "+15 pp", not "+30 %". |
| **PRM** | Process Reward Model | Planned but unshipped: a head that scores a prefix's likelihood of completing to a valid sequence; for beam-search re-ranking. |
| **PyPI** | Python Package Index | The repository where Python packages live (`pip install` source). |
| **R** | Running (SLURM state) | Job is actually executing. |
| **RMSNorm** | Root Mean Square Normalization | A layer-normalization variant used in our transformer. |
| **ROC** | Receiver Operating Characteristic | The curve underlying AUC; true-positive rate vs false-positive rate. |
| **RoPE** | Rotary Position Embedding | Position encoding used in our transformer (rotation matrices on Q and K). |
| **rsync** | remote synchronize | The file-transfer tool we use to sync code/checkpoints between local and Leonardo. |
| **SCRATCH** (`$SCRATCH`) | scratch filesystem | High-speed temporary file system on Leonardo; deleted 40 days after last access. |
| **SDPA** | Scaled Dot-Product Attention | PyTorch's fast attention kernel; the standard `F.scaled_dot_product_attention`. |
| **SLURM** | Simple Linux Utility for Resource Management | The job scheduler on Leonardo. |
| **sps** | Steps per second | Training throughput metric. |
| **SSH** | Secure Shell | The protocol we use to reach Leonardo (`ssh -i ~/.ssh/leonardo_hack`). |
| **SSL** | Self-Supervised Learning | A training approach with no human labels; used in main branch's SSL Transformer. |
| **SwiGLU** | Swish-Gated Linear Unit | MLP activation in our transformer (gated SiLU = SiLU(W₁x) ⊙ W₂x). |
| **TB** | TensorBoard | Training-log visualization tool. 106 TB run dirs in our repo. |
| **Top-K** (Top-1 / Top-3 / Top-5) | Top-K accuracy | Task 1 metric: % where the gold next-step is in the model's top-K predicted candidates. |
| **TP / FP / TN / FN** | True/False Positive/Negative | The 4 cells of a binary-classification confusion matrix. |
| **UNK** | Unknown token | Special token for OOV (out-of-vocabulary) or for `<FAMILY_UNK>` during family-token dropout. |
| **W&B** | Weights & Biases | Experiment-tracking SaaS; we used it opportunistically (always-on TensorBoard, W&B if env var set). |
| **xLSTM** | Extended Long Short-Term Memory | A recurrent architecture (mLSTM + sLSTM); we evaluated it as an alternative to the transformer. |

---

## Project-specific terms (not abbreviations, but worth defining)

| Term | Definition |
|---|---|
| **Backbone** | The fixed-order block structure shared across all product families: PREFIX → CLEAN → PREP → CYCLES → ILD → VIA → METAL → PASSIVATION → BACKSIDE → INSPECTION → TEST → SUFFIX. |
| **Beam search** | A decoding algorithm that maintains the top-`beam_width` candidate sequences at each step. We use `beam_width=5`, `max_words=6` for compositional decoding. |
| **Block-position head** | Planned but unshipped 12-way auxiliary head predicting which backbone block the current position is in. Cheap structural prior. |
| **Block signature** | The reduced sequence obtained by mapping each step to its coarse block (one of 10 in the official taxonomy) and de-duplicating consecutive duplicates. Used by Block-level Accuracy. |
| **Cell** | One training run = one (arch, size, heads, hyperparams) configuration. We trained 104 cells. |
| **Compositional tokenization** | Each step string is split into word tokens with a `<STEP>` delimiter. Vocab ~70 words. Alternative: step-as-token (vocab ~200). |
| **Family token** | A special token (`<FAMILY_MOSFET>`, `<FAMILY_IGBT>`, `<FAMILY_IC>`, `<FAMILY_UNK>`) prepended to every sequence after BOS. |
| **Grammar mask** | At decode time, filter candidate next-steps to those that don't introduce a rule violation at the candidate's position. |
| **Length normalization** | Divide a beam's cumulative log-probability by the number of word tokens, so longer beams aren't unfairly penalized. |
| **Online generator** | Training data source: each batch calls `generate_sequence(family, rng)` for fresh sequences. Infinite stream, no fixed dataset. |
| **PPM** (variable-order Markov model) | A ranker used by the neurosymbolic team; not the same as PRM. |
| **Role-induction anchors** | A neurosymbolic-team trick: anchor each rule trigger to canonical steps so renamed-but-equivalent steps in OOD families are caught. We didn't ship this. |
| **Rule attribution** | Task 3 sub-metric: given an invalid sequence, identify *which* of the 10 rules was violated. |
| **Rule mask / grammar mask** | Same thing as grammar mask above. |
| **Synonym randomization** | Training-time data aug: per step, with probability p, swap to a random synonym from its equivalence class. The inverse of canonicalization. |
| **Trigram-with-backoff** | Katz-style fall-through: try trigram counts first; if none, fall back to bigram; finally unigram. |
| **Vocab restriction** | At decode time, drop candidate step strings that aren't in the real training vocabulary (filters compositional beam-search hallucinations). |
| **Validator** | The organizers' `validate_sequence` function from `tracks/industrial-infineon/training_data/generate_sequences.py`. The oracle for the 10 known rules. |
| **Validator-dominant ensemble** | Trust the validator first; only override its "valid" verdict if the learned validity head is very sure (`P_valid < 0.1`). |

---

## Submission file format reference

| File | Schema (CSV columns) | Row count |
|---|---|--:|
| `eval_input_valid.csv` (input to Tasks 1+2) | `EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE` | 600 |
| `eval_input_anomaly.csv` (input to Task 3) | `EXAMPLE_ID, FAMILY, SEQUENCE` | 987 |
| `nextstep.csv` (Task 1 submission) | `EXAMPLE_ID, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5` | 600 |
| `completion.csv` (Task 2 submission) | `EXAMPLE_ID, PREDICTED_SEQUENCE` (`|`-separated) | 600 |
| `anomaly.csv` (Task 3 submission) | `EXAMPLE_ID, IS_VALID, SCORE, PREDICTED_RULE` | 987 |

`IS_VALID`: 1 = valid, 0 = invalid (rule violation).
`SCORE`: probability that the sequence is valid in [0, 1].
`PREDICTED_RULE`: the rule ID (e.g. `RULE_DEP_NO_CLEAN`) if invalid, else empty.

---

## Spec references

- Track briefing (English): `tracks/industrial-infineon/Track_industrial_en.md`
- Process grammar + 10 rules: `tracks/industrial-infineon/training_data/generation_rules.md`
- Submission requirements (Tally form): `submission/SUBMISSION.md`
- Report template: `submission/REPORT_TEMPLATE.md`
- Official scoring script: `participant_files/eval_metrics.py`
