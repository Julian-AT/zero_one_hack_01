# Data Augmentation Plan for Semiconductor Process Sequences

## Goal

The goal of this augmentation pipeline is to generate a high-quality synthetic training dataset for semiconductor process-sequence learning.

The model should learn:

1. next-step prediction,
2. sequence completion,
3. validity classification,
4. anomaly detection,
5. rule-violation attribution,
6. generalization to underrepresented or unseen process variants.

The main assumption is that dataset quality is more important than model size. Therefore, the augmentation strategy should not simply generate more random sequences. Instead, it should generate sequences that improve coverage of the process grammar, rare branches, rule boundaries, and invalid near-miss cases.

---

## Current Baseline

The current repository already contains:

* a grammar-based sequence generator,
* family-specific process flows for MOSFET, IGBT, and IC,
* a validator for the 10 process-logic rules,
* CSV export for generated sequences,
* duplicate filtering for generated variants,
* a combinatoric estimate of possible structural variants.

This is a good starting point. However, the current generator mainly samples random valid variants. It does not yet guarantee systematic coverage of rare transitions, optional branches, process-rule boundaries, or invalid near-valid examples.

The next step is therefore to convert the generator from a random valid-sequence generator into a coverage-guided dataset generation system.

---

## High-Level Strategy

We will combine the following approaches:

| Approach                      |  Use? | Role                                           |
| ----------------------------- | ----: | ---------------------------------------------- |
| Rule-to-grammar generation    |   Yes | Main backbone for valid sequence generation    |
| Coverage-guided generation    |   Yes | Main mechanism for dataset quality control     |
| Controlled invalid generation |   Yes | Generate anomaly and rule-attribution examples |
| Grammar-aware fuzzing         |   Yes | Add local diversity around valid sequences     |
| SMT/SAT-based generation      | Later | Only for hard-to-reach coverage targets        |

The intended pipeline is:

```text
process grammar
→ valid candidate generation
→ independent validation
→ coverage scoring
→ dataset selection
→ controlled invalid mutation
→ train / validation / OOD split
→ model training
→ error analysis
→ targeted regeneration
```

---

## Dataset Design Principles

### 1. Quality over volume

Do not generate millions of random valid traces without measuring coverage. Large datasets can still be weak if they mostly repeat common process paths.

A sequence should be kept if it improves at least one of:

* rule coverage,
* step coverage,
* transition coverage,
* block coverage,
* product-family coverage,
* rare optional-branch coverage,
* valid-next-step ambiguity coverage,
* invalid-rule coverage,
* OOD-style generalization coverage.

---

### 2. Valid and invalid examples are both required

The dataset should contain both:

* valid process sequences,
* invalid near-valid process sequences.

Invalid examples should be realistic. They should usually differ from valid sequences by one controlled process-rule violation.

Good invalid examples:

```text
Valid sequence
→ remove required CLEAN AFTER ETCH
→ label as RULE_DEP_NO_CLEAN
```

```text
Valid sequence
→ move SHIP LOT before WAFER SORT TEST
→ label as RULE_SHIP_BEFORE_TEST
```

```text
Valid sequence
→ remove DEVELOP PHOTORESIST before OXIDE ETCH
→ label as RULE_ETCH_NO_MASK
```

Bad invalid examples:

```text
Randomly shuffled sequence
```

```text
Randomly inserted unknown steps
```

```text
Completely broken sequence with many simultaneous violations
```

The model should learn process logic, not merely detect nonsense.

---

### 3. Coverage should be explicit and measurable

Every generated sequence should produce metadata describing what it covers.

Minimum metadata:

```json
{
  "sequence_id": "mosfet_valid_000001",
  "family": "mosfet",
  "is_valid": true,
  "steps": ["RECEIVE WAFER LOT", "..."],
  "length": 126,
  "variation_flags": {
    "has_post_expose_bake": true,
    "has_hard_bake": false,
    "has_optional_measurements": true,
    "has_extra_clean": false
  },
  "coverage": {
    "steps": ["THERMAL OXIDATION", "..."],
    "transitions": [["THERMAL OXIDATION", "MEASURE OXIDE THICKNESS"]],
    "trigrams": [["SPIN COAT PHOTORESIST", "SOFT BAKE", "ALIGN MASK LEVEL 1"]],
    "rules_triggered": ["RULE_DEP_NO_CLEAN_BOUNDARY_SAFE"],
    "blocks": ["PREFIX", "PRE_PROCESS_CLEAN", "LITHO", "ETCH", "IMPLANT"]
  }
}
```

---

## Implementation Plan

## Stage 1 — Add a Coverage Tracker

Create:

```text
src/coverage_tracker.py
```

or:

```text
coverage_tracker.py
```

The coverage tracker should compute coverage features for every generated sequence.

### 1.1 Coverage features

Track the following:

#### Family coverage

```text
mosfet
igbt
ic
```

#### Length-bin coverage

Example bins:

```text
short: 0–100 steps
medium: 101–140 steps
long: 141+ steps
```

#### Step coverage

Track every unique process step that appears in the dataset.

Example:

```text
THERMAL OXIDATION
DEPOSIT POLYSILICON
VIA ETCH
CURE PASSIVATION
SHIP LOT
```

#### Adjacent transition coverage

Track all ordered step pairs:

```text
(A, B)
```

Example:

```text
("DEVELOP PHOTORESIST", "INSPECT PATTERN LEVEL 1")
("OXIDE ETCH", "STRIP PHOTORESIST")
("CURE PASSIVATION", "MEASURE PASSIVATION THICKNESS")
```

#### Ordered trigram coverage

Track all ordered 3-step windows:

```text
(A, B, C)
```

This helps the model learn local process syntax.

#### Lithography-level coverage

Track:

```text
ALIGN MASK LEVEL 1
ALIGN MASK LEVEL 2
ALIGN MASK LEVEL 3
...
```

Also track whether mask levels are sequential and non-decreasing.

#### Optional-step coverage

Track whether optional steps appear or do not appear:

```text
POST EXPOSE BAKE present / absent
HARD BAKE present / absent
PRE ANNEAL CHECK present / absent
DRY WAFER present / absent
optional measurements present / absent
PACKAGE PREPARATION present / absent
```

#### Block coverage

Map each step to a high-level block:

```text
PREFIX
INITIAL_MEASUREMENTS
PRE_PROCESS_CLEAN
FAMILY_SPECIFIC_PREP
FIRST_OXIDATION
LITHO
ETCH
IMPLANT
ANNEAL
ILD
VIA
METAL
PASSIVATION
BACKSIDE
FINAL_INSPECTION
TEST
SUFFIX
```

Track block transitions:

```text
LITHO → ETCH
ETCH → CLEAN
CLEAN → IMPLANT
IMPLANT → ANNEAL
PASSIVATION → PAD_OPENING
TEST → SHIP
```

#### Rule-boundary coverage

For each rule, track not only invalid cases, but also valid boundary cases.

Example:

```text
RULE_DEP_NO_CLEAN:
- deposition with clean 1 step before
- deposition with clean 5 steps before
- deposition with clean 11 steps before
- deposition with clean exactly 12 steps before
```

This is important because the validator uses finite windows such as "within previous 12 steps" or "within previous 15 steps".

---

### 1.2 Coverage report

The coverage tracker should output:

```text
outputs/coverage_report.json
outputs/coverage_report.md
outputs/undercovered_targets.csv
```

The markdown report should include:

```markdown
# Coverage Report

## Summary

| Metric | Value |
|---|---:|
| Number of sequences | 10000 |
| Number of step rows | 1,250,000 |
| Unique steps covered | 98 / 104 |
| Unique transitions covered | 560 |
| Unique trigrams covered | 1450 |
| Invalid rules covered | 10 / 10 |
| Product families covered | 3 / 3 |

## Undercovered Targets

| Target Type | Target | Count |
|---|---|---:|
| step | DEPOSIT BACKSIDE PROTECTION | 12 |
| transition | PASSIVATION ETCH → CLEAN PAD OPENING | 18 |
| rule boundary | RULE_DEP_NO_CLEAN clean-distance=12 | 0 |
```

---

## Stage 2 — Add Metadata-Aware Generation

The current generator returns only a list of steps. Extend it so it can optionally return metadata.

Current form:

```python
steps = generate_sequence(family, rng)
```

Target form:

```python
sample = generate_sequence_with_metadata(family, rng)
```

Example output:

```python
{
    "family": "mosfet",
    "steps": [...],
    "blocks": [...],
    "variation_flags": {
        "post_expose_bake_levels": [1, 3],
        "hard_bake_levels": [2],
        "pre_anneal_check_count": 3,
        "optional_measurement_count": 9,
        "strip_variant": "STRIP PHOTORESIST"
    }
}
```

This metadata will make it easier to balance the dataset and debug weak model behavior.

---

## Stage 3 — Coverage-Guided Valid Generation

Replace pure random generation with coverage-guided generation.

### 3.1 Basic algorithm

```text
initialize empty dataset D
initialize empty coverage table C

while not enough coverage:
    sample candidate sequence from grammar
    validate candidate sequence
    compute coverage features
    compute coverage gain
    if candidate improves coverage:
        keep candidate
        update coverage table
    else:
        discard candidate or keep with low probability
```

### 3.2 Candidate acceptance rule

A sequence should be accepted if it improves any of:

```text
new step
new transition
new trigram
new block transition
rare optional-step combination
rare family-specific branch
rare length bin
rare rule-boundary case
rare ambiguity state
```

Example scoring:

```python
coverage_gain = (
    5.0 * new_step_count
    + 3.0 * new_transition_count
    + 1.0 * new_trigram_count
    + 4.0 * new_block_transition_count
    + 8.0 * new_rule_boundary_count
    + 2.0 * rare_optional_combo_count
)
```

Keep samples with high `coverage_gain`.

---

## Stage 4 — Add Controlled Invalid Sequence Generation

Create:

```text
src/invalid_mutations.py
```

or:

```text
invalid_mutations.py
```

Each invalid mutation should start from a valid sequence and introduce exactly one intended rule violation.

### 4.1 Invalid mutation functions

Implement one mutation function per rule.

```python
def mutate_RULE_DEP_NO_CLEAN(seq):
    ...

def mutate_RULE_METAL_ETCH_NO_LITHO(seq):
    ...

def mutate_RULE_ETCH_NO_MASK(seq):
    ...

def mutate_RULE_LITHO_LEVEL_SKIP(seq):
    ...

def mutate_RULE_IMPLANT_NO_MASK(seq):
    ...

def mutate_RULE_CMP_NO_DEP(seq):
    ...

def mutate_RULE_PAD_OPEN_BEFORE_DEP(seq):
    ...

def mutate_RULE_TEST_BEFORE_PASSIVATION(seq):
    ...

def mutate_RULE_SHIP_BEFORE_TEST(seq):
    ...

def mutate_RULE_BACKSIDE_BEFORE_PASSIVATION(seq):
    ...
```

### 4.2 Required labels

Every invalid example should include:

```json
{
  "sequence_id": "invalid_RULE_ETCH_NO_MASK_000123",
  "family": "igbt",
  "is_valid": false,
  "violated_rule": "RULE_ETCH_NO_MASK",
  "violation_position": 57,
  "mutation_type": "remove_develop_before_etch",
  "steps": [...]
}
```

### 4.3 Validation check

After mutation, always run the independent validator.

The mutation is accepted only if:

```text
1. the validator detects invalidity,
2. the intended violated rule appears in the validator output,
3. the example does not accidentally create too many unrelated violations.
```

Prefer examples with exactly one violation.

---

## Stage 5 — Grammar-Aware Fuzzing

Use fuzzing only after the grammar and validator are stable.

### 5.1 Valid-preserving mutations

These mutations should preserve validity:

```text
replace synonym step
toggle optional measurement
toggle POST EXPOSE BAKE
toggle HARD BAKE
toggle PRE ANNEAL CHECK
replace STRIP PHOTORESIST with STRIP RESIST
replace RCA CLEAN 1 with WET CLEAN RCA1
replace CMP METAL with CMP VIA FILL where valid
```

These examples increase local diversity without breaking process logic.

### 5.2 Invalid near-miss mutations

These mutations intentionally break one rule:

```text
remove clean before deposition
remove develop before etch
remove oxide/window opener before implant
move CMP away from deposition/fill
move pad opening before passivation cure
move test before passivation cure
move ship before wafer sort test
move backside metal before passivation cure
skip lithography level
decrease lithography level
```

Fuzzing should never be raw random shuffling. It should be validator-backed and grammar-aware.

---

## Stage 6 — Add Targeted SMT/SAT Generation Later

SMT/SAT generation should not be the first priority.

Use it only when the coverage report shows targets that random grammar sampling and grammar-aware fuzzing cannot reach.

Example SMT/SAT use cases:

```text
generate a valid sequence with:
- product_family = IGBT
- litho level 5 present
- no POST EXPOSE BAKE in any litho block
- HARD BAKE present in exactly one litho block
- rare etch synonym selected
- valid backside metallization after passivation
```

or:

```text
generate a sequence that covers:
- RULE_DEP_NO_CLEAN boundary case
- clean exactly 12 steps before deposition
```

SMT/SAT should be a rare-case generator, not the main generator.

---

## Stage 7 — Dataset Composition

Recommended initial dataset composition:

| Split Component                   | Share | Purpose                                   |
| --------------------------------- | ----: | ----------------------------------------- |
| Grammar-valid canonical variants  |   35% | Learn normal process logic                |
| Coverage-guided valid variants    |   30% | Cover rare valid transitions and branches |
| Grammar-aware valid fuzz variants |   10% | Local diversity around valid traces       |
| Controlled invalid near-misses    |   20% | Anomaly detection and rule attribution    |
| OOD-style held-out variants       |    5% | Generalization testing                    |

Do not mix OOD-style variants into normal training unless explicitly intended. Keep them separate for evaluation.

---

## Stage 8 — Dataset Files

Recommended output files:

```text
data/generated/train_valid.csv
data/generated/train_invalid.csv
data/generated/train_mixed.csv
data/generated/val_valid.csv
data/generated/val_invalid.csv
data/generated/test_id.csv
data/generated/test_ood.csv
data/generated/coverage_report.md
data/generated/coverage_report.json
data/generated/dataset_manifest.json
```

### 8.1 Long-format CSV

Use this for sequence-model training:

```csv
SEQUENCE_ID,FAMILY,STEP_INDEX,STEP,IS_VALID,VIOLATED_RULE,SPLIT
mosfet_valid_000001,mosfet,0,RECEIVE WAFER LOT,1,,train
mosfet_valid_000001,mosfet,1,LOT IDENTIFICATION,1,,train
```

### 8.2 Prefix-prediction CSV

Use this for next-step prediction:

```csv
EXAMPLE_ID,SEQUENCE_ID,FAMILY,PREFIX,NEXT_STEP,LEGAL_NEXT_STEPS,SPLIT
ex_000001,mosfet_valid_000001,mosfet,"RECEIVE WAFER LOT|LOT IDENTIFICATION","INITIAL WAFER INSPECTION","INITIAL WAFER INSPECTION|PRE CLEAN INSPECTION",train
```

### 8.3 Sequence-completion CSV

Use this for completion:

```csv
EXAMPLE_ID,SEQUENCE_ID,FAMILY,PARTIAL_SEQUENCE,TARGET_SUFFIX,SPLIT
completion_000001,mosfet_valid_000001,mosfet,"RECEIVE WAFER LOT|...","DEPOSIT POLYSILICON|...",train
```

### 8.4 Anomaly CSV

Use this for validity classification:

```csv
EXAMPLE_ID,SEQUENCE_ID,FAMILY,SEQUENCE,IS_VALID,VIOLATED_RULE,SCORE_TARGET,SPLIT
anom_000001,invalid_RULE_ETCH_NO_MASK_0001,igbt,"RECEIVE WAFER LOT|...",0,RULE_ETCH_NO_MASK,0.0,train
```

---

## Stage 9 — Training Paradigm

Use supervised learning, not reinforcement learning.

The main task is:

```text
given current prefix/state → predict next step
```

Recommended model setup:

```text
Transformer backbone
├── next-step prediction head
├── next-block prediction head
├── validity classification head
└── rule-violation attribution head
```

### 9.1 Losses

Use a multi-task loss:

```text
L = L_next_step
  + λ1 * L_next_block
  + λ2 * L_validity
  + λ3 * L_rule_attribution
```

### 9.2 Important detail: multiple valid next steps

Some states may have several legal next steps.

Do not punish the model for predicting a different legal continuation.

For ambiguous states, use one of:

```text
multi-label target over legal next steps
```

or:

```text
soft target distribution over legal next steps
```

This is important because the process grammar may allow multiple valid continuations, even if a particular generated sequence contains only one of them.

---

## Stage 10 — Evaluation

Track the following metrics.

### 10.1 Next-step prediction

```text
Top-1 Accuracy
Top-3 Accuracy
Top-5 Accuracy
Mean Reciprocal Rank
```

### 10.2 Sequence completion

```text
Exact Match Rate
Normalized Edit Distance
Token Accuracy
Block-level Accuracy
```

### 10.3 Anomaly detection

```text
Binary Accuracy
Precision
Recall
F1
ROC-AUC
Rule Attribution Accuracy
```

### 10.4 Generalization

```text
ID → OOD performance drop
```

Report performance separately for:

```text
MOSFET
IGBT
IC
held-out process branch
held-out rare variant
held-out product family if applicable
```

---

## Stage 11 — Error-Driven Regeneration Loop

After training the first model:

```text
1. Evaluate on validation set.
2. Collect failure cases.
3. Group failures by rule, family, block, and transition.
4. Add failed regions to undercovered targets.
5. Generate more data around these regions.
6. Retrain or fine-tune.
```

Example failure-driven generation:

```text
Model often fails around:
VIA ETCH → STRIP PHOTORESIST → CLEAN AFTER VIA ETCH

Action:
Generate more via-block variants with different etch synonyms, strip synonyms, and clean variants.
```

---

## Stage 12 — Implementation Milestones

### Milestone 1 — Coverage report

Implement:

```text
coverage_tracker.py
```

Acceptance criteria:

```text
- Can read generated CSV files.
- Computes step, transition, trigram, family, length-bin, and optional-step coverage.
- Writes coverage_report.json and coverage_report.md.
```

### Milestone 2 — Coverage-guided valid generation

Implement:

```text
generate_coverage_guided.py
```

Acceptance criteria:

```text
- Generates valid sequences.
- Keeps sequences based on coverage gain.
- Produces more diverse data than random generation.
- Reports undercovered targets.
```

### Milestone 3 — Invalid mutation generator

Implement:

```text
invalid_mutations.py
```

Acceptance criteria:

```text
- Supports one mutation per forbidden rule.
- Produces near-valid invalid examples.
- Stores violated rule labels.
- Validator confirms intended violation.
```

### Milestone 4 — Dataset builder

Implement:

```text
build_augmented_dataset.py
```

Acceptance criteria:

```text
- Builds train/val/test splits.
- Exports valid, invalid, mixed, prefix-prediction, completion, and anomaly files.
- Writes dataset_manifest.json.
```

### Milestone 5 — Error-driven regeneration

Implement:

```text
regenerate_from_errors.py
```

Acceptance criteria:

```text
- Reads model error reports.
- Maps errors to undercovered targets.
- Generates targeted additional samples.
```

---

## Recommended First Implementation Step

Start with coverage tracking.

Do not implement SMT/SAT first.

The first concrete implementation target should be:

```text
coverage_tracker.py
```

because it tells us whether the current data is actually diverse enough.

Without coverage tracking, we are only guessing.

---

## Summary

This augmentation plan turns the existing generator into a dataset-quality engine.

The final pipeline should be:

```text
grammar-based valid generation
→ independent validation
→ coverage-guided sample selection
→ controlled invalid near-miss generation
→ grammar-aware fuzzing
→ dataset split construction
→ model training
→ error analysis
→ targeted regeneration
```

The key principle is:

```text
Generate data to cover process logic, not just to increase row count.
```
