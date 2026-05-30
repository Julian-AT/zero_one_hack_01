# Implementation Guide — Neurosymbolic Process Engine (NSPE)

> **Audience:** an agent that will *execute* this guide end-to-end — build the
> code in `neurosymbolic-approach/`, run the CPU experiments locally, then use
> `LEONARDO_agent_guide.md` to connect to the A100s and launch the GPU tests.
>
> **Design thesis (read first).** Teammates are building a pure-transformer
> scaffold with **zero symbolic** components. This approach is the deliberate
> opposite: **symbolic-first.** The spine is a symbolic engine (grammar + 10
> rules + role ontology) that defines, at every prefix, *which next steps are
> even legal*. Learning is demoted to a subordinate job — **ranking preferences
> inside a symbolically-defined support.** Formally we reframe the whole track as:
>
> > *Learn `P(next | context)` restricted to the support `V(prefix)` that the
> > symbolic engine declares valid.* The symbolic engine owns **what is
> > possible**; the (small) learned model owns only **what is probable**.
>
> This is what makes the 4th-family (OOD) abstraction work: the support and the
> rules are **family-agnostic**, so they transfer with zero drop, while the
> learned ranker is kept small and role-factored so the part that *can't*
> transfer is minimized. The probe in `experiments/ood_symbolic_probe.py` already
> measured the key fact: symbolic anomaly detection on an unseen family is
> **0/900 false positives, 900/900 recall with role-induction**.

---

## 0. Data-flow at a glance

```
                official validator + grammar (REUSED, never modified)
                tracks/industrial-infineon/training_data/generate_sequences.py
                                       │ import by path
                                       ▼
  roles.py ──► rules.py ──► grammar.py(valid_next_set / prefix-automaton)
   (role        (rule          │
   ontology +   checker +      ├──────────────► anomaly.py ──► Task 3 (symbolic oracle + role-induction)
   induction)   role-aug)      │
                               ├──► ppm.py (symbolic role-factored ranker, CPU)   ┐
                               │                                                  ├─► decode.py ─► predict.py ─► 3 submission CSVs
                               └──► model.py (small constrained neural ranker, GPU)┘                 │
                                          ▲ losses.py (semantic/constraint loss)                     ▼
                                          │                                              eval.py ──► official eval_metrics.py
                                   data.py (LoFO splits, role-encode)                    (ID vs OOD drop = Task 4)
```

Every NSPE module lives in `neurosymbolic-approach/`. The **only** repo
dependency is *importing* (never editing) the organizers' ground-truth code:
`generate_sequences.py` (validator + grammar generator) and `eval_metrics.py`
(scorer). We import them by absolute path exactly like
`tracks/industrial-infineon/scripts/generate_ood_families.py` already does.

---

## 1. Reused repo assets (exact paths — import, don't copy logic)

| Asset | Path | Used for |
|---|---|---|
| Official validator + grammar | `tracks/industrial-infineon/training_data/generate_sequences.py` | `validate_sequence`, `generate_sequence`, `read_csv_sequences` |
| Official scorer | `tracks/industrial-infineon/scripts/eval_metrics.py` | scoring all 3 tasks (run via subprocess) |
| Training sequences (long fmt) | `tracks/industrial-infineon/training_data/{MOSFET,IGBT,IC}_variants.csv` | 1000 seqs/family |
| Fab parameters (optional) | `tracks/industrial-infineon/training_data/{MOSFET,IGBT,IC}_longdescription_parameters.csv` | optional role-induction fallback |
| Sample eval inputs | `tracks/industrial-infineon/scripts/eval_input_valid.csv` (600) · `eval_input_anomaly.csv` (987) | dry-run the submission pipeline |
| OOD family generator | `tracks/industrial-infineon/scripts/generate_ood_families.py` | simulate the unseen 4th family for OOD tests |
| Env (CUDA 12.1 torch) | `pixi.toml` (repo root) | GPU runs on Leonardo via `pixi run` |

**Self-containment rule:** all *new* code goes in `neurosymbolic-approach/`. The
symbolic core (`roles/rules/grammar/ppm/anomaly/decode`) imports **only stdlib +
the official validator** — it runs with no torch, no numpy. Only `model.py`,
`losses.py`, and the training experiments import `torch` (from the repo pixi env).

---

## 2. Folder layout to create

```
neurosymbolic-approach/
├── FINDINGS.md                 # (exists) the high-level plan
├── Implementation.md           # (this file)
├── README.md                   # NEW: 10-line quickstart + Leonardo recipe
├── nspe/
│   ├── __init__.py
│   ├── official.py             # locate + import generate_sequences.py & eval_metrics.py by path
│   ├── roles.py                # role ontology + step→role map + role INDUCTION (OOD lever)
│   ├── rules.py                # validate / validate_with_roles / would_violate / first_rule
│   ├── grammar.py              # valid_next_set(prefix,family) — the prefix automaton / support
│   ├── data.py                 # load variant CSVs, LoFO splits, role-encode, vocab
│   ├── simulate_eval.py        # build official-format GT (next-step + completion) from held-out seqs
│   ├── corrupt.py              # inject each of the 10 violation types (for anomaly OOD tests)
│   ├── ppm.py                  # symbolic role-factored variable-order Markov ranker (CPU)
│   ├── model.py                # small role-factored CONSTRAINED neural ranker (GPU, torch)
│   ├── losses.py               # CE + semantic/constraint loss
│   ├── decode.py               # constrained next-step top-k, beam completion, symbolic repair
│   ├── anomaly.py              # Task-3 symbolic oracle (+ optional learned residual)
│   ├── predict.py              # eval inputs → 3 submission CSVs (official formats)
│   └── eval.py                 # wrap eval_metrics.py + LoFO harness (ID vs OOD)
├── experiments/
│   ├── ood_symbolic_probe.py   # (exists) anomaly OOD sanity probe
│   ├── exp01_symbolic_anomaly.py   # Task 3: oracle ID + simulated-OOD, all 10 rules
│   ├── exp02_ppm_ranker.py         # Tasks 1&2: PPM (pure symbolic), ID + LoFO   [CPU]
│   ├── exp03_neural_ranker.py      # Tasks 1&2: constrained neural ranker, ID + LoFO   [GPU]
│   ├── exp04_constraint_loss.py    # ablation: +/- semantic loss, +/- mask        [GPU]
│   ├── exp05_scaling.py            # model-size & data-size sweep                  [GPU]
│   └── exp06_make_submission.py    # final all-3-family model → 3 CSVs            [CPU/GPU]
├── slurm/
│   ├── env_setup.sh            # login-node: pixi install + stage data to $SCRATCH
│   ├── test_debug.sbatch       # 1 GPU, debug QoS, 30 min — pipeline smoke test
│   ├── train_full.sbatch       # 1 GPU — full train+predict+score
│   └── grid_lofo.sbatch        # 4 GPUs — LoFO × configs in parallel
├── configs/
│   ├── small.yaml              # ~1M param ranker
│   ├── base.yaml               # ~4M param ranker
│   └── grid.yaml               # the LoFO × size matrix
└── outputs/                    # (gitignored) metrics json, submission csvs, checkpoints
```

Create skeleton:

```bash
cd /Users/kyrill/IdeaProjects/zero_one_hack_01/neurosymbolic-approach
mkdir -p nspe experiments slurm configs outputs
touch nspe/__init__.py
echo "outputs/" >> .gitignore
```

---

## 3. Environment

**Local (Mac, for building + CPU experiments):** symbolic core needs only Python
3.10+. For `model.py` use CPU torch (already in repo pixi `osx-arm64` target) or
just defer GPU experiments to Leonardo.

**Leonardo (GPU):** use the repo's `pixi.toml` (it pins `torch` cu121 for the
A100 driver). No custom CUDA kernels are used (plain PyTorch), so **no `module
load cuda` is required** — `pixi run` activates the bundled runtime. See §6.

Quick check after install:
```bash
pixi run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 4. Component specifications

Each subsection gives: **purpose · key API · algorithm · skeleton · Definition of
Done (DoD)**. Skeletons are near-complete for the crux modules; fill obvious gaps.

### 4.1 `nspe/official.py` — import the organizers' code by path

```python
"""Locate and import the organizers' ground-truth modules by absolute path."""
import importlib.util, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRACK = REPO / "tracks" / "industrial-infineon"
GEN_PATH   = TRACK / "training_data" / "generate_sequences.py"
EVAL_PATH  = TRACK / "scripts" / "eval_metrics.py"
OOD_PATH   = TRACK / "scripts" / "generate_ood_families.py"
DATA_DIR   = TRACK / "training_data"
EVAL_DIR   = TRACK / "scripts"

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

gs  = _load("generate_sequences", GEN_PATH)     # validator + grammar
ood = _load("generate_ood_families", OOD_PATH)  # unseen-family generators
validate_sequence = gs.validate_sequence
read_csv_sequences = gs.read_csv_sequences
FAMILY_FILES = {f: DATA_DIR / f"{f.upper()}_variants.csv" for f in ("mosfet","igbt","ic")}
```
**DoD:** `python -c "from nspe.official import validate_sequence; print('ok')"` runs
from the repo root (`PYTHONPATH=neurosymbolic-approach`).

### 4.2 `nspe/roles.py` — role ontology + induction (**the OOD lever**)

**Purpose:** map every step (known *or* unknown family-4 string) to one of ~16
functional roles. Known steps map via the validator's frozensets; unknown steps
map via surface patterns. This is what lets all downstream logic generalize.

```python
from nspe.official import gs

ROLES = ["LOGISTICS","INSPECT_MEASURE","CLEAN","SUBSTRATE_PREP","THERMAL_DEP",
         "LITHO","ETCH","STRIP","IMPLANT","ANNEAL_DIFFUSION","CMP","FILL",
         "PASSIVATION","BACKSIDE","TEST","OTHER"]

# Anchor known steps from the official frozensets (authoritative).
_KNOWN = {}
for s in gs.DEPOSITION_STEPS: _KNOWN[s] = "THERMAL_DEP"
for s in gs.ETCH_STEPS:       _KNOWN[s] = "ETCH"
for s in gs.IMPLANT_STEPS:    _KNOWN[s] = "IMPLANT"
for s in gs.CMP_STEPS:        _KNOWN[s] = "CMP"
for s in gs.CLEAN_STEPS:      _KNOWN.setdefault(s, "CLEAN")
for s in gs.ELECTRICAL_TEST_STEPS: _KNOWN[s] = "TEST"

def induce_role(step: str) -> str:
    s = step.upper().strip()
    if s in _KNOWN: return _KNOWN[s]
    # ---- pattern induction for UNKNOWN (family-4) strings ----
    if s.startswith("DEPOSIT ") or "OXIDATION" in s or s.endswith(" GROWTH") \
       or "EPITAXIAL DEPOSITION" in s:                      return "THERMAL_DEP"
    if " ETCH" in s or s.startswith("ETCH "):               return "ETCH"
    if s.startswith("STRIP "):                              return "STRIP"
    if s.startswith("IMPLANT "):                            return "IMPLANT"
    if s.startswith("CMP ") or "PLANAR" in s:               return "CMP"
    if s.startswith("FILL VIA"):                            return "FILL"
    if ("CLEAN" in s) or s.endswith(" RINSE") or s.startswith("DRY ") or s == "HF DIP": return "CLEAN"
    if any(k in s for k in ("SPIN COAT","SOFT BAKE","ALIGN MASK","EXPOSE LITHO",
            "DEVELOP","POST EXPOSE BAKE","HARD BAKE","MASK LEVEL","PATTERN INSPECTION","PAD WINDOW LITHO")): return "LITHO"
    if "PASSIVATION" in s or "PAD WINDOW" in s or s == "CURE PASSIVATION": return "PASSIVATION"
    if "ANNEAL" in s or "DIFFUSION" in s or "RAPID THERMAL" in s:          return "ANNEAL_DIFFUSION"
    if "BACKSIDE" in s or "GRIND" in s:                                    return "BACKSIDE"
    if s.endswith(" TEST") or "ANALYSIS" in s or "WAFER SORT" in s:        return "TEST"
    if s.startswith(("MEASURE","INSPECT","FINAL")) or s.endswith(("CHECK","INSPECTION")): return "INSPECT_MEASURE"
    if "EPITAX" in s or "SUBSTRATE" in s or "SURFACE PREP" in s:           return "SUBSTRATE_PREP"
    if any(k in s for k in ("RECEIVE WAFER","LOT ","SHIP LOT","RELEASE","PACKAGE")): return "LOGISTICS"
    return "OTHER"

ROLE_TO_IDX = {r:i for i,r in enumerate(ROLES)}
def role_idx(step): return ROLE_TO_IDX[induce_role(step)]
```
**DoD:** every step in all 3 variant CSVs maps to a non-`OTHER` role except a
small, inspected residual; print the `OTHER` set and confirm it's tiny/benign.

### 4.3 `nspe/rules.py` — checker, role-augmented checker, incremental check

```python
from nspe.official import gs, validate_sequence
from nspe.roles import induce_role
RULE_IDS = ["RULE_DEP_NO_CLEAN","RULE_METAL_ETCH_NO_LITHO","RULE_ETCH_NO_MASK",
  "RULE_LITHO_LEVEL_SKIP","RULE_IMPLANT_NO_MASK","RULE_CMP_NO_DEP",
  "RULE_PAD_OPEN_BEFORE_DEP","RULE_TEST_BEFORE_PASSIVATION","RULE_SHIP_BEFORE_TEST",
  "RULE_BACKSIDE_BEFORE_PASSIVATION"]

def validate(steps):                       # exact official semantics
    return validate_sequence(steps)

def validate_with_roles(steps):
    """Augment the official frozensets with role-induced members of THIS
    sequence's vocab, then run the exact official logic. (See ood_symbolic_probe.)"""
    vocab = set(steps)
    dep   = {s for s in vocab if induce_role(s)=="THERMAL_DEP"}
    clean = {s for s in vocab if induce_role(s)=="CLEAN"}
    etch  = {s for s in vocab if induce_role(s)=="ETCH"}
    imp   = {s for s in vocab if induce_role(s)=="IMPLANT"}
    orig = (gs.DEPOSITION_STEPS, gs.CLEAN_STEPS, gs.ETCH_STEPS, gs.IMPLANT_STEPS)
    try:
        gs.DEPOSITION_STEPS |= dep; gs.CLEAN_STEPS |= clean
        gs.ETCH_STEPS |= etch;      gs.IMPLANT_STEPS |= imp
        return validate_sequence(steps)
    finally:
        (gs.DEPOSITION_STEPS, gs.CLEAN_STEPS, gs.ETCH_STEPS, gs.IMPLANT_STEPS) = orig

def first_rule(steps, use_roles=True):
    v = (validate_with_roles if use_roles else validate)(steps)
    return v[0].rule if v else None

def would_violate(prefix, candidate, use_roles=False):
    """True iff appending `candidate` to an already-valid `prefix` introduces a
    violation AT the candidate's position. Sound because (a) windowed rules look
    back from their trigger and (b) global rules fire at the offending step — in
    both cases a candidate-induced violation lands at index len(prefix)."""
    seq = prefix + [candidate]
    v = (validate_with_roles if use_roles else validate)(seq)
    j = len(prefix)
    return any(viol.step_index == j for viol in v)
```
**DoD:** `would_violate(valid_prefix, gold_next)` is `False` for 100% of prefixes
of all 3000 training sequences (the gold next is always legal). Verify on a sample.

### 4.4 `nspe/grammar.py` — `valid_next_set` (the prefix automaton / support)

**Purpose:** the heart. Given a prefix (+ optional family), return the set of
candidate steps that keep the sequence valid. This is the *support* the ranker is
restricted to.

```python
from nspe.rules import would_violate

def valid_next_set(prefix, candidate_vocab, use_roles=False):
    """Rule-legal next steps. `candidate_vocab` = the step inventory to consider
    (e.g. all steps seen in training, optionally family-filtered)."""
    return {c for c in candidate_vocab if not would_violate(prefix, c, use_roles)}
```
Notes / refinements (document in code):
- `candidate_vocab` is built in `data.py` (all training steps; or family-restricted).
- The set is **necessary but permissive** (many steps pass at a given prefix) →
  ranking (PPM / neural) does the rest.
- **Guarantee:** the gold next step is always in `valid_next_set` (DoD 4.3), so
  masking never reduces Top-k recall — it can only help.
- Optional sharpening: intersect with steps whose induced role ∈ the PPM's
  top-`r` predicted roles (cuts the set ~5×; document the recall trade-off).

**DoD:** mean `|valid_next_set|` over training prefixes printed; gold-in-set rate
= 100%.

### 4.5 `nspe/data.py`, `nspe/simulate_eval.py`, `nspe/corrupt.py`

**`data.py`** — purpose: load + split + encode.
- `load_family(fam) -> list[list[str]]` via `read_csv_sequences`.
- `build_vocab(families) -> (steps_sorted, step_to_id)`; `candidate_vocab(families)`.
- `lofo_splits() -> [("mosfet",[igbt,ic]), ("igbt",[mosfet,ic]), ("ic",[mosfet,igbt])]`.
- `role_encode(seq)` → parallel role-id stream (uses `roles.role_idx`).

**`simulate_eval.py`** — purpose: build **official-format ground truth** from
held-out sequences so we can score Tasks 1&2 locally (the organizers' sample eval
inputs have no answers).
- `make_nextstep_gt(seqs) -> rows[EXAMPLE_ID,FAMILY,COMPLETION_FRACTION,PARTIAL_SEQUENCE,NEXT_STEP]`
  (sample several cut points per sequence; `NEXT_STEP` is the step after the cut).
- `make_completion_gt(seqs) -> rows[...,PARTIAL_SEQUENCE,FULL_SEQUENCE]` at fracs 0.6/0.8.
- Writers emit CSVs in the exact columns `eval_metrics.py` consumes (see its
  `_score_next_step`/`_score_completion`: needs `NEXT_STEP`, `FULL_SEQUENCE`).

**`corrupt.py`** — purpose: inject each of the 10 violations into a valid seq (for
anomaly ID + OOD recall, incl. novel-vocab). One function per rule returning a
corrupted copy + the rule id; verify with the official validator that exactly the
intended rule fires. Injection recipes:

| Rule | Minimal injection |
|---|---|
| DEP_NO_CLEAN | remove/blank all clean steps in the 12-window before a deposition (see probe) |
| ETCH_NO_MASK | delete the `DEVELOP PHOTORESIST` preceding a patterned etch |
| METAL_ETCH_NO_LITHO | delete `EXPOSE LITHO`+`DEVELOP` before a `METAL ETCH` |
| IMPLANT_NO_MASK | delete the oxide-etch/develop before an implant |
| CMP_NO_DEP | delete the deposition/fill in the 6-window before a CMP |
| LITHO_LEVEL_SKIP | renumber an `ALIGN MASK LEVEL n` → `n+2` |
| PAD_OPEN_BEFORE_DEP | move an `OPEN PAD WINDOW` before `DEPOSIT PASSIVATION` |
| TEST_BEFORE_PASSIVATION | move a `LEAKAGE TEST` before `CURE PASSIVATION` |
| SHIP_BEFORE_TEST | move `SHIP LOT` before `WAFER SORT TEST` |
| BACKSIDE_BEFORE_PASSIVATION | move `DEPOSIT BACKSIDE METAL` before `CURE PASSIVATION` |

**DoD:** for each rule, `validate(corrupt_x(seq))` contains that rule for ≥95% of
seqs (some seqs lack the trigger — skip those).

### 4.6 `nspe/ppm.py` — symbolic role-factored ranker (CPU, no torch)

**Purpose:** the *fully symbolic/statistical* preference model and the primary
OOD-robust ranker. Variable-order Markov (PPM/Katz backoff), **factored through
roles** so it transfers across families.

Algorithm:
- Train by counting on the training families:
  - Role model: `P(role_t | role_{t-k..t-1}, family)` with backoff over k=4→0.
  - Step model: `P(step_t | role_t, step_{t-j..t-1}, family)` backoff j=3→0,
    finally backing off to family-agnostic `P(step | role)`.
- `predict(prefix, family, candset) -> dict[step,prob]`:
  `score(c) = P(role(c)|role-context,fam) · P(c|role(c),step-context,fam)`,
  renormalized over `candset` (= `valid_next_set`).
- **OOD:** role context is shared across families → next-*role* prediction
  transfers even when step strings are novel; step backoff to `P(step|role)`
  keeps it sane.

**API:** `class PPM: fit(seqs_by_family); predict(prefix, family, candset)->{step:prob}`.

**DoD:** ID next-step Top-1 ≥ trigram baseline (~0.72) when unconstrained;
**Top-3/Top-5 strictly ≥ trigram** when restricted to `valid_next_set`.

### 4.7 `nspe/model.py` — small constrained neural ranker (GPU, torch)

**Purpose:** the learned ranker. Deliberately **small** (repo evidence: >5M
params is wasted on ID) and **role-factored + constraint-masked** — this is the
"neuro" the symbolic engine wraps, not a vanilla LM.

Architecture (≈1–4M params):
- Input per position: `step-embedding ⊕ role-embedding ⊕ family-embedding` (+ positional).
  Use the repo's `CompositionalTokenizer` *or* a folder-local step+role embedding.
- Backbone: 2–4 Transformer encoder layers (causal) **or** a 2-layer GRU. Keep it tiny.
- Two heads: **role head** (predict next role) and **step head** (predict next step).
- **Constraint masking at train & inference:** set logits of steps ∉
  `valid_next_set(prefix)` to `-inf` before softmax. (At train time, compute the
  mask from the gold prefix.)

```python
import torch, torch.nn as nn
class ConstrainedRanker(nn.Module):
    def __init__(self, vocab, n_roles, n_fam, d=128, layers=3, heads=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.role = nn.Embedding(n_roles, d)
        self.fam = nn.Embedding(n_fam, d); self.pos = nn.Embedding(512, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4*d, batch_first=True)
        self.body = nn.TransformerEncoder(enc, layers)
        self.step_head = nn.Linear(d, vocab); self.role_head = nn.Linear(d, n_roles)
    def forward(self, step_ids, role_ids, fam_id, attn_mask):
        T = step_ids.size(1); pos = torch.arange(T, device=step_ids.device)
        h = self.tok(step_ids)+self.role(role_ids)+self.fam(fam_id)[:,None,:]+self.pos(pos)[None]
        h = self.body(h, mask=attn_mask)          # causal mask
        return self.step_head(h), self.role_head(h)
def masked_logits(step_logits, valid_id_mask):     # valid_id_mask: [B,T,V] bool
    return step_logits.masked_fill(~valid_id_mask, float("-inf"))
```
**DoD:** trains to ID next-step Top-1 ≥ PPM on one family pair in <10 min on 1 A100;
masked decoding never emits a rule-violating step (assert via validator on outputs).

### 4.8 `nspe/losses.py` — CE + semantic/constraint loss

```python
import torch, torch.nn.functional as F
def step_ce(logits, target, pad_id):           # standard next-step CE
    return F.cross_entropy(logits.transpose(1,2), target, ignore_index=pad_id)
def role_ce(role_logits, role_target, pad):     # auxiliary next-role CE (transfer signal)
    return F.cross_entropy(role_logits.transpose(1,2), role_target, ignore_index=pad)
def semantic_loss(step_logits, valid_id_mask, pad_mask):
    """Penalize probability mass on RULE-INVALID continuations (Xu et al. 2018).
    L = -log( sum_{c in valid} softmax(logits)_c )  averaged over valid positions."""
    logp = F.log_softmax(step_logits, dim=-1)
    valid_logp = torch.logsumexp(logp.masked_fill(~valid_id_mask, float("-inf")), dim=-1)
    return -(valid_logp * pad_mask).sum() / pad_mask.sum().clamp(min=1)
# total = step_ce + 0.3*role_ce + lambda*semantic_loss   (lambda∈{0, 0.5}; ablate)
```
**DoD:** `exp04` shows semantic loss reduces invalid-mass and improves *OOD*
calibration/Top-1 vs `lambda=0` (mask still on); report both.

### 4.9 `nspe/decode.py` — constrained inference + repair

```python
from nspe.grammar import valid_next_set
def next_step_topk(prefix, family, ranker, candidate_vocab, k=5, use_roles=False):
    V = valid_next_set(prefix, candidate_vocab, use_roles)
    scores = ranker.predict(prefix, family, V or candidate_vocab)   # dict step->prob
    ranked = sorted(scores, key=scores.get, reverse=True)
    if len(ranked) < k:                       # pad from unconstrained ranker if needed
        extra = ranker.predict(prefix, family, candidate_vocab)
        ranked += [s for s in sorted(extra, key=extra.get, reverse=True) if s not in ranked]
    return ranked[:k]

def complete(prefix, family, ranker, candidate_vocab, max_len=200, use_roles=False):
    seq = list(prefix)
    while len(seq) < max_len:
        nxt = next_step_topk(seq, family, ranker, candidate_vocab, k=1, use_roles=use_roles)[0]
        seq.append(nxt)
        if nxt == "SHIP LOT": break
    return repair(seq)                        # safety net

def repair(steps, max_passes=3):
    """If (rarely) invalid, insert the missing prerequisite by role: a CLEAN
    before an uncleaned deposition, a DEVELOP before an unmasked etch, etc.
    Bounded greedy passes; return best-effort valid sequence."""
    # use rules.validate to find first violation, map rule->fix-insertion, repeat
    ...
```
- **Beam variant** (better Task-2 exact-match): keep top-`b` constrained prefixes,
  score by ranker log-prob, return best ending in `SHIP LOT`.
- **Guarantee:** constrained completion is rule-valid by construction → big NED /
  block-accuracy wins (mirrors the repo's grammar-decoder result: NED 0.999→0.126).

**DoD:** every completed sequence passes `validate_with_roles`; ID NED ≤ 0.15 at
frac 0.8 on at least one family.

### 4.10 `nspe/anomaly.py` — Task 3 (symbolic oracle + optional residual)

```python
from nspe.rules import validate_with_roles, first_rule
def classify(seq, ranker=None):
    v = validate_with_roles(seq)
    if v:
        return dict(is_valid=0, score=0.02, rule=v[0].rule)   # symbolic says INVALID
    # symbolic says valid; optional learned residual for novel-vocab anomalies:
    score = 0.98
    if ranker is not None:
        ppl = ranker.perplexity(seq)            # high ppl ⇒ structurally odd
        score = float(1.0 / (1.0 + max(0.0, (ppl - ranker.ppl_p95))))
    return dict(is_valid=1 if (ranker is None or score>0.5) else 0,
                score=score, rule=None if score>0.5 else "RULE_UNKNOWN")
```
- **Default submission = pure symbolic** (`ranker=None`): the probe shows 0 FP /
  100% recall on unseen families with role-induction. Rule attribution = `v[0].rule`.
- The residual path is for the *report's* OOD story (catching violations that use
  step strings even role-induction misses); keep it OFF for the headline submission
  unless it improves held-out F1.

**DoD:** on `eval_input_anomaly.csv` produces a valid Task-3 CSV; on
`corrupt.py`-injected sets, recall ≥0.98 per rule (ID) and ≥0.95 on simulated-OOD
families.

### 4.11 `nspe/predict.py` & `nspe/eval.py`

**`predict.py`** — read the two eval-input CSVs, emit the three submission CSVs in
the **exact** official formats (from `generation_rules.md` §5.3):
- Task 1: `EXAMPLE_ID,RANK_1,...,RANK_5` (from `decode.next_step_topk`).
- Task 2: `EXAMPLE_ID,PREDICTED_SEQUENCE` (pipe-joined steps **after** the cut only).
- Task 3: `EXAMPLE_ID,IS_VALID,SCORE,PREDICTED_RULE` (from `anomaly.classify`).

**`eval.py`** —
- `score(task, gt_csv, pred_csv)`: subprocess-call the official
  `eval_metrics.py` (`--task next-step|completion|anomaly`), capture the report.
- `lofo(make_ranker_fn)`: for each held-out family H — train ranker on the other
  two, build GT via `simulate_eval` on H (OOD) and on the train families (ID),
  predict, score, and record the **ID→OOD drop** per metric. This *is* Task 4.

**DoD:** `eval.score` reproduces numbers when fed the repo's own sample
predictions; `eval.lofo` prints an ID-vs-OOD table per metric.

---

## 5. Experiments (each = one runnable script)

| Exp | File | HW | What it proves | Key command |
|---|---|---|---|---|
| Probe | `ood_symbolic_probe.py` | CPU | symbolic anomaly OOD: 0 FP, 100% recall w/ roles | `python experiments/ood_symbolic_probe.py` |
| 01 | `exp01_symbolic_anomaly.py` | CPU | Task-3 oracle: ID + OOD, all 10 rules, attribution | `python experiments/exp01_symbolic_anomaly.py` |
| 02 | `exp02_ppm_ranker.py` | CPU | Tasks 1&2 PPM (pure symbolic): ID + LoFO | `python experiments/exp02_ppm_ranker.py` |
| 03 | `exp03_neural_ranker.py` | **GPU** | constrained neural ranker: ID + LoFO vs PPM | `pixi run python experiments/exp03_neural_ranker.py --config configs/base.yaml` |
| 04 | `exp04_constraint_loss.py` | **GPU** | ablation: mask on/off × semantic-loss on/off | `pixi run python experiments/exp04_constraint_loss.py` |
| 05 | `exp05_scaling.py` | **GPU** | size {1M,4M,16M} × data {2,4,8 families} curves | `pixi run python experiments/exp05_scaling.py` |
| 06 | `exp06_make_submission.py` | CPU/GPU | final all-3-family model → 3 submission CSVs | `pixi run python experiments/exp06_make_submission.py` |

Each experiment writes a JSON to `outputs/<exp>.json` and prints a table. Each
script's `__main__` must: set seed, build data, run, dump JSON, print summary.

**LoFO is the Task-4 measurement** — every prediction experiment (02–05) reports
ID vs held-out-family OOD, and the deliverable is the *drop*. Per FINDINGS, the
number to beat is the pure-neural baseline's drop (Top-1 0.72→~0.48).

---

## 6. Leonardo execution plan (connect → stage → launch → monitor)

> Follow `LEONARDO_agent_guide.md` for auth. Connection values come from the
> project `.env` (do **not** hardcode). Summary of the repo-specific path:

### 6.1 Connect (guide §2–§4)
```bash
# one-time: install step client + bootstrap CA + ssh-agent + cert (guide §2)
# then, using .env values:
ssh "${LEONARDO_SUPERCOMPUTER_SSH_USERNAME}@${LEONARDO_SUPERCOMPUTER_SSH_HOST}"
```

### 6.2 Stage code + data on a **login node** (has internet; guide §5–§7)
`slurm/env_setup.sh` (run on login node):
```bash
#!/bin/bash
set -e
# 1. clone repo into $HOME (code) using GITHUB_PERSONAL_TOKEN from .env
cd $HOME
git clone "https://${GITHUB_PERSONAL_TOKEN}@github.com/Julian-AT/zero_one_hack_01.git" || (cd zero_one_hack_01 && git pull)
cd zero_one_hack_01 && git checkout neurosymbolic-model
# 2. install pixi (login node has internet) and resolve the env
command -v pixi >/dev/null || curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
pixi install                     # resolves torch cu121 etc. (may take a few min)
pixi run python -c "import torch; print('torch', torch.__version__)"
# 3. data ships with the repo (training_data CSVs committed); point OUT to $SCRATCH
mkdir -p "$SCRATCH/nspe_outputs"
echo "Setup done. OUT=$SCRATCH/nspe_outputs"
```
> Heavy `pixi install`/downloads can exceed the login 10-min CPU cap — if so, run
> inside `srun --partition=lrd_all_serial --time 02:00:00 --mem=16G --pty bash` (guide §4).

### 6.3 Smoke test — 1 GPU, debug QoS (verify the pipeline before the grid)
`slurm/test_debug.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=nspe_debug
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out

cd $HOME/zero_one_hack_01
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PWD/neurosymbolic-approach:$PYTHONPATH"
export NSPE_OUT="$SCRATCH/nspe_outputs"

# CPU symbolic sanity (fast) + a tiny GPU train to prove the stack
pixi run python neurosymbolic-approach/experiments/ood_symbolic_probe.py
pixi run python neurosymbolic-approach/experiments/exp01_symbolic_anomaly.py
pixi run python neurosymbolic-approach/experiments/exp03_neural_ranker.py --config neurosymbolic-approach/configs/small.yaml --max-steps 200 --smoke
```
Submit & watch (guide §12):
```bash
sbatch neurosymbolic-approach/slurm/test_debug.sbatch
squeue --me
tail -c +0 -f nspe_debug-<jobid>.out
```

### 6.4 Full LoFO grid — 4 GPUs, one node (the real test battery)
`slurm/grid_lofo.sbatch` — runs the matrix in parallel, one run per GPU:
```bash
#!/bin/bash
#SBATCH --job-name=nspe_grid
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --mem=480GB
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out

cd $HOME/zero_one_hack_01
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PWD/neurosymbolic-approach:$PYTHONPATH"
export NSPE_OUT="$SCRATCH/nspe_outputs"
E=neurosymbolic-approach/experiments

# 4 independent runs, pinned to the 4 A100s (LoFO held-out family per GPU + a full-data run)
CUDA_VISIBLE_DEVICES=0 pixi run python $E/exp03_neural_ranker.py --holdout mosfet --config neurosymbolic-approach/configs/base.yaml &
CUDA_VISIBLE_DEVICES=1 pixi run python $E/exp03_neural_ranker.py --holdout igbt   --config neurosymbolic-approach/configs/base.yaml &
CUDA_VISIBLE_DEVICES=2 pixi run python $E/exp03_neural_ranker.py --holdout ic     --config neurosymbolic-approach/configs/base.yaml &
CUDA_VISIBLE_DEVICES=3 pixi run python $E/exp04_constraint_loss.py --config neurosymbolic-approach/configs/base.yaml &
wait
# aggregate + score everything with the official scorer
pixi run python $E/exp05_scaling.py --aggregate
echo "grid done -> $NSPE_OUT"
```
Submit:
```bash
sbatch neurosymbolic-approach/slurm/grid_lofo.sbatch
squeue --me ; tail -c +0 -f nspe_grid-<jobid>.out
```
> Each run writes to `$SCRATCH/nspe_outputs`. **Checkpoint there** (purged after 40
> days; copy final JSON/CSV back with `scp` from a login node — guide §6).

### 6.5 Produce the submission
```bash
sbatch neurosymbolic-approach/slurm/train_full.sbatch   # trains on ALL 3 families, runs exp06
# retrieve:
scp "${LEONARDO_SUPERCOMPUTER_SSH_USERNAME}@${LEONARDO_SUPERCOMPUTER_SSH_HOST}:\$SCRATCH/nspe_outputs/submission_task*.csv" ./outputs/
```

---

## 7. Acceptance criteria (what "done & working" means)

| Task | Metric | ID target | OOD (held-out family) target | Source of expectation |
|---|---|--:|--:|---|
| 3 anomaly | FP rate on valids | 0.00 | ~0.00 | probe [1] |
| 3 anomaly | recall (w/ role-induction) | ≥0.98 | ≥0.95 | probe [2]/[4] |
| 3 anomaly | rule-attribution acc | ≥0.95 | ≥0.90 | first-rule = injected rule |
| 1 next-step | Top-5 (constrained) | ≥0.99 | **beat 0.73** | repo trigram + grammar mask |
| 1 next-step | Top-1 | ~0.72 | **beat 0.48** | repo LoFO baseline (the prize) |
| 2 completion | NED @0.8 (constrained) | ≤0.15 | **well below pure-neural** | repo grammar-decoder 0.126 |
| 2 completion | every output rule-valid | 100% | 100% | constrained decode guarantee |

The headline result for the report is the **ID→OOD drop being far flatter than the
pure-neural scaffold's** — that is the whole point of going symbolic, and it is
what differentiates this from the teammates' approach.

---

## 8. Linear runbook (execute in this order)

1. **Build skeleton** (§2) and `nspe/official.py` (§4.1). DoD: import works.
2. **`roles.py` → `rules.py` → `grammar.py`** (§4.2–4.4). DoD: gold-in-set 100%,
   `OTHER` role set tiny.
3. **`data.py`, `simulate_eval.py`, `corrupt.py`** (§4.5). DoD: GT CSVs score
   cleanly through `eval_metrics.py`; each rule injects correctly.
4. **`anomaly.py` + `exp01`** (§4.10, §5). DoD: Task-3 CSV on `eval_input_anomaly.csv`;
   recall table per rule (ID + simulated OOD via `generate_ood_families`).
5. **`ppm.py` + `decode.py` + `exp02`** (§4.6, 4.9). DoD: PPM LoFO table; constrained
   completion all-valid.
6. **`predict.py` + `eval.py`** (§4.11). DoD: 3 submission CSVs from the sample eval
   inputs; LoFO harness prints ID-vs-OOD.
7. **`model.py` + `losses.py` + `exp03`/`exp04`** (§4.7–4.8). DoD: local CPU smoke
   (tiny), then Leonardo.
8. **Leonardo** (§6): `env_setup.sh` → `test_debug.sbatch` → `grid_lofo.sbatch` →
   `train_full.sbatch`. Retrieve outputs.
9. **Aggregate** into `outputs/` and update `FINDINGS.md` with the measured
   ID-vs-OOD tables.

---

## 9. Risks & fallbacks

- **Real eval inputs differ from samples.** Pipeline reads columns by name and is
  format-tolerant; if organizers add columns, `predict.py` ignores extras.
- **Family-4 renames many trigger steps** (the one real risk). Mitigation is built
  in: `validate_with_roles` + role-factored PPM/heads. `exp01`/`exp02` must report
  the *novel-vocab* variant (rename triggers in the OOD generator) to prove recovery.
- **Constrained set too permissive** (weak Top-1). Fallback: intersect with PPM
  top-`r` roles (§4.4) and/or rely on the neural ranker; never drop the gold (DoD 4.3).
- **GPU time tight.** The symbolic spine (Tasks 1-set, 2-constrained, 3-oracle)
  needs **no GPU** and already clears most targets; GPU only sharpens Top-1/Exact
  and produces the scaling story. Prioritize `exp01`/`exp02` if the node is busy.
- **`would_violate` cost.** O(n·|cand|) per position is fine (n≈150). If a sweep is
  slow, cache `validate(prefix)` and only recheck candidate-triggered rules.

---

*End of guide. The symbolic core is the contribution; the neural ranker is the
small, constrained, role-factored passenger. Keep it that way — that is the
differentiation and the OOD win.*
