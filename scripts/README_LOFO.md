# LoFO ablation grid — how to run

The hidden 4th product family (Task 4) is the actual competition. Leave-one-
family-out (LoFO) on the three known families is the only honest proxy we
have for it. This grid measures the ID→OOD drop across the recipe matrix,
then a 16-cell follow-up retrains the same recipes on all three families
for the final submission checkpoints.

## What the grid covers (64 cells)

Phase 1 — LoFO (48 cells):
- **arch**: `transformer`, `xlstm`
- **size**: `small` (~5M), `medium` (~25M)
- **heads**: `lm_only`, `multitask` (validity + rule-ID on `<EOS>`)
- **family_dropout**: `0.0`, `0.2`
- **fold**: `held_mosfet`, `held_igbt`, `held_ic`
- tokenization fixed to `compositional` (the only mode with an OOD story for
  unseen step strings in family 4)

Phase 1.5 — Final all-3 (16 cells): same `arch × size × heads × fdp` cross,
trained on all three families, no held-out.

Inspect locally before launching:
```bash
.venv/bin/python -m src.experiments.lofo_grid --list \
    | column -t -s$'\t' | less
.venv/bin/python -m src.experiments.lofo_grid --count   # → 64
```

## Launching on Leonardo

```bash
# 1. Sync repo to $SCRATCH (as before)
bash scripts/leonardo/deploy.sh

# 2. Launch training array (4 A100s saturated; ~3-4h wall total)
ssh leonardo
sbatch scripts/slurm/lofo_grid.sbatch
# → submits 64 array tasks, capped at 4 concurrent

# 3. Eval array — depends on training array completing
sbatch --dependency=afterok:<train_jobid> scripts/slurm/lofo_eval_grid.sbatch

# 4. Pull checkpoint summaries + eval metrics back, aggregate
rsync -av leonardo:$SCRATCH/zero_one_hack_01/extras/{checkpoints,results} extras/
.venv/bin/python scripts/aggregate_lofo.py
# → extras/results/lofo_ablation.{csv,md}
```

## Reading the ablation table

`extras/results/lofo_ablation.md` ranks cells by Top-1 on the held-out
family. Key columns:

- `Top1_held` — Top-1 next-step on the held-out family. The Task-4 proxy.
- `Top1_id_avg` — Top-1 averaged across the two training families.
- `top1_drop = Top1_id_avg − Top1_held` — smaller = less OOD penalty.
- `anom_AUC_held` — ROC-AUC of the anomaly ensemble on the held-out family.
  The brief's `eval_metrics.py` scores AUC; we shouldn't ship a model that
  flattens it.

The second table averages across the 3 LoFO folds per recipe — that's the
recipe-selection signal for which all-3 cell to ship for the final submission.

## Re-running a subset

```bash
# Re-run just cells 12, 13, 14 (e.g. one recipe's three LoFO folds)
sbatch --array=12-14 scripts/slurm/lofo_grid.sbatch
```

## Phase 2 (not yet wired)

These need code changes in `src/model/` and a follow-up grid:

- Physics-feature injection (`Linear(10, d_model) + missingness mask` added
  to token embedding, lookup from `data/processed/physics_features.json`)
- Synonym-randomization augmentation in `OnlineGeneratorIterableDataset`
- Block-position auxiliary head (12-way classification on `<EOS>` over the
  backbone blocks from `generation_rules.md §2`)
- Hybrid-block sequence augmentation (Frankenstein generation)
