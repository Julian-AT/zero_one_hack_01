# NSPE — Neurosymbolic Process Engine (Infineon process-logic track)

**Symbolic-first.** A symbolic engine (grammar + 10 rules + 16-role ontology) defines, at every
prefix, *which next steps are even legal*; a small, role-factored, constraint-masked ranker only
decides *what is probable* inside that support. This is the deliberate opposite of the teammates'
zero-symbolic, pure-transformer scaffold: the symbolic spine owns correctness (and transfers to the
unseen 4th family with zero drop), so the learned part stays small (<5M params) — which is exactly
where the OOD win comes from. Full design: `Implementation.md`. Spec contract: `FINDINGS.md`.

## Quickstart (local CPU — no GPU, no torch needed for the symbolic spine)
```bash
# from the repo root
PYTHONPATH=models/neurosymbolic python3 models/neurosymbolic/experiments/exp01_symbolic_anomaly.py   # Task 3 oracle
PYTHONPATH=models/neurosymbolic python3 models/neurosymbolic/experiments/exp02_ppm_ranker.py          # Tasks 1&2, PPM
```

## Leonardo (A100 GPU) — see `LEONARDO_agent_guide.md` for `step ssh` connect/auth
```bash
source .env && ssh "${LEONARDO_SUPERCOMPUTER_SSH_USERNAME}@${LEONARDO_SUPERCOMPUTER_SSH_HOST}"   # guide §2-§5
bash  models/neurosymbolic/slurm/env_setup.sh        # login node: clone + pixi install + mkdir $SCRATCH/nspe_outputs
sbatch models/neurosymbolic/slurm/test_debug.sbatch  # 1 GPU, 30 min: probe + exp01 + exp03 --smoke
sbatch models/neurosymbolic/slurm/grid_lofo.sbatch   # 4 GPUs: LoFO holdout mosfet/igbt/ic + exp04, then aggregate
sbatch models/neurosymbolic/slurm/train_full.sbatch  # 1 GPU: full train-all-3 + exp06 -> 3 submission CSVs
```

## Folder map
```
nspe/         symbolic core (official, roles, rules, grammar, data, ppm, decode, anomaly, predict, eval)
              + neural ranker (model.py, losses.py — the only torch users)
experiments/  exp01 anomaly · exp02 PPM · exp03 neural · exp04 constraint-loss · exp05 scaling · exp06 submission
slurm/        env_setup.sh + test_debug / grid_lofo / train_full .sbatch
configs/      small (~1M) · base (~4M) · grid (LoFO × size matrix)
outputs/      (gitignored) metrics JSON, submission CSVs, checkpoints — NSPE_OUT on Leonardo = $SCRATCH/nspe_outputs
```
