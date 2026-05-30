#!/bin/bash
# =============================================================================
# train_full.sbatch — NSPE full train + submission (1 GPU, longer walltime)
# -----------------------------------------------------------------------------
# Trains the final constrained neural ranker on ALL THREE families and emits the
# three submission CSVs (exp06). Also runs the full-data exp03 (holdout=none) so
# the in-distribution ranker numbers are recorded alongside the submission.
# Submit from the repo root after the smoke test passes:
#   sbatch neurosymbolic-approach/slurm/train_full.sbatch
#   squeue --me ; tail -f nspe_full-<jobid>.out
# =============================================================================
#SBATCH --job-name=nspe_full
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

set -euo pipefail

cd "$HOME/zero_one_hack_01"
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$PWD/neurosymbolic-approach:${PYTHONPATH:-}"
export NSPE_OUT="$SCRATCH/nspe_outputs"
mkdir -p "$NSPE_OUT"

E=neurosymbolic-approach/experiments
C=neurosymbolic-approach/configs

echo "[full] node=$(hostname)  NSPE_OUT=$NSPE_OUT"
pixi run python -c "import torch; print('[full] torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

# --- full in-distribution ranker (train on all 3 families, no held-out) ------
pixi run python "$E/exp03_neural_ranker.py" --holdout none --config "$C/base.yaml"

# --- final all-3-family model -> 3 submission CSVs ---------------------------
pixi run python "$E/exp06_make_submission.py" --config "$C/base.yaml"

echo "[full] done. submissions + metrics in $NSPE_OUT"
ls -la "$NSPE_OUT"/submission_task*.csv 2>/dev/null || echo "[full] (submission CSVs not found — check exp06 output)"
