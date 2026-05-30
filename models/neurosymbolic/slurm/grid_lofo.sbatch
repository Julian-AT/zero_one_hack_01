#!/bin/bash
# =============================================================================
# grid_lofo.sbatch — NSPE Leave-One-Family-Out grid (4 GPUs, one full node)
# -----------------------------------------------------------------------------
# The real test battery and the Task-4 (OOD) measurement. Launches four
# independent runs, one pinned per A100 via CUDA_VISIBLE_DEVICES, then waits and
# aggregates with the official scorer:
#   GPU 0: exp03 holdout=mosfet   (train igbt+ic, eval OOD on mosfet)
#   GPU 1: exp03 holdout=igbt     (train mosfet+ic, eval OOD on igbt)
#   GPU 2: exp03 holdout=ic       (train mosfet+igbt, eval OOD on ic)
#   GPU 3: exp04 constraint-loss ablation (mask/semantic-loss on|off)
# Submit from the repo root:
#   sbatch models/neurosymbolic/slurm/grid_lofo.sbatch
#   squeue --me ; tail -f nspe_grid-<jobid>.out
# =============================================================================
#SBATCH --job-name=nspe_grid
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --mem=480GB
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

set -euo pipefail

cd "$HOME/zero_one_hack_01"
# Use the pre-installed pixi env directly: pixi.toml has a duplicate
# [target.linux-64.dependencies] table that breaks `pixi run`, and compute nodes
# have no internet to re-solve. The env in .pixi/envs/default is already built.
export PXENV="$HOME/zero_one_hack_01/.pixi/envs/default"
export LD_LIBRARY_PATH="$PXENV/lib:${LD_LIBRARY_PATH:-}"
pixi() { shift 2; "$PXENV/bin/python" "$@"; }   # shim: `pixi run python ...` -> env python
export PYTHONPATH="$PWD/models/neurosymbolic:${PYTHONPATH:-}"
export NSPE_OUT="$SCRATCH/nspe_outputs"
mkdir -p "$NSPE_OUT"

E=models/neurosymbolic/experiments
C=models/neurosymbolic/configs

echo "[grid] node=$(hostname)  NSPE_OUT=$NSPE_OUT"
pixi run python -c "import torch; print('[grid] torch', torch.__version__, 'cuda devices', torch.cuda.device_count())"

# --- 4 independent runs, pinned to the 4 A100s ------------------------------
CUDA_VISIBLE_DEVICES=0 pixi run python "$E/exp03_neural_ranker.py" --holdout mosfet --config "$C/base.yaml" &
CUDA_VISIBLE_DEVICES=1 pixi run python "$E/exp03_neural_ranker.py" --holdout igbt   --config "$C/base.yaml" &
CUDA_VISIBLE_DEVICES=2 pixi run python "$E/exp03_neural_ranker.py" --holdout ic     --config "$C/base.yaml" &
CUDA_VISIBLE_DEVICES=3 pixi run python "$E/exp04_constraint_loss.py" --config "$C/base.yaml" &

wait

# --- aggregate + score everything with the official scorer ------------------
pixi run python "$E/exp05_scaling.py" --aggregate

# --- render every chart (anomaly + LoFO + neural scaling/ablation) -----------
pixi run python "$E/make_charts.py" || echo "[grid] charts step skipped"

echo "[grid] grid done -> $NSPE_OUT (charts in $NSPE_OUT/charts)"
