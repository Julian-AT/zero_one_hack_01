#!/bin/bash
# LoFO ablation grid — 48 LoFO cells + 16 final all-3 cells = 64 cells total.
# Each cell trains one (arch, size, heads, family_dropout, fold) recipe.
# Concurrency capped at 4 (one A100 per cell) — fills the 4-A100 reservation.
#
# Submit (after sync’ing the repo to $SCRATCH via scripts/leonardo/deploy.sh):
#   sbatch scripts/slurm/lofo_grid.sbatch
#
# Inspect the cell list locally first:
#   pixi run python -m src.experiments.lofo_grid --list | column -t -s$'\t'
#
# Re-run a subset by overriding --array:
#   sbatch --array=0,3,9 scripts/slurm/lofo_grid.sbatch

#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --array=0-63%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=lofo
#SBATCH --output=extras/logs/lofo-%A_%a.out
#SBATCH --error=extras/logs/lofo-%A_%a.err

set -euo pipefail
cd "${SCRATCH}/zero_one_hack_01"

# CUDA + GCC for xLSTM JIT
source /etc/profile.d/modules.sh 2>/dev/null || \
    source /usr/share/Modules/init/bash 2>/dev/null || true
module load gcc/12.2.0 2>&1 || true
module load cuda/12.6 2>&1 || true
echo "[lofo] task=$SLURM_ARRAY_TASK_ID gcc=$(gcc --version 2>/dev/null | head -1)"
echo "[lofo] CUDA_HOME=${CUDA_HOME:-unset}  nvcc=$(which nvcc 2>/dev/null || echo unset)"

# Compute-node proxy
export HTTP_PROXY="http://proxyuser:5dd1d2bd00@10.99.0.1:38425"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

export PATH="${SCRATCH}/.pixi/bin:${PATH}"
PIXI_RUN=(pixi run --manifest-path "${SCRATCH}/zero_one_hack_01/pixi.toml")

# Resolve cell — print one token per line, read into bash array.
mapfile -t CMD < <("${PIXI_RUN[@]}" python -m src.experiments.lofo_grid \
    --cell "$SLURM_ARRAY_TASK_ID")
CELL_ID=$("${PIXI_RUN[@]}" python -m src.experiments.lofo_grid \
    --cell-id "$SLURM_ARRAY_TASK_ID")
echo "[lofo] cell_id=$CELL_ID"
echo "[lofo] cmd: ${CMD[*]}"

"${PIXI_RUN[@]}" "${CMD[@]}"
