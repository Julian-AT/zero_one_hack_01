#!/bin/bash
# Phase-3 eval array.
#   sbatch --dependency=afterok:<phase3_train_jobid> shared/scripts/slurm/phase3_eval.sbatch

#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --array=0-7%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=p3eval
#SBATCH --output=shared/extras/logs/phase3-eval-%A_%a.out
#SBATCH --error=shared/extras/logs/phase3-eval-%A_%a.err

set -euo pipefail
cd "${SCRATCH}/zero_one_hack_01"

source /etc/profile.d/modules.sh 2>/dev/null || \
    source /usr/share/Modules/init/bash 2>/dev/null || true
module load gcc/12.2.0 2>&1 || true
module load cuda/12.6 2>&1 || true

export HTTP_PROXY="http://proxyuser:5dd1d2bd00@10.99.0.1:38425"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

export PATH="${SCRATCH}/.pixi/bin:${PATH}"
PIXI_RUN=(pixi run --manifest-path "${SCRATCH}/zero_one_hack_01/pixi.toml")

mapfile -t CMD < <("${PIXI_RUN[@]}" python -m transformer_xlstm.experiments.phase3_grid \
    --eval-cell "$SLURM_ARRAY_TASK_ID")
CELL_ID=$("${PIXI_RUN[@]}" python -m transformer_xlstm.experiments.phase3_grid \
    --cell-id "$SLURM_ARRAY_TASK_ID")
echo "[phase3-eval] cell_id=$CELL_ID  cmd: ${CMD[*]}"

"${PIXI_RUN[@]}" "${CMD[@]}"
