#!/bin/bash
# SLURM array job: runs the 7-cell scaling grid in parallel.
# Each task gets 1 A100. With reservation=s_tra_ncc (1 node = 4 GPUs),
# 4 cells run concurrently, the next 3 queue. Total wall ~ max(per-cell time).
#
# Submit:
#   sbatch scripts/slurm/grid.sbatch

#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --job-name=grid
#SBATCH --output=extras/logs/grid-%A_%a.out
#SBATCH --error=extras/logs/grid-%A_%a.err

set -euo pipefail
cd "${SCRATCH}/zero_one_hack_01"

# Load CUDA toolkit so xLSTM's sLSTM block can JIT-compile its kernels.
# (Compute nodes have the CUDA runtime via the driver, but not nvcc + headers
# until a module is loaded.)
source /etc/profile.d/modules.sh 2>/dev/null || \
    source /usr/share/Modules/init/bash 2>/dev/null || true
module load gcc/12.2.0 2>&1 || true
module load cuda/12.6 2>&1 || true
echo "[grid] gcc=$(gcc --version 2>/dev/null | head -1)"
echo "[grid] CUDA_HOME=${CUDA_HOME:-unset}  nvcc=$(which nvcc 2>/dev/null || echo unset)"

# Cell index → (arch_config, token_config)
CONFIGS=(
    "configs/arch/transformer_small.yaml configs/token/compositional.yaml"
    "configs/arch/transformer_medium.yaml configs/token/compositional.yaml"
    "configs/arch/transformer_large.yaml configs/token/compositional.yaml"
    "configs/arch/xlstm_small.yaml configs/token/compositional.yaml"
    "configs/arch/xlstm_medium.yaml configs/token/compositional.yaml"
    "configs/arch/xlstm_large.yaml configs/token/compositional.yaml"
    "configs/arch/transformer_medium.yaml configs/token/step.yaml"   # ablation
)

read -r ARCH_CFG TOKEN_CFG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="cell${SLURM_ARRAY_TASK_ID}-$(basename "$ARCH_CFG" .yaml)-$(basename "$TOKEN_CFG" .yaml)"

echo "[grid] task $SLURM_ARRAY_TASK_ID: $ARCH_CFG + $TOKEN_CFG"
echo "[grid] run_name=$RUN_NAME"

# Compute-node proxy
export HTTP_PROXY="http://proxyuser:5dd1d2bd00@10.99.0.1:38425"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

export PATH="${SCRATCH}/.pixi/bin:${PATH}"
pixi run --manifest-path "${SCRATCH}/zero_one_hack_01/pixi.toml" \
    python -m src.train.launch \
        --arch-config "$ARCH_CFG" \
        --train-config configs/train/default.yaml \
        --token-config "$TOKEN_CFG" \
        --run-name "$RUN_NAME"
