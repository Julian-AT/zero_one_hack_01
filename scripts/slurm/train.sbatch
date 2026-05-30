#!/bin/bash
# SLURM batch for a single training cell on Leonardo.
# Submit with:
#   sbatch --export=ALL,CONFIG=configs/arch/transformer_small.yaml \
#          scripts/slurm/train.sbatch
#
# For a 4-GPU job, change --gpus-per-task=4 (and bump --mem/--cpus accordingly).

#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=train
#SBATCH --output=extras/logs/slurm-%j.out
#SBATCH --error=extras/logs/slurm-%j.err

set -euo pipefail
cd "${SCRATCH}/zero_one_hack_01"

# Load CUDA toolkit so xLSTM's sLSTM block can JIT-compile its kernels.
source /etc/profile.d/modules.sh 2>/dev/null || \
    source /usr/share/Modules/init/bash 2>/dev/null || true
module load gcc/12.2.0 2>&1 || true
module load cuda/12.6 2>&1 || true
echo "[slurm] gcc=$(gcc --version 2>/dev/null | head -1)"
echo "[slurm] CUDA_HOME=${CUDA_HOME:-unset}  nvcc=$(which nvcc 2>/dev/null || echo unset)"

# Compute-node proxy (no direct internet on compute nodes)
export HTTP_PROXY="http://proxyuser:5dd1d2bd00@10.99.0.1:38425"
export HTTPS_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"

# Pick the config from environment, default to transformer_small
CONFIG="${CONFIG:-configs/arch/transformer_small.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/default.yaml}"
TOKEN_CONFIG="${TOKEN_CONFIG:-configs/token/compositional.yaml}"
RUN_NAME="${RUN_NAME:-$(basename "$CONFIG" .yaml)-$(date +%Y%m%d-%H%M%S)}"

echo "[slurm] running config=$CONFIG token=$TOKEN_CONFIG train=$TRAIN_CONFIG"
echo "[slurm] run_name=$RUN_NAME  node=$(hostname)  gpus=$(nvidia-smi -L | wc -l)"
mkdir -p extras/logs extras/checkpoints

# Run inside pixi env
export PATH="${SCRATCH}/.pixi/bin:${PATH}"
pixi run --manifest-path "${SCRATCH}/zero_one_hack_01/pixi.toml" \
    python -m src.train.launch \
        --arch-config "$CONFIG" \
        --train-config "$TRAIN_CONFIG" \
        --token-config "$TOKEN_CONFIG" \
        --run-name "$RUN_NAME"
