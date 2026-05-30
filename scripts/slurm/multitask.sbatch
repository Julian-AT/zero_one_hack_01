#!/bin/bash
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --job-name=multitask
#SBATCH --output=extras/logs/multitask-%j.out
#SBATCH --error=extras/logs/multitask-%j.err

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

ARCH_CFG="${ARCH_CFG:-configs/arch/transformer_medium.yaml}"
TOKEN_CFG="${TOKEN_CFG:-configs/token/compositional.yaml}"
RUN_NAME="${RUN_NAME:-multitask-$(basename "$ARCH_CFG" .yaml)-$(date +%Y%m%d-%H%M%S)}"
echo "[mt] $RUN_NAME on $(hostname)"

export PATH="${SCRATCH}/.pixi/bin:${PATH}"
pixi run --manifest-path "${SCRATCH}/zero_one_hack_01/pixi.toml" \
    python -m src.train.launch \
        --arch-config "$ARCH_CFG" \
        --train-config configs/train/multitask.yaml \
        --token-config "$TOKEN_CFG" \
        --run-name "$RUN_NAME"
