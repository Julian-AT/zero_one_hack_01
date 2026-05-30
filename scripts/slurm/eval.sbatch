#!/bin/bash
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=eval
#SBATCH --output=extras/logs/eval-%j.out
#SBATCH --error=extras/logs/eval-%j.err

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

CKPT="${CKPT:?Set CKPT to checkpoint path}"
OUT="${OUT:-extras/results/eval/$(basename $(dirname "$CKPT"))}"
echo "[eval] ckpt=$CKPT  out=$OUT"

export PATH="${SCRATCH}/.pixi/bin:${PATH}"
pixi run --manifest-path "${SCRATCH}/zero_one_hack_01/pixi.toml" \
    python -m src.eval.run_eval --checkpoint "$CKPT" --output-dir "$OUT"
