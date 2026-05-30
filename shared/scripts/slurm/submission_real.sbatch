#!/bin/bash
# Generate submission CSVs against the REAL eval inputs in competition/participant-files/.
# Submit:
#   sbatch --export=ALL,CKPT=<path>,OUT=<dir> shared/scripts/slurm/submission_real.sbatch

#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --job-name=submit-real
#SBATCH --output=shared/extras/logs/submit-real-%j.out
#SBATCH --error=shared/extras/logs/submit-real-%j.err

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
OUT="${OUT:-extras/results/submission_v2_real}"

export PATH="${SCRATCH}/.pixi/bin:${PATH}"
mkdir -p "$OUT"

pixi run --manifest-path "${SCRATCH}/zero_one_hack_01/pixi.toml" \
    python -m transformer_xlstm.eval.make_submission \
        --checkpoint "$CKPT" \
        --valid-input competition/participant-files/eval_input_valid.csv \
        --anomaly-input competition/participant-files/eval_input_anomaly.csv \
        --output-dir "$OUT"
