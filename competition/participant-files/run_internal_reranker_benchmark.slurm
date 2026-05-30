#!/bin/bash
#SBATCH --job-name=bench_rerank
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=competition/participant-files/bench_rerank_%j.out
#SBATCH --error=competition/participant-files/bench_rerank_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source competition/track-details/.venv/bin/activate

python competition/participant-files/benchmark_reranker_internal.py \
  --split test \
  --max-examples 20000 \
  --seed 42
