#!/bin/bash
#SBATCH --job-name=learn_rerank
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=competition/participant-files/learn_rerank_%j.out
#SBATCH --error=competition/participant-files/learn_rerank_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source competition/track-details/.venv/bin/activate

python3 competition/participant-files/train_learned_contrastive_reranker.py \
  --max-train 50000 \
  --max-val 8000 \
  --max-test 12000 \
  --epochs 8 \
  --batch-size 2048 \
  --lr 2e-3 \
  --seed 42

wc -l competition/participant-files/predictions/predictions_nextstep_learned_reranked.csv

echo "Done."
