#!/bin/bash
#SBATCH --job-name=gen_retrieval_bank
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=competition/participant-files/gen_retrieval_bank_%j.out
#SBATCH --error=competition/participant-files/gen_retrieval_bank_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source competition/track-details/.venv/bin/activate

mkdir -p competition/track-details/data/retrieval_bank_v1

python competition/track-details/training_data/generate_sequences.py \
  --family mosfet \
  --count 50000 \
  --output competition/track-details/data/retrieval_bank_v1/MOSFET_retrieval.csv \
  --seed 1001

python competition/track-details/training_data/generate_sequences.py \
  --family igbt \
  --count 50000 \
  --output competition/track-details/data/retrieval_bank_v1/IGBT_retrieval.csv \
  --seed 1002

python competition/track-details/training_data/generate_sequences.py \
  --family ic \
  --count 50000 \
  --output competition/track-details/data/retrieval_bank_v1/IC_retrieval.csv \
  --seed 1003

ls -lh competition/track-details/data/retrieval_bank_v1/

echo "Retrieval bank generated."
