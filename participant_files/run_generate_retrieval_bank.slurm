#!/bin/bash
#SBATCH --job-name=gen_retrieval_bank
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=participant_files/gen_retrieval_bank_%j.out
#SBATCH --error=participant_files/gen_retrieval_bank_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source tracks/industrial-infineon/.venv/bin/activate

mkdir -p tracks/industrial-infineon/data/retrieval_bank_v1

python tracks/industrial-infineon/training_data/generate_sequences.py \
  --family mosfet \
  --count 50000 \
  --output tracks/industrial-infineon/data/retrieval_bank_v1/MOSFET_retrieval.csv \
  --seed 1001

python tracks/industrial-infineon/training_data/generate_sequences.py \
  --family igbt \
  --count 50000 \
  --output tracks/industrial-infineon/data/retrieval_bank_v1/IGBT_retrieval.csv \
  --seed 1002

python tracks/industrial-infineon/training_data/generate_sequences.py \
  --family ic \
  --count 50000 \
  --output tracks/industrial-infineon/data/retrieval_bank_v1/IC_retrieval.csv \
  --seed 1003

ls -lh tracks/industrial-infineon/data/retrieval_bank_v1/

echo "Retrieval bank generated."
