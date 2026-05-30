#!/bin/bash
#SBATCH --job-name=retrieval_eval
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=participant_files/retrieval_eval_%j.out
#SBATCH --error=participant_files/retrieval_eval_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source tracks/industrial-infineon/.venv/bin/activate

python participant_files/retrieval_augmented_eval.py

wc -l participant_files/predictions/predictions_nextstep_retrieval.csv
wc -l participant_files/predictions/predictions_completion_retrieval.csv

echo "Done."
