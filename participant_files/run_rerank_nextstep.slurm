#!/bin/bash
#SBATCH --job-name=rerank_next
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=participant_files/rerank_next_%j.out
#SBATCH --error=participant_files/rerank_next_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source tracks/industrial-infineon/.venv/bin/activate

python participant_files/rerank_nextstep_with_rules.py

wc -l participant_files/predictions/predictions_nextstep.csv
wc -l participant_files/predictions/predictions_nextstep_reranked.csv

echo "Done."
