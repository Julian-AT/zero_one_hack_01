#!/bin/bash
#SBATCH --job-name=eval_preds
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=participant_files/eval_preds_%j.out
#SBATCH --error=participant_files/eval_preds_%j.err

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Submit dir: $SLURM_SUBMIT_DIR"

cd "$SLURM_SUBMIT_DIR"

source tracks/industrial-infineon/.venv/bin/activate

echo "Python: $(which python)"
python --version

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Checking required files..."
ls tracks/industrial-infineon/runs/ssl_hybrid_new_coverage_guided_v1/checkpoint_best.pt
ls participant_files/eval_input_valid.csv
ls participant_files/eval_input_anomaly.csv
ls participant_files/make_eval_predictions.py

echo "Running eval prediction generation..."
python participant_files/make_eval_predictions.py

echo "Checking outputs..."
ls -lh participant_files/predictions/
wc -l participant_files/eval_input_valid.csv
wc -l participant_files/predictions/predictions_nextstep.csv
wc -l participant_files/predictions/predictions_completion.csv
wc -l participant_files/eval_input_anomaly.csv
wc -l participant_files/predictions/predictions_anomaly.csv

echo "Job finished: $(date)"
