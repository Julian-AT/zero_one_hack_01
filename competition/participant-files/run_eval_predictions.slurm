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
#SBATCH --output=competition/participant-files/eval_preds_%j.out
#SBATCH --error=competition/participant-files/eval_preds_%j.err

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Submit dir: $SLURM_SUBMIT_DIR"

cd "$SLURM_SUBMIT_DIR"

source competition/track-details/.venv/bin/activate

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
ls competition/track-details/runs/ssl_hybrid_new_coverage_guided_v1/checkpoint_best.pt
ls competition/participant-files/eval_input_valid.csv
ls competition/participant-files/eval_input_anomaly.csv
ls competition/participant-files/make_eval_predictions.py

echo "Running eval prediction generation..."
python competition/participant-files/make_eval_predictions.py

echo "Checking outputs..."
ls -lh competition/participant-files/predictions/
wc -l competition/participant-files/eval_input_valid.csv
wc -l competition/participant-files/predictions/predictions_nextstep.csv
wc -l competition/participant-files/predictions/predictions_completion.csv
wc -l competition/participant-files/eval_input_anomaly.csv
wc -l competition/participant-files/predictions/predictions_anomaly.csv

echo "Job finished: $(date)"
