#!/bin/bash
#SBATCH --job-name=ssl_hybrid_newdata
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"

cd "$SLURM_SUBMIT_DIR"

mkdir -p runs/slurm
mkdir -p runs/ssl_hybrid_new_coverage_guided_v1

if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "No .venv found. Using current Python."
fi

echo "Python: $(which python)"
python --version

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
PY

echo ""
echo "Checking training data..."
ls -lh data/coverage_guided_v1/coverage_guided_sequences.csv
head -n 2 data/coverage_guided_v1/coverage_guided_sequences.csv

echo ""
echo "Starting hybrid SSL training..."

python shared/scripts/train_ssl_hybrid_process_transformer.py \
  --data data/coverage_guided_v1/coverage_guided_sequences.csv \
  --out-dir runs/ssl_hybrid_new_coverage_guided_v1 \
  --epochs 60 \
  --batch-size 256 \
  --d-model 256 \
  --layers 6 \
  --heads 8 \
  --ff-mult 4 \
  --dropout 0.15 \
  --lr 3e-4 \
  --weight-decay 0.05 \
  --warmup-steps 200 \
  --label-smoothing 0.05 \
  --family-dropout 0.25 \
  --input-step-mask-prob 0.03 \
  --class-weight none \
  --max-len 192 \
  --num-workers 8 \
  --seed 42

echo ""
echo "Training finished: $(date)"
echo "Final metrics:"
tail -n 20 runs/ssl_hybrid_new_coverage_guided_v1/metrics.csv
