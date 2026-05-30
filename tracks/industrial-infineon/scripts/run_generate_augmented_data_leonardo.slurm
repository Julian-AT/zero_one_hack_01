#!/bin/bash
#SBATCH --job-name=gen_semicond_data
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --account=euhpc_d30_031
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x_%j.out
#SBATCH --error=runs/slurm/%x_%j.err

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Submit dir: $SLURM_SUBMIT_DIR"

cd "$SLURM_SUBMIT_DIR"

mkdir -p runs/slurm
mkdir -p data/generated/base_seed

echo "Working directory: $(pwd)"

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "training_data/.venv" ]; then
    source training_data/.venv/bin/activate
elif [ -d "data/.venv" ]; then
    source data/.venv/bin/activate
else
    echo "No local .venv found. Using current Python."
fi

echo "Python: $(which python)"
python --version

echo ""
echo "Checking required scripts..."
ls training_data/generate_sequences.py
ls data/generate_coverage_guided.py
ls data/generate_invalid_sequences.py
ls data/generate_hard_invalid_sequences.py
ls data/build_task_datasets.py

echo ""
echo "============================================================"
echo "1/5 Generate base seed datasets"
echo "============================================================"

python training_data/generate_sequences.py \
  --family mosfet \
  --count 1000 \
  --output data/generated/base_seed/MOSFET_variants.csv \
  --seed 42

python training_data/generate_sequences.py \
  --family igbt \
  --count 1000 \
  --output data/generated/base_seed/IGBT_variants.csv \
  --seed 43

python training_data/generate_sequences.py \
  --family ic \
  --count 1000 \
  --output data/generated/base_seed/IC_variants.csv \
  --seed 44

echo ""
echo "============================================================"
echo "2/5 Generate coverage-guided valid data"
echo "============================================================"

python data/generate_coverage_guided.py \
  --existing data/generated/base_seed/MOSFET_variants.csv data/generated/base_seed/IGBT_variants.csv data/generated/base_seed/IC_variants.csv \
  --target-per-family 10000 \
  --max-candidates-per-family 400000 \
  --min-feature-count 250 \
  --output-dir data/coverage_guided_v1 \
  --include-existing-in-output \
  --seed 42

echo ""
echo "============================================================"
echo "3/5 Generate easy invalid data"
echo "============================================================"

python data/generate_invalid_sequences.py \
  --valid-input data/coverage_guided_v1/coverage_guided_sequences.csv \
  --target-per-rule-family 1000 \
  --max-tries-per-rule-family 200000 \
  --max-validator-violations 4 \
  --output-dir data/easy_invalid_v1 \
  --write-mixed \
  --seed 42

echo ""
echo "============================================================"
echo "4/5 Generate hard invalid data"
echo "============================================================"

python data/generate_hard_invalid_sequences.py \
  --valid-input data/coverage_guided_v1/coverage_guided_sequences.csv \
  --target-per-rule-family 1000 \
  --max-tries-per-rule-family 300000 \
  --max-validator-violations 4 \
  --output-dir data/hard_invalid_v1 \
  --write-mixed \
  --seed 42

echo ""
echo "============================================================"
echo "5/5 Build task datasets"
echo "============================================================"

python data/build_task_datasets.py \
  --valid-input data/coverage_guided_v1/coverage_guided_sequences.csv \
  --easy-invalid-input data/easy_invalid_v1/invalid_sequences.csv \
  --hard-invalid-input data/hard_invalid_v1/hard_invalid_sequences.csv \
  --output-dir data/task_datasets_v1 \
  --context-window 64 \
  --next-step-stride 1 \
  --completion-cut-fracs 0.25 0.5 0.75 \
  --completion-target-window 96 \
  --seed 42

echo ""
echo "============================================================"
echo "Generated files"
echo "============================================================"

ls -lh data/coverage_guided_v1/coverage_guided_sequences.csv
ls -lh data/easy_invalid_v1/invalid_sequences.csv
ls -lh data/hard_invalid_v1/hard_invalid_sequences.csv
ls -lh data/task_datasets_v1/next_step_prediction.csv
ls -lh data/task_datasets_v1/anomaly_detection.csv
ls -lh data/task_datasets_v1/rule_attribution.csv
ls -lh data/task_datasets_v1/task_dataset_report.md

echo "Job finished: $(date)"
