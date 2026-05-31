#!/bin/bash
# Reproduce the full cross-model benchmark on a Leonardo GPU node (~15 min on one A100).
# Same pipeline as ./reproduce.sh, just on the GPU.
#
# Submit from the repo root, with YOUR account:
#     sbatch --account=<your_account> reproduce.sbatch
# Add --reservation=<name> only if your project uses one. boost_usr_prod is Leonardo's GPU partition.

#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=01:30:00
#SBATCH --job-name=reproduce-benchmark
#SBATCH --output=reproduce-%j.out
#SBATCH --error=reproduce-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"          # run from wherever you submitted (the repo root)

# CUDA toolkit for the GPU build of PyTorch.
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module load gcc/12.2.0 2>&1 || true
module load cuda/12.6  2>&1 || true

# Use the environment built by setup_env.sh (no network needed on the compute node).
ENV_PY="$PWD/.pixi/envs/default/bin/python"
if [ ! -x "$ENV_PY" ]; then
  echo "ERROR: environment not found at $ENV_PY" >&2
  echo "Run 'bash shared/scripts/leonardo/setup_env.sh' on a login node first." >&2
  exit 1
fi
export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"

echo "node=$(hostname)  account=${SLURM_JOB_ACCOUNT:-?}"
"$ENV_PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# reproduce.sh auto-detects CUDA and runs the whole benchmark on the GPU.
PYTHON="$ENV_PY" bash reproduce.sh
