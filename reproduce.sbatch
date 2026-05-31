#!/bin/bash
# Run the full reproducible benchmark on a Leonardo GPU node (≈10-15 min).
# It is the exact same pipeline as ./reproduce.sh — just on an A100 instead of CPU.
#
#   # one-time, on a login node (has internet):
#   cd $SCRATCH && git clone <repo-url> zero_one_hack_01
#   bash zero_one_hack_01/shared/scripts/leonardo/setup_env.sh
#   # then:
#   cd zero_one_hack_01 && sbatch reproduce.sbatch
#   # watch:  tail -f shared/extras/logs/reproduce-<jobid>.out
#
# The --account / --reservation below are our hackathon allocation; change them to yours.

#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=01:30:00
#SBATCH --job-name=reproduce-benchmark
#SBATCH --output=shared/extras/logs/reproduce-%j.out
#SBATCH --error=shared/extras/logs/reproduce-%j.err

set -euo pipefail
cd "${SCRATCH}/zero_one_hack_01"

# CUDA toolkit (for the GPU build of torch; xLSTM's kernels if you extend to it)
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module load gcc/12.2.0 2>&1 || true
module load cuda/12.6  2>&1 || true

export PIXI_HOME="$(pwd)/.pixi"
export PATH="$PIXI_HOME/bin:$PATH"

# Same one-command reproducer, inside the pixi env, on the GPU.
# reproduce.sh auto-detects CUDA and runs there; results land in the same paths as a local run.
pixi run --manifest-path pixi.toml bash reproduce.sh
