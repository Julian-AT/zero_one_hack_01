#!/bin/bash
# =============================================================================
# test_debug.sbatch — NSPE pipeline smoke test (1 GPU, debug QoS, 30 min)
# -----------------------------------------------------------------------------
# Verifies the full stack end-to-end BEFORE spending the LoFO grid budget:
#   - CPU symbolic spine: OOD anomaly probe + exp01 (Task-3 oracle, all 10 rules)
#   - a tiny GPU training run (exp03 --smoke) to prove torch+CUDA+masking work.
# Submit from the repo root after env_setup.sh:
#   sbatch neurosymbolic-approach/slurm/test_debug.sbatch
#   squeue --me ; tail -f nspe_debug-<jobid>.out
# =============================================================================
#SBATCH --job-name=nspe_debug
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

set -euo pipefail

cd "$HOME/zero_one_hack_01"
# Use the pre-installed pixi env directly: pixi.toml has a duplicate
# [target.linux-64.dependencies] table that breaks `pixi run`, and compute nodes
# have no internet to re-solve. The env in .pixi/envs/default is already built.
export PXENV="$HOME/zero_one_hack_01/.pixi/envs/default"
export LD_LIBRARY_PATH="$PXENV/lib:${LD_LIBRARY_PATH:-}"
pixi() { shift 2; "$PXENV/bin/python" "$@"; }   # shim: `pixi run python ...` -> env python
export PYTHONPATH="$PWD/neurosymbolic-approach:${PYTHONPATH:-}"
export NSPE_OUT="$SCRATCH/nspe_outputs"
mkdir -p "$NSPE_OUT"

E=neurosymbolic-approach/experiments
C=neurosymbolic-approach/configs

echo "[debug] node=$(hostname)  NSPE_OUT=$NSPE_OUT"
pixi run python -c "import torch; print('[debug] torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

# --- CPU symbolic sanity (fast, no GPU needed) -------------------------------
pixi run python "$E/ood_symbolic_probe.py"
pixi run python "$E/exp01_symbolic_anomaly.py"

# --- tiny GPU train to prove the neural stack + constraint masking -----------
pixi run python "$E/exp03_neural_ranker.py" --config "$C/small.yaml" --smoke

echo "[debug] smoke test complete -> $NSPE_OUT"
