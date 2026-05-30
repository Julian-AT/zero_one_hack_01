#!/bin/bash
# =============================================================================
# extras_exp0405.sbatch — constraint-loss ablation (exp04) + scaling sweep (exp05)
# -----------------------------------------------------------------------------
# The two supplementary panels. Fast now that the semantic-loss legal-step mask
# uses the incremental legal_next_sets (~500x faster than per-candidate
# re-validation). 1 GPU, debug QoS is plenty.
#   sbatch --account=euhpc_d30_031 neurosymbolic-approach/slurm/extras_exp0405.sbatch
# =============================================================================
#SBATCH --job-name=nspe_extras
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
export PXENV="$HOME/zero_one_hack_01/.pixi/envs/default"
export LD_LIBRARY_PATH="$PXENV/lib:${LD_LIBRARY_PATH:-}"
pixi() { shift 2; "$PXENV/bin/python" "$@"; }   # shim: `pixi run python ...` -> env python
export PYTHONPATH="$PWD/neurosymbolic-approach:${PYTHONPATH:-}"
export NSPE_OUT="$SCRATCH/nspe_outputs"
mkdir -p "$NSPE_OUT"

E=neurosymbolic-approach/experiments
C=neurosymbolic-approach/configs

echo "[extras] node=$(hostname)  NSPE_OUT=$NSPE_OUT"
pixi run python -c "import torch; print('[extras] torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

# constraint-loss ablation: {mask off/on} x {sem_w 0/0.5}
pixi run python "$E/exp04_constraint_loss.py" --config "$C/base.yaml"

# size x data scaling sweep
pixi run python "$E/exp05_scaling.py" --config "$C/base.yaml"

# merge + render whatever charts are computable here (full set is rendered locally)
pixi run python "$E/exp05_scaling.py" --aggregate || true
pixi run python "$E/make_charts.py" || echo "[extras] charts step skipped"

echo "[extras] done -> $NSPE_OUT"
