#!/bin/bash
# =============================================================================
# train_length.sbatch — does TRAINING LONGER help Task 1 / OOD abstraction?
# -----------------------------------------------------------------------------
# Re-runs exp03 (holdout=ic, the worst OOD split) at three training lengths by
# varying epochs (the binding cap): ~500 / ~2500 / ~7500 optimizer steps, same
# d=128 model. Captures neural ID vs OOD next-step so we can see whether more
# gradient steps move in-distribution Top-1 and/or the held-out-family number.
#   sbatch --account=euhpc_d30_031 neurosymbolic-approach/slurm/train_length.sbatch
# =============================================================================
#SBATCH --job-name=nspe_trainlen
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

echo "[trainlen] node=$(hostname)"
pixi run python -c "import torch; print('[trainlen] torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

for EP in 8 40 120; do
  CFG="$NSPE_OUT/len_ep${EP}.yaml"
  cat > "$CFG" <<YAML
d_model: 128
layers: 3
heads: 4
batch: 32
lr: 0.0003
weight_decay: 0.01
role_w: 0.3
sem_w: 0.0
mask_train: false
max_len: 256
warmup: 50
seed: 0
steps: 20000
epochs: ${EP}
YAML
  echo "=== epochs=${EP} ==="
  pixi run python "$E/exp03_neural_ranker.py" --holdout ic --config "$CFG"
  cp "$NSPE_OUT/exp03_ic.json" "$NSPE_OUT/explen_ic_ep${EP}.json"
done

echo "[trainlen] done -> $NSPE_OUT (explen_ic_ep{8,40,120}.json)"
