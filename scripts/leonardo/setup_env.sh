#!/bin/bash
# One-time setup on a Leonardo login node.
# Run on login01-ext (or 02/05/07) after git-clone.
#
#   ssh USER@login01-ext.leonardo.cineca.it
#   git clone https://github.com/Julian-AT/zero_one_hack_01.git $SCRATCH/zero_one_hack_01
#   cd $SCRATCH/zero_one_hack_01
#   bash scripts/leonardo/setup_env.sh
set -euo pipefail

REPO_ROOT="${SCRATCH:?SCRATCH not set}/zero_one_hack_01"
PIXI_BIN="${SCRATCH}/.pixi/bin/pixi"

cd "$REPO_ROOT"

# 1. Install pixi if missing
if [[ ! -x "$PIXI_BIN" ]]; then
    echo "[setup] installing pixi to \$SCRATCH/.pixi/..."
    export PIXI_HOME="${SCRATCH}/.pixi"
    curl -fsSL https://pixi.sh/install.sh | bash
fi
export PATH="${SCRATCH}/.pixi/bin:${PATH}"

# 2. Install dependencies (downloads happen on login node where internet works)
# Leonardo login nodes have no GPU so pixi's resolver can't auto-detect CUDA.
# Pretend CUDA 12.0 is present so conda-forge picks the GPU build of pytorch.
export CONDA_OVERRIDE_CUDA="12.0"
echo "[setup] resolving + installing dependencies (pixi.toml)..."
pixi install --manifest-path "$REPO_ROOT/pixi.toml"

# 3. Smoke test
echo "[setup] smoke test:"
pixi run --manifest-path "$REPO_ROOT/pixi.toml" smoke

# 4. Build tokenizers (no GPU needed)
echo "[setup] building tokenizers..."
mkdir -p data/processed
pixi run --manifest-path "$REPO_ROOT/pixi.toml" python -m src.data.tokenizer --mode step
pixi run --manifest-path "$REPO_ROOT/pixi.toml" python -m src.data.tokenizer --mode compositional

echo "[setup] done. Next: 'sbatch scripts/slurm/train.sbatch'"
