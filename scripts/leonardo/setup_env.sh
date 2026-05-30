#!/bin/bash
# One-time setup on a Leonardo login node.
# Run on login01-ext (or 02/05/07) after git-clone.
#
#   ssh USER@login01-ext.leonardo.cineca.it
#   git clone https://github.com/Julian-AT/zero_one_hack_01.git
#   cd zero_one_hack_01
#   bash scripts/leonardo/setup_env.sh
set -euo pipefail

# Resolve repo root from this script's own location (scripts/leonardo/setup_env.sh -> repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PIXI_HOME="${REPO_ROOT}/.pixi"
PIXI_BIN="${PIXI_HOME}/bin/pixi"

cd "$REPO_ROOT"

# 1. Install pixi if missing (self-contained under the repo)
if [[ ! -x "$PIXI_BIN" ]]; then
    echo "[setup] installing pixi to ${PIXI_HOME}..."
    export PIXI_HOME
    curl -fsSL https://pixi.sh/install.sh | bash
fi
export PATH="${PIXI_HOME}/bin:${PATH}"

# 2. Install dependencies (downloads happen on login node where internet works)
#    cwd is REPO_ROOT, so pixi auto-discovers pixi.toml — no --manifest-path needed
echo "[setup] resolving + installing dependencies (pixi.toml)..."
pixi install

# 3. Smoke test
echo "[setup] smoke test:"
pixi run smoke

# 4. Build tokenizers (no GPU needed)
echo "[setup] building tokenizers..."
mkdir -p data/processed
pixi run python -m src.data.tokenizer --mode step
pixi run python -m src.data.tokenizer --mode compositional

echo "[setup] done. Next: 'sbatch scripts/slurm/train.sbatch'"