#!/bin/bash
# One-shot, location-agnostic training setup: pixi + all Python/PyTorch deps.
#
#   git clone https://github.com/Julian-AT/zero_one_hack_01.git
#   bash zero_one_hack_01/scripts/leonardo/setup_env.sh
#
# On Leonardo: clone into $SCRATCH and run from a login node (internet works there).
# Re-running is safe and idempotent.
set -euo pipefail

# --- resolve repo root from this script's own location -----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# --- keep pixi + envs inside the repo (no dependence on $HOME or $SCRATCH) ----
export PIXI_HOME="$REPO_ROOT/.pixi"
PIXI_BIN="$PIXI_HOME/bin/pixi"

# 1. install pixi if missing
if [[ ! -x "$PIXI_BIN" ]]; then
    echo "[setup] installing pixi -> $PIXI_HOME"
    curl -fsSL https://pixi.sh/install.sh | bash
fi
export PATH="$PIXI_HOME/bin:$PATH"

# 2. resolve + install everything declared in pixi.toml
#    (PyTorch cu121 on Linux, plus numpy/pandas/tokenizers/the rest)
echo "[setup] installing dependencies from pixi.toml ..."
pixi install --manifest-path pixi.toml

# 3. make PyPI wheels (numpy/torch) load the env's libstdc++ (GLIBCXX_3.4.29+)
#    instead of the older system /lib64/libstdc++.so.6. Dropped into each env's
#    conda activate.d so 'pixi run' / 'pixi shell' apply it automatically.
echo "[setup] wiring libstdc++ load-order fix ..."
shopt -s nullglob
for env in "$PIXI_HOME"/envs/*/; do
    mkdir -p "${env}etc/conda/activate.d"
    cat > "${env}etc/conda/activate.d/zz_libstdcpp.sh" <<'EOF'
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
EOF
done
shopt -u nullglob

# 4. smoke test — on a login node there's no GPU, so cuda shows False (expected).
#    Real GPU check: run 'pixi run smoke' inside an srun/sbatch job.
echo "[setup] smoke test:"
pixi run --manifest-path pixi.toml smoke

# 5. build tokenizers (CPU only) — uses the tasks defined in pixi.toml
echo "[setup] building tokenizers ..."
mkdir -p data/processed
pixi run --manifest-path pixi.toml tokenize-step
pixi run --manifest-path pixi.toml tokenize-comp

echo "[setup] done. GPU is verified inside the job: 'sbatch scripts/slurm/train.sbatch'"