#!/bin/bash
# =============================================================================
# env_setup.sh — NSPE (Neurosymbolic Process Engine) login-node bootstrap
# =============================================================================
# Run this ONCE on a LEONARDO *login node* (login nodes have internet; compute
# nodes do not). It clones the repo into $HOME, checks out the neurosymbolic
# branch, installs pixi and resolves the GPU environment, and creates the
# scratch output directory the sbatch scripts write to.
#
# Auth & connection values live in the project .env (NEVER hardcode secrets):
#   LEONARDO_SUPERCOMPUTER_SSH_USERNAME / _HOST  -> used to ssh in (see
#                                                   LEONARDO_agent_guide.md §2-§5)
#   GITHUB_PERSONAL_TOKEN                        -> used here to clone over HTTPS
#   GITHUB_PROJECT_URL                           -> repo URL (optional override)
#
# Usage (from the directory that holds your local .env, or after `scp`-ing .env
# to the login node):
#   source /path/to/.env   # or: set -a; . .env; set +a
#   bash env_setup.sh
#
# If `pixi install` exceeds the login-node 10-min CPU cap, run it inside a
# serial interactive allocation (guide §4):
#   srun --partition=lrd_all_serial --time 02:00:00 --mem=16G --pty bash
#   bash env_setup.sh
# =============================================================================
set -euo pipefail

# ---- 0. Resolve config from environment (.env), with safe fallbacks --------
: "${GITHUB_PERSONAL_TOKEN:?Set GITHUB_PERSONAL_TOKEN (export it or 'source .env') before running}"
GITHUB_PROJECT_URL="${GITHUB_PROJECT_URL:-https://github.com/Julian-AT/zero_one_hack_01}"
BRANCH="neurosymbolic-model"
# Strip any scheme so we can splice the token into the HTTPS clone URL.
REPO_HOST_PATH="${GITHUB_PROJECT_URL#https://}"
REPO_HOST_PATH="${REPO_HOST_PATH#http://}"
REPO_HOST_PATH="${REPO_HOST_PATH%.git}"
REPO_DIR_NAME="$(basename "$REPO_HOST_PATH")"
SCRATCH="${SCRATCH:-$HOME/scratch}"   # LEONARDO sets $SCRATCH; fall back for dry-runs

echo "[env_setup] repo      = $GITHUB_PROJECT_URL"
echo "[env_setup] branch    = $BRANCH"
echo "[env_setup] HOME      = $HOME"
echo "[env_setup] SCRATCH   = $SCRATCH"

# ---- 1. Clone (or update) the repo into $HOME using the token --------------
cd "$HOME"
if [ -d "$HOME/$REPO_DIR_NAME/.git" ]; then
    echo "[env_setup] repo already present -> updating"
    cd "$HOME/$REPO_DIR_NAME"
    git fetch origin
else
    echo "[env_setup] cloning repo"
    git clone "https://${GITHUB_PERSONAL_TOKEN}@${REPO_HOST_PATH}.git"
    cd "$HOME/$REPO_DIR_NAME"
fi
git checkout "$BRANCH"
git pull origin "$BRANCH" || echo "[env_setup] (pull skipped — branch may be local-only)"

# ---- 2. Install pixi (login node has internet) and resolve the GPU env -----
if ! command -v pixi >/dev/null 2>&1; then
    echo "[env_setup] installing pixi"
    curl -fsSL https://pixi.sh/install.sh | bash
fi
export PATH="$HOME/.pixi/bin:$PATH"
echo "[env_setup] pixi = $(command -v pixi) ($(pixi --version 2>/dev/null || echo '?'))"

echo "[env_setup] resolving environment (this can take several minutes) ..."
pixi install                    # resolves torch cu121 + deps from pixi.toml
pixi run python -c "import torch; print('[env_setup] torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

# ---- 3. Create the scratch output directory the sbatch scripts use ---------
mkdir -p "$SCRATCH/nspe_outputs"

echo "[env_setup] DONE."
echo "[env_setup]   code:   $HOME/$REPO_DIR_NAME  (branch $BRANCH)"
echo "[env_setup]   OUTPUT: $SCRATCH/nspe_outputs"
echo "[env_setup] next: sbatch neurosymbolic-approach/slurm/test_debug.sbatch"
