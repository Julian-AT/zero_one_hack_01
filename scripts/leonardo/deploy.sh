#!/bin/bash
# Drives the Leonardo deployment without ever surfacing credentials.
#
# Reads .env (Username, Password) into local env vars only.
# Installs a dedicated SSH key on Leonardo via `expect` (one-time, password
# never echoed). Subsequent ops use the key.
#
# Sub-commands (positional):
#   probe          – verify SSH key works (installs key if needed)
#   bootstrap      – rsync the repo to $SCRATCH and run setup_env.sh
#   smoke          – submit a tiny transformer_small / 100-step job
#   grid           – submit the full 7-cell scaling grid array
#   status         – `squeue --me`
#   tail JOBID     – tail the slurm log for a job id
#   run "cmd"      – run arbitrary command on Leonardo
#   pull-light     – rsync extras/logs + extras/results (no .pt) from $SCRATCH
#   pull           – full rsync of extras/ including checkpoints
#
# Usage:
#   bash scripts/leonardo/deploy.sh probe
#   bash scripts/leonardo/deploy.sh bootstrap
#   bash scripts/leonardo/deploy.sh smoke
#   bash scripts/leonardo/deploy.sh grid
#
# Default chains: probe → bootstrap (when called with no args).
set -eu

# Never trace; if you need to debug, comment this in.
# set -x

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
LEO_KEY="${HOME}/.ssh/leonardo_hack"
LEO_HOSTS=(
    "login01-ext.leonardo.cineca.it"
    "login02-ext.leonardo.cineca.it"
    "login05-ext.leonardo.cineca.it"
    "login07-ext.leonardo.cineca.it"
)

# ---------- 1. Load .env tolerantly (handles `K=V` and `K = V`) -----------

load_env() {
    [ -f "$ENV_FILE" ] || { echo "[leo] .env missing at $ENV_FILE" >&2; exit 1; }
    while IFS='' read -r raw || [ -n "$raw" ]; do
        case "$raw" in
            ''|\#*) continue ;;
        esac
        # split on first '=' tolerating whitespace around it
        local key="${raw%%=*}"
        local val="${raw#*=}"
        key="${key//[[:space:]]/}"
        val="${val#"${val%%[![:space:]]*}"}"   # ltrim
        val="${val%"${val##*[![:space:]]}"}"   # rtrim
        # strip surrounding quotes if present
        case "$val" in
            \"*\")  val="${val#\"}"; val="${val%\"}" ;;
            \'*\') val="${val#\'}"; val="${val%\'}" ;;
        esac
        [ -z "$key" ] && continue
        # Export to this process only (no parent shell, no logs).
        export "$key=$val"
    done < "$ENV_FILE"
}

load_env

# Required vars present? ${:?} prints name but NEVER the value.
: "${Username:?Username not set in .env}"
: "${Password:?Password not set in .env}"

# Pick a host; can be overridden via LEO_HOST in the environment.
LEO_HOST="${LEO_HOST:-${LEO_HOSTS[0]}}"

# ---------- 2. SSH key setup --------------------------------------------

ensure_key_exists() {
    if [ ! -f "$LEO_KEY" ]; then
        echo "[leo] generating dedicated SSH key (no passphrase) at $LEO_KEY"
        ssh-keygen -t ed25519 -f "$LEO_KEY" -N "" -C "leonardo-hack-abb" >/dev/null
        chmod 600 "$LEO_KEY"
    fi
}

key_auth_works() {
    ssh -i "$LEO_KEY" \
        -o BatchMode=yes \
        -o ConnectTimeout=8 \
        -o StrictHostKeyChecking=accept-new \
        -o UserKnownHostsFile="${HOME}/.ssh/known_hosts" \
        "${Username}@${LEO_HOST}" "true" >/dev/null 2>&1
}

install_key_with_password() {
    # Use expect to feed the password to ssh-copy-id; never log it.
    # The Password env var is read by Tcl as $env(Password); not echoed.
    echo "[leo] installing key on $LEO_HOST (one-time password prompt; silent)…"
    LEO_HOST="$LEO_HOST" LEO_KEY="$LEO_KEY" \
    expect <<'EXPECT_EOF'
set timeout 60
log_user 0
spawn ssh-copy-id -i $env(LEO_KEY).pub \
                  -o StrictHostKeyChecking=accept-new \
                  -o PreferredAuthentications=password \
                  -o PubkeyAuthentication=no \
                  $env(Username)@$env(LEO_HOST)
expect {
    -re "(yes/no|fingerprint)" { send "yes\r"; exp_continue }
    -re "(P|p)assword:" {
        send -- "$env(Password)\r"
        exp_continue
    }
    -re "Number of key.+added" { exp_continue }
    -re "ERROR|Permission denied" {
        log_user 1
        puts stderr "\[leo] key install failed; check Username/Password in .env"
        exit 2
    }
    eof
}
catch wait result
set status [lindex $result 3]
exit $status
EXPECT_EOF
}

probe_cmd() {
    ensure_key_exists
    if key_auth_works; then
        echo "[leo] key auth works ✓"
        return 0
    fi
    install_key_with_password
    if key_auth_works; then
        echo "[leo] key auth works ✓ (key installed)"
    else
        echo "[leo] still cannot authenticate. Verify Username/Password in .env." >&2
        exit 1
    fi
}

# ---------- 3. Wrappers (all silent on credentials) ---------------------

leo() {
    ssh -i "$LEO_KEY" \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        "${Username}@${LEO_HOST}" "$@"
}

# ---------- 4. Sub-commands ---------------------------------------------

bootstrap_cmd() {
    probe_cmd
    # Resolve $SCRATCH on the remote (varies per user).
    local remote_scratch
    remote_scratch="$(leo 'echo $SCRATCH')"
    [ -z "$remote_scratch" ] && { echo "[leo] could not resolve \$SCRATCH on remote" >&2; exit 1; }
    local remote_dir="${remote_scratch}/zero_one_hack_01"

    echo "[leo] rsyncing local repo → ${LEO_HOST}:${remote_dir}/ (skipping heavy/local-only dirs)"
    rsync -az --delete \
          -e "ssh -i ${LEO_KEY} -o BatchMode=yes" \
          --exclude='.venv/' \
          --exclude='__pycache__/' \
          --exclude='.DS_Store' \
          --exclude='extras/checkpoints/' \
          --exclude='extras/logs/' \
          --exclude='data/processed/' \
          --exclude='data/generated/' \
          --exclude='wandb/' \
          --exclude='runs/' \
          --exclude='.env' \
          "${REPO_ROOT}/" "${Username}@${LEO_HOST}:${remote_dir}/"

    echo "[leo] running setup_env.sh (pixi install + tokenizers)…"
    leo "bash ${remote_dir}/scripts/leonardo/setup_env.sh"
}

smoke_cmd() {
    probe_cmd
    echo "[leo] submitting smoke job: transformer_small, 100 steps…"
    leo 'cd "$SCRATCH/zero_one_hack_01"
         sbatch --time=00:15:00 \
                --export=ALL,CONFIG=configs/arch/transformer_small.yaml \
                --output=extras/logs/slurm-smoke-%j.out \
                --error=extras/logs/slurm-smoke-%j.err \
                scripts/slurm/train.sbatch
         squeue --me
        '
}

grid_cmd() {
    probe_cmd
    echo "[leo] submitting 7-cell scaling grid (array job)…"
    leo 'cd "$SCRATCH/zero_one_hack_01" && sbatch scripts/slurm/grid.sbatch && squeue --me'
}

status_cmd() {
    leo 'squeue --me'
}

tail_cmd() {
    local jobid="${1:-}"
    [ -z "$jobid" ] && { echo "usage: deploy.sh tail JOBID" >&2; exit 1; }
    leo "cd \"\$SCRATCH/zero_one_hack_01\" && tail -n 200 extras/logs/slurm-${jobid}.out 2>/dev/null || tail -n 200 extras/logs/grid-*_${jobid}.out 2>/dev/null"
}

run_cmd() {
    # Run an arbitrary command on Leonardo. Useful for ad-hoc queries.
    #   deploy.sh run 'sacctmgr show association where user=$USER ...'
    [ "$#" -lt 1 ] && { echo "usage: deploy.sh run 'command'" >&2; exit 1; }
    leo "$*"
}

# Resolve remote SCRATCH once for pull commands.
_remote_scratch() { leo 'echo $SCRATCH'; }

pull_light_cmd() {
    # Fast pull: just logs + summaries + result CSVs, no model .pt files.
    # Safe to run every few minutes during a long grid run.
    local remote_scratch
    remote_scratch="$(_remote_scratch)"
    local remote_dir="${remote_scratch}/zero_one_hack_01"
    echo "[leo] pulling logs + results (no .pt files) → ${REPO_ROOT}/extras/"
    rsync -avz \
          -e "ssh -i ${LEO_KEY} -o BatchMode=yes" \
          --include='extras/' \
          --include='extras/logs/***' \
          --include='extras/results/***' \
          --include='extras/checkpoints/' \
          --include='extras/checkpoints/*/' \
          --include='extras/checkpoints/*/summary.json' \
          --include='extras/checkpoints/*/*.txt' \
          --exclude='*.pt' \
          --exclude='*.safetensors' \
          --exclude='*.bin' \
          --exclude='*' \
          "${Username}@${LEO_HOST}:${remote_dir}/" "${REPO_ROOT}/"
}

pull_cmd() {
    # Full pull: includes the .pt checkpoints. Heavier.
    local remote_scratch
    remote_scratch="$(_remote_scratch)"
    local remote_dir="${remote_scratch}/zero_one_hack_01"
    echo "[leo] pulling extras/ (logs + results + checkpoints) → ${REPO_ROOT}/extras/"
    mkdir -p "${REPO_ROOT}/extras"
    rsync -avz --progress \
          -e "ssh -i ${LEO_KEY} -o BatchMode=yes" \
          "${Username}@${LEO_HOST}:${remote_dir}/extras/" "${REPO_ROOT}/extras/"
}

# ---------- 5. Dispatch -------------------------------------------------

case "${1:-default}" in
    probe)      probe_cmd ;;
    bootstrap)  bootstrap_cmd ;;
    smoke)      smoke_cmd ;;
    grid)       grid_cmd ;;
    status)     status_cmd ;;
    tail)       shift; tail_cmd "$@" ;;
    run)        shift; run_cmd "$@" ;;
    pull)       pull_cmd ;;
    pull-light) pull_light_cmd ;;
    default|"") probe_cmd; bootstrap_cmd ;;
    *)
        echo "Unknown command: $1" >&2
        echo "Usage: $0 {probe|bootstrap|smoke|grid|status|tail JOBID}" >&2
        exit 1
        ;;
esac
