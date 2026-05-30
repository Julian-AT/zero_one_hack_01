#!/bin/bash
# Background monitor for the LoFO training + eval pipeline.
#
# Polls squeue --me every $POLL_SEC. When the queue has been empty for two
# consecutive polls, runs sacct to detect retroactive failures and exits:
#   0  – all jobs done, no failures
#   1  – one or more cells FAILED / CANCELLED / TIMEOUT / NODE_FAIL / OOM
#   2  – safety timeout
#
# Job ids are passed via env so this works for re-launches.
set -u

TRAIN_JOBS="${TRAIN_JOBS:-43095030,43095215}"
EVAL_JOB="${EVAL_JOB:-43095220}"
POLL_SEC="${POLL_SEC:-300}"
MAX_HOURS="${MAX_HOURS:-8}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

start=$(date +%s)
deadline=$(( start + MAX_HOURS * 3600 ))
empty_streak=0
poll_num=0

echo "[monitor] start poll_sec=$POLL_SEC max_hours=$MAX_HOURS"
echo "[monitor] train_jobs=$TRAIN_JOBS eval_job=$EVAL_JOB"

while true; do
  poll_num=$(( poll_num + 1 ))
  now=$(date +%s)
  if [ "$now" -gt "$deadline" ]; then
    echo "[$(date +%H:%M:%S)] TIMEOUT after ${MAX_HOURS}h"
    exit 2
  fi

  remote=$(bash shared/scripts/leonardo/deploy.sh run \
    "squeue --me --noheader -o '%i %t %M %j' 2>&1" 2>&1 || true)

  total=$(echo "$remote" | grep -cv '^[[:space:]]*$' || true)
  running=$(echo "$remote" | awk 'BEGIN{n=0} $2=="R" {n++} END {print n}')
  pending=$(echo "$remote" | awk 'BEGIN{n=0} $2=="PD" {n++} END {print n}')

  # Snapshot first 3 lines of the queue for context.
  snippet=$(echo "$remote" | head -n 3 | sed 's/^/    /')
  echo "[$(date +%H:%M:%S)] poll=$poll_num total=$total running=$running pending=$pending"
  if [ -n "$snippet" ]; then
    echo "$snippet"
  fi

  if [ "$total" -eq 0 ]; then
    empty_streak=$(( empty_streak + 1 ))
    if [ "$empty_streak" -ge 2 ]; then
      sacct=$(bash shared/scripts/leonardo/deploy.sh run \
        "sacct -X --noheader -j ${TRAIN_JOBS},${EVAL_JOB} --format=jobid,state%20,exitcode 2>&1" 2>&1 || true)
      echo "[$(date +%H:%M:%S)] queue empty for 2 polls; sacct summary:"
      echo "$sacct" | sed 's/^/    /'

      fails=$(echo "$sacct" \
        | grep -iE 'FAILED|CANCELLED|TIMEOUT|NODE_FAIL|BOOT_FAIL|OUT_OF_MEMORY|DEADLINE' || true)
      if [ -n "$fails" ]; then
        echo "[$(date +%H:%M:%S)] FAILURES DETECTED:"
        echo "$fails" | sed 's/^/    /'
        exit 1
      fi
      echo "[$(date +%H:%M:%S)] all clean, exiting 0"
      exit 0
    fi
  else
    empty_streak=0
  fi

  sleep "$POLL_SEC"
done
