#!/usr/bin/env bash
# Train the 4 SSL-hybrid benchmark checkpoints (ID all-3 + 3 LoFO) on CPU, in parallel.
# Needs shared/benchmark/_train_data/ (run make_train_data.py first).
set -e
cd "$(dirname "$0")/../.."          # repo root
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
SSL_DEVICE="${SSL_DEVICE:-cpu}"   # set to 'auto' (cuda-or-cpu) or 'cuda' on a GPU box
PY="${PYTHON:-python}"
S=competition/track-details/scripts/train_ssl_hybrid_process_transformer.py
DATA=shared/benchmark/_train_data
CKPT=shared/benchmark/ssl_checkpoints
LOG=shared/benchmark/_trainlogs; mkdir -p "$LOG"
EPOCHS="${EPOCHS:-12}"

run() {  # run <name> <data-csv>
  "$PY" "$S" --data "$DATA/$2" --out-dir "$CKPT/$1" \
    --device "$SSL_DEVICE" --no-amp --epochs "$EPOCHS" --d-model 128 --layers 3 --heads 4 \
    --batch-size 64 --max-len 256 --num-workers 0 --seed 42 > "$LOG/$1.log" 2>&1
  echo "  done $1"
}

run ssl-all3        train_all3.csv &
run ssl-held_mosfet train_held_mosfet.csv &
run ssl-held_igbt   train_held_igbt.csv &
run ssl-held_ic     train_held_ic.csv &
wait
echo "SSL-hybrid checkpoints:"; ls "$CKPT"/ssl-*/checkpoint_best.pt
