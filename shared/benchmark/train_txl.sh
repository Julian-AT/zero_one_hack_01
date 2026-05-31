#!/usr/bin/env bash
# Train the 4 transformer_xlstm benchmark checkpoints (ID all-3 + 3 LoFO) on CPU, in parallel.
# Compact, laptop-reproducible budget (see submission/UNIFIED_BENCHMARK.md §8).
set -e
cd "$(dirname "$0")/../.."          # repo root
export PYTHONPATH=models
: "${PROCESS_LOGIC_DEVICE:=cpu}"; export PROCESS_LOGIC_DEVICE   # cuda if a GPU is present
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
PY="${PYTHON:-python}"
CKPT=shared/benchmark/checkpoints
LOG=shared/benchmark/_trainlogs; mkdir -p "$LOG"
STEPS="${STEPS:-2500}"
MAXLEN="${MAXLEN:-512}"   # max_len is the quality lever (256 plateaus ~0.63 Top-1; 512 -> ~0.78)

run() {  # run <name> <families>
  "$PY" -m transformer_xlstm.train.launch \
    --arch-config configs/arch/transformer_small.yaml \
    --train-config configs/train/default.yaml \
    --token-config configs/token/compositional.yaml \
    --run-name "$1" \
    --override train.max_steps="$STEPS" train.warmup_steps=250 train.eval_every=$((STEPS+1)) \
               train.save_every=$((STEPS+1)) train.batch_size=16 train.max_len="$MAXLEN" \
               "data.families=$2" loss.validity_weight=0.5 loss.rule_id_weight=0.3 \
               tracking.wandb_mode=disabled data.num_workers=0 out.checkpoint_dir="$CKPT" \
    > "$LOG/$1.log" 2>&1
  echo "  done $1"
}

run bench-tr-small-all3        "[mosfet,igbt,ic]" &
run bench-tr-small-held_mosfet "[igbt,ic]" &
run bench-tr-small-held_igbt   "[mosfet,ic]" &
run bench-tr-small-held_ic     "[mosfet,igbt]" &
wait
echo "transformer_xlstm checkpoints:"; ls "$CKPT"/bench-tr-small-*/final.pt
