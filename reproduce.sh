#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — one command to reproduce the unified cross-model benchmark.
#
#   git clone <repo> && cd <repo>
#   python -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   ./reproduce.sh                # full run  (CPU ~60-75 min, GPU ~10 min)
#   ./reproduce.sh quick          # smoke test (<2 min): baselines + neurosymbolic only
#
# Produces, all from a clean checkout with NO Leonardo access required:
#   shared/benchmark/eval_set_v1/        the frozen common eval set (deterministic seeds)
#   shared/benchmark/results_summary.csv ID / OOD / drop per model x task
#   submission/UNIFIED_BENCHMARK.md      the report (tables + figures regenerated)
#   submission/benchmark_assets/*.png    comparison figures
#
# Every score comes from the OFFICIAL scorer (competition/participant-files/eval_metrics.py).
# Device: CUDA if present, else CPU. MPS (Apple GPU) is intentionally skipped — its
# embedding-gather kernel is buggy for this model and would crash a Mac run.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"                       # repo root, regardless of caller cwd
PYTHON="${PYTHON:-python}"
MODE="${1:-full}"

echo "================ environment ================"
"$PYTHON" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
DEVICE="$("$PYTHON" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")"
export PROCESS_LOGIC_DEVICE="$DEVICE"
[ "$DEVICE" = "cuda" ] && export SSL_DEVICE="cuda"
echo "device: $DEVICE   (mps skipped — unstable for this model)"

echo "================ [1/5] common eval set (deterministic) ================"
"$PYTHON" shared/benchmark/make_eval_set.py

if [ "$MODE" != "quick" ]; then
  echo "================ [2/5] SSL training data ================"
  "$PYTHON" shared/benchmark/make_train_data.py
  echo "================ [3/5] train compact checkpoints (CPU ~60-75 min / GPU ~10 min) ================"
  PYTHON="$PYTHON" bash shared/benchmark/train_txl.sh
  PYTHON="$PYTHON" bash shared/benchmark/train_ssl.sh
else
  echo "================ [quick] skipping neural training — scoring baselines + neurosymbolic only ================"
fi

echo "================ [4/5] run unified benchmark (official eval_metrics.py) ================"
if [ "$MODE" = "quick" ]; then
  # only the no-training models — fast everywhere, even if stale checkpoints exist
  "$PYTHON" shared/benchmark/make_benchmark.py --models trigram grammar neurosymbolic
else
  # all models; make_benchmark skips any whose checkpoint is missing
  "$PYTHON" shared/benchmark/make_benchmark.py
fi

echo "================ [5/5] tables + figures ================"
"$PYTHON" shared/benchmark/report.py

echo
echo "================ DONE ================"
echo "results_summary.csv:"
cat shared/benchmark/results_summary.csv
echo
echo "Report : submission/UNIFIED_BENCHMARK.md"
echo "Figures: submission/benchmark_assets/"
echo
echo "To regenerate the official submission CSVs on the organizer eval inputs and score them:"
echo "  PYTHONPATH=models python competition/participant-files/make_eval_predictions.py \\"
echo "      --checkpoint shared/benchmark/ssl_checkpoints/ssl-all3/checkpoint_best.pt \\"
echo "      --vocab shared/benchmark/ssl_checkpoints/ssl-all3/vocab.json"
echo "  python competition/participant-files/eval_metrics.py --task anomaly \\"
echo "      --ground-truth shared/extras/results/eval_inputs/eval_input_anomaly_truth.csv \\"
echo "      --predictions competition/participant-files/predictions/predictions_anomaly.csv"
