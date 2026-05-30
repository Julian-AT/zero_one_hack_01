# Run NSPE on Leonardo — copy-paste recipe

> Everything is committed and pushed to branch **`neurosymbolic-model`**. The
> symbolic results (anomaly + PPM LoFO) and their charts are already produced on
> CPU and live in `models/neurosymbolic/outputs/charts/`. The Leonardo runs add
> the **neural** results (neural-vs-PPM, the constraint-loss ablation, scaling
> curves) and regenerate the full chart set including those panels.
>
> Auth is interactive (CINECA identity provider) — see `LEONARDO_agent_guide.md`
> §2–§3. The steps below assume you have an SSH cert / `ssh leonardo` working.

## 0. One-time auth (local machine)
```bash
brew install step                       # macOS; see guide §2 for others
step ca bootstrap --ca-url=https://sshproxy.hpc.cineca.it \
  --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
eval $(ssh-agent)
step ssh login "$LEONARDO_SUPERCOMPUTER_SSH_USERNAME@<your-idp-email>" --provisioner cineca-hpc
ssh "$LEONARDO_SUPERCOMPUTER_SSH_USERNAME@$LEONARDO_SUPERCOMPUTER_SSH_HOST"   # values from .env
```

## 1. Stage code + env (on a login node — has internet)
```bash
# fresh clone + pixi install + $SCRATCH output dir:
bash models/neurosymbolic/slurm/env_setup.sh
# --- OR, if the repo is already on Leonardo: ---
cd ~/zero_one_hack_01 && git fetch origin && git checkout neurosymbolic-model && git pull
export PATH="$HOME/.pixi/bin:$PATH" && pixi install
pixi run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
> If `pixi install` exceeds the 10-min login CPU cap, run it inside
> `srun --partition=lrd_all_serial --time 02:00:00 --mem=16G --pty bash` (guide §4).

## 2. Smoke test first (1 GPU, debug QoS, ~5 min)
```bash
cd ~/zero_one_hack_01
sbatch models/neurosymbolic/slurm/test_debug.sbatch
squeue --me
tail -f nspe_debug-<jobid>.out     # expect: probe + exp01 pass, tiny exp03 trains on GPU
```

## 3. The real runs (submit both; they write to `$SCRATCH/nspe_outputs`)
```bash
sbatch models/neurosymbolic/slurm/grid_lofo.sbatch   # 4×A100: LoFO mosfet/igbt/ic + constraint-loss ablation + aggregate + charts
sbatch models/neurosymbolic/slurm/train_full.sbatch  # 1×A100: full 3-family model -> submissions + charts
squeue --me
tail -f nspe_grid-<jobid>.out
```
Each job ends by running `make_charts.py`, so charts land in
`$SCRATCH/nspe_outputs/charts/` (now including the neural panels).

## 4. Pull results back (from your local machine)
```bash
H="$LEONARDO_SUPERCOMPUTER_SSH_USERNAME@$LEONARDO_SUPERCOMPUTER_SSH_HOST"
scp "$H:\$SCRATCH/nspe_outputs/charts/*.png"          models/neurosymbolic/outputs/charts/
scp "$H:\$SCRATCH/nspe_outputs/submission_task*.csv"  models/neurosymbolic/outputs/
scp "$H:\$SCRATCH/nspe_outputs/*.json"                models/neurosymbolic/outputs/
```

## What each job produces
| Job | GPUs | Time | Outputs |
|---|---|---|---|
| `test_debug` | 1 | <30 min | sanity: probe, exp01 (anomaly), tiny exp03 |
| `grid_lofo` | 4 | ≤4 h | exp03 LoFO ×3 families, exp04 ablation, exp05 aggregate, **all charts** |
| `train_full` | 1 | ≤8 h | full-data ranker, `submission_task{1,2,3}.csv`, **all charts** |

## Charts you will get
Already produced on CPU (in the repo now):
`anomaly_id_per_rule.png`, `anomaly_ood_recovery.png`, `ppm_lofo_nextstep.png`,
`ppm_lofo_completion.png`, `ood_drop_comparison.png`.
Added by the Leonardo runs: `exp03_neural_vs_ppm.png`, `exp04_constraint_loss.png`,
`exp05_scaling.png` (these are skipped on CPU because they need real GPU training).
