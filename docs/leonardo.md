# LEONARDO Supercomputer — Agent Operations Guide

> Purpose: a single, self-contained reference for an AI agent that must (1) authenticate and log in to the LEONARDO supercomputer, (2) navigate login and compute (training) nodes, and (3) submit and monitor GPU model-training jobs via SLURM.
> Context: AI Factory Austria (AI:AT) hackathon access to LEONARDO at CINECA. Hackathon-specific values are flagged with **[HACKATHON]**.

---

## 0. TL;DR — Minimal path to a running training job

1. Install `step` client → bootstrap CINECA CA → start ssh-agent → request ssh certificate.
2. `ssh <username>@login01-ext.leonardo.cineca.it` (one of login01/02/05/07-ext). **[HACKATHON]** no 2FA.
3. Put code/data in `$SCRATCH` (large) or `$HOME` (≤50 GB). **[HACKATHON]** do NOT use `$WORK` / `$FAST`.
4. Build/get a Singularity container or a Pixi/conda/venv environment on a **login node** (login nodes have internet; compute nodes do not).
5. Write a SLURM script (`job.sh`) requesting the `boost_usr_prod` GPU partition + **[HACKATHON]** `--reservation=s_tra_ncc`.
6. `sbatch job.sh` → `squeue --me` → `tail -f slurm-<jobid>.out` → `scancel <jobid>` if needed.

Key constants for LEONARDO Booster (GPU) nodes — **fair-share rule of thumb**:
- 1 node = 4× NVIDIA A100 (64 GB each), 1× Intel Ice Lake CPU (32 cores).
- `--mem` ≈ `120GB × gpus-per-task`
- `--cpus-per-task` ≈ `8 × gpus-per-task`
- `--gpus-per-task` up to 4; `--ntasks-per-node=1` (best practice on GPU nodes).
- Max walltime on the production GPU QoS examples here: `24:00:00`; debug QoS: `00:30:00`.

---

> **Credentials / config live in `.env`.** Read connection details and tokens from the project `.env` rather than hardcoding them. Relevant variables:
> - `LEONARDO_SUPERCOMPUTER_SSH_USERNAME` — your CINECA/LEONARDO username (part before the `@`)
> - `LEONARDO_SUPERCOMPUTER_SSH_PASSWORD` — identity-provider password used during `step ssh login` (hackathon: no 2FA code needed)
> - `LEONARDO_SUPERCOMPUTER_SSH_HOST` — login node, e.g. `login05-ext.leonardo.cineca.it` (also valid: `login01/02/07-ext`)
> - `GITHUB_PERSONAL_TOKEN` — token for cloning/pulling the repo (run git on a **login node**; compute nodes have no internet)
> - `GITHUB_PROJECT_URL` — the project repo, e.g. `https://github.com/Julian-AT/zero_one_hack_01`
>
> Copy `.env.example` → `.env` and fill in the blanks. Build the SSH target in code as `"${LEONARDO_SUPERCOMPUTER_SSH_USERNAME}@${LEONARDO_SUPERCOMPUTER_SSH_HOST}"` and use it as `ssh "<that>"`.

---

## 1. System Overview

LEONARDO is hosted at CINECA (Italy), a EuroHPC pre-exascale system (Nvidia A100, ranked #10 on the Top500 as of June 2025). It has two compute modules:

- **Booster module** — GPU-intensive workloads (large-scale AI training, deep learning). Each node: 1× Intel Ice Lake CPU (32 cores) + 4× NVIDIA A100 GPUs (64 GB each). GPUs interconnected via NVLink; CPU via PCIe Gen 4. **This is the module you use for model training.**
- **DCGP module** (Data-Centric General Purpose) — CPU-only nodes, 2× Intel Ice Lake CPUs, 100+ cores per node, for classical HPC / non-GPU workloads.
- Modules interconnected with HDR InfiniBand.

**Typical HPC topology** (how the pieces fit together):
- **You / agent** → SSH → **login nodes** (environment setup, data transfer, small tasks, internet access).
- **Login nodes** ↔ **SLURM scheduler** ↔ **compute nodes** (where real computation runs).
- A **fast network** connects login nodes, compute nodes, and shared **storage**.
- Login and compute nodes share the same storage filesystems.

> Rule: never run heavy compute on login nodes. They enforce a **10-minute CPU-time limit** per process. Everything heavy goes through SLURM onto compute nodes.

---

## 2. Authentication & First-Time Access (`step` client)

LEONARDO uses **certificate-based SSH authentication**. Temporary SSH certificates are issued/renewed with the `step` client. **[HACKATHON]** Two-factor authentication is NOT used during the hackathon.

### Step 1 — Install the `step` client
macOS (Homebrew):
```bash
brew install step
```
Other OSes: see https://smallstep.com/docs/step-ca/installation

### Step 2 — Bootstrap CINECA's Certificate Authority
```bash
step ca bootstrap \
  --ca-url=https://sshproxy.hpc.cineca.it \
  --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
```

### Step 3 — Start the SSH agent
```bash
eval $(ssh-agent)
```

### Step 4 — Request a short-lived SSH certificate
```bash
step ssh login 'USER@EMAIL' --provisioner cineca-hpc
```
You are redirected to your institution's identity provider to authenticate (password + one-time code).

### Step 5 — Connect to LEONARDO
```bash
ssh <username>@login.leonardo.cineca.it
```
End the session with:
```bash
logout
```

### Step 6 — Common issue: "REMOTE HOST IDENTIFICATION HAS CHANGED"
Edit known hosts and delete the stale LEONARDO line:
```bash
nano ~/.ssh/known_hosts
```
…or connect directly to a specific login node:
```bash
ssh <username>@login01-ext.leonardo.cineca.it
```

---

## 3. SSH Config — Simplified, Auto-Renewing Login

Add this to `~/.ssh/config` (adjust username and email). It auto-renews the certificate when it is close to expiring, so you can just run `ssh leonardo`.

```sshconfig
Match host leonardo exec "step ssh needs-renewal ~/.ssh/leonardo_key-cert.pub --expires-in=1m && step ssh certificate 'USER@EMAIL' ~/.ssh/leonardo_key --provisioner cineca-hpc --no-password --insecure --force"

Host leonardo
    HostName login01-ext.leonardo.cineca.it
    User <username>
    SetEnv LANG=C
    IdentityFile ~/.ssh/leonardo_key
```
Then simply:
```bash
ssh leonardo
```

---

## 4. Login Nodes

Use any of these (all equivalent). **[HACKATHON]** no 2FA:
```
login01-ext.leonardo.cineca.it
login02-ext.leonardo.cineca.it
login05-ext.leonardo.cineca.it
login07-ext.leonardo.cineca.it
```
Generic round-robin alias: `login.leonardo.cineca.it`.

What login nodes are for: editing/compiling code, data transfer, **package/container downloads (they have internet)**, environment setup, very short test runs.

**Login node CPU-time limit: 10 minutes per process.** For longer interactive work (e.g. downloading/converting a large container, heavy preprocessing), grab an interactive serial allocation instead:
```bash
srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty bash
```

---

## 5. Storage

Shared across all login and compute nodes.

| Area       | Quota / policy                          | Use for                                              |
|------------|-----------------------------------------|------------------------------------------------------|
| `$HOME`    | 50 GB, backed up daily                  | scripts, configs, small datasets, job scripts        |
| `$SCRATCH` | Large quota; files deleted after 40 days| **[HACKATHON] large files, datasets, checkpoints**   |
| `$PUBLIC`  | 50 GB                                   | sharing files between LEONARDO users                 |
| `$WORK`    | project-based, high-throughput I/O      | large I/O / collaborative work (**[HACKATHON] do NOT use**) |
| `$FAST`    | fast tier                               | **[HACKATHON] do NOT use**                           |

> **[HACKATHON] guidance:** keep code/job scripts in `$HOME`, put datasets/checkpoints/large outputs in `$SCRATCH`. Do not use `$WORK` or `$FAST` during the hackathon.
> Note: general CINECA docs recommend `$WORK`/`$SCRATCH` for batch jobs; the hackathon overrides this to `$SCRATCH` only.

Check usage / quota:
```bash
cindata
cinQuota
```

---

## 6. Data Transfer (from your local machine)

Small data → via login nodes; large transfers (> ~10 min) → via dedicated **data mover** nodes.

Login nodes:
```bash
# local -> LEONARDO
scp /absolute/path/from/file <username>@login.leonardo.cineca.it:/absolute/path/to/
# LEONARDO -> local
scp <username>@login.leonardo.cineca.it:/absolute/path/from/file /absolute/path/to/
```
Data mover nodes (large volumes):
```bash
scp /absolute/path/from/file <username>@data.leonardo.cineca.it:/absolute/path/to/
scp <username>@data.leonardo.cineca.it:/absolute/path/from/file /absolute/path/to/
```

---

## 7. Internet Access from Compute Nodes (proxy workaround)

**Compute nodes have NO internet access.** Login nodes do. Download large files (datasets, containers, model weights) on a login node and read them from `$SCRATCH` inside the job.

For unavoidable **low-bandwidth** traffic from inside a job, export these proxy variables in the SLURM script. **[HACKATHON]** credentials:
```bash
export HTTP_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export HTTPS_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export http_proxy=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export https_proxy=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
```
Caveat: the proxy restarts periodically (it is itself subject to the 10-min login CPU limit), so TCP connections drop occasionally. Use it only for small traffic; always download large files from login nodes.

---

## 8. Software Environment

### 8.1 Environment modules
```bash
module avail                 # list available modules
module load <appl>           # load a module into the current shell
module load autoload <appl>  # load a module + all dependencies
module load <appl>/<version> # load a specific (incl. hidden) version
module list                  # currently loaded modules
module help <appl>           # info / help for an application
module unload <appl>         # unload one module
module purge                 # unload everything
module av -a                 # also show hidden (unsupported) modules
```

### 8.2 Profiles & `modmap`
Modules are grouped into profiles; only `base` is loaded at login. Search all profiles at once:
```bash
modmap -m <name>             # find a module across all profiles, e.g. modmap -m python
modmap -c compilers          # list compilers
```
Then load the profile, then the module:
```bash
module load profile/chem-phys
module load lammps/29aug2024
```
Deep-learning software lives under the `deeplrn` profile (`module load profile/deeplrn`).

### 8.3 Compilers (quick reference)
- GPU (NVIDIA): GCC (offload target `nvptx`), NVIDIA `nvhpc` (`nvc`, `nvc++`, `nvfortran`, `nvcc`; `-cuda`, `-acc`, `-mp=gpu`), NVIDIA CUDA.
- Intel CPUs: Intel oneAPI (`icx`, `icpx`, `ifx`; classic `icc`/`icpc`/`ifort` via `intel-oneapi-compilers-classic`), GCC.
- AMD CPUs: AOCC (`clang`, `clang++`, `flang`), GCC.

### 8.4 Python virtual environments
Create envs in `$WORK`/`$SCRATCH` (not `$HOME`, quota is small):
```bash
module load python/<version>
# module load py-mpi4py/<version>   # if MPI4Py needed
python -m venv my_env
source my_env/bin/activate
pip install <package>
# ... work ...
deactivate
```

### 8.5 Conda
```bash
source activate my_conda_env
```

### 8.6 Pixi (fast, reproducible package manager — recommended for the hackathon)
Installs from both conda-forge and PyPI. https://pixi.sh/
```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi init myproject
cd myproject
pixi add python              # from conda-forge
pixi add --pypi openai       # from PyPI
pixi run python -c 'print("Hello World!")'
```
Run a script inside the pixi environment from a job (see §10 example):
```bash
/path/to/pixi run --as-is [--manifest-path <pixi_project_path>] python3 script.py
```

### 8.7 Spack (build software from source)
```bash
ml spack
spack spec -Il <package>     # ALWAYS check the spec first (verify CINECA-optimized deps with ^)
spack install <package>
ml <package>
# if a modulefile is missing:
spack module tcl refresh --upstream-modules <package>
module use $PUBLIC/spack-<version>/modules
```

---

## 9. Containers (Singularity / Apptainer)

HPC systems run **Singularity/Apptainer**, not Docker (the two are nearly identical). Docs: https://docs.sylabs.io/guides/latest/user-guide/

### Convert a Docker image to a `.sif` (do this with internet, via an interactive serial allocation):
```bash
srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty \
  singularity pull vllm-openai-v0.21.0-cu129.sif docker://docker.io/vllm/vllm-openai:0.21.0-cu129
```

### Run something inside a container (GPU-enabled):
```bash
singularity exec --nv --bind $SCRATCH:/scratch container.sif python3
```
- `--nv` exposes the NVIDIA GPUs to the container.
- `--bind <host>:<container>` mounts host paths inside the container.

---

## 10. SLURM — Job Submission

SLURM (Simple Linux Utility for Resource Management) allocates compute nodes, launches/monitors jobs, and queues pending work. You describe needs in a **job script** (`#SBATCH` directives) and submit with `sbatch`.

### 10.1 Core `#SBATCH` directives

| Directive | Meaning | Example |
|-----------|---------|---------|
| `-J` / `--job-name` | job name in the queue | `--job-name=my_job` |
| `--partition` | node group / queue | `--partition=boost_usr_prod` |
| `--account` | project to bill | `--account=EUHPC_1234` |
| `--qos` | quality of service / limits | `--qos=boost_qos_dbg` |
| `--reservation` | reserved node set | **[HACKATHON]** `--reservation=s_tra_ncc` |
| `--time` | max walltime HH:MM:SS | `--time=0:30:00` |
| `--nodes` | number of nodes | `--nodes=1` |
| `--ntasks-per-node` | srun tasks per node | `--ntasks-per-node=1` |
| `--gpus-per-task` | GPUs per task (≤4) | `--gpus-per-task=1` |
| `--cpus-per-task` | CPU cores per task | `--cpus-per-task=8` |
| `--mem` | memory per node | `--mem=120GB` |
| `--gres` | generic resource (alt GPU syntax) | `--gres=gpu:1` |
| `--output` | stdout file | `--output=job.out` |
| `--error` | stderr file | `--error=job.err` |

The first line `#!/bin/bash` sets the shell. Lines starting with `#SBATCH` are scheduler directives. The bottom of the script sets up the environment and launches the program (typically via `srun`).

> GPU best practice on LEONARDO: `--ntasks-per-node=1` and let the single task access multiple GPUs via `--gpus-per-task`. Multiple tasks per node complicate GPU binding.

### 10.2 Partition & QoS (Booster / GPU)

Production GPU partition: `boost_usr_prod`. QoS on that partition:

| QoS | Max nodes/job | Walltime | Max nodes/cores/GPUs | Priority |
|-----|---------------|----------|----------------------|----------|
| `boost_qos_dbg` (debug) | 2 | 00:30:00 | 2 / 64 / 8 | 80 |
| `boost_qos_bprod` | 65–256 | 24:00:00 | 256 nodes | 60 |
| `boost_qos_lprod` | 8 | 4-00:00:00 | 8 / – / 32 | 40 |

Start with short test jobs on the debug QoS before launching full production runs.

Other partitions seen here: `lrd_all_serial` (interactive serial / container building, has internet via login-side), `g100_usr_prod` (GALILEO100, not LEONARDO).

---

## 11. Example Job Scripts

### 11.1 LEONARDO — 1 GPU (recommended hackathon starting point)
```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc   # [HACKATHON] reservation
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # srun tasks per node (keep at 1 for GPU jobs)
#SBATCH --gpus-per-task=1         # GPUs (up to 4 on LEONARDO)
#SBATCH --mem=120GB               # 120GB * gpus-per-task
#SBATCH --cpus-per-task=8         # 8 * gpus-per-task
#SBATCH --time=0:30:00            # HH:MM:SS, up to 24:00:00

# Command(s) to run on the compute node:
python3 script.py
```

### 11.2 LEONARDO — 1 GPU inside a Pixi environment
```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00

export RUN_COMMAND="/path/to/pixi run --as-is [--manifest-path pixi_project_path]"
$RUN_COMMAND python3 script.py
```

### 11.3 LEONARDO — 1 GPU inside a Singularity container
```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00

export CONTAINER="singularity exec --nv container.sif"
$CONTAINER python3 script.py
```

### 11.4 LEONARDO — 2 GPUs (scale mem & cpus with GPUs)
```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --mem=240GB               # 120GB * 2
#SBATCH --cpus-per-task=16        # 8 * 2
#SBATCH --time=0:30:00

export CONTAINER="singularity exec --nv container.sif"
$CONTAINER python3 script.py
```

### 11.5 LEONARDO — 4 GPUs (full node)
```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --mem=480GB               # 120GB * 4
#SBATCH --cpus-per-task=32        # 8 * 4
#SBATCH --time=0:30:00

export CONTAINER="singularity exec --nv container.sif"
$CONTAINER python3 script.py
```

### 11.6 LEONARDO — 2 nodes × 4 GPUs (multi-node; use `srun`)
```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
# #SBATCH --reservation=s_tra_ncc   # [HACKATHON] reservation only covers 1 node per team — omit for multi-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --mem=480GB
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00

export CONTAINER="singularity exec --nv container.sif"
srun $CONTAINER python3 script.py   # NOTE: srun is required to launch across multiple nodes
```
> **[HACKATHON]** the `s_tra_ncc` reservation only has enough for **1 node per team**. For multi-node runs you must drop the reservation directive.

### 11.7 Multi-GPU / multi-node training template (CUDA + NCCL + MPI, generic CINECA style)
```bash
#!/bin/bash
#SBATCH --job-name=multi_gpu_job
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4          # e.g. 1 task per GPU
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4                 # GPUs per node
#SBATCH --partition=<gpu_partition>
#SBATCH --qos=<qos_name>
#SBATCH --output=multiGPUJob.out
#SBATCH --error=multiGPUJob.err
#SBATCH --account=<project_account>

module load cuda/12.2
module load openmpi
# module load <your_app_dependencies>

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO               # NCCL logging for multi-GPU comms

srun ./my_distributed_gpu_app --config config.yaml
```

### 11.8 Pure OpenMP (CPU, shared-memory) template
```bash
#!/bin/bash
#SBATCH --job-name=openmp_job
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48           # physical cores per task
#SBATCH --partition=<partition_name>
#SBATCH --qos=<qos_name>
#SBATCH --mem=<mem_per_node>         # e.g. 128G
#SBATCH --output=myJob.out
#SBATCH --error=myJob.err
#SBATCH --account=<project_account>

module load intel
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun ./myprogram < myinput > myoutput
```

### 11.9 MPI (CPU) template
```bash
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=<mem_per_node>
#SBATCH --partition=<partition_name>
#SBATCH --qos=<qos_name>
#SBATCH --job-name=jobMPI
#SBATCH --error=myJob.err
#SBATCH --output=myJob.out
#SBATCH --account=<project_account>

module load intel intelmpi
srun myprogram < myinput > myoutput
```

### 11.10 Serial (single-core) template
```bash
#!/bin/bash
#SBATCH --job-name=serial_job
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=<partition_name>
#SBATCH --qos=<qos_name>
#SBATCH --mem=2G
#SBATCH --output=serialJob.out
#SBATCH --account=<project_account>
```

---

## 12. SLURM — Submit, Monitor, Manage

```bash
# Submit a job
sbatch job.sh

# List your jobs
squeue --me
squeue -u <user>            # by user
squeue -j <job_id>          # by job id
squeue -p <partition>       # by partition
squeue -t R                 # running only  (PD = pending)
squeue --start              # estimated start times for pending jobs

# View output (files default to slurm-<job_id>.out unless --output set)
cat slurm-<job_id>.out
tail -c +0 -f slurm-<job_id>.out   # follow live output

# Cancel
scancel <job_id>
scancel -u $USER                   # all your jobs
scancel -p boost_usr_prod -t PD    # all pending in a partition

# Get an interactive shell on a node WHILE a job is running (for debugging)
srun --overlap --pty --jobid=<job_id> bash

# Inspect / control
scontrol show job <job_id>     # detailed job info
scontrol show node <name>      # node details
scontrol hold <job_id>         # prevent from starting
scontrol release <job_id>      # release a held job

# Cluster / partition state
sinfo -s                       # summary of node states
sinfo -N                       # per-node view
sinfo -p <partition>           # specific partition
```

### Interactive allocations (debug / testing)
```bash
# Allocate then run on compute nodes (preferred for multi-command sessions)
salloc -N 1 --ntasks-per-node=8
srun hostname        # runs on the compute node (plain commands run on login node!)
exit                 # end allocation

# Direct interactive shell on a compute node
srun -N 1 --ntasks-per-node=8 --pty /bin/bash    # add --overlap for extra srun calls inside
```
> Caveat: inside `salloc`, your prompt may still look like a login node. Any command **not** prefixed with `srun` runs on the login node, not the compute node.

---

## 13. Accounting / Budget

```bash
saldo -b            # show your project's budget / usage
saldo -b <user>     # budget associated with a specific user
```

---

## 14. Recommended Agent Workflow for Model Training

1. **Log in** to a login node (`ssh leonardo` after §3 config).
2. **Stage data** into `$SCRATCH` (download datasets/weights here — login nodes have internet). For very large/slow downloads or container conversion, use `srun --partition=lrd_all_serial ... --pty bash`.
3. **Prepare the environment** on the login node: Pixi project (§8.6), or a Singularity `.sif` (§9), or a Python venv (§8.4). Containers/Pixi are the most reproducible.
4. **Write `job.sh`** in `$HOME` using the matching template from §11 (start with 1 GPU + debug QoS / `s_tra_ncc` reservation, short `--time`).
5. **Submit** with `sbatch job.sh`; confirm with `squeue --me`.
6. **Monitor** with `tail -f slurm-<jobid>.out`; for live debugging, `srun --overlap --pty --jobid=<jobid> bash`.
7. **Scale up** to 2/4 GPUs (§11.4–11.5) once the 1-GPU run is verified. For multi-node, use `srun` to launch and drop the single-node hackathon reservation (§11.6).
8. **Checkpoint to `$SCRATCH`** regularly; remember files there are purged after 40 days, and walltime caps the run.

---

## 15. Gotchas & Hackathon-Specific Notes

- **[HACKATHON]** Reservation: add `#SBATCH --reservation=s_tra_ncc` for single-node GPU jobs. It only covers **1 node per team** — remove it for multi-node jobs.
- **[HACKATHON]** No 2FA; login nodes `login0{1,2,5,7}-ext.leonardo.cineca.it`.
- **[HACKATHON]** Storage: use `$SCRATCH` for big files, `$HOME` for scripts; do NOT use `$WORK` / `$FAST`.
- **Compute nodes have no internet.** Pre-download everything on login nodes; use the proxy (§7) only for small traffic.
- **Login nodes: 10-minute CPU-time limit.** Use `lrd_all_serial` interactive allocations for longer prep.
- **GPU jobs:** keep `--ntasks-per-node=1`; scale `--mem` (×120GB) and `--cpus-per-task` (×8) with `--gpus-per-task`. Use `--nv` with Singularity to expose GPUs.
- **Multi-node:** the program must be launched with `srun` (not a bare command) to span nodes.
- **Certificates expire** — the §3 SSH config auto-renews; otherwise re-run `step ssh login`.
- Hostnames are lowercase `leonardo.cineca.it` (the website sometimes capitalizes "LEONARDO" for emphasis).

---

## 16. References

- AI:AT HPC Onboarding Kit (Ch. 5 First steps on LEONARDO, Ch. 6 Software): https://ai-at.eu/hpc-onboarding/
- CINECA HPC docs: https://docs.hpc.cineca.it/
- step client install: https://smallstep.com/docs/step-ca/installation
- Singularity/Apptainer user guide: https://docs.sylabs.io/guides/latest/user-guide/
- Pixi: https://pixi.sh/
- Top500: https://www.top500.org/