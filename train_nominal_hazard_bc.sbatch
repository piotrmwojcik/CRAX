#!/bin/bash
#SBATCH --job-name=crax-nominal-hazard-bc
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Submit from the repository: mkdir -p logs && sbatch train_nominal_hazard_bc.sbatch
# Authenticate W&B on the login node first; do not put API keys in this file.
set -euo pipefail

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

cd "${SLURM_SUBMIT_DIR:?Submit this job from the CRAX repository}"
source .venv-py312/bin/activate

unset JAX_PLATFORMS
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Matches the existing cluster training jobs; override via sbatch --export.
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"
export WANDB_MODE=online

bc_checkpoint="${BC_CHECKPOINT:-models/safe_pathway_walker2d_Level_1_sac_lag_seed0_20261025_233216_014773/000500006912}"
bc_run_name="nominal-hazard-bc-${SLURM_JOB_ID}"
bc_output="results/${bc_run_name}"

test -f "${bc_checkpoint}/sac_network_config.json"
python -c "import jax, wandb; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices()), 'GPU not detected'"

srun python -u scripts/train_nominal_hazard_bc.py \
  "$bc_checkpoint" \
  --output "$bc_output" \
  --pairs 128 \
  --steps 2000 \
  --seed 15 \
  --hazard-seed 42 \
  --level 1 \
  --updates 100 \
  --batch-size 256 \
  --learning-rate 1e-4 \
  --kl-weight 1.0 \
  --teacher-momentum 0.99 \
  --reward-margin 1.0 \
  --cost-margin 0.5 \
  --eval-episodes 128 \
  --eval-every 25 \
  --eval-seed 10000 \
  --wandb-mode online \
  --wandb-project crax \
  --wandb-name "$bc_run_name"
