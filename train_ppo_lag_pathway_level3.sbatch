#!/bin/bash
# Baseline PPO-Lag Walker Pathway L3; no counterfactual relabelling.
# Submit: mkdir -p logs && sbatch train_ppo_lag_pathway_level3.sbatch
# Runs seed 3. Override with sbatch --array=1,2 ...
# Uses this repository's A100/Python module setup, not the example's Conda account.
#SBATCH --job-name=crax-ppolag-walker-l3
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

cd "$HOME/CRAX"
source .venv-py312/bin/activate

unset JAX_PLATFORMS
export MUJOCO_GL=egl
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"
export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

ppo_seed="${SLURM_ARRAY_TASK_ID:-3}"
echo "PPO-Lag baseline L3, seed=$ppo_seed, commit $(git rev-parse --short HEAD) on $(hostname)"
python -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices()), 'GPU not detected'"

# Match the reference PPO-Lag training arguments, using seed 3.
srun python -u -m training.train_env \
  --alg ppo_lag \
  --env_name safe_pathway_walker2d \
  --difficulty 3 \
  --safety_bound 25.0 \
  --lagrangian_coef_rate 3.0 \
  --learning_rate 5e-4 \
  --num_timesteps 500000000 \
  --seeds "$ppo_seed" \
  --num_evals 10 \
  --wandb_tags TRAIN \
  --skip_rollout \
  --video_fps 50
