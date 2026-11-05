#!/bin/bash
# Baseline PPO-Lag Walker Pathway L3; no counterfactual relabelling.
# Submit: mkdir -p logs && sbatch train_ppo_lag_pathway_level3.sbatch
# Seeds 1 and 2 run as separate jobs. Override with sbatch --array=3,4 ...
# Uses this repository's A100/Python module setup, not the example's Conda account.
#SBATCH --job-name=crax-ppolag-walker-l3
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --array=1-2
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
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

ppo_seed="${SLURM_ARRAY_TASK_ID:-1}"
echo "PPO-Lag baseline L3, seed=$ppo_seed, commit $(git rev-parse --short HEAD) on $(hostname)"
python -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices()), 'GPU not detected'"

# Local trainer has no counterfactual CLI options: ordinary PPO-Lag collection.
# Fixed per-step safety bound: 25 / 2000. No separate evaluation passes.
srun python -u -m training.train_env \
  --alg ppo_lag \
  --env_name safe_pathway_walker2d \
  --difficulty 3 \
  --safety_bound 25.0 \
  --observed_episode_length_bound false \
  --episode_length 2000 \
  --num_timesteps 500000000 \
  --num_envs 2048 \
  --unroll_length 8 \
  --batch_size 1024 \
  --num_minibatches 32 \
  --num_updates_per_batch 6 \
  --seeds "$ppo_seed" \
  --num_evals 0 \
  --use_wandb true \
  --store_model true \
  --wandb_group ppo_lag_pathway_l3_baseline \
  --wandb_tags BASELINE NO_RELABELLING BOUND_false \
  --skip_rollout \
  --skip_video
