#!/bin/bash
#SBATCH --job-name=crax-sac-walker-l3
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

cd "$HOME/CRAX"
source .venv-py312/bin/activate

unset JAX_PLATFORMS
export MUJOCO_GL=egl
export WANDB_BASE_URL=https://api.forge.coreweave.com
export PYTHONUNBUFFERED=1

# Fail early if JAX cannot access the allocated GPU.
python -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices()), 'GPU not detected'"

# Plain SAC baseline matching the Level 3 SAC-Lag training scale and resources.
# Reward-only optimization: no cost conditioning or Lagrange multiplier.
srun python -u -m training.train_env \
  --env_name safe_pathway_walker2d \
  --alg sac \
  --num_timesteps 500000000 \
  --num_envs 2048 \
  --batch_size 1024 \
  --grad_updates_per_step 32 \
  --max_replay_size 1000000 \
  --difficulty 3 \
  --seeds 0 \
  --use_wandb true \
  --store_model true \
  --skip_rollout \
  --skip_video
