#!/bin/bash
#SBATCH --job-name=crax-saclag-goal-point-l1
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

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

# SAC-Lagrangian on Safe Goal Point, difficulty/level 1.
# max_replay_size must be set: the SAC-Lag trainer cannot convert its CLI default None to int.
srun python -u -m training.train_env \
  --env_name safe_goal_point \
  --alg sac_lag \
  --difficulty 1 \
  --seeds 0 \
  --num_timesteps 100000000 \
  --num_envs 2048 \
  --batch_size 1024 \
  --num_eval_envs 128 \
  --max_replay_size 1000000 \
  --use_wandb true \
  --store_model true \
  --skip_rollout \
  --skip_video
