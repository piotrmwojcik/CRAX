#!/bin/bash
#SBATCH --job-name=crax-saclag-walker
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

cd "$HOME/CRAX"
source .venv-py312/bin/activate

unset JAX_PLATFORMS
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

# Fail early if JAX cannot access the allocated GPU.
python -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices()), 'GPU not detected'"

# Use this checkout's default SAC-Lagrange training hyperparameters.
srun python -u -m training.train_env \
  --env_name safe_pathway_walker2d \
  --alg sac_lag \
  --difficulty 1 \
  --seeds 0 \
  --use_wandb false \
  --store_model true \
  --skip_rollout \
  --cameras fixedfar \
  --num_video_episodes 1
