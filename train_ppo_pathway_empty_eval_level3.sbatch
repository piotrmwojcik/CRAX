#!/bin/bash
#SBATCH --job-name=crax-ppo-empty-eval-l3
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

python -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices()), 'GPU not detected'"

srun python -u -m training.train_env \
  --alg ppo \
  --env_name safe_pathway_walker2d \
  --difficulty 3 \
  --train_env_kwargs '{"num_hazards":0}' \
  --num_timesteps 500000000 \
  --num_envs 2048 \
  --batch_size 1024 \
  --unroll_length 8 \
  --num_minibatches 32 \
  --num_updates_per_batch 6 \
  --observed_episode_length_bound true \
  --episode_length_ema 0.9 \
  --safety_bound 25 \
  --num_evals 5 \
  --num_eval_envs 128 \
  --seeds 0 \
  --use_wandb true \
  --wandb_tags empty-train-level3-eval \
  --store_model true \
  --skip_rollout \
  --skip_video
