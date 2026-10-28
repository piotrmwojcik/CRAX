#!/bin/bash
#SBATCH --job-name=crax-saclag-walker-costcond
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=24:00:00
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

python -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices()), 'GPU not detected'"

# Match the Level 1 SAC-Lag baseline: 500M steps, seed 0, fixed reward.
# Bounds: [penetration exponent, hazard weight, terminal penalty].
# Nominal parameters are taken from the environment (defaults: [2, 1.5, 5]).
# Each replay sample selects nominal with probability 0.5, otherwise samples
# each parameter independently uniformly within its range.
# A bounded neural network learns lambda(parameters) with a combined budget.
# Collection and evaluation remain nominal; reward stays fixed.
# State observations and the trainer's action_repeat=1 default are required.
srun python -u -m training.train_env \
  --env_name safe_pathway_walker2d \
  --alg sac_lag \
  --difficulty 1 \
  --seeds 0 \
  --num_timesteps 500000000 \
  --num_envs 2048 \
  --batch_size 1024 \
  --grad_updates_per_step 32 \
  --max_replay_size 1000000 \
  --cost_conditioning true \
  --cost_parameter_bounds '[[1,4],[0.75,3],[2.5,10]]' \
  --cost_nominal_probability 0.5 \
  --lambda_network_lr 0.001 \
  --safety_bound 25 \
  --lagrangian_lr 0.01 \
  --initial_lambda 0 \
  --lambda_max 2 \
  --observed_episode_length_bound true \
  --episode_length_ema 0.9 \
  --wandb_group safe_pathway_walker2d_cost_uniform_lambda \
  --wandb_tags cost_conditioned uniform_cost learned_lambda sac_lag pathway_level1 \
  --use_wandb true \
  --store_model true \
  --skip_rollout \
  --skip_video
