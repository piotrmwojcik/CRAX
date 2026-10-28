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
# Triples: [penetration exponent, hazard weight, terminal penalty].
# Nominal parameters are taken from the environment (defaults: [2, 1.5, 5]).
# Each replay sample selects nominal with probability 0.5, otherwise one of
# the six alternatives uniformly. Collection and evaluation remain nominal.
# Each configuration has its own lambda and the same combined cost budget.
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
  --cost_configurations '[[1,1.5,5],[4,1.5,5],[2,0.75,5],[2,3,5],[2,1.5,2.5],[2,1.5,10]]' \
  --cost_nominal_probability 0.5 \
  --safety_bound 25 \
  --lagrangian_lr 0.01 \
  --initial_lambda 0 \
  --lambda_max 2 \
  --observed_episode_length_bound true \
  --episode_length_ema 0.9 \
  --wandb_group safe_pathway_walker2d_cost_conditioned \
  --wandb_tags cost_conditioned sac_lag pathway_level1 \
  --use_wandb true \
  --store_model true \
  --skip_rollout \
  --skip_video
