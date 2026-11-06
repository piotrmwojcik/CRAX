#!/bin/bash
# Run on Snellius: bash /home/pwojcik/CRAX/scripts/submit_ppo_lag_sweep.sh
set -euo pipefail

algos=("ppo_lag")
envs=("safe_pathway_walker2d")
learning_rates=(5e-4)
safety_bounds=(25.0)
lambdas=(3.0)
seeds=(1 2 3)
levels=(3)

ppo_repo="${CRAX_DIR:-/home/pwojcik/CRAX}"
ppo_partition="${PARTITION:-gpu_a100}"
ppo_time_limit="${TIME_LIMIT:-3:00:00}"
mkdir -p "$ppo_repo/logs"

for alg in "${algos[@]}"; do
  for env_name in "${envs[@]}"; do
    for safety_bound in "${safety_bounds[@]}"; do
      for lr in "${learning_rates[@]}"; do
        for lambda in "${lambdas[@]}"; do
          for level in "${levels[@]}"; do
            for seed in "${seeds[@]}"; do
              echo "Submitting $alg: env=$env_name level=$level bound=$safety_bound lambda=$lambda lr=$lr seed=$seed"
              sbatch \
                --partition="$ppo_partition" \
                --nodes=1 --ntasks=1 --cpus-per-task=18 --gpus=1 \
                --time="$ppo_time_limit" \
                --job-name="crax-${alg}-l${level}-s${seed}" \
                --output="$ppo_repo/logs/%x-%j.out" \
                --error="$ppo_repo/logs/%x-%j.err" \
                /dev/stdin "$ppo_repo" "$alg" "$env_name" "$safety_bound" \
                "$lambda" "$lr" "$level" "$seed" <<'SBATCH_SCRIPT'
#!/bin/bash
set -euo pipefail

module load 2024
module load Python/3.12.3-GCCcore-13.3.0

cd "$1"
source .venv-py312/bin/activate

unset JAX_PLATFORMS
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export JAX_TRACEBACK_FILTERING=off
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"

python -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices()), 'GPU not detected'"

srun python -u -m training.train_env \
  --alg "$2" \
  --env_name "$3" \
  --safety_bound "$4" \
  --lagrangian_coef_rate "$5" \
  --learning_rate "$6" \
  --difficulty "$7" \
  --seeds "$8" \
  --skip_rollout \
  --video_fps 50 \
  --num_evals 10 \
  --num_timesteps 5e8 \
  --use_wandb true \
  --wandb_tags TRAIN
SBATCH_SCRIPT
              sleep 0.1
            done
          done
        done
      done
    done
  done
done
