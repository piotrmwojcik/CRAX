#!/bin/bash
# Submit from CRAX: mkdir -p logs && sbatch collect_counterfactual_ppo_lag.sbatch
# Override checkpoint/batch count using PPO_CHECKPOINT and NUM_BATCHES.
#SBATCH --job-name=crax-ppo-collect
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
cd /home/pwojcik/CRAX
source .venv-py312/bin/activate
unset JAX_PLATFORMS
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

ppo_checkpoint="${PPO_CHECKPOINT:-/home/pwojcik/CRAX/models/safe_pathway_walker2d_Level_3_ppo_lag_seed3_20261106_154130_889149/000500170752}"
ppo_output="${OUTPUT_DIR:-results/counterfactual_ppo_collection_${SLURM_JOB_ID}}"
# DATASET_DIR selects an existing completed collection and skips collection.
# LIDAR_OUTPUT_DIR selects the new augmented dataset directory.
if [[ -n "${DATASET_DIR:-}" ]]; then
  ppo_output="$DATASET_DIR"
elif [[ "${SKIP_COLLECTION:-0}" == "1" ]]; then
  : "${OUTPUT_DIR:?Set DATASET_DIR or OUTPUT_DIR when skipping collection}"
else
python -c "import jax; print(jax.devices()); assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu', 'One GPU required'"

srun python -u -m training.agents.counterfactual.collect_rollouts \
  --checkpoint "$ppo_checkpoint" \
  --output "$ppo_output" \
  --env safe_pathway_walker2d \
  --level "${LEVEL:-3}" \
  --seed "${SEED:-1}" \
  --num-envs 2048 \
  --episode-length 2000 \
  --action-repeat 1 \
  --unroll-length 8 \
  --batch-size 1024 \
  --num-minibatches 32 \
  --num-batches "${NUM_BATCHES:-1}" \
  --num-added-hazards "${NUM_ADDED_HAZARDS:-3}" \
  --hazard-seed "${HAZARD_SEED:-42}" \
  --platform-half-width "${PLATFORM_HALF_WIDTH:-1.0}"
fi

srun python -u -m training.agents.counterfactual.edit_lidar \
  --dataset "$ppo_output" \
  --output "${LIDAR_OUTPUT_DIR:-${ppo_output}_fake_lidar_${SLURM_JOB_ID}}" \
  --num-added-hazards "${NUM_ADDED_HAZARDS:-3}" \
  --hazard-seed "${HAZARD_SEED:-42}"
