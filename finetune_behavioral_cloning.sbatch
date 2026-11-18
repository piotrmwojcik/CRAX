#!/bin/bash
#SBATCH --job-name=crax-behavioral-cloning
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
cd "${SLURM_SUBMIT_DIR:?Submit from the CRAX repository}"
source .venv-py312/bin/activate
unset JAX_PLATFORMS
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
DATASET_DIR="${DATASET_DIR:-results/counterfactual_ppo_collection_29375024_fake_lidar_29375024/hazards_013/generation_000}"
run_args=()
if [[ -n "${PPO_CHECKPOINT:-}" ]]; then
  run_args=(--checkpoint "$PPO_CHECKPOINT")
fi
if [[ -n "${SEED:-}" ]]; then
  run_args+=(--seed "$SEED")
fi
srun python -u -m training.agents.counterfactual.finetune_soft_guidance \
  --method cloning \
  --dataset "$DATASET_DIR" \
  --output "${OUTPUT_DIR:-results/behavioral_cloning_${SLURM_JOB_ID}}" \
  --iterations "${ITERATIONS:-100}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --learning-rate "${LEARNING_RATE:-0.0001}" \
  --num-evals "${NUM_EVALS:-10}" \
  --num-eval-envs "${NUM_EVAL_ENVS:-128}" \
  --video-every "${VIDEO_EVERY:-0}" \
  --video-seed "${VIDEO_SEED:-20000}" \
  --video-stride "${VIDEO_STRIDE:-4}" \
  --video-camera "${VIDEO_CAMERA:-fixedfar}" \
  --wandb-mode "${WANDB_MODE:-online}" \
  --wandb-project "${WANDB_PROJECT:-crax}" \
  --wandb-name "${WANDB_NAME:-behavioral-cloning-${SLURM_JOB_ID}}" \
  "${run_args[@]}"
