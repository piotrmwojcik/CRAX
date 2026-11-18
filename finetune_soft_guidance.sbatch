#!/bin/bash
#SBATCH --job-name=crax-mmd-imitation
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
: "${DATASET_DIR:?Set DATASET_DIR to the completed fake-lidar dataset}"
checkpoint_args=()
if [[ -n "${PPO_CHECKPOINT:-}" ]]; then
  checkpoint_args=(--checkpoint "$PPO_CHECKPOINT")
fi
srun python -u -m training.agents.counterfactual.finetune_soft_guidance \
  --dataset "$DATASET_DIR" \
  --output "${OUTPUT_DIR:-results/soft_guidance_${SLURM_JOB_ID}}" \
  --iterations "${ITERATIONS:-100}" \
  --num-envs "${NUM_ENVS:-16}" \
  --learning-rate "${LEARNING_RATE:-0.0005}" \
  --seed "${SEED:-3}" \
  --num-evals "${NUM_EVALS:-10}" \
  --num-eval-envs "${NUM_EVAL_ENVS:-128}" \
  --video-every "${VIDEO_EVERY:-0}" \
  --video-seed "${VIDEO_SEED:-20000}" \
  --video-stride "${VIDEO_STRIDE:-4}" \
  --video-camera "${VIDEO_CAMERA:-fixedfar}" \
  --wandb-mode "${WANDB_MODE:-online}" \
  --wandb-project "${WANDB_PROJECT:-crax}" \
  --wandb-name "${WANDB_NAME:-mmd-imitation-${SLURM_JOB_ID}}" \
  "${checkpoint_args[@]}"
