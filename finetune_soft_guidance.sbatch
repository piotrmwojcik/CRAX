#!/bin/bash
#SBATCH --job-name=crax-soft-guidance
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
  --initial-tolerance "${INITIAL_TOLERANCE:-0.05}" \
  --annealing "${ANNEALING:-0.01}" \
  --max-kl "${MAX_KL:-0.01}" \
  --seed "${SEED:-0}" \
  --eval-every "${EVAL_EVERY:-10}" \
  --eval-episodes "${EVAL_EPISODES:-64}" \
  --eval-seed "${EVAL_SEED:-10000}" \
  --final-eval-episodes "${FINAL_EVAL_EPISODES:-512}" \
  --final-eval-seed "${FINAL_EVAL_SEED:-60000}" \
  --wandb-mode "${WANDB_MODE:-online}" \
  --wandb-project "${WANDB_PROJECT:-crax}" \
  --wandb-name "${WANDB_NAME:-soft-guidance-${SLURM_JOB_ID}}" \
  "${checkpoint_args[@]}"
