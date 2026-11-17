#!/bin/bash
# Submit from CRAX: mkdir -p logs && sbatch collect_counterfactual_ppo_lag.sbatch
# Override checkpoint/batch count using PPO_CHECKPOINT and NUM_BATCHES.
# Default stages: caps 1, 2, 3; two generations each. STAGED_COLLECTION=0 disables staging.
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

# A run directory selects its highest numeric checkpoint step.
ppo_checkpoint="${PPO_CHECKPOINT:-/home/pwojcik/CRAX/models/safe_pathway_walker2d_Level_3_ppo_lag_seed4_20261116_223426_688934}"
ppo_output="${OUTPUT_DIR:-results/counterfactual_ppo_collection_${SLURM_JOB_ID}}"
# DATASET_DIR selects an existing completed collection and skips collection.
# LIDAR_OUTPUT_DIR selects the new augmented dataset directory.
if [[ -n "${DATASET_DIR:-}" ]]; then
  ppo_output="$DATASET_DIR"
elif [[ "${SKIP_COLLECTION:-0}" == "1" ]]; then
  : "${OUTPUT_DIR:?Set DATASET_DIR or OUTPUT_DIR when skipping collection}"
else
python -c "import jax; print(jax.devices()); assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu', 'One GPU required'"

stage_args=()
starting_hazards="${NUM_ADDED_HAZARDS:-3}"
if [[ "${STAGED_COLLECTION:-1}" == "1" ]]; then
  starting_hazards="${NUM_ADDED_HAZARDS:-1}"
  stage_args=(--max-added-hazards "${MAX_ADDED_HAZARDS:-3}"
              --generations-per-stage "${GENERATIONS_PER_STAGE:-2}"
              --hazard-increment "${HAZARD_INCREMENT:-1}")
fi

srun python -u -m training.agents.counterfactual.collect_rollouts \
  --checkpoint "$ppo_checkpoint" \
  --output "$ppo_output" \
  --env safe_pathway_walker2d \
  --level "${LEVEL:-3}" \
  --seed "${SEED:-4}" \
  --num-envs "${NUM_ENVS:-2048}" \
  --episode-length 2000 \
  --action-repeat 1 \
  --unroll-length "${UNROLL_LENGTH:-8}" \
  --batch-size "${BATCH_SIZE:-1024}" \
  --num-minibatches "${NUM_MINIBATCHES:-32}" \
  --num-batches "${NUM_BATCHES:-1}" \
  --num-added-hazards "$starting_hazards" \
  "${stage_args[@]}" \
  --hazard-seed "${HAZARD_SEED:-42}" \
  --comparison-batch-size "${COMPARISON_BATCH_SIZE:-16}" \
  --platform-half-width "${PLATFORM_HALF_WIDTH:-1.0}"
fi

# Read each saved generation's cap and seed, including when DATASET_DIR skips collection.
python - "$ppo_output" "${LIDAR_OUTPUT_DIR:-${ppo_output}_fake_lidar_${SLURM_JOB_ID}}" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

source, destination = map(Path, sys.argv[1:])
index = source / 'staged_collection.json'
if index.is_file():
    report = json.loads(index.read_text())
    if report['status'] != 'complete':
        raise ValueError('Staged collection must be complete before lidar editing')
    datasets = [Path(generation['dataset']) for stage in report['stages']
                for generation in stage['generations']]
else:
    datasets = [Path('.')]

# Validate all generations before starting post-processing.
commands = []
for relative in datasets:
    dataset = source / relative
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError(f'Incomplete dataset: {dataset}')
    config = manifest['configuration']
    output = destination / relative
    if output.exists():
        raise FileExistsError(f'Use a fresh LIDAR_OUTPUT_DIR: {output}')
    commands.append([
        'srun', sys.executable, '-u', '-m', 'training.agents.counterfactual.edit_lidar',
        '--dataset', str(dataset), '--output', str(output),
        '--num-added-hazards', str(config['num_added_hazards']),
        '--hazard-seed', str(config['hazard_seed']),
    ])
for command in commands:
    subprocess.run(command, check=True)
PY
