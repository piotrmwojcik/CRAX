# PPO checkpoint rollout collector

`collect_rollouts.py` first collects stochastic PPO-style batches from a
PPO/PPO-Lag or plain SAC checkpoint, then compares each completed episode against the same actor
with added real hazards. No policy updates or imitation are performed.
The checkpoint actor and observation normalizer remain fixed. Collection uses
the same `training.acting.generate_unroll`, environment wrappers, RNG splitting
within each batch, and reshape order as `training/agents/ppo/train.py`.

Default settings match the single-GPU PPO sbatch: 2,048 environments, fragment
length 8, batch size 1,024, and 32 minibatches. One collection batch contains
32,768 fragments / 262,144 transitions. `--num-batches` controls the total
initial collection size; optimization epochs are not applicable. After those
batches, collection continues until every pending episode ends. Finished lanes
are frozen, and no new episode is collected during this completion phase.
The final transition count can therefore exceed the requested batch budget.
Environment state and
action RNG continue across batches. Completed episodes automatically return to
their original sampled board, exactly as the current PPO wrappers do.

The first batch uses PPO's single-device initialization and first-epoch seed
scheme. Later batches keep that epoch's RNG stream; there are no evaluation
epoch boundaries. This collector currently supports one device on one host and
the ordinary state-observation environment path, not pixel/domain-randomized
training. A fixed policy will not reproduce later trajectories of a PPO run
whose actor and normalizer are being updated.

## HPC

From `/home/pwojcik/CRAX`:

```bash
mkdir -p logs
NUM_BATCHES=10 NUM_ADDED_HAZARDS=4 sbatch collect_counterfactual_ppo_lag.sbatch
```

Override `PPO_CHECKPOINT`, `LEVEL`, `SEED`, `OUTPUT_DIR`, `HAZARD_SEED`, or
`PLATFORM_HALF_WIDTH` as needed. The default
checkpoint is `/home/pwojcik/CRAX/models/safe_pathway_walker2d_Level_3_ppo_lag_seed3_20261106_154130_889149/000500170752`.
Run directories resolve to the latest numeric checkpoint. SAC uses its original
stochastic action distribution, with raw actions and log probabilities additionally
recorded for paired comparisons. Collection still uses PPO's batching and wrappers;
no SAC replay-buffer training is performed. `PPO_CHECKPOINT` remains the override
variable for compatibility, and accepts either supported checkpoint family.

## Small local collection

```bash
MUJOCO_GL=cgl JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python \
  -m training.agents.counterfactual.collect_rollouts \
  --checkpoint models/safe_pathway_walker2d_Level_3_ppo_lag_seed1_20261105_122816_749734 \
  --output results/ppo_collection_local \
  --level 3 --seed 1 --num-envs 4 --batch-size 4 --num-minibatches 2 \
  --unroll-length 8 --num-batches 2 --num-added-hazards 4 --hazard-seed 42
```

## Output

Each `batch_XXXXXX.npz` contains the full PPO `Transition` fields with shape
`[fragment, time, ...]`. Keys use slash-separated field names:
`observation`, `action`, `reward`, `discount`, `next_observation`,
`extras/policy_extras/log_prob`, `extras/policy_extras/raw_action`, and
`extras/state_extras/{cost,truncation,episode_done,episode_metrics/...}`.
Load with `np.load(path, allow_pickle=False)`.

Completion transitions are saved in `finish_XXXXXX.npz`, with shape
`[num_envs, unroll_length, ...]`. Every file contains a boolean `valid` array:
all regular-batch entries are valid, while completion files include zero-filled
padding for finished environments. Only entries where `valid` is true belong
to the dataset. Read files in manifest order and use terminal flags to assemble
complete episodes per environment. Final chunks can be shorter than an unroll.
The manifest reports actual valid transition counts and zero pending episodes
when successfully complete. Natural termination and time-limit truncation both
finish an episode; a run interrupted before completion can still contain partial
episodes and will not have status `complete`.

Fragment index is `scan_index * num_envs + env_index`. A fragment can cross an
episode boundary. Terminal `next_observation` is the auto-reset observation;
`discount=0` marks the boundary. For hazard comparisons, the batches additionally
store true successor `q`, `qd`, and link positions in
`extras/state_extras/counterfactual_state/*`, captured before auto-reset.
Initial physical states and original board positions are in `initial_states.npz`.
The manifest records the resolved checkpoint,
settings, file layout, counts, timings, and completed-episode metrics from the
existing PPO wrapper. A new output directory is required for every run.

## Hazard comparison phase

After every started nominal episode finishes, the program prints
`PHASE 1 COMPLETE` with a UTC timestamp, duration, episode count, and transition
count, and saves `collection_status=complete` in the manifest. It then reads the
stored fragments and reconstructs complete episodes per environment.

`--num-added-hazards N` places N simultaneous rectangular hazards in the widest
distinct safe gaps, using the same `choose_hazard` helper as
`scripts/compare_fake_real_hazard.py`. Each fills its safe gap. The nominal
trajectory remains unchanged; its added-hazard penetration must be zero. Episodes
with insufficient safe gaps are logged as skipped. No videos are rendered.

The real rollout starts from the same nominal initial pose, velocity, and
original board. All new rectangles are visible to lidar and contribute to the
normal environment cost. Their widths are dynamic inputs to a compiled rollout,
so placement changes do not require recompiling each episode. Real rollouts run
to termination or the configured time limit, even if nominal ended earlier.

Stochastic actions use the nominal trajectory's recovered Gaussian innovations
with the real policy's own means/scales (not replayed nominal actions). If the
real episode lasts longer, remaining innovations use a deterministic independent
extension keyed by collection seed, environment index, and episode index.

`hazard_comparisons.jsonl` is flushed after every episode, with nominal and real
reward, total cost, steps, added-hazard cost, cost excluding new hazards, placement,
and winner (`nominal`, `real`, or `none`; strict reward-and-cost dominance).
Successful real trajectories are saved in `real_rollouts/pair_XXXXXX.npz`.
`hazard_summary.json` contains means over compared pairs, skip counts, and winner
counts. The final manifest is marked complete only after both phases finish.

Use `--num-added-hazards 0` for collection only. The comparison phase currently
supports state-based Walker2d policies and action repeat 1. Physical recording
adds disk and memory use; each shard is processed once and completed episodes are
released as they are compared. Pre-existing datasets without recorded physics
cannot be used for this phase.

### Nominal demonstrations with fake-hazard lidar

Post-process a completed collection without loading a policy or stepping physics:

```bash
python -m training.agents.counterfactual.edit_lidar \
  --dataset results/existing_collection \
  --output results/existing_collection_fake_lidar \
  --num-added-hazards 3 --hazard-seed 42
```

The input must have `collection_status: complete` and recorded physics in
`initial_states.npz` and the transition batches. It can come from a run whose
real-hazard comparison failed after collection completed. The output must be a
new directory. Original batch shapes and all fields are preserved, except the
lidar slices of `observation` and `next_observation`. Actions, policy log
probabilities, rewards, costs, and physics remain those of the original rollout;
these are imperfect expert demonstrations, not freshly sampled on-policy data.
The initial-state file remains an unchanged record of the original physics.

The editor selects up to the requested number of widest safe gaps per episode,
using the same placement seed convention as hazard comparison. It uses the
actual environment observation function, including nearest-hazard selection and
lidar aliasing. Lidar detects hazard centers, as in the real environment; widths
control placement safety but do not change lidar. With no safe gap, the episode
is retained unchanged. `fake_hazards.jsonl` records placements or the reason no
placement was possible. Observation physics is aligned across file boundaries;
terminal next observations retain the collector's auto-reset semantics using
the ending episode's fake hazards.

The collection sbatch runs this edit after a successful collection/comparison.
To skip collection and edit an existing dataset:

```bash
sbatch --export=ALL,DATASET_DIR=results/existing_collection,LIDAR_OUTPUT_DIR=results/existing_collection_fake_lidar collect_counterfactual_ppo_lag.sbatch
```

Alternatively use `SKIP_COLLECTION=1,OUTPUT_DIR=results/existing_collection`.
`NUM_ADDED_HAZARDS` defaults to 3 and `HAZARD_SEED` to 42. Existing-dataset mode
uses the environment and level from the source manifest and requires no checkpoint.

## Fine-tuning with imperfect nominal demonstrations

`finetune_soft_guidance.py` implements the constrained policy update in
[Jing et al., AAAI 2020](https://arxiv.org/abs/1911.07109), Algorithm 1 and
Eqs. (3)--(9). All nominal episodes in the completed lidar-edited dataset are
eligible demonstrations, including unchanged episodes with no feasible fake
hazards. There is no winner filter, pairwise preference loss, behavior-cloning
loss, or use of archived real rollouts as on-policy experience.

```bash
python -m training.agents.counterfactual.finetune_soft_guidance \
  --dataset results/counterfactual_ppo_collection_28964946_fake_lidar \
  --output results/soft_guidance_trial \
  --iterations 100 --num-envs 16
```

The checkpoint defaults to `manifest.json`'s source checkpoint. Supply
`--checkpoint /path/to/checkpoint` if that path moved. State-based PPO/PPO-Lag
and plain SAC checkpoints supported by the collector are accepted. The actor
is updated, the observation normalizer and any other checkpoint parameters
remain frozen. Saved files preserve the original network configuration and
checkpoint tuple layout and can be loaded by the existing policy loaders.
This is an actor fine-tune, not a resumption of PPO/SAC optimizer state.

Submit from CRAX (create `logs` first):

```bash
sbatch --export=ALL,DATASET_DIR=/gpfs/home2/pwojcik/CRAX/results/counterfactual_ppo_collection_28964946_fake_lidar \
  finetune_soft_guidance.sbatch
```

### Objective and algorithm

Each iteration collects fresh stochastic episodes in the standard real-hazard
`safe_pathway_walker2d` environment at the manifest's level. It does not replay
the original paired boards or add their episode-specific rectangles online.
Only the first episode per lane contributes; terminal transitions are included
and all post-termination samples are masked. The default horizon comes from the
dataset. Native environment reward is the objective. Costs are logged but are
**not constrained**: this implements soft expert guidance, not PPO-Lag's safety
constraint, and does not promise lower cost.

The update maximizes a local reward surrogate subject to an expert discrepancy
bound and an old-to-new policy KL bound. It solves the local quadratic problem
in the span of the two inverse-Fisher gradients (equivalent to the dual solution),
uses conjugate gradients for Fisher-vector products, and backtracks to check
empirical discrepancy and KL. An infeasible local problem uses a discrepancy
recovery direction. Rejected proposals leave the actor unchanged. The expert
tolerance grows as `d *= 1 + annealing` each iteration.

### Explicit estimator choices

This is a new implementation of the method, not a reproduction of the authors'
experimental code. The paper leaves sampling/estimation details unspecified.
Our choices are:

- Squared, biased empirical Gaussian-kernel MMD on normalized discounted
  state-action occupancies. `--initial-tolerance` is in **MMD squared** units.
- Expert features are the saved edited observation and saved bounded action.
  Original policy log probabilities/raw actions are not used as likelihoods
  for these edited demonstrations. Saved rewards are not needed for guidance.
- A bounded expert reservoir samples with weights `gamma**t`; uniform sampling
  from that reservoir approximates the expert occupancy. Feature scaling is
  frozen from this reservoir. Kernel width defaults to its median nonzero
  pairwise distance; `--bandwidth` overrides it.
- For a candidate actor, online MMD sample weights use the product of action
  likelihood ratios from the start of an episode through each selected state
  and action. This includes the effect of preceding actions on occupancy;
  multiplying only the current action ratio would omit that effect. We use
  normalized weights and a fixed sample set throughout each line search.
- Reward gradients use discounted Monte Carlo returns, a leave-one-out
  timestep baseline, and a likelihood-ratio surrogate. No critic is trained.
- KL is analytic for the pre-tanh diagonal Gaussians (equivalently KL after
  their common invertible tanh transformation), measured on sampled rollout
  states. Fisher products include damping.

MMD and KL checks are finite-sample local estimates, not guarantees on the true
occupancy. Long trajectory importance weights can have high variance. Tune
`--mmd-samples`, `--max-kl`, `--initial-tolerance`, and `--annealing` using held-out
evaluation. The standard-environment rollouts logged each iteration are training
data, not held-out evaluation or measurements on the original paired tasks.

Outputs: `run.json` records configuration and estimator semantics;
`mmd_features.npz` records frozen scaling/bandwidth; `metrics.jsonl` records
reward, cost, valid length, MMD, tolerance, KL, recovery mode and accepted step;
`checkpoint/<iteration>` stores loadable actor snapshots. Defaults save every
10 iterations and at the end. Failed runs keep their prior snapshots; automatic
resume of optimizer/RNG state is not implemented.

### Evaluation and W&B

Fine-tuning now evaluates the initial actor (iteration 0), every `--eval-every`
iterations (default 10), and at the final iteration. Each evaluation runs
`--eval-episodes` stochastic episodes (default 64) with fixed independent reset
and action randomness controlled by `--eval-seed` (default 10000). These rollouts
use the standard real-hazard environment at the dataset level, not archived
paired boards; they are excluded from the training gradients and expert pool.
Reusing the evaluation seeds supports before/after comparisons.

W&B logs `eval/reward`, `eval/cost`, `eval/episode_length`, their population
standard deviations, `eval/zero_cost_fraction`, episode count and evaluation
wall time against `iteration`. Reward and cost are undiscounted episode totals,
averaged over episodes, including the terminal transition. Training diagnostics
use the separate `train/*` namespace. The latest evaluation and checkpoint path
are recorded in the run summary. All evaluations also persist to
`evaluation.jsonl`, including with W&B disabled. The W&B run is finalized by a
context manager on normal exit or an exception.

The sbatch defaults to W&B online in project `crax`, with run name
`soft-guidance-<job_id>`. Authenticate on the cluster beforehand using the
existing W&B setup. Override `WANDB_MODE`, `WANDB_PROJECT`, `WANDB_NAME`,
`EVAL_EVERY`, `EVAL_EPISODES`, or `EVAL_SEED` via sbatch environment variables.
The standard W&B `WANDB_ENTITY` and `WANDB_BASE_URL` settings are respected.
Direct Python usage defaults to `--wandb-mode disabled`; use `--wandb-mode
online --wandb-project crax` to enable uploading, or `offline` for local W&B logs.

### Episode records, comparison bars, and final independent evaluation

Each periodic evaluation now saves `evaluation/<iteration>/episodes.csv` with
matched before/after reward, cost, and length for every episode. The initial
actor's results are the baseline. Each directory also contains `summary.json`,
`comparison.png`, and `comparison.pdf`. Summary intervals are 95% percentile
bootstrap intervals from resampling matched episode pairs (2,000 resamples by
default). Bar whiskers show uncertainty in mean episode reward/cost; the chart
titles show paired changes and their intervals. These quantify episode sampling
uncertainty for this run, not variability between independent training runs.

W&B receives `eval/comparison` images and `eval/episode_results` tables at every
evaluation, along with before/after/change means and CI endpoints. Aggregate
curves remain available. Set `--eval-cost-limit` to show a specified episode-cost
budget; no threshold is invented by default.

After the last update, the original and final actors are evaluated on a separate
set of 512 episodes with seed 60000, used only at the end. Results, intervals,
and charts are saved under `independent_evaluation/` and uploaded under `test/*`.
This compares the final actor, not the best actor selected from periodic scores.
Configure via `--final-eval-episodes`, `--final-eval-seed`, and
`--eval-bootstrap-samples`. Sbatch exposes `FINAL_EVAL_EPISODES` and
`FINAL_EVAL_SEED`. Evaluation runs in batches of at most 32 environments to bound
memory. Its key derivation has changed from the previous all-at-once evaluator,
so new periodic curves use a different, consistently paired sample set.
