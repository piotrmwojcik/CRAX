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
checkpoint is `/home/pwojcik/CRAX/models/safe_pathway_walker2d_Level_3_sac_seed0_20261030_141532_851541`.
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
