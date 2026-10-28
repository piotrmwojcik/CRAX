# Cost-conditioned SAC-Lagrangian

This opt-in Safe Pathway extension adapts the replay relabeling and actor/critic
conditioning of [Reward-Conditioned Reinforcement Learning, §3](https://arxiv.org/abs/2603.05066)
to **costs**. The environment reward, reward scaling, dynamics, termination rules,
and nominal data-collection policy specification remain fixed. This is an
experimental constrained-RL adaptation, not a reproduction of the paper's reward
experiments or a claim of its reported performance gains.

## Cost definition

Each configuration is a triple `(penetration_exponent, hazard_weight, terminal_penalty)`:

```text
cost = hazard_weight * sum_h(penetration[h] ** penetration_exponent)
       + terminal_penalty * unhealthy_termination
```

Penetration is zero at/outside the hazard boundary and approximately one at its
center. Only feet passing the existing ground-height check contribute. Circular
hazards use radius-normalized distance; rectangles use the maximum distance along
either axis normalized by half-extents. We take the largest penetration among
feet for each hazard, preserving the original max-over-feet semantics. An exponent
of 1 penalizes shallow penetration more than the nominal exponent of 2; an
exponent of 4 concentrates cost nearer the center. Exponents must be positive;
weights can be zero but not negative. Ground-height thresholds, hazard geometry,
and separate left/right foot weights are not configurable by this implementation.

Replay stores the per-hazard penetration vector and the unhealthy-termination
indicator in addition to the original scalar cost. A time-limit truncation alone
does not set the unhealthy-termination indicator. Reward is not recomputed.

The nominal triple is derived from the environment: exponent 2, its cost scaler,
and its unhealthy-termination penalty. With the class defaults it is `(2, 1.5, 5)`;
environment overrides are respected. By default, six alternatives independently
halve or double each of the three nominal parameters (duplicates are removed).

## Training

Add `--cost_conditioning` to a Safe Pathway SAC-Lag command:

```bash
python -m training.train_env \
  --env_name safe_pathway_walker2d --alg sac_lag \
  --cost_conditioning \
  --cost_configurations '[[1,1.5,5],[4,1.5,5],[2,3,5],[2,1.5,10]]' \
  --cost_nominal_probability 0.5 \
  --max_replay_size 1000000 \
  --safety_bound 25
```

Omit `--cost_configurations` to use the default alternatives. Configuration zero
is always the nominal environment cost. Each gradient minibatch independently
assigns a configuration to every transition: nominal with probability 0.5 by
default, otherwise uniform over the alternatives. The current and next state
receive the **same** parameters. The policy, reward critic, cost critic, and
entropy-temperature loss all use these conditioned observations. Physical state
normalization excludes the cost parameters; their network features are `log1p`
of the three values, which also supports zero weights.

The actor uses the multiplier belonging to each sampled configuration:

```text
actor_loss = mean(alpha * log_probability - Q_reward(state, action, parameters)
                  + lambda[configuration] * Q_cost(state, action, parameters))
```

This implements one combined hazard-plus-terminal constraint per configuration,
not independent hazard and falling budgets. Every configuration uses the same
numeric `--safety_bound` in its own cost units. Thus changing weights with the
budget fixed changes the constraint, rather than merely rescaling its units.

Each multiplier retains the existing SAC-Lag **replay-average immediate-cost**
dual update, evaluated separately for every configuration on the original batch.
These means are averaged across devices before updating the multipliers. The
effective learning rate is `lagrangian_lr / episode_length`; bounds use either
the configured episode length or the nominal policy's observed-length EMA.

**Limitation:** counterfactual replay means describe the nominal behavior-data
distribution. They are not measured episodic costs of the alternative conditioned
policies. Neither this dual estimator nor relabeling guarantees that alternative
policies meet their budgets. Evaluate reward, termination rate, and the correctly
recomputed episodic cost under each deployed configuration before interpreting
the results as constraint satisfaction. The nominal-policy episode-length EMA
also need not match alternative policies' episode lengths.

Existing `lambda_lagr`, `mean_cost`, `cost_violation`, and evaluation metrics refer
to the nominal configuration. Additional training metrics under
`cost_configuration_<index>/` report each configuration's multiplier, replay mean
cost, and violation; `relabelled_mean_cost` reports the sampled training mixture.
Built-in environment evaluations use the nominal configuration only.

## Inference and checkpoints

Checkpoints include the nominal parameters, full configuration bank, multiplier
values, and conditioning settings. The existing SAC checkpoint loader detects
conditioned checkpoints automatically:

```python
from training.agents.sac.checkpoint import load_policy

# Nominal configuration is the default.
policy = load_policy("models/run/000500000000", deterministic=True)

# Choose a trained alternative without changing the environment reward.
policy = load_policy(
    "models/run/000500000000",
    deterministic=True,
    cost_parameters=(1.0, 1.5, 5.0),
)
action, extras = policy(observation, key)
```

The trainer's returned `make_policy` also accepts `cost_parameters`. Evaluating a
different policy condition does not change `env.info['cost']`: it remains nominal.
Use `training.agents.sac_lag.cost_conditioning.reconstruct_cost` with the returned
`info['hazard_penetration']`, `info['unhealthy_termination']`, and chosen triple to
measure that configuration's actual cost. Unseen triples are accepted for research
evaluation, but their behavior is unvalidated generalization. Existing checkpoints
without conditioning cannot acquire it at load time; train a new conditioned model.
As before, checkpoints contain inference parameters, not a resumable full learner.

## Scope and checks

Supported only by `sac_lag` on Safe Pathway with flat state observations and
`action_repeat=1`. Vision observations, custom network factories, custom multiplier
updates, and other environments are rejected. Other algorithms do not receive
cost-conditioned replay. Without the flag, the original scalar-cost SAC-Lag path
is used.

```bash
python -m unittest crax.envs.safe_pathway_cost_test \
  training.agents.sac_lag.cost_conditioning_test \
  training.agents.sac_lag.train_test
```
