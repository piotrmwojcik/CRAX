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
environment overrides are respected. By default, alternative values are drawn
independently and continuously uniformly from half to twice each nominal value:
exponent `[1,4]`, hazard weight `[0.75,3]`, terminal penalty `[2.5,10]` for the
class defaults. This is ordinary uniform sampling, not the paper's log-uniform
perturbation distribution. The learned multiplier described below is our
constrained-RL extension; the paper does not learn Lagrange multipliers.

## Training

Add `--cost_conditioning` to a Safe Pathway SAC-Lag command:

```bash
python -m training.train_env \
  --env_name safe_pathway_walker2d --alg sac_lag \
  --cost_conditioning \
  --cost_parameter_bounds '[[1,4],[0.75,3],[2.5,10]]' \
  --cost_nominal_probability 0.5 \
  --lambda_network_lr 0.001 \
  --max_replay_size 1000000 \
  --safety_bound 25
```

Omit `--cost_parameter_bounds` to derive bounds from the nominal parameters.
Each gradient minibatch independently assigns a fresh parameter triple to every
transition: exactly nominal with probability 0.5 by default, otherwise continuous
uniform draws in all three coordinates. Multiple parameters vary simultaneously;
there is no finite bank in this mode. Equal lower/upper bounds fix a coordinate.
The current and next state
receive the **same** parameters. The policy, reward critic, cost critic, and
entropy-temperature loss all use these conditioned observations. Physical state
normalization excludes the cost parameters; their network features are `log1p`
of the three values, which also supports zero weights.

The actor uses the learned multiplier evaluated at each sampled triple:

```text
actor_loss = mean(alpha * log_probability - Q_reward(state, action, parameters)
                  + stop_gradient(lambda_network(parameters))
                    * Q_cost(state, action, parameters))
```

This implements one combined hazard-plus-terminal constraint per configuration,
not independent hazard and falling budgets. Every configuration uses the same
numeric `--safety_bound` in its own cost units. Thus changing weights with the
budget fixed changes the constraint, rather than merely rescaling its units.

### Learned lambda(psi)

The multiplier network has two 32-unit tanh hidden layers and a sigmoid output
scaled by `lambda_max`. Its inputs are the three cost parameters centered on the
nominal triple and divided by their sampling-range widths (zero widths use 1).
It produces bounded, configuration-dependent penalties without a lookup table.
The network minimizes the negative dual objective:

```text
lambda_loss = -mean(lambda_network(parameters)
                    * stop_gradient(relabelled_cost - per_step_budget))
```

The signal remains **replay-distribution immediate cost**, not a return estimate
from the cost critic. Positive violations push penalties upward; negative ones
push them downward, with generalization/coupling through shared network weights.
Gradients are averaged across devices before the Adam update. `lambda_network_lr`
(default `0.001`) is the network's learning rate; unlike scalar `lagrangian_lr`,
it is **not divided by episode length**. The actor cannot update the multiplier
network, and the dual loss cannot update the policy or critics.

The initial output is constant across parameters. To avoid a dead or undefined
sigmoid initialization at a boundary, `initial_lambda=0` initializes to
`0.0001 * lambda_max` (0.0002 for the default maximum of 2). Initialization at
`lambda_max` similarly moves inward by this fraction. Intermediate values are
respected, subject to that endpoint clamp. Per-step budgets still use either
the configured episode length or the nominal policy's observed-length EMA.

### Legacy discrete mode

Providing `--cost_configurations '[[1,1.5,5],[4,1.5,5]]'` explicitly selects the
previous finite-bank implementation with one scalar multiplier per configuration.
Do not combine it with `--cost_parameter_bounds`. It uses `lagrangian_lr /
episode_length` for its projected scalar updates, not `lambda_network_lr`.
Old discrete checkpoints still load. New runs that only set `--cost_conditioning`
now use continuous sampling and the learned multiplier.

**Limitation:** counterfactual replay means describe the nominal behavior-data
distribution. They are not measured episodic costs of the alternative conditioned
policies. Neither this dual estimator nor relabeling guarantees that alternative
policies meet their budgets. Evaluate reward, termination rate, and the correctly
recomputed episodic cost under each deployed configuration before interpreting
the results as constraint satisfaction. The nominal-policy episode-length EMA
also need not match alternative policies' episode lengths.

Existing `lambda_lagr`, `mean_cost`, `cost_violation`, and evaluation metrics refer
to the nominal configuration. `lambda_sampled_mean/min/max` summarize learned
penalties on the sampled mixture; `lambda_dual_loss` reports the dual loss.
`cost_nominal_fraction` reports the sampled fraction of exact nominal triples,
and `cost_parameters/<name>_mean` reports each sampled coordinate's mean.
Logged statistics are averaged over training steps, so the sampled min/max are
averages of minibatch extrema rather than global extrema.
`relabelled_mean_cost` reports the sampled training mixture. Discrete mode retains
the `cost_configuration_<index>/` metrics instead of learned-network metrics.
Built-in environment evaluations use the nominal configuration only.

## Inference and checkpoints

Checkpoints include nominal parameters, sampling bounds, multiplier network
weights/architecture, and conditioning settings (or the configuration bank and
scalar multipliers in discrete mode). The existing SAC checkpoint loader detects
conditioned checkpoints automatically:

```python
from training.agents.sac.checkpoint import load_policy

# Nominal configuration is the default.
policy = load_policy("models/run/000500000000", deterministic=True)

# Choose a cost condition within the sampling ranges.
policy = load_policy(
    "models/run/000500000000",
    deterministic=True,
    cost_parameters=(1.0, 1.5, 5.0),
)
action, extras = policy(observation, key)

# Inspect the learned multiplier; it is not required to execute the policy.
from training.agents.sac.checkpoint import load_cost_multiplier
lambda_fn = load_cost_multiplier("models/run/000500000000")
penalty = lambda_fn((1.7, 2.1, 8.0))
```

The trainer's returned `make_policy` also accepts `cost_parameters`. Evaluating a
different policy condition does not change `env.info['cost']`: it remains nominal.
Use `training.agents.sac_lag.cost_conditioning.reconstruct_cost` with the returned
`info['hazard_penetration']`, `info['unhealthy_termination']`, and chosen triple to
measure that configuration's actual cost. Triples outside the training ranges
are accepted for research evaluation, but their behavior is unvalidated extrapolation. Existing checkpoints
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
