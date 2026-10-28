# Copyright 2024 The Brax Authors.
# Modifications Copyright 2026 CRAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is derived from Brax 0.12.3 (training/agents/sac/train.py) and has been
# substantially modified by the CRAX Authors: adding the Lagrangian constraint machinery,
# a cost critic, a Lagrange multiplier updated against the safety bound, and cost-aware actor losses.

"""SAC-Lagrangian training.

Off-policy safe RL via Soft Actor-Critic + Lagrangian dual ascent.

The method maintains:
  - A reward Q-network Q(s,a)  (same as vanilla SAC)
  - A cost Q-network Qc(s,a)   (new)
  - A Lagrange multiplier λ ≥ 0 updated by projected gradient ascent

Policy objective (per gradient step):
  L_pi = E[alpha * log_prob - Q(s,a) + lambda * Qc(s,a)]

Lagrange multiplier update (per gradient step):
  lambda <- clip(max(0, lambda + (lr_lambda/episode_length) * (mean_cost - per_step_safety_bound)), 0, lambda_max)

  Dividing by episode_length normalises the per-step update so lagrangian_lr is
  on the same scale as PPO-Lag's lagrangian_coef_rate (both are effectively
  per-episode rates). Without this, lambda would grow ~episode_length times
  faster than PPO-Lag with the same lr value, killing all reward learning.

References:
  - SAC:      https://arxiv.org/pdf/1812.05905.pdf
  - CMDP/Lag: https://arxiv.org/pdf/2001.06684.pdf  (Ray et al., 2019)
"""

import functools
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from absl import logging
from crax import base
from crax import envs
from training import acting
from training import gradients
from training import logger as metric_logger
from training import pmap
from training import replay_buffers
from training import types
from training.acme import running_statistics
from training.acme import specs
from training.agents.sac import checkpoint
from training.agents.sac_lag import losses as sac_lag_losses
from training.agents.sac_lag import networks as sac_lag_networks
from training.agents.sac_lag import cost_conditioning as conditioning
from training.types import Params, PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the SAC-Lag learner."""
    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    qc_optimizer_state: optax.OptState
    qc_params: Params
    target_qc_params: Params
    gradient_steps: types.UInt64
    env_steps: types.UInt64
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    lambda_lagr: jnp.ndarray   # Lagrange multiplier (scalar, >= 0)
    observed_episode_length: jnp.ndarray
    normalizer_params: running_statistics.RunningStatisticsState
    aux_state: Any              # Extra state for custom lambda update fns (e.g. PID)
    lambda_params: Any = ()
    lambda_optimizer_state: Any = ()


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def update_observed_episode_length(
    previous_length: jnp.ndarray,
    completed_length_sum: jnp.ndarray,
    completed_episode_count: jnp.ndarray,
    ema: float,
) -> jnp.ndarray:
    """Updates an EMA from newly completed episode lengths."""
    batch_length = completed_length_sum / jnp.maximum(completed_episode_count, 1.0)
    return jnp.where(
        completed_episode_count > 0,
        ema * previous_length + (1.0 - ema) * batch_length,
        previous_length,
    )


def _random_translate_pixels(obs: Dict[str, jax.Array], key: PRNGKey) -> Dict[str, jax.Array]:
    """DrQ-style random crop-pad shift on pixel obs, independent per sample.

    Unlike ppo.train's version of this (which shares one shift across its
    on-policy unroll_length axis), a replay-sampled batch has no temporal
    structure to share a shift across, so every sample gets its own.
    """

    def rt_view(img: jax.Array, padding: int, key: PRNGKey) -> jax.Array:  # HxWxC
        crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
        crop_from = jnp.concatenate([crop_from, jnp.zeros((1,), dtype=jnp.int32)])
        padded = jnp.pad(
            img, ((padding, padding), (padding, padding), (0, 0)), mode='edge'
        )
        return jax.lax.dynamic_slice(padded, crop_from, img.shape)

    out = {}
    for k, v in obs.items():
        if k.startswith('pixels/'):
            key, key_shift = jax.random.split(key)
            keys = jax.random.split(key_shift, v.shape[0])
            out[k] = jax.vmap(lambda img, kk: rt_view(img, 4, kk))(v, keys)
    return {**obs, **out}


def _remove_pixels(obs):
    """Drops 'pixels/*' keys -- those are normalized internally (/255) by
    the vision CNN encoder, not via running statistics. No-op for flat
    (non-dict) state observations. Mirrors training.agents.ppo.train.
    """
    if not isinstance(obs, dict):
        return obs
    return {k: v for k, v in obs.items() if not k.startswith('pixels/')}


def _init_training_state(
    key: PRNGKey,
    obs_size: Any,
    local_devices_to_use: int,
    sac_network: sac_lag_networks.SACLagNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    qc_optimizer: optax.GradientTransformation,
    initial_lambda: float,
    initial_observed_episode_length: float,
    aux_state: Any = (),
    lambda_params: Any = (),
    lambda_optimizer_state: Any = (),
) -> TrainingState:
    """Initialises the training state and replicates it over devices."""
    key_policy, key_q, key_qc = jax.random.split(key, 3)
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)

    q_params = sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    qc_params = sac_network.qc_network.init(key_qc)
    qc_optimizer_state = qc_optimizer.init(qc_params)

    if isinstance(obs_size, dict):
        obs_spec = _remove_pixels(
            {k: specs.Array(v, jnp.dtype('float32')) for k, v in obs_size.items()}
        )
    else:
        obs_spec = specs.Array((obs_size,), jnp.dtype('float32'))
    normalizer_params = running_statistics.init_state(obs_spec)
    lambda_lagr = jnp.asarray(initial_lambda, dtype=jnp.float32)

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        qc_optimizer_state=qc_optimizer_state,
        qc_params=qc_params,
        target_qc_params=qc_params,
        gradient_steps=types.UInt64(hi=0, lo=0),
        env_steps=types.UInt64(hi=0, lo=0),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        lambda_lagr=lambda_lagr,
        observed_episode_length=jnp.asarray(
            initial_observed_episode_length, dtype=jnp.float32
        ),
        normalizer_params=normalizer_params,
        aux_state=aux_state,
        lambda_params=lambda_params,
        lambda_optimizer_state=lambda_optimizer_state,
    )
    devices = jax.local_devices()[:local_devices_to_use]
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(devices), ('i',)), jax.sharding.PartitionSpec('i')
    )
    return jax.tree_util.tree_map(
        lambda value: jax.device_put(jnp.stack([value] * len(devices)), sharding),
        training_state,
    )


def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    wrap_env: bool = True,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    discounting: float = 0.99,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 0,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = 1e7,
    grad_updates_per_step: int = 32,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        sac_lag_networks.SACLagNetworks
    ] = sac_lag_networks.make_sac_lag_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    training_metrics_steps: int = 1_000_000,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # Vision (pixel-obs) training. Mirrors training.agents.ppo.train:
    # if given, applies GpuPixelObservationWrapper (MJWarp) to `environment`/
    # `eval_env` AFTER `wrap_for_training`, and `network_factory` must build
    # CNN-based networks (see sac_lag.networks.make_sac_lag_networks_vision).
    vision_kwargs: Optional[Dict[str, Any]] = None,
    augment_pixels: bool = False,
    # SAC-Lag specific
    safety_bound: float = 25.0,
    observed_episode_length_bound: bool = False,
    episode_length_ema: float = 0.9,
    lagrangian_lr: float = 0.01,
    initial_lambda: float = 0.0,
    lambda_max: float = 2.0,
    cost_conditioning: bool = False,
    cost_configurations: Optional[Sequence] = None,
    cost_parameter_bounds: Optional[Sequence] = None,
    lambda_network_lr: float = 0.001,
    cost_nominal_probability: float = 0.5,
    # Hooks for custom lambda update (used by SAC-PID etc.)
    lambda_update_fn: Optional[Callable] = None,
    init_aux_state_fn: Optional[Callable] = None,
    save_checkpoint_path: Optional[str] = None,
):
    """SAC-Lagrangian training.

    Args:
      environment: the environment to train.
      num_timesteps: total number of environment steps.
      episode_length: length of each episode (used to convert episodic safety
        bound to per-step bound).
      wrap_env: whether to wrap the environment for training.
      wrap_env_fn: optional custom environment wrapper.
      action_repeat: number of times each action is repeated.
      num_envs: number of parallel environments.
      num_eval_envs: number of environments used for evaluation.
      learning_rate: learning rate for policy, Q, and Qc networks.
      discounting: discount factor gamma.
      seed: random seed.
      batch_size: SGD mini-batch size.
      num_evals: number of evaluation checkpoints.
      normalize_observations: whether to z-score observations.
      max_devices_per_host: maximum number of accelerators to use.
      reward_scaling: scalar multiplier on environment rewards.
      tau: soft target-network update coefficient.
      min_replay_size: minimum replay buffer size before training starts.
      max_replay_size: maximum replay buffer capacity.
      grad_updates_per_step: gradient updates per environment step.
      deterministic_eval: use deterministic policy during evaluation.
      network_factory: factory for SAC-Lag networks.
      progress_fn: callback invoked at each eval checkpoint.
      eval_env: optional separate environment for evaluation.
      randomization_fn: optional domain-randomisation callback.
      vision_kwargs: if given, adds MJWarp-rendered pixel observations to
        `environment`/`eval_env` (see GpuPixelObservationWrapper) and the
        replay buffer stores transitions as a dtype-preserving pytree
        (`replay_buffers.PytreeUniformSamplingQueue`) instead of the
        default ravel_pytree'd single-dtype buffer, so uint8 frames aren't
        silently upcast to float32. `max_replay_size` still bounds it in
        *transitions*, not bytes — with pixel obs each transition is much
        bigger, so pass an explicit (much smaller) `max_replay_size`; the
        default (num_timesteps) is sized for flat state vectors and will
        try to allocate an enormous buffer for images.
      augment_pixels: whether to randomly translate (DrQ-style) pixel obs
        in each sampled training batch (same shift across a grad-update's
        batch, applied to both `observation` and `next_observation`).
      safety_bound: episodic cumulative cost limit (constraint threshold d).
      observed_episode_length_bound: derive the per-step cost limit from an
        EMA of completed training episode lengths instead of `episode_length`.
      episode_length_ema: EMA coefficient for the observed episode length.
      lagrangian_lr: learning rate for the Lagrange multiplier. Normalized
        internally by episode_length so the effective per-episode update rate is
        comparable to PPO-Lag's lagrangian_coef_rate.
      initial_lambda: initial value of the Lagrange multiplier.
      lambda_max: upper bound on the Lagrange multiplier (prevents runaway).
      cost_conditioning: enable Safe Pathway cost-conditioned replay (state observations,
        action_repeat=1). Keeps rewards and data collection at the nominal task.
      cost_configurations: alternative (exponent, hazard weight, terminal penalty)
        triples for legacy discrete mode; omitted selects continuous uniform sampling.
      cost_parameter_bounds: uniform [min, max] bounds for the three parameters;
        defaults to half/twice the nominal values.
      lambda_network_lr: Adam learning rate for lambda(psi), without episode-length division.
      cost_nominal_probability: probability of nominal conditioning per replay sample.
      save_checkpoint_path: directory for policy checkpoints. A checkpoint is
        saved after each evaluation and at the end of training. It includes the
        observation normalizer and policy parameters; the Lagrange multiplier
        is recorded in the checkpoint config.

    Returns:
      Tuple of (make_policy, params, metrics).
    """
    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    if max_devices_per_host is not None:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * jax.process_count()

    if min_replay_size >= num_timesteps:
        raise ValueError(
            'No training will happen because min_replay_size >= num_timesteps'
        )
    num_timesteps = int(num_timesteps)
    max_replay_size = int(max_replay_size if max_replay_size is not None else num_timesteps)
    min_replay_size = int(min_replay_size)

    cost_bank = None
    uniform_costs = cost_conditioning and cost_configurations is None
    lambda_params, lambda_optimizer_state = (), ()
    if cost_configurations is not None and not cost_conditioning:
        raise ValueError('cost_configurations requires cost_conditioning=True.')
    if cost_parameter_bounds is not None and not cost_conditioning:
        raise ValueError('cost_parameter_bounds requires cost_conditioning=True.')
    if cost_parameter_bounds is not None and cost_configurations is not None:
        raise ValueError('Choose continuous bounds or discrete configurations, not both.')
    if cost_conditioning:
        from crax.envs.safe_pathway import SafePathway
        base_env = environment.unwrapped
        if not isinstance(base_env, SafePathway):
            raise ValueError('Cost conditioning currently supports Safe Pathway only.')
        if vision_kwargs is not None or not isinstance(environment.observation_size, int):
            raise ValueError('Cost conditioning currently requires flat state observations.')
        if action_repeat != 1:
            raise ValueError('Cost conditioning requires action_repeat=1 to preserve per-step measurements.')
        if lambda_update_fn is not None:
            raise ValueError('Cost conditioning supports SAC-Lagrangian only, without a custom multiplier update.')
        if not 0.0 < cost_nominal_probability < 1.0:
            raise ValueError('cost_nominal_probability must lie strictly between zero and one.')
        if episode_length is None or episode_length <= 0:
            raise ValueError('Cost conditioning requires a positive episode_length.')
        if safety_bound < 0 or lagrangian_lr < 0 or not 0 <= initial_lambda <= lambda_max:
            raise ValueError('Invalid cost bound, multiplier learning rate, or multiplier range.')
        nominal_cost = conditioning.validate_parameters(
            (2.0, base_env._cost_scaler, base_env._unhealthy_termination_cost)
        )
        if uniform_costs:
            if not np.isfinite(lambda_network_lr) or lambda_network_lr <= 0 or not np.isfinite(lambda_max) or lambda_max <= 0:
                raise ValueError('Continuous lambda requires finite positive lambda_network_lr and lambda_max.')
            cost_bounds = conditioning.parameter_bounds(nominal_cost, cost_parameter_bounds)
            cost_bank = (nominal_cost,)
            lambda_network = conditioning.MultiplierNetwork(nominal_cost, cost_bounds, lambda_max, initial_lambda)
            lambda_params = lambda_network.init(jax.random.fold_in(jax.random.PRNGKey(seed), 1729), jnp.asarray(nominal_cost))
            lambda_optimizer = optax.adam(lambda_network_lr)
            lambda_optimizer_state = lambda_optimizer.init(lambda_params)
            initial_lambda = lambda_network.apply(lambda_params, jnp.asarray(nominal_cost))
            logging.info('Uniform cost bounds: %s; learned lambda(psi)', cost_bounds)
        else:
            cost_bank = conditioning.parameter_bank(nominal_cost, cost_configurations)
            initial_lambda = jnp.full((len(cost_bank),), initial_lambda, dtype=jnp.float32)
        if getattr(network_factory, 'func', network_factory) is not sac_lag_networks.make_sac_lag_networks:
            raise ValueError('Cost conditioning requires the standard SAC-Lag state network factory.')
        network_factory = functools.partial(
            sac_lag_networks.make_cost_conditioned_networks,
            **getattr(network_factory, 'keywords', {}),
            nominal_cost_parameters=cost_bank[0],
        )
        logging.info('Cost conditioning configurations (nominal first): %s', cost_bank)

    def add_cost_checkpoint_metadata(config, state):
        multipliers = _unpmap(state.lambda_lagr)
        config.algorithm = 'sac_lag'
        config.lambda_lagr = float(multipliers[0] if cost_conditioning and not uniform_costs else multipliers)
        if cost_conditioning:
            config.cost_conditioning = True
            config.cost_parameter_names = list(conditioning.PARAMETER_NAMES)
            config.cost_sampling = 'uniform' if uniform_costs else 'discrete'
            if uniform_costs:
                config.cost_parameter_bounds = [list(pair) for pair in cost_bounds]
                config.lambda_network_lr = lambda_network_lr
                config.lambda_max = lambda_max
                config.lambda_network_architecture = 'mlp_32_32_tanh_sigmoid_v1'
                config.lambda_network_params = jax.tree_util.tree_map(
                    lambda value: np.asarray(value).tolist(), _unpmap(state.lambda_params)
                )
            else:
                config.cost_configurations = [list(values) for values in cost_bank]
                config.cost_multipliers = jax.device_get(multipliers).tolist()
            config.cost_nominal_probability = cost_nominal_probability
            config.safety_bound = safety_bound
            config.cost_dual_estimator = 'nominal_replay_mean_cost'
        return config

    if observed_episode_length_bound:
        if episode_length is None or episode_length <= 0:
            raise ValueError(
                'observed_episode_length_bound requires a positive episode_length.'
            )
        if not 0.0 <= episode_length_ema < 1.0:
            raise ValueError('episode_length_ema must be in [0, 1).')
        if lambda_update_fn is not None:
            raise ValueError(
                'observed_episode_length_bound currently supports the built-in '
                'SAC-Lag multiplier update only.'
            )

    # Convert episodic safety bound to per-step bound
    per_step_safety_bound = safety_bound / episode_length if episode_length else safety_bound

    # Normalize lambda lr by episode_length so lagrangian_lr is on the same
    # per-episode scale as PPO-Lag's lagrangian_coef_rate. Without this,
    # SAC-Lag updates lambda every gradient step (~250K times for 5e8 steps)
    # whereas PPO-Lag updates once per rollout, causing lambda to explode.
    effective_lagrangian_lr = lagrangian_lr / max(episode_length or 1, 1)

    # Build the lambda update function if not provided by caller.
    # Default signature optionally also receives the observed episode length.
    if lambda_update_fn is None:
        _eff_lr = effective_lagrangian_lr
        _lmax = float(lambda_max)
        _bound = per_step_safety_bound
        def lambda_update_fn(lambda_lagr, aux_state, mean_cost, observed_length=None):
            if observed_episode_length_bound:
                step_bound = safety_bound / jnp.maximum(observed_length, 1.0)
            else:
                step_bound = _bound
            cost_violation = mean_cost - step_bound
            new_lambda = jnp.minimum(
                jax.nn.relu(lambda_lagr + _eff_lr * cost_violation),
                jnp.asarray(_lmax, jnp.float32),
            )
            metrics = {'lambda_lagr': new_lambda, 'cost_violation': cost_violation}
            if observed_episode_length_bound:
                metrics['observed_episode_length'] = observed_length
                metrics['per_step_safety_bound'] = step_bound
            return new_lambda, aux_state, metrics

    init_aux = init_aux_state_fn() if init_aux_state_fn is not None else ()

    env_steps_per_actor_step = action_repeat * num_envs
    num_prefill_actor_steps = -(-min_replay_size // num_envs)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps)
        // (num_evals_after_init * env_steps_per_actor_step)
    )

    assert num_envs % device_count == 0
    env = environment
    if wrap_env:
        if wrap_env_fn is not None:
            wrap_for_training = wrap_env_fn
        elif isinstance(env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            raise ValueError('Unsupported environment type: %s' % type(env))

        rng = jax.random.PRNGKey(seed)
        rng, key = jax.random.split(rng)
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(
                    key, num_envs // jax.process_count() // local_devices_to_use
                ),
            )
        env = wrap_for_training(
            env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )
        if vision_kwargs is not None:
            # Must wrap AFTER wrap_for_training (post-vmap/episode/autoreset)
            # -- see GpuPixelObservationWrapper's module docstring.
            from crax.envs.wrappers.pixel_observation_gpu import GpuPixelObservationWrapper
            env = GpuPixelObservationWrapper(env, num_envs=num_envs, **vision_kwargs)

    obs_size = env.observation_size
    action_size = env.action_size

    normalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize

    sac_network = network_factory(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
    )
    make_policy = sac_lag_networks.make_inference_fn(sac_network)

    alpha_optimizer = optax.adam(learning_rate=3e-4)
    policy_optimizer = optax.adam(learning_rate=learning_rate)
    q_optimizer = optax.adam(learning_rate=learning_rate)
    qc_optimizer = optax.adam(learning_rate=learning_rate)

    if isinstance(obs_size, dict):
        # uint8 for pixel keys (matches GpuPixelObservationWrapper's output dtype
        dummy_obs = {
            key: jnp.zeros(
                shape, jnp.uint8 if key.startswith('pixels/') else jnp.float32
            )
            for key, shape in obs_size.items()
        }
    else:
        dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    replay_extras = {'truncation': 0.0, 'cost': 0.0}
    if cost_conditioning:
        replay_extras.update(
            hazard_penetration=jnp.zeros((len(base_env._hazard_mocap_ids),)),
            unhealthy_termination=0.0,
        )
    replay_extra_fields = tuple(replay_extras)
    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        next_observation=dummy_obs,
        extras={
            'state_extras': replay_extras,
            'policy_extras': {},
        },
    )
    replay_buffer_cls = (
        replay_buffers.PytreeUniformSamplingQueue
        if isinstance(obs_size, dict)
        else replay_buffers.UniformSamplingQueue
    )
    replay_buffer = replay_buffer_cls(
        max_replay_size=max_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size * grad_updates_per_step // device_count,
    )

    metrics_aggregator = metric_logger.MetricsLogger(
        buffer_size=10,
        steps_between_logging=int(training_metrics_steps),
        progress_fn=progress_fn,
    )

    alpha_loss, critic_loss, cost_critic_loss, actor_loss = sac_lag_losses.make_losses(
        sac_network=sac_network,
        reward_scaling=reward_scaling,
        discounting=discounting,
        action_size=action_size,
    )
    alpha_update = gradients.gradient_update_fn(
        alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    critic_update = gradients.gradient_update_fn(
        critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    cost_critic_update = gradients.gradient_update_fn(
        cost_critic_loss, qc_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    actor_update = gradients.gradient_update_fn(
        actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )

    def sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        if cost_conditioning:
            # Keep the existing replay-cost dual-ascent estimator, separately
            # for each configuration. These are nominal replay-distribution
            # estimates, not measured returns of the alternative policies.
            configuration_costs = jax.lax.pmean(
                conditioning.counterfactual_mean_costs(transitions, cost_bank),
                axis_name=_PMAP_AXIS_NAME,
            )
            key, condition_key = jax.random.split(key)
            if uniform_costs:
                transitions, nominal_mask = conditioning.relabel_uniform(
                    transitions, condition_key, nominal_cost, cost_bounds, cost_nominal_probability
                )
                sampled_parameters = transitions.observation['cost_parameters']
                actor_lambda = jax.lax.stop_gradient(lambda_network.apply(
                    training_state.lambda_params, sampled_parameters
                ))
            else:
                transitions, configuration_ids = conditioning.relabel(
                    transitions, condition_key, cost_bank, cost_nominal_probability
                )
                actor_lambda = training_state.lambda_lagr[configuration_ids]
        else:
            actor_lambda = training_state.lambda_lagr
        key, key_alpha, key_critic, key_cost_critic, key_actor = jax.random.split(key, 5)

        # --- Entropy temperature update (unchanged from SAC) ---
        alpha_loss_val, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params)

        # --- Reward critic update (unchanged from SAC) ---
        critic_loss_val, q_params, q_optimizer_state = critic_update(
            training_state.q_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_q_params,
            alpha,
            transitions,
            key_critic,
            optimizer_state=training_state.q_optimizer_state,
        )

        # --- Cost critic update (new) ---
        cost_critic_loss_val, qc_params, qc_optimizer_state = cost_critic_update(
            training_state.qc_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_qc_params,
            transitions,
            key_cost_critic,
            optimizer_state=training_state.qc_optimizer_state,
        )

        # --- Lagrange-penalised actor update ---
        actor_loss_val, policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            training_state.qc_params,
            alpha,
            actor_lambda,
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state,
        )

        # --- Soft target-network updates ---
        new_target_q_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            training_state.target_q_params, q_params,
        )
        new_target_qc_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            training_state.target_qc_params, qc_params,
        )

        # --- Lagrange multiplier update ---
        mean_cost = jnp.mean(transitions.extras['state_extras']['cost'])
        new_lambda_params = training_state.lambda_params
        new_lambda_optimizer_state = training_state.lambda_optimizer_state
        if cost_conditioning:
            length = training_state.observed_episode_length if observed_episode_length_bound else episode_length
            step_bound = safety_bound / jnp.maximum(length, 1.0)
            new_aux_state = training_state.aux_state
            if uniform_costs:
                dual_loss, dual_grads = jax.value_and_grad(lambda params: conditioning.multiplier_loss(
                    lambda_network, params, sampled_parameters,
                    transitions.extras['state_extras']['cost'], step_bound,
                ))(training_state.lambda_params)
                dual_grads = jax.lax.pmean(dual_grads, axis_name=_PMAP_AXIS_NAME)
                dual_updates, new_lambda_optimizer_state = lambda_optimizer.update(
                    dual_grads, training_state.lambda_optimizer_state, training_state.lambda_params
                )
                new_lambda_params = optax.apply_updates(training_state.lambda_params, dual_updates)
                new_lambda_lagr = lambda_network.apply(new_lambda_params, jnp.asarray(nominal_cost))
                sampled_lambdas = lambda_network.apply(new_lambda_params, sampled_parameters)
                lambda_metrics = {
                    'lambda_lagr': new_lambda_lagr,
                    'cost_violation': configuration_costs[0] - step_bound,
                    'lambda_dual_loss': jax.lax.pmean(dual_loss, axis_name=_PMAP_AXIS_NAME),
                    'lambda_sampled_mean': jnp.mean(sampled_lambdas),
                    'lambda_sampled_min': jnp.min(sampled_lambdas),
                    'lambda_sampled_max': jnp.max(sampled_lambdas),
                    'cost_nominal_fraction': jnp.mean(nominal_mask),
                }
                for i, name in enumerate(conditioning.PARAMETER_NAMES):
                    lambda_metrics[f'cost_parameters/{name}_mean'] = jnp.mean(sampled_parameters[..., i])
            else:
                new_lambda_lagr, violations = conditioning.update_multipliers(
                    training_state.lambda_lagr, configuration_costs, step_bound,
                    effective_lagrangian_lr, lambda_max,
                )
                lambda_metrics = {'lambda_lagr': new_lambda_lagr[0], 'cost_violation': violations[0]}
                for i in range(len(cost_bank)):
                    lambda_metrics[f'cost_configuration_{i}/lambda'] = new_lambda_lagr[i]
                    lambda_metrics[f'cost_configuration_{i}/mean_cost'] = configuration_costs[i]
                    lambda_metrics[f'cost_configuration_{i}/violation'] = violations[i]
            lambda_metrics['per_step_safety_bound'] = step_bound
            lambda_metrics['relabelled_mean_cost'] = mean_cost
            if observed_episode_length_bound:
                lambda_metrics['observed_episode_length'] = length
            mean_cost = configuration_costs[0]
        elif observed_episode_length_bound:
            new_lambda_lagr, new_aux_state, lambda_metrics = lambda_update_fn(
                training_state.lambda_lagr,
                training_state.aux_state,
                mean_cost,
                training_state.observed_episode_length,
            )
        else:
            new_lambda_lagr, new_aux_state, lambda_metrics = lambda_update_fn(
                training_state.lambda_lagr, training_state.aux_state, mean_cost
            )

        metrics = {
            'critic_loss': critic_loss_val,
            'cost_critic_loss': cost_critic_loss_val,
            'actor_loss': actor_loss_val,
            'alpha_loss': alpha_loss_val,
            'alpha': jnp.exp(alpha_params),
            'mean_cost': mean_cost,
            **lambda_metrics,
        }

        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=new_target_q_params,
            qc_optimizer_state=qc_optimizer_state,
            qc_params=qc_params,
            target_qc_params=new_target_qc_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            lambda_lagr=new_lambda_lagr,
            observed_episode_length=training_state.observed_episode_length,
            normalizer_params=training_state.normalizer_params,
            aux_state=new_aux_state,
            lambda_params=new_lambda_params,
            lambda_optimizer_state=new_lambda_optimizer_state,
        )
        return (new_training_state, key), metrics

    def get_experience(
        normalizer_params: running_statistics.RunningStatisticsState,
        policy_params: Params,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[running_statistics.RunningStatisticsState, envs.State, ReplayBufferState]:
        policy = make_policy((normalizer_params, policy_params))
        env_state, transitions = acting.actor_step(
            env, env_state, policy, key, extra_fields=replay_extra_fields
        )
        normalizer_params = running_statistics.update(
            normalizer_params,
            _remove_pixels(transitions.observation),
            pmap_axis_name=_PMAP_AXIS_NAME,
        )
        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return normalizer_params, env_state, buffer_state

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state = get_experience(
            training_state.normalizer_params,
            training_state.policy_params,
            env_state,
            buffer_state,
            experience_key,
        )
        new_env_steps = training_state.env_steps + env_steps_per_actor_step
        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=new_env_steps,
        )
        if observed_episode_length_bound:
            episode_done = env_state.info['episode_done']
            episode_lengths = env_state.info['episode_metrics']['length']
            completed_episode_count = jax.lax.psum(
                jnp.sum(episode_done), axis_name=_PMAP_AXIS_NAME
            )
            completed_length_sum = jax.lax.psum(
                jnp.sum(episode_done * episode_lengths), axis_name=_PMAP_AXIS_NAME
            )
            observed_length = update_observed_episode_length(
                training_state.observed_episode_length,
                completed_length_sum,
                completed_episode_count,
                episode_length_ema,
            )
            training_state = training_state.replace(
                observed_episode_length=observed_length
            )
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        if augment_pixels:
            training_key, key_aug_obs, key_aug_next = jax.random.split(training_key, 3)
            transitions = transitions._replace(
                observation=_random_translate_pixels(transitions.observation, key_aug_obs),
                next_observation=_random_translate_pixels(
                    transitions.next_observation, key_aug_next
                ),
            )
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        (training_state, _), metrics = jax.lax.scan(
            sgd_step, (training_state, training_key), transitions
        )
        metrics['buffer_current_size'] = replay_buffer.size(buffer_state)
        jax.debug.callback(
            metrics_aggregator.update_step_metrics,
            env_state.info['episode_metrics'],
            env_state.info['episode_done'],
            metrics,
            new_env_steps,
        )
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            new_normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                env_state,
                buffer_state,
                key,
            )
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_prefill_actor_steps,
        )[0]

    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        nonlocal training_walltime
        t = time.time()
        (training_state, env_state, buffer_state, metrics) = training_epoch(
            training_state, env_state, buffer_state, key
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            'training/sps': sps,
            'training/walltime': training_walltime,
            **{f'training/{name}': value for name, value in metrics.items()},
        }
        return training_state, env_state, buffer_state, metrics

    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)

    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        local_devices_to_use=local_devices_to_use,
        sac_network=sac_network,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        qc_optimizer=qc_optimizer,
        initial_lambda=initial_lambda,
        initial_observed_episode_length=episode_length,
        aux_state=init_aux,
        lambda_params=lambda_params,
        lambda_optimizer_state=lambda_optimizer_state,
    )
    del global_key

    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
    env_state = jax.pmap(env.reset)(env_keys)
    if observed_episode_length_bound and (
        'episode_done' not in env_state.info
        or 'episode_metrics' not in env_state.info
        or 'length' not in env_state.info.get('episode_metrics', {})
    ):
        raise ValueError(
            'observed_episode_length_bound requires an environment wrapped '
            'with episode tracking.'
        )

    buffer_state = jax.pmap(replay_buffer.init)(
        jax.random.split(rb_key, local_devices_to_use)
    )

    if not eval_env:
        eval_env = environment
    if wrap_env:
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
            )
        eval_env = wrap_for_training(
            eval_env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )
        if vision_kwargs is not None:
            from crax.envs.wrappers.pixel_observation_gpu import GpuPixelObservationWrapper
            eval_env = GpuPixelObservationWrapper(
                eval_env, num_envs=num_eval_envs, **vision_kwargs
            )

    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap((training_state.normalizer_params, training_state.policy_params)),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    # Prefill replay buffer
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_keys
    )

    replay_size = (
        jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    )
    logging.info('replay size after prefill %s', replay_size)
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t

    current_step = 0
    for _ in range(num_evals_after_init):
        logging.info('step %s', current_step)

        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        (training_state, env_state, buffer_state, training_metrics) = (
            training_epoch_with_timing(
                training_state, env_state, buffer_state, epoch_keys
            )
        )
        current_step = int(_unpmap(training_state.env_steps))

        # Always log training metrics (losses, lambda, etc.) after every epoch,
        # matching PPO's pattern of calling progress_fn unconditionally.
        if process_id == 0:
            progress_fn(current_step, training_metrics)

        if process_id == 0 and num_evals > 0:
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (training_state.normalizer_params, training_state.policy_params)
                ),
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)

            if save_checkpoint_path is not None:
                params = _unpmap(
                    (training_state.normalizer_params, training_state.policy_params)
                )
                ckpt_config = checkpoint.network_config(
                    observation_size=obs_size,
                    action_size=env.action_size,
                    normalize_observations=normalize_observations,
                    network_factory=network_factory,
                )
                ckpt_config = add_cost_checkpoint_metadata(ckpt_config, training_state)
                checkpoint.save(save_checkpoint_path, current_step, params, ckpt_config)

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(
            f'Total steps {total_steps} is less than `num_timesteps`={num_timesteps}.'
        )

    params = _unpmap(
        (training_state.normalizer_params, training_state.policy_params)
    )
    if process_id == 0 and save_checkpoint_path is not None:
        ckpt_config = checkpoint.network_config(
            observation_size=obs_size,
            action_size=env.action_size,
            normalize_observations=normalize_observations,
            network_factory=network_factory,
        )
        ckpt_config = add_cost_checkpoint_metadata(ckpt_config, training_state)
        checkpoint.save(save_checkpoint_path, total_steps, params, ckpt_config)
    pmap.assert_is_replicated(training_state)
    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return make_policy, params, metrics, eval_env
