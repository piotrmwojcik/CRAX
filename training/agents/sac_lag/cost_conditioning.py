"""Cost-conditioned replay for SAC-Lag, inspired by arXiv:2603.05066.

Rewards and environment interaction remain nominal. Each replay transition is
paired with a cost specification, shared by its current and next observations.
"""

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn


PARAMETER_NAMES = ('penetration_exponent', 'hazard_weight', 'terminal_penalty')


class MultiplierNetwork(nn.Module):
    """Bounded lambda(psi); sigmoid avoids dead gradients at an initial zero.

    Exact endpoint initializations are moved 1e-4 of the range into the interior.
    Hidden features are centered on nominal parameters and scaled by box widths.
    """

    nominal: tuple
    bounds: tuple
    maximum: float = 2.0
    initial: float = 0.0

    @nn.compact
    def __call__(self, parameters):
        bounds = jnp.asarray(self.bounds)
        width = bounds[:, 1] - bounds[:, 0]
        features = (jnp.asarray(parameters) - jnp.asarray(self.nominal)) / jnp.where(width > 0, width, 1.)
        for _ in range(2):
            features = nn.tanh(nn.Dense(32)(features))
        fraction = min(max(self.initial / self.maximum, 1e-4), 1. - 1e-4)
        bias = math.log(fraction / (1. - fraction))
        logits = nn.Dense(1, kernel_init=nn.initializers.zeros,
                          bias_init=nn.initializers.constant(bias))(features)
        return self.maximum * jax.nn.sigmoid(logits[..., 0])


def multiplier_loss(network, network_params, parameters, costs, step_bound):
    """Minimize negative dual objective using replay-distribution violations."""
    violations = jax.lax.stop_gradient(costs - step_bound)
    return -jnp.mean(network.apply(network_params, parameters) * violations)


def validate_parameters(parameters):
    values = tuple(float(x) for x in parameters)
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError('Cost parameters must contain three finite numbers.')
    if values[0] <= 0 or min(values[1:]) < 0:
        raise ValueError('Penetration exponent must be positive; cost weights must be nonnegative.')
    return values


def parameter_bank(nominal, alternatives=None):
    """Returns nominal configuration first, followed by distinct alternatives."""
    nominal = validate_parameters(nominal)
    if alternatives is None:
        alternatives = []
        for index in range(3):
            for scale in (0.5, 2.0):
                values = list(nominal)
                values[index] *= scale
                alternatives.append(values)
    bank = [nominal]
    for values in alternatives:
        if isinstance(values, Mapping):
            if set(values) != set(PARAMETER_NAMES):
                raise ValueError(f'Each cost configuration must specify {PARAMETER_NAMES}.')
            values = [values[name] for name in PARAMETER_NAMES]
        values = validate_parameters(values)
        if values not in bank:
            bank.append(values)
    if len(bank) < 2:
        raise ValueError('Cost conditioning requires at least one non-nominal configuration.')
    return tuple(bank)


def parameter_bounds(nominal, bounds=None):
    """Validate uniform bounds in exponent, hazard-weight, terminal-penalty order."""
    nominal = validate_parameters(nominal)
    if bounds is None:
        bounds = [(0.5 * value, 2.0 * value) for value in nominal]
    if len(bounds) != 3 or any(len(pair) != 2 for pair in bounds):
        raise ValueError('Cost parameter bounds must contain three [minimum, maximum] pairs.')
    lower = validate_parameters([pair[0] for pair in bounds])
    upper = validate_parameters([pair[1] for pair in bounds])
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError('Each cost parameter minimum must be <= its maximum.')
    return tuple(zip(lower, upper))


def sample_uniform_parameters(key, shape, nominal, bounds, nominal_probability):
    """Independent uniform coordinates, mixed with an exact nominal point mass."""
    nominal_key, alternative_key = jax.random.split(key)
    bounds = jnp.asarray(bounds)
    parameters = jax.random.uniform(
        alternative_key, shape + (3,), minval=bounds[:, 0], maxval=bounds[:, 1]
    )
    is_nominal = jax.random.bernoulli(nominal_key, nominal_probability, shape)
    return jnp.where(is_nominal[..., None], jnp.asarray(nominal), parameters), is_nominal


def reconstruct_cost(penetration, termination, parameters):
    """Recompute scalar cost; last penetration axis enumerates hazards.

    Parameters may be a single triple or a triple per transition. Foot contact
    masking and max-over-feet have already been applied by the environment.
    """
    parameters = jnp.asarray(parameters)
    exponent, weight, terminal = (parameters[..., i] for i in range(3))
    return weight * jnp.sum(penetration ** exponent[..., None], axis=-1) + terminal * termination


def condition_observation(observation, parameters):
    parameters = jnp.broadcast_to(jnp.asarray(parameters), observation.shape[:-1] + (3,))
    return {'state': observation, 'cost_parameters': parameters}


def relabel(transitions, key, bank, nominal_probability):
    """Sample a cost configuration per transition, leaving rewards untouched."""
    nominal_key, alternative_key = jax.random.split(key)
    shape = transitions.reward.shape
    ids = jax.random.randint(alternative_key, shape, 1, len(bank))
    ids = jnp.where(jax.random.bernoulli(nominal_key, nominal_probability, shape), 0, ids)
    parameters = jnp.asarray(bank)[ids]
    return relabel_with_parameters(transitions, parameters), ids


def relabel_uniform(transitions, key, nominal, bounds, nominal_probability):
    """Draw fresh continuous cost parameters for every replayed transition."""
    parameters, is_nominal = sample_uniform_parameters(
        key, transitions.reward.shape, nominal, bounds, nominal_probability
    )
    return relabel_with_parameters(transitions, parameters), is_nominal


def relabel_with_parameters(transitions, parameters):
    """Apply a given parameter triple per transition without changing rewards."""
    state_extras = dict(transitions.extras['state_extras'])
    state_extras['cost'] = reconstruct_cost(
        state_extras['hazard_penetration'], state_extras['unhealthy_termination'], parameters
    )
    return transitions._replace(
        observation=condition_observation(transitions.observation, parameters),
        next_observation=condition_observation(transitions.next_observation, parameters),
        extras={**transitions.extras, 'state_extras': state_extras},
    )


def counterfactual_mean_costs(transitions, bank):
    """Replay-distribution costs, NOT on-policy returns of alternative policies."""
    extras = transitions.extras['state_extras']
    return jax.vmap(lambda parameters: jnp.mean(reconstruct_cost(
        extras['hazard_penetration'], extras['unhealthy_termination'], parameters
    )))(jnp.asarray(bank))


def update_multipliers(multipliers, mean_costs, step_bound, learning_rate, maximum):
    violation = mean_costs - step_bound
    return jnp.clip(multipliers + learning_rate * violation, 0.0, maximum), violation
