"""Cost-conditioned replay for SAC-Lag, inspired by arXiv:2603.05066.

Rewards and environment interaction remain nominal. Each replay transition is
paired with a cost specification, shared by its current and next observations.
"""

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp


PARAMETER_NAMES = ('penetration_exponent', 'hazard_weight', 'terminal_penalty')


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
    state_extras = dict(transitions.extras['state_extras'])
    state_extras['cost'] = reconstruct_cost(
        state_extras['hazard_penetration'], state_extras['unhealthy_termination'], parameters
    )
    return transitions._replace(
        observation=condition_observation(transitions.observation, parameters),
        next_observation=condition_observation(transitions.next_observation, parameters),
        extras={**transitions.extras, 'state_extras': state_extras},
    ), ids


def counterfactual_mean_costs(transitions, bank):
    """Replay-distribution costs, NOT on-policy returns of alternative policies."""
    extras = transitions.extras['state_extras']
    return jax.vmap(lambda parameters: jnp.mean(reconstruct_cost(
        extras['hazard_penetration'], extras['unhealthy_termination'], parameters
    )))(jnp.asarray(bank))


def update_multipliers(multipliers, mean_costs, step_bound, learning_rate, maximum):
    violation = mean_costs - step_bound
    return jnp.clip(multipliers + learning_rate * violation, 0.0, maximum), violation
