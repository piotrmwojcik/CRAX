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
# This file is derived from Brax 0.12.3 (training/agents/sac/networks.py) and has been
# substantially modified by the CRAX Authors: adds cost-Q critic networks alongside the reward critics.

"""SAC-Lagrangian networks — SAC extended with a cost Q-network."""

from typing import Mapping, Sequence, Tuple

from training import distribution
from training import networks
from training import types
from training.types import PRNGKey
import flax
from flax import linen
import jax.numpy as jnp

from training.agents.sac_lag.cost_conditioning import condition_observation, validate_parameters


@flax.struct.dataclass
class SACLagNetworks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    qc_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    nominal_cost_parameters: tuple | None = flax.struct.field(pytree_node=False, default=None)


def make_inference_fn(sac_lag_networks: SACLagNetworks):
    """Creates inference function for the SAC-Lag agent (same as SAC)."""

    def make_policy(
        params: types.PolicyParams, deterministic: bool = False, cost_parameters=None
    ) -> types.Policy:
        if cost_parameters is not None:
            if sac_lag_networks.nominal_cost_parameters is None:
                raise ValueError('Cost parameters require cost-conditioned SAC-Lag networks.')
            cost_parameters = validate_parameters(cost_parameters)

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            if cost_parameters is not None:
                observations = condition_observation(observations, cost_parameters)
            logits = sac_lag_networks.policy_network.apply(*params, observations)
            if deterministic:
                return sac_lag_networks.parametric_action_distribution.mode(logits), {}
            return (
                sac_lag_networks.parametric_action_distribution.sample(
                    logits, key_sample
                ),
                {},
            )

        return policy

    return make_policy


def make_cost_conditioned_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    nominal_cost_parameters: Sequence[float] = (2.0, 1.5, 5.0),
) -> SACLagNetworks:
    """Condition all three networks, normalizing only the physical state.

    Ordinary observations select the nominal configuration. Explicit conditioned
    observations carry physical state and three cost parameters separately.
    Parameter features use log1p so zero weights remain finite and distinguishable.
    """
    nominal = validate_parameters(nominal_cost_parameters)
    underlying = make_sac_lag_networks(
        observation_size=observation_size + 3,
        action_size=action_size,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        policy_network_layer_norm=policy_network_layer_norm,
        q_network_layer_norm=q_network_layer_norm,
    )

    def features(normalizer, observation):
        if isinstance(observation, Mapping):
            state, parameters = observation['state'], observation['cost_parameters']
        else:
            state = observation
            parameters = jnp.broadcast_to(jnp.asarray(nominal), state.shape[:-1] + (3,))
        state = preprocess_observations_fn(state, normalizer)
        return jnp.concatenate((state, jnp.log1p(parameters)), axis=-1)

    def wrap(network):
        def apply(normalizer, params, observation, *args):
            return network.apply(None, params, features(normalizer, observation), *args)
        return networks.FeedForwardNetwork(init=network.init, apply=apply)

    return underlying.replace(
        policy_network=wrap(underlying.policy_network),
        q_network=wrap(underlying.q_network),
        qc_network=wrap(underlying.qc_network),
        nominal_cost_parameters=nominal,
    )


def make_sac_lag_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
) -> SACLagNetworks:
    """Make SAC-Lag networks (adds cost Q-network to standard SAC)."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=policy_network_layer_norm,
    )
    q_network = networks.make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
    )
    qc_network = networks.make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
    )
    return SACLagNetworks(
        policy_network=policy_network,
        q_network=q_network,
        qc_network=qc_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_sac_lag_networks_vision(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    normalise_channels: bool = False,
    policy_obs_key: str = '',
    value_obs_key: str = '',
) -> SACLagNetworks:
    """Make vision SAC-Lag networks.

    Policy and both Q-critics (reward + cost) each get their own CNN encoder
    (Nature DQN backbone) fed by 'pixels/<camera>' observations, optionally
    concatenated with a 'state' slice when `policy_obs_key`/`value_obs_key`
    is set (--vision_obs_mode pixels+state).
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = networks.make_policy_network_vision(
        observation_size=observation_size,
        output_size=parametric_action_distribution.param_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=policy_network_layer_norm,
        state_obs_key=policy_obs_key,
        normalise_channels=normalise_channels,
    )
    q_network = networks.make_q_network_vision(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
        state_obs_key=value_obs_key,
        normalise_channels=normalise_channels,
    )
    qc_network = networks.make_q_network_vision(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
        state_obs_key=value_obs_key,
        normalise_channels=normalise_channels,
    )
    return SACLagNetworks(
        policy_network=policy_network,
        q_network=q_network,
        qc_network=qc_network,
        parametric_action_distribution=parametric_action_distribution,
    )
