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
# This file is derived from Brax 0.12.3 (training/agents/sac/checkpoint.py) and has been modified
# by the CRAX Authors: adds `load_config`.

"""Checkpointing for SAC."""

from typing import Any, Union

from training import checkpoint
from training import types
from training.agents.sac import networks as sac_networks
from etils import epath
from ml_collections import config_dict

_CONFIG_FNAME = 'sac_network_config.json'


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Any,
    config: config_dict.ConfigDict,
):
  """Saves a checkpoint."""
  return checkpoint.save(path, step, params, config, _CONFIG_FNAME)


def load(
    path: Union[str, epath.Path],
):
  """Loads SAC checkpoint."""
  return checkpoint.load(path)


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[sac_networks.SACNetworks],
) -> config_dict.ConfigDict:
  """Returns a config dict for re-creating a network from a checkpoint."""
  return checkpoint.network_config(
      observation_size, action_size, normalize_observations, network_factory
  )


def _get_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[sac_networks.SACNetworks],
) -> sac_networks.SACNetworks:
  """Generates a SAC network given config."""
  return checkpoint.get_network(config, network_factory)  # pytype: disable=bad-return-type


def load_config(
    path: Union[str, epath.Path],
) -> config_dict.ConfigDict:
  """Loads SAC config from checkpoint."""
  path = epath.Path(path)
  config_path = path / _CONFIG_FNAME
  return checkpoint.load_config(config_path)


def load_policy(
    path: Union[str, epath.Path],
    network_factory: types.NetworkFactory[
        sac_networks.SACNetworks
    ] = sac_networks.make_sac_networks,
    deterministic: bool = True,
    cost_parameters=None,
):
  """Loads policy inference function from SAC checkpoint."""
  path = epath.Path(path)
  config = load_config(path)
  params = load(path)
  if config.get('cost_conditioning', False):
    from training.agents.sac_lag import networks as lag_networks
    from training.agents.sac_lag.cost_conditioning import validate_parameters
    if cost_parameters is not None:
      cost_parameters = validate_parameters(cost_parameters)
    sac_network = _get_network(config, lag_networks.make_cost_conditioned_networks)
    return lag_networks.make_inference_fn(sac_network)(
        params, deterministic=deterministic, cost_parameters=cost_parameters
    )
  if cost_parameters is not None:
    raise ValueError('cost_parameters requires a cost-conditioned SAC-Lag checkpoint.')
  sac_network = _get_network(config, network_factory)
  make_inference_fn = sac_networks.make_inference_fn(sac_network)

  return make_inference_fn(params, deterministic=deterministic)


def load_cost_multiplier(path):
  """Restore lambda(psi) from a continuously cost-conditioned SAC-Lag checkpoint."""
  import jax
  import jax.numpy as jnp
  from training.agents.sac_lag.cost_conditioning import MultiplierNetwork

  config = load_config(path).to_dict()
  if config.get('lambda_network_architecture') != 'mlp_32_32_tanh_sigmoid_v1':
    raise ValueError('Checkpoint does not contain a supported learned lambda(psi).')
  network = MultiplierNetwork(
      tuple(config['network_factory_kwargs']['nominal_cost_parameters']),
      tuple(tuple(pair) for pair in config['cost_parameter_bounds']),
      config['lambda_max'],
  )
  params = jax.tree_util.tree_map(
      jnp.asarray, config['lambda_network_params'], is_leaf=lambda value: isinstance(value, list)
  )
  return lambda cost_parameters: network.apply(params, jnp.asarray(cost_parameters))
