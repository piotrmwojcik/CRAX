"""Regression and integration checks for cost-conditioned SAC-Lagrangian."""

import functools
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from crax.envs.base import State
from crax.envs.safe_pathway import SafePathway
from crax.envs.wrappers import training as env_wrappers
from training import acting
from training import types
from training.acme import running_statistics
from training.agents.sac import checkpoint
from training.agents.sac_lag import cost_conditioning as cc
from training.agents.sac_lag import networks
from training.agents.sac_lag.train import train
from training.config import build_base_parser


class TinyPathway(SafePathway):
    """Deterministic Pathway fixture exercising the real training/replay stack."""

    agent_xml_file = ''
    torso_link_name = ''
    foot_link_names = ()
    default_healthy_z_range = (0, 1)
    default_healthy_angle_range = (0, 1)
    observation_size = 2
    action_size = 1
    backend = 'generalized'

    def __init__(self):
        self._cost_scaler = 1.5
        self._unhealthy_termination_cost = 5.0
        self._hazard_mocap_ids = (0,)

    def reset(self, key):
        del key
        zero = jnp.array(0.0)
        return State(None, jnp.zeros(2), zero, zero, {'cost': zero}, {
            'cost': zero, 'hazard_penetration': jnp.zeros(1),
            'unhealthy_termination': zero,
        })

    def step(self, state, action):
        step = state.obs[0] + 1
        done = (step >= 3).astype(jnp.float32)
        cost = 1.5 * 0.5 ** 2 + 5.0 * done
        return state.replace(
            obs=jnp.stack((step, action[0])), reward=jnp.array(1.0), done=done,
            metrics={**state.metrics, 'cost': cost},
            info={**state.info, 'cost': cost, 'hazard_penetration': jnp.array([0.5]),
                  'unhealthy_termination': done},
        )


class CostConditioningTest(unittest.TestCase):
    def transition(self, size=32):
        return types.Transition(
            observation=jnp.zeros((size, 2)), next_observation=jnp.ones((size, 2)),
            action=jnp.zeros((size, 1)), reward=jnp.arange(size, dtype=jnp.float32),
            discount=jnp.ones(size), extras={'policy_extras': {}, 'state_extras': {
                'cost': jnp.zeros(size), 'truncation': jnp.zeros(size),
                'hazard_penetration': jnp.tile(jnp.array([0.0, 0.5, 1.0]), (size, 1)),
                'unhealthy_termination': jnp.ones(size),
            }},
        )

    def test_cost_shapes_and_terminal_penalty(self):
        p = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        costs = cc.reconstruct_cost(p, jnp.array([0., 1.]), jnp.array([[2., 1.5, 5.], [1., 2., 3.]]))
        np.testing.assert_allclose(costs, [1.875, 6.0])
        self.assertEqual(float(cc.reconstruct_cost(jnp.zeros(0), 1., (2., 1.5, 5.))), 5.)

    def test_relabel_preserves_rewards_and_shares_next_condition(self):
        batch = self.transition(2048)
        bank = ((2., 1.5, 5.), (1., 2., 3.))
        relabelled, ids = jax.jit(lambda b: cc.relabel(b, jax.random.PRNGKey(5), bank, 0.5))(batch)
        np.testing.assert_array_equal(relabelled.reward, batch.reward)
        np.testing.assert_array_equal(relabelled.discount, batch.discount)
        np.testing.assert_array_equal(relabelled.observation['state'], batch.observation)
        np.testing.assert_array_equal(relabelled.observation['cost_parameters'], relabelled.next_observation['cost_parameters'])
        self.assertAlmostEqual(float(jnp.mean(ids == 0)), 0.5, delta=0.04)
        expected = jnp.where(ids == 0, 6.875, 6.0)
        np.testing.assert_allclose(relabelled.extras['state_extras']['cost'], expected)
        np.testing.assert_array_equal(batch.extras['state_extras']['cost'], jnp.zeros(2048))

    def test_independent_configuration_dual_updates(self):
        multipliers, violation = cc.update_multipliers(jnp.array([0., 1., 1.99]), jnp.array([0., 0., 10.]), 1., 0.1, 2.)
        np.testing.assert_allclose(multipliers, [0., .9, 2.])
        np.testing.assert_allclose(violation, [-1., -1., 9.])
        np.testing.assert_allclose(cc.counterfactual_mean_costs(self.transition(), ((2., 1.5, 5.), (1., 2., 3.))), [6.875, 6.])

    def test_parameters_validation_and_cli(self):
        args = build_base_parser().parse_args(['--alg', 'sac_lag', '--cost_conditioning', '--cost_configurations', '[[1, 1.5, 5]]'])
        self.assertTrue(args.cost_conditioning)
        self.assertEqual(args.cost_configurations, [[1, 1.5, 5]])
        self.assertEqual(len(cc.parameter_bank((2., 1.5, 5.))), 7)
        for values in ((0, 1, 1), (2, -1, 1), (float('nan'), 1, 1), (2, 1)):
            with self.assertRaises(ValueError):
                cc.validate_parameters(values)

    def test_all_networks_condition_and_nominal_default_matches(self):
        network = networks.make_cost_conditioned_networks(
            2, 1, hidden_layer_sizes=(8,), preprocess_observations_fn=running_statistics.normalize
        )
        normalizer = running_statistics.init_state(jnp.zeros(2))
        state = jnp.zeros((4, 2))
        nominal = cc.condition_observation(state, (2., 1.5, 5.))
        alternative = cc.condition_observation(state, (1., 3., 0.))
        for index, net in enumerate((network.policy_network, network.q_network, network.qc_network)):
            params = net.init(jax.random.PRNGKey(index))
            args = () if index == 0 else (jnp.zeros((4, 1)),)
            output = net.apply(normalizer, params, nominal, *args)
            np.testing.assert_allclose(output, net.apply(normalizer, params, state, *args))
            self.assertGreater(float(jnp.max(jnp.abs(output - net.apply(normalizer, params, alternative, *args)))), 1e-6)

    def test_training_and_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            make_policy, params, metrics, _ = train(
                TinyPathway(), num_timesteps=16, episode_length=4, num_envs=2,
                num_eval_envs=2, num_evals=2, batch_size=4, grad_updates_per_step=1,
                min_replay_size=2, max_replay_size=32, normalize_observations=True,
                network_factory=functools.partial(networks.make_sac_lag_networks, hidden_layer_sizes=(8,)),
                cost_conditioning=True, cost_configurations=[(1., 2., 3.)],
                observed_episode_length_bound=True, save_checkpoint_path=directory,
            )
            folder = sorted(Path(directory).iterdir())[-1]
            config = checkpoint.load_config(folder)
            self.assertTrue(config.cost_conditioning)
            self.assertEqual(len(config.cost_multipliers), 2)
            self.assertEqual(config.cost_configurations[0], [2., 1.5, 5.])
            for parameters in (None, (1., 2., 3.)):
                original = make_policy(params, deterministic=True, cost_parameters=parameters)
                restored = checkpoint.load_policy(folder, deterministic=True, cost_parameters=parameters)
                obs, key = jnp.zeros(2), jax.random.PRNGKey(0)
                np.testing.assert_allclose(original(obs, key)[0], restored(obs, key)[0], atol=1e-6)
            self.assertIn('training/cost_configuration_1/lambda', metrics)

    def test_baseline_training_still_uses_scalar_multiplier(self):
        _, _, metrics, _ = train(
            TinyPathway(), num_timesteps=8, episode_length=4, num_envs=2,
            num_eval_envs=2, num_evals=2, batch_size=4, grad_updates_per_step=1,
            min_replay_size=2, max_replay_size=16,
            network_factory=functools.partial(networks.make_sac_lag_networks, hidden_layer_sizes=(8,)),
        )
        self.assertIn('training/lambda_lagr', metrics)
        self.assertNotIn('training/cost_configuration_0/lambda', metrics)

    def test_termination_measurements_survive_autoreset_and_timeout(self):
        for episode_length, expected_terminal in ((2, 0.), (4, 1.)):
            env = env_wrappers.wrap(TinyPathway(), episode_length=episode_length, action_repeat=1)
            state = env.reset(jax.random.split(jax.random.PRNGKey(0), 1))
            def policy(obs, key):
                return jnp.zeros(obs.shape[:-1] + (1,)), {}
            for _ in range(min(episode_length, 3)):
                state, transition = acting.actor_step(
                    env, state, policy, jax.random.PRNGKey(0),
                    extra_fields=('cost', 'hazard_penetration', 'unhealthy_termination'),
                )
            extras = transition.extras['state_extras']
            np.testing.assert_array_equal(extras['unhealthy_termination'], [expected_terminal])
            np.testing.assert_allclose(extras['cost'], cc.reconstruct_cost(
                extras['hazard_penetration'], extras['unhealthy_termination'], (2., 1.5, 5.)
            ))
            np.testing.assert_array_equal(state.obs, jnp.zeros((1, 2)))

    def test_unsupported_training_modes_fail_early(self):
        for kwargs, message in (
            ({'action_repeat': 2}, 'action_repeat'),
            ({'vision_kwargs': {}}, 'flat state'),
            ({'cost_nominal_probability': 0.0}, 'strictly between'),
            ({'lambda_update_fn': lambda *args: None}, 'SAC-Lagrangian only'),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                train(TinyPathway(), num_timesteps=16, episode_length=4,
                      cost_conditioning=True, **kwargs)


if __name__ == '__main__':
    unittest.main()
