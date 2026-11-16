"""Regression checks for the observed-length port from 8419f1d."""

import importlib
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

ppo_lag = importlib.import_module('training.agents.ppo_lag.train')


@struct.dataclass
class State:
    aux_state: object


class ObservedLengthTest(unittest.TestCase):
    def hooks(self, enabled):
        with mock.patch.object(ppo_lag.ppo_train, 'train') as base:
            ppo_lag.train(
                environment=None, num_timesteps=1, episode_length=2000,
                safety_bound=25, lagrangian_coef_rate=3,
                initial_lambda_lagr=1, observed_episode_length_bound=enabled,
            )
        return base.call_args.kwargs

    def metrics(self):
        # Only the final SGD epoch counts; completed episodes have lengths
        # 400, 600 and 800, while unfinished episodes contribute nothing.
        return {
            'mean_cost': jnp.array([[0.5, 0.5], [0.02, 0.02]]),
            'episodes_done': jnp.array([[100., 100.], [2., 1.]]),
            'episode_length_sum': jnp.array([[1., 1.], [1000., 800.]]),
        }

    def test_observed_update_and_no_completed_episodes(self):
        hooks = self.hooks(True)
        state = State(hooks['init_aux_state_fn']())
        update = jax.jit(hooks['post_step_fn'])
        state, metrics = update(state, self.metrics())
        # EMA: .9 * 2000 + .1 * 600 = 1860.
        np.testing.assert_allclose(state.aux_state[1], 1860)
        np.testing.assert_allclose(metrics['per_step_safety_bound'], 25 / 1860)
        np.testing.assert_allclose(state.aux_state[0], [1 + 3 * (.02 - 25 / 1860)])
        empty = {**self.metrics(), 'episodes_done': jnp.zeros((2, 2)),
                 'episode_length_sum': jnp.zeros((2, 2))}
        state, _ = update(state, empty)
        np.testing.assert_allclose(state.aux_state[1], 1860)

    def test_disabled_preserves_fixed_bound(self):
        hooks = self.hooks(False)
        state, metrics = jax.jit(hooks['post_step_fn'])(
            State(hooks['init_aux_state_fn']()), self.metrics())
        np.testing.assert_allclose(state.aux_state, [1 + 3 * (.02 - 25 / 2000)])
        self.assertNotIn('observed_episode_length', metrics)

    def test_loss_receives_multiplier_not_length(self):
        hooks = self.hooks(True)
        with mock.patch.object(ppo_lag.ppo_lag_losses, 'compute_ppo_lagrange_loss') as loss:
            hooks['loss_fn'](None, None, None, None, aux_state=hooks['init_aux_state_fn']())
        np.testing.assert_array_equal(loss.call_args.kwargs['lambda_lagr'], [1.])


if __name__ == '__main__':
    unittest.main()
