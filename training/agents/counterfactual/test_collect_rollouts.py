"""Small deterministic checks of PPO collection, boundaries, and saved layout."""
import unittest
from collections import namedtuple
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from crax.envs.base import State
from crax.envs.wrappers import training as wrappers
from training import acting
from training.agents.counterfactual.collect_rollouts import (
    EXTRA_FIELDS, initialize, make_collector, make_episode_finisher, transition_arrays,
)
from training.agents.counterfactual.hazard_comparison import PhysicsRecorder, iter_episodes


class ToyEnv:
    def reset(self, key):
        obs = jnp.array([0., jax.random.uniform(key)])
        return State(pipeline_state=obs, obs=obs, reward=jnp.array(0.),
                     done=jnp.array(0.), metrics={'cost': jnp.array(0.)},
                     info={'cost': jnp.array(0.)})

    def step(self, state, action):
        obs = state.obs.at[0].add(1.)
        cost = action[0] ** 2
        return state.replace(pipeline_state=obs, obs=obs, reward=action[0],
                             done=(obs[0] >= 3).astype(jnp.float32),
                             metrics={'cost': cost}, info=dict(state.info, cost=cost))


def make_policy(params, deterministic=False):
    def policy(obs, key):
        action = jax.random.uniform(key, obs.shape[:-1] + (1,))
        return action, {'raw_action': action, 'log_prob': jnp.zeros(obs.shape[:-1])}
    return policy


class CollectionTest(unittest.TestCase):
    def test_recording_preserves_terminal_physics_and_joins_episode_fragments(self):
        Position = namedtuple('Position', 'pos')
        Physics = namedtuple('Physics', 'q qd x')

        class PhysicalToy(ToyEnv):
            def physical(self, state):
                return state.replace(pipeline_state=Physics(state.obs, jnp.zeros_like(state.obs),
                                                           Position(state.obs[None])))

            def reset(self, key):
                return self.physical(super().reset(key))

            def step(self, state, action):
                return self.physical(super().step(state, action))

        env = wrappers.wrap(PhysicsRecorder(PhysicalToy()), episode_length=2)
        plain = wrappers.wrap(PhysicalToy(), episode_length=2)
        state, key = initialize(env, 1, 2)
        plain_state, plain_key = initialize(plain, 1, 2)
        collect = make_collector(env, make_policy, 2, 3, 2, 1,
                                 EXTRA_FIELDS + ('counterfactual_state',))
        plain_collect = make_collector(plain, make_policy, 2, 3, 2, 1)
        manifest = {'configuration': {'num_envs': 2}, 'batches': []}
        with tempfile.TemporaryDirectory() as tmp:
            for index in range(2):
                state, key, data = collect(state, (), key)
                plain_state, plain_key, reference = plain_collect(plain_state, (), plain_key)
                np.testing.assert_array_equal(data.action, reference.action)
                np.testing.assert_array_equal(data.observation, reference.observation)
                arrays = transition_arrays(data)
                arrays['valid'] = np.ones(data.discount.shape, dtype=bool)
                filename = f'batch_{index}.npz'
                np.savez(Path(tmp) / filename, **arrays)
                manifest['batches'].append({'file': filename})
            episodes = list(iter_episodes(tmp, manifest))
        self.assertEqual(len(episodes), 6)
        self.assertEqual({(lane, ep) for lane, ep, _ in episodes},
                         {(lane, ep) for lane in range(2) for ep in range(3)})
        for _, _, episode in episodes:
            self.assertEqual(len(episode['reward']), 2)
            np.testing.assert_array_equal(episode['observation'][:, 0], [0, 1])
            # True terminal physical state is 2, even though autoreset obs is 0.
            np.testing.assert_array_equal(episode['extras/state_extras/counterfactual_state/q'][:, 0], [1, 2])

    def test_finishes_only_pending_episodes(self):
        class VariableEnv(ToyEnv):
            def step(self, state, action):
                successor = super().step(state, action)
                limit = jnp.where(state.obs[1] < .5, 2, 5)
                return successor.replace(done=(successor.obs[0] >= limit).astype(jnp.float32))

        env = wrappers.wrap(VariableEnv(), episode_length=10)
        state, key = initialize(env, seed=1, num_envs=16)
        collect = make_collector(env, make_policy, 16, 2, 16, 1)
        state, key, _ = collect(state, (), key)
        active = ~state.done.astype(bool)
        pending = np.asarray(active).copy()
        self.assertTrue(pending.any())
        self.assertTrue((~pending).any())
        finish = make_episode_finisher(env, make_policy, 2)
        counts = np.zeros(16, dtype=int)
        terminals = np.zeros(16, dtype=int)
        frozen = np.asarray(state.obs)[~pending].copy()
        for _ in range(2):
            state, key, active, data, valid = finish(state, (), key, active)
            valid = np.asarray(valid)
            counts += valid.sum(axis=1)
            terminals += (np.asarray(data.extras['state_extras']['episode_done']).astype(bool) & valid).sum(axis=1)
            np.testing.assert_array_equal(np.asarray(state.obs)[~pending], frozen)
        self.assertFalse(np.asarray(active).any())
        np.testing.assert_array_equal(counts, pending.astype(int) * 3)
        np.testing.assert_array_equal(terminals, pending.astype(int))
        previous = state
        state, _, _, _, valid = finish(state, (), key, active)
        self.assertFalse(np.asarray(valid).any())
        for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(previous)):
            np.testing.assert_array_equal(a, b)

    def test_matches_ppo_loop_across_batches_and_episode_boundaries(self):
        # Horizon 2 forces truncation before ToyEnv's natural termination.
        env = wrappers.wrap(ToyEnv(), episode_length=2)
        state, key = initialize(env, seed=1, num_envs=2)
        reference_state, reference_key = state, key
        initial_obs = np.asarray(state.obs)
        collect = make_collector(env, make_policy, 2, 3, 2, 2)
        for _ in range(2):
            state, key, data = collect(state, (), key)
            _, scan_key, reference_key = jax.random.split(reference_key, 3)
            pieces = []
            for _ in range(2):
                action_key, scan_key = jax.random.split(scan_key)
                reference_state, piece = acting.generate_unroll(
                    env, reference_state, make_policy(()), action_key, 3,
                    extra_fields=EXTRA_FIELDS)
                pieces.append(piece)
            reference = jax.tree.map(lambda *x: jnp.stack(x), *pieces)
            reference = jax.tree.map(lambda x: x.swapaxes(1, 2), reference)
            reference = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), reference)
            for expected, actual in zip(jax.tree.leaves((reference_state, reference_key, reference)),
                                        jax.tree.leaves((state, key, data))):
                np.testing.assert_allclose(actual, expected, atol=1e-6)
            arrays = transition_arrays(data)
            self.assertEqual(arrays['observation'].shape, (4, 3, 2))
            done = arrays['extras/state_extras/episode_done'].astype(bool)
            self.assertTrue(done.any())
            np.testing.assert_array_equal(arrays['discount'][done], 0.)
            np.testing.assert_array_equal(arrays['extras/state_extras/truncation'][done], 1.)
            for fragment in range(4):
                terminal = arrays['next_observation'][fragment][done[fragment]]
                for obs in terminal:
                    np.testing.assert_array_equal(obs, initial_obs[fragment % 2])

    def test_invalid_batch_sizes(self):
        with self.assertRaises(ValueError):
            make_collector(None, None, 3, 8, 2, 2)


if __name__ == '__main__':
    unittest.main()
