"""Check saved-board reset, lidar reconstruction, and per-lane hazard extents."""
import json
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from training.agents.counterfactual.demonstration_env import load_boards, DemonstrationBoards
from training.agents.counterfactual.test_batched_comparison import BranchEnv


class BoardTest(unittest.TestCase):
    def test_saved_initials_placements_and_empty_board_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            np.savez(path/'initial_states.npz', q=[[1.], [2.]], qd=[[3.], [4.]],
                     hazard_positions=np.array([[[10., 0., .01]], [[20., 0., .01]]]))
            records = [dict(env_index=0, episode_index=0, positions=[[5., 0., .01]], half_lengths=[.7]),
                       dict(env_index=1, episode_index=0, positions=[], half_lengths=[])]
            (path/'fake_hazards.jsonl').write_text('\n'.join(map(json.dumps, records)))
            manifest = dict(configuration=dict(platform_half_width=1.), total_completed_episodes=2,
                            lidar_edit=dict(placements='fake_hazards.jsonl', max_added_hazards=2))
            bank, ids, cap, _ = load_boards(path, manifest)
            self.assertEqual(ids, [(0, 0), (1, 0)])
            self.assertEqual(cap, 2)
            np.testing.assert_allclose(bank['positions'][0, :2, 0], [10., 5.])
            np.testing.assert_array_equal(bank['q'], [[1.], [2.]])
            np.testing.assert_allclose(bank['widths'], [[.7, 0.], [0., 0.]])
            self.assertGreater(bank['positions'][1, 1, 0], 10000)
            records.append(records[0])
            (path/'fake_hazards.jsonl').write_text('\n'.join(map(json.dumps, records)))
            with self.assertRaises(ValueError):
                load_boards(path, manifest)

    def test_batched_resets_restore_boards_and_steps_use_lane_widths(self):
        class VisibleHazards(BranchEnv):
            def _get_obs(self, ps, positions):
                # Probe that reset observations use the saved hazard layout.
                return jnp.concatenate([ps.q, positions[:, 0]])
        real = VisibleHazards()
        bank = dict(q=np.array([[0.], [1.]], np.float32), qd=np.array([[4.], [5.]], np.float32),
                    positions=np.array([[[10.,0.,.01],[20.,0.,.01]],
                                        [[30.,0.,.01],[40.,0.,.01]]], np.float32),
                    widths=np.array([[.2],[.8]], np.float32))
        env = DemonstrationBoards(real, bank, 1, 1.)
        states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(0), 32))
        ids = np.asarray(states.info['demonstration_board'])
        self.assertEqual(set(ids), {0, 1})
        np.testing.assert_array_equal(states.pipeline_state.q, bank['q'][ids])
        np.testing.assert_array_equal(states.pipeline_state.qd, bank['qd'][ids])
        np.testing.assert_array_equal(states.pipeline_state.mocap_pos, bank['positions'][ids])
        np.testing.assert_array_equal(states.obs[:, 1:], bank['positions'][ids, :, 0])
        saved_extents = np.asarray(real._hazard_half_extents).copy()
        successors = jax.jit(jax.vmap(env.step))(states, jnp.zeros((32, 1)))
        expected = bank['widths'][ids, 0] * (bank['q'][ids, 0] + 1)
        np.testing.assert_allclose(successors.info['cost'], expected)
        np.testing.assert_array_equal(real._hazard_half_extents, saved_extents)


if __name__ == '__main__':
    unittest.main()
