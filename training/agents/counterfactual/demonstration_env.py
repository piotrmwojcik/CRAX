"""Reset online policies onto saved demonstration boards with real added hazards."""
import json

import jax
import jax.numpy as jnp
import numpy as np

from crax import envs
from crax.envs.base import Wrapper
from scripts.compare_fake_real_hazard import unwrap


def load_boards(dataset, manifest):
    config = manifest['configuration']
    records = [json.loads(line) for line in
               (dataset / manifest['lidar_edit']['placements']).read_text().splitlines() if line.strip()]
    records.sort(key=lambda r: (r['env_index'], r['episode_index']))
    ids = [(r['env_index'], r['episode_index']) for r in records]
    if len(ids) != manifest['total_completed_episodes'] or len(set(ids)) != len(ids):
        raise ValueError('Expected one unique hazard-placement record per demonstration episode')
    cap = int(manifest['lidar_edit']['max_added_hazards'])
    if cap < 1:
        raise ValueError('Demonstration hazard cap must be positive')
    with np.load(dataset / 'initial_states.npz', allow_pickle=False) as file:
        initial = {key: file[key] for key in ('q', 'qd', 'hazard_positions')}
    boards = {key: [] for key in ('q', 'qd', 'positions', 'widths')}
    for record in records:
        lane = record['env_index']
        if not 0 <= lane < len(initial['q']):
            raise ValueError(f'Invalid demonstration lane {lane}')
        positions = np.asarray(record.get('positions', []), dtype=np.float32).reshape(-1, 3)
        widths = np.asarray(record.get('half_lengths', []), dtype=np.float32)
        if widths.shape != (len(positions),) or len(widths) > cap or np.any(widths <= 0):
            raise ValueError('Invalid added hazard widths in demonstration')
        # Invisible, zero-extent slots preserve static JAX shapes for boards
        # with fewer hazards, including unchanged demonstrations.
        added = np.tile(np.array([1.e6, 0., .01], dtype=np.float32), (cap, 1))
        added[:len(positions)] = positions
        extents = np.zeros(cap, dtype=np.float32)
        extents[:len(widths)] = widths
        boards['q'].append(initial['q'][lane])
        boards['qd'].append(initial['qd'][lane])
        boards['positions'].append(np.concatenate([initial['hazard_positions'][lane], added]))
        boards['widths'].append(extents)
    if not records:
        raise ValueError('Empty demonstration board bank')
    bank = {key: np.asarray(values, dtype=np.float32) for key, values in boards.items()}
    if not all(np.isfinite(value).all() for value in bank.values()):
        raise ValueError('Nonfinite demonstration board state')
    return bank, ids, cap, float(config['platform_half_width'])


class DemonstrationBoards(Wrapper):
    """Each reset samples one stored episode's board, uniformly with replacement."""
    def __init__(self, real, bank, cap, half_width):
        super().__init__(real)
        self.bank = jax.tree.map(jnp.asarray, bank)
        self.board_count = len(bank['q'])
        self.cap, self.half_width = cap, half_width
        self.original_extents = real._hazard_half_extents[:-cap]
        real._hazard_is_rect = real._hazard_is_rect.at[-cap:].set(True)
        real._hazard_radii = real._hazard_radii.at[-cap:].set(0.)

    def reset(self, key):
        board_key, state_key = jax.random.split(key)
        index = jax.random.randint(board_key, (), 0, self.board_count)
        state = self.env.reset(state_key)
        positions = self.bank['positions'][index]
        ps = self.env.pipeline_init(self.bank['q'][index], self.bank['qd'][index])
        ps = ps.replace(mocap_pos=ps.mocap_pos.at[jnp.asarray(self.env._hazard_mocap_ids)].set(positions))
        return state.replace(pipeline_state=ps, obs=self.env._get_obs(ps, positions),
            info=dict(state.info, hazard_positions=positions, demonstration_board=index,
                      demonstration_widths=self.bank['widths'][index]))

    def step(self, state, action):
        # Widths vary per board/lane. As in hazard_comparison.make_branch_runner,
        # install runtime extents during JAX tracing and restore Python state.
        saved = self.env._hazard_half_extents
        widths = state.info['demonstration_widths']
        try:
            self.env._hazard_half_extents = jnp.concatenate([self.original_extents,
                jnp.stack([widths, jnp.full_like(widths, self.half_width)], axis=1)])
            return self.env.step(state, action)
        finally:
            self.env._hazard_half_extents = saved


def make_demonstration_env(dataset, manifest):
    bank, ids, cap, half_width = load_boards(dataset, manifest)
    config = manifest['configuration']
    base = unwrap(envs.get_environment(config['env'], level=config['level']))
    real = unwrap(envs.get_environment(config['env'], level=config['level'],
                                     num_hazards=base._num_hazards + cap))
    if bank['positions'].shape[1] != real._num_hazards:
        raise ValueError('Saved original hazard count does not match the configured environment')
    return DemonstrationBoards(real, bank, cap, half_width), ids
