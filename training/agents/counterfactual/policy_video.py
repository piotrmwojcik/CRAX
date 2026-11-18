"""Recorded nominal demonstration beside a fresh current-policy hazard rollout."""
from contextlib import ExitStack
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from crax import envs
from scripts.compare_fake_real_hazard import render, unwrap
from training.agents.counterfactual.hazard_comparison import iter_episodes

PREFIX = 'extras/state_extras/counterfactual_state/'
RED = (1., 0., 0., 1.)
YELLOW = (1., 1., 0., 1.)


def prepare_example(dataset, manifest):
    """Pick one fixed, successfully edited expert episode for the whole run."""
    records = [json.loads(line) for line in
               (dataset / manifest['lidar_edit']['placements']).read_text().splitlines() if line.strip()]
    eligible = sorted((r for r in records if r.get('positions')),
                      key=lambda r: (r['env_index'], r['episode_index']))
    if not eligible:
        raise ValueError('Video requires at least one demonstration with added hazard placements')
    placement = eligible[0]
    lane, episode_id = placement['env_index'], placement['episode_index']
    for ln, ep, episode in iter_episodes(dataset, manifest):
        if (ln, ep) == (lane, episode_id):
            break
    else:
        raise ValueError('Selected video episode is missing from the dataset')
    with np.load(dataset / 'initial_states.npz', allow_pickle=False) as initial:
        q = np.concatenate([initial['q'][lane:lane+1], episode[PREFIX+'q']])
        qd = np.concatenate([initial['qd'][lane:lane+1], episode[PREFIX+'qd']])
        original_positions = initial['hazard_positions'][lane].copy()
    return dict(env_index=lane, episode_index=episode_id, q=q, qd=qd,
                positions=np.asarray(placement['positions']),
                widths=np.asarray(placement['half_lengths']), original_positions=original_positions,
                reward=episode['reward'], cost=episode['extras/state_extras/cost'])


def make_video_rollout(real, network, normalizer, initial, horizon, seed):
    """Record physics during fresh inference; never replay archived real actions."""
    @jax.jit
    def run(actor):
        def step(carry, unused):
            state, key = carry
            key, action_key = jax.random.split(key)
            logits = network.policy_network.apply(normalizer, actor, state.obs)
            action = network.parametric_action_distribution.sample(logits, action_key)
            valid = ~state.done.astype(bool)
            successor = jax.lax.cond(valid, lambda s: real.step(s, action), lambda s: s, state)
            return (successor, key), dict(q=successor.pipeline_state.q, qd=successor.pipeline_state.qd,
                reward=jnp.where(valid, successor.reward, 0.),
                cost=jnp.where(valid, successor.info['cost'], 0.), valid=valid)
        return jax.lax.scan(step, (initial, jax.random.PRNGKey(seed)), None, length=horizon)[1]
    return run


def color_hazards(model, original_count, widths=None, half_width=1., height=.01):
    import mujoco
    extra_count = 0 if widths is None else len(widths)
    for index in range(1, original_count + extra_count + 1):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'hazard{index}')
        if body < 0:
            raise ValueError(f'Missing rendering body hazard{index}')
        geom = int(model.body_geomadr[body])
        model.geom_rgba[geom] = RED if index <= original_count else YELLOW
        if index > original_count:
            model.geom_type[geom] = mujoco.mjtGeom.mjGEOM_BOX
            model.geom_size[geom, :3] = [widths[index-original_count-1], half_width, height]


class PairedPolicyVideo:
    def __init__(self, dataset, manifest, network, normalizer, horizon, seed=20000,
                 stride=4, camera='fixedfar'):
        self.example = prepare_example(dataset, manifest)
        self.seed, self.stride, self.camera = seed, stride, camera
        config = manifest['configuration']
        self.half_width = config['platform_half_width']
        self.base = unwrap(envs.get_environment(config['env'], level=config['level']))
        count = len(self.example['widths'])
        self.real = unwrap(envs.get_environment(config['env'], level=config['level'],
                                               num_hazards=self.base._num_hazards + count))
        real = self.real
        real._hazard_is_rect = real._hazard_is_rect.at[-count:].set(True)
        real._hazard_radii = real._hazard_radii.at[-count:].set(0.)
        real._hazard_half_extents = real._hazard_half_extents.at[-count:].set(jnp.asarray(
            np.column_stack([self.example['widths'], np.full(count, self.half_width)])))
        positions = jnp.asarray(np.concatenate([self.example['original_positions'], self.example['positions']]))
        initial = real.reset(jax.random.PRNGKey(0))
        ps = real.pipeline_init(jnp.asarray(self.example['q'][0]), jnp.asarray(self.example['qd'][0]))
        ps = ps.replace(mocap_pos=ps.mocap_pos.at[jnp.asarray(real._hazard_mocap_ids)].set(positions))
        initial = initial.replace(pipeline_state=ps, obs=real._get_obs(ps, positions),
                                  info=dict(initial.info, hazard_positions=positions))
        self.real_template = ps
        nominal = self.base.pipeline_init(jnp.asarray(self.example['q'][0]), jnp.asarray(self.example['qd'][0]))
        self.nominal_template = nominal.replace(mocap_pos=nominal.mocap_pos.at[
            jnp.asarray(self.base._hazard_mocap_ids)].set(self.example['original_positions']))
        self.rollout = make_video_rollout(real, network, normalizer, initial, horizon, seed)

    def write(self, actor, directory, iteration):
        import imageio.v2 as iio
        import mujoco
        from PIL import Image, ImageDraw

        result = jax.device_get(self.rollout(actor))
        length = int(result['valid'].sum())
        real_q = np.concatenate([self.example['q'][:1], result['q'][:length]])
        real_qd = np.concatenate([self.example['qd'][:1], result['qd'][:length]])
        if not np.isfinite(real_q).all() or not np.isfinite(real_qd).all():
            raise ValueError('Nonfinite physics in diagnostic video rollout')
        color_hazards(self.base.sys.mj_model, self.base._num_hazards)
        color_hazards(self.real.sys.mj_model, self.base._num_hazards,
                      self.example['widths'], self.half_width, self.real._hazard_height)
        directory.mkdir(parents=True, exist_ok=False)
        filename = directory / 'current_vs_nominal.mp4'
        fps = 1. / float(self.base.dt) / self.stride
        trajectories = [(real_q, real_qd, result['reward'][:length], result['cost'][:length],
                         self.real.sys.mj_model, self.real_template, None,
                         'CURRENT POLICY | added hazards: YELLOW'),
                        (self.example['q'], self.example['qd'], self.example['reward'], self.example['cost'],
                         self.base.sys.mj_model, self.nominal_template, self.example['positions'],
                         'NOMINAL EXPERT (recorded) | added hazards: BLUE')]
        frames = list(range(0, max(len(real_q), len(self.example['q'])), self.stride))
        last = max(len(real_q), len(self.example['q'])) - 1
        if frames[-1] != last:
            frames.append(last)
        with ExitStack() as stack:
            renderers = [stack.enter_context(mujoco.Renderer(t[4], height=480, width=640)) for t in trajectories]
            writer = stack.enter_context(iio.get_writer(filename, fps=fps, codec='libx264'))
            totals = [(np.r_[0., np.cumsum(t[2])], np.r_[0., np.cumsum(t[3])]) for t in trajectories]
            for step in frames:
                panels = []
                for index, (q, qd, _, _, model, template, added, label) in enumerate(trajectories):
                    t = min(step, len(q)-1)
                    physics = SimpleNamespace(q=q[t], qd=qd[t], mocap_pos=template.mocap_pos,
                                              mocap_quat=template.mocap_quat)
                    pixels = render(renderers[index], model, SimpleNamespace(pipeline_state=physics),
                                    added, self.example['widths'], self.camera, self.half_width)
                    panel = Image.new('RGB', (640, 576), '#141820')
                    panel.paste(Image.fromarray(pixels), (0, 96))
                    ImageDraw.Draw(panel).multiline_text((10, 8),
                        f'{label}\nIteration {iteration} | Step {t}/{len(q)-1} | '
                        f'{"ENDED" if t == len(q)-1 else "RUNNING"}\n'
                        f'Reward {totals[index][0][t]:.3f} | Cost {totals[index][1][t]:.3f}\n'
                        'Original hazards: RED | fixed demonstration-board diagnostic', fill='white', spacing=3)
                    panels.append(np.asarray(panel))
                writer.append_data(np.concatenate(panels, axis=1))
        metadata = dict(iteration=iteration, env_index=self.example['env_index'],
                        episode_index=self.example['episode_index'], seed=self.seed,
                        added_hazards=len(self.example['widths']), current_steps=length,
                        nominal_steps=len(self.example['reward']),
                        current_reward=float(result['reward'].sum()), current_cost=float(result['cost'].sum()),
                        semantics='Fresh current-policy rollout; recorded nominal physics; same initial state and layout; '
                                  'fixed video randomness across iterations, not shared with recorded nominal actions')
        (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        return filename, fps
