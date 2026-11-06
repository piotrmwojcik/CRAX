"""Compare collected PPO episodes against the same policy with real hazards."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from crax import envs
from crax.envs.base import Wrapper
from scripts.compare_fake_real_hazard import choose_hazard, unwrap


class PhysicsRecorder(Wrapper):
    """Capture actual successor physics before AutoResetWrapper replaces it."""
    @staticmethod
    def record(state):
        ps = state.pipeline_state
        return state.replace(info=dict(state.info, counterfactual_state={
            'q': ps.q, 'qd': ps.qd, 'xyz': ps.x.pos}))

    def reset(self, key):
        return self.record(self.env.reset(key))

    def step(self, state, action):
        return self.record(self.env.step(state, action))


def iter_episodes(output, manifest):
    """Read valid fragments in lane order and join across files until done."""
    num_envs = manifest['configuration']['num_envs']
    pending = [[] for _ in range(num_envs)]
    episode_ids = np.zeros(num_envs, dtype=int)
    names = ['observation', 'action', 'reward', 'extras/state_extras/cost',
             'extras/policy_extras/raw_action'] + [
        f'extras/state_extras/counterfactual_state/{key}' for key in ('q', 'qd', 'xyz')]
    for entry in manifest['batches']:
        with np.load(Path(output) / entry['file'], allow_pickle=False) as file:
            arrays = {name: file[name] for name in names}
            dones = file['extras/state_extras/episode_done'].astype(bool)
            valid = file['valid']
        for fragment in range(len(valid)):
            lane = fragment % num_envs
            length = int(valid[fragment].sum())
            if not np.all(valid[fragment, :length]) or np.any(valid[fragment, length:]):
                raise ValueError('Expected contiguous valid entries in each fragment')
            start = 0
            for end in np.flatnonzero(dones[fragment, :length]) + 1:
                pending[lane].append({name: data[fragment, start:end].copy()
                                      for name, data in arrays.items()})
                episode = {name: np.concatenate([piece[name] for piece in pending[lane]])
                           for name in names}
                yield lane, int(episode_ids[lane]), episode
                episode_ids[lane] += 1
                pending[lane] = []
                start = end
            if start < length:
                pending[lane].append({name: data[fragment, start:length].copy()
                                      for name, data in arrays.items()})
    if any(pending):
        raise ValueError('Collected dataset contains incomplete episodes')


def make_branch_runner(real, network, params, horizon, count, half_width):
    """Compile once; hazard positions and extents remain runtime inputs."""
    original_extents = real._hazard_half_extents[:-count]
    real._hazard_is_rect = real._hazard_is_rect.at[-count:].set(True)
    real._hazard_radii = real._hazard_radii.at[-count:].set(0.)
    distribution = network.parametric_action_distribution

    @jax.jit
    def recover_noise(obs, raw):
        logits = network.policy_network.apply(params[0], params[1], obs)
        dist = distribution.create_dist(logits)
        return (raw - dist.loc) / dist.scale

    def step_with_extents(state, action, widths):
        saved = real._hazard_half_extents
        try:
            real._hazard_half_extents = jnp.concatenate([
                original_extents, jnp.stack([widths, jnp.full_like(widths, half_width)], axis=1)])
            return real.step(state, action)
        finally:
            real._hazard_half_extents = saved

    @jax.jit
    def run(q, qd, positions, widths, noise):
        initial = real.reset(jax.random.PRNGKey(0))
        ps = real.pipeline_init(q, qd)
        ps = ps.replace(mocap_pos=ps.mocap_pos.at[jnp.asarray(real._hazard_mocap_ids)].set(positions))
        initial = initial.replace(pipeline_state=ps, obs=real._get_obs(ps, positions),
                                  info=dict(initial.info, hazard_positions=positions))

        def body(state, epsilon):
            logits = network.policy_network.apply(params[0], params[1], state.obs)
            dist = distribution.create_dist(logits)
            action = distribution.postprocess(dist.loc + dist.scale * epsilon)
            valid = ~state.done.astype(bool)
            next_state = jax.lax.cond(valid, lambda s: step_with_extents(s, action, widths),
                                      lambda s: s, state)
            added_cost = jnp.sum(next_state.info['hazard_penetration'][-count:] ** 2) * real._cost_scaler
            return next_state, dict(observation=state.obs, next_observation=next_state.obs, action=action,
                reward=jnp.where(valid, next_state.reward, 0.),
                cost=jnp.where(valid, next_state.info['cost'], 0.),
                added_hazard_cost=jnp.where(valid, added_cost, 0.),
                done=next_state.done, valid=valid)

        result = jax.lax.scan(body, initial, noise, length=horizon)[1]
        result['truncation'] = result['valid'] & (jnp.arange(horizon) == horizon-1) & ~result['done'].astype(bool)
        result['done'] = result['done'].astype(bool) | result['truncation']
        return result

    return recover_noise, run


def compare_episodes(args, manifest, network, params):
    base = unwrap(envs.get_environment(args.env, level=args.level))
    real = unwrap(envs.get_environment(args.env, level=args.level,
                                      num_hazards=base._num_hazards + args.num_added_hazards))
    recover_noise, run = make_branch_runner(real, network, params, args.episode_length,
                                            args.num_added_hazards, args.platform_half_width)
    with np.load(args.output / 'initial_states.npz', allow_pickle=False) as file:
        initials = {key: file[key] for key in file.files}
    branch_dir = args.output / 'real_rollouts'
    branch_dir.mkdir()
    rows = []
    started = time.perf_counter()
    total = manifest['total_completed_episodes']
    print(f'PHASE 2 START: {total} episodes, {args.num_added_hazards} simultaneous hazards each.', flush=True)
    with (args.output / 'hazard_comparisons.jsonl').open('w') as log:
        for pair, (lane, episode_id, episode) in enumerate(iter_episodes(args.output, manifest)):
            reward = episode['reward']
            nominal = dict(reward=float(reward.sum(dtype=np.float64)),
                           cost=float(episode['extras/state_extras/cost'].sum(dtype=np.float64)),
                           steps=len(reward), added_hazard_cost=0.)
            row = dict(pair=pair, env_index=lane, episode_index=episode_id,
                       num_added_hazards=args.num_added_hazards, nominal=nominal)
            physical = {}
            for name in ('q', 'qd', 'xyz'):
                physical[name] = np.concatenate([initials[name][lane:lane+1],
                    episode[f'extras/state_extras/counterfactual_state/{name}']], axis=0)
            hazard_positions = initials['hazard_positions'][lane]
            states = [SimpleNamespace(pipeline_state=SimpleNamespace(q=q, qd=qd,
                      x=SimpleNamespace(pos=xyz)), info={'hazard_positions': hazard_positions})
                      for q, qd, xyz in zip(physical['q'], physical['qd'], physical['xyz'])]
            try:
                positions, widths = choose_hazard(base, states,
                    np.random.default_rng(np.random.SeedSequence([args.hazard_seed, lane, episode_id])),
                    count=args.num_added_hazards)
                # Verify all recorded nominal states against the actual rectangular cost region.
                feet = physical['xyz'][:, base._foot_link_ids]
                delta = np.abs(feet[:, :, None, :2] - positions[None, None, :, :2])
                normalized = delta / np.stack([widths, np.full_like(widths, args.platform_half_width)], axis=1)
                penetration = np.maximum(0., 1. - normalized.max(axis=-1))
                grounded = feet[..., 2] < real._hazard_height * 2 + .08
                if np.any(penetration * grounded[..., None] > 0):
                    raise ValueError('Nominal trajectory violates proposed hazards')
            except ValueError as error:
                row.update(status='skipped', reason=str(error))
            else:
                # Reuse nominal Gaussian innovations, not nominal actions: the
                # real policy still responds to its own hazard-visible observations.
                horizon = args.episode_length
                obs = np.pad(episode['observation'], ((0, horizon-len(reward)), (0, 0)))
                raw = np.pad(episode['extras/policy_extras/raw_action'], ((0, horizon-len(reward)), (0, 0)))
                noise = recover_noise(jnp.asarray(obs), jnp.asarray(raw))
                extension_key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(args.seed), lane), episode_id)
                extension = jax.random.normal(extension_key, noise.shape)
                noise = jnp.where(jnp.arange(horizon)[:, None] < len(reward), noise, extension)
                result = jax.device_get(run(jnp.asarray(initials['q'][lane]),
                    jnp.asarray(initials['qd'][lane]), jnp.asarray(np.concatenate([hazard_positions, positions])),
                    jnp.asarray(widths, dtype=jnp.float32), noise))
                length = int(result['valid'].sum())
                metrics = dict(reward=float(result['reward'].sum(dtype=np.float64)),
                               cost=float(result['cost'].sum(dtype=np.float64)), steps=length,
                               added_hazard_cost=float(result['added_hazard_cost'].sum(dtype=np.float64)))
                metrics['end_reason'] = 'time_limit' if result['truncation'][length-1] else 'environment_done'
                metrics['cost_without_added_hazards'] = metrics['cost'] - metrics['added_hazard_cost']
                nominal['cost_without_added_hazards'] = nominal['cost']
                winner = ('nominal' if nominal['reward'] > metrics['reward'] and nominal['cost'] < metrics['cost']
                          else 'real' if metrics['reward'] > nominal['reward'] and metrics['cost'] < nominal['cost']
                          else 'none')
                filename = f'real_rollouts/pair_{pair:06d}.npz'
                np.savez_compressed(args.output / filename, **{k: v[:length] for k, v in result.items()})
                row.update(status='compared', real=metrics, winner=winner,
                           positions=positions.tolist(), half_lengths=widths.tolist(),
                           nominal_added_hazard_penetration=0., real_rollout=filename)
            rows.append(row)
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(f"Pair {pair+1}/{total}: " + json.dumps(row), flush=True)
    if len(rows) != total:
        raise RuntimeError(f'Expected {total} completed episodes, read {len(rows)}')
    compared = [row for row in rows if row['status'] == 'compared']
    summary = dict(total_episodes=len(rows), compared=len(compared), skipped=len(rows)-len(compared),
                   seconds=time.perf_counter()-started,
                   winners={label: sum(r['winner'] == label for r in compared) for label in ('nominal', 'real', 'none')},
                   means={label: {metric: float(np.mean([r[label][metric] for r in compared]))
                          for metric in ('reward', 'cost', 'steps', 'added_hazard_cost', 'cost_without_added_hazards')}
                          for label in ('nominal', 'real')} if compared else {},
                   details='hazard_comparisons.jsonl',
                   randomness='shared Gaussian innovations until nominal termination; seeded extension thereafter')
    (args.output / 'hazard_summary.json').write_text(json.dumps(summary, indent=2))
    print('PHASE 2 COMPLETE: ' + json.dumps(summary), flush=True)
    return summary
