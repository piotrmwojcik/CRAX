"""Collect PPO-style transition batches from a frozen PPO/PPO-Lag checkpoint.

Uses the same wrappers, stochastic policy, generate_unroll, scan, and fragment
layout as PPO training, then compare completed episodes with added hazards.
"""
import argparse
import json
from pathlib import Path
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from crax import envs
from crax.envs.wrappers import training as wrappers
from training import acting, checkpoint as checkpoint_io
from training.agents.ppo import checkpoint, networks
from training.agents.sac import networks as sac_networks


EXTRA_FIELDS = ('truncation', 'episode_metrics', 'episode_done', 'cost')


def resolve_checkpoint(path):
    """Accept an exact checkpoint or a run directory (latest numeric step)."""
    path = Path(path).expanduser().resolve()
    def has_config(candidate):
        return any((candidate / name).is_file() for name in
                   ('ppo_network_config.json', 'sac_network_config.json'))
    for candidate in (path, path / '~'):
        if has_config(candidate):
            return candidate
    if path.is_dir():
        for step in sorted((p for p in path.iterdir() if p.is_dir() and p.name.isdigit()),
                           key=lambda p: int(p.name), reverse=True):
            for candidate in (step, step / '~'):
                if has_config(candidate):
                    return candidate
    raise ValueError(f'No PPO or SAC checkpoint found in {path}')


def load_collection_policy(path):
    """Load PPO or plain SAC and retain raw actions needed for noise pairing."""
    if (path / 'ppo_network_config.json').is_file():
        config = checkpoint.load_config(path)
        network = checkpoint_io.get_network(config, networks.make_ppo_networks)
        factory = networks.make_inference_fn(network)
        algorithm = 'ppo'
    else:
        config = checkpoint_io.load_config(path / 'sac_network_config.json')
        if config.get('cost_conditioning', False):
            raise ValueError('Cost-conditioned SAC checkpoints are not supported by this collector')
        network = checkpoint_io.get_network(config, sac_networks.make_sac_networks)
        algorithm = 'sac'

        def factory(params, deterministic=False):
            def policy(obs, key):
                logits = network.policy_network.apply(params[0], params[1], obs)
                distribution = network.parametric_action_distribution
                if deterministic:
                    return distribution.mode(logits), {}
                raw = distribution.sample_no_postprocessing(logits, key)
                return distribution.postprocess(raw), {
                    'raw_action': raw, 'log_prob': distribution.log_prob(logits, raw)}
            return policy

    params = jax.device_put(checkpoint_io.load(path), jax.devices()[0])
    return network, params, factory, algorithm


def initialize(env, seed, num_envs):
    """Match PPO's initial environment and first epoch RNG scheme on one device."""
    _, local_key = jax.random.split(jax.random.PRNGKey(seed))
    local_key = jax.random.fold_in(local_key, 0)
    local_key, key_env, _ = jax.random.split(local_key, 3)
    keys = jax.random.split(key_env, num_envs)
    state = jax.jit(env.reset)(keys)
    # PPO also strips weak types before collection to avoid retracing after reset.
    state = jax.tree.map(lambda x: jnp.asarray(x).astype(jnp.asarray(x).dtype), state)
    epoch_key, _ = jax.random.split(local_key)
    collection_key = jax.random.split(epoch_key, 1)[0]
    return state, collection_key


def make_collector(env, make_policy, num_envs, unroll_length, batch_size, num_minibatches,
                   extra_fields=EXTRA_FIELDS):
    """Return PPO's collection portion with explicit frozen policy parameters."""
    if min(num_envs, unroll_length, batch_size, num_minibatches) < 1:
        raise ValueError('Collection sizes must be positive')
    if batch_size * num_minibatches % num_envs:
        raise ValueError('batch_size * num_minibatches must be divisible by num_envs')

    @jax.jit
    def collect(state, params, key):
        _, key_generate_unroll, next_key = jax.random.split(key, 3)
        policy = make_policy(params, deterministic=False)

        def body(carry, unused):
            current_state, current_key = carry
            current_key, following_key = jax.random.split(current_key)
            successor, data = acting.generate_unroll(
                env, current_state, policy, current_key, unroll_length,
                extra_fields=extra_fields)
            return (successor, following_key), data

        (state, _), data = jax.lax.scan(
            body, (state, key_generate_unroll), (),
            length=batch_size * num_minibatches // num_envs)
        data = jax.tree.map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), data)
        return state, next_key, data

    return collect


def transition_arrays(data):
    """NPZ names use slash-separated Transition field paths; no pickle needed."""
    arrays = {}

    def visit(value, prefix=''):
        if isinstance(value, dict):
            for name, child in value.items():
                visit(child, f'{prefix}/{name}' if prefix else name)
        else:
            arrays[prefix] = np.asarray(value)

    visit(serialization.to_state_dict(data))
    return arrays


def make_episode_finisher(env, make_policy, unroll_length, extra_fields=EXTRA_FIELDS):
    """Finish pending episodes in bounded chunks; freeze finished environments.

    Batched physics may evaluate inactive lanes, but their successor states are
    discarded and their transitions are masked out. No new episode is collected.
    """
    @jax.jit
    def finish(state, params, key, active):
        policy = make_policy(params, deterministic=False)

        def body(carry, unused):
            state, key, active = carry
            action_key, key = jax.random.split(key)
            successor, data = acting.actor_step(
                env, state, policy, action_key, extra_fields=extra_fields)

            def select(new, old):
                mask = active.reshape(active.shape + (1,) * (new.ndim - active.ndim))
                return jnp.where(mask, new, old)

            successor = jax.tree.map(select, successor, state)
            still_active = active & ~successor.done.astype(bool)
            return (successor, key, still_active), (data, active)

        (state, key, active), (data, valid) = jax.lax.scan(
            body, (state, key, active), (), length=unroll_length)
        # Tail chunks contain one fragment per environment.
        data = jax.tree.map(lambda x: x.swapaxes(0, 1), data)
        return state, key, active, data, valid.T

    return finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--env', default='safe_pathway_walker2d')
    parser.add_argument('--level', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num-envs', type=int, default=2048)
    parser.add_argument('--episode-length', type=int, default=2000)
    parser.add_argument('--action-repeat', type=int, default=1)
    parser.add_argument('--unroll-length', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-minibatches', type=int, default=32)
    parser.add_argument('--num-batches', type=int, default=1,
                        help='PPO batches before finishing all pending episodes')
    parser.add_argument('--num-added-hazards', type=int, default=1,
                        help='Simultaneous hazards per episode; 0 disables comparison')
    parser.add_argument('--hazard-seed', type=int, default=42)
    parser.add_argument('--platform-half-width', type=float, default=1.0)
    args = parser.parse_args()
    if min(args.num_envs, args.episode_length, args.action_repeat,
           args.unroll_length, args.batch_size, args.num_minibatches, args.num_batches) < 1:
        parser.error('All counts must be positive')
    if args.batch_size * args.num_minibatches % args.num_envs:
        parser.error('--batch-size * --num-minibatches must be divisible by --num-envs')
    if jax.process_count() != 1 or jax.local_device_count() != 1:
        parser.error('This collector supports one CPU/GPU device on one host')
    if args.output.exists():
        parser.error('Use a new output directory to preserve existing datasets')
    if args.num_added_hazards < 0 or args.platform_half_width <= 0:
        parser.error('Hazard count must be nonnegative and platform half-width positive')
    if args.num_added_hazards and (args.env != 'safe_pathway_walker2d' or args.action_repeat != 1):
        parser.error('Hazard comparison currently requires safe_pathway_walker2d and action-repeat=1')

    path = resolve_checkpoint(args.checkpoint)
    network, params, make_policy, algorithm = load_collection_policy(path)
    base = envs.get_environment(args.env, level=args.level)
    extra_fields = EXTRA_FIELDS
    if args.num_added_hazards:
        from training.agents.counterfactual.hazard_comparison import PhysicsRecorder
        base = PhysicsRecorder(base)
        extra_fields += ('counterfactual_state',)
    env = wrappers.wrap(base,
                        episode_length=args.episode_length, action_repeat=args.action_repeat)
    collect = make_collector(env, make_policy, args.num_envs,
                             args.unroll_length, args.batch_size, args.num_minibatches, extra_fields)
    state, key = initialize(env, args.seed, args.num_envs)
    args.output.mkdir(parents=True)
    if args.num_added_hazards:
        np.savez_compressed(args.output / 'initial_states.npz',
                            **jax.device_get(dict(q=state.pipeline_state.q, qd=state.pipeline_state.qd,
                                                 xyz=state.pipeline_state.x.pos,
                                                 hazard_positions=state.info['hazard_positions'])))
    phase_one_started = time.perf_counter()
    transitions_per_batch = args.batch_size * args.num_minibatches * args.unroll_length
    manifest = dict(
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        checkpoint=str(path), checkpoint_algorithm=algorithm, device=str(jax.devices()[0]), status='collecting',
        policy='frozen, stochastic; checkpoint normalizer frozen',
        reset='automatic reset to each environment\'s stored initial state; no board resampling',
        layout='[fragment, time, ...]; fragment = scan_index * num_envs + env_index',
        transitions_per_batch=transitions_per_batch, batches=[], total_transitions=0,
        total_completed_episodes=0,
        completion='After requested batches, finish pending episodes without starting new ones',
        valid_mask='Use valid to exclude padding in finish files; all regular batch entries are valid')

    def save_manifest():
        temp = args.output / 'manifest.json.tmp'
        temp.write_text(json.dumps(manifest, indent=2))
        temp.replace(args.output / 'manifest.json')

    save_manifest()

    def save_batch(filename, data, valid, phase, started):
        arrays = transition_arrays(jax.device_get(data))
        arrays['valid'] = np.asarray(jax.device_get(valid), dtype=bool)
        # Zero padding so ignored slots cannot contribute reward/cost accidentally.
        for name, values in arrays.items():
            if name != 'valid':
                mask = arrays['valid'].reshape(arrays['valid'].shape + (1,) * (values.ndim - 2))
                arrays[name] = np.where(mask, values, np.zeros((), dtype=values.dtype))
        collection_seconds = time.perf_counter() - started
        np.savez_compressed(args.output / filename, **arrays)
        done = arrays['extras/state_extras/episode_done'].astype(bool) & arrays['valid']
        row = dict(file=filename, phase=phase, transitions=int(arrays['valid'].sum()),
                   completed_episodes=int(done.sum()), collection_seconds=collection_seconds,
                   total_seconds=time.perf_counter()-started)
        for metric in ('sum_reward', 'cost', 'length'):
            values = arrays.get(f'extras/state_extras/episode_metrics/{metric}')
            if values is not None:
                row[f'completed_episode_mean_{metric}'] = float(values[done].mean()) if done.any() else None
        manifest['batches'].append(row)
        manifest['total_transitions'] += row['transitions']
        manifest['total_completed_episodes'] += row['completed_episodes']
        manifest['total_environment_steps'] = manifest['total_transitions'] * args.action_repeat
        save_manifest()
        print(json.dumps(row), flush=True)

    print(f'Loaded {path}; collecting {args.num_batches} batches of '
          f'{transitions_per_batch:,} transitions on {jax.devices()[0]}. '
          'First batch includes compilation.', flush=True)
    for index in range(args.num_batches):
        started = time.perf_counter()
        state, key, data = collect(state, params, key)
        save_batch(f'batch_{index:06d}.npz', data, jnp.ones(data.discount.shape, dtype=bool),
                   'ppo_collection', started)

    # A done lane has just auto-reset, but no action has been taken in its next
    # episode. It must not be started merely to drain other lanes.
    active = ~state.done.astype(bool)
    pending = int(jnp.sum(active))
    manifest['pending_episodes_at_batch_limit'] = pending
    manifest['status'] = 'finishing_episodes'
    save_manifest()
    print(f'Finishing {pending} active episodes; no new episodes will be collected.', flush=True)
    finish = make_episode_finisher(env, make_policy, args.unroll_length, extra_fields)
    max_tail_steps = (args.episode_length + args.action_repeat - 1) // args.action_repeat
    max_chunks = (max_tail_steps + args.unroll_length - 1) // args.unroll_length
    for index in range(max_chunks):
        if not bool(jnp.any(active)):
            break
        started = time.perf_counter()
        state, key, active, data, valid = finish(state, params, key, active)
        save_batch(f'finish_{index:06d}.npz', data, valid, 'episode_completion', started)
    if bool(jnp.any(active)):
        raise RuntimeError('Episode completion exceeded the configured episode limit')
    manifest['pending_episodes'] = 0
    manifest['collection_status'] = 'complete'
    manifest['collection_completed_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    manifest['collection_seconds'] = time.perf_counter() - phase_one_started
    manifest['status'] = 'comparing_hazards' if args.num_added_hazards else 'complete'
    save_manifest()
    print(f"PHASE 1 COMPLETE at {manifest['collection_completed_at_utc']}: "
          f"{manifest['total_completed_episodes']} complete episodes, "
          f"{manifest['total_transitions']} transitions, "
          f"{manifest['collection_seconds']:.1f}s. Dataset: {args.output.resolve()}", flush=True)
    if args.num_added_hazards:
        from training.agents.counterfactual.hazard_comparison import compare_episodes
        manifest['hazard_comparison'] = compare_episodes(args, manifest, network, params)
    manifest['status'] = 'complete'
    save_manifest()
    print(f'Saved dataset: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
