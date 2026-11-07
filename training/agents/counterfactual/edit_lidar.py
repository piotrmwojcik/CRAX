"""Add fake-hazard lidar to a completed nominal dataset without rerunning physics."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from crax import envs
from scripts.compare_fake_real_hazard import choose_hazard, unwrap
from training.agents.counterfactual.hazard_comparison import iter_episodes


PREFIX = 'extras/state_extras/counterfactual_state/'


def edit_dataset(dataset, output, count=3, seed=42):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest.get('collection_status') != 'complete':
        raise ValueError('Dataset must have collection_status=complete')
    if manifest.get('lidar_edit'):
        raise ValueError('Use the original dataset, not an already edited copy')
    config = manifest['configuration']
    if config['env'] != 'safe_pathway_walker2d' or config['action_repeat'] != 1:
        raise ValueError('Requires safe_pathway_walker2d with action_repeat=1')
    if count < 1:
        raise ValueError('num-added-hazards must be positive')
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    with np.load(dataset / 'initial_states.npz', allow_pickle=False) as file:
        initials = {name: file[name] for name in file.files}
    base = unwrap(envs.get_environment(config['env'], level=config['level']))
    # Use the real environment observation implementation, including nearest-K
    # selection and angular aliasing. Physics is never stepped.
    observer = unwrap(envs.get_environment(config['env'], level=config['level'],
                         num_hazards=base._num_hazards + count))
    bins = observer._lidar_bins

    @jax.jit
    def lidar(q, qd, xyz, positions):
        def one(q, qd, xyz):
            ps = SimpleNamespace(q=q, qd=qd, x=SimpleNamespace(pos=xyz))
            return observer._get_obs(ps, positions)[-bins:]
        return jax.vmap(one)(q, qd, xyz)

    placements, records = {}, []
    for lane, episode_id, episode in iter_episodes(dataset, manifest):
        physical = {name: np.concatenate([initials[name][lane:lane+1],
                    episode[PREFIX + name]]) for name in ('q', 'qd', 'xyz')}
        states = [SimpleNamespace(pipeline_state=SimpleNamespace(q=q, qd=qd,
                  x=SimpleNamespace(pos=xyz)),
                  info={'hazard_positions': initials['hazard_positions'][lane]})
                  for q, qd, xyz in zip(physical['q'], physical['qd'], physical['xyz'])]
        record = dict(env_index=lane, episode_index=episode_id)
        try:
            positions, widths = choose_hazard(base, states,
                np.random.default_rng(np.random.SeedSequence([seed, lane, episode_id])), count=count)
            record.update(status='edited', positions=positions.tolist(), half_lengths=widths.tolist())
        except ValueError as error:
            positions = np.empty((0, 3))
            record.update(status='unchanged', reason=str(error), positions=[], half_lengths=[])
        placements[lane, episode_id] = positions
        records.append(record)
    if len(records) != manifest['total_completed_episodes']:
        raise ValueError('Episode count does not match manifest')

    output.mkdir(parents=True)
    result = copy.deepcopy(manifest)
    result.pop('hazard_comparison', None)
    result['status'] = 'editing_lidar'
    result['configuration']['output'] = str(output)
    result['lidar_edit'] = dict(source_dataset=str(dataset), max_added_hazards=count,
        seed=seed, placements='fake_hazards.jsonl', lidar_bins=bins,
        semantics='Only observation and next_observation lidar slices changed. Actions, rewards, costs, physics and policy extras retain original collection values. Terminal next_observation retains auto-reset semantics with the ending episode hazard placement.')
    (output / 'manifest.json').write_text(json.dumps(result, indent=2))
    shutil.copy2(dataset / 'initial_states.npz', output / 'initial_states.npz')
    (output / 'fake_hazards.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    ids = np.zeros(config['num_envs'], dtype=int)
    previous = {name: initials[name].copy() for name in ('q', 'qd', 'xyz')}
    for entry in manifest['batches']:
        source = dataset / entry['file']
        with np.load(source, allow_pickle=False) as file:
            arrays = {name: file[name] for name in file.files}
        for fragment, valid in enumerate(arrays['valid']):
            lane = fragment % config['num_envs']
            length = int(valid.sum())
            dones = arrays['extras/state_extras/episode_done'][fragment, :length].astype(bool)
            start = 0
            for end in sorted(set((np.flatnonzero(dones) + 1).tolist() + [length])):
                if end == start:
                    continue
                added = placements[lane, int(ids[lane])]
                successors = {name: arrays[PREFIX + name][fragment, start:end] for name in previous}
                before = {name: np.concatenate([previous[name][lane:lane+1], values[:-1]])
                          for name, values in successors.items()}
                after = {name: values.copy() for name, values in successors.items()}
                terminal = bool(dones[end-1])
                if terminal:
                    for name in after:
                        after[name][-1] = initials[name][lane]
                if len(added):
                    positions = jnp.asarray(np.concatenate([initials['hazard_positions'][lane], added]))
                    for key, physical in (('observation', before), ('next_observation', after)):
                        readings = np.asarray(lidar(*(jnp.asarray(physical[n]) for n in ('q', 'qd', 'xyz')), positions))
                        arrays[key][fragment, start:end, -bins:] = readings
                for name in previous:
                    previous[name][lane] = initials[name][lane] if terminal else successors[name][-1]
                if terminal:
                    ids[lane] += 1
                start = end
        destination = output / entry['file']
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)
        print(f'Edited lidar: {entry["file"]}', flush=True)
    result['status'] = 'complete'
    result['lidar_edit']['edited_episodes'] = sum(r['status'] == 'edited' for r in records)
    (output / 'manifest.json').write_text(json.dumps(result, indent=2))
    print(f'Saved augmented dataset: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--num-added-hazards', type=int, default=3)
    parser.add_argument('--hazard-seed', type=int, default=42)
    args = parser.parse_args()
    edit_dataset(args.dataset, args.output, args.num_added_hazards, args.hazard_seed)


if __name__ == '__main__':
    main()
