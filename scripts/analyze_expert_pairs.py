"""Summarize stored nominal expert episodes and their paired real rollouts.

Requires only NumPy; no policy, simulator, checkpoint or GPU is loaded.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


COST = 'extras/state_extras/cost'
DONE = 'extras/state_extras/episode_done'


def nominal_episodes(directory, manifest):
    """Stream scalar episode statistics, respecting lanes, padding and boundaries."""
    lanes = int(manifest['configuration']['num_envs'])
    pending = [dict(reward=0., cost=0., steps=0, cost_steps=0,
                    action_squared=0., action_values=0, saturated_actions=0) for _ in range(lanes)]
    ids = np.zeros(lanes, dtype=int)
    for entry in manifest['batches']:
        with np.load(directory/entry['file'], allow_pickle=False) as file:
            reward, cost, done, valid, action = (file[key] for key in ('reward', COST, DONE, 'valid', 'action'))
        if reward.shape != valid.shape or cost.shape != valid.shape or done.shape != valid.shape:
            raise ValueError(f'Invalid transition shapes in {entry["file"]}')
        for fragment, mask in enumerate(valid.astype(bool)):
            lane = fragment % lanes
            length = int(mask.sum())
            if not mask[:length].all() or mask[length:].any():
                raise ValueError('Expected contiguous valid masks')
            start = 0
            for end in sorted(set((np.flatnonzero(done[fragment, :length]) + 1).tolist() + [length])):
                if start == end:
                    continue
                r, c, a = reward[fragment,start:end], cost[fragment,start:end], action[fragment,start:end]
                if not all(np.isfinite(x).all() for x in (r,c,a)):
                    raise ValueError('Nonfinite valid nominal transitions')
                stats = pending[lane]
                stats['reward'] += float(r.sum(dtype=np.float64))
                stats['cost'] += float(c.sum(dtype=np.float64))
                stats['steps'] += end-start
                stats['cost_steps'] += int((c > 0).sum())
                stats['action_squared'] += float(np.square(a.astype(np.float64)).sum())
                stats['action_values'] += a.size
                stats['saturated_actions'] += int((np.abs(a) >= .99).sum())
                if done[fragment,end-1]:
                    yield lane, int(ids[lane]), finish_stats(stats)
                    ids[lane] += 1
                    pending[lane] = dict.fromkeys(stats, 0)
                start = end
    if any(s['steps'] for s in pending):
        raise ValueError('Dataset contains incomplete nominal episodes')


def finish_stats(stats):
    return dict(reward=stats['reward'], cost=stats['cost'], steps=stats['steps'],
                cost_per_step=stats['cost']/stats['steps'],
                cost_step_fraction=stats['cost_steps']/stats['steps'],
                action_rms=float(np.sqrt(stats['action_squared']/stats['action_values'])),
                action_saturation_fraction=stats['saturated_actions']/stats['action_values'])


def real_stats(path):
    with np.load(path, allow_pickle=False) as file:
        mask = file['valid'].astype(bool)
        r,c,a = (file[key][mask] for key in ('reward','cost','action'))
        added = file['added_hazard_cost'][mask]
    if not len(r) or not all(np.isfinite(x).all() for x in (r,c,a,added)):
        raise ValueError(f'Empty or nonfinite real trajectory: {path}')
    stats = finish_stats(dict(reward=float(r.sum(dtype=np.float64)),cost=float(c.sum(dtype=np.float64)),
        steps=len(r),cost_steps=int((c>0).sum()),action_squared=float(np.square(a.astype(np.float64)).sum()),
        action_values=a.size,saturated_actions=int((np.abs(a)>=.99).sum())))
    stats['added_hazard_cost'] = float(added.sum(dtype=np.float64))
    stats['cost_without_added_hazards'] = stats['cost']-stats['added_hazard_cost']
    return stats


def describe(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return dict(count=0)
    return dict(count=len(x), mean=float(x.mean()), std=float(x.std()),
                min=float(x.min()), p05=float(np.quantile(x,.05)), median=float(np.median(x)),
                p95=float(np.quantile(x,.95)), max=float(x.max()))


def summarize(episodes):
    if not episodes:
        return dict(episodes=0)
    metrics = sorted(set.intersection(*(set(e) for e in episodes)))
    return dict(episodes=len(episodes), total_transitions=sum(e['steps'] for e in episodes),
                zero_cost_fraction=float(np.mean([e['cost']==0 for e in episodes])),
                metrics={m:describe([e[m] for e in episodes]) for m in metrics})


def analyze(dataset, pairs, output):
    dataset = dataset.resolve()
    manifest = json.loads((dataset/'manifest.json').read_text())
    if manifest.get('collection_status') != 'complete' or manifest.get('status') == 'editing_lidar':
        raise ValueError('Nominal collection / lidar editing must be complete')
    if pairs is None:
        pairs = Path(manifest.get('lidar_edit',{}).get('source_dataset', str(dataset)))
    pairs = pairs.resolve()
    records = {}
    for line in (pairs/'hazard_comparisons.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row['env_index'],row['episode_index'])
        if key in records:
            raise ValueError(f'Duplicate pair identity: {key}')
        records[key] = row
    rows, nominal_all, nominal_paired, real_all, differences, warnings = [], [], [], [], [], []
    statuses, winners, reasons = Counter(),Counter(),Counter()
    for lane,episode,nominal in nominal_episodes(dataset,manifest):
        key = (lane,episode)
        pair = records.pop(key, None)
        nominal_all.append(nominal)
        row = dict(env_index=lane,episode_index=episode,
                   **{f'nominal_{k}':v for k,v in nominal.items()})
        if pair is None:
            row['status']='missing_pair_record'
        else:
            row.update(pair=pair['pair'],status=pair['status'],reason=pair.get('reason',''),
                       num_added_hazards=pair.get('num_added_hazards'),real_rollout=pair.get('real_rollout',''))
            for metric in ('reward','cost','steps'):
                if not np.isclose(nominal[metric],pair['nominal'][metric],rtol=1e-5,atol=1e-5):
                    warnings.append(f'Pair {pair["pair"]}: stored nominal {metric} differs from log')
            if pair['status']=='compared':
                real = real_stats(pairs/pair['real_rollout'])
                nominal_paired.append(nominal); real_all.append(real)
                diff = {m:real[m]-nominal[m] for m in ('reward','cost','steps')}
                differences.append(diff)
                winner = ('nominal' if diff['reward']<0 and diff['cost']>0 else
                          'real' if diff['reward']>0 and diff['cost']<0 else 'none')
                winners[winner] += 1
                row.update(winner=winner,**{f'real_{k}':v for k,v in real.items()},
                           **{f'real_minus_nominal_{k}':v for k,v in diff.items()})
            else:
                reasons[pair.get('reason','unspecified')] += 1
        statuses[row['status']] += 1
        rows.append(row)
    if len(rows) != manifest['total_completed_episodes']:
        raise ValueError('Nominal episode count differs from manifest')
    if records:
        raise ValueError(f'{len(records)} comparison records have no matching nominal episode')
    report = dict(dataset=str(dataset),pairs_dataset=str(pairs),statuses=dict(statuses),
        winners_strict_reward_cost_dominance=dict(winners),skip_reasons=dict(reasons),
        nominal_all=summarize(nominal_all),nominal_paired=summarize(nominal_paired),
        real_paired=summarize(real_all),
        paired_changes_real_minus_nominal={m:describe([d[m] for d in differences]) for m in ('reward','cost','steps')},
        warnings=warnings,
        notes=['Metrics recomputed from NPZ arrays; padding excluded.',
               'Compare nominal_paired with real_paired; nominal_all also contains unpaired/skipped episodes.',
               'Positive reward change favors real; negative cost change favors real.',
               'Cost includes the environment terminal cost; positive-cost steps are not necessarily hazard contacts.',
               'Action saturation means abs(action) >= 0.99; std is population standard deviation.'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'statistics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (output/'episodes.csv').open('w',newline='') as file:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        writer=csv.DictWriter(file,fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    print(f'Saved {len(rows)} nominal episodes and {len(real_all)} real pairs to {output}')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True,help='Original or lidar-edited nominal dataset')
    parser.add_argument('--pairs',type=Path,help='Original collection directory containing hazard_comparisons.jsonl and real_rollouts; inferred from edited manifest if omitted')
    parser.add_argument('--output',type=Path,required=True,help='New directory for statistics.json and episodes.csv')
    args=parser.parse_args()
    analyze(args.dataset,args.pairs,args.output)


if __name__=='__main__':
    main()
