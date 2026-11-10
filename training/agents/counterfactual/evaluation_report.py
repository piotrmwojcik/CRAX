"""Paired episode reports and before/after charts (no training side effects)."""
import csv
import json
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


METRICS = ('reward', 'cost', 'episode_length')


def paired_statistics(before, after, samples=2000, seed=123):
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 3 or len(before) < 2:
        raise ValueError('Need matching [episodes, 3] arrays with at least two episodes')
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('Nonfinite evaluation results')
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, samples, 256):
        ids = rng.integers(len(before), size=(min(256, samples-start), len(before)))
        estimates.append(np.stack([before[ids].mean(1), after[ids].mean(1),
                                    (after-before)[ids].mean(1)], axis=1))
    intervals = np.percentile(np.concatenate(estimates), [2.5, 97.5], axis=0)
    return {name: {label: dict(mean=float(values[:, i].mean()),
                              ci95=intervals[:, j, i].tolist())
                   for j, (label, values) in enumerate((('before', before), ('after', after),
                                                        ('change', after-before)))}
            for i, name in enumerate(METRICS)}


def save_report(directory, before, after, seed, iteration, samples=2000, cost_limit=None):
    directory.mkdir(parents=True, exist_ok=False)
    stats = paired_statistics(before, after, samples)
    with (directory/'episodes.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['episode', 'eval_seed'] + [f'{side}_{m}' for side in ('before','after') for m in METRICS])
        writer.writerows([i, seed, *b, *a] for i, (b,a) in enumerate(zip(before,after)))
    summary = dict(iteration=iteration, seed=seed, episodes=len(before), metrics=stats,
                   uncertainty='95% percentile bootstrap over matched episodes; conditional on one training run',
                   bootstrap_samples=samples, cost_limit=cost_limit)
    (directory/'summary.json').write_text(json.dumps(summary, indent=2))
    fig = Figure(figsize=(9,4), constrained_layout=True)
    FigureCanvasAgg(fig)
    for axis, metric in zip(fig.subplots(1,2), ('reward','cost')):
        values = [stats[metric][side]['mean'] for side in ('before','after')]
        intervals = np.array([stats[metric][side]['ci95'] for side in ('before','after')])
        axis.bar([0,1],values,color=['#4285b4','#d16bb8'])
        # Draw percentile interval endpoints directly: the empirical mean need
        # not lie inside a finite bootstrap percentile interval.
        axis.vlines([0,1], intervals[:,0],intervals[:,1],color='black')
        for x,(lo,hi) in enumerate(intervals):
            axis.hlines([lo,hi],x-.06,x+.06,color='black')
        axis.set_xticks([0,1],['Before',f'After ({iteration})'])
        axis.set_ylabel(f'Episode {metric}')
        change = stats[metric]['change']
        axis.set_title(f"{metric.capitalize()}: change {change['mean']:+.3g}\n"
                       f"paired 95% CI [{change['ci95'][0]:.3g}, {change['ci95'][1]:.3g}]")
        if metric == 'cost' and cost_limit is not None:
            axis.axhline(cost_limit,color='red',linestyle='--',label='Cost budget')
            axis.legend()
    fig.suptitle(f'{len(before)} matched stochastic episodes; bars: mean, whiskers: 95% CI')
    fig.savefig(directory/'comparison.png',dpi=160)
    fig.savefig(directory/'comparison.pdf')
    return summary
