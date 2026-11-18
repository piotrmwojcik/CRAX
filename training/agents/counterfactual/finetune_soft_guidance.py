"""Online imitation by gradient descent on squared MMD to one expert dataset."""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

from crax import envs
from crax.envs.wrappers import training as wrappers
from training import acting, checkpoint as checkpoint_io
from training.agents.counterfactual.collect_rollouts import resolve_checkpoint, load_collection_policy
from training.agents.counterfactual.hazard_comparison import iter_episodes
from training.agents.counterfactual.soft_guidance import (
    gaussian_kernel, weighted_mmd, conjugate_gradient, constrained_step, discounted_returns)


def load_demonstrations(dataset, manifest, capacity, gamma, rng):
    """Weighted reservoir: gamma**t per valid nominal state-action sample.

    All nominal episodes are eligible, including episodes without a safe gap.
    Raw actions/log probabilities are deliberately not used after lidar editing.
    """
    features, priorities = None, np.empty(0)
    episodes = transitions = 0
    for _, _, episode in iter_episodes(dataset, manifest):
        x = np.concatenate([episode['observation'], episode['action']], axis=-1)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite expert observations/actions')
        if np.any(np.abs(episode['action']) > 1.00001):
            raise ValueError('Expected tanh-bounded expert actions')
        # Largest log(U)/weight keys give weighted sampling without replacement.
        keys = np.log(rng.uniform(np.finfo(float).tiny, 1., len(x))) / gamma**np.arange(len(x))
        features = x if features is None else np.concatenate([features, x])
        priorities = np.concatenate([priorities, keys])
        if len(priorities) > capacity:
            keep = np.argpartition(priorities, -capacity)[-capacity:]
            features, priorities = features[keep], priorities[keep]
        episodes += 1
        transitions += len(x)
    if episodes != manifest['total_completed_episodes'] or features is None:
        raise ValueError('Incomplete or empty demonstration dataset')
    return features.astype(np.float32), dict(episodes=episodes, transitions=transitions,
                                            reservoir_size=len(features))


def make_rollout(env, network, normalizer, horizon):
    distribution = network.parametric_action_distribution
    @jax.jit
    def rollout(actor, key, reset_keys):
        initial = env.reset(reset_keys)
        def body(carry, unused):
            state, key, active = carry
            key, action_key = jax.random.split(key)
            logits = network.policy_network.apply(normalizer, actor, state.obs)
            raw = distribution.sample_no_postprocessing(logits, action_key)
            action = distribution.postprocess(raw)
            successor = env.step(state, action)
            data = dict(obs=state.obs, raw=raw, action=action,
                        logp=distribution.log_prob(logits, raw),
                        reward=successor.reward*active, cost=successor.info['cost']*active,
                        valid=active)
            return (successor, key, active & ~successor.done.astype(bool)), data
        return jax.lax.scan(body, (initial, key, jnp.ones(initial.done.shape, dtype=bool)),
                            None, length=horizon)[1]
    return rollout


def make_mmd_update(network, normalizer, unravel):
    """Score-function occupancy gradient; rewards and costs are not inputs."""
    def discrepancy(theta, data, indices, kernels):
        logits = network.policy_network.apply(normalizer, unravel(theta), data['obs'])
        log_ratio = network.parametric_action_distribution.log_prob(logits, data['raw']) - data['logp']
        prefix = jnp.cumsum(jnp.where(data['valid'], log_ratio, 0.), axis=0)
        weights = jax.nn.softmax(prefix.reshape(-1)[indices])
        return weighted_mmd(weights, *kernels)
    return jax.jit(discrepancy), jax.jit(jax.grad(discrepancy))


def make_ppo_evaluation(env, network, normalizer, horizon):
    """Use PPO's EvalWrapper and generate_unroll with its metric conventions.

    Keep fixed reset/action keys for paired comparisons and retain individual
    episode totals, unlike the aggregate-only default of acting.Evaluator.
    """
    eval_env = wrappers.EvalWrapper(env)
    walltime = 0.

    @jax.jit
    def run(actor, reset_keys, key):
        def policy(obs, action_key):
            logits = network.policy_network.apply(normalizer, actor, obs)
            dist = network.parametric_action_distribution
            return dist.sample(logits, action_key), {}
        state = eval_env.reset(reset_keys)
        state, _ = acting.generate_unroll(eval_env, state, policy, key, horizon)
        return state.info['eval_metrics']

    def evaluate(actor, seed, episodes):
        nonlocal walltime
        started = time.perf_counter()
        # Match the first evaluation key split in acting.Evaluator.
        _, key = jax.random.split(jax.random.PRNGKey(seed))
        result = jax.device_get(run(actor, jax.random.split(key, episodes), key))
        seconds = time.perf_counter() - started
        walltime += seconds
        metrics = {}
        for name, values in result.episode_metrics.items():
            metrics[f'eval/episode_{name}'] = float(np.mean(values))
            metrics[f'eval/episode_{name}_std'] = float(np.std(values))
        metrics.update({
            'eval/avg_episode_length': float(np.mean(result.episode_steps)),
            'eval/std_episode_length': float(np.std(result.episode_steps)),
            'eval/epoch_eval_time': seconds,
            'eval/sps': horizon * episodes / seconds,
            'eval/walltime': walltime,
            'eval/episodes': episodes,
            'eval/zero_cost_fraction': float(np.mean(result.episode_metrics['cost'] == 0)),
        })
        totals = np.column_stack([result.episode_metrics['reward'],
                                  result.episode_metrics['cost'], result.episode_steps])
        return metrics, totals
    return evaluate


def make_update(network, normalizer, unravel):
    """Legacy constrained updater, retained for regression tests; not used by main."""
    distribution = network.parametric_action_distribution
    def stats(theta, obs):
        logits = network.policy_network.apply(normalizer, unravel(theta), obs)
        return logits, distribution.create_dist(logits)

    def measures(theta, old, data, indices, kernels):
        logits, _ = stats(theta, data['obs'])
        log_ratio = distribution.log_prob(logits, data['raw']) - data['logp']
        # Include all preceding actions: occupancy gradients must account for
        # how policy actions change the visited state distribution.
        prefix_log_ratio = jnp.cumsum(jnp.where(data['valid'], log_ratio, 0.), axis=0)
        selected = prefix_log_ratio.reshape(-1)[indices]
        weights = jax.nn.softmax(selected)
        discrepancy = weighted_mmd(weights, *kernels)
        reward = jnp.sum(jnp.exp(log_ratio)*data['advantage']*data['occupancy'])
        _, before = stats(old, data['kl_obs'])
        _, after = stats(theta, data['kl_obs'])
        kl = jnp.mean(jnp.sum(jnp.log(after.scale/before.scale)
               + (before.scale**2 + (before.loc-after.loc)**2)/(2*after.scale**2) - .5, axis=-1))
        return reward, discrepancy, kl

    reward_grad = jax.jit(jax.grad(lambda theta, *args: measures(theta, *args)[0]))
    discrepancy_grad = jax.jit(jax.grad(lambda theta, *args: measures(theta, *args)[1]))
    kl_grad = jax.grad(lambda theta, *args: measures(theta, *args)[2])
    @jax.jit
    def hessian_product(theta, vector, old, data, indices, kernels, damping):
        return jax.jvp(lambda p: kl_grad(p, old, data, indices, kernels),
                       (theta,), (vector,))[1] + damping*vector
    return jax.jit(measures), reward_grad, discrepancy_grad, hessian_product


def evaluate_policy(rollout, actor, seed, episodes, return_episodes=False):
    """Fixed independent random stream; does not consume training RNG."""
    started = time.perf_counter()
    key = jax.random.fold_in(jax.random.PRNGKey(seed), 1)
    batches = []
    batch_size = min(episodes, 32)
    for start in range(0, episodes, batch_size):
        action_key, reset_key = jax.random.split(jax.random.fold_in(key, start))
        data = jax.device_get(rollout(actor, action_key, jax.random.split(reset_key, batch_size)))
        totals = np.column_stack([data['reward'].sum(0), data['cost'].sum(0), data['valid'].sum(0)])
        batches.append(totals[:min(batch_size, episodes-start)])
    totals = np.concatenate(batches)
    metrics = {'eval/episodes': episodes}
    for i, name in enumerate(('reward', 'cost', 'episode_length')):
        metrics[f'eval/{name}'] = float(np.mean(totals[:, i]))
        metrics[f'eval/{name}_std'] = float(np.std(totals[:, i]))
    metrics['eval/zero_cost_fraction'] = float(np.mean(totals[:, 1] == 0))
    metrics['eval/seconds'] = time.perf_counter()-started
    if return_episodes:
        return metrics, totals
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path, help='Completed lidar-edited nominal dataset')
    p.add_argument('--checkpoint', type=Path, help='Defaults to source checkpoint in manifest')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--num-envs', type=int, default=16)
    p.add_argument('--episode-length', type=int, help='Defaults to dataset episode length')
    p.add_argument('--expert-capacity', type=int, default=100000)
    p.add_argument('--mmd-samples', type=int, default=512)
    p.add_argument('--gamma', type=float, default=.99)
    p.add_argument('--learning-rate', type=float, default=5e-4, help='MMD gradient descent step size')
    p.add_argument('--bandwidth', type=float, help='Gaussian kernel width; defaults to median distance')
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--eval-episodes', type=int, default=128)
    p.add_argument('--eval-seed', type=int, default=10000)
    p.add_argument('--final-eval-episodes', type=int, default=512)
    p.add_argument('--final-eval-seed', type=int, default=60000)
    p.add_argument('--eval-bootstrap-samples', type=int, default=2000)
    p.add_argument('--eval-cost-limit', type=float, default=None, help='Optional episode cost budget line on charts')
    p.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='disabled')
    p.add_argument('--wandb-project', default='crax')
    p.add_argument('--wandb-entity', default=None)
    p.add_argument('--wandb-name', default=None)
    args = p.parse_args()
    positive = (args.iterations, args.num_envs, args.expert_capacity, args.mmd_samples,
                args.save_every, args.learning_rate, args.eval_every, args.eval_episodes)
    if min(positive) <= 0 or not 0 < args.gamma <= 1:
        p.error('Counts and tolerances must be positive; 0 < gamma <= 1')
    if min(args.eval_episodes, args.final_eval_episodes, args.eval_bootstrap_samples) < 2:
        p.error('Evaluation episode counts and bootstrap samples must be at least 2')
    if args.final_eval_seed == args.eval_seed:
        p.error('Final evaluation seed must differ from periodic evaluation seed')
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    if manifest.get('status') != 'complete' or not manifest.get('lidar_edit'):
        p.error('Provide a completed lidar-edited dataset')
    config = manifest['configuration']
    if config['env'] != 'safe_pathway_walker2d' or config['action_repeat'] != 1:
        p.error('Only state-based safe_pathway_walker2d with action_repeat=1 is supported')
    horizon = args.episode_length or config['episode_length']
    if horizon < 1 or (args.bandwidth is not None and args.bandwidth <= 0):
        p.error('Episode length and bandwidth must be positive')
    if args.output.exists():
        p.error('Use a new output directory')
    path = resolve_checkpoint(args.checkpoint or Path(manifest['checkpoint']))
    network, params, _, algorithm = load_collection_policy(path)
    normalizer, actor = params[0], params[1]
    theta, unravel = ravel_pytree(actor)
    rng = np.random.default_rng(args.seed)
    expert, counts = load_demonstrations(args.dataset, manifest, args.expert_capacity, args.gamma, rng)
    feature_mean = expert.mean(axis=0)
    feature_scale = np.maximum(expert.std(axis=0), .1)
    expert = (expert-feature_mean)/feature_scale
    probe = expert[rng.choice(len(expert), min(512, len(expert)), replace=False)]
    distances = np.sqrt(np.maximum(((probe[:, None]-probe[None, :])**2).sum(-1), 0.))
    nonzero = distances[distances > 0]
    bandwidth = args.bandwidth or (float(np.median(nonzero)) if len(nonzero) else 1.)
    env = wrappers.wrap(envs.get_environment(config['env'], level=config['level']),
                        episode_length=horizon, action_repeat=1)
    if expert.shape[1] != env.observation_size + env.action_size:
        raise ValueError('Demonstration dimensions do not match environment')
    rollout = make_rollout(env, network, normalizer, horizon)
    mmd_measure, mmd_gradient = make_mmd_update(network, normalizer, unravel)
    evaluation = make_ppo_evaluation(env, network, normalizer, horizon)
    config_name = 'ppo_network_config.json' if algorithm == 'ppo' else 'sac_network_config.json'
    checkpoint_config = checkpoint_io.load_config(path/config_name)
    args.output.mkdir(parents=True)
    report = dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  source_checkpoint=str(path), demonstrations=counts, bandwidth=bandwidth,
                  method='MMD imitation: theta <- theta - learning_rate * grad(MMD squared)',
                  online_environment=dict(env=config['env'], level=config['level'], episode_length=horizon),
                  objective='MMD squared only; reward and safety cost are evaluation metrics',
                  expert_policy=str(manifest['checkpoint']),
                  normalizer='frozen', expert_filter='all nominal episodes; no winner filtering',
                  estimator='normalized discounted occupancy; prefix importance weights; empirical squared MMD',
                  paper='https://arxiv.org/abs/1911.07109')
    report['evaluation'] = dict(scope='standard real-hazard environment, independent fixed reset/action randomness',
                                seed=args.eval_seed, episodes=args.eval_episodes,
                                schedule='before training, every eval-every iterations, and final iteration',
                                policy='stochastic')
    report['evaluation']['final_test'] = dict(seed=args.final_eval_seed, episodes=args.final_eval_episodes)
    (args.output/'run.json').write_text(json.dumps(report, indent=2))
    np.savez(args.output/'mmd_features.npz', mean=feature_mean, scale=feature_scale, bandwidth=bandwidth)
    environment_steps = 0
    key = jax.random.PRNGKey(args.seed)
    import wandb
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                    name=args.wandb_name or args.output.name, mode=args.wandb_mode,
                    dir=str(args.output.resolve()), config=report) as run_log, \
            (args.output/'metrics.jsonl').open('w') as log, \
            (args.output/'evaluation.jsonl').open('w') as eval_log:
        run_log.define_metric('iteration')
        run_log.define_metric('num_steps')
        run_log.define_metric('train/*', step_metric='num_steps')
        run_log.define_metric('eval/*', step_metric='num_steps')
        run_log.define_metric('test/*', step_metric='num_steps')
        from training.agents.counterfactual.evaluation_report import save_report, METRICS
        evaluation_baseline = None
        def publish_comparison(before, after, seed, iteration, namespace, directory):
            summary = save_report(directory, before, after, seed, iteration,
                                  args.eval_bootstrap_samples, args.eval_cost_limit)
            table = wandb.Table(columns=['episode', 'eval_seed'] +
                               [f'{side}_{m}' for side in ('before','after') for m in METRICS],
                               data=[[i, seed, *b.tolist(), *a.tolist()]
                                     for i,(b,a) in enumerate(zip(before,after))])
            payload = {'iteration': iteration, 'num_steps': environment_steps,
                       f'{namespace}/comparison': wandb.Image(str(directory/'comparison.png')),
                       f'{namespace}/episode_results': table}
            for metric, values in summary['metrics'].items():
                for label, value in values.items():
                    prefix = f'{namespace}/{metric}_{label}'
                    payload[prefix] = value['mean']
                    payload[prefix+'_ci_low'], payload[prefix+'_ci_high'] = value['ci95']
            run_log.log(payload, step=environment_steps, commit=False)

        def evaluate(iteration):
            nonlocal evaluation_baseline
            metrics, episodes = evaluation(unravel(theta), args.eval_seed, args.eval_episodes)
            if evaluation_baseline is None:
                evaluation_baseline = episodes.copy()
            publish_comparison(evaluation_baseline, episodes, args.eval_seed, iteration, 'eval',
                               args.output/'evaluation'/f'{iteration:06d}')
            record = dict(iteration=iteration, num_steps=environment_steps, **metrics)
            eval_log.write(json.dumps(record)+'\n'); eval_log.flush()
            run_log.log(record, step=environment_steps, commit=False)
            for name, value in metrics.items():
                run_log.summary[name] = value
            print(json.dumps(record), flush=True)

        evaluate(0)
        run_log.log({}, step=0, commit=True)
        for iteration in range(1, args.iterations+1):
            started = time.perf_counter()
            key, action_key, reset_key = jax.random.split(key, 3)
            data = rollout(unravel(theta), action_key, jax.random.split(reset_key, args.num_envs))
            valid = np.asarray(data['valid'])
            occupancy = valid * args.gamma**np.arange(horizon)[:, None]
            probability = occupancy.reshape(-1)/occupancy.sum()
            indices = rng.choice(len(probability), args.mmd_samples, p=probability)
            online = np.concatenate([np.asarray(data['obs']),np.asarray(data['action'])],axis=-1).reshape(-1, expert.shape[1])
            online = (online[indices]-feature_mean)/feature_scale
            demos = expert[rng.integers(len(expert),size=args.mmd_samples)]
            kernels = tuple(gaussian_kernel(jnp.asarray(x),jnp.asarray(y),bandwidth)
                            for x,y in ((online,online),(online,demos),(demos,demos)))
            indices = jnp.asarray(indices)
            old = theta
            discrepancy = float(mmd_measure(old, data, indices, kernels))
            gradient = mmd_gradient(old, data, indices, kernels)
            if not np.isfinite(np.asarray(gradient)).all():
                raise FloatingPointError('Nonfinite MMD gradient')
            theta = old - args.learning_rate * gradient
            if not np.isfinite(np.asarray(theta)).all():
                raise FloatingPointError('Nonfinite actor parameters')
            after = float(mmd_measure(theta, data, indices, kernels))
            # The full scan executes horizon steps in every lane, including
            # masked post-termination steps, as PPO evaluation does.
            environment_steps += horizon * args.num_envs
            row = dict(iteration=iteration, num_steps=environment_steps,
                       mode='mmd_gradient_descent', learning_rate=args.learning_rate,
                       mmd_before=discrepancy, mmd_after=after,
                       gradient_norm=float(jnp.linalg.norm(gradient)),
                       update_norm=float(jnp.linalg.norm(theta-old)),
                       rollout_reward=float(data['reward'].sum()/args.num_envs),
                       rollout_cost=float(data['cost'].sum()/args.num_envs),
                       rollout_steps=float(data['valid'].sum()/args.num_envs),
                       seconds=time.perf_counter()-started)
            log.write(json.dumps(row)+'\n'); log.flush()
            print(json.dumps(row),flush=True)
            run_log.log({'iteration': iteration, 'num_steps': environment_steps,
                         **{f'train/{k}': v for k, v in row.items() if k not in ('iteration', 'num_steps')}},
                        step=environment_steps, commit=False)
            if iteration % args.save_every == 0 or iteration == args.iterations:
                updated = list(params); updated[1] = unravel(theta)
                checkpoint_io.save(args.output/'checkpoint', iteration, tuple(updated), checkpoint_config, config_name)
                run_log.summary['saved_checkpoint'] = str((args.output/'checkpoint'/f'{iteration:012d}').resolve())
            if iteration % args.eval_every == 0 or iteration == args.iterations:
                evaluate(iteration)
            if iteration != args.iterations:
                run_log.log({}, step=environment_steps, commit=True)
        # Untouched test randomness, used only after training finishes. The
        # original actor and final actor share reset/action keys for pairing.
        _, test_before = evaluation(actor, args.final_eval_seed, args.final_eval_episodes)
        test_metrics, test_after = evaluation(unravel(theta), args.final_eval_seed, args.final_eval_episodes)
        publish_comparison(test_before, test_after, args.final_eval_seed, args.iterations,
                           'test', args.output/'independent_evaluation')
        run_log.log({'iteration': args.iterations, 'num_steps': environment_steps,
                     **{k.replace('eval/', 'test/'): v for k,v in test_metrics.items()}},
                    step=environment_steps, commit=True)
    print(f'Saved fine-tuned checkpoint: {args.output}/checkpoint/{args.iterations:012d}',flush=True)


if __name__ == '__main__':
    main()
