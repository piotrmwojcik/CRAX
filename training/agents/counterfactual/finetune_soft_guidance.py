"""Online fine-tuning with nominal demonstrations and soft MMD guidance.

An implementation of Algorithm 1 in https://arxiv.org/abs/1911.07109,
with explicit finite-horizon, squared-MMD estimator choices; see README.
"""
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
from training import checkpoint as checkpoint_io
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


def make_update(network, normalizer, unravel):
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


def evaluate_policy(rollout, actor, seed, episodes):
    """Fixed independent random stream; does not consume training RNG."""
    started = time.perf_counter()
    key = jax.random.fold_in(jax.random.PRNGKey(seed), 1)
    action_key, reset_key = jax.random.split(key)
    data = jax.device_get(rollout(actor, action_key, jax.random.split(reset_key, episodes)))
    metrics = {'eval/episodes': episodes}
    for name, values in (('reward', data['reward'].sum(0)),
                         ('cost', data['cost'].sum(0)),
                         ('episode_length', data['valid'].sum(0))):
        metrics[f'eval/{name}'] = float(np.mean(values))
        metrics[f'eval/{name}_std'] = float(np.std(values))
    metrics['eval/zero_cost_fraction'] = float(np.mean(data['cost'].sum(0) == 0))
    metrics['eval/seconds'] = time.perf_counter()-started
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
    p.add_argument('--kl-samples', type=int, default=2048)
    p.add_argument('--gamma', type=float, default=.99)
    p.add_argument('--initial-tolerance', type=float, default=.05, help='Squared normalized MMD bound')
    p.add_argument('--annealing', type=float, default=.01)
    p.add_argument('--max-kl', type=float, default=.01)
    p.add_argument('--damping', type=float, default=.01)
    p.add_argument('--cg-iterations', type=int, default=10)
    p.add_argument('--backtracks', type=int, default=15)
    p.add_argument('--bandwidth', type=float, help='Gaussian kernel width; defaults to median distance')
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--eval-episodes', type=int, default=64)
    p.add_argument('--eval-seed', type=int, default=10000)
    p.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='disabled')
    p.add_argument('--wandb-project', default='crax')
    p.add_argument('--wandb-entity', default=None)
    p.add_argument('--wandb-name', default=None)
    args = p.parse_args()
    positive = (args.iterations, args.num_envs, args.expert_capacity, args.mmd_samples,
                args.kl_samples, args.cg_iterations, args.backtracks, args.save_every,
                args.initial_tolerance, args.max_kl, args.damping, args.eval_every, args.eval_episodes)
    if min(positive) <= 0 or not 0 < args.gamma <= 1 or args.annealing < 0:
        p.error('Counts and tolerances must be positive; 0 < gamma <= 1; annealing >= 0')
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
    measures, reward_grad, discrepancy_grad, hvp = make_update(network, normalizer, unravel)
    config_name = 'ppo_network_config.json' if algorithm == 'ppo' else 'sac_network_config.json'
    checkpoint_config = checkpoint_io.load_config(path/config_name)
    args.output.mkdir(parents=True)
    report = dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  source_checkpoint=str(path), demonstrations=counts, bandwidth=bandwidth,
                  method='Soft expert guidance: MMD squared, KL trust region, recovery, annealed tolerance',
                  online_environment=dict(env=config['env'], level=config['level'], episode_length=horizon),
                  objective='environment reward; safety cost logged but not constrained',
                  normalizer='frozen', expert_filter='all nominal episodes; no winner filtering',
                  estimator='normalized discounted occupancy; prefix importance weights; empirical squared MMD',
                  paper='https://arxiv.org/abs/1911.07109')
    report['evaluation'] = dict(scope='standard real-hazard environment, independent fixed reset/action randomness',
                                seed=args.eval_seed, episodes=args.eval_episodes,
                                schedule='before training, every eval-every iterations, and final iteration',
                                policy='stochastic')
    (args.output/'run.json').write_text(json.dumps(report, indent=2))
    np.savez(args.output/'mmd_features.npz', mean=feature_mean, scale=feature_scale, bandwidth=bandwidth)
    tolerance = args.initial_tolerance
    key = jax.random.PRNGKey(args.seed)
    import wandb
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                    name=args.wandb_name or args.output.name, mode=args.wandb_mode,
                    dir=str(args.output.resolve()), config=report) as run_log, \
            (args.output/'metrics.jsonl').open('w') as log, \
            (args.output/'evaluation.jsonl').open('w') as eval_log:
        run_log.define_metric('iteration')
        run_log.define_metric('train/*', step_metric='iteration')
        run_log.define_metric('eval/*', step_metric='iteration')
        def evaluate(iteration):
            metrics = evaluate_policy(rollout, unravel(theta), args.eval_seed, args.eval_episodes)
            record = dict(iteration=iteration, **metrics)
            eval_log.write(json.dumps(record)+'\n'); eval_log.flush()
            run_log.log(record)
            for name, value in metrics.items():
                run_log.summary[name] = value
            print(json.dumps(record), flush=True)

        evaluate(0)
        for iteration in range(1, args.iterations+1):
            started = time.perf_counter()
            key, action_key, reset_key = jax.random.split(key, 3)
            data = rollout(unravel(theta), action_key, jax.random.split(reset_key, args.num_envs))
            valid = np.asarray(data['valid'])
            occupancy = valid * args.gamma**np.arange(horizon)[:, None]
            probability = occupancy.reshape(-1)/occupancy.sum()
            indices = rng.choice(len(probability), args.mmd_samples, p=probability)
            kl_indices = rng.choice(len(probability), args.kl_samples, p=probability)
            online = np.concatenate([np.asarray(data['obs']),np.asarray(data['action'])],axis=-1).reshape(-1, expert.shape[1])
            online = (online[indices]-feature_mean)/feature_scale
            demos = expert[rng.integers(len(expert),size=args.mmd_samples)]
            kernels = tuple(gaussian_kernel(jnp.asarray(x),jnp.asarray(y),bandwidth)
                            for x,y in ((online,online),(online,demos),(demos,demos)))
            returns = discounted_returns(data['reward'],data['valid'],args.gamma)
            # State-independent baseline at each timestep, estimated across lanes.
            # Leave-one-out baseline avoids cancelling the policy gradient
            # with a lane's own return (especially with a single environment).
            other_count = data['valid'].sum(1,keepdims=True)-data['valid']
            baseline = jnp.where(other_count > 0,
                (returns.sum(1,keepdims=True)-returns)/jnp.maximum(other_count,1), 0.)
            advantage = (returns-baseline)*data['valid']
            data = dict(data, advantage=advantage/(jnp.std(advantage)+1e-8),
                        occupancy=jnp.asarray(occupancy/occupancy.sum(),dtype=jnp.float32),
                        kl_obs=data['obs'].reshape(-1,data['obs'].shape[-1])[kl_indices])
            indices = jnp.asarray(indices)
            old = theta
            arguments = (old,data,indices,kernels)
            objective, discrepancy, _ = map(float,measures(old,*arguments))
            g, b = reward_grad(old,*arguments), discrepancy_grad(old,*arguments)
            if not np.isfinite(np.asarray(g)).all() or not np.isfinite(np.asarray(b)).all():
                raise FloatingPointError('Nonfinite policy gradients')
            matvec = lambda v: hvp(old,v,*arguments,args.damping)
            u = conjugate_gradient(matvec,g,args.cg_iterations)
            r = conjugate_gradient(matvec,b,args.cg_iterations)
            step, mode = constrained_step(g,b,u,r,tolerance-discrepancy,args.max_kl)
            accepted = False
            new_values = (objective,discrepancy,0.)
            fraction = 0.
            for backtrack in range(args.backtracks):
                candidate_fraction = .5**backtrack
                candidate = old + candidate_fraction*step
                reward, distance, kl = map(float,measures(candidate,*arguments))
                finite = np.isfinite([reward,distance,kl]).all()
                recovery = mode.startswith('infeasible') or mode == 'recovery'
                satisfies = distance < discrepancy if recovery else distance <= tolerance
                improves = recovery or discrepancy > tolerance or reward >= objective
                if finite and kl <= args.max_kl and satisfies and improves and np.linalg.norm(np.asarray(step)) > 0:
                    theta, new_values, fraction, accepted = candidate,(reward,distance,kl),candidate_fraction,True
                    break
            row = dict(iteration=iteration, accepted=accepted, mode=mode, step_fraction=fraction,
                       tolerance=tolerance, mmd_before=discrepancy, mmd_after=new_values[1], kl=new_values[2],
                       surrogate_before=objective,surrogate_after=new_values[0],
                       rollout_reward=float(data['reward'].sum()/args.num_envs),
                       rollout_cost=float(data['cost'].sum()/args.num_envs),
                       rollout_steps=float(data['valid'].sum()/args.num_envs),
                       seconds=time.perf_counter()-started)
            log.write(json.dumps(row)+'\n'); log.flush()
            print(json.dumps(row),flush=True)
            run_log.log({'iteration': iteration,
                         **{f'train/{k}': v for k, v in row.items() if k != 'iteration'}})
            tolerance *= 1+args.annealing
            if iteration % args.save_every == 0 or iteration == args.iterations:
                updated = list(params); updated[1] = unravel(theta)
                checkpoint_io.save(args.output/'checkpoint', iteration, tuple(updated), checkpoint_config, config_name)
                run_log.summary['saved_checkpoint'] = str((args.output/'checkpoint'/f'{iteration:012d}').resolve())
            if iteration % args.eval_every == 0 or iteration == args.iterations:
                evaluate(iteration)
    print(f'Saved fine-tuned checkpoint: {args.output}/checkpoint/{args.iterations:012d}',flush=True)


if __name__ == '__main__':
    main()
