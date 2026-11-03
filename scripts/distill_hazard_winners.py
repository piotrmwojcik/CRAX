"""Winner-filtered SAC imitation (not an implementation of SDPO).

Select strictly Pareto-dominant nominal/added-hazard rollouts and imitate
their sampled actions at their own observations, with original-policy KL.
"""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL', 'cgl' if sys.platform == 'darwin' else 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from crax import envs
from scripts.counterpol_hazard import load_actor
from scripts.compare_fake_real_hazard import choose_hazard, unwrap, rollout as comparison_rollout
from training.agents.sac import networks, checkpoint as checkpoint_io
from training import acting
from crax.envs.wrappers import training as wrappers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairs', type=int, default=32)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=15)
    p.add_argument('--hazard-seed', type=int, default=42)
    p.add_argument('--level', type=int, default=1)
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--kl-weight', type=float, default=1.)
    p.add_argument('--reward-margin', type=float, default=1.)
    p.add_argument('--cost-margin', type=float, default=.5)
    p.add_argument('--eval-episodes', type=int, default=16)
    p.add_argument('--nominal-relabel', action='store_true',
                   help='Keep only nominal winners; rebuild their observations with the added hazard')
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--eval-seed', type=int, default=10000)
    p.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='disabled')
    p.add_argument('--wandb-project', default='crax')
    p.add_argument('--wandb-entity', default=None)
    p.add_argument('--wandb-name', default=None)
    args = p.parse_args()
    if min(args.pairs, args.steps, args.updates, args.batch_size, args.eval_episodes, args.eval_every) <= 0:
        p.error('Counts must be positive')
    if min(args.reward_margin, args.cost_margin, args.kl_weight) < 0:
        p.error('Margins and KL weight must be nonnegative')
    if args.output.exists():
        p.error('Use a new output directory to preserve previous experiments')
    cfg, net, (norm, original) = load_actor(args.checkpoint.resolve())
    # Restore portably on CPU, then colocate actor and normalizer with physics
    # on the default accelerator (or CPU for local runs).
    norm, original = jax.device_put((norm, original), jax.devices()[0])
    if cfg.get('cost_conditioning', False):
        p.error('Only unconditioned SAC-Lag checkpoints are supported')
    args.output.mkdir(parents=True)
    import wandb
    run_log = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name or args.output.name, mode=args.wandb_mode,
                         dir=str(args.output.resolve()),
                         config={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    def log(values):
        run_log.log(values)
    env = envs.get_environment('safe_pathway_walker2d', level=args.level)
    base = unwrap(env)
    real_env = envs.get_environment('safe_pathway_walker2d', level=args.level, num_hazards=base._num_hazards+1)
    real = unwrap(real_env)
    reset, real_reset = jax.jit(env.reset), jax.jit(real_env.reset)
    step = jax.jit(env.step)
    radii = real._hazard_radii[:-1]
    infer = networks.make_inference_fn(net)
    policy = jax.jit(lambda params, obs, key: infer((norm, params), deterministic=False)(obs, key)[0])

    def run(initial, params, root, step_fn, record=False):
        if record:
            keys = [jax.random.fold_in(root, t) for t in range(args.steps)]
            actor = lambda obs, key: (policy(params, obs, key), {})
            states = comparison_rollout(initial, keys, actor, step_fn)
            obs = [np.asarray(s.obs) for s in states[:-1]]
            actions = [np.asarray(policy(params, s.obs, key))
                       for s, key in zip(states[:-1], keys)]
            metrics = dict(reward=sum(float(s.reward) for s in states[1:]),
                           cost=sum(float(s.info['cost']) for s in states[1:]), steps=len(actions))
            return metrics, states, obs, actions
        state = initial
        states, obs, actions = [state], [], []
        reward = cost = 0.
        for t in range(args.steps):
            action = policy(params, state.obs, jax.random.fold_in(root, t))
            if record:
                obs.append(np.asarray(state.obs)); actions.append(np.asarray(action))
            state = step_fn(state, action)
            reward += float(state.reward); cost += float(state.info['cost'])
            if record: states.append(state)
            if bool(state.done): break
        return dict(reward=reward, cost=cost, steps=t+1), states, obs, actions

    records, observations, actions, anchors = [], [], [], []
    for i in range(args.pairs):
        seed = args.seed+i
        rk, ak = jax.random.split(jax.random.PRNGKey(seed))
        initial = reset(rk)
        nm, states, no, na = run(initial, original, ak, step, True)
        try:
            pos, radius = choose_hazard(base, states, np.random.default_rng(np.random.SeedSequence([args.hazard_seed, i])))
        except ValueError as error:
            records.append(dict(seed=seed, skipped=str(error)))
            print(f'Pair {i+1}: skipped ({error})', flush=True)
            continue
        real._hazard_radii = jnp.concatenate([radii, jnp.array([radius])])
        ri = real_reset(rk)
        np.testing.assert_allclose(ri.pipeline_state.q, initial.pipeline_state.q)
        positions = jnp.concatenate([initial.info['hazard_positions'], jnp.asarray(pos[None])])
        ps = ri.pipeline_state.replace(mocap_pos=ri.pipeline_state.mocap_pos.at[jnp.array(real._hazard_mocap_ids)].set(positions))
        ri = ri.replace(pipeline_state=ps, obs=real._get_obs(ps, positions),
                        info=dict(ri.info, hazard_positions=positions))
        real_step = jax.jit(lambda s, a: real_env.step(s, a))
        rm, _, ro, ra = run(ri, original, ak, real_step, True)
        def wins(a, b):
            return a['reward'] > b['reward'] + args.reward_margin and a['cost'] < b['cost'] - args.cost_margin
        winner = 'nominal' if wins(nm, rm) else 'real' if wins(rm, nm) else None
        records.append(dict(seed=seed, position=pos.tolist(), radius=radius, nominal=nm, real=rm, winner=winner))
        accepted = winner == 'nominal' if args.nominal_relabel else winner is not None
        records[-1]['accepted'] = accepted
        if accepted:
            if args.nominal_relabel:
                # Same physical states and sampled actions; only hazard sensing
                # changes. Exclude the terminal successor (no action target).
                relabel = jax.jit(lambda ps: real._get_obs(ps, positions))
                rebuilt = [np.asarray(relabel(s.pipeline_state)) for s in states[:-1]]
                probe = jax.jit(lambda ps: real._calculate_hazard_penetration(ps, positions)[-1])
                assert all(float(probe(s.pipeline_state)) == 0. for s in states)
                assert len(rebuilt) == len(na)
                observations.extend(rebuilt)
                actions.extend(na)
            else:
                observations.extend(no if winner == 'nominal' else ro)
                actions.extend(na if winner == 'nominal' else ra)
            anchors.extend(no); anchors.extend(ro)
        log({'collection/pair':i+1, 'collection/accepted':int(accepted),
             'collection/nominal_reward':nm['reward'], 'collection/real_reward':rm['reward'],
             'collection/nominal_cost':nm['cost'], 'collection/real_cost':rm['cost'],
             'collection/transitions':len(actions)})
        print(f'Pair {i+1}/{args.pairs}: winner={winner}, nominal={nm}, real={rm}', flush=True)
    report = dict(method='nominal-winner BC on hazard-visible observations (uniform weights)' if args.nominal_relabel else 'winner-filtered imitation, not SDPO', configuration={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, pairs=records)
    report_path = args.output/'report.json'
    report_path.write_text(json.dumps(report, indent=2))
    if not actions:
        print('No strictly dominant pairs found; no policy update or checkpoint saved.', flush=True)
        run_log.summary['status'] = 'no_qualifying_pairs'
        run_log.finish()
        return
    x, a, anchor = map(jnp.asarray, (np.stack(observations), np.stack(actions), np.stack(anchors)))
    np.savez_compressed(args.output/'winner_data.npz', observations=x, actions=a)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.learning_rate))
    params, opt_state = original, optimizer.init(original)
    distribution = net.parametric_action_distribution
    def loss(params, obs, acts, reference_obs):
        logits = net.policy_network.apply(norm, params, obs)
        raw_actions = distribution.inverse_postprocess(jnp.clip(acts, -1+1e-6, 1-1e-6))
        nll = -jnp.mean(distribution.log_prob(logits, raw_actions))
        old = distribution.create_dist(net.policy_network.apply(norm, original, reference_obs))
        new = distribution.create_dist(net.policy_network.apply(norm, params, reference_obs))
        kl = jnp.mean(jnp.sum(jnp.log(new.scale/old.scale) + (old.scale**2+(old.loc-new.loc)**2)/(2*new.scale**2)-.5, axis=-1))
        return nll+args.kl_weight*kl, (nll, kl)
    @jax.jit
    def update(params, opt_state, obs, acts, ref):
        (value, metrics), grad = jax.value_and_grad(loss, has_aux=True)(params, obs, acts, ref)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, value, metrics
    rng = np.random.default_rng(args.seed)
    history = []
    # Same wrapper and Evaluator used by SAC-Lag training. Reuse held-out
    # evaluation randomness for paired before/after comparisons.
    eval_env = wrappers.wrap(envs.get_environment('safe_pathway_walker2d', level=args.level),
                             episode_length=args.steps, action_repeat=1)
    evaluator = acting.Evaluator(eval_env, lambda p: infer(p, deterministic=False),
                                 num_eval_envs=args.eval_episodes, episode_length=args.steps,
                                 action_repeat=1, key=jax.random.PRNGKey(args.eval_seed))
    evaluations = []
    def evaluate(params, update):
        evaluator._key = jax.random.PRNGKey(args.eval_seed)
        metrics = {k:float(v) for k,v in evaluator.run_evaluation((norm, params), {}).items()}
        evaluations.append(dict(update=update, **metrics))
        log(dict(update=update, **metrics))
        print(f'Evaluation update {update}: {metrics}', flush=True)
    evaluate(original, 0)
    for i in range(args.updates):
        ids = rng.integers(len(x), size=args.batch_size)
        refs = rng.integers(len(anchor), size=args.batch_size)
        params, opt_state, value, (nll, kl) = update(params, opt_state, x[ids], a[ids], anchor[refs])
        if not np.isfinite(float(value)): raise RuntimeError('Nonfinite training loss; checkpoint not saved')
        history.append(dict(update=i+1, loss=float(value), nll=float(nll), kl=float(kl)))
        log(dict(update=i+1, **{'train/loss':float(value), 'train/bc_nll':float(nll), 'train/reference_kl':float(kl)}))
        if (i+1) % args.eval_every == 0 or i+1 == args.updates:
            evaluate(params, i+1)
    evaluation = []
    for i in range(args.eval_episodes):
        seed = args.seed+args.pairs+10000+i
        rk, ak = jax.random.split(jax.random.PRNGKey(seed))
        initial = reset(rk)
        before, *_ = run(initial, original, ak, step)
        after, *_ = run(initial, params, ak, step)
        evaluation.append(dict(seed=seed, before=before, after=after))
        print(f'Evaluation {i+1}: before={before}, after={after}', flush=True)
    aggregate = {label:{metric:float(np.mean([r[label][metric] for r in evaluation])) for metric in ('reward','cost','steps')} for label in ('before','after')}
    report.update(training=history, standard_evaluation=evaluations, winner_transitions=len(a), evaluation=evaluation, evaluation_mean=aggregate,
                  evaluation_scope='fresh nominal environments; no added hazard', checkpoint_status='experimental, not automatically promoted')
    checkpoint_io.save(args.output/'checkpoint', args.updates, (norm, params), cfg)
    saved = args.output/'checkpoint'/f'{args.updates:012d}'
    _, _, (_, restored) = load_actor(saved.resolve())
    for expected, actual in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(expected, actual)
    report['saved_checkpoint'] = str(saved)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f'Saved and verified checkpoint: {saved}', flush=True)
    run_log.summary['saved_checkpoint'] = str(saved)
    run_log.summary['accepted_pairs'] = sum(r.get('accepted', False) for r in records)
    run_log.finish()


if __name__ == '__main__':
    main()
