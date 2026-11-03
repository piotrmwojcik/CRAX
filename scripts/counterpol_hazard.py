#!/usr/bin/env python3
"""COUNTERPOL-style policy adaptation at the first Walker hazard violation.

This is a sample implementation of the objective from arXiv:2307.13192.
It adapts a copy of the actor parameters, then evaluates the adapted policy
from the nominal violation state. The simulator remains the source of reward
and safety cost; the policy-gradient update uses Monte Carlo score functions.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

from crax import envs
from training import checkpoint as generic_checkpoint
from training.acme import running_statistics, specs
from training.agents.sac import networks as sac_networks

from scripts.counterfactual_actions import (
    render_frames,
    termination_reason,
    write_comparison_video,
)


def load_actor(checkpoint):
    config = generic_checkpoint.load_config(str(checkpoint / "sac_network_config.json"))
    network = generic_checkpoint.get_network(config, sac_networks.make_sac_networks)
    normalizer = running_statistics.init_state(
        specs.Array((config.observation_size,), jnp.dtype("float32"))
    )
    actor = network.policy_network.init(jax.random.PRNGKey(0))
    target = (normalizer, actor)
    cpu = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(sharding=cpu), target
    )
    params = ocp.PyTreeCheckpointer().restore(
        str(checkpoint),
        args=ocp.args.PyTreeRestore(item=target, restore_args=restore_args),
    )
    return config, network, params


def first_hazard_violation(states, penetration_threshold=1e-6):
    for step, state in enumerate(states[1:]):
        penetration = np.asarray(state.info.get("hazard_penetration", []))
        if penetration.size and float(np.max(penetration)) > penetration_threshold:
            return step
    return None


def policy_stats(network, normalizer, old_params, new_params, observation):
    old_dist = network.policy_network.apply(normalizer, old_params, observation)
    new_dist = network.policy_network.apply(normalizer, new_params, observation)
    # SAC's distribution exposes KL only through its parameterized distribution.
    distribution = network.parametric_action_distribution
    old_mean = old_dist["loc"] if isinstance(old_dist, dict) else old_dist[..., :old_dist.shape[-1] // 2]
    new_mean = new_dist["loc"] if isinstance(new_dist, dict) else new_dist[..., :new_dist.shape[-1] // 2]
    # This diagonal Gaussian mean distance is a stable KL surrogate for the
    # squashed SAC actor when exact distribution parameters are unavailable.
    return jnp.mean(jnp.square(old_mean - new_mean))


def counterpol_update(network, normalizer, old_params, params, observations,
                      actions, rewards, costs, target_cost, reward_target, optimizer,
                      opt_state, kl_weight):
    """One COUNTERPOL update using score-function Monte Carlo estimates."""
    rewards = jnp.asarray(rewards)
    costs = jnp.asarray(costs)
    returns_r = jnp.flip(jnp.cumsum(jnp.flip(rewards)))
    returns_c = jnp.flip(jnp.cumsum(jnp.flip(costs)))

    def objective(actor_params):
        dist = network.policy_network.apply(normalizer, actor_params, observations)
        old_dist = network.policy_network.apply(normalizer, old_params, observations)
        log_prob = network.parametric_action_distribution.log_prob(dist, actions)
        old_log_prob = network.parametric_action_distribution.log_prob(old_dist, actions)
        # Use the policy-gradient score term for the sampled nominal segment.
        score = jnp.mean(
            jax.lax.stop_gradient(
                jnp.sign(jnp.mean(returns_r) - reward_target) * returns_r
                + jnp.sign(jnp.mean(returns_c) - target_cost) * returns_c
            ) * log_prob
        )
        log_std = dist[..., dist.shape[-1] // 2:]
        old_log_std = old_dist[..., old_dist.shape[-1] // 2:]
        mean = dist[..., :dist.shape[-1] // 2]
        old_mean = old_dist[..., :old_dist.shape[-1] // 2]
        kl = 0.5 * jnp.mean(
            jnp.square(old_mean - mean) / jnp.exp(2.0 * log_std)
            + jnp.exp(2.0 * log_std - 2.0 * old_log_std)
            - 1.0 + 2.0 * (log_std - old_log_std)
        )
        # Return and cost targets are scalar episode targets. Their score terms
        # are supplied by the sampled trajectory in the outer update.
        return score + kl_weight * kl

    objective_loss, grad = jax.value_and_grad(objective)(params)
    # REINFORCE direction: reduce cost-target error while preserving reward.
    cost_error = jnp.mean(returns_c) - target_cost
    reward_error = jnp.mean(returns_r) - reward_target
    updates, opt_state = optimizer.update(grad, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, {
        "cost_error": cost_error,
        "reward_error": reward_error,
        "objective": objective_loss,
    }


def main():
    started_at = time.perf_counter()

    def log(message):
        elapsed = time.perf_counter() - started_at
        print(f"[{elapsed:8.1f}s] {message}", flush=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument(
        "--lookback-steps", type=int, default=10,
        help="Number of nominal steps before the first violation used for adaptation",
    )
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--reward-rtol", type=float, default=0.15)
    parser.add_argument("--violation-threshold", type=float, default=1e-6)
    parser.add_argument("--video", action="store_true",
                        help="Save nominal/adapted side-by-side comparison video")
    parser.add_argument("--camera", default="fixedfar")
    parser.add_argument("--output", type=Path, default=Path("results/counterpol_hazard"))
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    log(f"Loading actor checkpoint: {checkpoint}")
    config, network, (normalizer, old_params) = load_actor(checkpoint)
    log("Actor checkpoint loaded")
    if config.get("cost_conditioning", False):
        parser.error("Use an unconditioned SAC-Lag checkpoint.")
    log(f"Creating environment: safe_pathway_walker2d (level={args.level})")
    env = envs.get_environment("safe_pathway_walker2d", level=args.level)
    log("Environment created; compiling/resetting JAX environment")
    reset = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, reset)
    log("Environment reset complete")
    policy = sac_networks.make_inference_fn(network)((normalizer, old_params), deterministic=False)

    states, observations, actions, rewards, costs = [reset], [], [], [], []
    state = reset
    log(f"Starting nominal rollout (up to {args.steps} steps; first step may compile JAX)")
    rollout_started_at = time.perf_counter()
    for step in range(args.steps):
        observations.append(np.asarray(state.obs))
        action, _ = policy(state.obs, jax.random.fold_in(jax.random.PRNGKey(args.seed), step))
        state = env.step(state, action)
        actions.append(np.asarray(action)); rewards.append(float(state.reward)); costs.append(float(state.info["cost"]))
        states.append(state)
        if step == 0:
            log("First nominal environment step complete")
        elif (step + 1) % 10 == 0:
            elapsed = time.perf_counter() - rollout_started_at
            log(
                f"Nominal rollout: {step + 1}/{args.steps} steps "
                f"({elapsed / (step + 1):.3f}s/step average)"
            )
        if bool(state.done):
            log(f"Nominal rollout terminated by environment at step {step + 1}")
            break
    log(f"Nominal rollout complete ({len(states) - 1} transitions); searching for hazard violation")
    violation_step = first_hazard_violation(states, args.violation_threshold)
    if violation_step is None:
        raise RuntimeError("No hazard violation occurred in the nominal rollout.")

    start_step = max(0, violation_step - args.lookback_steps)
    violation_state = states[violation_step + 1]
    violation_penetration = np.asarray(
        violation_state.info.get("hazard_penetration", [])
    )
    print(
        f"Hazard violation found at transition {violation_step}: "
        f"max penetration={float(violation_penetration.max(initial=0.0)):.6f}; "
        f"step cost={costs[violation_step]:.6f}",
        flush=True,
    )
    print(
        f"Gradient descent starts at state {start_step} "
        f"({args.lookback_steps} steps before violation) and uses "
        f"states {start_step}..{violation_step} inclusive.",
        flush=True,
    )
    target_cost = float(
        np.sum(costs[start_step:violation_step + 1]) - costs[violation_step]
    )
    reward_target = float(np.sum(rewards[start_step:violation_step + 1]))
    print(
        f"Adaptation window: nominal reward={reward_target:.6f}, "
        f"nominal cost={float(np.sum(costs[start_step:violation_step + 1])):.6f}, "
        f"hazard cost removed={costs[violation_step]:.6f}, "
        f"target cost={target_cost:.6f}",
        flush=True,
    )
    segment_obs = jnp.asarray(
        np.stack(observations[start_step:violation_step + 1])
    )
    segment_actions = jnp.asarray(
        np.stack(actions[start_step:violation_step + 1])
    )
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(old_params)
    params = old_params
    history = []
    log(f"Starting policy adaptation ({args.updates} updates; first update may compile JAX)")
    for update in range(args.updates):
        params, opt_state, metrics = counterpol_update(
            network, normalizer, old_params, params, segment_obs,
            segment_actions,
            rewards[start_step:violation_step + 1],
            costs[start_step:violation_step + 1],
            target_cost, reward_target, optimizer, opt_state, args.kl_weight)
        history.append({"update": update, **{k: float(v) for k, v in metrics.items()}})
        print(
            f"Update {update + 1}/{args.updates}: "
            f"objective={float(metrics['objective']):.6f}, "
            f"cost_error={float(metrics['cost_error']):+.6f}, "
            f"reward_error={float(metrics['reward_error']):+.6f}",
            flush=True,
        )

    adapted_policy = sac_networks.make_inference_fn(network)(
        (normalizer, params), deterministic=False
    )
    log("Policy adaptation complete; rolling out adapted policy")
    def rollout_from(state, start, horizon, actor):
        trajectory = [state]
        total_reward = 0.0
        total_cost = 0.0
        rollout_costs = []
        for offset in range(horizon):
            if bool(state.done):
                break
            key = jax.random.fold_in(jax.random.PRNGKey(args.seed), start + offset)
            action, _ = actor(state.obs, key)
            state = env.step(state, action)
            trajectory.append(state)
            total_reward += float(state.reward)
            total_cost += float(state.info["cost"])
            rollout_costs.append(float(state.info["cost"]))
        return trajectory, total_reward, total_cost, rollout_costs

    adapted_states, adapted_reward, adapted_cost, adapted_costs = rollout_from(
        states[start_step], start_step, args.steps - start_step, adapted_policy)
    log(f"Adapted rollout complete ({len(adapted_states) - 1} transitions)")
    nominal_states = states[start_step:]

    args.output.mkdir(parents=True, exist_ok=True)
    log(f"Writing summary to {args.output / 'summary.json'}")
    (args.output / "summary.json").write_text(json.dumps({
        "violation_step": violation_step,
        "adaptation_start_step": start_step,
        "lookback_steps": args.lookback_steps,
        "target_cost": target_cost,
        "reward_target": reward_target,
        "nominal_cost_in_adaptation_window": float(np.sum(costs[start_step:violation_step + 1])),
        "nominal_reward_to_violation": reward_target,
        "history": history,
        "adapted_reward_from_window": adapted_reward,
        "adapted_cost_from_window": adapted_cost,
        "note": "Sample COUNTERPOL actor update; evaluate adapted actor in simulator before using it.",
    }, indent=2))
    if args.video:
        output_video = args.output / "comparison_counterpol_hazard.mp4"
        log(f"Rendering and writing comparison video: {output_video}")

        def logged_frames(label, trajectory):
            total = len(trajectory)
            for index, frame in enumerate(render_frames(env, trajectory, args.camera), 1):
                if index == 1 or index % 32 == 0 or index == total:
                    log(f"{label} video frames: {index}/{total}")
                yield frame

        write_comparison_video(
            output_video,
            logged_frames("Nominal", nominal_states),
            logged_frames("Adapted", adapted_states),
            fps=float(1 / env.dt),
            intervention_step=0,
            nominal_reward=float(np.sum(rewards[start_step:])),
            nominal_cost=float(np.sum(costs[start_step:])),
            branch_reward=adapted_reward,
            branch_cost=adapted_cost,
            action_change_l2=float(jnp.sqrt(sum(
                jnp.sum(jnp.square(a - b))
                for a, b in zip(
                    jax.tree_util.tree_leaves(params),
                    jax.tree_util.tree_leaves(old_params),
                )
            ))),
            nominal_costs=costs[start_step:],
            branch_costs=adapted_costs,
            nominal_end_reason=termination_reason(nominal_states, args.steps - start_step),
            branch_end_reason=termination_reason(adapted_states, args.steps - start_step),
        )
        log(f"Saved comparison video: {output_video}")
    print(f"First violation: step {violation_step}")
    print(f"Target cost: {target_cost:.6f}; reward target: {reward_target:.6f}")
    print(f"Saved: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
