#!/usr/bin/env python3
"""Search single-action counterfactuals from a nominal, unconditioned SAC rollout.

Costs and rewards are undiscounted simulator episode sums. The optimizer is a
fixed-budget projected gradient search, not a certificate of global optimality.
Policy actions are stochastic, with shared per-step noise across all branches.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from itertools import zip_longest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

import jax
import jax.numpy as jnp
import numpy as np
from crax import envs


def render_frames(env, trajectory, camera):
    """Render bounded chunks rather than materializing an entire episode video."""
    for start in range(0, len(trajectory), 32):
        yield from env.render(
            [s.pipeline_state for s in trajectory[start : start + 32]],
            width=640,
            height=480,
            camera=camera,
        )


def save_nominal_charts(output, frames, velocity, rewards, costs, fps):
    """Save per-transition plots and a synchronized simulation/chart video.

    Frame zero is the initial state. Frame k shows the state after transition k
    and only the first k measurements, including the terminal transition.
    """
    import imageio.v2 as iio
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    velocity, rewards, costs = map(np.asarray, (velocity, rewards, costs))
    count = len(rewards)
    if count == 0 or len(velocity) != count or len(costs) != count:
        raise ValueError("Velocity, reward and cost must have equal nonzero lengths.")
    steps = np.arange(1, count + 1)
    np.savetxt(
        output / "nominal_metrics.csv",
        np.column_stack(
            (steps, velocity, rewards, costs, np.cumsum(rewards), np.cumsum(costs))
        ),
        delimiter=",",
        comments="",
        header="timestep,x_velocity,reward,cost,cumulative_reward,cumulative_cost",
    )
    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True)
    lines, cursors = [], []
    for ax, values, label, color in zip(
        axes,
        (velocity, rewards, costs),
        ("X velocity (m/s)", "Step reward", "Step cost"),
        ("#2878b5", "#299d55", "#d75043"),
    ):
        (line,) = ax.plot(steps, values, color=color, linewidth=1.5)
        lines.append(line)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, max(count, 1))
        finite = values[np.isfinite(values)]
        low, high = (
            (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
        )
        pad = max((high - low) * 0.1, abs(high) * 0.05, 0.01)
        ax.set_ylim(low - pad, high + pad)
    axes[-1].set_xlabel("Episode timestep (completed transitions)")
    title = fig.suptitle("Nominal rollout: per-step measurements", fontsize=11)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.12, top=0.86, hspace=0.22)
    fig.savefig(output / "nominal_charts.png", dpi=200)
    fig.savefig(output / "nominal_charts.pdf")
    for ax in axes:
        cursors.append(ax.axvline(0, color="#666666", linestyle="--", linewidth=1))
    cumulative_reward = np.concatenate(([0.0], np.cumsum(rewards)))
    cumulative_cost = np.concatenate(([0.0], np.cumsum(costs)))
    frame_count = 0
    with iio.get_writer(
        str(output / "nominal_with_charts.mp4"), fps=float(fps)
    ) as writer:
        for k, frame in enumerate(frames):
            if k > count:
                raise ValueError("More video frames than trajectory states.")
            for line, cursor, values in zip(lines, cursors, (velocity, rewards, costs)):
                line.set_data(steps[:k], values[:k])
                cursor.set_xdata([k, k])
            title.set_text(
                f"Nominal rollout | step {k}/{count}\n"
                f"Reward so far: {cumulative_reward[k]:.3f} | "
                f"Cost so far: {cumulative_cost[k]:.3f}"
            )
            canvas.draw()
            chart = np.asarray(canvas.buffer_rgba())[..., :3]
            frame = np.asarray(frame)[..., :3]
            if frame.shape != chart.shape:
                raise ValueError("Expected simulation frames of shape (480, 640, 3).")
            writer.append_data(np.concatenate((frame, chart), axis=1))
            frame_count += 1
    if frame_count != count + 1:
        raise ValueError("Video must include initial state plus every successor state.")


def save_lidar_visualization(output, observations, visible_threshold,
                             violation_steps=()):
    """Save a heatmap showing when lidar bins become active."""
    import matplotlib.pyplot as plt

    observations = np.asarray(observations)
    lidar = observations[:, -16:]
    visible = np.max(lidar, axis=1) > visible_threshold
    active_bins = np.sum(lidar > visible_threshold, axis=1)
    np.savetxt(
        output / 'nominal_lidar.csv',
        np.column_stack((np.arange(len(lidar)), lidar, visible, active_bins)),
        delimiter=',', comments='',
        header='timestep,' + ','.join(f'lidar_bin_{i}' for i in range(16))
        + ',visible,active_bins',
    )
    fig, (ax_heat, ax_count) = plt.subplots(
        2, 1, figsize=(11, 5), sharex=True,
        gridspec_kw={'height_ratios': (3, 1)},
    )
    image = ax_heat.imshow(
        lidar.T, origin='lower', aspect='auto', interpolation='nearest',
        cmap='viridis', vmin=0.0, vmax=max(float(lidar.max()), 1e-6),
    )
    ax_heat.set_ylabel('Lidar bin')
    ax_heat.set_title('When hazards enter the lidar observation')
    ax_heat.set_yticks(range(16))
    fig.colorbar(image, ax=ax_heat, label='Hazard proximity')
    timesteps = np.arange(len(active_bins))
    ax_count.plot(timesteps, active_bins, color='#2878b5')
    ax_count.set_ylabel('Active bins')
    ax_count.set_xlabel('Episode timestep')
    ax_count.set_ylim(bottom=0)
    for step in violation_steps:
        ax_heat.axvline(step, color='#d75043', linestyle='--', linewidth=1)
        ax_count.axvline(step, color='#d75043', linestyle='--', linewidth=1)
    fig.tight_layout()
    fig.savefig(output / 'nominal_lidar.png', dpi=180)
    fig.savefig(output / 'nominal_lidar.pdf')
    plt.close(fig)
    return int(np.sum(visible)), int(np.max(active_bins, initial=0))


def termination_reason(trajectory, horizon):
    """Report only causes recorded by the environment; do not guess a fall/collision."""
    final = trajectory[-1]
    info = final.info
    for key in ("termination_reason", "done_reason", "terminal_reason"):
        if key in info:
            value = info[key]
            if isinstance(value, str) and value:
                return value
    for key, label in (
        ("truncated", "environment time limit"),
        ("time_limit", "environment time limit"),
        ("timeout", "environment timeout"),
    ):
        if key in info and bool(np.asarray(info[key])):
            return label
    if bool(final.done):
        return "environment done (specific cause not exposed)"
    if len(trajectory) - 1 >= horizon:
        return "requested rollout horizon reached"
    return "rollout ended (cause unavailable)"


def write_comparison_video(
    path,
    nominal_frames,
    branch_frames,
    *,
    fps,
    intervention_step,
    nominal_reward,
    nominal_cost,
    branch_reward,
    branch_cost,
    action_change_l2,
    nominal_costs,
    branch_costs,
    nominal_end_reason,
    branch_end_reason,
):
    """Display running costs and both termination reasons until both episodes finish.

    The shorter episode holds its last frame while the other finishes. This does
    NOT step an environment after it has returned done=True.
    """
    import imageio.v2 as iio
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except OSError:
            font = ImageFont.load_default()

    nominal_costs = np.asarray(nominal_costs, dtype=float)
    branch_costs = np.asarray(branch_costs, dtype=float)
    running_costs = [np.r_[0.0, np.cumsum(c)] for c in (nominal_costs, branch_costs)]
    lengths = [len(nominal_costs), len(branch_costs)]
    reasons = [nominal_end_reason, branch_end_reason]

    def change(after, before):
        delta = after - before
        percent = (
            f"{100 * delta / abs(before):+.2f}%"
            if abs(before) > 1e-12
            else "percent n/a"
        )
        return f"{delta:+.4f} ({percent})"

    last_left = last_right = None
    with iio.get_writer(str(path), fps=float(fps)) as writer:
        for step, (left, right) in enumerate(
            zip_longest(nominal_frames, branch_frames)
        ):
            if left is not None:
                last_left = left
            if right is not None:
                last_right = right
            if last_left is None or last_right is None:
                raise ValueError(
                    "Both comparison rollouts must contain at least one frame."
                )
            height, width = np.asarray(last_left).shape[:2]
            canvas = Image.new("RGB", (2 * width, height + 200), "#141820")
            canvas.paste(
                Image.fromarray(np.asarray(last_left).astype(np.uint8)), (0, 112)
            )
            canvas.paste(
                Image.fromarray(np.asarray(last_right).astype(np.uint8)), (width, 112)
            )
            draw = ImageDraw.Draw(canvas)
            for idx, (x, title, reward, cost) in enumerate(
                [
                    (12, "NOMINAL / original action", nominal_reward, nominal_cost),
                    (
                        width + 12,
                        "COUNTERFACTUAL / action + delta",
                        branch_reward,
                        branch_cost,
                    ),
                ]
            ):
                actual_step = min(step, lengths[idx])
                finished = step >= lengths[idx]
                draw.text((x, 8), title, font=font, fill="white")
                draw.text(
                    (x, 33),
                    f"Cumulative cost: {running_costs[idx][actual_step]:.4f}  "
                    f"(final: {cost:.4f})",
                    font=font,
                    fill="#ffce64",
                )
                draw.text(
                    (x, 59),
                    f"Final reward: {reward:.4f} | Step: {actual_step}/{lengths[idx]}",
                    font=font,
                    fill="white",
                )
                status = ("TERMINATED: " + reasons[idx]) if finished else "RUNNING"
                draw.text(
                    (x, 85),
                    status,
                    font=font,
                    fill="#ff8c80" if finished else "#83e1a3",
                )
            draw.line((width, 0, width, canvas.height), fill="white", width=2)
            event = (
                "NEXT ACTION IS THE INTERVENTION"
                if step == intervention_step
                else (
                    "INTERVENTION JUST APPLIED"
                    if step == intervention_step + 1
                    else (
                        "Before intervention"
                        if step < intervention_step
                        else "Same policy continues"
                    )
                )
            )
            draw.text(
                (12, height + 128),
                f"Intervention at step {intervention_step} | {event} | ||delta||2 = {action_change_l2:.5f}",
                font=font,
                fill="#ffce64",
            )
            draw.text(
                (12, height + 157),
                f"Final reward change: {change(branch_reward, nominal_reward)}    "
                f"Final cost change: {change(branch_cost, nominal_cost)}",
                font=font,
                fill="white",
            )
            writer.append_data(np.asarray(canvas))


def make_action_objective(
    evaluate, cost_penalty, reward_penalty, finite_difference_step=1e-3
):
    """Finite-difference gradients of the exact simulator oracle used for video.

    Perturb only the action; all calls replay identical policy noise. Central
    differences become one-sided at the action bounds. No transformed physics
    executable is used for scoring or gradient estimation.
    """

    def objective(
        action,
        nominal_action,
        nominal_cost,
        nominal_reward,
        reward_tolerance,
        min_cost_reduction,
        *eval_args,
    ):
        cost, reward = map(float, evaluate(action, *eval_args))
        cost_violation = max(cost - nominal_cost + min_cost_reduction, 0.0)
        reward_violation = max(abs(reward - nominal_reward) - reward_tolerance, 0.0)
        loss = (
            0.5 * np.sum((np.asarray(action) - nominal_action) ** 2)
            + cost_penalty * cost_violation / max(abs(nominal_cost), 1.0)
            + reward_penalty * reward_violation / max(abs(nominal_reward), 1.0)
        )
        return float(loss), (cost, reward)

    def loss_and_grad(action, *args):
        action = np.asarray(action, dtype=np.float64)
        value = objective(action, *args)
        gradient = np.empty_like(action)
        for i in range(action.size):
            plus, minus = action.copy(), action.copy()
            plus[i] = min(1.0, action[i] + finite_difference_step)
            minus[i] = max(-1.0, action[i] - finite_difference_step)
            gradient[i] = (objective(plus, *args)[0] - objective(minus, *args)[0]) / (
                plus[i] - minus[i]
            )
        return value, jnp.asarray(gradient, dtype=jnp.float32)

    return loss_and_grad


def search_action(
    evaluate,
    nominal_action,
    nominal_cost,
    nominal_reward,
    reward_tolerance,
    min_cost_reduction,
    gradient_steps=100,
    learning_rate=0.01,
    cost_penalty=10.0,
    reward_penalty=10.0,
    max_grad_norm=10.0,
    eval_args=(),
    loss_and_grad=None,
):
    """Projected gradient descent on one action; policy parameters stay fixed.

    Return the smallest feasible visited change, not necessarily the last iterate.
    Gradients are finite-difference estimates of a possibly nonsmooth function.
    Feasibility is always checked by simulation.
    """
    nominal_action = jnp.asarray(nominal_action, dtype=jnp.float32)
    action = nominal_action
    loss_and_grad = loss_and_grad or make_action_objective(
        evaluate, cost_penalty, reward_penalty
    )
    best, history = None, []
    status = "completed"
    for iteration in range(gradient_steps + 1):
        # The final iteration evaluates the action after the last requested step.
        (loss, (cost, reward)), gradient = loss_and_grad(
            action,
            nominal_action,
            nominal_cost,
            nominal_reward,
            reward_tolerance,
            min_cost_reduction,
            *eval_args,
        )
        cost, reward, loss = float(cost), float(reward), float(loss)
        norm = float(jnp.linalg.norm(action - nominal_action))
        feasible = (
            np.isfinite([cost, reward, loss]).all()
            and cost <= nominal_cost - min_cost_reduction
            and abs(reward - nominal_reward) <= reward_tolerance
        )
        grad_norm = float(jnp.linalg.norm(gradient))
        row = dict(
            iteration=iteration,
            action=np.asarray(action).tolist(),
            cost=cost,
            reward=reward,
            loss=loss,
            action_change_l2=norm,
            gradient_norm=grad_norm,
            feasible=bool(feasible),
        )
        history.append(row)
        if feasible and (best is None or norm < best["action_change_l2"]):
            best = row.copy()
        if iteration == gradient_steps:
            break
        if not np.isfinite(grad_norm) or not np.isfinite(loss):
            status = "stopped_nonfinite_gradient_or_loss"
            break
        # Clip gradients for long-horizon stability, then project onto action bounds.
        gradient = gradient * jnp.minimum(
            1.0, max_grad_norm / jnp.maximum(grad_norm, 1e-12)
        )
        action = jnp.clip(action - learning_rate * gradient, -1.0, 1.0)
    return best, history, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, help="Run or numeric checkpoint directory"
    )
    parser.add_argument("--env", default="safe_pathway_walker2d")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument(
        "--env-kwargs",
        type=json.loads,
        default={},
        help="JSON environment overrides; match the training configuration",
    )
    parser.add_argument(
        "--steps", type=int, default=2000, help="Original episode time limit"
    )
    parser.add_argument(
        "--states", type=int, default=5,
        help="Maximum number of evenly spaced states per visibility-to-violation window"
    )
    parser.add_argument(
        "--visibility-threshold", type=float, default=1e-6,
        help="Minimum lidar signal considered visible",
    )
    parser.add_argument(
        "--violation-threshold", type=float, default=1e-6,
        help="Minimum grounded-foot penetration considered a safety violation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-rtol", type=float, default=0.05)
    parser.add_argument("--reward-atol", type=float, default=0.0)
    parser.add_argument("--min-cost-reduction", type=float, default=0.01)
    parser.add_argument("--gradient-steps", type=int, default=100)
    parser.add_argument("--action-lr", type=float, default=0.01)
    parser.add_argument(
        "--finite-difference-step",
        type=float,
        default=1e-3,
        help="Action perturbation for simulator gradient estimates",
    )
    parser.add_argument("--cost-penalty", type=float, default=10.0)
    parser.add_argument("--reward-penalty", type=float, default=10.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/counterfactual_actions")
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Render verified branches and side-by-side comparisons; nominal video is always saved first",
    )
    parser.add_argument(
        "--rollout-only",
        action="store_true",
        help="Save nominal rollout, charts and videos, then skip action optimization",
    )
    parser.add_argument("--camera", default="fixedfar")
    args = parser.parse_args()
    if min(args.steps, args.states, args.gradient_steps) < 1:
        parser.error("Steps, states and gradient-steps must be positive.")
    if (
        min(
            args.action_lr,
            args.cost_penalty,
            args.reward_penalty,
            args.max_grad_norm,
            args.finite_difference_step,
        )
        <= 0
    ):
        parser.error(
            "Learning rate, penalty weights, gradient clip and finite-difference step must be positive."
        )
    if args.min_cost_reduction <= 0 or min(args.reward_rtol, args.reward_atol) < 0:
        parser.error(
            "Cost reduction must be positive and reward tolerances nonnegative."
        )

    checkpoint = args.checkpoint.expanduser().resolve()
    if (
        not (checkpoint / "sac_network_config.json").exists()
        and not (checkpoint / "~" / "sac_network_config.json").exists()
    ):
        steps = sorted(
            p for p in checkpoint.iterdir() if p.is_dir() and p.name.isdigit()
        )
        if steps:
            checkpoint = max(steps, key=lambda p: int(p.name))
    if not (checkpoint / "sac_network_config.json").exists():
        checkpoint = checkpoint / "~"
    config = json.loads((checkpoint / "sac_network_config.json").read_text())
    if config.get("cost_conditioning", False):
        parser.error("This script requires an unconditioned SAC/SAC-Lag checkpoint.")
    if not isinstance(config["observation_size"], int):
        parser.error("Only vector-observation checkpoints are supported.")

    spec = importlib.util.spec_from_file_location(
        "renderer", ROOT / "examples/02_render_checkpoint.py"
    )
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    policy = renderer.load_checkpoint_policy(checkpoint, deterministic=False)
    env = envs.get_environment(args.env, level=args.level, **args.env_kwargs)
    step_env = jax.jit(env.step)
    # Index randomness by absolute episode step, not by branch-local step.
    # Every candidate sees the same noise, including after the nominal episode
    # ends if the intervention allows the counterfactual to survive longer.
    action_root_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1)
    act = jax.jit(
        lambda obs, step: policy(obs, jax.random.fold_in(action_root_key, step))[0]
    )
    initial = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
    if initial.obs.shape != (config["observation_size"],):
        raise ValueError("Environment observations do not match checkpoint dimensions.")

    def rollout(state, horizon, first_action=None, start_step=0, record=True):
        states, actions, rewards, costs, penetrations = [state], [], [], [], []
        for t in range(horizon):
            if bool(state.done):
                break
            action = (
                act(state.obs, start_step + t)
                if t or first_action is None
                else jnp.asarray(first_action, dtype=jnp.float32)
            )
            state = step_env(state, action)
            if record:
                actions.append(np.asarray(action))
            rewards.append(float(state.reward))
            costs.append(float(state.info["cost"]))
            penetrations.append(np.asarray(state.info.get("hazard_penetration", [])))
            if record:
                states.append(state)
        return (states, np.asarray(actions), np.asarray(rewards),
                np.asarray(costs), penetrations)

    def evaluate(action, state, horizon, prefix_cost, prefix_reward):
        # Reuse the exact policy/physics execution path that records videos.
        # Converting the injected action to float32 happens inside rollout.
        _, _, suffix_rewards, suffix_costs, _ = rollout(
            state, horizon, action, start_step=args.steps - horizon, record=False
        )
        return (
            float(suffix_costs.sum()) + prefix_cost,
            float(suffix_rewards.sum()) + prefix_reward,
        )

    loss_and_grad = make_action_objective(
        evaluate, args.cost_penalty, args.reward_penalty, args.finite_difference_step
    )

    args.output.mkdir(parents=True, exist_ok=True)
    states, actions, rewards, costs, penetrations = rollout(initial, args.steps)
    if not len(actions):
        raise ValueError("Nominal episode has no valid pre-action states.")
    total_cost, total_reward = float(costs.sum()), float(rewards.sum())
    nominal_end_reason = termination_reason(states, args.steps)
    tolerance = args.reward_atol + args.reward_rtol * abs(total_reward)
    # A hazard is considered visible when any lidar bin is active. A violation
    # starts when any grounded foot has positive penetration. For each violation,
    # optimize every state from first visibility through the violating action.
    lidar = np.asarray([np.asarray(s.obs)[-16:] for s in states[:-1]])
    visible = np.max(lidar, axis=1) > args.visibility_threshold
    violation = np.asarray([
        p.size > 0 and np.max(p) > args.violation_threshold
        for p in penetrations
    ], dtype=bool)
    violation_indices = np.flatnonzero(violation)
    windows = []
    for end in violation_indices:
        starts = np.flatnonzero(visible[: end + 1])
        if starts.size:
            windows.append((int(starts[0]), int(end)))
    selected = np.unique(np.concatenate([
        np.linspace(start, end, min(args.states, end - start + 1), dtype=int)
        for start, end in windows
    ])) if windows else np.empty(0, dtype=int)

    def save_rollout(name, trajectory, acts, rews, cs, hazard_data=None, force_video=False):
        np.savez_compressed(
            args.output / f"{name}.npz",
            observations=np.stack([np.asarray(s.obs) for s in trajectory]),
            q=np.stack([np.asarray(s.pipeline_state.q) for s in trajectory]),
            qd=np.stack([np.asarray(s.pipeline_state.qd) for s in trajectory]),
            actions=acts,
            rewards=rews,
            costs=cs,
            hazard_penetration=(
                np.asarray(hazard_data, dtype=object) if hazard_data is not None
                else np.empty((0,), dtype=np.float32)
            ),
            x_velocity=np.asarray(
                [float(s.metrics.get("x_velocity", np.nan)) for s in trajectory[1:]]
            ),
            done=np.array([bool(s.done) for s in trajectory]),
        )
        if args.video or force_video:
            import imageio.v2 as iio

            with iio.get_writer(
                str(args.output / f"{name}.mp4"), fps=float(1 / env.dt)
            ) as writer:
                for frame in render_frames(env, trajectory, args.camera):
                    writer.append_data(frame)

    print(
        "Saving complete nominal rollout video before counterfactual optimization...",
        flush=True,
    )
    save_rollout("nominal", states, actions, rewards, costs,
                 hazard_data=penetrations, force_video=True)
    print(f'Nominal video saved: {(args.output / "nominal.mp4").resolve()}', flush=True)
    save_nominal_charts(
        args.output,
        render_frames(env, states, args.camera),
        [float(s.metrics.get("x_velocity", np.nan)) for s in states[1:]],
        rewards,
        costs,
        fps=float(1 / env.dt),
    )
    print(
        f'Nominal video with charts saved: {(args.output / "nominal_with_charts.mp4").resolve()}',
        flush=True,
    )
    visible_steps, max_active_bins = save_lidar_visualization(
        args.output,
        np.asarray([np.asarray(s.obs) for s in states]),
        args.visibility_threshold,
        violation_indices,
    )
    print(
        f'Lidar visualization saved: {(args.output / "nominal_lidar.png").resolve()} '
        f'({visible_steps} visible states; max {max_active_bins} active bins)',
        flush=True,
    )
    report = dict(
        checkpoint=str(checkpoint),
        settings={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        nominal_cost=total_cost,
        nominal_reward=total_reward,
        nominal_length=len(actions),
        reward_tolerance=tolerance,
        policy_mode="stochastic",
        action_noise="shared_by_absolute_episode_step",
        gradient_method="finite_difference",
        search="Projected gradient descent; best feasible iterate; no global minimum guarantee",
        branches=[],
        safety_detection={
            "visibility_threshold": args.visibility_threshold,
            "violation_threshold": args.violation_threshold,
            "violation_steps": violation_indices.tolist(),
            "visibility_violation_windows": [list(w) for w in windows],
            "selected_states": selected.tolist(),
        },
    )
    print(
        f"Nominal: steps={len(actions)}, cost={total_cost:.6f}, reward={total_reward:.6f}",
        flush=True,
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    if args.rollout_only:
        return
    for index in selected:
        index = int(index)
        prefix_cost, prefix_reward = costs[:index].sum(), rewards[:index].sum()
        eval_args = (
            states[index],
            args.steps - index,
            float(prefix_cost),
            float(prefix_reward),
        )
        # Check that stepping the saved state reproduces the nominal continuation.
        c, r = evaluate(actions[index], *eval_args)
        if not np.allclose([c, r], [total_cost, total_reward], rtol=1e-4, atol=1e-4):
            raise RuntimeError(
                f"Nominal continuation mismatch at step {index}: "
                f"original cost/reward=({total_cost:.9f}, {total_reward:.9f}), "
                f"replayed=({c:.9f}, {r:.9f})."
            )
        print(
            f"State {index}: nominal replay verified; starting {args.gradient_steps} "
            "finite-difference gradient steps...",
            flush=True,
        )
        best, history, status = search_action(
            evaluate,
            actions[index],
            total_cost,
            total_reward,
            tolerance,
            args.min_cost_reduction,
            args.gradient_steps,
            args.action_lr,
            args.cost_penalty,
            args.reward_penalty,
            args.max_grad_norm,
            eval_args=eval_args,
            loss_and_grad=loss_and_grad,
        )
        row = dict(
            step=index,
            nominal_action=actions[index].tolist(),
            gradient_steps_completed=len(history) - 1,
            optimizer_status=status,
            optimization_history=history,
            result=best,
        )
        if best is not None:
            branch_states, ba, br, bc, bp = rollout(
                states[index], args.steps - index, best["action"], start_step=index
            )
            full_rewards, full_costs = np.concatenate(
                [rewards[:index], br]
            ), np.concatenate([costs[:index], bc])
            # Independently verify the saved full rollout, including terminal cost.
            verified = (
                full_costs.sum() <= total_cost - args.min_cost_reduction
                and abs(full_rewards.sum() - total_reward) <= tolerance
            )
            row["verified"] = bool(verified)
            row["full_cost"] = float(full_costs.sum())
            row["full_reward"] = float(full_rewards.sum())
            full_states = states[:index] + branch_states
            branch_end_reason = termination_reason(full_states, args.steps)
            row["counterfactual_termination_reason"] = branch_end_reason
            if verified:
                save_rollout(
                    f"branch_{index:05d}",
                    full_states,
                    np.concatenate([actions[:index], ba]),
                    full_rewards,
                    full_costs,
                    hazard_data=penetrations[:index] + bp,
                )
                if args.video:
                    filename = f"comparison_{index:05d}.mp4"
                    write_comparison_video(
                        args.output / filename,
                        render_frames(env, states, args.camera),
                        render_frames(env, full_states, args.camera),
                        fps=float(1 / env.dt),
                        intervention_step=index,
                        nominal_reward=total_reward,
                        nominal_cost=total_cost,
                        branch_reward=row["full_reward"],
                        branch_cost=row["full_cost"],
                        action_change_l2=best["action_change_l2"],
                        nominal_costs=costs,
                        branch_costs=full_costs,
                        nominal_end_reason=nominal_end_reason,
                        branch_end_reason=branch_end_reason,
                    )
                    row["comparison_video"] = filename
        report["branches"].append(row)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2))
        print(
            f'State {index}: {"feasible" if best is not None else "no feasible candidate found"}; '
            f'{len(history) - 1} gradient steps; status={status}; verified={row.get("verified", False)}',
            flush=True,
        )
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
