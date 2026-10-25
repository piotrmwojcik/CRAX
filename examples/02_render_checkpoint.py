#!/usr/bin/env python3
"""Render evaluation episodes from a saved SAC/SAC-Lagrange checkpoint.

Example:
    MUJOCO_GL=cgl JAX_PLATFORMS=cpu python examples/02_render_checkpoint.py \
      models/<run-name>/000100007936 --episodes 3 --camera fixedfar
"""
import argparse
import os
from pathlib import Path

# Must be set before MuJoCo is imported. Override with MUJOCO_GL if needed.
os.environ.setdefault("MUJOCO_GL", "egl")

from crax import envs
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from training import checkpoint as generic_checkpoint
from training.acme import running_statistics, specs
from training.agents.sac import networks as sac_networks


def load_checkpoint_policy(checkpoint_dir, deterministic):
    """Restore the policy onto the current device, even from a GPU checkpoint."""
    config = generic_checkpoint.load_config(
        str(checkpoint_dir / "sac_network_config.json")
    )
    network = generic_checkpoint.get_network(
        config, sac_networks.make_sac_networks
    )
    normalizer = running_statistics.init_state(
        specs.Array((config.observation_size,), jnp.dtype("float32"))
    )
    policy_params = network.policy_network.init(jax.random.PRNGKey(0))
    target = (normalizer, policy_params)

    # Build explicit restore args for this machine. In particular, point arrays
    # at the local CPU instead of trying to restore the saved cuda:0 sharding.
    cpu_sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(sharding=cpu_sharding), target
    )
    restored = ocp.PyTreeCheckpointer().restore(
        str(checkpoint_dir),
        args=ocp.args.PyTreeRestore(item=target, restore_args=restore_args),
    )
    make_inference_fn = sac_networks.make_inference_fn(network)
    return make_inference_fn(restored, deterministic=deterministic)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Checkpoint step directory")
    parser.add_argument("--env_name", default="safe_pathway_walker2d")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--steps", type=int, default=2000,
                        help="Maximum steps per episode")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", default="fixedfar")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--out_name", default="pathway_walker_final_eval")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions from the policy instead of using its mean")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint).expanduser().resolve()

    # Some Orbax versions place the checkpoint item and config under `~/`.
    if not (checkpoint_dir / "sac_network_config.json").exists():
        nested_item = checkpoint_dir / "~"
        if (nested_item / "sac_network_config.json").exists():
            checkpoint_dir = nested_item

    if not (checkpoint_dir / "sac_network_config.json").exists():
        raise FileNotFoundError(
            f"Could not find sac_network_config.json under {args.checkpoint}"
        )

    policy = load_checkpoint_policy(checkpoint_dir, deterministic=not args.stochastic)
    env = envs.get_environment(args.env_name, level=args.level)
    @jax.jit
    def rollout(key):
        state = env.reset(key)

        def step(carry, _):
            state, key = carry
            key, action_key = jax.random.split(key)
            action, _ = policy(state.obs, action_key)
            next_state = env.step(state, action)
            cost = next_state.info.get("cost", jnp.zeros_like(next_state.reward))
            return (next_state, key), (
                next_state.pipeline_state,
                next_state.reward,
                cost,
                next_state.done,
            )

        _, data = jax.lax.scan(step, (state, key), xs=None, length=args.steps)
        return data

    key = jax.random.PRNGKey(args.seed)
    video_frames = []
    for episode in range(args.episodes):
        key, episode_key = jax.random.split(key)
        pipeline_states, rewards, costs, dones = jax.device_get(rollout(episode_key))
        dones = np.asarray(dones, dtype=bool)
        done_indices = np.flatnonzero(dones)
        episode_steps = int(done_indices[0] + 1) if done_indices.size else args.steps
        pipeline_states = jax.tree_util.tree_map(
            lambda values: values[:episode_steps], pipeline_states
        )
        trajectory = [
            jax.tree_util.tree_map(lambda values, i=i: values[i], pipeline_states)
            for i in range(episode_steps)
        ]
        video_frames.extend(
            env.render(trajectory, width=args.width, height=args.height,
                       camera=args.camera)
        )
        print(
            f"episode {episode + 1}: steps={episode_steps} "
            f"return={float(np.asarray(rewards[:episode_steps]).sum()):.2f} "
            f"cost={float(np.asarray(costs[:episode_steps]).sum()):.2f}"
        )

    output = Path("videos") / f"{args.out_name}_{args.camera}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, np.stack(video_frames), fps=args.fps)
    print(f"Saved video: {output}")


if __name__ == "__main__":
    main()
