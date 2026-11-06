"""Compare a visual-only hazard against the same additional real cost hazard."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys

os.environ.setdefault(
    "MUJOCO_GL",
    "cgl" if sys.platform == "darwin" else "egl",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as iio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from crax import envs
from scripts.visualize_policy import load_policy, save_video

# ============================================================
# Environment helpers
# ============================================================


def unwrap(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def rollout(initial, keys, policy, step):
    states = [initial]

    for key in keys:
        action, _ = policy(states[-1].obs, key)
        state = step(states[-1], action)
        states.append(state)

        if bool(state.done):
            break

    return states


# ============================================================
# Find the largest safe rectangular hazard
# ============================================================


def choose_hazard(
    base,
    states,
    rng,
    clearance=0.02,
    min_half_length=0.03,
    randomize=False,
    count=None,
):
    """
    Find the widest available interval along x that does
    not overlap the nominal walker's near-ground feet
    or existing hazard regions.

    The selected interval defines the same rectangular
    geometry for both the nominal and real simulations.

    Returns:
        position: [x, y, z]
        half_length: half-length of the hazard along x

    With randomize=True, sample a feasible gap, size, and center instead
    of always filling the widest gap.
    With count set, fill the widest distinct safe gaps to maximize total
    hazard length, retaining clearance and a numerical boundary margin.
    Randomness only breaks width ties. Raise ValueError if too few gaps exist.
    """

    # Extract foot positions across the nominal trajectory.
    feet = np.stack(
        [np.asarray(s.pipeline_state.x.pos)[base._foot_link_ids] for s in states]
    )

    # Only consider feet near the ground.
    grounded = feet[
        feet[..., 2] < 2 * base._hazard_height + 0.08,
        :2,
    ]

    if len(grounded) == 0:
        raise ValueError("No near-ground foot positions in nominal rollout")

    model = base.sys.mj_model
    data = mujoco.MjData(model)

    foot_geoms = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )
        for name in ("foot_geom", "foot_left_geom")
    ]

    if any(gid < 0 for gid in foot_geoms):
        raise ValueError("Could not find Walker2d foot geoms")

    # Each forbidden interval is protected from hazard
    # placement along the walking direction (x).
    forbidden = []

    # Protect nominal foot contact reference points.
    for x in grounded[:, 0]:
        forbidden.append(
            (
                float(x - clearance),
                float(x + clearance),
            )
        )

    # Protect the visible geometry of the feet as well.
    for state in states:
        data.qpos[:] = np.asarray(state.pipeline_state.q)

        mujoco.mj_forward(model, data)

        for gid in foot_geoms:
            foot_bound = float(model.geom_size[gid, 0] + model.geom_size[gid, 1])

            if data.geom_xpos[gid, 2] - foot_bound <= 2 * base._hazard_height + 0.02:
                x = float(data.geom_xpos[gid, 0])

                forbidden.append(
                    (
                        x - foot_bound - clearance,
                        x + foot_bound + clearance,
                    )
                )

    # Also avoid overlapping existing hazards.
    real_xy = np.asarray(states[0].info["hazard_positions"])[:, :2]

    is_rect = np.asarray(
        base._hazard_is_rect,
        dtype=bool,
    )

    extents = np.asarray(base._hazard_half_extents)

    radii = np.asarray(base._hazard_radii)

    for idx, (x, _y) in enumerate(real_xy):
        half_x = float(extents[idx, 0] if is_rect[idx] else radii[idx])

        forbidden.append(
            (
                float(x - half_x - clearance),
                float(x + half_x + clearance),
            )
        )

    # Determine search range from the nominal torso.
    torso = np.array([float(s.pipeline_state.x.pos[0, 0]) for s in states])

    low = float(torso.min() + 0.15)
    high = float(torso.max() - 0.15)

    if high <= low:
        raise ValueError("Episode has insufficient walking distance")

    # Clip occupied regions to the available range.
    occupied = sorted(
        (
            max(low, left),
            min(high, right),
        )
        for left, right in forbidden
        if right > low and left < high
    )

    # Merge overlapping forbidden intervals.
    merged = []

    for left, right in occupied:
        if merged and left <= merged[-1][1]:
            merged[-1] = (
                merged[-1][0],
                max(merged[-1][1], right),
            )
        else:
            merged.append((left, right))

    # Find the free intervals between occupied regions.
    # Only consider interior gaps.
    gaps = [
        (a[1], b[0])
        for a, b in zip(
            merged[:-1],
            merged[1:],
        )
        if b[0] - a[1] >= 2 * min_half_length
    ]

    if not gaps:
        raise ValueError("No safe between-footsteps interval " "for a full-width box")

    if count is not None:
        if count < 1:
            raise ValueError('Hazard count must be positive')
        if len(gaps) < count:
            raise ValueError(f'Only {len(gaps)} safe gaps available for {count} hazards')
        # Sort a random permutation stably so equally wide gaps use the seed,
        # while the chosen set maximizes total length across distinct gaps.
        selected = sorted(rng.permutation(len(gaps)),
                          key=lambda index: gaps[index][1] - gaps[index][0],
                          reverse=True)[:count]
        positions, sizes = [], []
        for index in selected:
            left, right = gaps[index]
            size = float((right - left) / 2 - 1e-5)
            center = float((left + right) / 2)
            positions.append([center, 0.0, float(base._hazard_height)])
            sizes.append(size)
        return np.asarray(positions), np.asarray(sizes)

    if randomize:
        left, right = gaps[int(rng.integers(len(gaps)))]
        maximum = (right - left) / 2 - 1e-5
        half_length = float(rng.uniform(min(min_half_length, maximum), maximum))
        center_x = float(rng.uniform(left + half_length, right - half_length))
        return np.array([center_x, 0.0, float(base._hazard_height)]), half_length

    # Select the widest feasible gap.
    width = max(right - left for left, right in gaps)

    best = [
        (left, right)
        for left, right in gaps
        if np.isclose(
            right - left,
            width,
            atol=1e-8,
        )
    ]

    # Randomness is only used to break ties.
    left, right = best[int(rng.integers(len(best)))]

    # Small numerical margin at boundaries.
    half_length = (right - left) / 2 - 1e-5

    center_x = (left + right) / 2

    position = np.array(
        [
            center_x,
            0.0,
            float(base._hazard_height),
        ]
    )

    return position, float(half_length)


# ============================================================
# Render the nominal BLUE visual-only hazard
# ============================================================


def render(
    renderer,
    model,
    state,
    position,
    radius,
    camera,
    platform_half_width=1.0,
):
    ps = state.pipeline_state
    data = mujoco.MjData(model)

    data.qpos[:] = np.asarray(ps.q)
    data.qvel[:] = np.asarray(ps.qd)
    data.mocap_pos[:] = np.asarray(ps.mocap_pos)
    data.mocap_quat[:] = np.asarray(ps.mocap_quat)

    mujoco.mj_forward(model, data)

    renderer.update_scene(
        data,
        camera=camera,
    )

    if position is None:
        return renderer.render().copy()

    scene = renderer.scene

    positions = np.atleast_2d(position)
    radii = np.atleast_1d(radius)
    if scene.ngeom + len(positions) > scene.maxgeom:
        raise RuntimeError("No free rendering geom slots")
    for center, half_length in zip(positions, radii, strict=True):
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
            np.array([half_length, platform_half_width, center[2]]),
            center, np.eye(3).ravel(),
            np.array([0.0, 0.25, 1.0, 1.0], dtype=np.float32),
        )
        scene.ngeom += 1

    return renderer.render().copy()


# ============================================================
# Main experiment
# ============================================================


def main():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--checkpoint",
        required=True,
    )

    p.add_argument(
        "--level",
        type=int,
        default=1,
    )

    p.add_argument(
        "--episode_length",
        type=int,
        default=2000,
    )

    p.add_argument(
        "--num_episodes",
        type=int,
        default=5,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    p.add_argument(
        "--hazard-seed",
        type=int,
        default=0,
        help="Independent seed for added-hazard placement",
    )

    p.add_argument(
        "--video_fps",
        type=int,
        default=125,
    )

    p.add_argument(
        "--deterministic",
        action="store_true",
    )

    p.add_argument(
        "--camera",
        default="fixedfar",
    )

    p.add_argument(
        "--platform-half-width",
        type=float,
        default=1.0,
        help=("Half-width of both rectangular " "hazard strips along y"),
    )

    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("visualizations/fake_real_hazard"),
    )

    p.add_argument('--num-added-hazards', type=int, default=1,
                   help='Number of simultaneous blue/orange hazards per episode pair')
    args = p.parse_args()
    if args.num_added_hazards < 1:
        p.error('--num-added-hazards must be positive')

    if args.platform_half_width <= 0:
        p.error("--platform-half-width must be positive")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load trained SAC-Lagrangian actor.
    policy = jax.jit(
        load_policy(
            args.checkpoint,
            deterministic=args.deterministic,
        )
    )

    # Nominal: original CRAX environment.
    nominal_env = envs.get_environment(
        "safe_pathway_walker2d",
        level=args.level,
    )

    # Real: original hazards + the requested added hazards.
    real_env = envs.get_environment(
        "safe_pathway_walker2d",
        level=args.level,
        num_hazards=(unwrap(nominal_env)._num_hazards + args.num_added_hazards),
    )

    base = unwrap(nominal_env)
    real = unwrap(real_env)

    # --------------------------------------------------------
    # Configure added REAL hazard as a rectangular box.
    # --------------------------------------------------------

    original_radii = real._hazard_radii[:-args.num_added_hazards]
    original_extents = real._hazard_half_extents[:-args.num_added_hazards]

    platform_half_width = args.platform_half_width

    real._hazard_is_rect = real._hazard_is_rect.at[-args.num_added_hazards:].set(True)
    real_model = real_env.sys.mj_model
    added_geoms = []
    for hazard_id in range(base._num_hazards + 1, real._num_hazards + 1):
        body = mujoco.mj_name2id(real_model, mujoco.mjtObj.mjOBJ_BODY, f'hazard{hazard_id}')
        geom = int(real_model.body_geomadr[body])
        added_geoms.append(geom)
        real_model.geom_type[geom] = mujoco.mjtGeom.mjGEOM_BOX
        real_model.geom_rgba[geom] = [1.0, 0.5, 0.0, 1.0]

    real_reset = jax.jit(real_env.reset)

    reset = jax.jit(nominal_env.reset)

    nominal_step = jax.jit(nominal_env.step)

    key = jax.random.PRNGKey(args.seed)

    font = ImageFont.load_default(size=18)

    records = []

    # --------------------------------------------------------
    # Rendering
    # --------------------------------------------------------

    with (
        mujoco.Renderer(
            nominal_env.sys.mj_model,
            height=480,
            width=640,
        ) as renderer,
        mujoco.Renderer(
            real_model,
            height=480,
            width=640,
        ) as real_renderer,
        ExitStack() as writers,
    ):

        for ep in range(args.num_episodes):

            placement_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        args.hazard_seed,
                        ep,
                    ]
                )
            )

            key, reset_key = jax.random.split(key)

            initial = reset(reset_key)

            # ------------------------------------------------
            # Step 1: Collect nominal rollout.
            # ------------------------------------------------

            nominal = [initial]
            keys = []

            for _ in range(args.episode_length):
                key, ak = jax.random.split(key)

                keys.append(ak)

                action, _ = policy(
                    nominal[-1].obs,
                    ak,
                )

                nominal.append(
                    nominal_step(
                        nominal[-1],
                        action,
                    )
                )

                if bool(nominal[-1].done):
                    break

            # Extend action randomness for the real rollout.
            branch_key = key

            while len(keys) < args.episode_length:
                branch_key, ak = jax.random.split(branch_key)
                keys.append(ak)

            # ------------------------------------------------
            # Step 2: Find largest feasible hazard.
            # ------------------------------------------------

            try:
                if args.num_added_hazards == 1:
                    position, radius = choose_hazard(base, nominal, placement_rng)
                    added_positions, added_radii = position[None], np.asarray([radius])
                else:
                    added_positions, added_radii = choose_hazard(
                        base, nominal, placement_rng, count=args.num_added_hazards)

            except ValueError as error:
                print(
                    f"Skipping episode {ep + 1}: {error}",
                    flush=True,
                )

                records.append(
                    dict(
                        episode=ep + 1,
                        status="skipped",
                        num_added_hazards=args.num_added_hazards,
                        reason=str(error),
                        seed=args.seed,
                        hazard_seed=args.hazard_seed,
                        nominal_steps=len(nominal) - 1,
                    )
                )

                (args.output_dir / "summary.json").write_text(
                    json.dumps(
                        records,
                        indent=2,
                    )
                )

                continue

            print(f'Episode {ep+1}: {args.num_added_hazards} added hazards; '
                  f'x={added_positions[:, 0].tolist()}, '
                  f'half-lengths={added_radii.tolist()}', flush=True)

            # ------------------------------------------------
            # Step 3: Nominal rollout metrics.
            # ------------------------------------------------

            rewards = np.r_[
                0.0,
                [float(s.reward) for s in nominal[1:]],
            ]

            costs = np.r_[
                0.0,
                [float(s.info["cost"]) for s in nominal[1:]],
            ]

            final = nominal[-1]

            nominal_metrics = dict(
                episode=ep + 1,
                length=len(nominal) - 1,
                cost=float(costs.sum()),
                end_reason=(
                    "unhealthy termination"
                    if bool(
                        final.info.get(
                            "unhealthy_termination",
                            False,
                        )
                    )
                    else (
                        "environment done" if bool(final.done) else "step limit reached"
                    )
                ),
                terminal_cost=float(
                    final.metrics.get(
                        "terminal_cost",
                        0.0,
                    )
                ),
            )

            nominal_output = args.output_dir / f"episode_{ep+1:02d}_nominal.mp4"

            # Original nominal video, without added marker.
            save_video(
                nominal_env,
                nominal,
                list(
                    zip(
                        rewards,
                        np.cumsum(rewards),
                    )
                ),
                list(
                    zip(
                        costs,
                        np.cumsum(costs),
                    )
                ),
                str(nominal_output),
                fps=args.video_fps,
                camera=args.camera,
                show_metrics=True,
                episode_metrics=[nominal_metrics],
            )

            # ------------------------------------------------
            # Step 4: Apply identical rectangle in REAL env.
            # ------------------------------------------------

            # Match all active rectangles to the blue nominal overlays.
            real._hazard_radii = jnp.concatenate([
                original_radii, jnp.zeros(args.num_added_hazards)])
            added_extents = np.column_stack([
                added_radii, np.full(args.num_added_hazards, platform_half_width)])
            real._hazard_half_extents = jnp.concatenate([
                original_extents, jnp.asarray(added_extents)], axis=0)
            for geom, half_length in zip(added_geoms, added_radii, strict=True):
                real_model.geom_size[geom, :3] = [
                    half_length, platform_half_width, real._hazard_height]
            positions = jnp.concatenate([
                initial.info['hazard_positions'], jnp.asarray(added_positions)])

            info = dict(
                initial.info,
                hazard_positions=positions,
                hazard_penetration=jnp.zeros(positions.shape[0]),
            )

            real_initial = real_reset(reset_key)

            # Verify matched initial physical states.
            np.testing.assert_allclose(
                real_initial.pipeline_state.q,
                initial.pipeline_state.q,
            )

            np.testing.assert_allclose(
                real_initial.pipeline_state.qd,
                initial.pipeline_state.qd,
            )

            ps = real_initial.pipeline_state

            ps = ps.replace(
                mocap_pos=ps.mocap_pos.at[jnp.array(real._hazard_mocap_ids)].set(
                    positions
                )
            )

            real_initial = real_initial.replace(
                pipeline_state=ps,
                info=info,
                obs=real._get_obs(
                    ps,
                    positions,
                ),
            )

            # Recompile with the current hazard dimensions.
            real_step = jax.jit(
                lambda s, a: real_env.step(
                    s,
                    a,
                )
            )

            # ------------------------------------------------
            # Step 5: Collect the real-hazard rollout.
            # ------------------------------------------------

            adapted = rollout(
                real_initial,
                keys,
                policy,
                real_step,
            )

            # Check whether the nominal trajectory would
            # penetrate the added real rectangle.
            probe = jax.jit(
                lambda ps: (
                    real._calculate_hazard_penetration(
                        ps,
                        positions,
                    )[-args.num_added_hazards:].max()
                )
            )

            nominal_penetration = max(float(probe(s.pipeline_state)) for s in nominal)

            assert nominal_penetration == 0.0, nominal_penetration

            trajectories = [
                nominal,
                adapted,
            ]

            # ------------------------------------------------
            # Step 6: Compute reward and cost metrics.
            # ------------------------------------------------

            cumulative = [
                np.r_[
                    0.0,
                    np.cumsum([float(s.info["cost"]) for s in t[1:]]),
                ]
                for t in trajectories
            ]

            added_cost = sum(
                float(jnp.sum(s.info["hazard_penetration"][-args.num_added_hazards:] ** 2)) * real._cost_scaler
                for s in adapted[1:]
            )

            record = dict(
                episode=ep + 1,
                num_added_hazards=args.num_added_hazards,
                positions=added_positions.tolist(),
                radii=added_radii.tolist(),
                half_extents=[
                    [float(r), platform_half_width, float(real._hazard_height)]
                    for r in added_radii
                ],
                seed=args.seed,
                hazard_seed=args.hazard_seed,
                nominal_video=str(nominal_output),
                nominal_steps=len(nominal) - 1,
                real_steps=len(adapted) - 1,
                nominal_reward=sum(float(s.reward) for s in nominal[1:]),
                real_reward=sum(float(s.reward) for s in adapted[1:]),
                nominal_cost=float(cumulative[0][-1]),
                real_cost=float(cumulative[1][-1]),
                nominal_added_hazard_cost=0.0,
                nominal_cost_without_added_hazard=float(cumulative[0][-1]),
                real_cost_without_added_hazard=(float(cumulative[1][-1]) - added_cost),
                nominal_cost_with_added_hazard=float(cumulative[0][-1]),
                real_cost_with_added_hazard=float(cumulative[1][-1]),
                hypothetical_nominal_penetration=(nominal_penetration),
                real_added_hazard_cost=added_cost,
            )

            reward_delta = record['nominal_reward'] - record['real_reward']
            cost_delta = record['nominal_cost'] - record['real_cost']
            winner = (
                'nominal' if reward_delta > 0 and cost_delta < 0
                else 'real' if reward_delta < 0 and cost_delta > 0
                else 'none'
            )
            record['winner'] = winner
            records.append(record)

            print(
                f"Episode {ep+1}: winner={winner} | "
                f"nominal reward={record['nominal_reward']:.6f}, "
                f"cost={record['nominal_cost']:.6f} | "
                f"real reward={record['real_reward']:.6f}, "
                f"cost={record['real_cost']:.6f}",
                flush=True,
            )

            print(
                json.dumps(record),
                flush=True,
            )

            # ------------------------------------------------
            # Step 7: Render side-by-side video.
            # ------------------------------------------------

            output = args.output_dir / f"episode_{ep+1:02d}_fake_vs_real.mp4"

            writer = writers.enter_context(
                iio.get_writer(
                    str(output),
                    fps=args.video_fps,
                )
            )

            for i in range(
                max(
                    map(
                        len,
                        trajectories,
                    )
                )
            ):
                panels = []

                for side, trajectory in enumerate(trajectories):
                    j = min(
                        i,
                        len(trajectory) - 1,
                    )

                    state = trajectory[j]

                    panel = Image.new(
                        "RGB",
                        (640, 600),
                        "#141820",
                    )

                    if side == 0:
                        # Nominal:
                        # BLUE rectangular visual-only hazard.
                        pixels = render(
                            renderer,
                            nominal_env.sys.mj_model,
                            state,
                            added_positions,
                            added_radii,
                            args.camera,
                            platform_half_width,
                        )
                    else:
                        # Real:
                        # ORANGE rectangular active hazard.
                        pixels = render(
                            real_renderer,
                            real_model,
                            state,
                            None,
                            added_radii,
                            args.camera,
                        )

                    panel.paste(
                        Image.fromarray(pixels),
                        (0, 120),
                    )

                    draw = ImageDraw.Draw(panel)

                    ended = j == len(trajectory) - 1

                    status = (
                        (
                            "UNHEALTHY"
                            if bool(
                                state.info.get(
                                    "unhealthy_termination",
                                    False,
                                )
                            )
                            else "STEP LIMIT"
                        )
                        if ended
                        else "RUNNING"
                    )

                    label = "VISUAL ONLY" if side == 0 else "REAL: OBSERVATIONS + COST"

                    draw.multiline_text(
                        (10, 8),
                        (
                            f"Episode {ep+1}: {label} ({args.num_added_hazards} hazards)\n"
                            f"Step {j}/{len(trajectory)-1} "
                            f"| {status}\n"
                            f"Cost: {cumulative[side][j]:.3f} "
                            f"| Terminal: "
                            f"{float(state.metrics.get('terminal_cost', 0)):.3f}"
                        ),
                        font=font,
                        fill="white",
                        spacing=8,
                    )

                    panels.append(np.asarray(panel))

                frame = np.concatenate(
                    panels,
                    axis=1,
                )

                writer.append_data(frame)

            # Keep final frame visible for two seconds.
            for _ in range(args.video_fps * 2):
                writer.append_data(frame)

            writer.close()

            print(
                f"Saved: {output}",
                flush=True,
            )

            (args.output_dir / "summary.json").write_text(
                json.dumps(
                    records,
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
