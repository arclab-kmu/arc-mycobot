# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure how often a trained reach policy puts the arm through itself.

Rolls the policy out for one episode in ``--num_envs`` environments, records the joint positions every ``--stride``
steps, rebuilds the link poses with the URDF forward kinematics (``mycobot_collision.py``, which matches the simulator's
body poses to about 1 micrometre), and reports the overlap of the collision-checked link hulls. It also reports the
tracking error at the end of each goal window and, on the Newton MuJoCo Warp backend, the contact count per world.

    uv run python scripts/reach_self_collision_check.py --checkpoint logs/rsl_rl/reach_mycobot_280_jn/<run>/model_499.pt
"""

from __future__ import annotations

import argparse
import sys
from typing import cast

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import arc_mycobot.tasks  # noqa: F401
from arc_mycobot.tasks.reach.config.mycobot.joint_pos_env_cfg import MyCobotReachEnvCfg

parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
parser.add_argument("--task", default="ARC-Reach-MyCobot280JN")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--stride", type=int, default=3, help="Record every N env steps.")
parser.add_argument("--threshold", type=float, default=0.002, help="Overlap depth counted as a collision [m].")
add_launcher_args(parser)
parser.set_defaults(device=None, visualizer=[])
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args

task_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
env_cfg = cast(MyCobotReachEnvCfg, task_cfg)
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.observations.policy.enable_corruption = False
env_cfg.seed = agent_cfg.seed
args_cli.device = env_cfg.sim.device

from mycobot_collision import ARM_JOINTS, FLANGE_LINK, MyCobotGeometry  # noqa: E402

from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)

agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())


def mujoco_contacts_per_world(num_envs: int) -> np.ndarray | None:
    """MuJoCo Warp contacts per world ``[2, num_envs]``: generated (within margin) and penetrating; None elsewhere."""
    try:
        from isaaclab_newton.physics import NewtonManager

        data = NewtonManager._solver.mjw_data  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    except (ImportError, AttributeError):
        return None
    count = int(data.nacon.numpy()[0])
    world = data.contact.worldid.numpy()[:count]
    penetrating = data.contact.dist.numpy()[:count] < 0
    return np.stack([np.bincount(world, minlength=num_envs), np.bincount(world[penetrating], minlength=num_envs)])


with launch_simulation(env_cfg, args_cli):
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = cast(ManagerBasedRLEnv, env.unwrapped)
    vec_env = RslRlVecEnvWrapper(cast(ManagerBasedRLEnv, env), clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(vec_env, agent_cfg)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=unwrapped.device)
    robot = unwrapped.scene["robot"]
    assert list(robot.joint_names) == ARM_JOINTS, robot.joint_names
    window = round(env_cfg.commands.ee_pose.resampling_time_range[0] / unwrapped.step_dt)
    episode = round(env_cfg.episode_length_s / unwrapped.step_dt)
    num_steps = episode - 5  # stop before the time-out reset

    steps, joint_pos, goals, contacts = [], [], [], []
    obs = vec_env.get_observations()
    with torch.inference_mode():
        for k in range(num_steps):
            obs, _, dones, _ = vec_env.step(policy(obs))
            assert not dones.any(), f"unexpected reset at step {k}"
            if k % args_cli.stride == 0 or k == num_steps - 1:
                steps.append(k)
                joint_pos.append(robot.data.joint_pos.torch.clone())
                goals.append(unwrapped.command_manager.get_command("ee_pose")[:, :3].clone())
                per_world = mujoco_contacts_per_world(args_cli.num_envs)
                if per_world is not None:
                    contacts.append(per_world)
    nconmax = getattr(getattr(env_cfg.sim.physics, "solver_cfg", None), "nconmax", None)
    env.close()

steps = np.array(steps)
q = torch.stack(joint_pos)  # [T, N, 6]
goal = torch.stack(goals)  # [T, N, 3], root frame
geometry = MyCobotGeometry(device=str(q.device))
poses, _ = geometry.forward_kinematics(q.reshape(-1, 6))
depth = geometry.pair_depths(poses).reshape(len(geometry.pairs), *q.shape[:2]).cpu().numpy()  # [P, T, N]
error = (poses[FLANGE_LINK][:, :3, 3].reshape(*q.shape[:2], 3) - goal).norm(dim=-1).cpu().numpy()  # [T, N]

threshold = args_cli.threshold
worst = depth.max(0)
settled = [int(np.flatnonzero(steps <= end - 3)[-1]) for end in range(window, num_steps + window, window)]
settled_depth = worst[settled]
print(f"\ncheckpoint: {args_cli.checkpoint}")
print(
    f"{args_cli.num_envs} envs x {num_steps} steps, recorded every {args_cli.stride} steps; {len(geometry.pairs)} pairs"
)
print(f"overlap > {1e3 * threshold:.0f} mm: {100 * (worst > threshold).mean():.2f}% of recorded samples")
print(
    f"settled goals (last sample of each {window}-step window, {settled_depth.size} goals): "
    f"{100 * (settled_depth > threshold).mean():.2f}% overlap, max depth {1e3 * settled_depth.max():.1f} mm"
)
print(f"episodes with an overlap at any recorded step: {100 * (worst > threshold).any(0).mean():.2f}%")
print(f"max depth over all samples: {1e3 * worst.max():.1f} mm")
settled_error = error[settled]
print(
    f"settled tracking error: mean {1e3 * settled_error.mean():.2f} mm, "
    f"within 20 mm {100 * (settled_error <= 0.02).mean():.1f}%"
)
if contacts:
    generated, penetrating = np.stack(contacts).transpose(1, 0, 2)  # [T, N] each
    print(
        f"MuJoCo contacts: penetrating in {100 * (penetrating > 0).mean():.2f}% of samples; generated (incl. margin) "
        f"max {generated.max()} per world"
        + (f", at the nconmax cap ({nconmax}) in {100 * (generated >= nconmax).mean():.3f}%" if nconmax else "")
    )
order = np.argsort(-(depth > threshold).mean((1, 2)))
for k in order[:10]:
    share = 100 * (depth[k] > threshold).mean()
    if share == 0:
        break
    a, b = geometry.pairs[k]
    print(f"  {a:>14} x {b:<14} {share:5.2f}% of samples, max {1e3 * depth[k].max():.1f} mm")
