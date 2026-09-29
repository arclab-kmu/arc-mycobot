"""Evaluate the camera-box alignment task with zero action or a PPO checkpoint.

Run with the Isaac Lab Python environment. The same seed and reset call make
baseline and policy evaluations use the same randomized target/joint starts.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Visual-Align-MyCobot280JN-v0")
parser.add_argument("--checkpoint", type=str, default=None, help="RSL-RL checkpoint; omit for zero-action baseline")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--seed", type=int, default=20260929)
parser.add_argument("--image", type=str, default=None, help="Optional raw camera RGB PNG path")
args = parser.parse_args()
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import json

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import arc_mycobot.tasks  # noqa: F401
from arc_mycobot.tasks.visual_align import mdp


def main() -> None:
    torch.manual_seed(args.seed)
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)

    policy = None
    policy_model = None
    if args.checkpoint:
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device="cuda:0")
        policy_model = runner.alg.policy

    # Runner initialization consumes random numbers; reseed immediately before
    # the explicit reset so baseline and checkpoint see the same initial scene.
    torch.manual_seed(args.seed)
    obs, _ = wrapped.reset()
    box_func = env.unwrapped.cfg.observations.policy.box.func
    if box_func is mdp.RandomizedHandBox or isinstance(box_func, mdp.RandomizedHandBox):
        # Score the actual YOLO detection, not the box corruption seen by PPO.
        box_func = mdp.detected_hand_box
    initial = box_func(env.unwrapped).cpu()
    target = env.unwrapped.scene["target"]
    target_yz_history = [target.data.root_pos_w[:, 1:3].clone().cpu()]
    history = []
    for _ in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs) if policy else torch.zeros((args.num_envs, wrapped.num_actions), device="cuda:0")
            obs, _, dones, _ = wrapped.step(actions)
            if policy_model is not None:
                policy_model.reset(dones)
            history.append(box_func(env.unwrapped).cpu())
            target_yz_history.append(target.data.root_pos_w[:, 1:3].clone().cpu())

    boxes = torch.stack(history)
    target_yz = torch.stack(target_yz_history)
    target_yz_path = torch.linalg.vector_norm(target_yz[1:] - target_yz[:-1], dim=2).sum(dim=0)
    robot_masses = env.unwrapped.scene["robot"].root_physx_view.get_masses().sum(dim=1)
    visible = boxes[:, :, 4] > 0
    error = torch.linalg.vector_norm(boxes[:, :, :2], dim=2)
    late_visible = visible[-20:]
    late_error = error[-20:]
    metrics = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "initial_box_sum": initial.sum(dim=0).tolist(),
        "detected_fraction": float(visible.float().mean()),
        "centered_fraction": float((visible & (error < 0.1)).float().mean()),
        "late_centered_fraction": float((late_visible & (late_error < 0.1)).float().mean()),
        "late_mean_error_visible": float((late_error * late_visible).sum() / late_visible.sum().clamp(min=1)),
        "mean_target_yz_path_m": float(target_yz_path.mean()),
        "mean_target_yz_displacement_m": float(torch.linalg.vector_norm(target_yz[-1] - target_yz[0], dim=1).mean()),
        "robot_mass_range_kg": [float(robot_masses.min()), float(robot_masses.max())],
    }
    print("METRICS " + json.dumps(metrics), flush=True)

    if args.image:
        from PIL import Image

        rgb = env.unwrapped.scene["camera"].data.output["rgb"][0].cpu().numpy()
        Image.fromarray(rgb).save(args.image)

    wrapped.close()


def cli() -> int:
    from arc_mycobot.scripts._entrypoint import run_and_exit

    run_and_exit(main, simulation_app)


if __name__ == "__main__":
    cli()
