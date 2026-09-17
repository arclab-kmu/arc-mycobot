# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0
#
# Derived from isaaclab_tasks/.../manipulation/reach/config/franka/agents/rsl_rl_ppo_cfg.py

"""PPO runner configuration for the myCobot reach task.

The ``algorithm`` block is Isaac Lab's Franka-reach configuration **verbatim** --
same clip, epochs, mini-batches, learning rate, adaptive KL schedule, entropy
coefficient and discounting. This task is a standard position reach on a small
arm, not a new algorithmic problem, so those are left alone.

Four things differ, and all four are in the runner or the network rather than
the algorithm:

* ``experiment_name`` and ``max_iterations`` -- naming and budget;
* ``actor_hidden_dims`` / ``critic_hidden_dims`` are ``[256, 128, 64]``, not
  upstream reach's ``[64, 64]``. Those are the networks Isaac Lab's
  deployment-oriented reach uses;
* ``empirical_normalization = True``, which upstream reach leaves off. This
  task's observation mixes joint angles (order 1), joint velocities (order 0.1)
  and positions in metres (order 0.2), and that spread is what the normalizer is
  for.

The last two were **compared** rather than assumed, because a first training run
improved and then regressed and both were suspects. At 4096 envs, seed 0, the
configuration above reached 39 mm at iteration 121; upstream's ``[64, 64]``
without normalization reached 44 mm at iteration 193. Neither fixed the
regression, because neither was causing it -- ``JOINT_ACTION_SCALE`` was. See
that constant's docstring. The larger network is kept because it was no worse
and converged sooner.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class MyCobotReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # Measured, not inherited. A full 1500-iteration run (4096 envs, seed 0,
    # 1004 s) peaks at **iteration 400** -- 2.9 mm mean tracking error, 93.0% of
    # control steps on target -- and then slowly *degrades* to 4.9 mm and 88.6%
    # by iteration 1499.
    #
    # The cause is visible in `Mean action noise std`, which collapses from 1.0
    # to 0.04: past roughly iteration 450 the policy is effectively
    # deterministic, stops exploring, and drifts. Training longer does not help
    # and mildly hurts.
    #
    # So: 500, which is past the peak with margin. save_interval is 50, so the
    # best checkpoint is captured either way -- and note that the *last*
    # checkpoint is not the best one, which matters because `play` loads the
    # latest by default. Pass --checkpoint <abs path> to pick another.
    max_iterations = 500
    save_interval = 50
    experiment_name = "reach_mycobot_280_jn"
    # See the module docstring: on here, off in upstream's reach, and measured
    # rather than assumed. Baked into the exported policy, so deployment feeds
    # raw observations.
    empirical_normalization = True
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # Isaac Lab's *deployment* reach networks, not its textbook reach's
        # [64, 64]. See the module docstring for the comparison.
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    # Isaac Lab's Franka-reach algorithm block, verbatim.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
