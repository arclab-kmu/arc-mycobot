# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0
#
# Derived from isaaclab_tasks/core/reach/config/franka/agents/rsl_rl_ppo_cfg.py in Isaac Lab 3.0.0 (BSD-3-Clause).

"""PPO runner configuration for the myCobot reach task.

The algorithm block is upstream reach's except ``entropy_coef`` (0.001, v1's value). The networks are ``[256, 128,
64]`` with observation normalization instead of upstream's ``[64, 64]`` without: the observation mixes joint angles
(order 1), joint velocities (order 0.1), and positions in metres (order 0.2). In v1 (4096 envs, seed 0) this reached
39 mm at iteration 121 against 44 mm at iteration 193 for upstream's networks; it was kept because it was no worse and
converged sooner. v1's ``empirical_normalization`` is ``obs_normalization`` on each model in rsl_rl 5.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class MyCobotReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    # v1's 1500-iteration run (PhysX, 4096 envs, seed 0) peaked at iteration 400 (2.9 mm mean error, 93.0% of steps on
    # target) and then drifted to 4.9 mm and 88.6% as the action noise std collapsed from 1.0 to 0.04. 500 is past that
    # peak; the last checkpoint is not necessarily the best one. The v2 Newton run with self-collision and the
    # self-contact penalty (seed 0) had not degraded by 499 (3.45 mm at 400, lowest 2.38 mm at 493, 2.61 mm and 96.1%
    # at 499), so the budget is v1's and not yet re-measured here.
    max_iterations = 500
    save_interval = 50
    experiment_name = "reach_mycobot_280_jn"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
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
