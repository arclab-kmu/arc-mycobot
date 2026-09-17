"""MDP terms for the myCobot position-reach task.

A thin aggregation layer. Three sources, in order of preference:

* the generic framework terms (``joint_pos_rel``, ``joint_vel_rel``,
  ``last_action``, ``action_rate_l2``, ``joint_vel_l2``, ``time_out``,
  ``reset_joints_by_scale``, ``UniformPoseCommandCfg``,
  ``JointPositionActionCfg`` ...) come from ``isaaclab.envs.mdp``;

* the two position-tracking rewards come from Isaac Lab's own
  ``manipulation/reach`` package. They are **imported, not vendored**: they are
  the reference implementation of this exact reward, and a copy here would
  silently drift from the version this project pins
  (``isaaclab==2.3.2.post1``). ``~/IsaacLab`` is only ever read, never modified.

  ``orientation_command_error`` is deliberately *not* imported. This task tracks
  position only -- see ``RewardsCfg`` in
  :mod:`arc_mycobot.tasks.reach.reach_env_cfg` for the reasoning and for the
  one-line change that turns orientation tracking back on.

* the success indicator and the position-only goal observation are local, in
  :mod:`.rewards` and :mod:`.observations`. The framework has no equivalent of
  either.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.manipulation.reach.mdp.rewards import (  # noqa: F401
    position_command_error,
    position_command_error_tanh,
)

from .observations import position_command  # noqa: F401
from .rewards import position_command_success  # noqa: F401
