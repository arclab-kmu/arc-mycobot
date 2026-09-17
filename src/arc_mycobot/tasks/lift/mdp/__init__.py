"""MDP terms for the myCobot cube-lift task.

Aggregation only. The lift-specific terms -- ``object_ee_distance``,
``object_is_lifted``, ``object_goal_distance``,
``object_position_in_robot_root_frame`` -- are **imported** from Isaac Lab's own
``manipulation/lift`` package rather than vendored, for the same reason the
reach task imports its tracking rewards: they are the reference implementation
and a copy here would drift from the pinned ``isaaclab==2.3.2.post1``.

``position_command`` is reused from the reach task: this task's goal orientation
is pinned too, so feeding the policy the command's constant quaternion would add
four dead inputs.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.manipulation.lift.mdp import (  # noqa: F401
    object_ee_distance,
    object_goal_distance,
    object_is_lifted,
    object_position_in_robot_root_frame,
)

from arc_mycobot.tasks.reach.mdp.observations import position_command  # noqa: F401

from .rewards import grasping_object  # noqa: F401
