# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Elephant Robotics myCobot 280 JetsonNano articulation configuration.

Ported to the Isaac Lab 3.0 API from v1 (``assets/robots/mycobot_280.py`` and ``mycobot_urdf.py`` at tag ``v1.0.0``).
The packaged URDFs and meshes under ``assets/data/mycobot_280_jn`` are v1's repaired vendor files (meshes BSD-2-Clause,
Elephant Robotics; see ``meshes/LICENSE``). The API changes are the backend-specific ``PhysxRigidBodyCfg`` /
``PhysxArticulationCfg`` / ``NewtonArticulationCfg`` schemas, ``collision_type`` in place of ``collider_type``, and
``joint_effort_limit`` / ``joint_velocity_limit`` in place of ``effort_limit_sim`` / ``velocity_limit_sim``.

The vendor names its *links* ``joint1`` .. ``joint6``; joints are ``jointN_to_jointM``. The names are passed through so
the URDF, ``pymycobot``, and the vendor MoveIt config agree.

.. warning::
    The actuator numbers are **not** vendor-published (``TODO(unverified)``). Elephant Robotics quotes a 250 g payload,
    a 280 mm working radius, and +-0.5 mm repeatability, but no torque limit or servo gains. They are modelling choices
    for a nominal 1.20 kg arm, gripper, and camera assembly, good enough for simulation and not for sim-to-real.
"""

import contextlib

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from . import ARC_MYCOBOT_ASSETS_DIR


def _register_physx_schemas() -> None:
    """Register the PhysX USD schemas that the kit-less URDF importer applies (e.g. ``PhysxJointAPI``).

    USD ignores schema plugins registered after the first stage exists, and the environment creates its stage before
    spawning the robot, so this runs while the task config loads. When ``ovphysx`` is installed its full codeless
    schemas are registered first, because they collide with the importer's ``physxSchemaFallback`` types (same fix as
    arc-humanoid v2 ``f8a343b``).
    """
    try:
        import ovphysx  # pyright: ignore[reportMissingImports]  (optional ``ovphysx`` extra)
    except ImportError:
        pass
    else:
        from pxr import Plug

        Plug.Registry().RegisterPlugins([str(path) for path in ovphysx.codeless_schema_paths()])
    with contextlib.suppress(ImportError):
        import isaacsim.asset.importer.utils.impl.physx_types  # noqa: F401


_register_physx_schemas()

MYCOBOT_280_JN_DIR = ARC_MYCOBOT_ASSETS_DIR / "mycobot_280_jn"
MYCOBOT_280_JN_URDF_PATH = MYCOBOT_280_JN_DIR / "urdf" / "mycobot_280_jn_camera_gripper.urdf"
"""Camera flange and fixed (welded) gripper; used by reach."""
MYCOBOT_280_JN_PARALLEL_URDF_PATH = MYCOBOT_280_JN_DIR / "urdf" / "mycobot_280_jn_camera_gripper_parallel.urdf"
"""Camera flange and sliding parallel-jaw gripper; used by lift."""

MYCOBOT_ARM_JOINTS = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]
"""The six actuated revolute joints in base-to-flange order.

This is the observation and action layout of the trained policies and the order ``pymycobot.get_angles()`` returns.
Listed rather than matched by regex so the order is stated here, not left to the backend's joint sort.
"""

MYCOBOT_ROOT_LINK = "joint1"
"""The articulation root, and the frame goal positions are expressed in."""

MYCOBOT_FLANGE_BODY = "joint6_flange"
"""The tool flange; the reach task tracks its origin (not a TCP)."""

MYCOBOT_HOME_POSE = {
    "joint2_to_joint1": 0.0,
    "joint3_to_joint2": -0.5,
    "joint4_to_joint3": -1.0,
    "joint5_to_joint4": -0.6,
    "joint6_to_joint5": 0.0,
    "joint6output_to_joint6": 0.0,
}
"""Reach reset posture [rad]: elbow bent, flange at ``(0.189, -0.065, 0.184)`` m, inside the reach goal box."""

MYCOBOT_ARM_VELOCITY_LIMIT = 2.0944
"""Per-joint velocity limit [rad/s]: 120 deg/s from the spec sheet. ``TODO(unverified)`` per joint."""

# -- actuator model (TODO(unverified)) -----------------------------------------------------------------------------
# Stiffness tapers base to wrist; gravity is on. In v1 (PhysX) holding the home posture sagged at most 32.8 mrad on
# joint3_to_joint2, 8.8 mm at the flange. The action commands offsets from home, so a constant sag shifts the
# operating point rather than the settled error.
_ARM_STIFFNESS = {
    "joint2_to_joint1": 40.0,
    "joint3_to_joint2": 40.0,
    "joint4_to_joint3": 30.0,
    "joint5_to_joint4": 20.0,
    "joint6_to_joint5": 15.0,
    "joint6output_to_joint6": 10.0,
}
_ARM_DAMPING = {name: round(value / 10.0, 1) for name, value in _ARM_STIFFNESS.items()}
_ARM_EFFORT_LIMIT = 3.0
"""Torque ceiling per joint [N·m], about 3x the shoulder's gravity load; the URDF's ``effort="1000"`` is a dummy."""

MYCOBOT_280_JN_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(MYCOBOT_280_JN_URDF_PATH),
        # The URDF's only root is joint1 (URDF importer 3.0 dropped `root_link_name`).
        fix_base=True,
        # Keep the flange and camera_link frames as separate bodies.
        merge_fixed_joints=False,
        # The URDF carries convex-hull collision STLs; do not rebuild collision from the decorative visuals.
        collision_from_visuals=False,
        collision_type="Convex Hull",
        # Self-collision on (v1 had it off): without it a reach policy folded the gripper into the base and lower arm
        # for 24% of goals, up to 41 mm deep. Links welded by fixed joints, and links one joint apart, are not checked;
        # no checked pair touches at home. The same flag is set for PhysX and Newton in `articulation_props`. Contacts
        # are soft, so a task still has to penalize self-contact (see reach's `self_contact` reward term).
        self_collision=True,
        # Every link has an inertial block; four reference frames carry 0.1 g each.
        link_density=0.0,
        # PhysX needs this for contact sensors (the reach self-contact penalty); Newton ignores it.
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            # Gains come from the actuator below, not the URDF.
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
            NewtonArticulationCfg(self_collision_enabled=True),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=MYCOBOT_HOME_POSE, joint_vel={".*": 0.0}),
    # The vendor's ranges are already inside the mechanical ones, and v1's goal box was verified reachable against them.
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=MYCOBOT_ARM_JOINTS,
            joint_effort_limit=_ARM_EFFORT_LIMIT,
            joint_velocity_limit=MYCOBOT_ARM_VELOCITY_LIMIT,
            stiffness=_ARM_STIFFNESS,
            damping=_ARM_DAMPING,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""myCobot 280 JN with the camera flange and fixed gripper, fixed to the world at its base."""
