"""Articulation configuration for the myCobot 280 JetsonNano with adaptive gripper.

The articulation is built by converting the *repaired* vendor URDF on the fly --
see :mod:`.mycobot_urdf` for the four defects that repair fixes and why the
vendor file cannot be imported as it stands. What comes out:

* root link ``joint1``, fixed to the world (this is a desktop arm bolted down);
* six actuated revolute joints, :data:`MYCOBOT_ARM_JOINTS`, in base-to-flange
  order -- which is also the order the Elephant Robotics ``pymycobot`` API's
  ``get_angles()`` returns;
* ``joint6_flange`` -- the tool flange, and the body the reach task tracks;
* the gripper, welded and merged into the flange. It contributes its mass and
  its geometry and nothing else.

.. note::
    The vendor names its *links* ``joint1`` .. ``joint6``. That is confusing and
    it is theirs, not ours: renaming them here would put this file and the
    ``mycobot_ros2`` checkout into permanent disagreement, and any cross-check
    against the vendor's MoveIt config or ``pymycobot`` would have to translate.
    So the names are passed through. Links are ``jointN``; joints are
    ``jointN_to_jointM``.

.. warning::
    The actuator numbers below are **not** vendor-published. Elephant Robotics
    quotes a payload (250 g), a working radius (280 mm) and a repeatability
    (+-0.5 mm) for this arm, but no torque limit and no servo gains. Every value
    in the two blocks marked ``TODO(unverified)`` is therefore a modelling
    choice, chosen to be physically sane for a 0.97 kg arm rather than measured.
    They are good enough for a simulation-only reach task -- the policy learns
    against whatever dynamics it is given -- and they are *not* good enough for
    sim-to-real transfer.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .mycobot_urdf import (
    ARM_JOINTS,
    ARM_VELOCITY_LIMIT,
    FLANGE_BODY,
    HOME_POSE,
    PARALLEL_FINGER_JOINTS,
    PARALLEL_OPEN,
    ROOT_LINK,
    build_mycobot_urdf,
)

__all__ = [
    "MYCOBOT_280_JN_CFG",
    "MYCOBOT_280_JN_LIFT_CFG",
    "MYCOBOT_FINGER_BODIES",
    "MYCOBOT_GRIPPER_COMMAND_JOINTS",
    "MYCOBOT_GRIPPER_JOINTS",
    "MYCOBOT_ARM_JOINTS",
    "MYCOBOT_FLANGE_BODY",
    "MYCOBOT_HOME_POSE",
    "MYCOBOT_LIFT_HOME_POSE",
    "MYCOBOT_ROOT_LINK",
    "MYCOBOT_URDF_PATH",
]

MYCOBOT_URDF_PATH = str(build_mycobot_urdf())
"""Absolute path to the repaired URDF. Generated on import; see :mod:`.mycobot_urdf`."""

# The structural constants -- joint names, root link, flange, home posture --
# live in mycobot_urdf.py, not here. They are statements about the URDF's
# contents, and keeping them in the Isaac-free module is what lets the CPU-only
# workspace sweep and the test suite read them without starting Omniverse. They
# are re-exported under MYCOBOT_* names so that a reader of a task config sees
# one consistent prefix.
MYCOBOT_ROOT_LINK = ROOT_LINK
MYCOBOT_ARM_JOINTS = list(ARM_JOINTS)
MYCOBOT_FLANGE_BODY = FLANGE_BODY
MYCOBOT_HOME_POSE = HOME_POSE
MYCOBOT_GRIPPER_JOINTS = list(PARALLEL_FINGER_JOINTS)
"""Every gripper DOF. With the parallel jaw these are the two sliding fingers."""

MYCOBOT_GRIPPER_COMMAND_JOINTS = list(PARALLEL_FINGER_JOINTS)
"""The two joints the action term drives: the sliding fingers, mirrored.

Same list as :data:`MYCOBOT_GRIPPER_JOINTS` now -- with a parallel jaw there are
no followers to couple, which is precisely what makes it robust. The rotating
linkage needed a mimic constraint per side and still swept its pads 15 mm
forward; see ``PARALLEL_FINGER_JOINTS`` in :mod:`.mycobot_urdf`."""
MYCOBOT_FINGER_BODIES = ["gripper_left1", "gripper_right1"]
"""The two fingertip pads -- the bodies that actually touch a grasped object."""


# -- actuator model -----------------------------------------------------------
# TODO(unverified): no torque limit or servo gain is published for this arm.
#
# Stiffness tapers base to wrist in the usual way for a serial arm: the shoulder
# holds the most mass at the longest lever, the wrist holds almost nothing. The
# sizing constraint is gravity, which is *enabled* for this task.
#
# Measured, holding the home posture for 600 physics steps: the worst
# steady-state sag is 32.8 mrad (1.9 deg) on `joint3_to_joint2`, which puts the
# flange 8.8 mm below where forward kinematics says the commanded angles should
# put it -- FK predicts (0.1888, -0.0646, 0.1839), PhysX settles at
# (0.1880, -0.0646, 0.1751). That is the entire discrepancy between this
# repository's FK and its simulation, and it is gravity, not a frame error.
#
# 8.8 mm is a large fraction of the task's 20 mm success threshold, but it is
# not a tracking bias: the action term commands offsets from the home posture
# and the policy learns whatever offset reaches the goal, so a constant sag
# shifts the operating point rather than the settled error. Raise these gains if
# a trained policy's error turns out to have a constant vertical component.
_ARM_STIFFNESS = {
    "joint2_to_joint1": 40.0,
    "joint3_to_joint2": 40.0,
    "joint4_to_joint3": 30.0,
    "joint5_to_joint4": 20.0,
    "joint6_to_joint5": 15.0,
    "joint6output_to_joint6": 10.0,
}
# stiffness/10, the usual starting ratio for Isaac Lab implicit position
# actuators (Isaac Lab's own arm configs sit between stiffness/4 and /18).
_ARM_DAMPING = {name: round(value / 10.0, 1) for name, value in _ARM_STIFFNESS.items()}

_EFFORT_LIMIT = 3.0
"""Torque ceiling per joint [Nm]. ``TODO(unverified)``.

The URDF's own ``effort="1000"`` is a placeholder and an absurd one for an arm
rated to lift 250 g; passing it through would mean the sim has no torque ceiling
at all. 3.0 Nm is about 3x the shoulder's gravity load, which leaves the PD gains
rather than the ceiling shaping the response while still bounding it.
"""

MYCOBOT_280_JN_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=MYCOBOT_URDF_PATH,
        fix_base=True,
        root_link_name=MYCOBOT_ROOT_LINK,
        # True, unlike the NERO config in the sibling repository, and for a
        # reason specific to this robot: the gripper is welded (see
        # mycobot_urdf.GRIPPER_FIXED_AT), so merging folds its seven links into
        # `joint6_flange` and the articulation drops from 14 rigid bodies to 7.
        # At 4096 environments that halves the body count for geometry that
        # cannot move relative to the flange anyway. The two names this
        # repository depends on -- `joint1` and `joint6_flange` -- are both on
        # the parent side of every merge, so both survive.
        merge_fixed_joints=True,
        # The vendor gives the six arm links no <collision> at all -- only the
        # seven gripper links have any -- so the arm's collision geometry can
        # only come from its visual meshes, which are heavy: 140k faces across
        # the six links, 77k of them in `joint1_jet.dae` alone.
        #
        # Measured rather than assumed, because the size of those meshes makes
        # it look expensive: converting with this flag on takes 1.6 s, and with
        # it off, also 1.6 s. It is not a cost worth trading anything for.
        #
        # (If a run appears to hang during setup, this is not the cause. The
        # renderer is: `SimulationContext.step()` renders by default, and the
        # first RTX pipeline compile ran past 16 minutes on this machine. Pass
        # `render=False` when stepping a physics-only check.)
        #
        # Convex hulls rather than decomposition: this task has nothing to
        # collide with -- no object, no obstacle, self-collision off, and the
        # ground plane 1.05 m below the mounting plane -- so the geometry is
        # inert here and a closer fit would buy nothing. It is generated anyway
        # so that the arm is a physical object the day the scene gains one.
        collision_from_visuals=True,
        collider_type="convex_hull",
        self_collision=False,
        activate_contact_sensors=False,
        # <inertial> is written for every link by the URDF repair, so this is a
        # fallback that should never fire. Left non-zero deliberately: if a
        # future vendor link slips through without one, a light body is far
        # easier to debug than a zero-mass PhysX assertion.
        link_density=100.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Gravity ON. This is a simulation-only task, the arm is light, and
            # the actuators are sized to hold it, so there is no reason to hide
            # the load the way a sim-to-real config might.
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            # None on both: take whatever the URDF declares at conversion time
            # and let the ImplicitActuatorCfg below set the gains that matter.
            # Keeping the authority in one place means the actuator block is the
            # only thing to edit when the gains are eventually measured.
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=MYCOBOT_HOME_POSE,
        joint_vel={".*": 0.0},
    ),
    # 1.0: the soft limits equal the URDF's hard limits. The vendor's declared
    # ranges are already inside the mechanical ones, and the goal box was
    # verified reachable against exactly these numbers
    # (scripts/tools/workspace_sweep.py), so shrinking them here would
    # invalidate that check.
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=MYCOBOT_ARM_JOINTS,
            effort_limit_sim=_EFFORT_LIMIT,
            # 120 deg/s, from the spec sheet, written into the URDF over the
            # vendor's velocity="0". Taken from the same constant the repair
            # uses rather than repeated, so the two cannot disagree.
            velocity_limit_sim=ARM_VELOCITY_LIMIT,
            stiffness=_ARM_STIFFNESS,
            damping=_ARM_DAMPING,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""myCobot 280 JetsonNano with adaptive gripper, converted from the repaired URDF."""


# -----------------------------------------------------------------------------
# Lift variant: the same arm with a gripper that can close.
# -----------------------------------------------------------------------------

MYCOBOT_LIFT_HOME_POSE = {
    "joint2_to_joint1": 0.7787,
    "joint3_to_joint2": -0.4468,
    "joint4_to_joint3": -2.5084,
    "joint5_to_joint4": -0.1863,
    "joint6_to_joint5": -2.3629,
    "joint6output_to_joint6": 3.1000,
}
"""Reset posture for the lift task [rad]. **Not** the reach task's home.

The reach home is elbow-bent with the wrist at zero. Reaching a side grasp from
there needs the wrist to travel -2.36 and +3.14 rad, and with
``JointPositionAction(scale=0.5, use_default_offset=True)`` the policy would have
to output **6.3 sigma** on a unit-Gaussian action to get there. It never did:
trained from the reach home, the jaw plateaued about 45 mm short of the cube for
1500 iterations and the lift rate stuck near 10%.

This pose is the *same IK branch* as the grasp, lifted 70 mm: solved at the
grasp and then stepped up in 10 mm increments so the solution never jumps
branches. From here the excursion to the grasp is 0.93 rad -- **1.86 sigma** --
and to the middle of the goal region 0.55 rad, 1.10 sigma.

Starting the arm near its work is standard (Isaac Lab's Franka lift does the
same) and it is not what makes the task easy: the cube spawn, the goal and the
grasp itself are all still randomized.

The wrist roll is 3.10 rather than the IK's 3.1416, which is exactly the joint's
limit -- sitting on a limit would clip half the action range on that joint. The
2.4 deg of roll it gives up is immaterial to a symmetric jaw.
"""

# The lift task drives the arm to poses the reach task never visits: jaw at the
# ground plane, arm near full extension, holding against contact. Measured there,
# the reach gains leave 0.10-0.20 rad of steady-state joint error -- about 30 mm
# at the jaw, which is larger than the cube. Two causes, both fixed here:
#
#  * `_EFFORT_LIMIT` of 3.0 Nm is marginal. The arm is 0.97 kg and at this
#    extension the shoulder carries roughly 0.97 * 9.81 * 0.25 ~= 2.4 Nm of pure
#    gravity, before any contact or dynamics. 10 Nm gives real headroom.
#  * the gains are three times the reach values, because here a joint error is
#    not a tracking statistic the policy can learn around -- it decides whether
#    the jaw closes on the cube or 30 mm above it.
#
# TODO(unverified), like every other actuator number in this file: these are
# sized for a stable sim grasp, not measured on hardware. They are further from
# the real servo than the reach values, not closer.
_LIFT_EFFORT_LIMIT = 10.0
_LIFT_ARM_STIFFNESS = {name: value * 3.0 for name, value in _ARM_STIFFNESS.items()}
_LIFT_ARM_DAMPING = {name: round(value / 10.0, 2) for name, value in _LIFT_ARM_STIFFNESS.items()}

_GRIPPER_STIFFNESS = 2000.0
_GRIPPER_DAMPING = 100.0
_GRIPPER_EFFORT = 5.0
"""Gripper drive gains, in **linear** units now that the fingers slide: N/m and N.

Grasping with a position-controlled jaw works by *commanding past* the object:
the finger stops on the cube, the drive keeps pulling toward the closed target,
and the residual angle times the stiffness is the grip force. The stiffness is
therefore a responsiveness knob and the **effort limit is the grip force**.

Unlike every other actuator number in this file, the force has a published figure
behind it: Elephant Robotics rate the adaptive gripper at a **150 g** maximum
grip, about 1.47 N at the pad. 5 N is a few times that -- headroom for a stable
sim contact rather than a fidelity claim.

A 20 g cube needs 0.196 N held by friction; at 1.5 static friction the pads only
have to press with 0.13 N, so even the published figure has a 10x margin.
"""

MYCOBOT_280_JN_LIFT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(build_mycobot_urdf("parallel")),
        # The flag's name is misleading and it was checked rather than trusted:
        # True makes the importer parse <mimic> into PhysxMimicJointAPI prims
        # (measured: 2 created, both with their reference bound), False leaves
        # the joints free. It does *not* mean "flatten the coupling away".
        convert_mimic_joints_to_normal_joints=True,
        fix_base=True,
        root_link_name=MYCOBOT_ROOT_LINK,
        # False, unlike the reach config. Merging would fold the fingertip links
        # into their knuckles, and the fingertips are exactly the bodies that
        # have to exist as separate colliders for a grasp to happen at all.
        merge_fixed_joints=False,
        # False, unlike the reach config, and this is the single most important
        # line in this file for whether a grasp works.
        #
        # With it True the arm's collision geometry is convex hulls taken from
        # decorative visual meshes. Two things went wrong, both measured: the
        # hulls of the forearm and gripper body fight the ground plane at the
        # low poses a grasp needs, leaving ~0.19 rad of joint error with the arm
        # merely holding still; and the fingertip hulls are so asymmetric that
        # the jaw closes past a centred cube on one side and short of it on the
        # other (see _PAD_INSET in mycobot_urdf.py).
        #
        # Off, the only colliders on the robot are the two box grip pads the
        # URDF repair writes onto the fingertips. That is the whole contact
        # model, and it is the right one for this task: a pick-and-lift needs
        # accurate pads and nothing else.
        #
        # The cost is real and worth stating: the arm's links can pass through
        # the ground plane and through each other. Nothing in this task rewards
        # that, but nothing forbids it either.
        collision_from_visuals=False,
        # Off for the same reason it is off in the reach config: with only two
        # colliders on the whole robot there is nothing meaningful to
        # self-collide, and leaving it on was measured to cost ~0.5 rad of
        # tracking error through spurious contacts.
        self_collision=False,
        activate_contact_sensors=False,
        link_density=100.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # Higher than the reach config's 8. Contact-rich tasks need the
            # extra position iterations or the grasp jitters and drops.
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={**MYCOBOT_LIFT_HOME_POSE, **PARALLEL_OPEN},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=MYCOBOT_ARM_JOINTS,
            effort_limit_sim=_LIFT_EFFORT_LIMIT,
            velocity_limit_sim=ARM_VELOCITY_LIMIT,
            stiffness=_LIFT_ARM_STIFFNESS,
            damping=_LIFT_ARM_DAMPING,
            friction=0.0,
            armature=0.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=MYCOBOT_GRIPPER_JOINTS,
            effort_limit_sim=_GRIPPER_EFFORT,
            velocity_limit_sim=0.2,
            stiffness=_GRIPPER_STIFFNESS,
            damping=_GRIPPER_DAMPING,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""myCobot 280 JN with a gripper that can close, for the lift task.

Differs from :data:`MYCOBOT_280_JN_CFG` in five places, all of them consequences
of the task now involving contact: the actuated URDF, fixed joints kept
unmerged so the fingertips stay separate colliders, self-collision on, more
solver position iterations, and a gripper actuator.
"""
