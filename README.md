# arc-mycobot

<p align="center">
  <a href="https://docs.isaacsim.omniverse.nvidia.com/latest/index.html"><img src="https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg" alt="Isaac Sim 5.1.0"/></a>
  <a href="https://isaac-sim.github.io/IsaacLab/"><img src="https://img.shields.io/badge/IsaacLab-2.3.2-silver.svg" alt="Isaac Lab 2.3.2"/></a>
  <a href="https://docs.python.org/3/whatsnew/3.11.html"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11"/></a>
  <a href="https://releases.ubuntu.com/"><img src="https://img.shields.io/badge/platform-linux--64-orange.svg" alt="Linux platform"/></a>
</p>

End-effector **position reach** for the Elephant Robotics **myCobot 280
JetsonNano** with its adaptive gripper, trained in Isaac Lab with PPO.

The robot comes from the vendor's own `mycobot_ros2` description. That file
cannot be imported as it stands — it is not well-formed XML, every joint
declares a zero velocity limit, no link has any inertia, and the gripper is a
five-joint `<mimic>` cluster. Repairing it is a real part of this repository and
it happens in one documented place: **`assets/robots/mycobot_urdf.py`**. The
vendor checkout is never modified.

## Two tasks

| task id | what it does | gripper |
| --- | --- | --- |
| `Isaac-Reach-MyCobot280JN-v0` | end-effector **position reach** | welded open |
| `Isaac-Lift-Cube-MyCobot280JN-v0` | **pick up a 32 mm cube** and carry it to a goal | actuated |

The two share the arm, the URDF repair and the kinematics; they differ in what
the gripper is allowed to do, which turns out to change almost everything about
the physics setup. See [The lift task](#the-lift-task).

## Scope — reach

Position reach in simulation. The policy sees its own joint state and a goal
**position**, and emits joint-position offsets. No grasping, no contact, no
orientation tracking, no hardware.

The gripper is **welded open** and merged into the flange: it contributes its
mass and geometry and nothing else. See `GRIPPER_FIXED_AT` for why that beats
keeping five mimic joints alive for a task that never closes them.

## Quick start

```bash
# 1. the vendor robot description, as a sibling checkout
git clone --depth 1 https://github.com/elephantrobotics/mycobot_ros2.git ../mycobot_ros2
#    (or point $MYCOBOT_ROS2_DIR at an existing one)

# 2. the environment — Isaac Sim 5.1 and Isaac Lab 2.3.2 come in as pip deps
uv sync

# 3. Isaac Sim asks to accept the NVIDIA Omniverse licence on first launch, and
#    the prompt has no stdin under uv. Accept it once, deliberately:
export OMNI_KIT_ACCEPT_EULA=YES

# 4. check the robot and the task before spending a GPU-hour on them
uv run pytest                   # ~0.3 s, no GPU: URDF repair + task geometry
uv run workspace_sweep          # ~10 s, no GPU: is every goal actually reachable?
uv run list_envs                # registered task ids

# 5. see the environment run before training it. Pass --max_steps, or these
#    loop until you interrupt them.
uv run zero_agent   --task Isaac-Reach-MyCobot280JN-v0 --num_envs 16 --headless --max_steps 200
uv run random_agent --task Isaac-Reach-MyCobot280JN-v0 --num_envs 16 --headless --max_steps 200

# 6. train
uv run train --task Isaac-Reach-MyCobot280JN-v0 --headless

# 7. watch a checkpoint
uv run play --task Isaac-Reach-MyCobot280JN-Play-v0 --num_envs 16
```

`uv run train` writes to `logs/rsl_rl/reach_mycobot_280_jn/<timestamp>/`, saving
a checkpoint every 50 iterations and dumping the resolved env and agent configs
into `params/`. `uv run play` picks up the latest run by default; `--load_run`
and `--checkpoint` select another.

A smoke run before the real thing:

```bash
uv run train --task Isaac-Reach-MyCobot280JN-v0 --headless --num_envs 64 --max_iterations 5
```

Watch **`Episode_Reward/end_effector_position_success`** — the fraction of
control steps spent within `SUCCESS_THRESHOLD` (20 mm) of the goal. It is the
one number that says whether the policy is working.

### Troubleshooting

**Setup appears to hang for minutes with no output.** That is the renderer, not
the URDF conversion — converting this robot takes about 1.6 s. Isaac Sim's first
RTX pipeline compile ran past 16 minutes on this machine, and
`SimulationContext.step()` renders by default even under `--headless`. In your
own physics-only scripts, step with `sim.step(render=False)`. The scripts here
already do the right thing.

## The robot

| | |
| --- | --- |
| source | `mycobot_ros2/mycobot_description/urdf/mycobot_280_jn/mycobot_280_jn_adaptive_gripper.urdf` |
| DOF | 6 revolute, `joint2_to_joint1` … `joint6output_to_joint6` |
| root link | `joint1` — fixed base, and the frame goals are expressed in |
| tracked body | `joint6_flange` — the tool flange, not a TCP |
| mass | 0.968 kg = 0.85 kg arm + 0.118 kg gripper. PhysX reports `joint6_flange` as **0.138 kg**, because the welded gripper merges into it (0.020 + 0.118). |
| reach | 302 mm horizontal at the flange; `z` ∈ [−103, +447] mm |

The vendor names its **links** `joint1` … `joint6`. That is confusing and it is
theirs — renaming would put this repository and `mycobot_ros2` into permanent
disagreement. Links are `jointN`, joints are `jointN_to_jointM`.

### What the URDF repair fixes

All four were confirmed against the checkout, not assumed:

| defect | effect if unrepaired | fix |
| --- | --- | --- |
| `lower = "-2.932"1 upper = …` | not well-formed XML; every parser stops at line 89 | text-level repair before the parse |
| `velocity="0"` on all 13 joints | imports as a PhysX joint that cannot move | 2.0944 rad/s (120 °/s, spec sheet) |
| no `<inertial>` anywhere; no `<collision>` on the 6 arm links | zero-mass articulation | explicit inertials + `collision_from_visuals` |
| 5 `<mimic>` joints on the gripper | only one of the five binds (see below) | welded (reach) / re-targeted to adjacent pairs (lift) |

Plus an `<?xml version="1.1"?>` declaration, a stray `<xacro:property>`, and
`package://` mesh URIs that urdfdom resolves only under ROS. The repaired file
lands in `generated/` (git-ignored — it is derived, not authored) and is rebuilt
whenever the vendor file is newer.

## The task

`Isaac-Reach-MyCobot280JN-v0`, a manager-based `ManagerBasedRLEnv`.

| | |
| --- | --- |
| observation | 21 = 6 joint pos + 6 joint vel + 3 goal position + 6 last action |
| action | 6 joint-position offsets from the home posture, scale 0.5 rad |
| control | 60 Hz policy over 120 Hz physics (`decimation=2`) |
| episode | 12 s, goal resampled every 3 s → four goals per episode |
| goal box | 160 × 240 × 160 mm centred at (0.16, 0, 0.20) m in `joint1` |
| reward | −0.2·‖e‖ + 0.1·(1 − tanh(‖e‖/0.05)) + 0.05·[‖e‖ ≤ 0.02], minus action-rate and joint-velocity penalties |
| termination | time-out only |
| success | flange within 20 mm of the goal |

**Position only.** Upstream's orientation-tracking reward is deliberately
absent: on a 6-DOF arm with a 280 mm reach, a position the arm reaches easily
can be one it reaches in only a single wrist configuration, so a pose reward
would spend most of its gradient on goals that are position-feasible and
pose-infeasible. `RewardsCfg` documents the three changes needed to turn it
back on — and why the third one is not optional.

**The goal box is derived, not guessed.** `uv run workspace_sweep` re-derives
it: it loads the same repaired URDF Isaac Lab converts, runs bounded
least-squares IK against the declared joint limits, and checks the eight corners
explicitly as well as the interior. Corners matter — an earlier box passed 300
uniformly-sampled goals while one of its corners was 9.3 mm out of reach, because
uniform sampling essentially never visits the corner of a 3-box.

## Results

Measured on an RTX 5090, 4096 environments, seed 0. A full 1500-iteration run
(1004 s) to find where the task actually converges:

| iteration | mean tracking error | control steps within 20 mm |
| --- | --- | --- |
| 150 | 5.0 mm | 92.4% |
| **400** | **2.9 mm** | **93.0%** |
| 750 | 6.8 mm | 88.0% |
| 1499 | 4.9 mm | 88.6% |

**It peaks at iteration 400 and then degrades.** `Mean action noise std`
collapses from 1.0 to 0.04 over the run: past roughly iteration 450 the policy
is effectively deterministic, stops exploring and drifts. `max_iterations` is
therefore set to **500**, not the 1500 that was first configured — training
longer costs 15 minutes and makes the policy slightly worse.

Two consequences worth knowing:

* **The last checkpoint is not the best one.** `uv run play` loads the latest by
  default; pass `--checkpoint <absolute path to model_400.pt>` to load another.
  Note `--checkpoint` takes a *file path*, not a run-relative name.
* At 500 iterations the run takes about 5½ minutes.

`Episode_Reward/end_effector_position_success` divided by that term's 0.05
weight reads directly as the fraction of control steps on target — 0.0465 means
93%. The 7% that miss are mostly the moments just after a goal resamples, while
the arm is still traversing.

**The one number that mattered.** `JOINT_ACTION_SCALE` started at 0.15, reasoned
down from upstream's 0.5 because this arm is a third the size of a UR10. That
reasoning is wrong: the action is in *joint* space, and joint excursion does not
shrink with arm size — only Cartesian distance does. At 0.15 the goal box
demanded actions of 9.8σ at the 95th percentile, and training improved to 39 mm
and then *regressed* to 56 mm as exploration noise decayed. At 0.5 it converges
monotonically to 3 mm. The constant's docstring carries the derivation.

Success is a **reward term, not a termination**. The goal resamples inside the
episode, so the task is to track, not to arrive; and a non-time-out termination
on a task whose dominant term is a negative distance penalty would make ending
the episode early a reward in itself.

## The lift task

`Isaac-Lift-Cube-MyCobot280JN-v0`. The policy sees its joint state, the cube's
position and a goal position, and emits six joint-position offsets plus one
binary open/close.

| | |
| --- | --- |
| observation | 27 = 7 joint pos + 7 joint vel + 3 cube pos + 3 goal pos + 7 last action |
| action | 6 joint offsets (scale 0.5) + 1 binary gripper |
| control | 50 Hz policy over 100 Hz physics |
| episode | 5 s, goal resampled every 5 s |
| cube | 32 mm, 20 g, friction 1.5 / 1.2 |
| jaw | opens to 50.2 mm, closes to 24.0 mm |
| lifted | cube centre above 60 mm (it rests at 17 mm) |

### Getting the gripper to grip

This took four measured fixes, and the order they were found in is the useful
part — each one looked like the whole problem until it wasn't.

**1. The mimic cluster.** The vendor gripper is a four-bar per side:
`gripper_base → {knuckle, parallel link}` and `knuckle → fingertip`, with the
fingertip coupled at ×−1.0 so the pad stays parallel as the knuckle swings.

PhysX **does** have mimic joints (`PhysxMimicJointAPI`), and Isaac Lab's
`convert_mimic_joints_to_normal_joints=True` is what makes the importer create
them — the name is misleading, so it was checked rather than trusted. But
articulation mimic joints only bind **parent-child adjacent** joints. Importing
the vendor's five tags verbatim produced five mimic prims of which exactly *one*
had its reference bound; the other four named `gripper_controller` from a sibling
branch and silently did nothing, and the right finger then swung to 1.10 rad
against a 0.7 rad limit.

`gripper="mimic"` re-expresses the couplings against adjacent joints: each
fingertip mimics **its own knuckle**, and the two knuckles are commanded
(mirrored) by one binary action. The pads — the parts that touch the object — are
therefore held parallel by a real constraint. The two dangling parallel links are
the open end of a loop URDF cannot express, so they stay welded.

**2. Top-down grasping is impossible.** The pads hang about 15 mm below the jaw
centre, so centring the jaw on a cube resting on a surface drives them through
it. Measured, not guessed. The task uses a **side grasp** instead — approach
horizontal, pads in a vertical plane, measured span z ∈ [5.5, 27.8] mm for a
35 mm cube, entirely above the ground.

**3. The vendor collision meshes cannot close on anything.** `gripper_left1`
spans 59 mm and `gripper_right1` 13 mm, and their convex hulls sit at very
different distances from the centreline — 13.6 mm on the left, 37.9 mm on the
right. A cube placed at the jaw centre is struck by one pad and missed by the
other. The URDF repair replaces both with **symmetric box pads** (6 × 24 × 20 mm,
inset 16 mm from the link origin) and strips every other gripper collider.

**4. The arm's own hulls were fighting the floor.** With
`collision_from_visuals=True` the arm carries convex hulls off its decorative
visual meshes, and at the low poses a grasp needs they press into the ground:
the arm held **0.19 rad** of joint error while merely standing still, which is
~30 mm at the jaw — larger than the cube. Turning it off drops that to
**0.005 rad**. The lift config therefore has exactly two colliders on the whole
robot: the grip pads.

With all four in place, a scripted grasp lifts the cube ~98 mm from every
approach offset between 5 and 25 mm.

The cost of (4) is worth stating plainly: **the arm's links can pass through the
ground plane and through each other.** Nothing in the task rewards that, and
nothing forbids it.

### Lift results

4096 environments, seed 0, 1500 iterations in 483 s (8 min):

| iteration | cube lifted | cube at goal | mean reward |
| --- | --- | --- | --- |
| 250 | 81.3% | 61.7% | 129 |
| 500 | 90.9% | 74.5% | 153 |
| 1000 | 93.0% | 81.0% | 163 |
| 1499 | **93.5%** | **84.2%** | 158 |

Read those as fractions of control steps: the cube spends 94% of the time above
60 mm and 84% of it near the goal. The curve does not turn over, so the last
checkpoint is usable.

**What it took.** The first working gripper was a fiction — the cube was being
wedged into the housing, not pinched — and fixing that honestly made the task
*unlearnable* for four successive runs. Each run isolated one real defect:

| fix | evidence | result |
| --- | --- | --- |
| PhysX mimic, retargeted to adjacent joints | 4 of the vendor's 5 `<mimic>` tags imported with an empty reference; the right finger swung to 1.10 rad against a 0.7 rad limit | couplings exact to 0.6% |
| grasp point moved out of the housing | origin midpoint sits 16.8 mm from a body that reaches 13.9 mm — a 32 mm cube overlapped it by 13 mm | cube pinched, not wedged |
| lift-specific home pose | side grasp needed **6.3σ** of action from the reach home | jaw reached 4–7 mm, from a 45 mm plateau |
| `grasping_object` reward | probing the policy: gripper command positive — open — at *every* one of 2000 samples; it never closed, so never saw a lift | jaw closes 75% of steps |
| **sliding fingers** | pads swept **15.2 mm forward** as the knuckles rotated, shoving the cube 14–21 mm; invariant to pad placement | 0% → 94% |

The last one is the whole story. A rotating linkage keeps its pads *parallel* but
not *stationary*, and the arc drags whatever it closes on. Sliding fingers have
no arc. It also removed the size limit: scripted grasps hold 20, 22, 25, 28, 32,
36 and 40 mm cubes, where the rotating jaw only worked from 34 to 42 mm.

## Layout

```
src/arc_mycobot/
├── assets/robots/
│   ├── mycobot_urdf.py        # the vendor-URDF repair + every structural constant
│   └── mycobot_280.py         # MYCOBOT_280_JN_CFG: spawn, actuators, gains
├── kinematics/urdf_fk.py      # numpy FK off the repaired URDF; no Isaac import
├── tasks/reach/
│   ├── reach_env_cfg.py       # robot-agnostic base (scene, MDP, 60 Hz)
│   ├── mdp/                   # framework terms + success indicator + goal obs
│   └── config/mycobot/
│       ├── geometry.py        # goal box and length scales — Isaac-free on purpose
│       ├── joint_pos_env_cfg.py
│       └── agents/rsl_rl_ppo_cfg.py
└── scripts/
    ├── rsl_rl/{train,play}.py
    ├── environments/{zero_agent,random_agent,list_envs}.py
    └── tools/workspace_sweep.py
```

Two boundaries are load-bearing:

* **Gym registration is lazy.** `config/mycobot/__init__.py` registers *string*
  entry points and `agents/__init__.py` imports nothing, so `import
  arc_mycobot.tasks` costs milliseconds and needs no GPU.
  `tests/test_task_registration.py` asserts that `isaaclab` stays out of
  `sys.modules`.
* **The geometric constants avoid Isaac Lab.** `geometry.py` and
  `mycobot_urdf.py` import no `isaaclab`, which is what lets `workspace_sweep`
  and the whole test suite run on CPU in under ten seconds instead of paying an
  Omniverse Kit bootstrap to read six floats.

## What is not verified

Marked `TODO(unverified)` in place, and worth knowing before trusting any number
that comes out of this:

* **Link masses and inertias.** Elephant Robotics publish a total mass (850 g),
  a payload (250 g) and a working radius; they publish no per-link breakdown and
  no inertia tensors. The masses here are distributed by the usual taper for a
  serial arm and normalized to the published total; the inertias are isotropic
  solid-sphere approximations at a 30 mm radius of gyration. Right order of
  magnitude, right total — not good enough for torque-level work.
* **Joint stiffness, damping and torque limits.** No servo gains are published.
  The values here were chosen, not measured — what *was* measured is their
  effect: holding the home posture, the worst steady-state gravity sag is
  32.8 mrad at the shoulder, which puts the flange 8.8 mm below where forward
  kinematics says it should be. That 8.8 mm is the whole discrepancy between
  this repository's FK and its simulation.
* **Joint velocity limit.** 120 °/s is the spec sheet's single figure for the
  whole arm, applied to all six joints.

None of this blocks a simulation-only reach task — the policy learns against
whatever dynamics it is given. All of it would have to be measured before any
sim-to-real transfer.
