# AGENTS.md

Guidance for coding agents in this repository. `CLAUDE.md` is a symlink to this file.

## Project

`arc-mycobot` v2: an Isaac Lab **3.0.0** external project (import package `arc_mycobot`) for the myCobot 280
JetsonNano. The layout is exactly the Isaac Lab 3.0.0 template generator's external project (`isaaclab --new`,
blank content, wheel mode). v2 is being rebuilt task by task: reach, then cube lift, then wrist-camera visual alignment
(YOLO hand), then the Jetson `deploy/` runtime. Until a task lands here, it exists only in v1.

The Isaac Lab 2.3.2 implementation (v1) is frozen at tag `v1.0.0` on `main`; its checkout and runtime data (`logs/`,
`outputs/`, `dist/`) live in the sibling folder `../arc-mycobot-v1`, a linked worktree of this repository's `.git`.
Read v1 for task design, geometry, assets, and reference numbers, but do not copy v1 code wholesale: Isaac Lab 3
changed quaternion order, data accessors, writers, camera configs, rsl_rl configs, checkpoints, and the CLI.

## Layout

```text
src/arc_mycobot/
├── __init__.py                 # passive: never import tasks or isaaclab here
├── assets/                     # ARC_MYCOBOT_ASSETS_DIR -> assets/data (URDF, meshes, USD)
└── tasks/
    └── __init__.py             # import_packages(): imports every task package, skips `utils` and `.mdp`
```

- Tasks are discovered through `[project.entry-points."isaaclab.tasks"]` → `arc_mycobot.tasks`.
- A task family is `tasks/<task>/{<task>_env_cfg.py, mdp/, config/<robot>/{__init__.py, *_env_cfg.py, agents/}}`.
  Terms shared by several task families go in `tasks/mdp/` (skipped by the task importer).
- `mdp/` packages export through `lazy_export()` and an `__init__.pyi` stub; when you add a term, add it to `__all__`
  and the import list in the stub.
- Task IDs use the `ARC-` prefix and no `-v0` suffix (for example `ARC-Reach-MyCobot280JN`); registrations keep string
  entry points and set `"default_agent": "rsl_rl"`. Play-time changes go in the env cfg's `play_mode()` method, not in
  separate `-Play` IDs or `*_PLAY` classes.
- Do not add packages the 3.0.0 template does not generate until real code needs them.
- Use the shared `isaaclab` CLI (`train`, `play`, `zero_agent`, `random_agent`, `list_envs`); do not copy launchers into
  `scripts/`.
- Large data (checkpoints, logs, datasets) stays out of Git.

## Environment

- Python `>=3.12,<3.13`, managed by `uv`; build backend `uv_build`. Commit `uv.lock`.
- Dependency: `isaaclab[rsl-rl,importers]==3.0.0` from PyPI (`importers` = kit-less URDF → USD). The default `uv sync`
  installs the kit-less Newton backend without Isaac Sim. Other features are forwarded extras, e.g.
  `uv run --extra isaacsim ...`, `--extra ovphysx`, `--extra video`. `uv sync` removes extras you do not pass again.
- `[tool.uv]` mirrors Isaac Lab v3.0.0's `override-dependencies`, `environments`, indexes, and sources (copied from
  `arc-humanoid` v2). Without the overrides `uv lock` fails (usd-exchange 2.3.0 vs 3.0.1). Update them together with the
  Isaac Lab pin.

## Commands

```bash
uv sync
uv run pytest                                   # tests/test_registration.py
uv run ruff check --fix . && uv run ruff format .
uv run pre-commit run --all-files               # ruff, hygiene hooks, codespell
uv run pyright
uv run isaaclab list_envs --keyword ARC
uv build
```

- Ruff ignores B008 because Isaac Lab MDP terms use `asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")` defaults.
- Commit messages use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`; `!` for breaking changes).

## Porting a v1 task

1. Diff against the matching upstream task in the installed wheel (`.venv/lib/python3.12/site-packages/isaaclab_tasks/
   core/{reach,lift}`) and port v1's robot-specific choices onto it.
2. Gate each task before starting the next: registration test and `isaaclab list_envs`; `isaaclab zero_agent --task ...
   --num_envs 16 --max_steps 200` with observation/action sizes equal to v1 and joint names, order, and limits equal to
   the URDF; `isaaclab train --rl_library rsl_rl --num_envs 512 --max_iterations 3` smoke; then a full run compared with
   v1's numbers (reach v1: 2.9 mm, 93.0% on-target at iteration 400). A short smoke run does not show a task learns.
3. `deploy/` (Jetson, Python 3.8) must stay 3.8-parseable: exclude it from the py312 ruff target or give it its own.
   rsl_rl 5.5.1 checkpoints store `actor_state_dict`/`critic_state_dict`, not v1's `model_state_dict` with `actor.`
   keys; `isaaclab play` exports `exported/policy.pt` and `policy.onnx`. Redefine the deploy contract before reuse.

## Isaac Lab 3 conventions to respect

- Quaternions are `XYZW` (2.x was `WXYZ`); never infer the order from array width. Recheck every camera mount,
  calibration, and observation that v1 wrote as `wxyz`.
- Asset and sensor data return `ProxyArray`; convert explicitly with `.torch` or `.warp`.
- Use `write_*_to_sim_index` / `write_*_to_sim_mask` writers; actuators live on `robot.actuators`.
  `effort_limit_sim`/`velocity_limit_sim` became `joint_effort_limit`/`joint_velocity_limit`.
- Physics-specific schemas live in `isaaclab_physx` / `isaaclab_newton`; the physics backend is a Hydra preset
  (`physics=newton_mjwarp` default, `ovphysx`, `isaacsim_physx`).
- `TiledCameraCfg` is deprecated: use `CameraCfg` with a `renderer_cfg`; moving cameras need
  `update_latest_camera_pose=True`. There is no `--enable_cameras` or `--headless`; windows open only with `--viz`.
- The kit-less URDF importer nests bodies under `Robot/Geometry/<root>/...` and merges fixed-joint links into their
  parent, so sensor and FrameTransformer prim paths written as `Robot/<link>` may break. Its fallback PhysX schema
  plugin must be registered before the first USD stage (import `isaacsim.asset.importer.utils.impl.physx_types` at the
  top of the robot cfg module).
