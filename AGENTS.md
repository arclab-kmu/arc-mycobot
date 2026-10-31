# arc-mycobot agent guide

Isaac Lab 2.3.2 / Isaac Sim 5.1 project for myCobot 280 JetsonNano reach, cube lift, and wrist-camera visual alignment (including YOLO hand detection). `src/arc_mycobot/` contains simulation assets, tasks, vision, and CLI entry points; `deploy/` is a separate Jetson Nano real-arm runtime. Read `README.md` for task assumptions and `deploy/README.md` before touching hardware.

## Environment

- Root project: Python 3.11 only (`pyproject.toml`, `uv.lock`); `uv sync` installs Isaac Lab/Sim and dev tools into `.venv`. For YOLO tasks use `uv sync --extra yolo`.
- Before launching Isaac Sim via `uv run list_envs`, `train`, `play`, or an agent script, explicitly accept its license with `export OMNI_KIT_ACCEPT_EULA=YES` (first-run stdin prompt does not work under uv). Simulation/rendering and camera tasks need a compatible GPU; CPU checks below do not.
- Jetson deployment is different: JetPack 4.6 / Python 3.8 with its JetPack-compatible Torch/Torchvision, then `python -m pip install -r deploy/requirements.txt` in that same environment. Do not run root `uv sync` on the Jetson merely to run `deploy/`. `deploy` source is kept Python 3.8-parseable (`tests/test_deploy.py`).

## Build and checks (repo root)

```bash
uv sync
uv build --offline
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync workspace_sweep
```

- `uv run --no-sync pytest` is the CPU suite (`tests/test_*.py`); vendor-source URDF tests skip if the sibling `mycobot_ros2` checkout is absent. `workspace_sweep` uses CPU-only bounded IK to check reach-goal corners and samples; rerun after changing reach geometry, joint limits, or packaged URDF.
- Ruff uses 120 columns, Python 3.11, and E/W/F/I/UP/SIM/RET (`pyproject.toml`); `E402` is exempt because simulator entry points must launch the app before importing Isaac modules. Pyright is configured in `pyproject.toml` but currently reports 18 existing errors in `tests/test_deploy.py` (fake arm/config types and one float assignment); don't mistake that baseline for a new failure.
- For a GPU smoke run, the README uses `uv run zero_agent --task Isaac-Reach-MyCobot280JN-v0 --num_envs 16 --headless --max_steps 200` and `uv run train --task Isaac-Reach-MyCobot280JN-v0 --headless --num_envs 64 --max_iterations 5`. Visual-align training needs `--enable_cameras`; YOLO tasks also need `--extra yolo`.

## Code boundaries

- Task implementations live in `src/arc_mycobot/tasks/{reach,lift,visual_align}/`: base env config, `mdp/` terms, and `config/mycobot/` robot-specific config/agent. CLI entry points are declared in `[project.scripts]` in `pyproject.toml`.
- Keep `config/mycobot/__init__.py` Gym registration entry points as strings: importing `arc_mycobot.tasks` must not import Isaac Lab/Sim (`tests/test_task_registration.py`). Reach/lift `geometry.py`, `assets/robots/mycobot_urdf.py`, and `kinematics/urdf_fk.py` remain Isaac-free so CPU tests and `workspace_sweep` don't bootstrap Kit.
- Isaac scripts launch `AppLauncher` before importing the simulator-dependent modules and use `scripts/_entrypoint.py:run_and_exit` to flush output and preserve failure exit codes; a plain `simulation_app.close()` on an error can exit with status 0.
- Runtime URDFs and relative-path STL meshes are packaged under `src/arc_mycobot/assets/robots/`; reach uses the fixed gripper, lift the parallel-jaw variant. Vendor recovery via `build_urdf` is optional and writes machine-specific absolute mesh paths to the XDG cache, not the runtime assets. Keep tests for joint order, limits, mass, and jaw geometry aligned with any asset change.

## Pitfalls

- `logs/`, `outputs/`, `.venv/`, and `dist/` are ignored local checkpoints/runs/builds; don't treat local checkpoints as committed inputs. `play` defaults to the latest run, not necessarily the best checkpoint; `--checkpoint` takes a file path.
- Isaac's first RTX pipeline compilation may take many minutes even with `--headless`; physics-only probes should use `sim.step(render=False)` (the repo's scripts already do).
- `deploy.run` defaults to read-only preview. `--execute` sends physical joint commands only after safety gates; don't use it for a software smoke test. On Jetson `/dev/ttyTHS1` must not be shared with another robot process; read `deploy/README.md` for pose, freshness, stop, and camera requirements first.
