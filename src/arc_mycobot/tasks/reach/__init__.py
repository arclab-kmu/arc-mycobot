"""End-effector position reach for the myCobot 280.

The robot-agnostic base lives in :mod:`.reach_env_cfg`; the myCobot's wiring and
Gym registration live in ``config/mycobot``. Add another arm by creating
``config/<robot>`` and appending it to the import below.
"""

from .config import mycobot  # noqa: F401
