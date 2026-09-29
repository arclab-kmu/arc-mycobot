# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Gym registration must not need a GPU.

``config/mycobot/__init__.py`` registers with *string* entry points precisely so
that importing the package costs nothing. If someone replaces one with a real
class object, this file starts an Omniverse Kit bootstrap and these tests turn
from milliseconds into minutes -- which is the signal.
"""

from __future__ import annotations

import gymnasium as gym
import pytest

import arc_mycobot.tasks  # noqa: F401  (registers the environments)

TASKS = [
    "Isaac-Reach-MyCobot280JN-v0",
    "Isaac-Reach-MyCobot280JN-Play-v0",
    "Isaac-Visual-Align-MyCobot280JN-v0",
    "Isaac-Visual-Align-MyCobot280JN-Play-v0",
    "Isaac-Visual-Align-YOLO-Hand-MyCobot280JN-v0",
    "Isaac-Visual-Align-YOLO-Hand-MyCobot280JN-Play-v0",
]


@pytest.mark.parametrize("task", TASKS)
def test_task_is_registered(task):
    assert task in gym.registry


@pytest.mark.parametrize("task", TASKS)
def test_entry_points_are_strings(task):
    """Strings, not imported objects: see the module docstring."""
    kwargs = gym.registry[task].kwargs
    for key in ("env_cfg_entry_point", "rsl_rl_cfg_entry_point"):
        assert isinstance(kwargs[key], str), f"{task}.{key} is not a lazy string entry point"


def test_importing_tasks_does_not_import_isaaclab():
    """The whole point of the lazy registration. Also guards the CPU-only tools."""
    import sys

    assert "isaaclab" not in sys.modules
    assert "isaacsim" not in sys.modules
