# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the environments registered by arc_mycobot.

The script iterates over the registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

By default only this project's environments (task ids containing ``MyCobot``) are shown;
pass ``--keyword`` to filter by any other substring (default: 'MyCobot').
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="List the environments registered by arc_mycobot.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments (default: 'MyCobot').")
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import arc_mycobot.tasks  # noqa: F401


def main():
    """Print all environments registered by the arc_mycobot extension."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Environments registered by arc_mycobot"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # default to this project's tasks; --keyword overrides the filter
    keyword = args_cli.keyword if args_cli.keyword is not None else "MyCobot"
    # count of environments
    index = 0
    # acquire all matching environment names
    for task_spec in gym.registry.values():
        if keyword not in task_spec.id:
            continue
        # add details to table
        env_cfg = task_spec.kwargs.get("env_cfg_entry_point", "")
        table.add_row([index + 1, task_spec.id, task_spec.entry_point, env_cfg])
        # increment count
        index += 1

    print(table)


def cli() -> int:
    """Console-script entry point; see :mod:`arc_mycobot.scripts._entrypoint`."""
    from arc_mycobot.scripts._entrypoint import run_and_exit

    run_and_exit(main, simulation_app)


if __name__ == "__main__":
    cli()
