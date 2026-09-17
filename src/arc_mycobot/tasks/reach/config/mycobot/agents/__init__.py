# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Agent configurations for the myCobot reach task.

Deliberately empty of imports. ``config/mycobot/__init__.py`` registers the Gym
environments using ``agents.__name__`` to build a *string* entry point, so the
package only has to exist -- and importing ``rsl_rl_ppo_cfg`` here would pull in
``isaaclab_rl`` and start the Omniverse Kit bootstrap for anything that merely
touches this package, including the CPU-only ``workspace_sweep`` tool and the
test suite.
"""
