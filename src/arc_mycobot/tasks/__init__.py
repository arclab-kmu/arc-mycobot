"""Task implementations for arc_mycobot.

Importing this package registers every Gym environment. Registration is
CPU-safe by design: the ``config/<robot>/__init__.py`` modules call
``gym.register`` with *string* entry points, so nothing from ``isaaclab`` is
imported until an environment is actually constructed.
"""

from . import (
    lift,  # noqa: F401
    reach,  # noqa: F401
    visual_align,  # noqa: F401
)
