# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared shutdown handling for every script that launches Isaac Sim.

Isaac Sim makes three things awkward for a script that has to report a result,
and all three bit this project before this module existed:

1. ``isaacsim`` replaces :func:`sys.exit` with ``app.post_quit(status)``, which
   rejects ``None`` -- so ``sys.exit(main())`` on a function returning ``None``
   dies with a ``TypeError`` after the work is already done.
2. ``simulation_app.close()`` can terminate the process itself, **always with
   status 0**. Calling it on an error path turns a crash into a reported
   success, which is exactly what a pre-flight gate must never do.
3. That same hard exit discards a buffered ``stdout``. When stdout is a pipe
   rather than a terminal -- a log file, a CI capture -- the script's entire
   output disappears while the exit code still says everything is fine.

So: flush first, shut the simulator down only when the work succeeded, and force
the real status code out with :func:`os._exit` either way.
"""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable
from typing import NoReturn

__all__ = ["run_and_exit"]


def run_and_exit(main: Callable[[], int | None], simulation_app: object) -> NoReturn:
    """Run ``main``, then terminate the process with a truthful exit code.

    Args:
        main: The script body. A returned ``int`` is used as the exit status;
            ``None`` counts as success.
        simulation_app: The object returned by ``AppLauncher(...).app``.
    """
    status = 1
    try:
        status = main() or 0
    except SystemExit as exit_request:  # an explicit exit from inside main()
        status = exit_request.code if isinstance(exit_request.code, int) else 0
    except BaseException:
        traceback.print_exc()
        status = 1

    sys.stdout.flush()
    sys.stderr.flush()

    if status == 0:
        # Only shut the simulator down on the happy path: close() may exit the
        # process, and it would do so with status 0.
        try:
            simulation_app.close()  # type: ignore[attr-defined]
        except BaseException:
            traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(status)
