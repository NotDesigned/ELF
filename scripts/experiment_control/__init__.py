"""Reusable experiment-control primitives.

The public ``scripts/experimentctl.py`` entry point remains backward compatible;
this package contains scheduler-neutral pieces that can be tested without SSH,
Slurm, SCO, Docker, or a GPU.
"""

from .runner import CommandResult, CommandRunner, SubprocessRunner
from .states import FailureClass, normalize_sensecore_state, normalize_slurm_state

__all__ = [
    "CommandResult",
    "CommandRunner",
    "FailureClass",
    "SubprocessRunner",
    "normalize_sensecore_state",
    "normalize_slurm_state",
]
