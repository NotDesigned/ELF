"""Reusable experiment-control primitives.

The public ``scripts/experimentctl.py`` entry point uses these scheduler-neutral
pieces, which can be tested without an external backend or accelerator.
"""

from .runner import CommandResult, CommandRunner, SubprocessRunner
from .states import FailureClass

__all__ = [
    "CommandResult",
    "CommandRunner",
    "FailureClass",
    "SubprocessRunner",
]
