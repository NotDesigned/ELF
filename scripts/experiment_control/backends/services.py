"""Narrow dependency bundle injected into platform adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..runner import CommandResult


@dataclass(frozen=True)
class BackendServices:
    repo_root: Path
    script_dir: Path
    ssh_control_path: str
    run_command: Callable[..., CommandResult]
    remote_exec: Callable[..., CommandResult]
    local_run_dir: Callable[[dict[str, Any], dict[str, Any]], Path]
    backend_record: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    summarize_run: Callable[[Path], dict[str, Any]]
