"""Backend contract and registry used by the controller dispatch loop."""

from __future__ import annotations

from typing import Any, Protocol


class Backend(Protocol):
    kind: str

    def stage(self, campaign: dict[str, Any], run: dict[str, Any], source_id: str) -> bool: ...
    def render(self, manifest: dict[str, Any]) -> str: ...
    def submit(self, campaign: dict[str, Any], run: dict[str, Any], manifest: dict[str, Any], *, dry_run: bool) -> str: ...
    def status(self, campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]: ...
    def collect(self, campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]: ...
    def cancel(self, campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]: ...


class BackendRegistry:
    """Small explicit registry that removes backend branching from CLI code."""

    def __init__(self, *backends: Backend):
        self._backends = {backend.kind: backend for backend in backends}

    def get(self, kind: str) -> Backend:
        try:
            return self._backends[kind]
        except KeyError as error:
            raise ValueError(f"unsupported experiment backend: {kind}") from error
