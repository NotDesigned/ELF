"""Normalized scheduler states and conservative failure classification."""

from __future__ import annotations

from enum import Enum


class FailureClass(str, Enum):
    TRANSPORT = "transport"
    SCHEDULER = "scheduler"
    PREEMPTION = "preemption"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    MODEL = "model"
    EVALUATION = "evaluation"
    UNKNOWN = "unknown"


SLURM_STATES = {
    "PENDING": "QUEUED", "CONFIGURING": "QUEUED", "REQUEUED": "QUEUED",
    "REQUEUE_FED": "QUEUED", "RUNNING": "RUNNING", "COMPLETING": "RUNNING",
    "COMPLETED": "SUCCEEDED", "PREEMPTED": "PREEMPTED",
    "FAILED": "FAILED", "NODE_FAIL": "FAILED", "OUT_OF_MEMORY": "FAILED",
    "TIMEOUT": "FAILED", "CANCELLED": "CANCELLED",
}


def normalize_slurm_state(raw_state: str, exit_code: str | None = None) -> str:
    """Normalize a Slurm state; success additionally requires exit ``0:0``."""
    raw = raw_state.split()[0].rstrip("+").upper()
    state = SLURM_STATES.get(raw, "UNKNOWN")
    if raw == "COMPLETED" and exit_code not in {None, "", "0:0"}:
        return "FAILED"
    return state


def normalize_sensecore_state(raw_state: str, *, cancellation_requested: bool = False) -> str:
    """Normalize a sanitized SenseCore state without inspecting raw job JSON."""
    raw = raw_state.upper()
    if cancellation_requested and raw in {"SUSPENDING", "SUSPENDED", "DELETING", "DELETED"}:
        return "CANCELLED"
    if raw in {"WAITING", "INIT", "QUEUEING", "CREATING"}:
        return "QUEUED"
    if raw in {"STARTING", "RECOVERING"}:
        return "STARTING"
    if raw in {"RUNNING", "RESTARTING"}:
        return "RUNNING"
    if raw == "SUCCEEDED":
        return "SUCCEEDED"
    if raw in {"SUSPENDING", "SUSPENDED"}:
        return "PREEMPTED"
    if raw == "FAILED":
        return "FAILED"
    if raw in {"DELETING", "DELETED"}:
        return "CANCELLED"
    return "UNKNOWN"


def classify_failure(raw_state: str = "", text: str = "") -> FailureClass:
    """Classify known failures without automatically authorizing a retry."""
    raw = raw_state.upper()
    haystack = text.lower()
    if raw in {"PREEMPTED", "SUSPENDED", "SUSPENDING"} or "spot eviction" in haystack:
        return FailureClass.PREEMPTION
    if raw in {"NODE_FAIL", "BOOT_FAIL"}:
        return FailureClass.SCHEDULER
    if raw in {"OUT_OF_MEMORY", "TIMEOUT"} or "out of memory" in haystack or "cuda oom" in haystack:
        return FailureClass.RESOURCE
    if any(token in haystack for token in ("tls", "eof", "502", "connection reset", "expired log")):
        return FailureClass.TRANSPORT
    if any(token in haystack for token in ("missing cache", "no such file", "invalid config", "not mounted")):
        return FailureClass.CONFIGURATION
    if any(token in haystack for token in ("nan", "diverg", "collapse")):
        return FailureClass.MODEL
    if "required metric" in haystack or "evaluation" in haystack and "missing" in haystack:
        return FailureClass.EVALUATION
    return FailureClass.UNKNOWN
