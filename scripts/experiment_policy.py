"""Pure observation classification and bounded retry decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from experiment_control.states import FailureClass, classify_failure


@dataclass(frozen=True)
class Decision:
    action: str
    failure_class: str
    reason: str
    retries_used: int
    retries_allowed: int
    resume_checkpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_next_action(
    status: dict[str, Any], *, retries_used: int, max_infra_retries: int,
    diagnostic_text: str = "", completed_checkpoint: str | None = None,
) -> Decision:
    """Recommend, but never execute, a bounded action from normalized evidence."""
    state = str(status.get("state", "UNKNOWN"))
    if state in {"CREATED", "NOT_SUBMITTED"}:
        return Decision("SUBMIT", FailureClass.UNKNOWN.value, "run has no scheduler job", retries_used, max_infra_retries)
    if state in {"QUEUED", "STARTING", "RUNNING", "EVALUATING", "SUBMITTING"}:
        return Decision("OBSERVE", FailureClass.UNKNOWN.value, "run is nonterminal", retries_used, max_infra_retries)
    if state == "SUCCEEDED":
        return Decision("VERIFY_RESULTS", FailureClass.NONE.value, "scheduler succeeded; verify required metrics", retries_used, max_infra_retries)
    declared = str(status.get("failure_class") or "")
    try:
        failure = FailureClass(declared) if declared else classify_failure(diagnostic_text)
    except ValueError:
        failure = FailureClass.UNKNOWN
    retryable = failure in {FailureClass.TRANSPORT, FailureClass.SCHEDULER, FailureClass.PREEMPTION}
    if retryable and retries_used < max_infra_retries:
        return Decision(
            "RETRY_ALLOWED", failure.value, "declared infrastructure retry budget remains",
            retries_used, max_infra_retries, completed_checkpoint,
        )
    reason = "retry budget exhausted" if retryable else "failure requires explicit scientific/resource decision"
    return Decision("DO_NOT_RETRY", failure.value, reason, retries_used, max_infra_retries, completed_checkpoint)
