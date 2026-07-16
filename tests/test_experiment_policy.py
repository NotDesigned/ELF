from elf_experiments.policy import decide_next_action


def test_policy_allows_only_bounded_infrastructure_retry():
    allowed = decide_next_action(
        {"state": "PREEMPTED", "failure_class": "preemption"}, retries_used=0, max_infra_retries=2
    )
    assert allowed.action == "RETRY_ALLOWED"
    exhausted = decide_next_action(
        {"state": "PREEMPTED", "failure_class": "preemption"}, retries_used=2, max_infra_retries=2
    )
    assert exhausted.action == "DO_NOT_RETRY"


def test_policy_never_relabels_oom_as_infrastructure():
    decision = decide_next_action(
        {"state": "FAILED", "failure_class": "resource"}, retries_used=0, max_infra_retries=3
    )
    assert decision.action == "DO_NOT_RETRY"
    assert decision.failure_class == "resource"


def test_unknown_failure_with_budget_requires_review_instead_of_forbidding_retry():
    decision = decide_next_action(
        {"state": "FAILED"}, retries_used=0, max_infra_retries=1,
    )
    assert decision.action == "REVIEW_RETRY"
    assert decision.failure_class == "unknown"


def test_unknown_failure_without_budget_remains_forbidden():
    decision = decide_next_action(
        {"state": "FAILED"}, retries_used=0, max_infra_retries=0,
    )
    assert decision.action == "DO_NOT_RETRY"


def test_retry_decision_carries_only_observed_completed_checkpoint():
    decision = decide_next_action(
        {"state": "PREEMPTED", "failure_class": "preemption"},
        retries_used=0, max_infra_retries=1,
        completed_checkpoint="/runs/project/run/checkpoint_21",
    )
    assert decision.action == "RETRY_ALLOWED"
    assert decision.resume_checkpoint.endswith("checkpoint_21")


def test_scheduler_success_has_no_failure_class():
    decision = decide_next_action(
        {"state": "SUCCEEDED"}, retries_used=0, max_infra_retries=0
    )
    assert decision.action == "VERIFY_RESULTS"
    assert decision.failure_class == "none"
    assert decision.evidence_outcome == "INCONCLUSIVE"


def test_cancelled_run_is_not_a_failure_but_remains_inconclusive():
    decision = decide_next_action(
        {"state": "CANCELLED"}, retries_used=0, max_infra_retries=2
    )
    assert decision.action == "DO_NOT_RETRY"
    assert decision.failure_class == "none"
    assert decision.evidence_outcome == "INCONCLUSIVE"
