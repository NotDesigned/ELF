from experiment_policy import decide_next_action


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
