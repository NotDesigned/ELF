from experiment_overrides import operational_overrides


def test_batch_size_has_explicit_precedence_over_global_batch_size():
    overrides = operational_overrides(
        {"RUN_ID": "run", "GLOBAL_BATCH_SIZE": "512", "BATCH_SIZE": "64"}, "/data/run"
    )
    assert overrides[-3:] == ["global_batch_size=512", "global_batch_size=null", "batch_size=64"]


def test_wandb_attempt_identity_defaults_are_deterministic():
    overrides = operational_overrides({"RUN_ID": "run-42"}, "/data/run-42")
    assert "wandb_run_name=run-42" in overrides
    assert "wandb_run_id=run-42" in overrides
    assert "wandb_resume=allow" in overrides
