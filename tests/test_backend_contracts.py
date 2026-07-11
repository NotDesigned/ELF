from experiment_control.backends.base import BackendRegistry
from experiment_control.project import ProjectRegistry
from experiment_control.backends.wyd import parse_accounting, scheduler_job_name
import experimentctl


def test_slurm_accounting_contract_normalizes_exit_code():
    result = parse_accounting(
        "42|run|h100|COMPLETED|00:10:00|1:0\n",
        job_id="42", run_id="run", partition="h100",
    )
    assert result["state"] == "FAILED"
    assert result["exit_code"] == "1:0"


def test_backend_registry_rejects_unknown_kind():
    registry = BackendRegistry()
    try:
        registry.get("other")
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("unknown backend was accepted")


def test_attempt_qualified_slurm_name_is_bounded_and_deterministic():
    name = scheduler_job_name("r" * 128, "attempt-123")
    assert len(name) <= 128
    assert name == scheduler_job_name("r" * 128, "attempt-123")
    assert name != scheduler_job_name("r" * 128, "attempt-124")


def test_controller_core_accepts_a_registered_backend_without_platform_branch(monkeypatch):
    class FakeBackend:
        kind = "fake"

        def validate(self, run):
            run["validated_by_fake"] = True

        def environment(self, campaign, run, source_id, attempt_id):
            return {"FAKE_ALLOCATION": "ready"}

    class FakeProject:
        name = "project"
        safe_env_keys = frozenset()

        def environment(self, campaign, run):
            return {"PROJECT_RUNTIME": "ready"}

    registry = BackendRegistry(FakeBackend())
    monkeypatch.setattr(experimentctl, "BACKENDS", registry)
    monkeypatch.setattr(experimentctl, "PROJECTS", ProjectRegistry(FakeProject()))
    run = {
        "run_id": "abstract-run", "config": "config.yml", "image_id": "immutable",
        "backend": {"kind": "fake"}, "env": {},
        "storage": {
            "run_dir": "/persistent/run", "data_root": "/persistent",
            "project_data_root": "/persistent/project", "hf_home": "/persistent/hf",
            "hf_datasets_cache": "/persistent/hf/datasets",
        },
    }
    experimentctl.validate_run(run)
    env = experimentctl.command_environment(
        {"project": "project", "campaign": "campaign"}, run, "source", "attempt-001"
    )
    assert run["validated_by_fake"] is True
    assert env["BACKEND"] == "fake"
    assert env["FAKE_ALLOCATION"] == "ready"
    assert env["PROJECT_RUNTIME"] == "ready"
