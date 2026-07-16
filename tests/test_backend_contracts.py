import experiment_control
from experiment_control.backends.base import BackendRegistry
from experiment_control.project import ProjectRegistry
from elf_experiments import controller as experimentctl


def test_package_dependency_uses_documented_public_surface():
    experiment_control.validate_identity("run_id", "elf-run")
    experiment_control.require_immutable("source_id", "source-deadbeef")
    assert callable(experiment_control.append_event)
    assert callable(experiment_control.atomic_write)
    assert callable(experiment_control.sanitize_command)
    assert callable(experiment_control.utc_now)


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
