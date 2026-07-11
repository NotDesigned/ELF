import json
from pathlib import Path

from experiment_control.runner import CommandResult, SubprocessRunner
from experiment_control.backends.wyd import WydSlurmBackend
from experiment_control.projects.elf import ElfProjectAdapter
from experiment_control.backends.sensecore import SenseCoreBackend
from experimentctl import (
    backend_services,
    materialize_run,
    prepare_run,
    reconcile_submission,
    record_submission,
    record_submission_intent,
    set_command_runner,
)
from test_experimentctl import slurm_campaign


class QueueRunner:
    def __init__(self, results):
        self.results = list(results)
        self.commands = []

    def run(self, command, **kwargs):
        self.commands.append(tuple(command))
        result = self.results.pop(0)
        if kwargs.get("check", True):
            result.check_returncode()
        return result


def test_slurm_status_contract_uses_injected_runner(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1234")
    fake = QueueRunner([CommandResult(("ssh",), 0, "1234|smoke-h100|h100|COMPLETED|00:01:00|0:0\n")])
    set_command_runner(fake)
    try:
        status = WydSlurmBackend(backend_services()).status(campaign, run)
    finally:
        set_command_runner(SubprocessRunner())
    assert status["state"] == "SUCCEEDED"
    assert any("sacct -j 1234" in argument for argument in fake.commands[0])


def test_slurm_submission_intent_recovers_job_by_unique_comment(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-009")
    intent = record_submission_intent(campaign, run, "attempt-009")
    token = intent["request"]["submission_token"]
    fake = QueueRunner([CommandResult(("ssh",), 0, f"9876|smoke-h100--attempt-009|{token}\n")])
    set_command_runner(fake)
    try:
        assert reconcile_submission(campaign, run, "attempt-009") == "9876"
    finally:
        set_command_runner(SubprocessRunner())
    backend = json.loads(
        (tmp_path / "local/controller-test/smoke-h100/backend.json").read_text(encoding="utf-8")
    )
    assert backend["backend_job_id"] == "9876"


def test_slurm_submission_intent_recovers_terminal_job_by_attempt_name(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-010")
    record_submission_intent(campaign, run, "attempt-010")
    fake = QueueRunner(
        [
            CommandResult(("squeue",), 0, ""),
            CommandResult(("sacct",), 0, "9988|smoke-h100--attempt-010\n"),
        ]
    )
    set_command_runner(fake)
    try:
        assert reconcile_submission(campaign, run, "attempt-010") == "9988"
    finally:
        set_command_runner(SubprocessRunner())


def test_sensecore_submit_contract_checks_exact_job_after_create(tmp_path: Path):
    campaign = {"project": "elf", "campaign": "sensecore-test"}
    run = {
        "run_id": "sensecore-a0",
        "backend": {
            "kind": "sensecore", "workspace": "workspace", "aec2": "cluster",
            "job_name": "elf-sensecore-a0", "display_name": "ELF A0",
            "image": "registry/elf@sha256:" + "a" * 64,
            "worker_spec": "gpu.4", "quota_type": "spot", "storage_mount": "volume:/data",
        },
    }
    fake = QueueRunner(
        [
            CommandResult(("safe-list",), 0, "[]\n"),
            CommandResult(("sco-create",), 0, ""),
            CommandResult(
                ("safe-describe",), 0,
                json.dumps({"name": "elf-sensecore-a0", "state": "WAITING", "normalized_state": "QUEUED"}),
            ),
        ]
    )
    set_command_runner(fake)
    try:
        job_id = SenseCoreBackend(backend_services()).submit(
            campaign, run, {"command": ["bash", "scripts/cloud_train.sh", "config.yml"]},
            dry_run=False,
        )
    finally:
        set_command_runner(SubprocessRunner())
    assert job_id == "elf-sensecore-a0"
    create = fake.commands[1]
    assert "--quota-type" in create and "spot" in create
    assert "--wait" in create


def test_slurm_stage_reuses_remote_source_and_sif_markers(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner(
        [
            CommandResult(("mkdir",), 0),
            CommandResult(("source-marker",), 0),
            CommandResult(("sif-marker",), 0),
        ]
    )
    set_command_runner(fake)
    try:
        bundle = ElfProjectAdapter().source_bundle(Path(__file__).resolve().parents[1])
        WydSlurmBackend(backend_services()).stage(campaign, run, "source-fixed", bundle)
    finally:
        set_command_runner(SubprocessRunner())
    assert len(fake.commands) == 3
    assert not any(command and command[0] == "rsync" for command in fake.commands)


def test_slurm_logs_bound_carriage_return_progress(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1234")
    fake = QueueRunner(
        [
            CommandResult(("stdout",), 0, "one\rtwo\rthree\n"),
            CommandResult(("stderr",), 0, "four\rfive\rsix\n"),
        ]
    )
    set_command_runner(fake)
    try:
        logs = WydSlurmBackend(backend_services()).logs(campaign, run, tail=2)
    finally:
        set_command_runner(SubprocessRunner())
    assert logs["stdout"] == ["two", "three"]
    assert logs["stderr"] == ["five", "six"]


def test_sensecore_logs_classify_expired_stream(tmp_path: Path):
    run = {
        "run_id": "sensecore-expired",
        "backend": {"kind": "sensecore", "job_name": "sensecore-expired", "workspace": "workspace"},
    }
    raw = "real-time job logs have expired (403); token=secret\n"
    fake = QueueRunner(
        [CommandResult(("stream",), 1, stderr=raw), CommandResult(("redact",), 0, stdout="real-time job logs have expired (403); token=<redacted>\n")]
    )
    set_command_runner(fake)
    try:
        logs = SenseCoreBackend(backend_services()).logs({}, run, tail=5)
    finally:
        set_command_runner(SubprocessRunner())
    assert logs["expired"] is True
    assert "secret" not in "\n".join(logs["lines"])
