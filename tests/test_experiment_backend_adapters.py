import hashlib
import json
import os
from pathlib import Path

import pytest

from experiment_control.runner import CommandResult, SubprocessRunner
from experiment_control.backends.wyd import WydSlurmBackend
from elf_experiments.projects.elf import ElfProjectAdapter
from experiment_control.backends.sensecore import (
    SenseCoreBackend,
    digest_pinned_image,
    scheduler_job_name as sensecore_scheduler_job_name,
)
from experiment_control.project import AssetProbe, AssetRequirement
from elf_experiments.controller import (
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


def test_sensecore_preflight_checks_cli_and_sanitized_workspace_access():
    run = {
        "run_id": "sensecore-preflight",
        "backend": {
            "kind": "sensecore", "workspace": "workspace",
            "job_name": "sensecore-preflight",
        },
    }
    fake = QueueRunner([
        CommandResult(("sco-version",), 0, "v1.2.0\n"),
        CommandResult(("safe-list",), 0, "[]\n"),
    ])
    set_command_runner(fake)
    try:
        report = SenseCoreBackend(backend_services()).preflight(run, scope="submit")
    finally:
        set_command_runner(SubprocessRunner())
    assert report.ready is True
    assert [check.name for check in report.checks] == ["sco-cli", "workspace-access"]


def test_sensecore_preflight_fails_closed_on_malformed_sanitized_response():
    run = {
        "run_id": "sensecore-preflight",
        "backend": {"kind": "sensecore", "workspace": "workspace", "job_name": "job"},
    }
    fake = QueueRunner([
        CommandResult(("sco-version",), 0, "v1.2.0\n"),
        CommandResult(("safe-list",), 1, "", "safe_sco: input was not valid JSON; raw response suppressed"),
        CommandResult(("redact",), 0, "safe_sco: input was not valid JSON; raw response suppressed"),
    ])
    set_command_runner(fake)
    try:
        report = SenseCoreBackend(backend_services()).preflight(run, scope="submit")
    finally:
        set_command_runner(SubprocessRunner())
    assert report.ready is False


def test_sensecore_identity_probe_reports_consumed_exact_name():
    run = {
        "run_id": "sensecore-identity",
        "backend": {
            "kind": "sensecore", "workspace": "workspace",
            "job_name": "sensecore-identity",
        },
    }
    fake = QueueRunner([
        CommandResult(
            ("safe-list",), 0,
            '[{"name":"sensecore-identity--attempt-001","state":"RUNNING"}]\n',
        ),
    ])
    services = backend_services()
    backend = SenseCoreBackend(type(services)(
        fake.run, services.local_run_dir, services.backend_record,
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    ))
    report = backend.identity(
        {"campaign": "identity-test"}, run, "attempt-001"
    ).to_dict()
    assert report == {
        "available": False,
        "ambiguous": False,
        "scheduler_job_ids": ["sensecore-identity--attempt-001"],
        "remote_manifest_exists": None,
        "remote_manifest_matches": None,
    }


def test_sensecore_attempt_identity_and_render_pin_the_recorded_digest():
    digest = "sha256:" + "b" * 64
    run = {
        "run_id": "sensecore-render",
        "image_id": digest,
        "backend": {
            "kind": "sensecore", "workspace": "workspace", "aec2": "cluster",
            "job_name": "sensecore-render", "display_name": "render test",
            "image": "registry.example/project/image:runtime-source-fixed",
            "worker_spec": "gpu.4", "quota_type": "spot",
            "storage_mount": "volume/subdir:/data",
        },
    }
    manifest = {
        **run, "attempt_id": "attempt-002",
        "command": ["python", "train.py"],
    }
    fake = QueueRunner([])
    services = backend_services()
    backend = SenseCoreBackend(type(services)(
        fake.run, services.local_run_dir, services.backend_record,
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    ))

    request = backend.submission_request({}, run, "attempt-002")
    assert request == {
        "scheduler_name": "sensecore-render--attempt-002",
        "image_tag": "registry.example/project/image:runtime-source-fixed",
        "image_digest": digest,
        "image_reference": f"registry.example/project/image@{digest}",
    }
    rendered = backend.render(manifest)
    assert "--name sensecore-render--attempt-002" in rendered
    assert f"--container-image-url registry.example/project/image@{digest}" in rendered
    assert "image:runtime-source-fixed" not in rendered
    assert rendered.startswith("timeout 120s env ")
    assert backend.submit({}, run, manifest, dry_run=True) == "DRY_RUN"
    assert fake.commands == []

    conflicting = {**manifest, "image_id": "sha256:" + "e" * 64}
    with pytest.raises(ValueError, match="image_id conflicts"):
        backend.submit({}, run, conflicting, dry_run=True)


def test_sensecore_attempt_resource_names_are_distinct_and_bounded():
    first = sensecore_scheduler_job_name("run", "attempt-001")
    second = sensecore_scheduler_job_name("run", "attempt-002")
    assert first == "run--attempt-001"
    assert first != second
    assert len(sensecore_scheduler_job_name("r" * 80, "attempt-001")) <= 63
    assert digest_pinned_image(
        "registry.example:5000/ns/image:source-abc", "sha256:" + "c" * 64
    ) == "registry.example:5000/ns/image@sha256:" + "c" * 64


@pytest.mark.parametrize(
    ("base_name", "attempt_id"),
    [
        ("Run", "attempt-001"),
        ("run_name", "attempt-001"),
        ("1run", "attempt-001"),
        ("run", "Attempt-001"),
        ("run", "attempt-001-"),
    ],
)
def test_sensecore_resource_name_rejects_values_the_api_would_reject(
    base_name, attempt_id
):
    with pytest.raises(ValueError, match="SenseCore"):
        sensecore_scheduler_job_name(base_name, attempt_id)


def test_sensecore_create_timeout_is_bounded(monkeypatch):
    backend = SenseCoreBackend(backend_services())
    monkeypatch.setenv("EXPERIMENTCTL_SCO_CREATE_TIMEOUT_SECONDS", "9")
    with pytest.raises(ValueError, match="integer from 10 to 600"):
        backend.create_timeout_seconds()


@pytest.mark.parametrize("method_name", ["find", "describe"])
def test_sensecore_query_errors_are_redacted_before_exception(method_name: str):
    secret = "credential-value-that-must-not-escape"
    run = {
        "run_id": "sensecore-secret",
        "backend": {
            "kind": "sensecore", "workspace": "workspace", "job_name": "job",
        },
    }
    fake = QueueRunner([
        CommandResult(("safe-query",), 1, stderr=f"access_key_secret={secret}\n"),
        CommandResult(("redact",), 0, stdout="access_key_secret=<redacted>\n"),
    ])
    services = backend_services()
    backend = SenseCoreBackend(type(services)(
        fake.run, services.local_run_dir, services.backend_record,
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    ))

    with pytest.raises(RuntimeError) as captured:
        getattr(backend, method_name)(run, "job--attempt-001")
    assert secret not in str(captured.value)
    assert "<redacted>" in str(captured.value)
    assert "redact-lines" in fake.commands[0][-1]
    assert "2>&1" not in fake.commands[0][-1]


def test_slurm_preflight_checks_tools_live_resources_and_storage(tmp_path: Path):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner([
        CommandResult(("ssh-version",), 0),
        CommandResult(("rsync-version",), 0),
        CommandResult(
            ("slurm-live",), 0,
            "h100|up|3-00:00:00|gpu:h100:8\nuser|lab||normal|normal\n",
        ),
        CommandResult(("runtime-storage",), 0),
    ])
    set_command_runner(fake)
    try:
        report = WydSlurmBackend(backend_services()).preflight(run, scope="stage")
    finally:
        set_command_runner(SubprocessRunner())
    assert report.ready is True
    assert [check.name for check in report.checks] == [
        "ssh-cli", "rsync-cli", "slurm-access", "runtime-storage",
    ]


def test_slurm_observe_preflight_only_requires_control_access(tmp_path: Path):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner([
        CommandResult(("ssh-version",), 0),
        CommandResult(("squeue",), 0, ""),
    ])
    set_command_runner(fake)
    try:
        report = WydSlurmBackend(backend_services()).preflight(run, scope="observe")
    finally:
        set_command_runner(SubprocessRunner())
    assert report.ready is True
    assert [check.name for check in report.checks] == ["ssh-cli", "slurm-access"]


def test_slurm_preflight_fails_closed_before_remote_access(tmp_path: Path):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner([
        CommandResult(("ssh-version",), 127),
        CommandResult(("rsync-version",), 0),
    ])
    set_command_runner(fake)
    try:
        report = WydSlurmBackend(backend_services()).preflight(run, scope="submit")
    finally:
        set_command_runner(SubprocessRunner())
    assert report.ready is False
    assert len(fake.commands) == 1


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


def test_slurm_submit_stages_canonical_manifest_before_job_script(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    manifest = prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    fake = QueueRunner([
        CommandResult(("validate-live",), 0, "h100|up|3-00:00:00|gpu:h100:8\nuser|lab||normal|normal\n"),
        CommandResult(("mkdir",), 0),
        CommandResult(("manifest-rsync",), 0),
        CommandResult(("script-rsync",), 0),
        CommandResult(("sbatch",), 0, "4321\n"),
    ])
    set_command_runner(fake)
    try:
        job_id = WydSlurmBackend(backend_services()).submit(
            campaign, run, manifest, dry_run=False
        )
    finally:
        set_command_runner(SubprocessRunner())
    assert job_id == "4321"
    assert ".submission-attempt-001" in " ".join(fake.commands[1])
    assert fake.commands[2][-1].endswith("/manifest.yaml")
    assert "controller-attempt-001.sbatch" in fake.commands[3][-1]


def test_slurm_submit_claim_blocks_duplicate_scheduler_mutation(tmp_path):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    manifest = prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    fake = QueueRunner([
        CommandResult(
            ("validate-live",), 0,
            "h100|up|3-00:00:00|gpu:h100:8\nuser|lab||normal|normal\n",
        ),
        CommandResult(("claim",), 1, "", "already exists"),
    ])
    services = backend_services()
    backend = WydSlurmBackend(type(services)(
        fake.run, services.local_run_dir, services.backend_record,
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    ))
    with pytest.raises(FileExistsError, match="submission claim"):
        backend.submit(campaign, run, manifest, dry_run=False)
    assert len(fake.commands) == 2


def test_slurm_collection_reports_latest_remote_completed_checkpoint(tmp_path):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner([
        CommandResult(("collect-rsync",), 0),
        CommandResult(("checkpoint-probe",), 0, "checkpoint_8\ncheckpoint_21\n"),
        CommandResult(("stdout-probe",), 1),
        CommandResult(("stderr-probe",), 1),
    ])
    services = backend_services()
    services = type(services)(
        fake.run, services.local_run_dir,
        lambda campaign, run: {
            "attempt_id": "attempt-001", "backend_job_id": "1760"
        },
        lambda campaign, path: {"run_id": run["run_id"]}, services.parse_metric,
        services.parse_checkpoint, services.atomic_write, services.utc_now,
    )
    summary = WydSlurmBackend(services).collect(campaign, run)
    assert summary["latest_completed_checkpoint"].endswith("/checkpoint_21")
    assert summary["latest_completed_checkpoint_step"] == 21
    assert summary["process_evidence"] == {
        "observed": False,
        "sources": {"stdout": None, "stderr": None},
        "stdout_tail": [],
        "stderr_tail": [],
    }


def test_slurm_submission_intent_recovers_job_by_unique_comment(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-009")
    intent = record_submission_intent(campaign, run, "attempt-009")
    token = intent["request"]["submission_token"]
    fake = QueueRunner([
        CommandResult(("squeue",), 0, f"9876|smoke-h100--attempt-009|{token}\n"),
        CommandResult(("sacct",), 0, "9876|smoke-h100--attempt-009\n"),
    ])
    set_command_runner(fake)
    try:
        assert reconcile_submission(campaign, run, "attempt-009") == "9876"
    finally:
        set_command_runner(SubprocessRunner())
    backend_record = json.loads(
        (tmp_path / "local/controller-test/smoke-h100/backend.json").read_text(
            encoding="utf-8"
        )
    )
    assert backend_record["backend_job_id"] == "9876"


def test_slurm_submission_recovery_rejects_multiple_matching_jobs(tmp_path: Path):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner([
        CommandResult(("squeue",), 0, "1732|smoke-h100--attempt-001|token\n"),
        CommandResult(
            ("sacct",), 0,
            "1731|smoke-h100--attempt-001\n1732|smoke-h100--attempt-001\n",
        ),
    ])
    backend = WydSlurmBackend(type(backend_services())(
        fake.run,
        backend_services().local_run_dir,
        backend_services().backend_record,
        backend_services().summarize_run,
        backend_services().parse_metric,
        backend_services().parse_checkpoint,
        backend_services().atomic_write,
        backend_services().utc_now,
    ))
    with pytest.raises(RuntimeError, match="2 jobs match"):
        backend.recover_submission(
            run, {"submission_token": "token"}, "attempt-001"
        )


def test_slurm_identity_probe_is_read_only_and_reports_availability(tmp_path: Path):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner([
        CommandResult(("squeue",), 0, ""),
        CommandResult(("sacct",), 0, ""),
        CommandResult(("manifest",), 1, ""),
    ])
    services = backend_services()
    backend = WydSlurmBackend(type(services)(
        fake.run, services.local_run_dir, services.backend_record,
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    ))
    report = backend.identity(campaign, run, "attempt-001").to_dict()
    assert report == {
        "available": True,
        "ambiguous": False,
        "scheduler_job_ids": [],
        "remote_manifest_exists": False,
        "remote_manifest_matches": None,
    }
    assert len(fake.commands) == 3


def test_slurm_remote_manifest_is_owned_only_when_digest_matches(
    tmp_path: Path, monkeypatch
):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    local_manifest = (
        tmp_path / "local" / "controller-test" / "smoke-h100" / "manifest.yaml"
    )
    digest = hashlib.sha256(local_manifest.read_bytes()).hexdigest()
    fake = QueueRunner([
        CommandResult(("squeue",), 0, ""),
        CommandResult(("sacct",), 0, ""),
        CommandResult(("manifest",), 0, ""),
        CommandResult(("sha256sum",), 0, f"{digest}\n"),
    ])
    services = backend_services()
    backend = WydSlurmBackend(type(services)(
        fake.run, services.local_run_dir, services.backend_record,
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    ))
    report = backend.identity(campaign, run, "attempt-002").to_dict()
    assert report["available"] is False
    assert report["remote_manifest_exists"] is True
    assert report["remote_manifest_matches"] is True


def test_slurm_remote_manifest_digest_mismatch_is_not_owned(
    tmp_path: Path, monkeypatch
):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    fake = QueueRunner([
        CommandResult(("squeue",), 0, ""),
        CommandResult(("sacct",), 0, ""),
        CommandResult(("manifest",), 0, ""),
        CommandResult(("sha256sum",), 0, f"{'0' * 64}\n"),
    ])
    services = backend_services()
    backend = WydSlurmBackend(type(services)(
        fake.run, services.local_run_dir, services.backend_record,
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    ))
    assert backend.identity(campaign, run, "attempt-002").remote_manifest_matches is False


@pytest.mark.parametrize(
    "results",
    [
        [CommandResult(("squeue",), 255, stderr="ssh unavailable")],
        [
            CommandResult(("squeue",), 0, ""),
            CommandResult(("sacct",), 255, stderr="ssh unavailable"),
        ],
        [
            CommandResult(("squeue",), 0, ""),
            CommandResult(("sacct",), 0, ""),
            CommandResult(("manifest",), 255, stderr="ssh unavailable"),
        ],
    ],
)
def test_slurm_identity_fails_closed_when_remote_evidence_is_unavailable(
    tmp_path: Path, results,
):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner(results)
    services = backend_services()
    backend = WydSlurmBackend(type(services)(
        fake.run, services.local_run_dir, services.backend_record,
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    ))
    with pytest.raises(RuntimeError, match="evidence is unavailable"):
        backend.identity(campaign, run, "attempt-001")


def test_slurm_status_does_not_treat_transport_failure_as_missing_job(tmp_path: Path):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner([
        CommandResult(("sacct",), 255, stderr="connection timed out"),
    ])
    services = backend_services()
    services = type(services)(
        fake.run, services.local_run_dir,
        lambda campaign, run: {
            "attempt_id": "attempt-001", "backend_job_id": "1234",
        },
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    )
    with pytest.raises(RuntimeError, match="accounting status query"):
        WydSlurmBackend(services).status(campaign, run)
    assert len(fake.commands) == 1


def test_slurm_asset_probe_distinguishes_missing_from_transport_failure(tmp_path: Path):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    requirement = AssetRequirement("dataset", "dataset-id", "training")
    probe = AssetProbe(requirement, "/data/dataset", file=False)
    services = backend_services()

    missing_runner = QueueRunner([CommandResult(("test",), 1)])
    missing_services = type(services)(
        missing_runner.run, services.local_run_dir, services.backend_record,
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    )
    report = WydSlurmBackend(missing_services).verify_assets(run, [probe])
    assert report["missing"][0]["identity"] == "dataset-id"

    failed_runner = QueueRunner([CommandResult(("test",), 255, stderr="ssh failed")])
    failed_services = type(services)(
        failed_runner.run, services.local_run_dir, services.backend_record,
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    )
    with pytest.raises(RuntimeError, match="evidence is unavailable"):
        WydSlurmBackend(failed_services).verify_assets(run, [probe])


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
            "image": "registry/elf:runtime-source-fixed",
            "worker_spec": "gpu.4", "quota_type": "spot", "storage_mount": "volume:/data",
        },
        "image_id": "sha256:" + "a" * 64,
    }
    fake = QueueRunner(
        [
            CommandResult(("safe-list",), 0, "[]\n"),
            CommandResult(("sco-create",), 0, ""),
            CommandResult(
                ("safe-describe",), 0,
                json.dumps({
                    "name": "elf-sensecore-a0--attempt-001",
                    "state": "WAITING", "normalized_state": "QUEUED",
                }),
            ),
        ]
    )
    set_command_runner(fake)
    try:
        job_id = SenseCoreBackend(backend_services()).submit(
            campaign, run, {
                **run, "attempt_id": "attempt-001",
                "command": ["bash", "scripts/cloud_train.sh", "config.yml"],
            },
            dry_run=False,
        )
    finally:
        set_command_runner(SubprocessRunner())
    assert job_id == "elf-sensecore-a0--attempt-001"
    create = fake.commands[1]
    assert "--quota-type" in create and "spot" in create
    assert "--wait" in create
    image_index = create.index("--container-image-url") + 1
    assert create[image_index] == "registry/elf@sha256:" + "a" * 64


def test_slurm_stage_reuses_remote_source_and_sif_markers(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner(
        [
            CommandResult(("mkdir",), 0),
            CommandResult(("source-marker",), 0),
            CommandResult(("sif-marker",), 0),
            CommandResult(("required-path",), 0),
            CommandResult(("required-path",), 0),
        ]
    )
    set_command_runner(fake)
    try:
        bundle = ElfProjectAdapter().source_bundle(Path(__file__).resolve().parents[1])
        WydSlurmBackend(backend_services()).stage(campaign, run, "source-fixed", bundle)
    finally:
        set_command_runner(SubprocessRunner())
    assert len(fake.commands) == 5
    assert not any(command and command[0] == "rsync" for command in fake.commands)
    required_path_commands = [" ".join(command) for command in fake.commands[-2:]]
    assert all("test -s" in command for command in required_path_commands)
    assert "scripts/cloud_train.sh" in required_path_commands[0]
    assert "src/train.py" in required_path_commands[1]


def test_slurm_ssh_control_socket_is_scoped_to_current_process():
    backend = WydSlurmBackend(backend_services())
    assert f"-{os.getpid()}-" in backend.ssh_control_path
    assert backend.ssh_control_path in backend.ssh_transport()


def test_slurm_stage_fails_when_required_source_path_is_missing(tmp_path: Path):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner([
        CommandResult(("mkdir",), 0),
        CommandResult(("source-marker",), 0),
        CommandResult(("sif-marker",), 0),
        CommandResult(("required-path",), 1, "", "missing path"),
    ])
    services = backend_services()
    services = type(services)(
        fake.run, services.local_run_dir, services.backend_record,
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    )
    bundle = ElfProjectAdapter().source_bundle(Path(__file__).resolve().parents[1])
    with pytest.raises(RuntimeError, match="missing required project path"):
        WydSlurmBackend(services).stage(campaign, run, "source-fixed", bundle)


def test_slurm_stage_required_path_probe_fails_closed_on_transport(tmp_path: Path):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner([
        CommandResult(("mkdir",), 0),
        CommandResult(("source-marker",), 0),
        CommandResult(("sif-marker",), 0),
        CommandResult(("required-path",), 255, stderr="ssh unavailable"),
    ])
    services = backend_services()
    services = type(services)(
        fake.run, services.local_run_dir, services.backend_record,
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    )
    bundle = ElfProjectAdapter().source_bundle(Path(__file__).resolve().parents[1])
    with pytest.raises(RuntimeError, match="evidence is unavailable"):
        WydSlurmBackend(services).stage(campaign, run, "source-fixed", bundle)


@pytest.mark.parametrize(
    "resources,gres,error",
    [
        ({"gpus": 2}, "gpu:h100:1", "does not match"),
        ({"gpus": 1, "nodes": 2}, "gpu:h100:1", "resources.nodes=1"),
    ],
)
def test_slurm_validation_rejects_resource_request_drift(
    tmp_path: Path, resources, gres: str, error: str,
):
    campaign = slurm_campaign(tmp_path)
    authored = campaign["runs"][0]
    authored["resources"] = resources
    authored["backend"]["gres"] = gres
    with pytest.raises(ValueError, match=error):
        materialize_run(campaign, authored, "source-fixed")


def test_slurm_logs_bound_carriage_return_progress(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1234")
    fake = QueueRunner(
        [
            CommandResult(
                ("stdout",), 0,
                f"{run['storage']['run_dir']}/attempts/attempt-001/stdout.log\n"
                "one\rtwo\rthree\n",
            ),
            CommandResult(
                ("stderr",), 0,
                f"{run['storage']['run_dir']}/attempts/attempt-001/stderr.log\n"
                "four\rfive\rsix\n",
            ),
        ]
    )
    set_command_runner(fake)
    try:
        logs = WydSlurmBackend(backend_services()).logs(campaign, run, tail=2)
    finally:
        set_command_runner(SubprocessRunner())
    assert logs["stdout"] == ["two", "three"]
    assert logs["stderr"] == ["five", "six"]


def test_slurm_logs_fall_back_to_job_qualified_files_and_redact(tmp_path: Path):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    run_dir = run["storage"]["run_dir"]
    fake = QueueRunner([
        CommandResult(
            ("stdout",), 0,
            f"{run_dir}/slurm-1760.out\nlauncher started\n",
        ),
        CommandResult(
            ("stderr",), 0,
            f"{run_dir}/slurm-1760.err\n"
            "token=top-secret\nModuleNotFoundError: No module named 'dependency'\n",
        ),
    ])
    services = backend_services()
    services = type(services)(
        fake.run, services.local_run_dir,
        lambda campaign, run: {
            "attempt_id": "attempt-001", "backend_job_id": "1760"
        },
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    )
    logs = WydSlurmBackend(services).logs(campaign, run, tail=20)
    assert logs["sources"] == {
        "stdout": f"{run_dir}/slurm-1760.out",
        "stderr": f"{run_dir}/slurm-1760.err",
    }
    assert logs["stdout"] == ["launcher started"]
    assert logs["stderr"] == [
        "token=<redacted>",
        "ModuleNotFoundError: No module named 'dependency'",
    ]
    commands = "\n".join(" ".join(command) for command in fake.commands)
    assert "slurm-1760.out" in commands
    assert "slurm-1760.err" in commands
    assert "*" not in commands


def test_slurm_collection_includes_sanitized_process_failure_evidence(
    tmp_path: Path,
):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    run_dir = run["storage"]["run_dir"]
    fake = QueueRunner([
        CommandResult(("collect-rsync",), 0),
        CommandResult(("checkpoint-probe",), 0),
        CommandResult(("stdout",), 1),
        CommandResult(
            ("stderr",), 0,
            f"{run_dir}/attempts/attempt-001/slurm-1760.err\n"
            "access_key_secret=do-not-persist\n"
            "ModuleNotFoundError: No module named 'experiment_control'\n",
        ),
    ])
    services = backend_services()
    services = type(services)(
        fake.run, services.local_run_dir,
        lambda campaign, run: {
            "attempt_id": "attempt-001", "backend_job_id": "1760"
        },
        lambda campaign, path: {"run_id": run["run_id"]},
        services.parse_metric, services.parse_checkpoint, services.atomic_write,
        services.utc_now,
    )
    summary = WydSlurmBackend(services).collect(campaign, run)
    evidence = summary["process_evidence"]
    assert evidence["observed"] is True
    assert evidence["stdout_tail"] == []
    assert evidence["stderr_tail"] == [
        "access_key_secret=<redacted>",
        "ModuleNotFoundError: No module named 'experiment_control'",
    ]
    assert "do-not-persist" not in json.dumps(summary)


@pytest.mark.parametrize("tail", [0, 10001])
def test_slurm_logs_reject_unbounded_tail_before_remote_access(tmp_path: Path, tail: int):
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner([])
    services = backend_services()
    services = type(services)(
        fake.run, services.local_run_dir,
        lambda campaign, run: {
            "attempt_id": "attempt-001", "backend_job_id": "1760"
        },
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    )
    with pytest.raises(ValueError, match="tail must be between"):
        WydSlurmBackend(services).logs(campaign, run, tail=tail)
    assert fake.commands == []


def test_sensecore_logs_classify_expired_stream(tmp_path: Path):
    run = {
        "run_id": "sensecore-expired",
        "backend": {"kind": "sensecore", "job_name": "sensecore-expired", "workspace": "workspace"},
    }
    raw = "real-time job logs have expired (403); token=secret\n"
    fake = QueueRunner(
        [CommandResult(("stream",), 1, stderr=raw), CommandResult(("redact",), 0, stdout="real-time job logs have expired (403); token=<redacted>\n")]
    )
    services = backend_services()
    services = type(services)(
        fake.run, services.local_run_dir,
        lambda campaign, run: {
            "attempt_id": "attempt-002",
            "backend_job_id": "sensecore-expired--attempt-002",
        },
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    )
    logs = SenseCoreBackend(services).logs({}, run, tail=5)
    assert logs["expired"] is True
    assert "secret" not in "\n".join(logs["lines"])
    assert logs["backend_job_id"] == "sensecore-expired--attempt-002"
    assert "sensecore-expired--attempt-002" in fake.commands[0]


def test_sensecore_cancel_preserves_terminal_preemption(tmp_path, monkeypatch):
    writes = []
    services = backend_services()
    services = type(services)(
        services.run_command, lambda campaign, run: tmp_path,
        services.backend_record, services.summarize_run, services.parse_metric,
        services.parse_checkpoint,
        lambda *args, **kwargs: writes.append((args, kwargs)), services.utc_now,
    )
    backend = SenseCoreBackend(services)
    monkeypatch.setattr(backend, "status", lambda campaign, run: {
        "state": "PREEMPTED", "raw_state": "SUSPENDED", "backend_job_id": "job",
    })
    result = backend.cancel({}, {"run_id": "run", "backend": {}})
    assert result["state"] == "PREEMPTED"
    assert writes == []


def test_sensecore_status_and_cancel_use_recorded_attempt_resource(tmp_path: Path):
    resource_name = "sensecore-run--attempt-002"
    run = {
        "run_id": "sensecore-run",
        "backend": {
            "kind": "sensecore", "workspace": "workspace", "job_name": "sensecore-run",
        },
    }
    fake = QueueRunner([
        CommandResult(("describe-running",), 0, json.dumps({
            "name": resource_name, "state": "RUNNING",
        })),
        CommandResult(("stop",), 0),
        CommandResult(("describe-deleted",), 0, json.dumps({
            "name": resource_name, "state": "DELETED",
        })),
    ])
    services = backend_services()
    services = type(services)(
        fake.run, lambda campaign, run: tmp_path,
        lambda campaign, run: {
            "attempt_id": "attempt-002", "backend_job_id": resource_name,
        },
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    )
    status = SenseCoreBackend(services).cancel({}, run)
    assert status["state"] == "CANCELLED"
    assert resource_name in fake.commands[0][-1]
    assert resource_name in fake.commands[1]
    assert resource_name in fake.commands[2][-1]


def test_sensecore_status_honors_matching_legacy_run_cancel_marker(tmp_path: Path):
    resource_name = "sensecore-run--attempt-001"
    attempt_dir = tmp_path / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    (tmp_path / "cancel_requested.json").write_text(
        json.dumps({"backend_job_id": resource_name}), encoding="utf-8"
    )
    run = {
        "run_id": "sensecore-run",
        "backend": {"kind": "sensecore", "workspace": "workspace"},
    }
    fake = QueueRunner([
        CommandResult(("describe",), 0, json.dumps({
            "name": resource_name, "state": "SUSPENDED",
        })),
    ])
    services = backend_services()
    services = type(services)(
        fake.run, lambda campaign, run: attempt_dir,
        lambda campaign, run: {
            "attempt_id": "attempt-001", "backend_job_id": resource_name,
        },
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    )
    assert SenseCoreBackend(services).status({}, run)["state"] == "CANCELLED"


def test_sensecore_collection_extracts_committed_checkpoint_from_logs(monkeypatch):
    backend = SenseCoreBackend(backend_services())
    monkeypatch.setattr(backend, "logs", lambda campaign, run, tail: {
        "lines": [
            "Checkpoint committed to /data/elf/runs/run/checkpoint_8 (120 bytes)",
            "Checkpoint committed to /data/elf/runs/run/checkpoint_21 (240 bytes)",
        ],
        "expired": False,
    })
    monkeypatch.setattr(backend, "workers", lambda campaign, run: {
        "worker_state": "RELEASED", "worker_phases": ["Deleted"],
        "worker_evidence_available": True,
    })
    summary = backend.collect(
        {"project": "elf"}, {"run_id": "run", "backend": {}}
    )
    assert summary["latest_completed_checkpoint"].endswith("checkpoint_21")
    assert summary["latest_completed_checkpoint_step"] == 21
    assert summary["worker_state"] == "RELEASED"


@pytest.mark.parametrize(
    ("phase", "expected"),
    [("Pending", "PENDING"), ("Running", "ALLOCATED"), ("Deleted", "RELEASED")],
)
def test_sensecore_worker_query_is_sanitized_and_normalized(
    tmp_path: Path, phase: str, expected: str
):
    resource_name = "sensecore-run--attempt-001"
    fake = QueueRunner([
        CommandResult(("workers",), 0, json.dumps([{
            "worker_name": "worker-0", "resource": "4 accelerators",
            "phase": phase,
        }])),
    ])
    services = backend_services()
    services = type(services)(
        fake.run, lambda campaign, run: tmp_path,
        lambda campaign, run: {
            "attempt_id": "attempt-001", "backend_job_id": resource_name,
        },
        services.summarize_run, services.parse_metric, services.parse_checkpoint,
        services.atomic_write, services.utc_now,
    )
    result = SenseCoreBackend(services).workers(
        {}, {"run_id": "sensecore-run", "backend": {
            "kind": "sensecore", "workspace": "workspace",
        }}
    )
    assert result["worker_state"] == expected
    assert resource_name in fake.commands[0][-1]
    assert "worker-list" in fake.commands[0][-1]
