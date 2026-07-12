"""Thin integration coverage for ELF's injection of ml-experiment-control."""

from __future__ import annotations

import json
from pathlib import Path

from backend_fixtures import slurm_campaign
from elf_experiments import controller as experimentctl
from elf_experiments.controller import (
    annotate_collection,
    backend_services,
    materialize_run,
    prepare_run,
    reconcile_submission,
    record_submission,
    record_submission_intent,
    set_command_runner,
)
from elf_experiments.projects.elf import ElfProjectAdapter
from experiment_control.backends.wyd import WydSlurmBackend
from experiment_control.runner import CommandResult, SubprocessRunner


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


def test_elf_controller_registers_installed_package_backends():
    assert experimentctl.BACKENDS.kinds == frozenset({"sensecore", "slurm"})


def test_slurm_backend_submits_an_elf_prepared_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    manifest = prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    fake = QueueRunner([
        CommandResult(
            ("validate-live",), 0,
            "h100|up|3-00:00:00|gpu:h100:8\nuser|lab||normal|normal\n",
        ),
        CommandResult(("claim",), 0),
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
    assert fake.commands[2][-1].endswith("/manifest.yaml")
    assert "controller-attempt-001.sbatch" in fake.commands[3][-1]


def test_controller_reconciles_attempt_identity_through_package_backend(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-009")
    intent = record_submission_intent(campaign, run, "attempt-009")
    token = intent["request"]["submission_token"]
    fake = QueueRunner([
        CommandResult(
            ("squeue",), 0,
            f"9876|smoke-h100--attempt-009|{token}\n",
        ),
        CommandResult(("sacct",), 0, "9876|smoke-h100--attempt-009\n"),
    ])
    set_command_runner(fake)
    try:
        assert reconcile_submission(campaign, run, "attempt-009") == "9876"
    finally:
        set_command_runner(SubprocessRunner())

    backend = json.loads(
        (tmp_path / "local/controller-test/smoke-h100/backend.json").read_text()
    )
    assert backend["backend_job_id"] == "9876"


def test_slurm_stage_honors_elf_source_bundle_required_paths(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    fake = QueueRunner([
        CommandResult(("mkdir",), 0),
        CommandResult(("source-marker",), 0),
        CommandResult(("sif-marker",), 0),
        CommandResult(("required-path",), 0),
        CommandResult(("required-path",), 0),
    ])
    set_command_runner(fake)
    try:
        bundle = ElfProjectAdapter().source_bundle(repo_root)
        assert WydSlurmBackend(backend_services()).stage(
            campaign, run, "source-fixed", bundle
        ) is True
    finally:
        set_command_runner(SubprocessRunner())

    required = [" ".join(command) for command in fake.commands[-2:]]
    assert "scripts/cloud_train.sh" in required[0]
    assert "src/train.py" in required[1]


def test_package_process_evidence_feeds_elf_failure_classification(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1760")
    run_dir = run["storage"]["run_dir"]
    fake = QueueRunner([
        CommandResult(("collect-rsync",), 0),
        CommandResult(("checkpoint-probe",), 0),
        CommandResult(("stdout",), 1),
        CommandResult(
            ("stderr",), 0,
            f"{run_dir}/slurm-1760.err\n"
            "ModuleNotFoundError: No module named 'experiment_control'\n",
        ),
    ])
    service_bundle = backend_services()
    injected = type(service_bundle)(
        fake.run,
        service_bundle.local_run_dir,
        service_bundle.backend_record,
        lambda _campaign, _path: {"run_id": run["run_id"]},
        service_bundle.parse_metric,
        service_bundle.parse_checkpoint,
        service_bundle.atomic_write,
        service_bundle.utc_now,
    )
    summary = WydSlurmBackend(injected).collect(campaign, run)

    annotated = annotate_collection(summary, {"state": "FAILED"})
    assert annotated["failure_class"] == "configuration"
