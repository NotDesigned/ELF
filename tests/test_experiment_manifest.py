import json
import subprocess
import sys
from pathlib import Path

import yaml
import pytest

from elf_experiments.manifest import ExperimentStateStore, RunState, sanitize_command


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "experiment_manifest.py"
CONFIG = REPO_ROOT / "src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf_len256.yml"


def prepare_args(run_dir: Path, attempt_id: str = "attempt-001") -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--project",
        "elf",
        "--run-id",
        "test-run-s0",
        "--attempt-id",
        attempt_id,
        "--backend",
        "sensecore",
        "--backend-job-id",
        "job-123",
        "--config",
        str(CONFIG),
        "--config-override",
        f"output_dir={run_dir}",
        "--output-dir",
        str(run_dir),
        "--source-id",
        "source-deadbeef",
        "--image-id",
        "elf@sha256-deadbeef",
        "--gpus",
        "4",
        "--quota",
        "spot",
        "--require-immutable-identities",
        "--",
        "bash",
        "scripts/launch.sh",
        "train",
        str(CONFIG),
    ]


def test_legacy_control_records_are_observable_but_not_mutable(tmp_path):
    run_dir = tmp_path / "legacy-run"
    attempt_dir = run_dir / "attempts" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    legacy = {
        "schema_version": 1,
        "project": "elf",
        "run_id": "legacy-run",
        "attempt_id": "attempt-001",
        "created_at": "2026-07-11T00:00:00Z",
        "backend": {"kind": "slurm"},
    }
    (run_dir / "control_manifest.yaml").write_text(yaml.safe_dump(legacy))
    (attempt_dir / "control_attempt.yaml").write_text(yaml.safe_dump(legacy))
    store = ExperimentStateStore(run_dir)

    assert store.load_manifest()["run_id"] == "legacy-run"
    assert store.load_attempt("attempt-001")["attempt_id"] == "attempt-001"
    assert store.read_status("attempt-001").state == RunState.CREATED
    assert not store.manifest_path.exists()
    assert not store.attempt_path("attempt-001").exists()

    with pytest.raises(ValueError, match="observation-only"):
        store.ensure_manifest(legacy)


def test_prepare_writes_durable_run_and_attempt_records(tmp_path):
    run_dir = tmp_path / "run"
    subprocess.run(prepare_args(run_dir), cwd=REPO_ROOT, check=True, capture_output=True)

    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text(encoding="utf-8"))
    attempt = yaml.safe_load(
        (run_dir / "attempts/attempt-001/attempt.yaml").read_text(encoding="utf-8")
    )
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]

    assert manifest["run_id"] == "test-run-s0"
    assert manifest["resolved_config"]["output_dir"] == str(run_dir)
    assert manifest["seed"] == 42
    assert manifest["identity_version"] == 2
    assert manifest["backend"]["kind"] == "sensecore"
    assert manifest["resources"]["gpus"] == 4
    assert manifest["command"][-1] == str(CONFIG)
    assert manifest["assets"]
    assert "save_freq" in manifest["checkpoint"]
    assert attempt["attempt_id"] == "attempt-001"
    assert attempt["resources"]["quota"] == "spot"
    assert status["state"] == "CREATED"
    assert events[-1]["event"] == "attempt_created"


def test_prepare_refuses_attempt_overwrite_and_accepts_new_attempt(tmp_path):
    run_dir = tmp_path / "run"
    subprocess.run(prepare_args(run_dir), cwd=REPO_ROOT, check=True, capture_output=True)

    duplicate = subprocess.run(
        prepare_args(run_dir), cwd=REPO_ROOT, text=True, capture_output=True
    )
    assert duplicate.returncode != 0
    assert "attempt already exists" in duplicate.stderr

    subprocess.run(
        prepare_args(run_dir, "attempt-002"), cwd=REPO_ROOT, check=True, capture_output=True
    )
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 2


def test_prepare_requires_immutable_identities(tmp_path):
    args = prepare_args(tmp_path / "run")
    args[args.index("source-deadbeef")] = "unknown"
    result = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True)
    assert result.returncode != 0
    assert "source_id must be an immutable" in result.stderr


def test_record_updates_status_and_appends_lifecycle_event(tmp_path):
    run_dir = tmp_path / "run"
    subprocess.run(prepare_args(run_dir), cwd=REPO_ROOT, check=True, capture_output=True)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "record",
            "--project",
            "elf",
            "--run-id",
            "test-run-s0",
            "--attempt-id",
            "attempt-001",
            "--output-dir",
            str(run_dir),
            "--state",
            "SUCCEEDED",
            "--event",
            "process_exited",
            "--exit-code",
            "0",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert status["state"] == "SUCCEEDED"
    assert status["exit_code"] == 0
    assert events[-1]["event"] == "process_exited"
    assert events[-1]["payload"]["exit_code"] == 0


def test_manifest_command_redacts_secret_values():
    assert sanitize_command(
        ["train", "WANDB_API_KEY=secret-value", "--access-token", "secret-value", "seed=42"]
    ) == ["train", "WANDB_API_KEY=<redacted>", "--access-token", "<redacted>", "seed=42"]
