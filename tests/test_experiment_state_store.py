import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_manifest import ExperimentStateStore, RunState


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "experiment_manifest.py"
CONFIG = REPO_ROOT / "src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf_len256.yml"


def prepare(run_dir: Path) -> ExperimentStateStore:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--project",
            "elf",
            "--run-id",
            "store-test-s0",
            "--attempt-id",
            "attempt-001",
            "--backend",
            "slurm",
            "--config",
            str(CONFIG),
            "--config-override",
            f"output_dir={run_dir}",
            "--output-dir",
            str(run_dir),
            "--source-id",
            "source-deadbeef",
            "--image-id",
            "sif-sha256-deadbeef",
            "--gpus",
            "1",
            "--require-immutable-identities",
            "--",
            "bash",
            "scripts/launch.sh",
            "train",
            str(CONFIG),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return ExperimentStateStore(run_dir)


def events(store: ExperimentStateStore) -> list[dict]:
    return [json.loads(line) for line in store.events_path.read_text().splitlines()]


def test_read_status_distinguishes_not_submitted_and_created(tmp_path):
    empty = ExperimentStateStore(tmp_path / "empty")
    assert empty.read_status().state == RunState.NOT_SUBMITTED

    store = prepare(tmp_path / "run")
    store.status_path.unlink()
    assert store.read_status("attempt-001").state == RunState.CREATED
    assert store.read_status("attempt-999").state == RunState.NOT_SUBMITTED


def test_submission_intent_is_idempotent_and_redacted(tmp_path):
    store = prepare(tmp_path / "run")
    request = {
        "argv": ["sbatch", "run.sbatch"],
        "environment": {"WANDB_API_KEY": "must-not-persist", "SEED": "42"},
        "callback": "https://user:password@example.invalid/status",
    }

    first = store.begin_submission(
        project="elf",
        run_id="store-test-s0",
        attempt_id="attempt-001",
        backend="slurm",
        request=request,
    )
    second = store.begin_submission(
        project="elf",
        run_id="store-test-s0",
        attempt_id="attempt-001",
        backend="slurm",
        request=request,
    )

    assert first == second
    assert first["request"]["environment"]["WANDB_API_KEY"] == "<redacted>"
    assert first["request"]["callback"] == "https://<redacted>@example.invalid/status"
    assert store.read_status().state == RunState.SUBMITTING
    assert [event["event"] for event in events(store)].count("submission_intent_created") == 1


def test_repeating_intent_repairs_derived_state_after_crash(tmp_path):
    store = prepare(tmp_path / "run")
    kwargs = {
        "project": "elf",
        "run_id": "store-test-s0",
        "attempt_id": "attempt-001",
        "backend": "slurm",
        "request": {"argv": ["sbatch", "run.sbatch"]},
    }
    store.begin_submission(**kwargs)

    # Simulate a process dying after the durable intent write but before all
    # Derived read-model files were updated idempotently.
    store.status_path.unlink()
    store.backend_path.unlink()
    store.events_path.write_text(
        "\n".join(
            json.dumps(event)
            for event in events(store)
            if event["event"] != "submission_intent_created"
        )
        + "\n"
    )

    store.begin_submission(**kwargs)
    assert store.read_status().state == RunState.SUBMITTING
    assert json.loads(store.backend_path.read_text())["backend"] == "slurm"
    assert [event["event"] for event in events(store)].count("submission_intent_created") == 1


def test_reconcile_is_idempotent_and_rejects_a_different_job(tmp_path):
    store = prepare(tmp_path / "run")
    store.begin_submission(
        project="elf",
        run_id="store-test-s0",
        attempt_id="attempt-001",
        backend="slurm",
        request={"argv": ["sbatch", "run.sbatch"]},
    )
    kwargs = {
        "project": "elf",
        "run_id": "store-test-s0",
        "attempt_id": "attempt-001",
        "backend_job_id": "12345",
    }
    first = store.reconcile_submission(**kwargs)

    # Runtime initialization recovery must not regress a submitted attempt.
    store.initialize_attempt_records("attempt-001")
    assert store.read_status().state == RunState.QUEUED

    # Repeating this also repairs files lost after scheduler acceptance.
    store.status_path.unlink()
    store.backend_path.unlink()
    second = store.reconcile_submission(**kwargs)
    assert first == second
    assert store.read_status().state == RunState.QUEUED
    assert json.loads(store.backend_path.read_text())["backend_job_id"] == "12345"
    assert [event["event"] for event in events(store)].count("submission_reconciled") == 1

    with pytest.raises(ValueError, match="already reconciled"):
        store.reconcile_submission(**{**kwargs, "backend_job_id": "67890"})


def test_transition_supports_event_idempotency(tmp_path):
    store = prepare(tmp_path / "run")
    kwargs = {
        "project": "elf",
        "run_id": "store-test-s0",
        "attempt_id": "attempt-001",
        "state": RunState.RUNNING,
        "event": "worker_observed",
        "event_id": "worker-observed:attempt-001",
    }
    store.transition(**kwargs)
    store.transition(**kwargs)
    assert store.read_status().state == RunState.RUNNING
    assert [event["event"] for event in events(store)].count("worker_observed") == 1
