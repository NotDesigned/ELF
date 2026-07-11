import json
import subprocess
import sys
from pathlib import Path

import pytest

from elf_experiments.manifest import ExperimentStateStore, RunState


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "experiment_manifest.py"
CONFIG = REPO_ROOT / "src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf_len256.yml"


def prepare(
    run_dir: Path,
    *,
    attempt_id: str = "attempt-001",
    resume_from: str | None = None,
) -> ExperimentStateStore:
    overrides = [f"output_dir={run_dir}"]
    if resume_from:
        overrides.append(f"resume={resume_from}")
    override_args = [item for override in overrides for item in ("--config-override", override)]
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--project",
            "elf",
            "--run-id",
            "store-test-s0",
            "--attempt-id",
            attempt_id,
            "--backend",
            "slurm",
            "--config",
            str(CONFIG),
            *override_args,
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


def test_attempt_records_are_canonical_and_root_is_only_current_mirror(tmp_path):
    run_dir = tmp_path / "run"
    store = prepare(run_dir)
    store.begin_submission(
        project="elf", run_id="store-test-s0", attempt_id="attempt-001",
        backend="slurm", request={"argv": ["sbatch", "first.sbatch"]},
    )
    store.reconcile_submission(
        project="elf", run_id="store-test-s0", attempt_id="attempt-001",
        backend_job_id="12345",
    )

    checkpoint = str(run_dir / "checkpoint_10")
    prepare(run_dir, attempt_id="attempt-002", resume_from=checkpoint)

    first_backend = json.loads(store.attempt_backend_path("attempt-001").read_text())
    first_status = json.loads(store.attempt_status_path("attempt-001").read_text())
    current_backend = json.loads(store.backend_path.read_text())
    current_status = json.loads(store.status_path.read_text())
    second_attempt = store.load_attempt("attempt-002")

    assert first_backend["backend_job_id"] == "12345"
    assert first_status["state"] == "QUEUED"
    assert current_backend == json.loads(
        store.attempt_backend_path("attempt-002").read_text()
    )
    assert current_backend["attempt_id"] == "attempt-002"
    assert current_backend["backend_job_id"] is None
    assert current_status["attempt_id"] == "attempt-002"
    assert current_status["state"] == "CREATED"
    assert second_attempt["resume_from"] == checkpoint
    assert "resume" not in store.load_manifest()["resolved_config"]

    # A late observation for an older attempt updates its canonical record but
    # cannot move the run-level current-attempt mirror backwards.
    store.transition(
        project="elf", run_id="store-test-s0", attempt_id="attempt-001",
        state=RunState.FAILED, event="late_terminal_observation",
    )
    assert store.read_status("attempt-001").state == RunState.FAILED
    assert store.read_status("attempt-002").state == RunState.CREATED
    assert json.loads(store.status_path.read_text())["attempt_id"] == "attempt-002"


def test_new_attempt_fails_closed_when_root_mirror_drifted(tmp_path):
    run_dir = tmp_path / "run"
    prepare(run_dir, attempt_id="attempt-001")
    store = ExperimentStateStore(run_dir)
    drifted = json.loads(store.status_path.read_text())
    drifted["state"] = "RUNNING"
    store.status_path.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError) as captured:
        prepare(run_dir, attempt_id="attempt-002")
    assert b"root mirror conflicts" in captured.value.stderr
