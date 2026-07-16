import json
import os
import subprocess
import sys
import shutil
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from elf_experiments import controller as experimentctl
from elf_experiments.controller import (
    annotate_collection,
    ensure_attempt_not_submitted,
    frozen_source_identity,
    load_campaign,
    materialize_run,
    prepare_run,
    reconcile_submission,
    record_submission,
    record_submission_intent,
    resolved_run_overrides,
    identity_report,
)
from experiment_control.identity import IdentityReport
from elf_experiments.manifest import prepare as runtime_prepare
from experiment_control.backends.wyd import render_job
from elf_experiments.projects.elf import parse_training_metric_line
from backend_fixtures import slurm_campaign as backend_slurm_campaign


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = "src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf_len256.yml"


class NeutralBackend:
    kind = "test-backend"

    def validate(self, _run):
        return None

    def environment(self, _campaign, _run, _source_id, _attempt_id):
        return {}

    def submission_request(self, _campaign, run, attempt_id):
        return {
            "backend": self.kind,
            "run_id": run["run_id"],
            "attempt_id": attempt_id,
        }


class ControllerBackendRegistry:
    def __init__(self, real_registry):
        self.real_registry = real_registry
        self.kinds = frozenset({*real_registry.kinds, "test-backend"})

    def get(self, kind):
        if kind == "test-backend":
            return NeutralBackend()
        return self.real_registry.get(kind)


@pytest.fixture(autouse=True)
def register_neutral_controller_backend(monkeypatch):
    monkeypatch.setattr(
        experimentctl, "BACKENDS", ControllerBackendRegistry(experimentctl.BACKENDS)
    )


def test_preflight_cli_returns_nonzero_when_a_required_tool_is_unavailable(tmp_path):
    campaign = backend_slurm_campaign(tmp_path)
    campaign_path = tmp_path / "backend-campaign.yml"
    campaign_path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    env = os.environ.copy()
    env["EXPERIMENTCTL_SSH_BIN"] = "/bin/false"
    result = subprocess.run(
        [
            sys.executable, "tools/experimentctl.py",
            str(campaign_path), "preflight", "--run", "smoke-h100",
        ],
        cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 1
    assert '"ready": false' in result.stdout


def test_cli_formats_expected_operational_error_without_traceback(monkeypatch, capsys):
    def fail(_argv):
        raise FileExistsError("identity consumed")

    monkeypatch.setattr(experimentctl, "main", fail)
    assert experimentctl.cli([]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error == {"error": "FileExistsError", "message": "identity consumed"}


def local_evidence_fixture(tmp_path: Path):
    """Create separate reviewed identity copies and original local data."""
    campaign_id = "local-evidence-study"
    run_id = "run-local"
    attempt_id = "attempt-001"
    local_root = tmp_path / "local-root"
    attempt_dir = local_root / campaign_id / run_id / "attempts" / attempt_id
    collected = attempt_dir / "collected_run"
    collected.mkdir(parents=True)
    campaign = {
        "schema_version": 1, "project": "elf", "campaign": campaign_id,
        "runs": [{"run_id": run_id}],
    }
    identity = tmp_path / "private" / "inputs"
    (identity / "run").mkdir(parents=True)
    (identity / "attempt").mkdir(parents=True)
    campaign_path = identity / "campaign.yml"
    campaign_path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    run_manifest = {
        "schema_version": 1, "project": "elf", "campaign": campaign_id,
        "run_id": run_id, "source_id": "source-fixed", "image_id": "image-fixed",
        "resolved_config": {"seed": 7, "max_length": 256},
    }
    (identity / "run" / "manifest.yaml").write_text(
        yaml.safe_dump(run_manifest), encoding="utf-8",
    )
    attempt = {
        **run_manifest, "attempt_id": attempt_id,
    }
    (identity / "attempt" / "attempt.yaml").write_text(
        yaml.safe_dump(attempt), encoding="utf-8",
    )
    backend = {
        "project": "elf", "run_id": run_id, "attempt_id": attempt_id,
        "backend": "slurm", "backend_job_id": "123",
    }
    (identity / "attempt" / "backend.json").write_text(
        json.dumps(backend), encoding="utf-8",
    )
    previous = {
        "project": "elf", "run_id": run_id, "attempt_id": attempt_id,
        "state": "SUCCEEDED", "train_loss": 9.0,
    }
    collection = attempt_dir / "collection.json"
    collection.write_text(json.dumps(previous) + "\n", encoding="utf-8")
    (identity / "attempt" / "collection.json").write_bytes(collection.read_bytes())
    (collected / "manifest.yaml").write_text(
        yaml.safe_dump(run_manifest), encoding="utf-8",
    )
    (collected / "status.json").write_text(
        json.dumps({"state": "SUCCEEDED", "attempt_id": attempt_id}) + "\n",
        encoding="utf-8",
    )
    (collected / "backend.json").write_text(
        json.dumps(backend) + "\n", encoding="utf-8",
    )
    (collected / "train_metrics.jsonl").write_text(
        json.dumps({"step": 4, "train_loss": 1.25}) + "\n",
        encoding="utf-8",
    )
    (attempt_dir / "attempt.yaml").write_text(
        yaml.safe_dump(attempt), encoding="utf-8",
    )
    (attempt_dir / "backend.json").write_text(
        json.dumps(backend) + "\n", encoding="utf-8",
    )
    (attempt_dir / "status.json").write_text(
        json.dumps({"state": "SUCCEEDED", "attempt_id": attempt_id}) + "\n",
        encoding="utf-8",
    )
    (attempt_dir / "events.jsonl").write_text(
        json.dumps({"event": "finished", "attempt_id": attempt_id}) + "\n",
        encoding="utf-8",
    )
    arguments = [
        str(campaign_path), "refresh-evidence-local", "--run", run_id,
        "--attempt-id", attempt_id, "--local-root", str(local_root),
        "--identity-root", str(identity),
    ]
    return arguments, identity, attempt_dir, collection


def test_refresh_evidence_local_has_no_backend_or_command_boundary(
    tmp_path, monkeypatch, capsys,
):
    arguments, _identity, attempt_dir, collection = local_evidence_fixture(tmp_path)
    snapshot_digest = "sha256:" + "d" * 64
    monkeypatch.setenv("ML_EXPD_CONTROLLER_SNAPSHOT_SHA256", snapshot_digest)

    class ForbiddenBackends:
        @property
        def kinds(self):
            raise AssertionError("backend registry must not be inspected")

        def get(self, _kind):
            raise AssertionError("backend adapter must not be selected")

    class ForbiddenRunner:
        def run(self, *args, **kwargs):
            raise AssertionError("scheduler/local command runner must not be called")

    monkeypatch.setattr(experimentctl, "BACKENDS", ForbiddenBackends())
    monkeypatch.setattr(experimentctl, "_COMMAND_RUNNER", ForbiddenRunner())
    immutable_paths = [
        attempt_dir / "attempt.yaml", attempt_dir / "backend.json",
        attempt_dir / "status.json", attempt_dir / "events.jsonl",
        *(path for path in (attempt_dir / "collected_run").rglob("*") if path.is_file()),
    ]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in immutable_paths}

    assert experimentctl.main([*arguments, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)[0]
    assert preview["dry_run"] is True
    assert preview["local_only"] is True
    assert preview["backend_accessed"] is False
    assert preview["scheduler_accessed"] is False
    assert preview["controller_snapshot_sha256"] == snapshot_digest
    assert preview["expected_new_collection_digest"] == preview["new_digest"]
    assert preview["atomic_collection_replace"] is True
    assert preview["write_protocol"] == "dirfd-fsync-rename-v1"
    old_bytes = collection.read_bytes()

    assert experimentctl.main([
        *arguments, "--expected-input-digest", preview["input_digest"],
    ]) == 0
    result = json.loads(capsys.readouterr().out)[0]

    assert result["old_digest"] == preview["old_digest"]
    assert result["new_digest"] == experimentctl._regular_file_digest(collection)
    assert (
        result["expected_new_collection_digest"]
        == preview["expected_new_collection_digest"]
        == result["new_digest"]
    )
    assert result["new_digest"] != result["old_digest"]
    assert collection.read_bytes() != old_bytes
    rebuilt = json.loads(collection.read_text(encoding="utf-8"))
    assert rebuilt["project"] == "elf"
    assert rebuilt["run_id"] == "run-local"
    assert rebuilt["attempt_id"] == "attempt-001"
    assert rebuilt["train_loss"] == 1.25
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in immutable_paths} == before


def test_refresh_evidence_local_keeps_exact_execution_and_checkpoint_evidence(
    tmp_path, monkeypatch, capsys,
):
    arguments, identity, _attempt_dir, collection = local_evidence_fixture(tmp_path)
    previous = {
        "project": "elf",
        "run_id": "run-local",
        "attempt_id": "attempt-001",
        "source_id": "source-fixed",
        "image_id": "image-fixed",
        "state": "SUCCEEDED",
        "backend": "slurm",
        "run_dir": "/remote/run-local",
        "collected_from": "/remote/run-local",
        "scheduler_state": "SUCCEEDED",
        "worker_state": "RELEASED",
        "process_state": "SUCCEEDED",
        "runtime_state": "SUCCEEDED",
        "model_state": "OBSERVED",
        "process_evidence": {
            "observed": True,
            "stdout_tail": ["Final checkpoint saved"],
            "stderr_tail": [],
        },
        "evidence_outcome": "OBSERVED",
        "evidence_unavailable_reason": None,
        "latest_completed_checkpoint": "/remote/run-local/checkpoint_38035",
        "latest_completed_checkpoint_step": 38035,
        "collection_provenance": {"backend_job_id": "123"},
        "train_loss": 9.0,
        "g_ppl": 30.0,
        "oracle_plan_ppl": 20.0,
        "shuffled_plan_ppl": 25.0,
        "token_recon_ppl": 10.0,
        "plan_ppl_gap": 5.0,
        "generation_mean_entropy": 4.5,
        "metric_evidence": {"g_ppl": {"step": 38035, "value": 30.0}},
        "evidence_conflicts": [{"metric": "g_ppl"}],
        "warnings": ["stale conflict"],
    }
    encoded = (json.dumps(previous, sort_keys=True) + "\n").encode("utf-8")
    collection.write_bytes(encoded)
    (identity / "attempt" / "collection.json").write_bytes(encoded)
    monkeypatch.setenv(
        "ML_EXPD_CONTROLLER_SNAPSHOT_SHA256", "sha256:" + "6" * 64,
    )

    assert experimentctl.main([*arguments, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)[0]
    assert experimentctl.main([
        *arguments, "--expected-input-digest", preview["input_digest"],
    ]) == 0
    capsys.readouterr()
    rebuilt = json.loads(collection.read_text(encoding="utf-8"))

    operational = (
        "state", "backend", "run_dir", "collected_from", "scheduler_state",
        "worker_state", "process_state", "runtime_state", "model_state",
        "process_evidence", "evidence_outcome", "evidence_unavailable_reason",
        "latest_completed_checkpoint", "latest_completed_checkpoint_step",
        "collection_provenance",
    )
    assert {key: rebuilt[key] for key in operational} == {
        key: previous[key] for key in operational
    }
    # These exact fields are the inputs used by ml-expd's execution-layer and
    # checkpoint gates, so the rebuilt collection remains fully observed.
    assert rebuilt["scheduler_state"] == "SUCCEEDED"
    assert rebuilt["process_state"] == "SUCCEEDED"
    assert rebuilt["model_state"] == "OBSERVED"
    assert rebuilt["latest_completed_checkpoint_step"] == 38035
    assert rebuilt["train_loss"] == 1.25
    assert "g_ppl" not in rebuilt
    assert "oracle_plan_ppl" not in rebuilt
    assert "shuffled_plan_ppl" not in rebuilt
    assert "token_recon_ppl" not in rebuilt
    assert "plan_ppl_gap" not in rebuilt
    assert "generation_mean_entropy" not in rebuilt
    assert "evidence_conflicts" not in rebuilt
    assert "warnings" not in rebuilt


def test_refresh_evidence_local_uses_reviewed_recovery_baseline_with_live_cas(
    tmp_path, monkeypatch, capsys,
):
    arguments, identity, _attempt_dir, collection = local_evidence_fixture(tmp_path)
    baseline = {
        "project": "elf", "run_id": "run-local", "attempt_id": "attempt-001",
        "state": "SUCCEEDED", "backend": "slurm",
        "scheduler_state": "SUCCEEDED", "worker_state": "RELEASED",
        "process_state": "SUCCEEDED", "runtime_state": "SUCCEEDED",
        "model_state": "OBSERVED", "evidence_outcome": "OBSERVED",
        "latest_completed_checkpoint": "/remote/checkpoint_38035",
        "latest_completed_checkpoint_step": 38035,
        # Old science is intentionally untrusted during recovery.
        "train_loss": 99.0, "g_ppl": 88.0,
        "evidence_conflicts": [{"metric": "g_ppl"}],
    }
    baseline_bytes = (json.dumps(baseline, sort_keys=True) + "\n").encode()
    (identity / "attempt" / "collection.json").write_bytes(baseline_bytes)
    damaged = {
        "project": "elf", "run_id": "run-local", "attempt_id": "attempt-001",
        "state": "SUCCEEDED", "backend": "slurm",
        "train_loss": 2.0,
        "evaluation_family_state": "UNRESOLVED",
    }
    collection.write_text(json.dumps(damaged, sort_keys=True) + "\n")
    current_digest = experimentctl._regular_file_digest(collection)
    arguments.extend(["--expected-current-collection-digest", current_digest])
    monkeypatch.setenv(
        "ML_EXPD_CONTROLLER_SNAPSHOT_SHA256", "sha256:" + "5" * 64,
    )

    assert experimentctl.main([*arguments, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)[0]
    assert preview["old_digest"] == current_digest
    assert preview["recovery_baseline_digest"] == experimentctl._sha256_bytes(
        baseline_bytes
    )
    assert experimentctl.main([
        *arguments, "--expected-input-digest", preview["input_digest"],
    ]) == 0
    capsys.readouterr()

    rebuilt = json.loads(collection.read_text())
    assert rebuilt["scheduler_state"] == "SUCCEEDED"
    assert rebuilt["process_state"] == "SUCCEEDED"
    assert rebuilt["model_state"] == "OBSERVED"
    assert rebuilt["latest_completed_checkpoint_step"] == 38035
    assert rebuilt["train_loss"] == 1.25
    assert "g_ppl" not in rebuilt
    assert "evidence_conflicts" not in rebuilt


def test_controller_import_does_not_construct_backend_registry(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        str(REPO_ROOT / "src"),
        str(REPO_ROOT.parent / "ml-experiment-control-local-evidence" / "src"),
        environment.get("PYTHONPATH", ""),
    ))
    code = """
import experiment_control.backends as backends
def forbidden(*args, **kwargs):
    raise AssertionError('backend registry constructed during controller import')
backends.build_registry = forbidden
import elf_experiments.controller as controller
assert controller.BACKENDS._registry is None
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=environment,
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_refresh_evidence_local_survives_attempt_directory_rename_swap(
    tmp_path, monkeypatch, capsys,
):
    arguments, _identity, attempt_dir, collection = local_evidence_fixture(tmp_path)
    monkeypatch.setenv(
        "ML_EXPD_CONTROLLER_SNAPSHOT_SHA256", "sha256:" + "a" * 64,
    )
    assert experimentctl.main([*arguments, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)[0]
    original_projects = experimentctl.PROJECTS
    moved_attempt = attempt_dir.with_name(attempt_dir.name + "-anchored")
    replacement_payload = b'{"attacker":"replacement"}\n'

    class SwapAdapter:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def summarize(self, run_dir):
            attempt_dir.rename(moved_attempt)
            attempt_dir.mkdir()
            (attempt_dir / "collection.json").write_bytes(replacement_payload)
            return self.wrapped.summarize(run_dir)

    class SwapProjects:
        def get(self, project):
            return SwapAdapter(original_projects.get(project))

    monkeypatch.setattr(experimentctl, "PROJECTS", SwapProjects())
    assert experimentctl.main([
        *arguments, "--expected-input-digest", preview["input_digest"],
    ]) == 0
    result = json.loads(capsys.readouterr().out)[0]

    assert collection.read_bytes() == replacement_payload
    anchored_collection = moved_attempt / "collection.json"
    assert experimentctl._regular_file_digest(anchored_collection) == result["new_digest"]
    assert result["new_digest"] == preview["expected_new_collection_digest"]


def test_refresh_evidence_local_is_deterministic_across_independent_processes(
    tmp_path,
):
    arguments, _identity, attempt_dir, collection = local_evidence_fixture(tmp_path)
    collected = attempt_dir / "collected_run"
    evaluation = collected / "train_sampling_eval" / "generation"
    evaluation.mkdir(parents=True)
    dimensions = {
        "sampling_method": "sde", "num_sampling_steps": 32, "cfg": 1.0,
        "self_cond_cfg_scale": 3.0, "time_schedule": "logit_normal",
        "time_warp_gamma": 1.5,
    }
    metrics = evaluation / "metrics.jsonl"
    metrics.write_text(json.dumps({
        "epoch": 1, "step": 4, "mode": "generation_refine_decode",
        "g_ppl": 31.0, "sampling_config": dimensions,
    }) + "\n", encoding="utf-8")
    reviewed_mtime_ns = 1_784_029_337_104_056_800
    os.utime(metrics, ns=(reviewed_mtime_ns, reviewed_mtime_ns))
    checkpoint = collected / "checkpoint_4"
    checkpoint.write_bytes(b"stable checkpoint")
    (collected / "checkpoint_4.complete").write_text(json.dumps({
        "step": 4, "bytes": checkpoint.stat().st_size,
        "completed_at": "2026-07-14T00:00:00Z",
    }) + "\n", encoding="utf-8")
    snapshot_digest = "sha256:" + "9" * 64
    environment = os.environ.copy()
    environment["ML_EXPD_CONTROLLER_SNAPSHOT_SHA256"] = snapshot_digest
    environment["PYTHONPATH"] = os.pathsep.join((
        str(REPO_ROOT / "src"), environment.get("PYTHONPATH", ""),
    ))

    def invoke(extra: list[str]) -> dict:
        completed = subprocess.run(
            [sys.executable, "tools/experimentctl.py", *arguments, *extra],
            cwd=REPO_ROOT, env=environment, text=True, capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert isinstance(payload, list) and len(payload) == 1
        return payload[0]

    first = invoke(["--dry-run"])
    second = invoke(["--dry-run"])

    assert second == first
    assert first["new_digest"] == first["expected_new_collection_digest"]
    result = invoke([
        "--expected-input-digest", first["input_digest"],
    ])
    assert result["new_digest"] == first["expected_new_collection_digest"]
    assert experimentctl._regular_file_digest(collection) == result["new_digest"]
    rebuilt = json.loads(collection.read_text(encoding="utf-8"))
    # An existing collection owns operational and checkpoint provenance.  A
    # scientific refresh cannot manufacture either from its local snapshot.
    assert "run_dir" not in rebuilt
    assert "latest_completed_checkpoint" not in rebuilt
    assert "latest_completed_checkpoint_step" not in rebuilt
    observations = rebuilt["metric_evidence"]["observations"]
    assert {item["observed_at"] for item in observations} == {
        reviewed_mtime_ns / 1_000_000_000,
    }

    def strings(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield str(key)
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    assert all(
        "/proc/self/fd/" not in value and "elf-local-evidence-" not in value
        for value in strings(rebuilt)
    )


def test_refresh_evidence_local_rejects_collection_cas_and_artifact_drift(
    tmp_path, monkeypatch, capsys,
):
    arguments, identity, attempt_dir, collection = local_evidence_fixture(tmp_path)
    monkeypatch.setenv(
        "ML_EXPD_CONTROLLER_SNAPSHOT_SHA256", "sha256:" + "e" * 64,
    )
    assert experimentctl.main([*arguments, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)[0]
    collection.write_text('{"raced":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed after private review"):
        experimentctl.main([
            *arguments, "--expected-input-digest", preview["input_digest"],
        ])

    # Restore reviewed collection, then drift one durable artifact.
    collection.write_bytes((identity / "attempt" / "collection.json").read_bytes())
    (attempt_dir / "collected_run" / "train_metrics.jsonl").write_text(
        json.dumps({"step": 5, "train_loss": 0.5}) + "\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="input digest changed"):
        experimentctl.main([
            *arguments, "--expected-input-digest", preview["input_digest"],
        ])


def test_refresh_evidence_local_rejects_wrong_identity_and_nonterminal_summary(
    tmp_path, monkeypatch,
):
    arguments, identity, attempt_dir, _collection = local_evidence_fixture(tmp_path)
    monkeypatch.setenv(
        "ML_EXPD_CONTROLLER_SNAPSHOT_SHA256", "sha256:" + "f" * 64,
    )
    backend = json.loads((identity / "attempt" / "backend.json").read_text())
    backend["attempt_id"] = "attempt-999"
    (identity / "attempt" / "backend.json").write_text(json.dumps(backend))
    with pytest.raises(ValueError, match="attempt_id conflicts"):
        experimentctl.main([*arguments, "--dry-run"])

    backend["attempt_id"] = "attempt-001"
    (identity / "attempt" / "backend.json").write_text(json.dumps(backend))
    (attempt_dir / "collected_run" / "status.json").write_text(
        json.dumps({"state": "RUNNING", "attempt_id": "attempt-001"}) + "\n"
    )
    with pytest.raises(ValueError, match="terminal Attempt"):
        experimentctl.main([*arguments, "--dry-run"])


def test_refresh_evidence_local_accepts_minimal_backend_contract_and_rejects_spoof(
    tmp_path, monkeypatch, capsys,
):
    arguments, identity, _attempt_dir, _collection = local_evidence_fixture(tmp_path)
    monkeypatch.setenv(
        "ML_EXPD_CONTROLLER_SNAPSHOT_SHA256", "sha256:" + "7" * 64,
    )
    backend_path = identity / "attempt" / "backend.json"
    minimal_backend = {
        "attempt_id": "attempt-001", "backend": "slurm",
        "backend_job_id": "7433",
    }
    backend_path.write_text(json.dumps(minimal_backend) + "\n", encoding="utf-8")

    assert experimentctl.main([*arguments, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)[0]
    assert preview["local_only"] is True
    assert preview["attempt_id"] == "attempt-001"

    backend_path.write_text(json.dumps({
        **minimal_backend, "project": "spoofed",
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="backend project conflicts"):
        experimentctl.main([*arguments, "--dry-run"])


def test_refresh_evidence_local_is_content_idempotent_for_a_fresh_review(
    tmp_path, monkeypatch, capsys,
):
    arguments, identity, _attempt_dir, collection = local_evidence_fixture(tmp_path)
    monkeypatch.setenv(
        "ML_EXPD_CONTROLLER_SNAPSHOT_SHA256", "sha256:" + "1" * 64,
    )
    assert experimentctl.main([*arguments, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)[0]
    assert experimentctl.main([
        *arguments, "--expected-input-digest", preview["input_digest"],
    ]) == 0
    first = json.loads(capsys.readouterr().out)[0]
    first_bytes = collection.read_bytes()

    # A later Action snapshots the new collection preimage and recomputes it.
    (identity / "attempt" / "collection.json").write_bytes(first_bytes)
    assert experimentctl.main([*arguments, "--dry-run"]) == 0
    second_preview = json.loads(capsys.readouterr().out)[0]
    assert experimentctl.main([
        *arguments, "--expected-input-digest", second_preview["input_digest"],
    ]) == 0
    second = json.loads(capsys.readouterr().out)[0]

    assert collection.read_bytes() == first_bytes
    assert second["old_digest"] == first["new_digest"]
    assert second["new_digest"] == first["new_digest"]


def test_refresh_evidence_local_rejects_missing_or_linked_durable_evidence(
    tmp_path, monkeypatch,
):
    arguments, _identity, attempt_dir, _collection = local_evidence_fixture(tmp_path)
    monkeypatch.setenv(
        "ML_EXPD_CONTROLLER_SNAPSHOT_SHA256", "sha256:" + "2" * 64,
    )
    manifest = attempt_dir / "collected_run" / "manifest.yaml"
    manifest.unlink()
    with pytest.raises(FileNotFoundError, match="manifest.yaml"):
        experimentctl.main([*arguments, "--dry-run"])

    outside = tmp_path / "outside.txt"
    outside.write_text("not reviewed local evidence\n", encoding="utf-8")
    (attempt_dir / "collected_run" / "linked.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="rejects symlink"):
        experimentctl.main([*arguments, "--dry-run"])


def controller_campaign(tmp_path: Path) -> dict:
    """Return a deployment-neutral campaign for controller tests."""
    return {
        "schema_version": 1,
        "campaign": "controller-test",
        "project": "elf",
        "source_id": "source-fixed",
        "local_root": str(tmp_path / "local"),
        "runs": [
            {
                "run_id": "controller-run",
                "config": CONFIG,
                "config_overrides": ["epochs=1", "save_freq=0.1"],
                "image_id": "sha256:" + "a" * 64,
                "resources": {"gpus": 1, "cpus": 8},
                "storage": {
                    "run_dir": "/mnt/test-project/runs/controller-run",
                    "data_root": "/mnt",
                    "project_data_root": "/mnt/test-project",
                    "hf_home": "/mnt/test-project/cache/huggingface",
                    "hf_datasets_cache": "/mnt/test-project/cache/huggingface/datasets",
                },
                "env": {"BATCH_SIZE": "4", "LOG_FREQ": "10"},
                "backend": {
                    "kind": "test-backend",
                    "source_dir": "/mnt/test-project/sources/{source_id}",
                },
            }
        ],
    }


def test_prepare_and_render_preserve_explicit_partition(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = backend_slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    manifest = prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    script = render_job(manifest)

    assert "#SBATCH --partition=h100" in script
    assert "#SBATCH --gres=gpu:h100:1" in script
    assert "#SBATCH --job-name=smoke-h100--attempt-001" in script
    assert "#SBATCH --output=/dev/null" in script
    assert "attempts/attempt-001" in script
    assert "--bind /mnt/test-project/sources/source-fixed:/app" in script
    assert 'export BACKEND_JOB_ID="$SLURM_JOB_ID"' in script
    assert "WANDB_DIR=/mnt/test-project/wandb" in script
    assert "CHECKPOINT_ROOT=/mnt/test-project/checkpoints" in script
    assert manifest["resolved_config"]["epochs"] == 1
    assert manifest["resolved_config"]["save_freq"] == 0.1
    assert manifest["resolved_config"]["global_batch_size"] is None
    assert manifest["resolved_config"]["batch_size"] == 4
    assert manifest["resolved_config"]["log_freq"] == 10


def test_submit_dry_run_reports_local_artifacts_and_next_gates(tmp_path):
    campaign = backend_slurm_campaign(tmp_path)
    path = tmp_path / "campaign.yml"
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable, "tools/experimentctl.py", str(path), "submit",
            "--run", "smoke-h100", "--attempt-id", "attempt-001", "--dry-run",
        ],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)[0]
    assert payload["scheduler_mutated"] is False
    assert payload["state"] == "CREATED"
    assert Path(payload["manifest_path"]).is_file()
    assert Path(payload["submission_preview_path"]).is_file()
    assert payload["next_gates"] == [
        "check-identity", "assets-verify", "stage", "submit",
    ]


def test_controller_manifest_is_accepted_unchanged_by_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    campaign.update({"git_commit": "commit", "campaign_id": "campaign-id"})
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    remote_dir = tmp_path / "remote-run"
    run["storage"]["run_dir"] = str(remote_dir)
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    local_manifest = tmp_path / "local/controller-test/controller-run/manifest.yaml"
    frozen = yaml.safe_load(local_manifest.read_text(encoding="utf-8"))
    assert frozen["git_commit"] == "commit"
    assert frozen["runtime_tree_id"] == "source-fixed"
    assert frozen["campaign_id"] == "campaign-id"
    assert frozen["image_id"] == run["image_id"]
    remote_dir.mkdir()
    shutil.copy2(local_manifest, remote_dir / "manifest.yaml")
    runtime_prepare(Namespace(
        project="elf", run_id="controller-run", attempt_id="attempt-001",
        backend="test-backend", backend_job_id="123", config=CONFIG,
        config_override=resolved_run_overrides(campaign, run, str(remote_dir)),
        output_dir=str(remote_dir), source_id="source-fixed", runtime_tree_id="source-fixed",
        git_commit="commit", campaign_id="campaign-id", campaign="controller-test",
        image_id=run["image_id"], gpus=1, nodes=1, quota="normal",
        resource_spec="", max_infra_retries=0, require_immutable_identities=True,
        command=["true"],
    ))
    assert (remote_dir / "attempts/attempt-001/attempt.yaml").is_file()


def test_prepared_run_keeps_frozen_provenance_when_submit_checkout_moves(tmp_path):
    campaign = controller_campaign(tmp_path)
    campaign.update({"git_commit": "commit-before", "campaign_id": "campaign-before"})
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepared = prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")

    moved = dict(campaign)
    moved.update({"git_commit": "commit-after", "campaign_id": "campaign-after"})
    submitted = prepare_run(moved, run, "source-fixed", attempt_id="attempt-001")

    assert submitted == prepared
    assert submitted["git_commit"] == "commit-before"
    assert submitted["campaign_id"] == "campaign-before"
    assert "GIT_COMMIT=commit-before" in submitted["command"]
    assert "CAMPAIGN_ID=campaign-before" in submitted["command"]


def test_render_supports_alternate_storage_mount(tmp_path, monkeypatch):
    """Jobs bind and cache on their explicitly declared filesystem."""
    monkeypatch.chdir(REPO_ROOT)
    campaign = backend_slurm_campaign(tmp_path)
    backend = campaign["runs"][0]["backend"]
    backend.update(
        {
            "mount_root": "/alternate",
            "apptainer_cache_dir": "/alternate/test-project/apptainer/cache",
            "apptainer_tmp_dir": "/alternate/test-project/apptainer/tmp",
            "source_dir": "/alternate/test-project/sources/{source_id}",
            "sif_path": "/alternate/test-project/images/test.sif",
        }
    )
    campaign["runs"][0]["storage"].update(
        {
            "run_dir": "/alternate/test-project/runs/smoke-h100",
            "data_root": "/alternate",
            "project_data_root": "/alternate/test-project",
            "hf_home": "/alternate/cache/huggingface",
            "hf_datasets_cache": "/alternate/cache/huggingface/datasets",
        }
    )
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    manifest = prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    script = render_job(manifest)

    assert "--bind /alternate:/alternate" in script
    assert "export APPTAINER_CACHEDIR=/alternate/test-project/apptainer/cache" in script
    assert "export APPTAINER_TMPDIR=/alternate/test-project/apptainer/tmp" in script


def test_rejects_relative_slurm_mount_root(tmp_path):
    campaign = backend_slurm_campaign(tmp_path)
    campaign["runs"][0]["backend"]["mount_root"] = "alternate"
    path = tmp_path / "campaign.yml"
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="mount_root must be an absolute path"):
        load_campaign(path)


def test_rejects_mixed_slurm_storage_profiles(tmp_path):
    campaign = backend_slurm_campaign(tmp_path)
    campaign["runs"][0]["backend"]["mount_root"] = "/alternate"
    path = tmp_path / "campaign.yml"
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="must be under declared mount_root"):
        load_campaign(path)


def test_load_campaign_validates_controller_tokens_after_materialization(tmp_path):
    source = REPO_ROOT / "experiments/campaigns/backend_smoke_sensecore_20260711.yml"
    authored = yaml.safe_load(source.read_text(encoding="utf-8"))
    authored["local_root"] = str(tmp_path / "state")
    authored["runs"].append({
        "run_id": "elf-smoke-sco-0711-1618-b",
        "profile": "sensecore-4gpu-spot",
    })
    path = tmp_path / "sensecore-tokens.yml"
    path.write_text(yaml.safe_dump(authored, sort_keys=False), encoding="utf-8")

    campaign = load_campaign(path)
    first, second = campaign["runs"]

    # The authored document remains reviewable and unresolved, while load has
    # already validated the exact recursively materialized backend shape.
    assert first["backend"]["job_name"] == second["backend"]["job_name"] == "{run_id}"
    assert first["backend"]["display_name"] == second["backend"]["display_name"] == "{run_id}"
    first_materialized = materialize_run(campaign, first, "source-fixed")
    second_materialized = materialize_run(campaign, second, "source-fixed")
    assert first_materialized["backend"]["job_name"] == first["run_id"]
    assert first_materialized["backend"]["display_name"] == first["run_id"]
    assert second_materialized["backend"]["job_name"] == second["run_id"]
    assert second_materialized["backend"]["display_name"] == second["run_id"]
    assert first_materialized["backend"]["job_name"] != second_materialized["backend"]["job_name"]
    assert first_materialized["backend"]["display_name"] != second_materialized["backend"]["display_name"]

    first_manifest = prepare_run(
        campaign, first_materialized, "source-fixed", attempt_id="attempt-001"
    )
    second_manifest = prepare_run(
        campaign, second_materialized, "source-fixed", attempt_id="attempt-001"
    )
    serialized_outputs = (
        yaml.safe_dump(campaign, sort_keys=True),
        yaml.safe_dump(first_materialized, sort_keys=True),
        yaml.safe_dump(second_materialized, sort_keys=True),
        json.dumps(first_manifest, sort_keys=True),
        json.dumps(second_manifest, sort_keys=True),
    )
    assert all("validation-source" not in output for output in serialized_outputs)
    assert first_manifest["backend"]["job_name"] == first["run_id"]
    assert second_manifest["backend"]["job_name"] == second["run_id"]
    assert first_manifest["run_id"] != second_manifest["run_id"]


def test_load_campaign_rejects_unreviewed_or_secret_env(tmp_path):
    campaign = controller_campaign(tmp_path)
    campaign["runs"][0]["env"]["WANDB_API_KEY"] = "secret"
    path = tmp_path / "campaign.yml"
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden env keys"):
        load_campaign(path)


def test_load_campaign_rejects_nested_credentials_and_url_userinfo(tmp_path):
    campaign = controller_campaign(tmp_path)
    campaign["runs"][0]["backend"]["api_token"] = "must-not-persist"
    path = tmp_path / "nested-secret.yml"
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="credential-bearing campaign field"):
        load_campaign(path)

    campaign = controller_campaign(tmp_path)
    campaign["runs"][0]["config_overrides"].append(
        "endpoint=https://user:password@example.invalid/api"
    )
    path = tmp_path / "url-userinfo.yml"
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="URL userinfo"):
        load_campaign(path)


def test_prepare_refuses_changed_scientific_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    run["config_overrides"] = ["epochs=2"]
    with pytest.raises(ValueError, match="conflicts"):
        prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")


def test_control_status_is_created_before_submission(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-007")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-007")
    run_dir = tmp_path / "local/controller-test/controller-run"
    status = json.loads(
        (run_dir / "status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "CREATED"
    assert status["attempt_id"] == "attempt-007"
    assert json.loads((run_dir / "backend.json").read_text())["backend"] == "test-backend"
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events].count("attempt_created") == 1


def test_new_attempt_keeps_run_manifest_and_gets_its_own_command(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    second = prepare_run(campaign, run, "source-fixed", attempt_id="attempt-002")
    assert second["attempt_id"] == "attempt-002"
    assert "ATTEMPT_ID=attempt-002" in second["command"]
    assert (tmp_path / "local/controller-test/controller-run/attempts/attempt-002/attempt.yaml").is_file()


@pytest.mark.parametrize("mutation", ["resources", "backend", "command_env"])
def test_new_attempt_cannot_change_run_identity(tmp_path, monkeypatch, mutation):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    changed = dict(run)
    if mutation == "resources":
        changed["resources"] = {**run["resources"], "gpus": run["resources"]["gpus"] + 1}
    elif mutation == "backend":
        changed["backend"] = {**run["backend"], "time": "48:00:00"}
    else:
        changed["env"] = {**run["env"], "NUM_WORKERS": "99"}

    with pytest.raises(ValueError, match="existing run manifest conflicts"):
        prepare_run(campaign, changed, "source-fixed", attempt_id="attempt-002")


def test_run_manifest_freezes_execution_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    manifest = yaml.safe_load(
        (tmp_path / "local/controller-test/controller-run/manifest.yaml").read_text()
    )

    assert manifest["identity_version"] == 2
    assert manifest["backend"] == run["backend"]
    assert manifest["resources"]["gpus"] == run["resources"]["gpus"]
    assert manifest["resources"]["nodes"] == 1
    assert manifest["storage"] == run["storage"]
    assert "ATTEMPT_ID={attempt_id}" in manifest["command"]
    assert manifest["execution"]["source_mount"]
    assert {item["kind"] for item in manifest["assets"]} >= {"model", "dataset"}
    assert manifest["checkpoint"]["save_freq"] == 0.1


def test_retry_resume_is_attempt_operational_state_not_run_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")

    checkpoint = "/mnt/test-project/runs/controller-run/checkpoint_100"
    resumed = {**run, "env": {**run["env"], "RESUME": checkpoint}}
    second = prepare_run(campaign, resumed, "source-fixed", attempt_id="attempt-002")
    run_dir = tmp_path / "local/controller-test/controller-run"
    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())

    assert "resume" not in manifest["resolved_config"]
    assert second["resume_from"] == checkpoint
    assert f"RESUME={checkpoint}" in second["command"]
    assert json.loads((run_dir / "status.json").read_text())["attempt_id"] == "attempt-002"
    assert json.loads((run_dir / "status.json").read_text())["state"] == "CREATED"
    assert json.loads((run_dir / "backend.json").read_text())["backend_job_id"] is None


def test_owned_remote_run_manifest_allows_a_new_attempt_only_with_local_ownership(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")

    class RemoteManifestBackend:
        kind = "test-backend"

        def identity(self, _campaign, _run, _attempt_id):
                return IdentityReport(
                    available=False, ambiguous=False, remote_manifest_exists=True,
                    remote_manifest_matches=True,
                )

    class Registry:
        kinds = frozenset({"test-backend"})

        def get(self, _kind):
            return RemoteManifestBackend()

    real_backends = experimentctl.BACKENDS
    monkeypatch.setattr(experimentctl, "BACKENDS", Registry())
    without_owner = identity_report(campaign, run, "attempt-002")
    assert without_owner["available"] is False
    assert without_owner["remote_manifest_owned"] is False

    # Restore the real adapter while preparing the locally owned run.
    monkeypatch.setattr(experimentctl, "BACKENDS", real_backends)
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    monkeypatch.setattr(experimentctl, "BACKENDS", Registry())
    owned = identity_report(campaign, run, "attempt-002")
    assert owned["available"] is True
    assert owned["remote_manifest_owned"] is True


def test_cli_read_and_cancel_operations_target_explicit_historical_attempt(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "111")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-002")
    record_submission_intent(campaign, run, "attempt-002")
    record_submission(campaign, run, "attempt-002", "222")

    run_dir = tmp_path / "local/controller-test/controller-run"
    (run_dir / "collection.json").write_text(
        json.dumps({"attempt": "attempt-002"}), encoding="utf-8"
    )
    campaign_path = tmp_path / "campaign.yml"
    campaign_path.write_text(yaml.safe_dump(campaign), encoding="utf-8")

    class AttemptBackend:
        kind = "test-backend"

        def __init__(self):
            self.seen: list[tuple[str, str, str]] = []

        def validate(self, _run):
            return None

        def _record(self, operation, selected_campaign, selected_run):
            record = experimentctl.backend_record(selected_campaign, selected_run)
            selected_dir = experimentctl.local_run_dir(selected_campaign, selected_run)
            self.seen.append((operation, str(record["backend_job_id"]), str(selected_dir)))
            return record

        def recover_submission(self, _run, _intent, _attempt_id):
            return None

        def status(self, selected_campaign, selected_run):
            record = self._record("status", selected_campaign, selected_run)
            return {
                "run_id": selected_run["run_id"], "backend": "test-backend",
                "backend_job_id": record["backend_job_id"], "state": "RUNNING",
                "raw_state": "RUNNING",
            }

        def logs(self, selected_campaign, selected_run, *, tail):
            record = self._record("logs", selected_campaign, selected_run)
            return {
                "run_id": selected_run["run_id"], "backend": "test-backend",
                "backend_job_id": record["backend_job_id"], "tail": tail,
                "stdout": ["historical attempt"], "stderr": [],
            }

        def collect(self, selected_campaign, selected_run):
            record = self._record("collect", selected_campaign, selected_run)
            return {
                "run_id": selected_run["run_id"], "backend": "test-backend",
                "backend_job_id": record["backend_job_id"], "state": "RUNNING",
                "step": 7,
            }

        def cancel(self, selected_campaign, selected_run):
            record = self._record("cancel", selected_campaign, selected_run)
            return {
                "run_id": selected_run["run_id"], "backend": "test-backend",
                "backend_job_id": record["backend_job_id"], "state": "CANCELLED",
                "raw_state": "CANCELLED",
            }

    fake = AttemptBackend()

    class Registry:
        kinds = frozenset({"test-backend"})

        def get(self, _kind):
            return fake

    monkeypatch.setattr(experimentctl, "BACKENDS", Registry())
    for command in ("status", "logs", "collect", "observe", "cancel"):
        assert experimentctl.main([
            str(campaign_path), command, "--run", "controller-run",
            "--attempt-id", "attempt-001",
        ]) == 0
        output = json.loads(capsys.readouterr().out)
        rendered = json.dumps(output)
        assert "111" in rendered
        assert "222" not in rendered

    assert experimentctl.main([
        str(campaign_path), "cancel", "--run", "controller-run",
        "--attempt-id", "attempt-001",
    ]) == 0
    capsys.readouterr()
    assert [operation for operation, _, _ in fake.seen].count("cancel") == 1

    assert fake.seen
    assert {job_id for _, job_id, _ in fake.seen} == {"111"}
    assert all(path.endswith("attempts/attempt-001") for _, _, path in fake.seen)
    assert json.loads((run_dir / "backend.json").read_text())["backend_job_id"] == "222"
    assert json.loads((run_dir / "status.json").read_text())["attempt_id"] == "attempt-002"
    assert json.loads((run_dir / "status.json").read_text())["state"] == "QUEUED"
    assert json.loads((run_dir / "attempts/attempt-001/status.json").read_text())[
        "state"
    ] == "CANCELLED"
    assert (run_dir / "attempts/attempt-001/collection.json").is_file()
    assert json.loads((run_dir / "collection.json").read_text()) == {
        "attempt": "attempt-002"
    }


def test_submitted_attempt_cannot_be_submitted_twice(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1234")
    with pytest.raises(FileExistsError, match="already has backend job 1234"):
        ensure_attempt_not_submitted(campaign, run, "attempt-001")


def test_reconcile_rejects_two_recorded_jobs_for_one_attempt(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    events_path = tmp_path / "local/controller-test/controller-run/events.jsonl"
    with events_path.open("a", encoding="utf-8") as handle:
        for job_id in ("1731", "1732"):
            handle.write(json.dumps({
                "attempt_id": "attempt-001",
                "backend_job_id": job_id,
                "event": "scheduler_accepted",
            }) + "\n")
    with pytest.raises(RuntimeError, match="records jobs.*1731.*1732"):
        reconcile_submission(campaign, run, "attempt-001")


def test_reconcile_fails_closed_on_corrupt_local_event_history(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    events_path = tmp_path / "local/controller-test/controller-run/events.jsonl"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")
    with pytest.raises(ValueError, match="invalid lifecycle event"):
        reconcile_submission(campaign, run, "attempt-001")


def test_read_operations_can_recover_frozen_source_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    assert frozen_source_identity(campaign, run, "new-dirty-source") == "source-fixed"


def test_controller_uses_baked_identity_without_git(tmp_path, monkeypatch):
    monkeypatch.setenv("ELF_SOURCE_ID", "runtime.baked-source")
    monkeypatch.setenv("ELF_GIT_COMMIT", "baked-commit")
    monkeypatch.setattr(
        experimentctl.PROJECTS,
        "get",
        lambda _project: (_ for _ in ()).throw(AssertionError("source probe should not run")),
    )
    assert experimentctl.source_identity({"project": "elf", "source_id": "auto"}) == "runtime.baked-source"

    def fake_run(command, *, cwd=None, **_kwargs):
        if command[:2] == ["git", "rev-parse"]:
            raise subprocess.CalledProcessError(128, command)
        return experimentctl.CommandResult(tuple(command), 0, "campaign.baked\n", "")

    monkeypatch.setattr(experimentctl, "run_command", fake_run)
    result = experimentctl.provenance_identity(tmp_path / "campaign.yml")
    assert result == {"git_commit": "baked-commit", "campaign_id": "campaign.baked"}


def test_provenance_identity_preserves_daemon_reviewed_authored_revision(
    tmp_path, monkeypatch,
):
    reviewed_revision = "campaign." + "a" * 64

    def fake_run(command, *, cwd=None, **_kwargs):
        assert command[:2] == ["git", "rev-parse"]
        return experimentctl.CommandResult(tuple(command), 0, "commit-reviewed\n", "")

    monkeypatch.setattr(experimentctl, "run_command", fake_run)
    result = experimentctl.provenance_identity(
        tmp_path / "campaign.execution.yml", campaign_id=reviewed_revision,
    )

    assert result == {
        "git_commit": "commit-reviewed",
        "campaign_id": reviewed_revision,
    }
    with pytest.raises(ValueError, match="campaign.<sha256>"):
        experimentctl.provenance_identity(
            tmp_path / "campaign.execution.yml", campaign_id="campaign.mutable",
        )


def test_parses_structured_training_metric_from_sensecore_log():
    record = parse_training_metric_line(
        "INFO - engine - Step 120: loss=3.1, l2=1.2, ce=9.8, plan=0.0, "
        "plan_aux=0.0, emb_var=0.000e+00, pred_var=0.0, emb_norm=0.00, "
        "pred_norm=0.00, p_phase=0.49, t_phase=0.51, "
        "lr=2.5e-05, steps/sec=2.95"
    )
    assert record == {
        "step": 120,
        "train_loss": 3.1,
        "train_l2_loss": 1.2,
        "train_ce_loss": 9.8,
        "train_plan_loss": 0.0,
        "train_plan_aux_loss": 0.0,
        "train_plan_emb_batch_var": 0.0,
        "train_plan_pred_batch_var": 0.0,
        "train_plan_emb_norm": 0.0,
        "train_plan_pred_norm": 0.0,
        "train_plan_phase_fraction": 0.49,
        "train_token_phase_fraction": 0.51,
        "lr": 2.5e-05,
        "steps_per_sec": 2.95,
    }


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("INFO - generation - gPPL: 152.9692", {"g_ppl": 152.9692}),
        (
            "INFO - generation - oracle_plan_ppl: 132.7387",
            {"oracle_plan_ppl": 132.7387},
        ),
        (
            "INFO - generation - shuffled_plan_ppl: 133.2057",
            {"shuffled_plan_ppl": 133.2057},
        ),
        (
            "INFO - generation - Token reconstruction PPL: 20.0156",
            {"token_recon_ppl": 20.0156},
        ),
    ],
)
def test_parses_evaluation_metric_lines(line, expected):
    assert parse_training_metric_line(line) == expected


def test_collection_separates_stale_runtime_from_scheduler_truth():
    result = annotate_collection({"state": "RUNNING", "step": 10}, {"state": "CANCELLED"})
    assert result["runtime_state"] == "RUNNING"
    assert result["scheduler_state"] == "CANCELLED"
    assert result["worker_state"] == "RELEASED"
    assert result["process_state"] == "CANCELLED"
    assert result["model_state"] == "OBSERVED"


def test_terminal_scheduler_state_overrides_stale_worker_and_process_state():
    result = annotate_collection(
        {"state": "RUNNING", "worker_state": "ALLOCATED", "step": 10},
        {"state": "SUCCEEDED"},
    )
    assert result["runtime_state"] == "RUNNING"
    assert result["scheduler_state"] == "SUCCEEDED"
    assert result["worker_state"] == "RELEASED"
    assert result["process_state"] == "SUCCEEDED"


def test_collection_cleans_nul_padded_process_logs():
    result = annotate_collection(
        {
            "state": "RUNNING",
            "process_evidence": {
                "observed": True,
                "stdout_tail": ["metric\x00\x00"],
                "stderr_tail": ["progress\x00"],
            },
        },
        {"state": "RUNNING"},
    )
    assert result["process_evidence"]["stdout_tail"] == ["metric"]
    assert result["process_evidence"]["stderr_tail"] == ["progress"]


def test_logs_fall_back_to_attempt_collection_when_live_probe_is_unavailable(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1234")
    selected = experimentctl.select_attempt(run, "attempt-001")
    experimentctl.write_local_collection(campaign, selected, {
        "process_evidence": {
            "observed": True,
            "sources": {"stdout": "/remote/stdout", "stderr": "/remote/stderr"},
            "stdout_tail": ["one", "two", "three"],
            "stderr_tail": ["warning"],
        }
    })

    class UnavailableLogs:
        def logs(self, _campaign, _run, *, tail):
            raise RuntimeError(f"live log probe unavailable at tail {tail}")

    result = experimentctl.read_logs(
        campaign, selected, UnavailableLogs(), tail=2
    )
    assert result["live"] is False
    assert result["evidence_source"] == "cached_collection"
    assert result["stdout"] == ["two", "three"]
    assert result["stderr"] == ["warning"]
    assert result["backend_job_id"] == "1234"


def test_sensecore_cancel_terminal_collection_retains_first_metric(
    tmp_path, monkeypatch,
):
    from elf_experiments.controller import status_for_decision
    from elf_experiments.policy import decide_next_action

    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "sensecore-exact-job")
    selected = experimentctl.select_attempt(run, "attempt-001")
    running = annotate_collection(
        {
            "run_id": run["run_id"],
            "state": "RUNNING",
            "model_observed": True,
            "latest_metric": {
                "step": 1539, "train_loss": 3.5847, "steps_per_sec": 16.42,
            },
            "metric_source": "/data/run/train_metrics.jsonl",
            "process_evidence": {
                "observed": True,
                "sources": {"stdout": "sensecore_stream_logs"},
                "stdout_tail": ["Step 1539 train_loss=3.5847"],
                "stderr_tail": [],
            },
        },
        {"state": "RUNNING"},
    )
    experimentctl.write_local_collection(campaign, selected, running)

    terminal = annotate_collection(
        {
            "run_id": run["run_id"],
            "state": None,
            "model_observed": False,
            "worker_state": "RELEASED",
            "worker_phases": ["Deleted"],
            "process_evidence": {
                "observed": False, "stdout_tail": [], "stderr_tail": [],
            },
        },
        {"state": "CANCELLED"},
    )
    merged = experimentctl.write_local_collection(campaign, selected, terminal)
    decision = decide_next_action(
        status_for_decision({"state": "CANCELLED"}, merged),
        retries_used=0,
        max_infra_retries=0,
        diagnostic_text=json.dumps(merged),
    )

    assert merged["scheduler_state"] == "CANCELLED"
    assert merged["worker_state"] == "RELEASED"
    assert merged["process_state"] == "UNKNOWN"
    assert merged["latest_metric"]["step"] == 1539
    assert merged["metric_source"] == "/data/run/train_metrics.jsonl"
    assert merged["model_state"] == "OBSERVED"
    assert merged["process_evidence"]["stdout_tail"] == [
        "Step 1539 train_loss=3.5847"
    ]
    assert merged["retained_evidence"]["retained"] is True
    assert decision.action == "DO_NOT_RETRY"
    assert decision.failure_class == "none"


def test_watch_streams_first_metric_and_persists_decision(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1234")
    selected = experimentctl.select_attempt(run, "attempt-001")

    class RunningBackend:
        def status(self, _campaign, selected_run):
            return {
                "run_id": selected_run["run_id"], "backend": "test-backend",
                "backend_job_id": "1234", "state": "RUNNING",
                "raw_state": "RUNNING",
            }

        def collect(self, _campaign, selected_run):
            return {
                "run_id": selected_run["run_id"], "state": "RUNNING",
                "step": 100, "optimizer_step": 50, "train_loss": 3.0,
            }

    backend = RunningBackend()

    class Registry:
        kinds = frozenset({"test-backend"})

        def get(self, _kind):
            return backend

    monkeypatch.setattr(experimentctl, "BACKENDS", Registry())
    assert experimentctl.watch_runs(
        campaign, [selected], attempt_id="attempt-001",
        interval_seconds=1, timeout_seconds=0, poll_timeout_seconds=60,
        until="first-metric",
    ) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == [
        "watch_observation", "watch_run_complete", "watch_complete",
    ]
    assert events[0]["model_state"] == "OBSERVED"
    assert events[0]["optimizer_step"] == 50
    assert events[1]["reason"] == "first-metric"
    assert events[1]["gate_passed"] is True
    assert events[1]["decision"]["action"] == "OBSERVE"
    assert events[2]["gate_passed"] is True
    assert events[2]["failed_gate_run_ids"] == []
    assert (
        tmp_path / "local/controller-test/controller-run/attempts/attempt-001/decision.json"
    ).is_file()


def test_first_metric_gate_does_not_accept_checkpoint_only_evidence():
    assert experimentctl.has_model_metric({
        "model_state": "OBSERVED", "latest_completed_checkpoint": "/ckpt"
    }) is False
    assert experimentctl.has_model_metric({"step": 0}) is True
    assert experimentctl.has_model_metric({
        "evaluation_metrics_by_variant": {"generation": {"g_ppl": 12.5}}
    }) is True


def test_collection_treats_evaluation_metrics_as_model_evidence():
    result = annotate_collection(
        {
            "state": "SUCCEEDED",
            "evaluation_metrics_by_variant": {
                "generation": {"g_ppl": 12.5},
            },
        },
        {"state": "SUCCEEDED"},
    )

    assert result["model_state"] == "OBSERVED"


def test_watch_first_metric_gate_fails_when_run_terminates_without_metric(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1234")
    selected = experimentctl.select_attempt(run, "attempt-001")

    class FailedBeforeMetricBackend:
        def status(self, _campaign, selected_run):
            return {
                "run_id": selected_run["run_id"], "backend": "test-backend",
                "backend_job_id": "1234", "state": "FAILED",
                "raw_state": "FAILED", "exit_code": "1:0",
            }

        def collect(self, _campaign, selected_run):
            return {
                "run_id": selected_run["run_id"], "state": "FAILED",
                "process_evidence": {
                    "observed": True,
                    "stdout_tail": [],
                    "stderr_tail": ["startup failed"],
                },
            }

    backend = FailedBeforeMetricBackend()

    class Registry:
        kinds = frozenset({"test-backend"})

        def get(self, _kind):
            return backend

    monkeypatch.setattr(experimentctl, "BACKENDS", Registry())
    monkeypatch.setattr(
        experimentctl.time, "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("must not sleep")),
    )
    assert experimentctl.watch_runs(
        campaign, [selected], attempt_id="attempt-001",
        interval_seconds=60, timeout_seconds=0, poll_timeout_seconds=60,
        until="first-metric",
    ) == 1

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == [
        "watch_observation", "watch_run_complete", "watch_complete",
    ]
    assert events[0]["model_state"] == "NOT_OBSERVED"
    assert events[1]["reason"] == "terminal-without-first-metric"
    assert events[1]["gate_passed"] is False
    assert events[1]["decision"]["action"] == "DO_NOT_RETRY"
    assert events[2]["gate_passed"] is False
    assert events[2]["failed_gate_run_ids"] == ["controller-run"]
    assert (
        tmp_path / "local/controller-test/controller-run/attempts/attempt-001/decision.json"
    ).is_file()


def test_watch_terminal_state_collects_and_decides_without_sleeping(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1234")
    selected = experimentctl.select_attempt(run, "attempt-001")

    class TerminalBackend:
        def status(self, _campaign, selected_run):
            return {
                "run_id": selected_run["run_id"], "backend": "test-backend",
                "backend_job_id": "1234", "state": "SUCCEEDED",
                "raw_state": "COMPLETED", "exit_code": "0:0",
            }

        def collect(self, _campaign, selected_run):
            return {
                "run_id": selected_run["run_id"], "state": "RUNNING",
                "step": 100, "latest_completed_checkpoint": "/remote/ckpt",
            }

    backend = TerminalBackend()

    class Registry:
        kinds = frozenset({"test-backend"})

        def get(self, _kind):
            return backend

    monkeypatch.setattr(experimentctl, "BACKENDS", Registry())
    monkeypatch.setattr(
        experimentctl.time, "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("must not sleep")),
    )
    assert experimentctl.watch_runs(
        campaign, [selected], attempt_id="attempt-001",
        interval_seconds=60, timeout_seconds=0, poll_timeout_seconds=60,
        until="terminal",
    ) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    terminal = events[1]
    assert terminal["reason"] == "terminal"
    assert terminal["gate_passed"] is True
    assert terminal["worker_state"] == "RELEASED"
    assert terminal["process_state"] == "SUCCEEDED"
    assert terminal["decision"]["action"] == "VERIFY_RESULTS"


def test_watch_hard_times_out_one_blocked_poll(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(REPO_ROOT)
    campaign = controller_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1234")
    selected = experimentctl.select_attempt(run, "attempt-001")

    class BlockedBackend:
        def status(self, _campaign, _run):
            raise subprocess.TimeoutExpired(["ssh", "test-login"], 0.01)

    backend = BlockedBackend()

    class Registry:
        kinds = frozenset({"test-backend"})

        def get(self, _kind):
            return backend

    monkeypatch.setattr(experimentctl, "BACKENDS", Registry())
    assert experimentctl.watch_runs(
        campaign, [selected], attempt_id="attempt-001",
        interval_seconds=60, timeout_seconds=0.01,
        poll_timeout_seconds=30, until="terminal",
    ) == 1
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == [
        "watch_poll_timeout", "watch_timeout",
    ]
    assert events[0]["action"] == "RETRY_OBSERVATION"
    assert events[0]["run_id"] == "controller-run"


def test_collection_classifies_pretraining_import_failure():
    result = annotate_collection(
        {
            "state": "UNKNOWN",
            "process_evidence": {
                "observed": True,
                "stderr_tail": [
                    "ModuleNotFoundError: No module named 'experiment_control'"
                ],
            },
        },
        {"state": "FAILED"},
    )
    assert result["worker_state"] == "RELEASED"
    assert result["process_state"] == "FAILED"
    assert result["model_state"] == "NOT_OBSERVED"
    assert result["failure_class"] == "configuration"


def test_decision_prefers_collected_import_failure_over_transport_status():
    from elf_experiments.controller import status_for_decision
    from elf_experiments.policy import decide_next_action

    collection = annotate_collection(
        {
            "state": "UNKNOWN",
            "process_evidence": {
                "observed": True,
                "stderr_tail": [
                    "ModuleNotFoundError: No module named 'experiment_control'"
                ],
            },
        },
        {"state": "FAILED", "failure_class": "transport"},
    )
    decision = decide_next_action(
        status_for_decision(
            {"state": "FAILED", "failure_class": "transport"}, collection
        ),
        retries_used=0,
        max_infra_retries=2,
        diagnostic_text=json.dumps(collection),
    )
    assert collection["failure_class"] == "configuration"
    assert decision.failure_class == "configuration"
    assert decision.action == "DO_NOT_RETRY"


def test_decision_prefers_collected_oom_over_transport_and_blocks_retry():
    from elf_experiments.controller import status_for_decision
    from elf_experiments.policy import decide_next_action

    collection = annotate_collection(
        {
            "state": "UNKNOWN",
            "live_logs_expired": False,
            "process_evidence": {
                "observed": True,
                "stdout_tail": [
                    "502 log stream closed after process exit",
                    "torch.OutOfMemoryError: CUDA out of memory",
                ],
                "stderr_tail": [],
            },
        },
        {"state": "FAILED", "failure_class": "transport"},
    )
    decision = decide_next_action(
        status_for_decision(
            {"state": "FAILED", "failure_class": "transport"}, collection
        ),
        retries_used=0,
        max_infra_retries=2,
        diagnostic_text=json.dumps(collection),
    )
    assert collection["failure_class"] == "resource"
    assert decision.failure_class == "resource"
    assert decision.action == "DO_NOT_RETRY"


def test_collection_marks_expired_external_evidence_inconclusive():
    result = annotate_collection(
        {
            "state": None,
            "model_observed": False,
            "evidence_unavailable_reason": "live_logs_expired",
        },
        {"state": "SUCCEEDED"},
    )
    assert result["worker_state"] == "RELEASED"
    assert result["process_state"] == "UNKNOWN"
    assert result["model_state"] == "UNKNOWN"
    assert result["evidence_outcome"] == "INCONCLUSIVE"


def test_cancelled_without_process_or_model_evidence_is_inconclusive():
    result = annotate_collection(
        {
            "state": None, "model_observed": False,
            "worker_state": "RELEASED", "worker_phases": ["Deleted"],
        },
        {"state": "CANCELLED"},
    )
    assert result["model_state"] == "NOT_OBSERVED"
    assert result["evidence_outcome"] == "INCONCLUSIVE"
    assert result["evidence_unavailable_reason"] == "cancelled_before_observation"
