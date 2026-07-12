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


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = "src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf_len256.yml"


def test_preflight_cli_returns_nonzero_when_a_required_tool_is_unavailable():
    env = os.environ.copy()
    env["EXPERIMENTCTL_SSH_BIN"] = "/bin/false"
    result = subprocess.run(
        [
            sys.executable, "tools/experimentctl.py",
            "experiments/campaigns/backend_smoke_slurm_20260711.yml",
            "preflight", "--run", "elf-smoke-slurm-l40s-0711-1642",
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


def slurm_campaign(tmp_path: Path) -> dict:
    """Return a minimal campaign using H100 to guard against L40S assumptions."""
    return {
        "schema_version": 1,
        "campaign": "controller-test",
        "project": "elf",
        "source_id": "source-fixed",
        "local_root": str(tmp_path / "local"),
        "runs": [
            {
                "run_id": "smoke-h100",
                "config": CONFIG,
                "config_overrides": ["epochs=1", "save_freq=0.1"],
                "image_id": "sha256:" + "a" * 64,
                "resources": {"gpus": 1, "cpus": 8},
                "storage": {
                    "run_dir": "/data/liangluocheng/elf/runs/smoke-h100",
                    "data_root": "/data/liangluocheng",
                    "project_data_root": "/data/liangluocheng/elf",
                    "hf_home": "/data/liangluocheng/elf/cache/huggingface",
                    "hf_datasets_cache": "/data/liangluocheng/elf/cache/huggingface/datasets",
                },
                "env": {"BATCH_SIZE": "4", "LOG_FREQ": "10"},
                "backend": {
                    "kind": "slurm",
                    "ssh_alias": "wyd-l40s",
                    "partition": "h100",
                    "account": "lab",
                    "qos": "normal",
                    "gres": "gpu:h100:1",
                    "time": "00:10:00",
                    "mount_root": "/data",
                    "source_dir": "/data/liangluocheng/elf/sources/{source_id}",
                    "sif_path": "/data/liangluocheng/elf/images/test.sif",
                },
            }
        ],
    }


def test_prepare_and_render_preserve_explicit_partition(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    manifest = prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    script = render_job(manifest)

    assert "#SBATCH --partition=h100" in script
    assert "#SBATCH --gres=gpu:h100:1" in script
    assert "#SBATCH --job-name=smoke-h100--attempt-001" in script
    assert "#SBATCH --output=/dev/null" in script
    assert "attempts/attempt-001" in script
    assert "--bind /data/liangluocheng/elf/sources/source-fixed:/app" in script
    assert 'export BACKEND_JOB_ID="$SLURM_JOB_ID"' in script
    assert "WANDB_DIR=/data/liangluocheng/elf/wandb" in script
    assert "CHECKPOINT_ROOT=/data/liangluocheng/elf/checkpoints" in script
    assert manifest["resolved_config"]["epochs"] == 1
    assert manifest["resolved_config"]["save_freq"] == 0.1
    assert manifest["resolved_config"]["global_batch_size"] is None
    assert manifest["resolved_config"]["batch_size"] == 4
    assert manifest["resolved_config"]["log_freq"] == 10


def test_submit_dry_run_reports_local_artifacts_and_next_gates(tmp_path):
    campaign = slurm_campaign(tmp_path)
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
    campaign = slurm_campaign(tmp_path)
    campaign.update({"git_commit": "commit", "campaign_id": "campaign-id"})
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    remote_dir = tmp_path / "remote-run"
    run["storage"]["run_dir"] = str(remote_dir)
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    local_manifest = tmp_path / "local/controller-test/smoke-h100/manifest.yaml"
    frozen = yaml.safe_load(local_manifest.read_text(encoding="utf-8"))
    assert frozen["git_commit"] == "commit"
    assert frozen["runtime_tree_id"] == "source-fixed"
    assert frozen["campaign_id"] == "campaign-id"
    assert frozen["image_id"] == run["image_id"]
    remote_dir.mkdir()
    shutil.copy2(local_manifest, remote_dir / "manifest.yaml")
    runtime_prepare(Namespace(
        project="elf", run_id="smoke-h100", attempt_id="attempt-001",
        backend="slurm", backend_job_id="123", config=CONFIG,
        config_override=resolved_run_overrides(campaign, run, str(remote_dir)),
        output_dir=str(remote_dir), source_id="source-fixed", runtime_tree_id="source-fixed",
        git_commit="commit", campaign_id="campaign-id", campaign="controller-test",
        image_id=run["image_id"], gpus=1, nodes=1, quota="normal",
        resource_spec="", max_infra_retries=0, require_immutable_identities=True,
        command=["true"],
    ))
    assert (remote_dir / "attempts/attempt-001/attempt.yaml").is_file()


def test_render_supports_datapool_storage_mount(tmp_path, monkeypatch):
    """H100 jobs bind and cache on their declared /datapool filesystem."""
    monkeypatch.chdir(REPO_ROOT)
    campaign = slurm_campaign(tmp_path)
    backend = campaign["runs"][0]["backend"]
    backend.update(
        {
            "mount_root": "/datapool",
            "apptainer_cache_dir": "/datapool/liangluocheng/elf/apptainer/cache",
            "apptainer_tmp_dir": "/datapool/liangluocheng/elf/apptainer/tmp",
            "source_dir": "/datapool/liangluocheng/elf/sources/{source_id}",
            "sif_path": "/datapool/liangluocheng/elf/images/test.sif",
        }
    )
    campaign["runs"][0]["storage"].update(
        {
            "run_dir": "/datapool/liangluocheng/elf/runs/smoke-h100",
            "data_root": "/datapool/liangluocheng",
            "project_data_root": "/datapool/liangluocheng/elf",
            "hf_home": "/datapool/liangluocheng/.cache/huggingface",
            "hf_datasets_cache": "/datapool/liangluocheng/.cache/huggingface/datasets",
        }
    )
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    manifest = prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    script = render_job(manifest)

    assert "--bind /datapool:/datapool" in script
    assert "export APPTAINER_CACHEDIR=/datapool/liangluocheng/elf/apptainer/cache" in script
    assert "export APPTAINER_TMPDIR=/datapool/liangluocheng/elf/apptainer/tmp" in script


def test_rejects_relative_slurm_mount_root(tmp_path):
    campaign = slurm_campaign(tmp_path)
    campaign["runs"][0]["backend"]["mount_root"] = "datapool"
    path = tmp_path / "campaign.yml"
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="mount_root must be an absolute path"):
        load_campaign(path)


def test_rejects_mixed_slurm_storage_profiles(tmp_path):
    campaign = slurm_campaign(tmp_path)
    campaign["runs"][0]["backend"]["mount_root"] = "/datapool"
    path = tmp_path / "campaign.yml"
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="must be under declared mount_root"):
        load_campaign(path)


def test_load_campaign_rejects_unreviewed_or_secret_env(tmp_path):
    campaign = slurm_campaign(tmp_path)
    campaign["runs"][0]["env"]["WANDB_API_KEY"] = "secret"
    path = tmp_path / "campaign.yml"
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden env keys"):
        load_campaign(path)


def test_load_campaign_rejects_nested_credentials_and_url_userinfo(tmp_path):
    campaign = slurm_campaign(tmp_path)
    campaign["runs"][0]["backend"]["api_token"] = "must-not-persist"
    path = tmp_path / "nested-secret.yml"
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="credential-bearing campaign field"):
        load_campaign(path)

    campaign = slurm_campaign(tmp_path)
    campaign["runs"][0]["config_overrides"].append(
        "endpoint=https://user:password@example.invalid/api"
    )
    path = tmp_path / "url-userinfo.yml"
    path.write_text(yaml.safe_dump(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="URL userinfo"):
        load_campaign(path)


def test_prepare_refuses_changed_scientific_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    run["config_overrides"] = ["epochs=2"]
    with pytest.raises(ValueError, match="conflicts"):
        prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")


def test_control_status_is_created_before_submission(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-007")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-007")
    run_dir = tmp_path / "local/controller-test/smoke-h100"
    status = json.loads(
        (run_dir / "status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "CREATED"
    assert status["attempt_id"] == "attempt-007"
    assert json.loads((run_dir / "backend.json").read_text())["backend"] == "slurm"
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events].count("attempt_created") == 1


def test_new_attempt_keeps_run_manifest_and_gets_its_own_command(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    second = prepare_run(campaign, run, "source-fixed", attempt_id="attempt-002")
    assert second["attempt_id"] == "attempt-002"
    assert "ATTEMPT_ID=attempt-002" in second["command"]
    assert (tmp_path / "local/controller-test/smoke-h100/attempts/attempt-002/attempt.yaml").is_file()


@pytest.mark.parametrize("mutation", ["resources", "backend", "command_env"])
def test_new_attempt_cannot_change_run_identity(tmp_path, monkeypatch, mutation):
    monkeypatch.chdir(REPO_ROOT)
    campaign = slurm_campaign(tmp_path)
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
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    manifest = yaml.safe_load(
        (tmp_path / "local/controller-test/smoke-h100/manifest.yaml").read_text()
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
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")

    checkpoint = "/data/liangluocheng/elf/runs/smoke-h100/checkpoint_100"
    resumed = {**run, "env": {**run["env"], "RESUME": checkpoint}}
    second = prepare_run(campaign, resumed, "source-fixed", attempt_id="attempt-002")
    run_dir = tmp_path / "local/controller-test/smoke-h100"
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
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")

    class RemoteManifestBackend:
        kind = "slurm"

        def identity(self, _campaign, _run, _attempt_id):
                return IdentityReport(
                    available=False, ambiguous=False, remote_manifest_exists=True,
                    remote_manifest_matches=True,
                )

    class Registry:
        kinds = frozenset({"slurm"})

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
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "111")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-002")
    record_submission_intent(campaign, run, "attempt-002")
    record_submission(campaign, run, "attempt-002", "222")

    run_dir = tmp_path / "local/controller-test/smoke-h100"
    (run_dir / "collection.json").write_text(
        json.dumps({"attempt": "attempt-002"}), encoding="utf-8"
    )
    campaign_path = tmp_path / "campaign.yml"
    campaign_path.write_text(yaml.safe_dump(campaign), encoding="utf-8")

    class AttemptBackend:
        kind = "slurm"

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
                "run_id": selected_run["run_id"], "backend": "slurm",
                "backend_job_id": record["backend_job_id"], "state": "RUNNING",
                "raw_state": "RUNNING",
            }

        def logs(self, selected_campaign, selected_run, *, tail):
            record = self._record("logs", selected_campaign, selected_run)
            return {
                "run_id": selected_run["run_id"], "backend": "slurm",
                "backend_job_id": record["backend_job_id"], "tail": tail,
                "stdout": ["historical attempt"], "stderr": [],
            }

        def collect(self, selected_campaign, selected_run):
            record = self._record("collect", selected_campaign, selected_run)
            return {
                "run_id": selected_run["run_id"], "backend": "slurm",
                "backend_job_id": record["backend_job_id"], "state": "RUNNING",
                "step": 7,
            }

        def cancel(self, selected_campaign, selected_run):
            record = self._record("cancel", selected_campaign, selected_run)
            return {
                "run_id": selected_run["run_id"], "backend": "slurm",
                "backend_job_id": record["backend_job_id"], "state": "CANCELLED",
                "raw_state": "CANCELLED",
            }

    fake = AttemptBackend()

    class Registry:
        kinds = frozenset({"slurm"})

        def get(self, _kind):
            return fake

    monkeypatch.setattr(experimentctl, "BACKENDS", Registry())
    for command in ("status", "logs", "collect", "observe", "cancel"):
        assert experimentctl.main([
            str(campaign_path), command, "--run", "smoke-h100",
            "--attempt-id", "attempt-001",
        ]) == 0
        output = json.loads(capsys.readouterr().out)
        rendered = json.dumps(output)
        assert "111" in rendered
        assert "222" not in rendered

    assert experimentctl.main([
        str(campaign_path), "cancel", "--run", "smoke-h100",
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
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1234")
    with pytest.raises(FileExistsError, match="already has backend job 1234"):
        ensure_attempt_not_submitted(campaign, run, "attempt-001")


def test_reconcile_rejects_two_recorded_jobs_for_one_attempt(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    events_path = tmp_path / "local/controller-test/smoke-h100/events.jsonl"
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
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    events_path = tmp_path / "local/controller-test/smoke-h100/events.jsonl"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")
    with pytest.raises(ValueError, match="invalid lifecycle event"):
        reconcile_submission(campaign, run, "attempt-001")


def test_read_operations_can_recover_frozen_source_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = slurm_campaign(tmp_path)
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


def test_parses_structured_training_metric_from_sensecore_log():
    record = parse_training_metric_line(
        "INFO - engine - Step 120: loss=3.1, l2=1.2, ce=9.8, plan=0.0, "
        "plan_aux=0.0, emb_var=0.000e+00, pred_var=0.0, emb_norm=0.00, "
        "pred_norm=0.00, lr=2.5e-05, steps/sec=2.95"
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
        "lr": 2.5e-05,
        "steps_per_sec": 2.95,
    }


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
    campaign = slurm_campaign(tmp_path)
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


def test_watch_streams_first_metric_and_persists_decision(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.chdir(REPO_ROOT)
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1234")
    selected = experimentctl.select_attempt(run, "attempt-001")

    class RunningBackend:
        def status(self, _campaign, selected_run):
            return {
                "run_id": selected_run["run_id"], "backend": "slurm",
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
        kinds = frozenset({"slurm"})

        def get(self, _kind):
            return backend

    monkeypatch.setattr(experimentctl, "BACKENDS", Registry())
    assert experimentctl.watch_runs(
        campaign, [selected], attempt_id="attempt-001",
        interval_seconds=1, timeout_seconds=0, until="first-metric",
    ) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["event"] for event in events] == [
        "watch_observation", "watch_run_complete", "watch_complete",
    ]
    assert events[0]["model_state"] == "OBSERVED"
    assert events[0]["optimizer_step"] == 50
    assert events[1]["reason"] == "first-metric"
    assert events[1]["decision"]["action"] == "OBSERVE"
    assert (
        tmp_path / "local/controller-test/smoke-h100/attempts/attempt-001/decision.json"
    ).is_file()


def test_first_metric_gate_does_not_accept_checkpoint_only_evidence():
    assert experimentctl.has_model_metric({
        "model_state": "OBSERVED", "latest_completed_checkpoint": "/ckpt"
    }) is False
    assert experimentctl.has_model_metric({"step": 0}) is True


def test_watch_terminal_state_collects_and_decides_without_sleeping(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.chdir(REPO_ROOT)
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1234")
    selected = experimentctl.select_attempt(run, "attempt-001")

    class TerminalBackend:
        def status(self, _campaign, selected_run):
            return {
                "run_id": selected_run["run_id"], "backend": "slurm",
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
        kinds = frozenset({"slurm"})

        def get(self, _kind):
            return backend

    monkeypatch.setattr(experimentctl, "BACKENDS", Registry())
    monkeypatch.setattr(
        experimentctl.time, "sleep",
        lambda _seconds: (_ for _ in ()).throw(AssertionError("must not sleep")),
    )
    assert experimentctl.watch_runs(
        campaign, [selected], attempt_id="attempt-001",
        interval_seconds=60, timeout_seconds=0, until="terminal",
    ) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    terminal = events[1]
    assert terminal["reason"] == "terminal"
    assert terminal["worker_state"] == "RELEASED"
    assert terminal["process_state"] == "SUCCEEDED"
    assert terminal["decision"]["action"] == "VERIFY_RESULTS"


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
