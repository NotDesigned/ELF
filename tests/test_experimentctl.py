import json
import os
import subprocess
import sys
import shutil
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experimentctl import (
    annotate_collection,
    ensure_attempt_not_submitted,
    frozen_source_identity,
    load_campaign,
    materialize_run,
    prepare_run,
    record_submission,
    record_submission_intent,
    resolved_run_overrides,
)
from experiment_manifest import prepare as runtime_prepare
from experiment_control.backends.wyd import render_job
from experiment_projects.elf import parse_training_metric_line


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = "src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf_len256.yml"


def test_preflight_cli_returns_nonzero_when_a_required_tool_is_unavailable():
    env = os.environ.copy()
    env["EXPERIMENTCTL_SSH_BIN"] = "/bin/false"
    result = subprocess.run(
        [
            sys.executable, "scripts/experimentctl.py",
            "experiments/campaigns/backend_smoke_slurm_20260711.yml",
            "preflight", "--run", "elf-smoke-slurm-l40s-0711-1642",
        ],
        cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 1
    assert '"ready": false' in result.stdout


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


def test_controller_manifest_is_accepted_unchanged_by_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = slurm_campaign(tmp_path)
    campaign.update({"git_commit": "commit", "campaign_id": "campaign-id"})
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    remote_dir = tmp_path / "remote-run"
    run["storage"]["run_dir"] = str(remote_dir)
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    local_manifest = tmp_path / "local/controller-test/smoke-h100/manifest.yaml"
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


def test_submitted_attempt_cannot_be_submitted_twice(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    record_submission_intent(campaign, run, "attempt-001")
    record_submission(campaign, run, "attempt-001", "1234")
    with pytest.raises(FileExistsError, match="already has backend job 1234"):
        ensure_attempt_not_submitted(campaign, run, "attempt-001")


def test_read_operations_can_recover_frozen_source_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = slurm_campaign(tmp_path)
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    assert frozen_source_identity(campaign, run, "new-dirty-source") == "source-fixed"


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
