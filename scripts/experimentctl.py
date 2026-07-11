#!/usr/bin/env python
"""Prepare, submit, inspect, and collect ELF campaigns across SenseCore and Slurm."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_manifest import (  # noqa: E402
    IDENTITY_RE,
    SECRET_KEY_RE,
    ExperimentStateStore,
    RunState,
    append_event,
    atomic_write,
    resolved_config,
    sanitize_command,
    utc_now,
)
from experiment_campaign import load_and_resolve_campaign  # noqa: E402
from experiment_assets import cache_path, plan_assets  # noqa: E402
from experiment_overrides import operational_overrides  # noqa: E402
from experiment_policy import decide_next_action  # noqa: E402
from experiment_control.backends.base import BackendRegistry  # noqa: E402
from experiment_control.backends.services import BackendServices  # noqa: E402
from experiment_control.backends.wyd import WydSlurmBackend  # noqa: E402
from experiment_control.backends.sensecore import SenseCoreBackend as SenseCoreAdapter  # noqa: E402
from experiment_control.backends.slurm import scheduler_job_name, shell_join  # noqa: E402
from experiment_control.runner import (  # noqa: E402
    CommandResult,
    CommandRunner,
    SubprocessRunner,
)
from summarize_experiments import summarize_run  # noqa: E402


SAFE_ENV_KEYS = {
    "BATCH_SIZE",
    "DATA_ROOT",
    "GLOBAL_BATCH_SIZE",
    "HF_DATASETS_OFFLINE",
    "HF_DATASETS_CACHE",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "LOG_FREQ",
    "MAX_INFRA_RETRIES",
    "NUM_WORKERS",
    "PROJECT_DATA_ROOT",
    "REQUIRE_OFFLINE_CACHE",
    "TRANSFORMERS_OFFLINE",
    "USE_COMPILE",
    "USE_WANDB",
}
SSH_CONTROL_PATH = "/tmp/elf-experimentctl-%C"
_COMMAND_RUNNER: CommandRunner = SubprocessRunner()


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    input_text: str | None = None,
) -> CommandResult:
    """Run a local command with captured text output and no shell expansion.

    Args:
        command: Exact argument vector. Secret-bearing arguments are forbidden
            by campaign validation before reaching this function.
        cwd: Optional working directory.
        check: Raise ``CalledProcessError`` on non-zero exit when true.
        input_text: Optional UTF-8 text supplied to stdin.

    Returns:
        The completed process with stdout and stderr captured.
    """
    return _COMMAND_RUNNER.run(
        command, cwd=cwd, check=check, input_text=input_text
    )


def set_command_runner(runner: CommandRunner) -> None:
    """Inject a hermetic runner for adapter contract and recovery tests."""
    global _COMMAND_RUNNER
    _COMMAND_RUNNER = runner


def load_campaign(path: Path) -> dict[str, Any]:
    """Load and structurally validate one campaign YAML document."""
    payload = load_and_resolve_campaign(path)
    if not isinstance(payload, dict):
        raise ValueError(f"campaign must be a mapping: {path}")
    if payload.get("schema_version") != 1:
        raise ValueError("campaign schema_version must be 1")
    for key in ("campaign", "project", "runs"):
        if not payload.get(key):
            raise ValueError(f"campaign is missing {key}")
    if not IDENTITY_RE.fullmatch(str(payload["campaign"])):
        raise ValueError("campaign identity contains unsupported characters")
    if not isinstance(payload["runs"], list) or not payload["runs"]:
        raise ValueError("campaign runs must be a non-empty list")
    seen: set[str] = set()
    for run in payload["runs"]:
        validate_run(run)
        if run["run_id"] in seen:
            raise ValueError(f"duplicate run_id: {run['run_id']}")
        seen.add(run["run_id"])
    return payload


def validate_run(run: Any) -> None:
    """Validate one backend-neutral run and reject secret-bearing settings."""
    if not isinstance(run, dict):
        raise ValueError("each campaign run must be a mapping")
    for key in ("run_id", "config", "backend", "storage", "image_id"):
        if not run.get(key):
            raise ValueError(f"run is missing {key}")
    if not IDENTITY_RE.fullmatch(str(run["run_id"])):
        raise ValueError(f"invalid run_id: {run['run_id']!r}")
    backend = run["backend"]
    if not isinstance(backend, dict) or backend.get("kind") not in {"sensecore", "slurm"}:
        raise ValueError(f"run {run['run_id']} backend.kind must be sensecore or slurm")
    env = run.get("env", {})
    if not isinstance(env, dict):
        raise ValueError(f"run {run['run_id']} env must be a mapping")
    forbidden = [key for key in env if key not in SAFE_ENV_KEYS or SECRET_KEY_RE.search(key)]
    if forbidden:
        raise ValueError(f"run {run['run_id']} has forbidden env keys: {sorted(forbidden)}")
    for value in env.values():
        if "\n" in str(value) or "\x00" in str(value):
            raise ValueError(f"run {run['run_id']} env values must be single-line text")
    storage = run["storage"]
    if not isinstance(storage, dict) or not storage.get("run_dir"):
        raise ValueError(f"run {run['run_id']} storage.run_dir is required")
    for value in [*backend.values(), *storage.values()]:
        if isinstance(value, str) and ("\n" in value or "\x00" in value):
            raise ValueError(f"run {run['run_id']} backend/storage values must be single-line text")
    if backend["kind"] == "slurm":
        required = {
            "ssh_alias", "partition", "account", "qos", "gres", "time", "source_dir", "sif_path",
            "data_root", "project_data_root", "hf_home", "hf_datasets_cache",
        }
    else:
        required = {"workspace", "aec2", "worker_spec", "image", "storage_mount", "quota_type"}
        if backend.get("quota_type") != "spot":
            raise ValueError("SenseCore runs for this account must use spot quota")
        image = str(backend.get("image", ""))
        tag = image.rsplit(":", 1)[-1]
        if tag in {"latest", "runtime", "seed"} or ":" not in image:
            raise ValueError(f"run {run['run_id']} SenseCore image must use an immutable source-qualified tag")
    missing = sorted(key for key in required if not backend.get(key))
    if missing:
        raise ValueError(f"run {run['run_id']} backend is missing: {missing}")
    if backend["kind"] == "slurm" and not re.fullmatch(r"gpu:[A-Za-z0-9_-]+:[1-9][0-9]*", backend["gres"]):
        raise ValueError(f"run {run['run_id']} has invalid Slurm gres: {backend['gres']!r}")
    if backend["kind"] == "slurm" and not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", str(run["image_id"])):
        raise ValueError(f"run {run['run_id']} Slurm image_id must be a SIF sha256 digest")
    if backend["kind"] == "slurm":
        for field in ("mount_root", "apptainer_cache_dir", "apptainer_tmp_dir"):
            value = backend.get(field)
            if value is not None and not Path(str(value)).is_absolute():
                raise ValueError(f"run {run['run_id']} backend.{field} must be an absolute path")
        mount_root = Path(str(backend.get("mount_root", "/data")))
        for field, value in (
            ("storage.run_dir", storage["run_dir"]),
            ("backend.source_dir", backend["source_dir"]),
            ("backend.sif_path", backend["sif_path"]),
            ("backend.project_data_root", backend["project_data_root"]),
            ("backend.hf_home", backend["hf_home"]),
            ("backend.hf_datasets_cache", backend["hf_datasets_cache"]),
        ):
            path = Path(str(value))
            if not path.is_absolute() or not path.is_relative_to(mount_root):
                raise ValueError(
                    f"run {run['run_id']} {field} must be under declared mount_root {mount_root}"
                )
    for field in ("partition", "account", "qos"):
        if backend["kind"] == "slurm" and not re.fullmatch(r"[A-Za-z0-9_.-]+", str(backend[field])):
            raise ValueError(f"run {run['run_id']} has invalid Slurm {field}: {backend[field]!r}")


def source_identity(campaign: dict[str, Any]) -> str:
    """Resolve the runtime-tree identity, computing it when set to ``auto``."""
    configured = str(campaign.get("source_id", "auto"))
    if configured != "auto":
        return configured
    result = run_command(["bash", "scripts/source_identity.sh", "--runtime"], cwd=REPO_ROOT)
    return result.stdout.strip()


def provenance_identity(campaign_path: Path) -> dict[str, str]:
    """Keep Git and campaign provenance separate from runtime artifact identity."""
    commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).stdout.strip()
    campaign_id = run_command(
        ["bash", "scripts/source_identity.sh", "--campaign", str(campaign_path)], cwd=REPO_ROOT
    ).stdout.strip()
    return {"git_commit": commit, "campaign_id": campaign_id}


def selected_runs(campaign: dict[str, Any], names: Iterable[str]) -> list[dict[str, Any]]:
    """Return runs selected by ID, or every run when no IDs were supplied."""
    requested = set(names)
    runs = list(campaign["runs"])
    unknown = requested - {run["run_id"] for run in runs}
    if unknown:
        raise ValueError(f"unknown run IDs: {sorted(unknown)}")
    return [run for run in runs if not requested or run["run_id"] in requested]


def materialize_run(campaign: dict[str, Any], run: dict[str, Any], identity: str) -> dict[str, Any]:
    """Expand stable campaign placeholders in a copied run mapping.

    Supported placeholders are ``{source_id}``, ``{run_id}``, ``{project}``,
    and ``{campaign}``. This lets immutable Slurm source paths be derived only
    after the dirty-tree identity has been computed.
    """
    values = {
        "source_id": identity,
        "run_id": str(run["run_id"]),
        "project": str(campaign["project"]),
        "campaign": str(campaign["campaign"]),
    }

    def expand(value: Any) -> Any:
        """Recursively format strings while preserving container/value types."""
        if isinstance(value, str):
            return value.format(**values)
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    materialized = expand(run)
    validate_run(materialized)
    return materialized


def local_run_dir(campaign: dict[str, Any], run: dict[str, Any]) -> Path:
    """Return the controller-owned local metadata directory for one run."""
    root = Path(campaign.get("local_root", "outputs/experiment_campaigns"))
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / campaign["campaign"] / run["run_id"]


def frozen_source_identity(
    campaign: dict[str, Any], run: dict[str, Any], fallback: str
) -> str:
    """Return a prepared run's frozen source identity, or a supplied fallback."""
    local_dir = local_run_dir(campaign, run)
    for name in ("manifest.yaml", "control_manifest.yaml"):
        manifest_path = local_dir / name
        if manifest_path.is_file():
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            if payload.get("source_id"):
                return str(payload["source_id"])
    return fallback


def command_environment(
    campaign: dict[str, Any], run: dict[str, Any], source_id: str, attempt_id: str
) -> dict[str, str]:
    """Build the reviewed non-secret environment passed to the run launcher."""
    backend = run["backend"]
    env = {str(key): str(value) for key, value in run.get("env", {}).items()}
    env.update(
        {
            "PROJECT_NAME": str(campaign["project"]),
            "RUN_ID": str(run["run_id"]),
            "ATTEMPT_ID": attempt_id,
            "BACKEND": str(backend["kind"]),
            "SOURCE_ID": source_id,
            "RUNTIME_TREE_ID": source_id,
            "GIT_COMMIT": str(campaign.get("git_commit") or "unknown"),
            "CAMPAIGN_ID": str(campaign.get("campaign_id") or "unknown"),
            "CAMPAIGN_NAME": str(campaign["campaign"]),
            "IMAGE_ID": str(run["image_id"]),
            "OUTPUT_DIR": str(run["storage"]["run_dir"]),
            "NGPU": str(run.get("resources", {}).get("gpus", 1)),
        }
    )
    if backend["kind"] == "sensecore":
        env.update(
            {
                "BACKEND_JOB_ID": str(backend["job_name"]),
                "QUOTA_TYPE": str(backend["quota_type"]),
                "RESOURCE_SPEC": str(backend["worker_spec"]),
            }
        )
    else:
        env["QUOTA_TYPE"] = "normal"
        project_root = str(backend["project_data_root"])
        env.update(
            {
                "DATA_ROOT": str(backend["data_root"]),
                "PROJECT_DATA_ROOT": project_root,
                "HF_HOME": str(backend["hf_home"]),
                "HF_DATASETS_CACHE": str(backend["hf_datasets_cache"]),
                "CHECKPOINT_ROOT": f"{project_root}/checkpoints",
                "ELF_B_OWT_CHECKPOINT": f"{project_root}/checkpoints/ELF-B-owt-torch/checkpoint_95085",
                "SAVE_DIR": f"{project_root}/saved_models",
                "WANDB_CACHE_DIR": f"{project_root}/wandb_cache",
                "WANDB_DIR": f"{project_root}/wandb",
            }
        )
    return env


def launcher_command(
    campaign: dict[str, Any], run: dict[str, Any], source_id: str, attempt_id: str
) -> list[str]:
    """Build the container-side launcher command with frozen config overrides."""
    env = command_environment(campaign, run, source_id, attempt_id)
    command: list[str] = ["env"]
    command.extend(f"{key}={value}" for key, value in sorted(env.items()))
    command.extend(["bash", "scripts/cloud_train.sh", str(run["config"])])
    for override in run.get("config_overrides", []):
        command.extend(["--config_override", str(override)])
    return command


def resolved_run_overrides(run: dict[str, Any], remote_run_dir: str) -> list[str]:
    """Mirror launcher environment and explicit CLI overrides in execution order."""
    env = {str(key): str(value) for key, value in run.get("env", {}).items()}
    env.setdefault("RUN_ID", str(run["run_id"]))
    overrides = operational_overrides(env, remote_run_dir)
    overrides.extend(map(str, run.get("config_overrides", [])))
    return overrides


def prepare_run(
    campaign: dict[str, Any], run: dict[str, Any], source_id: str, *, attempt_id: str
) -> dict[str, Any]:
    """Freeze local control metadata before any scheduler mutation occurs."""
    local_dir = local_run_dir(campaign, run)
    manifest_path = local_dir / "manifest.yaml"
    legacy_manifest_path = local_dir / "control_manifest.yaml"
    if not manifest_path.exists() and legacy_manifest_path.is_file() and not legacy_manifest_path.is_symlink():
        legacy_manifest_path.replace(manifest_path)
    remote_run_dir = str(run["storage"]["run_dir"])
    overrides = resolved_run_overrides(run, remote_run_dir)
    resolved = resolved_config(str(run["config"]), overrides)
    command = launcher_command(campaign, run, source_id, attempt_id)
    manifest = {
        "schema_version": 1,
        "campaign": campaign["campaign"],
        "project": campaign["project"],
        "run_id": run["run_id"],
        "attempt_id": attempt_id,
        "created_at": utc_now(),
        "source_id": source_id,
        "runtime_tree_id": source_id,
        "git_commit": campaign.get("git_commit"),
        "campaign_id": campaign.get("campaign_id"),
        "image_id": run["image_id"],
        "config_path": run["config"],
        "config_overrides": list(run.get("config_overrides", [])),
        "resolved_config": resolved,
        "backend": run["backend"],
        "resources": run.get("resources", {}),
        "storage": run["storage"],
        "command": sanitize_command(command),
        "retry": run.get("retry", {"max_infra_retries": 0}),
    }
    if manifest_path.exists():
        existing = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        immutable_keys = ("campaign", "project", "run_id", "source_id", "image_id", "resolved_config", "backend")
        conflicts = [key for key in immutable_keys if existing.get(key) != manifest.get(key)]
        if conflicts:
            raise ValueError(f"existing control manifest conflicts in {conflicts}: {manifest_path}")
        base_manifest = existing
    else:
        atomic_write(manifest_path, manifest, yaml_format=True)
        base_manifest = manifest
    if not legacy_manifest_path.exists():
        legacy_manifest_path.symlink_to(manifest_path.name)

    effective = dict(base_manifest)
    effective["attempt_id"] = attempt_id
    effective["command"] = sanitize_command(command)
    attempt_path = local_dir / "attempts" / attempt_id / "attempt.yaml"
    legacy_attempt_path = attempt_path.with_name("control_attempt.yaml")
    if not attempt_path.exists() and legacy_attempt_path.is_file() and not legacy_attempt_path.is_symlink():
        legacy_attempt_path.replace(attempt_path)
    if attempt_path.exists():
        previous_attempt = yaml.safe_load(attempt_path.read_text(encoding="utf-8"))
        if previous_attempt != effective:
            raise ValueError(f"existing control attempt conflicts: {attempt_path}")
    else:
        atomic_write(attempt_path, effective, yaml_format=True)
        append_event(
            local_dir / "events.jsonl",
            {
                "timestamp": utc_now(),
                "run_id": run["run_id"],
                "attempt_id": attempt_id,
                "backend": run["backend"]["kind"],
                "event": "control_attempt_created",
                "payload": {"remote_run_dir": remote_run_dir},
            },
        )
    if not legacy_attempt_path.exists():
        legacy_attempt_path.symlink_to(attempt_path.name)
    status_path = local_dir / "status.json"
    if not status_path.exists():
        atomic_write(
            status_path,
            {"run_id": run["run_id"], "attempt_id": attempt_id, "state": "CREATED", "updated_at": utc_now()},
        )
    return effective


def record_submission(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str, backend_job_id: str
) -> None:
    """Atomically record scheduler acceptance in controller-owned metadata."""
    local_dir = local_run_dir(campaign, run)
    backend_payload = {
        "backend": run["backend"]["kind"],
        "backend_job_id": backend_job_id,
        "attempt_id": attempt_id,
    }
    store = ExperimentStateStore(local_dir)
    if store.read_submission(attempt_id) is None:
        record_submission_intent(campaign, run, attempt_id)
    store.reconcile_submission(
        project=str(campaign["project"]), run_id=str(run["run_id"]),
        attempt_id=attempt_id, backend_job_id=backend_job_id, state=RunState.QUEUED,
    )
    # Attempt-local mirror remains for old readers; the canonical scheduler
    # identity is the state store's root backend.json/submission.json.
    atomic_write(local_dir / "attempts" / attempt_id / "backend.json", backend_payload)


def record_submission_intent(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str
) -> dict[str, Any]:
    """Durably record scheduler mutation intent before crossing the API boundary."""
    local_dir = local_run_dir(campaign, run)
    token = f"{campaign['campaign']}/{run['run_id']}/{attempt_id}"
    return ExperimentStateStore(local_dir).begin_submission(
        project=str(campaign["project"]), run_id=str(run["run_id"]),
        attempt_id=attempt_id, backend=str(run["backend"]["kind"]),
        request={
            "submission_token": token,
            "scheduler_name": run["backend"].get("job_name", run["run_id"]),
        },
    )


def pending_submission_intent(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str
) -> dict[str, Any] | None:
    payload = ExperimentStateStore(local_run_dir(campaign, run)).read_submission(attempt_id)
    if payload and payload.get("state") == "SUBMITTING":
        # Compatibility with the original controller helper.
        return {**payload, **payload.get("request", {})}
    return None


def reconcile_submission(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str
) -> str | None:
    """Recover a scheduler identity after acceptance/local-record crash."""
    local_dir = local_run_dir(campaign, run)
    record_path = local_dir / "attempts" / attempt_id / "backend.json"
    if record_path.is_file():
        return str(json.loads(record_path.read_text(encoding="utf-8"))["backend_job_id"])
    root_record = local_dir / "backend.json"
    if root_record.is_file():
        payload = json.loads(root_record.read_text(encoding="utf-8"))
        if payload.get("backend_job_id") and payload.get("attempt_id") == attempt_id:
            atomic_write(record_path, payload)
            return str(payload["backend_job_id"])
    intent = pending_submission_intent(campaign, run, attempt_id)
    if not intent:
        return None
    backend = run["backend"]
    if backend["kind"] == "sensecore":
        matches = SenseCoreAdapter(backend_services(), parse_training_metric_line).find(run)
        job_id = backend["job_name"] if matches else None
    else:
        token = str(intent["submission_token"])
        expected_name = scheduler_job_name(str(run["run_id"]), attempt_id)
        query = "squeue -u $(id -un) -h -o '%i|%j|%k'"
        result = remote_exec(backend["ssh_alias"], query, check=False)
        matches = []
        for line in result.stdout.splitlines():
            fields = line.split("|", 2)
            if len(fields) == 3 and (fields[1] == expected_name or fields[2] == token):
                matches.append(fields[0])
        if not matches:
            accounting = remote_exec(
                backend["ssh_alias"],
                "sacct -S now-7days -u $(id -un) -X -n -P -o JobIDRaw,JobName",
                check=False,
            )
            matches = [
                line.split("|", 1)[0] for line in accounting.stdout.splitlines()
                if line.endswith(f"|{expected_name}") and line.split("|", 1)[0].isdigit()
            ]
        job_id = matches[0] if len(matches) == 1 else None
    if job_id:
        record_submission(campaign, run, attempt_id, str(job_id))
        return str(job_id)
    return None


def ensure_attempt_not_submitted(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str
) -> None:
    """Reject a second scheduler mutation for an already submitted attempt."""
    path = local_run_dir(campaign, run) / "attempts" / attempt_id / "backend.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        raise FileExistsError(
            f"attempt {attempt_id} already has backend job {payload.get('backend_job_id')}; "
            "use a new attempt ID"
        )
    intent = pending_submission_intent(campaign, run, attempt_id)
    if intent:
        raise RuntimeError(
            f"attempt {attempt_id} has an unresolved submission intent; run status to reconcile "
            "before creating another scheduler job"
        )


def remote_exec(alias: str, remote_command: str, *, check: bool = True) -> CommandResult:
    """Execute through a configured SSH alias and reuse a bounded master socket."""
    return run_command(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=auto",
            "-o", "ControlPersist=900", "-o", f"ControlPath={SSH_CONTROL_PATH}",
            alias, remote_command,
        ],
        check=check,
    )


def backend_services() -> BackendServices:
    """Inject controller IO boundaries into platform-specific adapters."""
    return BackendServices(
        repo_root=REPO_ROOT, script_dir=SCRIPT_DIR, ssh_control_path=SSH_CONTROL_PATH,
        run_command=run_command, remote_exec=remote_exec, local_run_dir=local_run_dir,
        backend_record=backend_record, summarize_run=summarize_run,
    )


def backend_record(campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Load the recorded scheduler identity for a submitted run."""
    path = local_run_dir(campaign, run) / "backend.json"
    if not path.is_file():
        raise FileNotFoundError(f"run has not been submitted: {run['run_id']}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("backend_job_id"):
        raise ValueError(f"invalid backend record: {path}")
    return payload


def update_observed_status(campaign: dict[str, Any], run: dict[str, Any], status: dict[str, Any]) -> None:
    """Persist one normalized scheduler observation without claiming model progress."""
    local_dir = local_run_dir(campaign, run)
    payload = dict(status)
    payload["updated_at"] = utc_now()
    atomic_write(local_dir / "status.json", payload)
    append_event(
        local_dir / "events.jsonl",
        {
            "timestamp": payload["updated_at"],
            "run_id": run["run_id"],
            "attempt_id": backend_record(campaign, run).get("attempt_id"),
            "backend": status["backend"],
            "backend_job_id": status["backend_job_id"],
            "event": "scheduler_observed",
            "payload": {"state": status["state"], "raw_state": status.get("raw_state")},
        },
    )


def parse_training_metric_line(line: str) -> dict[str, Any] | None:
    """Parse one rank-zero ``Step N`` log line into a structured metric record."""
    match = re.search(r"Step\s+(\d+):\s+(.*)$", line)
    if not match:
        return None
    record: dict[str, Any] = {"step": int(match.group(1))}
    key_map = {
        "loss": "train_loss",
        "l2": "train_l2_loss",
        "ce": "train_ce_loss",
        "plan": "train_plan_loss",
        "plan_aux": "train_plan_aux_loss",
        "emb_var": "train_plan_emb_batch_var",
        "pred_var": "train_plan_pred_batch_var",
        "emb_norm": "train_plan_emb_norm",
        "pred_norm": "train_plan_pred_norm",
        "lr": "lr",
        "steps/sec": "steps_per_sec",
    }
    for key, value in re.findall(r"([A-Za-z0-9_/]+)=([-+0-9.eE]+)", match.group(2)):
        if key in key_map:
            record[key_map[key]] = float(value)
    return record


def write_local_collection(campaign: dict[str, Any], run: dict[str, Any], summary: dict[str, Any]) -> None:
    """Persist the latest collected scientific/process observation locally."""
    atomic_write(local_run_dir(campaign, run) / "collection.json", summary)


def annotate_collection(
    summary: dict[str, Any], scheduler_status: dict[str, Any]
) -> dict[str, Any]:
    """Keep scheduler truth separate from possibly stale runtime state."""
    annotated = dict(summary)
    annotated["runtime_state"] = summary.get("state")
    annotated["scheduler_state"] = scheduler_status.get("state")
    return annotated


BACKENDS = BackendRegistry(
    WydSlurmBackend(backend_services()),
    SenseCoreAdapter(backend_services(), parse_training_metric_line),
)


def unsubmitted_status(campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Return controller state for a prepared run without raising on no job ID."""
    path = local_run_dir(campaign, run) / "status.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"run_id": run["run_id"], "state": "NOT_SUBMITTED"}
    payload.setdefault("backend", run["backend"]["kind"])
    payload.setdefault("backend_job_id", None)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse campaign path, operation, optional run filters, and dry-run policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "command", choices=(
            "prepare", "stage", "render", "submit", "status", "collect", "cancel",
            "observe", "logs", "decide", "assets-plan", "assets-verify",
        )
    )
    parser.add_argument("--run", action="append", default=[], help="limit to this run ID; repeatable")
    parser.add_argument("--attempt-id", default="attempt-001")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tail", type=int, default=100, help="maximum log lines per stream")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute one deterministic controller operation for selected campaign runs."""
    args = parse_args(argv)
    campaign = load_campaign(args.campaign)
    campaign.update(provenance_identity(args.campaign))
    default_identity = source_identity(campaign)
    selected = selected_runs(campaign, args.run)
    runs_with_identity = []
    for run in selected:
        identity = str(run.get("source_id", default_identity))
        if args.command in {"status", "collect", "cancel", "observe", "logs", "decide"}:
            identity = frozen_source_identity(campaign, run, identity)
        runs_with_identity.append((materialize_run(campaign, run, identity), identity))
    outputs: list[dict[str, Any]] = []
    for run, identity in runs_with_identity:
        backend_kind = run["backend"]["kind"]
        backend_adapter = BACKENDS.get(backend_kind)
        manifest = None
        if args.command in {"prepare", "stage", "render", "submit"}:
            manifest = prepare_run(campaign, run, identity, attempt_id=args.attempt_id)
        if args.command == "prepare":
            outputs.append({"run_id": run["run_id"], "state": "CREATED"})
        elif args.command == "stage":
            staged = backend_adapter.stage(campaign, run, identity)
            outputs.append({"run_id": run["run_id"], "staged": staged})
        elif args.command == "render":
            assert manifest is not None
            rendered = backend_adapter.render(manifest)
            outputs.append({"run_id": run["run_id"], "rendered": rendered})
        elif args.command == "submit":
            assert manifest is not None
            if not args.dry_run:
                recovered = reconcile_submission(campaign, run, args.attempt_id)
                if recovered:
                    outputs.append({"run_id": run["run_id"], "backend_job_id": recovered, "reconciled": True})
                    continue
                ensure_attempt_not_submitted(campaign, run, args.attempt_id)
                record_submission_intent(campaign, run, args.attempt_id)
            job_id = backend_adapter.submit(campaign, run, manifest, dry_run=args.dry_run)
            if not args.dry_run:
                record_submission(campaign, run, args.attempt_id, job_id)
            outputs.append({"run_id": run["run_id"], "backend_job_id": job_id})
        elif args.command == "status":
            reconcile_submission(campaign, run, args.attempt_id)
            backend_path = local_run_dir(campaign, run) / "backend.json"
            submitted = False
            if backend_path.is_file():
                submitted = bool(json.loads(backend_path.read_text(encoding="utf-8")).get("backend_job_id"))
            if not submitted:
                status = unsubmitted_status(campaign, run)
            else:
                status = backend_adapter.status(campaign, run)
                update_observed_status(campaign, run, status)
            outputs.append(status)
        elif args.command == "collect":
            status = backend_adapter.status(campaign, run)
            update_observed_status(campaign, run, status)
            summary = annotate_collection(backend_adapter.collect(campaign, run), status)
            write_local_collection(campaign, run, summary)
            outputs.append(summary)
        elif args.command == "logs":
            if args.tail < 1 or args.tail > 10000:
                raise ValueError("--tail must be between 1 and 10000")
            outputs.append(backend_adapter.logs(campaign, run, tail=args.tail))
        elif args.command == "cancel":
            status = backend_adapter.cancel(campaign, run)
            update_observed_status(campaign, run, status)
            outputs.append(status)
        elif args.command == "observe":
            reconcile_submission(campaign, run, args.attempt_id)
            backend_path = local_run_dir(campaign, run) / "backend.json"
            backend_payload = json.loads(backend_path.read_text(encoding="utf-8")) if backend_path.is_file() else {}
            if backend_payload.get("backend_job_id"):
                status = backend_adapter.status(campaign, run)
                update_observed_status(campaign, run, status)
                collection = annotate_collection(
                    backend_adapter.collect(campaign, run), status
                )
                write_local_collection(campaign, run, collection)
            else:
                status, collection = unsubmitted_status(campaign, run), None
            outputs.append({"run_id": run["run_id"], "scheduler": status, "model": collection})
        elif args.command == "decide":
            local_dir = local_run_dir(campaign, run)
            status = unsubmitted_status(campaign, run)
            collection_path = local_dir / "collection.json"
            collection = json.loads(collection_path.read_text(encoding="utf-8")) if collection_path.is_file() else {}
            attempts = [path for path in (local_dir / "attempts").glob("attempt-*") if path.is_dir()]
            retry = run.get("retry", {"max_infra_retries": 0})
            diagnostic = json.dumps(collection, ensure_ascii=False)
            decision = decide_next_action(
                status, retries_used=max(0, len(attempts) - 1),
                max_infra_retries=int(retry.get("max_infra_retries", 0)),
                diagnostic_text=diagnostic,
                completed_checkpoint=collection.get("latest_completed_checkpoint"),
            )
            payload = decision.to_dict()
            atomic_write(local_dir / "decision.json", payload)
            outputs.append({"run_id": run["run_id"], **payload})
        elif args.command in {"assets-plan", "assets-verify"}:
            requirements = plan_assets(
                str(run["config"]), list(map(str, run.get("config_overrides", [])))
            )
            result: dict[str, Any] = {
                "run_id": run["run_id"],
                "requirements": [asdict(item) for item in requirements],
            }
            if args.command == "assets-verify":
                env = command_environment(campaign, run, identity, args.attempt_id)
                hf_home = Path(env.get("HF_HOME", "/data/.cache/huggingface"))
                datasets_cache = Path(env.get("HF_DATASETS_CACHE", str(hf_home / "datasets")))
                if backend_kind == "slurm":
                    missing = []
                    alias = str(run["backend"]["ssh_alias"])
                    for requirement in requirements:
                        path = cache_path(requirement, hf_home, datasets_cache)
                        predicate = "-s" if requirement.kind == "file" else "-d"
                        check = remote_exec(
                            alias, shell_join(["test", predicate, str(path)]), check=False
                        )
                        if check.returncode != 0:
                            missing.append({**asdict(requirement), "path": str(path)})
                    result.update(
                        {"missing": missing, "verification": "remote-ssh", "verified_on": alias}
                    )
                else:
                    result.update(
                        {
                            "missing": None,
                            "verification": "requires-running-sensecore-worker",
                            "verified_on": None,
                        }
                    )
            outputs.append(result)
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
