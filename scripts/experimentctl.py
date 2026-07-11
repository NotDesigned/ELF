#!/usr/bin/env python
"""Prepare, submit, inspect, and collect ELF campaigns across SenseCore and Slurm."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
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
    append_event,
    atomic_write,
    resolved_config,
    sanitize_command,
    utc_now,
)
from summarize_experiments import summarize_run  # noqa: E402


NORMAL_STATES = {
    "PENDING": "QUEUED",
    "CONFIGURING": "QUEUED",
    "RUNNING": "RUNNING",
    "COMPLETING": "RUNNING",
    "COMPLETED": "SUCCEEDED",
    "PREEMPTED": "PREEMPTED",
    "REQUEUED": "QUEUED",
    "REQUEUE_FED": "QUEUED",
    "FAILED": "FAILED",
    "NODE_FAIL": "FAILED",
    "OUT_OF_MEMORY": "FAILED",
    "TIMEOUT": "FAILED",
    "CANCELLED": "CANCELLED",
}
TERMINAL_STATES = {"SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED"}
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


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
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
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        input=input_text,
        text=True,
        capture_output=True,
    )


def load_campaign(path: Path) -> dict[str, Any]:
    """Load and structurally validate one campaign YAML document."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
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
    missing = sorted(key for key in required if not backend.get(key))
    if missing:
        raise ValueError(f"run {run['run_id']} backend is missing: {missing}")
    if backend["kind"] == "slurm" and not re.fullmatch(r"gpu:[A-Za-z0-9_-]+:[1-9][0-9]*", backend["gres"]):
        raise ValueError(f"run {run['run_id']} has invalid Slurm gres: {backend['gres']!r}")
    for field in ("partition", "account", "qos"):
        if backend["kind"] == "slurm" and not re.fullmatch(r"[A-Za-z0-9_.-]+", str(backend[field])):
            raise ValueError(f"run {run['run_id']} has invalid Slurm {field}: {backend[field]!r}")


def source_identity(campaign: dict[str, Any]) -> str:
    """Resolve the campaign source identity, computing it when set to ``auto``."""
    configured = str(campaign.get("source_id", "auto"))
    if configured != "auto":
        return configured
    result = run_command(["bash", "scripts/source_identity.sh"], cwd=REPO_ROOT)
    return result.stdout.strip()


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
    manifest_path = local_run_dir(campaign, run) / "control_manifest.yaml"
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
    overrides = [f"output_dir={remote_run_dir}"]
    if "USE_WANDB" in env:
        overrides.append(f"use_wandb={env['USE_WANDB']}")
    if "GLOBAL_BATCH_SIZE" in env:
        overrides.append(f"global_batch_size={env['GLOBAL_BATCH_SIZE']}")
    if "BATCH_SIZE" in env:
        overrides.extend(["global_batch_size=null", f"batch_size={env['BATCH_SIZE']}"])
    for env_key, config_key in (
        ("NUM_WORKERS", "num_workers"),
        ("LOG_FREQ", "log_freq"),
        ("USE_COMPILE", "use_compile"),
    ):
        if env_key in env:
            overrides.append(f"{config_key}={env[env_key]}")
    overrides.extend(map(str, run.get("config_overrides", [])))
    return overrides


def prepare_run(
    campaign: dict[str, Any], run: dict[str, Any], source_id: str, *, attempt_id: str
) -> dict[str, Any]:
    """Freeze local control metadata before any scheduler mutation occurs."""
    local_dir = local_run_dir(campaign, run)
    manifest_path = local_dir / "control_manifest.yaml"
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

    effective = dict(base_manifest)
    effective["attempt_id"] = attempt_id
    effective["command"] = sanitize_command(command)
    attempt_path = local_dir / "attempts" / attempt_id / "control_attempt.yaml"
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
    status_path = local_dir / "status.json"
    if not status_path.exists():
        atomic_write(
            status_path,
            {"run_id": run["run_id"], "attempt_id": attempt_id, "state": "CREATED", "updated_at": utc_now()},
        )
    return effective


def shell_join(command: Iterable[str]) -> str:
    """Quote an argument vector into one POSIX shell command string."""
    return " ".join(shlex.quote(str(argument)) for argument in command)


def render_slurm_script(manifest: dict[str, Any]) -> str:
    """Render an explicit-partition Slurm script for one prepared attempt."""
    backend = manifest["backend"]
    resources = manifest.get("resources", {})
    run_dir = manifest["storage"]["run_dir"]
    source_dir = backend["source_dir"]
    sif_path = backend["sif_path"]
    command = shell_join(manifest["command"])
    cpus = int(resources.get("cpus", 8))
    return f"""#!/usr/bin/env bash
#SBATCH --partition={backend['partition']}
#SBATCH --account={backend['account']}
#SBATCH --qos={backend['qos']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres={backend['gres']}
#SBATCH --time={backend['time']}
#SBATCH --job-name={manifest['run_id']}
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
export APPTAINER_CACHEDIR=/data/apptainer/cache/liangluocheng
export APPTAINER_TMPDIR=/data/apptainer/tmp/liangluocheng
export BACKEND_JOB_ID="$SLURM_JOB_ID"
mkdir -p {shlex.quote(run_dir)}
attempt_log_dir={shlex.quote(f"{run_dir}/attempts/{manifest['attempt_id']}")}
mkdir -p "$attempt_log_dir"
exec > >(tee -a "$attempt_log_dir/slurm-$SLURM_JOB_ID.out") \
     2> >(tee -a "$attempt_log_dir/slurm-$SLURM_JOB_ID.err" >&2)
test -d {shlex.quote(source_dir)}
test -s {shlex.quote(sif_path)}
srun apptainer exec --nv \
  --bind /data:/data \
  --bind {shlex.quote(source_dir)}:/app \
  --pwd /app \
  {shlex.quote(sif_path)} \
  {command}
"""


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
    atomic_write(local_dir / "attempts" / attempt_id / "backend.json", backend_payload)
    atomic_write(local_dir / "backend.json", backend_payload)
    atomic_write(
        local_dir / "status.json",
        {
            "run_id": run["run_id"],
            "attempt_id": attempt_id,
            "state": "QUEUED",
            "backend_job_id": backend_job_id,
            "updated_at": utc_now(),
        },
    )
    append_event(
        local_dir / "events.jsonl",
        {
            "timestamp": utc_now(),
            "run_id": run["run_id"],
            "attempt_id": attempt_id,
            "backend": run["backend"]["kind"],
            "backend_job_id": backend_job_id,
            "event": "scheduler_accepted",
            "payload": {},
        },
    )


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


def remote_exec(alias: str, remote_command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Execute through a configured SSH alias and reuse a bounded master socket."""
    return run_command(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=auto",
            "-o", "ControlPersist=900", "-o", f"ControlPath={SSH_CONTROL_PATH}",
            alias, remote_command,
        ],
        check=check,
    )


def stage_slurm(campaign: dict[str, Any], run: dict[str, Any], source_id: str) -> None:
    """Stage an immutable source snapshot and verify the configured remote SIF."""
    backend = run["backend"]
    expected_suffix = f"/sources/{source_id}"
    if not str(backend["source_dir"]).endswith(expected_suffix):
        raise ValueError(f"source_dir must end with {expected_suffix}")
    remote_exec(
        backend["ssh_alias"],
        shell_join(["mkdir", "-p", backend["source_dir"], str(Path(run["storage"]["run_dir"]).parent)]),
    )
    rsync_command = [
        "rsync", "-a", "--delete", "-e",
        f"ssh -o ControlMaster=auto -o ControlPersist=900 -o ControlPath={SSH_CONTROL_PATH}",
        "--exclude=.git/", "--exclude=outputs/", "--exclude=runs/",
        "--exclude=checkpoints/", "--exclude=wandb/", "--exclude=*.log",
        f"{REPO_ROOT}/", f"{backend['ssh_alias']}:{backend['source_dir']}/",
    ]
    run_command(rsync_command)
    verify = remote_exec(
        backend["ssh_alias"],
        f"test -s {shlex.quote(backend['sif_path'])} && sha256sum {shlex.quote(backend['sif_path'])}",
    )
    actual_sha = verify.stdout.split()[0]
    expected_image = str(run["image_id"])
    if expected_image.startswith("sha256:") and actual_sha != expected_image.removeprefix("sha256:"):
        raise ValueError(f"SIF checksum mismatch: expected {expected_image}, got sha256:{actual_sha}")


def validate_slurm_live(run: dict[str, Any]) -> dict[str, str]:
    """Verify the declared WYD partition/GRES and user association before submit.

    The SSH alias is deliberately not used to infer placement. This function
    queries the exact campaign partition and confirms that its advertised GRES
    contains the requested GPU type, then confirms the configured account/QOS
    appears in the current user's association.
    """
    backend = run["backend"]
    partition = backend["partition"]
    expected_gpu = backend["gres"].split(":", 2)[1]
    query = (
        f"sinfo -h -p {shlex.quote(partition)} -o '%P|%a|%l|%G'; "
        "sacctmgr -n -P show assoc where user=$(id -un) format=User,Account,Partition,QOS,DefaultQOS"
    )
    result = remote_exec(backend["ssh_alias"], query)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    partition_lines = [line for line in lines if line.split("|", 1)[0].rstrip("*") == partition]
    if not partition_lines:
        raise RuntimeError(f"Slurm partition is not currently visible: {partition}")
    fields = partition_lines[0].split("|")
    if len(fields) < 4 or fields[1] != "up" or f"gpu:{expected_gpu}:" not in fields[3]:
        raise RuntimeError(f"Slurm partition/GRES is not currently usable: {partition}/{backend['gres']}")
    association_lines = [line for line in lines if line not in partition_lines and "|" in line]
    if not any(backend["account"] in line.split("|") and backend["qos"] in line.split("|") for line in association_lines):
        raise RuntimeError(
            f"Slurm association does not expose account={backend['account']} qos={backend['qos']}"
        )
    return {"partition": partition, "availability": fields[1], "gres": fields[3]}


def submit_slurm(campaign: dict[str, Any], run: dict[str, Any], manifest: dict[str, Any], *, dry_run: bool) -> str:
    """Stage controller metadata, submit via ``sbatch --parsable``, and return job ID."""
    local_dir = local_run_dir(campaign, run)
    script_path = local_dir / "attempts" / manifest["attempt_id"] / "job.sbatch"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(render_slurm_script(manifest), encoding="utf-8")
    if dry_run:
        return "DRY_RUN"
    backend = run["backend"]
    validate_slurm_live(run)
    remote_run_dir = run["storage"]["run_dir"]
    remote_script = f"{remote_run_dir}/controller-{manifest['attempt_id']}.sbatch"
    remote_exec(backend["ssh_alias"], shell_join(["mkdir", "-p", remote_run_dir]))
    run_command(
        [
            "rsync", "-a", "-e",
            f"ssh -o ControlMaster=auto -o ControlPersist=900 -o ControlPath={SSH_CONTROL_PATH}",
            str(script_path), f"{backend['ssh_alias']}:{remote_script}",
        ]
    )
    result = remote_exec(backend["ssh_alias"], f"sbatch --parsable {shlex.quote(remote_script)}")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError(f"unexpected sbatch response: {result.stdout!r}")
    return job_id


def sensecore_safe_command(arguments: list[str], sanitizer_mode: str) -> list[str]:
    """Build a proxy-free shell pipeline that sanitizes SCO JSON immediately."""
    safe_script = Path.home() / ".codex/skills/operate-sensecore/scripts/safe_sco.py"
    sco = shell_join(["env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy", "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY", *arguments])
    pipeline = f"{sco} | {shell_join([sys.executable, str(safe_script), sanitizer_mode])}"
    return ["bash", "-o", "pipefail", "-c", pipeline]


def sensecore_describe(run: dict[str, Any]) -> dict[str, Any]:
    """Return an allowlisted SenseCore job summary for the exact resource name."""
    backend = run["backend"]
    command = sensecore_safe_command(
        ["sco", "acp", "jobs", "describe", backend["job_name"], "--workspace-name", backend["workspace"], "-o", "json"],
        "job-summary",
    )
    result = run_command(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "sanitized SenseCore describe failed")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("SenseCore describe sanitizer returned a non-object")
    return payload


def sensecore_find(run: dict[str, Any]) -> list[dict[str, Any]]:
    """List allowlisted SenseCore summaries matching the exact intended name."""
    backend = run["backend"]
    command = sensecore_safe_command(
        [
            "sco", "acp", "jobs", "list", "--workspace-name", backend["workspace"],
            "--name", backend["job_name"], "--page-size", "5", "-o", "json",
        ],
        "job-list",
    )
    result = run_command(command, check=False)
    if result.returncode != 0:
        empty_result_message = "safe_sco: input was not valid JSON; raw response suppressed"
        if not result.stdout.strip() and result.stderr.strip() == empty_result_message:
            return []
        raise RuntimeError(result.stderr.strip() or "sanitized SenseCore list failed")
    payload = json.loads(result.stdout)
    if not isinstance(payload, list):
        raise ValueError("SenseCore list sanitizer returned a non-list")
    return [item for item in payload if item.get("name") == backend["job_name"]]


def submit_sensecore(
    campaign: dict[str, Any], run: dict[str, Any], manifest: dict[str, Any], *, dry_run: bool
) -> str:
    """Submit one exact SenseCore spot job after checking that its name is unused."""
    backend = run["backend"]
    if dry_run:
        return "DRY_RUN"
    existing = sensecore_find(run)
    if existing:
        raise FileExistsError(f"SenseCore job already exists: {backend['job_name']}")
    command_text = shell_join(manifest["command"])
    create = [
        "env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
        "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY",
        "sco", "acp", "jobs", "create",
        "--workspace-name", backend["workspace"],
        "--aec2-name", backend["aec2"],
        "--name", backend["job_name"],
        "--job-name", backend["display_name"],
        "--container-image-url", backend["image"],
        "--training-framework", "pytorch",
        "--worker-spec", backend["worker_spec"],
        "--worker-nodes", str(backend.get("worker_nodes", 1)),
        "--priority", str(backend.get("priority", "NORMAL")),
        "--quota-type", backend["quota_type"],
        "--storage-mount", backend["storage_mount"],
        "--wait",
        "--command", command_text,
    ]
    result = run_command(create, check=False)
    if result.returncode != 0:
        redactor = Path.home() / ".codex/skills/operate-sensecore/scripts/safe_sco.py"
        redacted = run_command([sys.executable, str(redactor), "redact-lines"], input_text=result.stderr, check=False)
        raise RuntimeError(redacted.stdout.strip() or "SenseCore create failed")
    summary = sensecore_describe(run)
    if summary.get("name") != backend["job_name"]:
        raise RuntimeError("SenseCore accepted create but exact job was not observable")
    return backend["job_name"]


def backend_record(campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Load the recorded scheduler identity for a submitted run."""
    path = local_run_dir(campaign, run) / "backend.json"
    if not path.is_file():
        raise FileNotFoundError(f"run has not been submitted: {run['run_id']}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("backend_job_id"):
        raise ValueError(f"invalid backend record: {path}")
    return payload


def status_slurm(campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Query Slurm accounting and normalize scheduler state independently of metrics."""
    record = backend_record(campaign, run)
    backend = run["backend"]
    job_id = str(record["backend_job_id"])
    result = remote_exec(
        backend["ssh_alias"],
        f"sacct -j {shlex.quote(job_id)} -X -n -P -o JobID,JobName,Partition,State,Elapsed,ExitCode",
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        queue = remote_exec(
            backend["ssh_alias"],
            f"squeue -j {shlex.quote(job_id)} -h -o '%i|%j|%P|%T|%M|0:0'",
            check=False,
        )
        lines = [line for line in queue.stdout.splitlines() if line.strip()]
    if not lines:
        state, raw = "UNKNOWN", "UNKNOWN"
        fields = [job_id, run["run_id"], backend["partition"], raw, "", ""]
    else:
        fields = lines[0].split("|")
        raw = fields[3].split()[0].rstrip("+")
        state = NORMAL_STATES.get(raw, "UNKNOWN")
        if raw == "COMPLETED" and len(fields) > 5 and fields[5] != "0:0":
            state = "FAILED"
    return {
        "run_id": run["run_id"],
        "backend": "slurm",
        "backend_job_id": job_id,
        "state": state,
        "raw_state": raw,
        "partition": fields[2] if len(fields) > 2 else backend["partition"],
        "elapsed": fields[4] if len(fields) > 4 else None,
        "exit_code": fields[5] if len(fields) > 5 else None,
    }


def status_sensecore(campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Query an exact sanitized SenseCore job and return normalized scheduler state."""
    record = backend_record(campaign, run)
    summary = sensecore_describe(run)
    state = summary.get("normalized_state", "UNKNOWN")
    cancellation_marker = local_run_dir(campaign, run) / "cancel_requested.json"
    if cancellation_marker.is_file() and summary.get("state") in {
        "SUSPENDING", "SUSPENDED", "DELETING", "DELETED"
    }:
        state = "CANCELLED"
    return {
        "run_id": run["run_id"],
        "backend": "sensecore",
        "backend_job_id": record["backend_job_id"],
        "state": state,
        "raw_state": summary.get("state"),
        "pool": summary.get("pool"),
        "spec": summary.get("spec"),
    }


def cancel_slurm(campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Cancel exactly the recorded nonterminal Slurm job and re-observe it."""
    current = status_slurm(campaign, run)
    if current["state"] in TERMINAL_STATES:
        return current
    backend = run["backend"]
    remote_exec(backend["ssh_alias"], f"scancel {shlex.quote(str(current['backend_job_id']))}")
    return status_slurm(campaign, run)


def cancel_sensecore(campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Stop exactly the recorded nonterminal SenseCore job and re-observe it."""
    current = status_sensecore(campaign, run)
    marker = local_run_dir(campaign, run) / "cancel_requested.json"
    atomic_write(
        marker,
        {
            "run_id": run["run_id"],
            "backend_job_id": current["backend_job_id"],
            "requested_at": utc_now(),
        },
    )
    if current.get("raw_state") in {"SUSPENDING", "SUSPENDED", "DELETING", "DELETED"}:
        current["state"] = "CANCELLED"
        return current
    if current["state"] in TERMINAL_STATES:
        return current
    backend = run["backend"]
    command = [
        "env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
        "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY",
        "sco", "acp", "jobs", "stop", backend["job_name"],
        "--workspace-name", backend["workspace"],
    ]
    result = run_command(command, check=False)
    if result.returncode != 0:
        redactor = Path.home() / ".codex/skills/operate-sensecore/scripts/safe_sco.py"
        redacted = run_command(
            [sys.executable, str(redactor), "redact-lines"], input_text=result.stderr, check=False
        )
        raise RuntimeError(redacted.stdout.strip() or "SenseCore stop failed")
    return status_sensecore(campaign, run)


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


def collect_slurm(campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Mirror small canonical Slurm records and summarize them locally.

    Checkpoints and generated sample payloads are deliberately excluded. This
    avoids requiring Python or a usable FUSE mount on the login host while
    preserving the files needed for campaign metrics.
    """
    backend = run["backend"]
    mirror = local_run_dir(campaign, run) / "collected_run"
    mirror.mkdir(parents=True, exist_ok=True)
    ssh_transport = (
        f"ssh -o ControlMaster=auto -o ControlPersist=900 -o ControlPath={SSH_CONTROL_PATH}"
    )
    run_command(
        [
            "rsync", "-a", "--delete", "-e", ssh_transport,
            "--include=*/", "--include=manifest.yaml", "--include=status.json",
            "--include=backend.json", "--include=train_metrics.jsonl",
            "--include=metrics.jsonl", "--exclude=*",
            f"{backend['ssh_alias']}:{run['storage']['run_dir']}/", f"{mirror}/",
        ]
    )
    summary = summarize_run(mirror)
    summary["collected_from"] = run["storage"]["run_dir"]
    summary["run_dir"] = run["storage"]["run_dir"]
    return summary


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


def collect_sensecore_logs(run: dict[str, Any]) -> dict[str, Any]:
    """Fetch and redact a bounded SenseCore log snapshot for metric observation."""
    backend = run["backend"]
    safe_script = Path.home() / ".codex/skills/operate-sensecore/scripts/safe_sco.py"
    command = [
        "timeout", "20s",
        "env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
        "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY",
        "sco", "acp", "jobs", "stream-logs", backend["job_name"],
        "--workspace-name", backend["workspace"],
    ]
    result = run_command(command, check=False)
    redacted = run_command(
        [sys.executable, str(safe_script), "redact-lines"],
        input_text=(result.stdout + "\n" + result.stderr),
        check=False,
    ).stdout
    metric_lines = [
        line for line in redacted.splitlines()
        if "Step " in line or "gPPL:" in line or "plan" in line.lower() and "ppl" in line.lower()
    ]
    parsed_metrics = [metric for line in metric_lines if (metric := parse_training_metric_line(line))]
    return {
        "run_id": run["run_id"],
        "backend": "sensecore",
        "model_observed": bool(parsed_metrics),
        "latest_metric": parsed_metrics[-1] if parsed_metrics else None,
        "metric_log_lines": metric_lines[-20:],
    }


def write_local_collection(campaign: dict[str, Any], run: dict[str, Any], summary: dict[str, Any]) -> None:
    """Persist the latest collected scientific/process observation locally."""
    atomic_write(local_run_dir(campaign, run) / "collection.json", summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse campaign path, operation, optional run filters, and dry-run policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "command", choices=("prepare", "stage", "render", "submit", "status", "collect", "cancel")
    )
    parser.add_argument("--run", action="append", default=[], help="limit to this run ID; repeatable")
    parser.add_argument("--attempt-id", default="attempt-001")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute one deterministic controller operation for selected campaign runs."""
    args = parse_args(argv)
    campaign = load_campaign(args.campaign)
    default_identity = source_identity(campaign)
    selected = selected_runs(campaign, args.run)
    runs_with_identity = []
    for run in selected:
        identity = str(run.get("source_id", default_identity))
        if args.command in {"status", "collect", "cancel"}:
            identity = frozen_source_identity(campaign, run, identity)
        runs_with_identity.append((materialize_run(campaign, run, identity), identity))
    outputs: list[dict[str, Any]] = []
    for run, identity in runs_with_identity:
        backend_kind = run["backend"]["kind"]
        manifest = None
        if args.command in {"prepare", "stage", "render", "submit"}:
            manifest = prepare_run(campaign, run, identity, attempt_id=args.attempt_id)
        if args.command == "prepare":
            outputs.append({"run_id": run["run_id"], "state": "CREATED"})
        elif args.command == "stage":
            if backend_kind == "slurm":
                stage_slurm(campaign, run, identity)
            outputs.append({"run_id": run["run_id"], "staged": backend_kind == "slurm"})
        elif args.command == "render":
            assert manifest is not None
            rendered = render_slurm_script(manifest) if backend_kind == "slurm" else shell_join(manifest["command"])
            outputs.append({"run_id": run["run_id"], "rendered": rendered})
        elif args.command == "submit":
            assert manifest is not None
            if not args.dry_run:
                ensure_attempt_not_submitted(campaign, run, args.attempt_id)
            job_id = (
                submit_slurm(campaign, run, manifest, dry_run=args.dry_run)
                if backend_kind == "slurm"
                else submit_sensecore(campaign, run, manifest, dry_run=args.dry_run)
            )
            if not args.dry_run:
                record_submission(campaign, run, args.attempt_id, job_id)
            outputs.append({"run_id": run["run_id"], "backend_job_id": job_id})
        elif args.command == "status":
            status = status_slurm(campaign, run) if backend_kind == "slurm" else status_sensecore(campaign, run)
            update_observed_status(campaign, run, status)
            outputs.append(status)
        elif args.command == "collect":
            summary = collect_slurm(campaign, run) if backend_kind == "slurm" else collect_sensecore_logs(run)
            write_local_collection(campaign, run, summary)
            outputs.append(summary)
        elif args.command == "cancel":
            status = cancel_slurm(campaign, run) if backend_kind == "slurm" else cancel_sensecore(campaign, run)
            update_observed_status(campaign, run, status)
            outputs.append(status)
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
