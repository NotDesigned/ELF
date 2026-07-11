#!/usr/bin/env python
"""Create durable run/attempt metadata before an ELF training process starts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs.config import Config, apply_config_overrides, load_config_from_yaml  # noqa: E402


IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SECRET_KEY_RE = re.compile(
    r"(?:secret|token|password|credential|access[_-]?key|api[_-]?key|proxy|authorization|cookie)",
    re.IGNORECASE,
)
OPERATIONAL_CONFIG_FIELDS = {
    "output_dir",
    "resume",
    "wandb_run_name",
    "wandb_run_id",
    "wandb_resume",
}


def utc_now() -> str:
    """Return the current UTC timestamp in RFC 3339 form with a ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _plain(value: Any) -> Any:
    """Recursively convert config objects into YAML/JSON-safe primitive values.

    Dataclass-like config objects are detected through class annotations. A
    ``TypeError`` is raised for unknown values so manifests never silently
    stringify a scientific setting.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    annotations = getattr(type(value), "__annotations__", {})
    if annotations:
        return {name: _plain(getattr(value, name)) for name in annotations}
    raise TypeError(f"cannot serialize manifest value of type {type(value).__name__}")


def resolved_config(config_path: str, overrides: list[str]) -> dict[str, Any]:
    """Load inheritance, apply typed overrides, and return every Config field."""
    config = load_config_from_yaml(config_path)
    config = apply_config_overrides(config, overrides)
    return {name: _plain(getattr(config, name)) for name in Config.__annotations__}


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove retry/attempt-specific fields before manifest compatibility checks."""
    return {
        key: value
        for key, value in config.items()
        if key not in OPERATIONAL_CONFIG_FIELDS
    }


def sanitize_command(command: list[str]) -> list[str]:
    """Redact secret-bearing ``KEY=value`` and flag-following command arguments."""
    sanitized: list[str] = []
    redact_next = False
    for argument in command:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if "=" in argument:
            key, value = argument.split("=", 1)
            sanitized.append(f"{key}=<redacted>" if SECRET_KEY_RE.search(key) else argument)
            continue
        sanitized.append(argument)
        if argument.startswith("-") and SECRET_KEY_RE.search(argument):
            redact_next = True
    return sanitized


def _fsync_dir(path: Path) -> None:
    """Persist a directory entry update after an atomic file replacement."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, payload: Any, *, yaml_format: bool = False) -> None:
    """Durably replace a JSON or YAML file using fsync plus atomic rename.

    The temporary file is created in the destination directory so ``os.replace``
    stays on one filesystem. Any leftover temporary file is removed on failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".yaml" if yaml_format else ".json"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if yaml_format:
                yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
            else:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Append and fsync one compact JSON object to a lifecycle event stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_identity(label: str, value: str) -> None:
    """Require a scheduler/filesystem-safe run or attempt identity."""
    if not IDENTITY_RE.fullmatch(value):
        raise ValueError(
            f"{label}={value!r} is invalid; use 1-128 letters, digits, '.', '_' or '-'"
        )


def _require_immutable(label: str, value: str) -> None:
    """Reject missing, mutable, or placeholder source/image identities."""
    if not value or value.lower() in {"unknown", "latest", "runtime", "seed"}:
        raise ValueError(f"{label} must be an immutable, non-placeholder identity")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Create or validate a scientific run and append one immutable attempt.

    A new run writes its resolved manifest. A retry may reuse the run only when
    project, run ID, source, image, and scientific config are identical. Attempt
    directories are never overwritten. The function also initializes backend
    and normalized status files and records ``attempt_created``.

    Returns:
        Paths to the run manifest and newly created attempt manifest.
    """
    _validate_identity("run_id", args.run_id)
    _validate_identity("attempt_id", args.attempt_id)
    if args.require_immutable_identities:
        _require_immutable("source_id", args.source_id)
        _require_immutable("image_id", args.image_id)

    run_dir = Path(args.output_dir).resolve()
    attempt_dir = run_dir / "attempts" / args.attempt_id
    manifest_path = run_dir / "manifest.yaml"
    attempt_path = attempt_dir / "attempt.yaml"
    if attempt_path.exists():
        raise FileExistsError(
            f"attempt already exists: {attempt_path}; choose a new ATTEMPT_ID"
        )

    config = resolved_config(args.config, args.config_override)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("resolved training command must not be empty")
    command = sanitize_command(command)

    manifest = {
        "schema_version": 1,
        "project": args.project,
        "run_id": args.run_id,
        "created_at": utc_now(),
        "config_path": args.config,
        "resolved_config": config,
        "source_id": args.source_id,
        "image_id": args.image_id,
        "seed": config.get("seed"),
        "storage": {
            "run_dir": str(run_dir),
            "checkpoint_dir": str(run_dir),
        },
        "resume_policy": {
            "enabled": True,
            "max_infra_retries": args.max_infra_retries,
        },
    }

    if manifest_path.exists():
        existing = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        immutable_pairs = {
            "project": (existing.get("project"), manifest["project"]),
            "run_id": (existing.get("run_id"), manifest["run_id"]),
            "source_id": (existing.get("source_id"), manifest["source_id"]),
            "image_id": (existing.get("image_id"), manifest["image_id"]),
            "scientific config": (
                scientific_config(existing.get("resolved_config", {})),
                scientific_config(manifest["resolved_config"]),
            ),
        }
        conflicts = [label for label, (old, new) in immutable_pairs.items() if old != new]
        if conflicts:
            raise ValueError(
                "existing run manifest conflicts in " + ", ".join(conflicts)
            )
        manifest = existing
    else:
        atomic_write(manifest_path, manifest, yaml_format=True)

    attempt = {
        "schema_version": 1,
        "project": args.project,
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "created_at": utc_now(),
        "backend": args.backend,
        "backend_job_id": args.backend_job_id or None,
        "source_id": args.source_id,
        "image_id": args.image_id,
        "command": command,
        "resources": {
            "gpus": args.gpus,
            "nodes": args.nodes,
            "quota": args.quota,
            "resource_spec": args.resource_spec or None,
        },
        "resume_from": config.get("resume"),
    }
    atomic_write(attempt_path, attempt, yaml_format=True)
    atomic_write(
        run_dir / "backend.json",
        {
            "backend": args.backend,
            "backend_job_id": args.backend_job_id or None,
            "attempt_id": args.attempt_id,
        },
    )
    atomic_write(
        run_dir / "status.json",
        {
            "project": args.project,
            "run_id": args.run_id,
            "attempt_id": args.attempt_id,
            "state": "CREATED",
            "updated_at": utc_now(),
        },
    )
    event = {
        "timestamp": utc_now(),
        "project": args.project,
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "backend": args.backend,
        "backend_job_id": args.backend_job_id or None,
        "event": "attempt_created",
        "payload": {
            "command": command,
            "output_dir": str(run_dir),
            "resume_from": config.get("resume"),
        },
    }
    append_event(run_dir / "events.jsonl", event)
    return {"manifest": str(manifest_path), "attempt": str(attempt_path)}


def record(args: argparse.Namespace) -> dict[str, Any]:
    """Atomically update normalized status and append a lifecycle event.

    Recording is allowed only after the run and attempt manifests exist and
    their identities agree with the command arguments.
    """
    _validate_identity("run_id", args.run_id)
    _validate_identity("attempt_id", args.attempt_id)
    run_dir = Path(args.output_dir).resolve()
    manifest_path = run_dir / "manifest.yaml"
    attempt_path = run_dir / "attempts" / args.attempt_id / "attempt.yaml"
    if not manifest_path.is_file() or not attempt_path.is_file():
        raise FileNotFoundError(
            f"cannot record lifecycle event before manifest/attempt exists in {run_dir}"
        )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("project") != args.project or manifest.get("run_id") != args.run_id:
        raise ValueError("lifecycle event identity conflicts with run manifest")

    backend_path = run_dir / "backend.json"
    backend = json.loads(backend_path.read_text(encoding="utf-8")) if backend_path.is_file() else {}
    timestamp = utc_now()
    status = {
        "project": args.project,
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "state": args.state,
        "updated_at": timestamp,
    }
    if args.exit_code is not None:
        status["exit_code"] = args.exit_code
    atomic_write(run_dir / "status.json", status)

    payload: dict[str, Any] = {}
    if args.exit_code is not None:
        payload["exit_code"] = args.exit_code
    if args.reason:
        payload["reason"] = args.reason
    event = {
        "timestamp": timestamp,
        "project": args.project,
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "backend": backend.get("backend"),
        "backend_job_id": backend.get("backend_job_id"),
        "event": args.event,
        "payload": payload,
    }
    append_event(run_dir / "events.jsonl", event)
    return {"status": str(run_dir / "status.json"), "event": args.event}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments for initial run/attempt manifest creation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--backend", choices=("sensecore", "slurm", "local"), required=True)
    parser.add_argument("--backend-job-id", default="")
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--quota", default="spot")
    parser.add_argument("--resource-spec", default="")
    parser.add_argument("--max-infra-retries", type=int, default=2)
    parser.add_argument("--require-immutable-identities", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def parse_record_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments for a normalized lifecycle state transition."""
    parser = argparse.ArgumentParser(description="Record an ELF experiment lifecycle transition.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--state",
        choices=("CREATED", "QUEUED", "STARTING", "RUNNING", "EVALUATING", "SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED", "UNKNOWN"),
        required=True,
    )
    parser.add_argument("--event", required=True)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--reason", default="")
    return parser.parse_args(argv)


def main() -> None:
    """Dispatch the default prepare command or the explicit ``record`` command."""
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        result = record(parse_record_args(sys.argv[2:]))
    else:
        result = prepare(parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
