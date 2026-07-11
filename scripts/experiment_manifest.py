#!/usr/bin/env python
"""Create durable run/attempt metadata before a training process starts."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml
from experiment_run_manifest import build_run_manifest, comparable_manifest

IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SECRET_KEY_RE = re.compile(
    r"(?:secret|token|password|credential|access[_-]?key|api[_-]?key|proxy|authorization|cookie)",
    re.IGNORECASE,
)
URL_USERINFO_RE = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@")


class RunState(str, Enum):
    """Controller/runtime states persisted in ``status.json``.

    ``NOT_SUBMITTED`` is a read-model value for a run without an attempt or
    submission record. ``SUBMITTING`` is the durable outbox state written
    before contacting a scheduler, closing the otherwise unavoidable crash
    window between scheduler acceptance and local bookkeeping.
    """

    NOT_SUBMITTED = "NOT_SUBMITTED"
    CREATED = "CREATED"
    SUBMITTING = "SUBMITTING"
    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    PREEMPTED = "PREEMPTED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class LifecycleStatus:
    """Typed normalized status read from or written to ``status.json``."""

    project: str | None
    run_id: str | None
    attempt_id: str | None
    state: RunState
    updated_at: str | None = None
    exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class SubmissionIntent:
    """Durable scheduler submission outbox record for one attempt."""

    project: str
    run_id: str
    attempt_id: str
    backend: str
    request: dict[str, Any]
    state: str = "SUBMITTING"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    backend_job_id: str | None = None
    reconciled_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def utc_now() -> str:
    """Return the current UTC timestamp in RFC 3339 form with a ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _write_payload(handle: Any, payload: Any, *, yaml_format: bool) -> None:
    if yaml_format:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    else:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
        handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def _durable_temp(path: Path, payload: Any, *, yaml_format: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".yaml" if yaml_format else ".json"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        _write_payload(handle, payload, yaml_format=yaml_format)
    return temp_name


def atomic_write(path: Path, payload: Any, *, yaml_format: bool = False) -> None:
    """Durably replace a JSON or YAML file using fsync plus atomic rename."""
    temp_name = _durable_temp(path, payload, yaml_format=yaml_format)
    try:
        os.replace(temp_name, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_create(path: Path, payload: Any, *, yaml_format: bool = False) -> None:
    """Durably create an immutable file, failing instead of overwriting it."""
    temp_name = _durable_temp(path, payload, yaml_format=yaml_format)
    try:
        # Hard-linking a complete temporary file makes publication atomic while
        # preserving O_EXCL-like no-overwrite behavior for concurrent writers.
        os.link(temp_name, path)
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


def append_event_once(path: Path, event: dict[str, Any], event_id: str) -> bool:
    """Append an event once, using a filesystem lock as the idempotency gate.

    Returns ``True`` when a new line was appended and ``False`` when the event
    ID was already present. The lock is separate from ``events.jsonl`` so an
    atomic replacement or log rotation cannot invalidate the lock inode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            with path.open(encoding="utf-8") as existing:
                for line in existing:
                    try:
                        if json.loads(line).get("event_id") == event_id:
                            return False
                    except json.JSONDecodeError:
                        continue
        append_event(path, {**event, "event_id": event_id})
        return True


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


def _sanitize_mapping(value: Any, key: str = "") -> Any:
    """Return a JSON-safe submission request with secret-bearing values removed."""
    if SECRET_KEY_RE.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_mapping(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_mapping(item) for item in value]
    if isinstance(value, str):
        return URL_USERINFO_RE.sub(r"\g<scheme><redacted>@", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"cannot serialize submission request value of type {type(value).__name__}")


class ExperimentStateStore:
    """Reusable durable store shared by controllers and training runtimes.

    Immutable run/attempt manifests are kept separate from mutable scheduler
    submission and lifecycle observations. Submission uses a tiny durable
    outbox: write ``SUBMITTING`` first, contact the scheduler, then reconcile
    the returned job ID. Repeating either operation with the same data is safe.
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).resolve()

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.yaml"

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def status_path(self) -> Path:
        return self.run_dir / "status.json"

    @property
    def backend_path(self) -> Path:
        return self.run_dir / "backend.json"

    def attempt_dir(self, attempt_id: str) -> Path:
        _validate_identity("attempt_id", attempt_id)
        return self.run_dir / "attempts" / attempt_id

    def attempt_path(self, attempt_id: str) -> Path:
        return self.attempt_dir(attempt_id) / "attempt.yaml"

    def submission_path(self, attempt_id: str) -> Path:
        return self.attempt_dir(attempt_id) / "submission.json"

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"run manifest does not exist: {self.manifest_path}")
        return yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))

    def load_attempt(self, attempt_id: str) -> dict[str, Any]:
        path = self.attempt_path(attempt_id)
        if not path.is_file():
            raise FileNotFoundError(f"attempt manifest does not exist: {path}")
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def create_attempt(self, attempt: Mapping[str, Any]) -> Path:
        """Atomically publish one immutable attempt manifest."""
        attempt_id = str(attempt.get("attempt_id", ""))
        _validate_identity("attempt_id", attempt_id)
        if not self.manifest_path.is_file():
            raise FileNotFoundError("cannot create an attempt before the run manifest")
        manifest = self.load_manifest()
        if (
            attempt.get("project") != manifest.get("project")
            or attempt.get("run_id") != manifest.get("run_id")
        ):
            raise ValueError("attempt identity conflicts with run manifest")
        path = self.attempt_path(attempt_id)
        atomic_create(path, dict(attempt), yaml_format=True)
        return path

    def initialize_attempt_records(self, attempt_id: str) -> LifecycleStatus:
        """Idempotently initialize derived status, backend, and event records."""
        attempt = self.load_attempt(attempt_id)
        timestamp = attempt["created_at"]
        backend = (
            json.loads(self.backend_path.read_text(encoding="utf-8"))
            if self.backend_path.is_file()
            else {}
        )
        if backend.get("attempt_id") != attempt_id:
            atomic_write(
                self.backend_path,
                {
                    "backend": attempt["backend"],
                    "backend_job_id": attempt.get("backend_job_id"),
                    "attempt_id": attempt_id,
                },
            )
        existing_status = self.read_status(attempt_id)
        if (
            not self.status_path.is_file()
            or existing_status.attempt_id != attempt_id
            or existing_status.state == RunState.NOT_SUBMITTED
        ):
            status = self._write_status(
                project=attempt["project"],
                run_id=attempt["run_id"],
                attempt_id=attempt_id,
                state=RunState.CREATED,
                timestamp=timestamp,
            )
        else:
            status = existing_status
        append_event_once(
            self.events_path,
            {
                "timestamp": timestamp,
                "project": attempt["project"],
                "run_id": attempt["run_id"],
                "attempt_id": attempt_id,
                "backend": attempt["backend"],
                "backend_job_id": attempt.get("backend_job_id"),
                "event": "attempt_created",
                "payload": {
                    "command": attempt["command"],
                    "output_dir": str(self.run_dir),
                    "resume_from": attempt.get("resume_from"),
                },
            },
            f"attempt-created:{attempt_id}",
        )
        return status

    def read_status(self, attempt_id: str | None = None) -> LifecycleStatus:
        """Read normalized state without raising for an unsubmitted run."""
        if self.status_path.is_file():
            payload = json.loads(self.status_path.read_text(encoding="utf-8"))
            if attempt_id is None or payload.get("attempt_id") == attempt_id:
                return LifecycleStatus(
                    project=payload.get("project"),
                    run_id=payload.get("run_id"),
                    attempt_id=payload.get("attempt_id"),
                    state=RunState(payload.get("state", RunState.UNKNOWN.value)),
                    updated_at=payload.get("updated_at"),
                    exit_code=payload.get("exit_code"),
                )
        if attempt_id and self.attempt_path(attempt_id).is_file():
            attempt = self.load_attempt(attempt_id)
            return LifecycleStatus(
                project=attempt.get("project"),
                run_id=attempt.get("run_id"),
                attempt_id=attempt_id,
                state=RunState.CREATED,
            )
        manifest = (
            yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path.is_file()
            else {}
        )
        return LifecycleStatus(
            project=manifest.get("project"),
            run_id=manifest.get("run_id"),
            attempt_id=attempt_id,
            state=RunState.NOT_SUBMITTED,
        )

    def read_submission(self, attempt_id: str) -> dict[str, Any] | None:
        path = self.submission_path(attempt_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def _validate_attempt_identity(
        self, project: str, run_id: str, attempt_id: str
    ) -> dict[str, Any]:
        manifest = self.load_manifest()
        attempt = self.load_attempt(attempt_id)
        expected = (project, run_id, attempt_id)
        actual = (attempt.get("project"), attempt.get("run_id"), attempt.get("attempt_id"))
        if (manifest.get("project"), manifest.get("run_id")) != expected[:2] or actual != expected:
            raise ValueError("submission identity conflicts with run or attempt manifest")
        return attempt

    def _write_status(
        self,
        *,
        project: str,
        run_id: str,
        attempt_id: str,
        state: RunState,
        exit_code: int | None = None,
        timestamp: str | None = None,
    ) -> LifecycleStatus:
        status = LifecycleStatus(
            project=project,
            run_id=run_id,
            attempt_id=attempt_id,
            state=state,
            updated_at=timestamp or utc_now(),
            exit_code=exit_code,
        )
        atomic_write(self.status_path, status.to_dict())
        return status

    def begin_submission(
        self,
        *,
        project: str,
        run_id: str,
        attempt_id: str,
        backend: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist a submission intent before making an external scheduler call."""
        self._validate_attempt_identity(project, run_id, attempt_id)
        sanitized_request = _sanitize_mapping(request)
        path = self.submission_path(attempt_id)
        existing = self.read_submission(attempt_id)
        if existing:
            immutable = {
                "project": project,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "backend": backend,
                "request": sanitized_request,
            }
            conflicts = [key for key, value in immutable.items() if existing.get(key) != value]
            if conflicts:
                raise ValueError("existing submission intent conflicts in " + ", ".join(conflicts))
            intent = existing
        else:
            intent = SubmissionIntent(
                project=project,
                run_id=run_id,
                attempt_id=attempt_id,
                backend=backend,
                request=sanitized_request,
            ).to_dict()
            try:
                atomic_create(path, intent)
            except FileExistsError:
                # A concurrent controller won publication. Re-enter through
                # the same conflict checks instead of overwriting its intent.
                return self.begin_submission(
                    project=project,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    backend=backend,
                    request=request,
                )

        # Repair derived files/events after a crash, but never regress a
        # reconciled submission back to SUBMITTING.
        if intent.get("state") == "SUBMITTING":
            timestamp = intent["created_at"]
            atomic_write(
                self.backend_path,
                {"backend": backend, "backend_job_id": None, "attempt_id": attempt_id},
            )
            self._write_status(
                project=project,
                run_id=run_id,
                attempt_id=attempt_id,
                state=RunState.SUBMITTING,
                timestamp=timestamp,
            )
            append_event_once(
                self.events_path,
                {
                    "timestamp": timestamp,
                    "project": project,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "backend": backend,
                    "backend_job_id": None,
                    "event": "submission_intent_created",
                    "payload": {"request": sanitized_request},
                },
                f"submission-intent:{attempt_id}",
            )
        return intent

    def reconcile_submission(
        self,
        *,
        project: str,
        run_id: str,
        attempt_id: str,
        backend_job_id: str,
        state: RunState = RunState.QUEUED,
    ) -> dict[str, Any]:
        """Attach a scheduler job to a prior intent, safely and idempotently."""
        self._validate_attempt_identity(project, run_id, attempt_id)
        if not backend_job_id:
            raise ValueError("backend_job_id must not be empty")
        intent = self.read_submission(attempt_id)
        if not intent:
            raise FileNotFoundError("cannot reconcile scheduler job before submission intent")
        existing_job_id = intent.get("backend_job_id")
        if existing_job_id and existing_job_id != backend_job_id:
            raise ValueError(
                f"attempt is already reconciled to backend job {existing_job_id!r}"
            )
        timestamp = intent.get("reconciled_at") or utc_now()
        reconciled = {
            **intent,
            "state": "SUBMITTED",
            "backend_job_id": backend_job_id,
            "reconciled_at": timestamp,
        }
        atomic_write(self.submission_path(attempt_id), reconciled)
        atomic_write(
            self.backend_path,
            {
                "backend": intent["backend"],
                "backend_job_id": backend_job_id,
                "attempt_id": attempt_id,
            },
        )
        self._write_status(
            project=project,
            run_id=run_id,
            attempt_id=attempt_id,
            state=state,
            timestamp=timestamp,
        )
        append_event_once(
            self.events_path,
            {
                "timestamp": timestamp,
                "project": project,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "backend": intent["backend"],
                "backend_job_id": backend_job_id,
                "event": "submission_reconciled",
                "payload": {"state": state.value},
            },
            f"submission-reconciled:{attempt_id}:{backend_job_id}",
        )
        return reconciled

    def transition(
        self,
        *,
        project: str,
        run_id: str,
        attempt_id: str,
        state: RunState,
        event: str,
        payload: Mapping[str, Any] | None = None,
        event_id: str | None = None,
        exit_code: int | None = None,
    ) -> LifecycleStatus:
        """Write status atomically and append an optionally idempotent event."""
        self._validate_attempt_identity(project, run_id, attempt_id)
        timestamp = utc_now()
        status = self._write_status(
            project=project,
            run_id=run_id,
            attempt_id=attempt_id,
            state=state,
            exit_code=exit_code,
            timestamp=timestamp,
        )
        backend = (
            json.loads(self.backend_path.read_text(encoding="utf-8"))
            if self.backend_path.is_file()
            else {}
        )
        record = {
            "timestamp": timestamp,
            "project": project,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "backend": backend.get("backend"),
            "backend_job_id": backend.get("backend_job_id"),
            "event": event,
            "payload": _sanitize_mapping(payload or {}),
        }
        if event_id:
            append_event_once(self.events_path, record, event_id)
        else:
            append_event(self.events_path, record)
        return status


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

    store = ExperimentStateStore(args.output_dir)
    run_dir = store.run_dir
    manifest_path = store.manifest_path
    attempt_path = store.attempt_path(args.attempt_id)
    if attempt_path.exists():
        raise FileExistsError(
            f"attempt already exists: {attempt_path}; choose a new ATTEMPT_ID"
        )

    from experiment_projects import build_project_registry

    project = build_project_registry().get(args.project)
    config = project.resolve_config(args.config, args.config_override)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("resolved training command must not be empty")
    command = sanitize_command(command)

    manifest = build_run_manifest(
        project=args.project, run_id=args.run_id, created_at=utc_now(),
        config_path=args.config, resolved_config=config, source_id=args.source_id,
        runtime_tree_id=args.runtime_tree_id or args.source_id,
        git_commit=args.git_commit or None, campaign_id=args.campaign_id or None,
        campaign=args.campaign or None, image_id=args.image_id,
        run_dir=str(run_dir), max_infra_retries=args.max_infra_retries,
    )

    if manifest_path.exists():
        existing = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if comparable_manifest(existing) != comparable_manifest(manifest):
            raise ValueError("existing run manifest conflicts")
        manifest = existing
    else:
        try:
            atomic_create(manifest_path, manifest, yaml_format=True)
        except FileExistsError:
            # A concurrent prepare published the run identity. Re-enter the
            # normal immutable-manifest validation path before creating this attempt.
            return prepare(args)

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
    store.create_attempt(attempt)
    store.initialize_attempt_records(args.attempt_id)
    return {"manifest": str(manifest_path), "attempt": str(attempt_path)}


def record(args: argparse.Namespace) -> dict[str, Any]:
    """Atomically update normalized status and append a lifecycle event.

    Recording is allowed only after the run and attempt manifests exist and
    their identities agree with the command arguments.
    """
    _validate_identity("run_id", args.run_id)
    _validate_identity("attempt_id", args.attempt_id)
    store = ExperimentStateStore(args.output_dir)
    run_dir = store.run_dir
    manifest_path = store.manifest_path
    attempt_path = store.attempt_path(args.attempt_id)
    if not manifest_path.is_file() or not attempt_path.is_file():
        raise FileNotFoundError(
            f"cannot record lifecycle event before manifest/attempt exists in {run_dir}"
        )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("project") != args.project or manifest.get("run_id") != args.run_id:
        raise ValueError("lifecycle event identity conflicts with run manifest")

    payload: dict[str, Any] = {}
    if args.exit_code is not None:
        payload["exit_code"] = args.exit_code
    if args.reason:
        payload["reason"] = args.reason
    store.transition(
        project=args.project,
        run_id=args.run_id,
        attempt_id=args.attempt_id,
        state=RunState(args.state),
        event=args.event,
        payload=payload,
        exit_code=args.exit_code,
    )
    return {"status": str(store.status_path), "event": args.event}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments for initial run/attempt manifest creation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--backend-job-id", default="")
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--runtime-tree-id", default="")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--campaign-id", default="")
    parser.add_argument("--campaign", default="")
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--quota", default="")
    parser.add_argument("--resource-spec", default="")
    parser.add_argument("--max-infra-retries", type=int, default=2)
    parser.add_argument("--require-immutable-identities", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def parse_record_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments for a normalized lifecycle state transition."""
    parser = argparse.ArgumentParser(description="Record an experiment lifecycle transition.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--state",
        choices=tuple(state.value for state in RunState if state != RunState.NOT_SUBMITTED),
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
