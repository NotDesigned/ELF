#!/usr/bin/env python
"""Prepare, submit, inspect, and collect project campaigns through registered backends."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

from experiment_control import (  # noqa: E402
    ExperimentStateStore,
    RunState,
    append_event,
    atomic_write,
    sanitize_command,
    utc_now,
    validate_identity,
)
from .campaign import load_and_resolve_campaign, resolve_campaign  # noqa: E402
from .policy import decide_next_action  # noqa: E402
from .run_manifest import build_run_manifest  # noqa: E402
from .research_contract import (  # noqa: E402
    evaluate_research_block,
    evaluate_research_run,
    validate_research_contract,
)
from experiment_control.backends import build_registry  # noqa: E402
from experiment_control.backends.services import BackendServices  # noqa: E402
from .projects import build_project_registry  # noqa: E402
from .summary import merge_local_scientific_evidence  # noqa: E402
from experiment_control.runner import (  # noqa: E402
    CommandResult,
    CommandRunner,
    SubprocessRunner,
)
from experiment_control.states import FailureClass, classify_failure  # noqa: E402
from experiment_control.outbox import (  # noqa: E402
    cancel_intent_path as package_cancel_intent_path,
    execute_cancel_outbox,
)
from experiment_control.observations import merge_terminal_observation  # noqa: E402


_COMMAND_RUNNER: CommandRunner = SubprocessRunner()
_COMMAND_DEADLINE: ContextVar[float | None] = ContextVar(
    "experimentctl_command_deadline", default=None
)
_ATTEMPT_SELECTOR = "__experiment_attempt_id"
_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED"})
_CAMPAIGN_SECRET_KEY_RE = re.compile(
    r"(?:^|[_-])(?:secret|password|credential|access[_-]?key(?:[_-](?:id|secret))?"
    r"|api[_-]?key|authorization|cookie|proxy|token)(?:$|[_-])",
    re.IGNORECASE,
)
_URL_USERINFO_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^/@\s]+@")
_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CAMPAIGN_REVISION_RE = re.compile(r"^campaign\.[0-9a-f]{64}$")


def _reject_embedded_credentials(value: Any, *, path: str) -> None:
    """Reject campaign fields that could persist credentials in run metadata."""
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key)
            child = f"{path}.{name}"
            if _CAMPAIGN_SECRET_KEY_RE.search(name):
                raise ValueError(f"credential-bearing campaign field is forbidden: {child}")
            _reject_embedded_credentials(item, path=child)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_embedded_credentials(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _URL_USERINFO_RE.search(value):
        raise ValueError(f"URL userinfo is forbidden in campaign metadata: {path}")


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
    deadline = _COMMAND_DEADLINE.get()
    timeout_seconds = None
    if deadline is not None:
        timeout_seconds = max(0.001, deadline - time.monotonic())
    return _COMMAND_RUNNER.run(
        command, cwd=cwd, check=check, input_text=input_text,
        timeout_seconds=timeout_seconds,
    )


@contextmanager
def command_deadline(timeout_seconds: float):
    """Apply one monotonic hard deadline to all backend commands in a poll."""
    if timeout_seconds <= 0:
        raise ValueError("command deadline must be greater than zero")
    token = _COMMAND_DEADLINE.set(time.monotonic() + timeout_seconds)
    try:
        yield
    finally:
        _COMMAND_DEADLINE.reset(token)


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
    try:
        validate_identity("campaign", str(payload["campaign"]))
    except ValueError as error:
        raise ValueError("campaign identity contains unsupported characters") from error
    if not isinstance(payload["runs"], list) or not payload["runs"]:
        raise ValueError("campaign runs must be a non-empty list")
    seen: set[str] = set()
    for run in payload["runs"]:
        # Campaign authoring deliberately retains controller-time placeholders
        # (including backend.job_name={run_id}) until immutable source and Run
        # identities are known.  Validate the recursively materialized shape,
        # while returning the authored unresolved payload for real execution.
        materialize_run(payload, run, "validation-source")
        if run["run_id"] in seen:
            raise ValueError(f"duplicate run_id: {run['run_id']}")
        seen.add(run["run_id"])
    _reject_embedded_credentials(payload, path="campaign")
    validate_research_contract(payload)
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _regular_file_digest(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"local evidence input is not a regular file: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return "sha256:" + digest.hexdigest()


def _load_identity_mapping(path: Path) -> dict[str, Any]:
    """Read one reviewed regular YAML/JSON identity file without following links."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"reviewed identity input is not a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    payload = yaml.safe_load(b"".join(chunks).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"reviewed identity input must be a mapping: {path}")
    return payload


def _read_regular_at(directory_fd: int, name: str, *, display: str) -> bytes:
    descriptor = os.open(
        name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"local evidence input is not a regular file: {display}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _regular_file_digest_at(directory_fd: int, name: str, *, display: str) -> str:
    return _sha256_bytes(_read_regular_at(directory_fd, name, display=display))


def _optional_regular_digest_at(
    directory_fd: int, name: str, *, display: str,
) -> str | None:
    try:
        return _regular_file_digest_at(directory_fd, name, display=display)
    except FileNotFoundError:
        return None


def _local_tree_manifest_at(
    root_fd: int, *, display: str, snapshot_fd: int | None = None,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """Read an anchored tree without following links, optionally snapshotting it."""
    records: list[dict[str, Any]] = []
    for name in sorted(os.listdir(root_fd)):
        relative = f"{prefix}{name}"
        item = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if stat.S_ISLNK(item.st_mode):
            raise ValueError(f"local evidence rejects symlink: {display}/{relative}")
        if stat.S_ISDIR(item.st_mode):
            records.append({
                "path": relative + "/", "kind": "directory",
                "mode": stat.S_IMODE(item.st_mode),
                "mtime_ns": item.st_mtime_ns,
            })
            child_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
            destination_fd: int | None = None
            try:
                if snapshot_fd is not None:
                    os.mkdir(name, mode=0o700, dir_fd=snapshot_fd)
                    destination_fd = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=snapshot_fd,
                    )
                records.extend(_local_tree_manifest_at(
                    child_fd, display=display, snapshot_fd=destination_fd,
                    prefix=relative + "/",
                ))
                if destination_fd is not None:
                    os.fchmod(destination_fd, stat.S_IMODE(item.st_mode))
                    os.utime(
                        destination_fd,
                        ns=(item.st_atime_ns, item.st_mtime_ns),
                    )
                    os.fsync(destination_fd)
            finally:
                if destination_fd is not None:
                    os.close(destination_fd)
                os.close(child_fd)
            continue
        if not stat.S_ISREG(item.st_mode):
            raise ValueError(f"local evidence rejects special file: {display}/{relative}")
        payload = _read_regular_at(root_fd, name, display=f"{display}/{relative}")
        if snapshot_fd is not None:
            copied_fd = os.open(
                name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600, dir_fd=snapshot_fd,
            )
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(copied_fd, view)
                    view = view[written:]
                os.fchmod(copied_fd, stat.S_IMODE(item.st_mode))
                os.utime(
                    copied_fd, ns=(item.st_atime_ns, item.st_mtime_ns),
                )
                os.fsync(copied_fd)
            finally:
                os.close(copied_fd)
        records.append({
            "path": relative, "kind": "file", "size": item.st_size,
            "mode": stat.S_IMODE(item.st_mode),
            "mtime_ns": item.st_mtime_ns,
            "sha256": _sha256_bytes(payload),
        })
    return records


def _stable_local_evidence_paths(
    value: Any, *, logical_root: Path, snapshot_roots: tuple[str, ...],
) -> Any:
    """Project private snapshot paths back to one reviewed logical run root."""
    def project(relative: str) -> str:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "local evidence summary contains an unsafe private path"
            )
        return str(logical_root / path)

    if isinstance(value, dict):
        return {
            str(_stable_local_evidence_paths(
                str(key), logical_root=logical_root,
                snapshot_roots=snapshot_roots,
            )): _stable_local_evidence_paths(
                item, logical_root=logical_root, snapshot_roots=snapshot_roots,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _stable_local_evidence_paths(
                item, logical_root=logical_root, snapshot_roots=snapshot_roots,
            )
            for item in value
        ]
    if not isinstance(value, str):
        return value
    for root in snapshot_roots:
        if value == root:
            return str(logical_root)
        prefix = root.rstrip("/") + "/"
        if value.startswith(prefix):
            return project(value[len(prefix):])
    proc_match = re.fullmatch(r"/proc/self/fd/[0-9]+(?:/(.*))?", value)
    if proc_match is not None:
        relative = proc_match.group(1)
        return str(logical_root) if not relative else project(relative)
    if (
        value.startswith("/proc/")
        or "elf-local-evidence-" in value
        or "{controller_snapshot}" in value
    ):
        raise ValueError("local evidence summary contains an unmappable private path")
    return value


def _local_evidence_input_digest(
    *, identity_digests: dict[str, str], durable: list[dict[str, Any]],
    old_digest: str | None, controller_snapshot_sha256: str,
) -> str:
    encoded = json.dumps({
        "schema_version": 1,
        "controller_snapshot_sha256": controller_snapshot_sha256,
        "identity_files": identity_digests,
        "durable_artifacts": durable,
        "old_collection_sha256": old_digest,
    }, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


@contextmanager
def _locked_attempt_directory(attempt_dir: Path):
    descriptor = os.open(
        attempt_dir,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield descriptor
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _atomic_replace_at(
    directory_fd: int, name: str, payload: bytes, *, display: str,
) -> None:
    temp_name = f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
    descriptor = os.open(
        temp_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    published = False
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.rename(
            temp_name, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
        )
        published = True
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass


def _validate_local_evidence_identity(
    *, campaign_path: Path, identity_root: Path,
    run_id: str, attempt_id: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    authored = _load_identity_mapping(campaign_path)
    campaign = resolve_campaign(authored)
    project = str(campaign.get("project") or "")
    campaign_id = str(campaign.get("campaign") or "")
    if campaign.get("schema_version") != 1 or not project or not campaign_id:
        raise ValueError("reviewed campaign identity is incomplete")
    matches = [
        item for item in campaign.get("runs", [])
        if isinstance(item, dict) and item.get("run_id") == run_id
    ]
    if len(matches) != 1:
        raise ValueError("reviewed campaign does not select exactly one requested Run")
    run_manifest = _load_identity_mapping(identity_root / "run" / "manifest.yaml")
    attempt = _load_identity_mapping(identity_root / "attempt" / "attempt.yaml")
    backend = _load_identity_mapping(identity_root / "attempt" / "backend.json")
    expectations = (
        (run_manifest, "project", project),
        (run_manifest, "run_id", run_id),
        (run_manifest, "campaign", campaign_id),
        (attempt, "project", project),
        (attempt, "run_id", run_id),
        (attempt, "attempt_id", attempt_id),
    )
    for payload, key, expected in expectations:
        if str(payload.get(key) or "") != expected:
            raise ValueError(
                f"reviewed {key} conflicts with exact local evidence identity"
            )
    for required in ("attempt_id", "backend", "backend_job_id"):
        if required not in backend:
            raise ValueError(f"reviewed backend is missing required {required}")
    if str(backend.get("attempt_id") or "") != attempt_id:
        raise ValueError(
            "reviewed backend attempt_id conflicts with exact local evidence identity"
        )
    if not isinstance(backend.get("backend"), str) or not backend["backend"].strip():
        raise ValueError("reviewed backend backend must be a non-empty string")
    for key, expected in (("project", project), ("run_id", run_id)):
        if key in backend and str(backend.get(key) or "") != expected:
            raise ValueError(
                f"reviewed backend {key} conflicts with exact local evidence identity"
            )
    status_path = identity_root / "attempt" / "status.json"
    if status_path.is_file():
        status = _load_identity_mapping(status_path)
        for key, expected in (
            ("project", project), ("run_id", run_id),
            ("attempt_id", attempt_id),
        ):
            if str(status.get(key) or "") != expected:
                raise ValueError(
                    f"reviewed status {key} conflicts with exact local evidence identity"
                )
    collection_path = identity_root / "attempt" / "collection.json"
    previous = (
        _load_identity_mapping(collection_path) if collection_path.is_file() else None
    )
    if previous is not None:
        for key, expected in (
            ("project", project), ("run_id", run_id),
            ("attempt_id", attempt_id),
        ):
            value = previous.get(key)
            if value is not None and str(value) != expected:
                raise ValueError(
                    f"reviewed collection {key} conflicts with exact Attempt"
                )
    return campaign, previous


def rebuild_local_evidence(args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild only exact-Attempt collection.json from reviewed local inputs."""
    if len(args.run) != 1:
        raise ValueError("refresh-evidence-local requires exactly one --run")
    if args.local_root is None or not args.local_root.is_absolute():
        raise ValueError("refresh-evidence-local requires an absolute --local-root")
    if args.identity_root is None or not args.identity_root.is_absolute():
        raise ValueError("refresh-evidence-local requires an absolute --identity-root")
    if args.campaign.resolve() != (args.identity_root / "campaign.yml").resolve():
        raise ValueError("campaign must be the reviewed private identity copy")
    if (
        args.expected_input_digest is not None
        and _SHA256_DIGEST_RE.fullmatch(args.expected_input_digest) is None
    ):
        raise ValueError("--expected-input-digest must be a sha256 digest")
    if (
        args.expected_current_collection_digest is not None
        and _SHA256_DIGEST_RE.fullmatch(
            args.expected_current_collection_digest
        ) is None
    ):
        raise ValueError(
            "--expected-current-collection-digest must be a sha256 digest"
        )
    controller_digest = os.environ.get("ML_EXPD_CONTROLLER_SNAPSHOT_SHA256", "")
    if _SHA256_DIGEST_RE.fullmatch(controller_digest) is None:
        raise ValueError("reviewed controller snapshot identity is unavailable")
    run_id = str(args.run[0])
    attempt_id = str(args.attempt_id)
    validate_identity("run_id", run_id)
    validate_identity("attempt_id", attempt_id)
    campaign, previous = _validate_local_evidence_identity(
        campaign_path=args.campaign, identity_root=args.identity_root,
        run_id=run_id, attempt_id=attempt_id,
    )
    identity_files = (
        "campaign.yml", "run/manifest.yaml", "attempt/attempt.yaml",
        "attempt/backend.json",
    )
    identity_digests = {
        name: _regular_file_digest(args.identity_root / name)
        for name in identity_files
    }
    reviewed_collection = args.identity_root / "attempt" / "collection.json"
    reviewed_status = args.identity_root / "attempt" / "status.json"
    if reviewed_status.is_file():
        identity_digests["attempt/status.json"] = _regular_file_digest(
            reviewed_status
        )
    reviewed_old_digest = (
        _regular_file_digest(reviewed_collection)
        if reviewed_collection.is_file() else None
    )
    expected_current_digest = (
        args.expected_current_collection_digest
        if args.expected_current_collection_digest is not None
        else reviewed_old_digest
    )
    if (
        args.expected_current_collection_digest is not None
        and reviewed_old_digest is None
    ):
        raise ValueError("recovery baseline collection is required")
    if reviewed_old_digest is not None:
        identity_digests["attempt/collection.json"] = reviewed_old_digest
    local_root = args.local_root
    if local_root.is_symlink() or not local_root.is_dir():
        raise ValueError("local_root must be a real already-local data directory")
    run_dir = local_root / str(campaign["campaign"]) / run_id
    attempt_dir = run_dir / "attempts" / attempt_id
    collected_run = attempt_dir / "collected_run"
    collection_path = attempt_dir / "collection.json"
    for directory in (
        local_root / str(campaign["campaign"]), run_dir,
        run_dir / "attempts", attempt_dir,
    ):
        metadata = directory.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"local evidence path is not a real directory: {directory}")
    with tempfile.TemporaryDirectory(prefix="elf-local-evidence-") as temporary:
        snapshot_path = Path(temporary) / "collected_run"
        snapshot_path.mkdir(mode=0o700)
        snapshot_fd = os.open(
            snapshot_path,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            with _locked_attempt_directory(attempt_dir) as attempt_fd:
                old_digest = _optional_regular_digest_at(
                    attempt_fd, "collection.json", display=str(collection_path),
                )
                if old_digest != expected_current_digest:
                    raise ValueError(
                        "original collection changed after private review snapshot"
                    )
                collected_fd = os.open(
                    "collected_run",
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=attempt_fd,
                )
                try:
                    durable_before = _local_tree_manifest_at(
                        collected_fd, display=str(collected_run),
                        snapshot_fd=snapshot_fd,
                    )
                    if not any(
                        record["kind"] == "file" for record in durable_before
                    ):
                        raise ValueError(
                            f"collected_run has no durable regular files: {collected_run}"
                        )
                    input_digest = _local_evidence_input_digest(
                        identity_digests=identity_digests,
                        durable=durable_before,
                        old_digest=old_digest,
                        controller_snapshot_sha256=controller_digest,
                    )
                    if (
                        args.expected_input_digest is not None
                        and args.expected_input_digest != input_digest
                    ):
                        raise ValueError(
                            "local evidence input digest changed after Action review"
                        )
                    summary_root = Path(f"/proc/self/fd/{snapshot_fd}")
                    summary = PROJECTS.get(
                        str(campaign["project"])
                    ).summarize(summary_root)
                    summary = _stable_local_evidence_paths(
                        summary,
                        logical_root=collected_run,
                        snapshot_roots=(str(summary_root), str(snapshot_path)),
                    )
                    summary["run_dir"] = str(collected_run)
                    if summary.get("project") != campaign["project"]:
                        raise ValueError(
                            "local summary project conflicts with reviewed identity"
                        )
                    if summary.get("run_id") != run_id:
                        raise ValueError(
                            "local summary run_id conflicts with reviewed identity"
                        )
                    if summary.get("attempt_id") not in {None, attempt_id}:
                        raise ValueError(
                            "local summary attempt_id conflicts with reviewed identity"
                        )
                    if (
                        str(summary.get("state") or "").upper()
                        not in _TERMINAL_STATES
                    ):
                        raise ValueError(
                            "local summary does not prove an exact terminal Attempt"
                        )
                    summary.update({
                        "project": campaign["project"], "run_id": run_id,
                        "attempt_id": attempt_id,
                    })
                    rebuilt = merge_local_scientific_evidence(previous, summary)
                    rebuilt = _stable_local_evidence_paths(
                        rebuilt,
                        logical_root=collected_run,
                        snapshot_roots=(str(summary_root), str(snapshot_path)),
                    )
                    durable_after = _local_tree_manifest_at(
                        collected_fd, display=str(collected_run),
                    )
                    stable_digest = _local_evidence_input_digest(
                        identity_digests=identity_digests,
                        durable=durable_after,
                        old_digest=old_digest,
                        controller_snapshot_sha256=controller_digest,
                    )
                    if (
                        stable_digest != input_digest
                        or durable_after != durable_before
                    ):
                        raise ValueError(
                            "local durable evidence changed while rebuilding"
                        )
                finally:
                    os.close(collected_fd)
                proposed_bytes = (
                    json.dumps(
                        rebuilt, ensure_ascii=False, sort_keys=True,
                        allow_nan=False,
                    ) + "\n"
                ).encode("utf-8")
                expected_new_digest = _sha256_bytes(proposed_bytes)
                if not args.dry_run:
                    actual_old_digest = _optional_regular_digest_at(
                        attempt_fd, "collection.json", display=str(collection_path),
                    )
                    if actual_old_digest != old_digest:
                        raise ValueError(
                            "collection changed before atomic rebuild write"
                        )
                    _atomic_replace_at(
                        attempt_fd, "collection.json", proposed_bytes,
                        display=str(collection_path),
                    )
                    new_digest = _regular_file_digest_at(
                        attempt_fd, "collection.json", display=str(collection_path),
                    )
                    if new_digest != expected_new_digest:
                        raise ValueError(
                            "atomic rebuild produced an unexpected collection digest"
                        )
                else:
                    new_digest = expected_new_digest
        finally:
            os.close(snapshot_fd)
    return {
        "project": campaign["project"], "run_id": run_id,
        "attempt_id": attempt_id, "collection_path": str(collection_path),
        "input_digest": input_digest, "old_digest": old_digest,
        "recovery_baseline_digest": (
            reviewed_old_digest
            if args.expected_current_collection_digest is not None else None
        ),
        "new_digest": new_digest,
        "expected_new_collection_digest": expected_new_digest,
        "atomic_collection_replace": True,
        "write_protocol": "dirfd-fsync-rename-v1",
        "dry_run": bool(args.dry_run),
        "local_only": True, "backend_accessed": False,
        "scheduler_accessed": False,
        "controller_snapshot_sha256": controller_digest,
    }


def validate_run(run: Any, *, project: str | None = None) -> None:
    """Validate one backend-neutral run and reject secret-bearing settings."""
    if not isinstance(run, dict):
        raise ValueError("each campaign run must be a mapping")
    for key in ("run_id", "config", "backend", "storage", "image_id"):
        if not run.get(key):
            raise ValueError(f"run is missing {key}")
    try:
        validate_identity("run_id", str(run["run_id"]))
    except ValueError as error:
        raise ValueError(f"invalid run_id: {run['run_id']!r}") from error
    backend = run["backend"]
    if not isinstance(backend, dict) or backend.get("kind") not in BACKENDS.kinds:
        raise ValueError(
            f"run {run['run_id']} backend.kind must be one of {sorted(BACKENDS.kinds)}"
        )
    env = run.get("env", {})
    if not isinstance(env, dict):
        raise ValueError(f"run {run['run_id']} env must be a mapping")
    allowed_env = PROJECTS.get(project).safe_env_keys if project is not None else frozenset()
    forbidden = [
        key for key in env
        if key not in allowed_env or _CAMPAIGN_SECRET_KEY_RE.search(key)
    ]
    if forbidden:
        raise ValueError(f"run {run['run_id']} has forbidden env keys: {sorted(forbidden)}")
    for value in env.values():
        if "\n" in str(value) or "\x00" in str(value):
            raise ValueError(f"run {run['run_id']} env values must be single-line text")
    storage = run["storage"]
    required_storage = {
        "run_dir", "data_root", "project_data_root", "hf_home", "hf_datasets_cache"
    }
    if not isinstance(storage, dict):
        raise ValueError(f"run {run['run_id']} storage must be a mapping")
    missing_storage = sorted(key for key in required_storage if not storage.get(key))
    if missing_storage:
        raise ValueError(f"run {run['run_id']} storage is missing: {missing_storage}")
    for field in required_storage:
        if not Path(str(storage[field])).is_absolute():
            raise ValueError(f"run {run['run_id']} storage.{field} must be an absolute path")
    for value in [*backend.values(), *storage.values()]:
        if isinstance(value, str) and ("\n" in value or "\x00" in value):
            raise ValueError(f"run {run['run_id']} backend/storage values must be single-line text")
    _reject_embedded_credentials(run, path=f"run {run['run_id']}")
    BACKENDS.get(str(backend["kind"])).validate(run)
    if project is not None:
        PROJECTS.get(project).validate_run(run)


def source_identity(campaign: dict[str, Any]) -> str:
    """Resolve the runtime-tree identity, computing it when set to ``auto``."""
    configured = str(campaign.get("source_id", "auto"))
    if configured != "auto":
        return configured
    baked = os.environ.get("ELF_SOURCE_ID", "")
    if baked and baked != "unknown":
        return baked
    bundle = PROJECTS.get(str(campaign["project"])).source_bundle(REPO_ROOT)
    if not bundle.identity_command:
        raise ValueError(f"project {campaign['project']} has no source identity command")
    result = run_command(list(bundle.identity_command), cwd=bundle.root)
    return result.stdout.strip()


def provenance_identity(
    campaign_path: Path, *, campaign_id: str | None = None,
) -> dict[str, str]:
    """Keep Git and authored Campaign provenance separate from runtime identity.

    A daemon-owned execution copy may differ byte-for-byte solely because its
    operational ``local_root`` was rebound.  In that path the daemon supplies
    the exact catalog revision explicitly; direct CLI use continues to hash the
    supplied Campaign file.
    """
    frozen_commit = ""
    if campaign_id is not None and campaign_path.is_file():
        payload = yaml.safe_load(campaign_path.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict):
            frozen_commit = str(payload.get("git_commit") or "")
        if frozen_commit and re.fullmatch(r"[0-9a-f]{40}", frozen_commit) is None:
            raise ValueError("execution Campaign git_commit must be a 40-digit lowercase SHA")
    if frozen_commit:
        commit = frozen_commit
    else:
        try:
            commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).stdout.strip()
        except subprocess.CalledProcessError as error:
            commit = os.environ.get("ELF_GIT_COMMIT", "")
            if not commit or commit == "unknown":
                raise RuntimeError(
                    "Git metadata is unavailable; build the image with GIT_COMMIT"
                ) from error
    resolved_campaign_id = campaign_id
    if resolved_campaign_id is not None:
        if _CAMPAIGN_REVISION_RE.fullmatch(resolved_campaign_id) is None:
            raise ValueError(
                "--campaign-id must be an immutable campaign.<sha256> revision"
            )
    else:
        resolved_campaign_id = run_command(
            ["bash", "scripts/source_identity.sh", "--campaign", str(campaign_path)],
            cwd=REPO_ROOT,
        ).stdout.strip()
    return {"git_commit": commit, "campaign_id": resolved_campaign_id}


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
    and ``{campaign}``. This lets immutable backend source paths be derived only
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
    validate_run(materialized, project=str(campaign["project"]))
    return materialized


def run_root_dir(campaign: dict[str, Any], run: dict[str, Any]) -> Path:
    """Return the controller-owned local metadata root for one scientific run."""
    root = Path(campaign.get("local_root", "outputs/experiment_campaigns"))
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root / campaign["campaign"] / run["run_id"]


def selected_attempt_id(run: dict[str, Any]) -> str | None:
    """Return the private controller selector injected for attempt reads."""
    value = run.get(_ATTEMPT_SELECTOR)
    if value is None:
        return None
    attempt_id = str(value)
    try:
        validate_identity("attempt_id", attempt_id)
    except ValueError as error:
        raise ValueError(f"invalid internal attempt selector: {attempt_id!r}") from error
    return attempt_id


def select_attempt(run: dict[str, Any], attempt_id: str) -> dict[str, Any]:
    """Return a shallow materialized-run view targeting one durable attempt."""
    selected = dict(run)
    selected[_ATTEMPT_SELECTOR] = attempt_id
    return selected


def local_run_dir(campaign: dict[str, Any], run: dict[str, Any]) -> Path:
    """Return the backend-local directory for the selected attempt or run root."""
    root = run_root_dir(campaign, run)
    attempt_id = selected_attempt_id(run)
    return root / "attempts" / attempt_id if attempt_id else root


def frozen_source_identity(
    campaign: dict[str, Any], run: dict[str, Any], fallback: str
) -> str:
    """Return a prepared run's frozen source identity, or a supplied fallback."""
    manifest_path = run_root_dir(campaign, run) / "manifest.yaml"
    if manifest_path.is_file():
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if payload.get("source_id"):
            return str(payload["source_id"])
    return fallback


def command_environment(
    campaign: dict[str, Any], run: dict[str, Any], source_id: str, attempt_id: str
) -> dict[str, str]:
    """Build the reviewed non-secret environment passed to the run launcher."""
    backend = BACKENDS.get(str(run["backend"]["kind"]))
    storage = run["storage"]
    env = {str(key): str(value) for key, value in run.get("env", {}).items()}
    env.update(
        {
            "PROJECT_NAME": str(campaign["project"]),
            "RUN_ID": str(run["run_id"]),
            "ATTEMPT_ID": attempt_id,
            "BACKEND": backend.kind,
            "SOURCE_ID": source_id,
            "RUNTIME_TREE_ID": source_id,
            "GIT_COMMIT": str(campaign.get("git_commit") or "unknown"),
            "CAMPAIGN_ID": str(campaign.get("campaign_id") or "unknown"),
            "CAMPAIGN_NAME": str(campaign["campaign"]),
            "IMAGE_ID": str(run["image_id"]),
            "OUTPUT_DIR": str(run["storage"]["run_dir"]),
            "NGPU": str(run.get("resources", {}).get("gpus", 1)),
            "MAX_INFRA_RETRIES": str(run.get("retry", {}).get("max_infra_retries", 0)),
            "DATA_ROOT": str(storage["data_root"]),
            "PROJECT_DATA_ROOT": str(storage["project_data_root"]),
            "HF_HOME": str(storage["hf_home"]),
            "HF_DATASETS_CACHE": str(storage["hf_datasets_cache"]),
        }
    )
    env.update(PROJECTS.get(str(campaign["project"])).environment(campaign, run))
    env.update(backend.environment(campaign, run, source_id, attempt_id))
    if campaign.get("research_contract") is not None:
        encoded = base64.urlsafe_b64encode(
            json.dumps(
                campaign["research_contract"],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).decode("ascii")
        env["RESEARCH_CONTRACT_B64"] = encoded
        env["RESEARCH_ROLE"] = str(run["research_role"])
    return env


def launcher_command(
    campaign: dict[str, Any], run: dict[str, Any], source_id: str, attempt_id: str
) -> list[str]:
    """Build the container-side launcher command with frozen config overrides."""
    env = command_environment(campaign, run, source_id, attempt_id)
    command: list[str] = ["env"]
    command.extend(f"{key}={value}" for key, value in sorted(env.items()))
    command.extend(PROJECTS.get(str(campaign["project"])).command(run))
    return command


def resolved_run_overrides(
    campaign: dict[str, Any], run: dict[str, Any], remote_run_dir: str
) -> list[str]:
    """Mirror launcher environment and explicit CLI overrides in execution order."""
    env = {str(key): str(value) for key, value in run.get("env", {}).items()}
    env.setdefault("RUN_ID", str(run["run_id"]))
    overrides = PROJECTS.get(str(campaign["project"])).operational_overrides(
        env, remote_run_dir
    )
    overrides.extend(map(str, run.get("config_overrides", [])))
    return overrides


def prepare_run(
    campaign: dict[str, Any], run: dict[str, Any], source_id: str, *, attempt_id: str
) -> dict[str, Any]:
    """Freeze local control metadata before any scheduler mutation occurs."""
    local_dir = run_root_dir(campaign, run)
    store = ExperimentStateStore(local_dir)
    identity_campaign = dict(campaign)
    if store.manifest_path.is_file():
        # Preparation freezes controller provenance before scheduler mutation.
        # A later submit may run from a newer checkout, but must render the
        # already-reviewed Attempt with the original immutable identities.
        frozen_manifest = store.load_manifest()
        for key in ("git_commit", "campaign_id"):
            if frozen_manifest.get(key) is not None:
                identity_campaign[key] = frozen_manifest[key]
    remote_run_dir = str(run["storage"]["run_dir"])
    project = PROJECTS.get(str(campaign["project"]))
    overrides = resolved_run_overrides(campaign, run, remote_run_dir)
    resolved = project.resolve_config(str(run["config"]), overrides)
    command = launcher_command(identity_campaign, run, source_id, attempt_id)
    bundle = project.source_bundle(REPO_ROOT)
    identity_run = dict(run)
    identity_run["env"] = {
        str(key): value for key, value in run.get("env", {}).items()
        if str(key).upper() not in {"RESUME", "RESUME_FROM", "CHECKPOINT_PATH"}
    }
    command_template = sanitize_command(
        launcher_command(identity_campaign, identity_run, source_id, "{attempt_id}")
    )
    retry = run.get("retry", {"max_infra_retries": 0})
    execution_identity = {
        "source_mount": bundle.container_path,
        "workdir": bundle.container_path,
    }
    run_resources = dict(run.get("resources", {}))
    run_resources.setdefault("nodes", 1)
    asset_requirements = [
        asdict(item) for item in project.plan_assets(str(run["config"]), overrides)
    ]
    checkpoint_identity = dict(run.get("checkpoint", {}))
    checkpoint_identity.setdefault("save_freq", resolved.get("save_freq"))
    manifest = build_run_manifest(
        project=str(campaign["project"]), run_id=str(run["run_id"]),
        created_at=utc_now(), config_path=str(run["config"]), resolved_config=resolved,
        source_id=source_id, runtime_tree_id=source_id,
        git_commit=identity_campaign.get("git_commit"),
        campaign_id=identity_campaign.get("campaign_id"),
        campaign=str(campaign["campaign"]), image_id=str(run["image_id"]),
        run_dir=remote_run_dir,
        max_infra_retries=int(retry.get("max_infra_retries", 0)),
        backend=dict(run["backend"]), resources=run_resources,
        storage=dict(run["storage"]), command=command_template,
        execution=execution_identity,
        config_overrides=list(map(str, run.get("config_overrides", []))),
        assets=asset_requirements, checkpoint=checkpoint_identity,
        evaluation=dict(run.get("evaluation", {})),
        research_contract=campaign.get("research_contract"),
        research_role=run.get("research_role"),
    )
    base_manifest = store.ensure_manifest(manifest)
    effective = dict(base_manifest)
    effective["attempt_id"] = attempt_id
    effective["backend"] = run["backend"]
    effective["resources"] = run_resources
    effective["storage"] = run["storage"]
    effective["config_overrides"] = list(run.get("config_overrides", []))
    effective["retry"] = retry
    effective["command"] = sanitize_command(command)
    effective["execution"] = execution_identity
    effective["resume_from"] = resolved.get("resume")
    attempt_path = store.attempt_path(attempt_id)
    if attempt_path.exists():
        previous_attempt = store.load_attempt(attempt_id)
        effective["created_at"] = previous_attempt["created_at"]
        if previous_attempt != effective:
            raise ValueError(f"existing control attempt conflicts: {attempt_path}")
    else:
        effective["created_at"] = utc_now()
        store.create_attempt(effective)
    store.initialize_attempt_records(attempt_id)
    return effective


def record_submission(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str, backend_job_id: str
) -> None:
    """Atomically record scheduler acceptance in controller-owned metadata."""
    local_dir = run_root_dir(campaign, run)
    store = ExperimentStateStore(local_dir)
    store.reconcile_submission(
        project=str(campaign["project"]), run_id=str(run["run_id"]),
        attempt_id=attempt_id, backend_job_id=backend_job_id, state=RunState.QUEUED,
    )


def record_submission_intent(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str
) -> dict[str, Any]:
    """Durably record scheduler mutation intent before crossing the API boundary."""
    local_dir = run_root_dir(campaign, run)
    backend = BACKENDS.get(str(run["backend"]["kind"]))
    request = backend.submission_request(campaign, run, attempt_id)
    return ExperimentStateStore(local_dir).begin_submission(
        project=str(campaign["project"]), run_id=str(run["run_id"]),
        attempt_id=attempt_id, backend=str(run["backend"]["kind"]),
        request=request,
    )


def reconcile_submission(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str
) -> str | None:
    """Recover a scheduler identity after acceptance/local-record crash."""
    local_dir = run_root_dir(campaign, run)
    recorded = recorded_scheduler_job_ids(local_dir, attempt_id)
    if len(recorded) > 1:
        raise RuntimeError(
            f"ambiguous scheduler identity: attempt {attempt_id} records jobs {recorded}"
        )
    store = ExperimentStateStore(local_dir)
    payload = store.load_backend(attempt_id) or {}
    if payload.get("backend_job_id"):
        return str(payload["backend_job_id"])
    submission = store.read_submission(attempt_id)
    if len(recorded) == 1 and submission:
        store.reconcile_submission(
            project=str(campaign["project"]), run_id=str(run["run_id"]),
            attempt_id=attempt_id, backend_job_id=recorded[0], state=RunState.QUEUED,
        )
        return recorded[0]
    if len(recorded) == 1:
        raise RuntimeError(
            f"attempt {attempt_id} records scheduler job {recorded[0]} without a "
            "durable submission intent"
        )
    if not submission or submission.get("state") != "SUBMITTING":
        return None
    intent = {**submission, **submission.get("request", {})}
    backend = BACKENDS.get(str(run["backend"]["kind"]))
    job_id = backend.recover_submission(run, intent, attempt_id)
    if job_id:
        record_submission(campaign, run, attempt_id, str(job_id))
        return str(job_id)
    return None


def cancel_intent_path(campaign: dict[str, Any], run: dict[str, Any]) -> Path:
    attempt_id = selected_attempt_id(run)
    if not attempt_id:
        raise ValueError("cancel requires an explicit attempt selector")
    return package_cancel_intent_path(run_root_dir(campaign, run), attempt_id)


def cancel_with_intent(campaign: dict[str, Any], run: dict[str, Any], backend_adapter) -> dict[str, Any]:
    """Bind ELF/backend context to the package-owned durable cancel outbox."""
    record = backend_record(campaign, run)
    attempt_id = str(record["attempt_id"])
    job_id = str(record["backend_job_id"])
    return execute_cancel_outbox(
        run_dir=run_root_dir(campaign, run),
        project=str(campaign["project"]),
        run_id=str(run["run_id"]),
        attempt_id=attempt_id,
        backend=record.get("backend"),
        backend_job_id=job_id,
        status_call=lambda: backend_adapter.status(campaign, run),
        cancel_call=lambda: backend_adapter.cancel(campaign, run),
    )


def recorded_scheduler_job_ids(local_dir: Path, attempt_id: str) -> list[str]:
    """Return every distinct scheduler job recorded for one local attempt."""
    found: set[str] = set()
    events_path = local_dir / "events.jsonl"
    if events_path.is_file():
        for line_number, line in enumerate(
            events_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid lifecycle event: {events_path}:{line_number}"
                ) from error
            if not isinstance(event, dict):
                raise ValueError(
                    f"lifecycle event is not an object: {events_path}:{line_number}"
                )
            if event.get("attempt_id") == attempt_id and event.get("backend_job_id"):
                found.add(str(event["backend_job_id"]))
    for path in (
        local_dir / "backend.json",
        local_dir / "attempts" / attempt_id / "backend.json",
        local_dir / "attempts" / attempt_id / "submission.json",
    ):
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("attempt_id") == attempt_id and payload.get("backend_job_id"):
                found.add(str(payload["backend_job_id"]))
    return sorted(found, key=lambda value: (len(value), value))


def identity_report(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str
) -> dict[str, Any]:
    """Combine local durable history with the backend's read-only identity probe."""
    local_dir = run_root_dir(campaign, run)
    local_jobs = recorded_scheduler_job_ids(local_dir, attempt_id)
    remote = BACKENDS.get(str(run["backend"]["kind"])).identity(
        campaign, run, attempt_id
    ).to_dict()
    all_jobs = sorted(
        {*local_jobs, *map(str, remote.get("scheduler_job_ids", []))},
        key=lambda value: (len(value), value),
    )
    ambiguous = bool(remote.get("ambiguous")) or len(all_jobs) > 1
    local_manifest_exists = (local_dir / "manifest.yaml").is_file()
    owned_remote_manifest = bool(
        local_manifest_exists and remote.get("remote_manifest_matches") is True
    )
    available = (
        (bool(remote.get("available")) or owned_remote_manifest)
        and not local_jobs
        and not ambiguous
    )
    return {
        "run_id": run["run_id"],
        "attempt_id": attempt_id,
        "available": available,
        "ambiguous": ambiguous,
        "scheduler_job_ids": all_jobs,
        "local_manifest_exists": local_manifest_exists,
        "remote_manifest_exists": remote.get("remote_manifest_exists"),
        "remote_manifest_matches": remote.get("remote_manifest_matches"),
        "remote_manifest_owned": owned_remote_manifest,
    }


def require_identity_available(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str
) -> dict[str, Any]:
    """Fail closed unless one run/attempt identity is unused and unambiguous."""
    report = identity_report(campaign, run, attempt_id)
    if not report["available"]:
        raise FileExistsError(
            f"run/attempt identity is already consumed or ambiguous: {report}"
        )
    return report


def ensure_attempt_not_submitted(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str
) -> None:
    """Reject a second scheduler mutation for an already submitted attempt."""
    submission = ExperimentStateStore(run_root_dir(campaign, run)).read_submission(attempt_id)
    if submission and submission.get("backend_job_id"):
        raise FileExistsError(
            f"attempt {attempt_id} already has backend job {submission.get('backend_job_id')}; "
            "use a new attempt ID"
        )
    if submission and submission.get("state") == "SUBMITTING":
        raise RuntimeError(
            f"attempt {attempt_id} has an unresolved submission intent; run status to reconcile "
            "before creating another scheduler job"
        )


def backend_services() -> BackendServices:
    """Inject controller IO boundaries into platform-specific adapters."""
    return BackendServices(
        run_command=run_command, local_run_dir=local_run_dir,
        backend_record=backend_record, summarize_run=summarize_project_run,
        parse_metric=parse_project_metric, parse_checkpoint=parse_project_checkpoint,
        atomic_write=atomic_write, utc_now=utc_now,
    )


def summarize_project_run(campaign: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Dispatch result interpretation to the campaign's scientific project."""
    return PROJECTS.get(str(campaign["project"])).summarize(run_dir)


def parse_project_metric(campaign: dict[str, Any], line: str) -> dict[str, Any] | None:
    """Dispatch training-log interpretation without teaching a backend project syntax."""
    return PROJECTS.get(str(campaign["project"])).parse_metric(line)


def parse_project_checkpoint(campaign: dict[str, Any], line: str) -> dict[str, Any] | None:
    """Dispatch project checkpoint log parsing without backend format knowledge."""
    return PROJECTS.get(str(campaign["project"])).parse_checkpoint(line)


def backend_record(campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Load the recorded scheduler identity for a submitted run."""
    attempt_id = selected_attempt_id(run)
    store = ExperimentStateStore(run_root_dir(campaign, run))
    payload = store.load_backend(attempt_id)
    path = store.attempt_backend_path(attempt_id) if attempt_id else store.backend_path
    if payload is None:
        raise FileNotFoundError(f"run has not been submitted: {run['run_id']}")
    if not isinstance(payload, dict) or not payload.get("backend_job_id"):
        raise ValueError(f"invalid backend record: {path}")
    return payload


def update_observed_status(campaign: dict[str, Any], run: dict[str, Any], status: dict[str, Any]) -> None:
    """Persist one normalized scheduler observation without claiming model progress."""
    local_dir = run_root_dir(campaign, run)
    record = backend_record(campaign, run)
    attempt_id = str(record["attempt_id"])
    payload = dict(status)
    payload["updated_at"] = utc_now()
    payload.setdefault("project", campaign["project"])
    payload.setdefault("run_id", run["run_id"])
    ExperimentStateStore(local_dir).write_status_payload(attempt_id, payload)
    append_event(
        local_dir / "events.jsonl",
        {
            "timestamp": payload["updated_at"],
            "run_id": run["run_id"],
            "attempt_id": attempt_id,
            "backend": status["backend"],
            "backend_job_id": status["backend_job_id"],
            "event": "scheduler_observed",
            "payload": {"state": status["state"], "raw_state": status.get("raw_state")},
        },
    )


def write_local_collection(
    campaign: dict[str, Any], run: dict[str, Any], summary: dict[str, Any],
) -> dict[str, Any]:
    """Persist one observation without erasing stronger exact-Attempt evidence."""
    root = run_root_dir(campaign, run)
    attempt_id = selected_attempt_id(run) or str(backend_record(campaign, run)["attempt_id"])
    attempt_path = root / "attempts" / attempt_id / "collection.json"
    previous = (
        json.loads(attempt_path.read_text(encoding="utf-8"))
        if attempt_path.is_file() else None
    )
    merged = merge_terminal_observation(previous, summary)
    atomic_write(attempt_path, merged)
    current = ExperimentStateStore(root).load_backend()
    if current and current.get("attempt_id") == attempt_id:
        atomic_write(root / "collection.json", merged)
    return merged


def campaign_research_block(
    campaign: dict[str, Any], *, current_run_id: str,
    current_decision: dict[str, Any], attempt_id: str,
) -> dict[str, Any]:
    """Combine current-attempt role decisions without mixing stale attempts."""
    contract = campaign["research_contract"]
    records: dict[str, dict[str, Any]] = {}
    for candidate in campaign["runs"]:
        role = candidate.get("research_role")
        if not role:
            continue
        root = run_root_dir(campaign, candidate)
        manifest_path = root / "manifest.yaml"
        if not manifest_path.is_file():
            continue
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        if not isinstance(manifest, dict):
            continue
        if candidate["run_id"] == current_run_id:
            decision = current_decision
            selected_attempt = attempt_id
        else:
            decision_path = root / "decision.json"
            if not decision_path.is_file():
                continue
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            selected_attempt = str(decision.get("attempt_id", ""))
        current_backend = ExperimentStateStore(root).load_backend() or {}
        if current_backend.get("attempt_id") != selected_attempt:
            continue
        records[str(role)] = {
            "research_outcome": decision.get("research_outcome"),
            "manifest": manifest,
            "run": candidate,
        }
    return evaluate_research_block(contract=contract, role_records=records)


def clean_log_lines(value: Any, *, max_chars: int = 32768) -> list[str]:
    """Normalize persisted/tool-facing log lines and bound pathological rows."""
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for value_line in value:
        line = str(value_line).replace("\x00", "")
        if len(line) > max_chars:
            line = line[:max_chars] + "…<truncated>"
        cleaned.append(line)
    return cleaned


def annotate_collection(
    summary: dict[str, Any], scheduler_status: dict[str, Any]
) -> dict[str, Any]:
    """Keep scheduler, worker, process, and model evidence explicitly separate."""
    annotated = dict(summary)
    annotated["runtime_state"] = summary.get("state")
    annotated["scheduler_state"] = scheduler_status.get("state")
    process_evidence = summary.get("process_evidence", {})
    if isinstance(process_evidence, dict):
        process_evidence = dict(process_evidence)
        process_evidence["stdout_tail"] = clean_log_lines(
            process_evidence.get("stdout_tail")
        )
        process_evidence["stderr_tail"] = clean_log_lines(
            process_evidence.get("stderr_tail")
        )
        annotated["process_evidence"] = process_evidence
    process_observed = bool(
        isinstance(process_evidence, dict) and process_evidence.get("observed")
    )
    scheduler_state = str(scheduler_status.get("state") or "UNKNOWN")
    scheduler_terminal = scheduler_state in _TERMINAL_STATES
    inferred_worker = (
        "RELEASED" if scheduler_terminal
        else "ALLOCATED" if process_observed
        else "UNKNOWN"
    )
    # Scheduler terminality proves that the allocation and its processes can
    # no longer be running. Preserve the adapter's possibly stale runtime
    # observation separately in ``runtime_state``, but never expose that stale
    # value as current worker/process state.
    annotated["worker_state"] = (
        "RELEASED" if scheduler_terminal
        else summary.get("worker_state", inferred_worker)
    )
    runtime_state = summary.get("state")
    process_was_observed = process_observed or runtime_state not in {None, "", "UNKNOWN"}
    annotated["process_state"] = (
        scheduler_state if scheduler_terminal and process_was_observed
        else "UNKNOWN" if scheduler_terminal
        else runtime_state or "UNKNOWN"
    )
    if scheduler_state == "FAILED" and process_observed:
        failure = classify_failure(json.dumps(process_evidence, ensure_ascii=False))
        if failure is not FailureClass.UNKNOWN:
            annotated["failure_class"] = failure.value
    model_evidence = (
        summary.get("model_observed"),
        summary.get("step"),
        summary.get("latest_metric"),
        summary.get("latest_completed_checkpoint"),
        summary.get("evaluation_metrics_by_variant") or None,
    )
    annotated["model_state"] = (
        "UNKNOWN" if summary.get("evidence_unavailable_reason")
        else "OBSERVED" if any(value is not None and value is not False for value in model_evidence)
        else "NOT_OBSERVED"
    )
    evidence_reason = summary.get("evidence_unavailable_reason")
    if not evidence_reason and scheduler_terminal and annotated["model_state"] != "OBSERVED" and not process_observed:
        evidence_reason = (
            "cancelled_before_observation"
            if scheduler_status.get("state") == "CANCELLED"
            else "terminal_without_process_or_model_evidence"
        )
    annotated["evidence_unavailable_reason"] = evidence_reason
    annotated["evidence_outcome"] = (
        "INCONCLUSIVE" if evidence_reason
        else "PENDING" if not scheduler_terminal
        else "OBSERVED"
    )
    return annotated


def cached_attempt_logs(
    campaign: dict[str, Any], run: dict[str, Any], *, tail: int,
    live_error: str,
) -> dict[str, Any] | None:
    """Return bounded attempt logs cached by collect/observe, when available.

    A transient backend log probe must not make logs appear unavailable when a
    prior controller observation already persisted sanitized process evidence.
    The response is explicitly marked non-live so callers cannot mistake it
    for a successful remote read.
    """
    attempt_id = selected_attempt_id(run)
    if not attempt_id:
        return None
    path = run_root_dir(campaign, run) / "attempts" / attempt_id / "collection.json"
    if not path.is_file():
        return None
    collection = json.loads(path.read_text(encoding="utf-8"))
    evidence = collection.get("process_evidence")
    if not isinstance(evidence, dict) or not evidence.get("observed"):
        return None
    stdout = evidence.get("stdout_tail")
    stderr = evidence.get("stderr_tail")
    if not isinstance(stdout, list) or not isinstance(stderr, list):
        return None
    record = backend_record(campaign, run)
    return {
        "run_id": run["run_id"],
        "backend": run["backend"]["kind"],
        "backend_job_id": record["backend_job_id"],
        "attempt_id": attempt_id,
        "tail": tail,
        "sources": evidence.get("sources", {}),
        "stdout": clean_log_lines(stdout[-tail:]),
        "stderr": clean_log_lines(stderr[-tail:]),
        "live": False,
        "evidence_source": "cached_collection",
        "live_error": live_error,
    }


def read_logs(
    campaign: dict[str, Any], run: dict[str, Any], backend_adapter,
    *, tail: int,
) -> dict[str, Any]:
    """Read live backend logs, falling back to explicit cached evidence."""
    try:
        payload = backend_adapter.logs(campaign, run, tail=tail)
    except RuntimeError:
        cached = cached_attempt_logs(
            campaign, run, tail=tail,
            live_error="live log probe unavailable; cached evidence returned",
        )
        if cached is None:
            raise
        return cached
    payload = dict(payload)
    for key in ("stdout", "stderr", "lines"):
        if key in payload:
            payload[key] = clean_log_lines(payload[key])
    payload.setdefault("live", True)
    payload.setdefault("evidence_source", "backend")
    return payload


def status_for_decision(
    status: dict[str, Any], collection: dict[str, Any]
) -> dict[str, Any]:
    """Prefer attempt diagnostics over a coarser scheduler classification."""
    merged = dict(status)
    if collection.get("failure_class"):
        merged["failure_class"] = collection["failure_class"]
    return merged


PROJECTS = build_project_registry()


class _LazyBackendRegistry:
    """Construct scheduler adapters only when a backend verb actually needs them."""

    def __init__(self) -> None:
        self._registry = None

    def _get_registry(self):
        if self._registry is None:
            self._registry = build_registry(backend_services())
        return self._registry

    @property
    def kinds(self):
        return self._get_registry().kinds

    def get(self, kind: str):
        return self._get_registry().get(kind)


BACKENDS = _LazyBackendRegistry()


def unsubmitted_status(campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Return controller state for a prepared run without raising on no job ID."""
    store = ExperimentStateStore(run_root_dir(campaign, run))
    attempt_id = selected_attempt_id(run)
    payload = store.load_status_payload(attempt_id)
    if payload is None:
        payload = store.read_status(attempt_id).to_dict()
    backend = store.load_backend(attempt_id) or {}
    payload.setdefault("backend", run["backend"]["kind"])
    payload.setdefault("backend_job_id", backend.get("backend_job_id"))
    return payload


def observe_run(
    campaign: dict[str, Any], run: dict[str, Any], backend_adapter,
    *, attempt_id: str,
) -> dict[str, Any]:
    """Collect one four-layer observation and persist its attempt read model."""
    reconcile_submission(campaign, run, attempt_id)
    backend_payload = ExperimentStateStore(
        run_root_dir(campaign, run)
    ).load_backend(attempt_id) or {}
    if backend_payload.get("backend_job_id"):
        status = backend_adapter.status(campaign, run)
        update_observed_status(campaign, run, status)
        collection = annotate_collection(
            backend_adapter.collect(campaign, run), status
        )
        collection = write_local_collection(campaign, run, collection)
    else:
        status, collection = unsubmitted_status(campaign, run), None
    return {
        "run_id": run["run_id"],
        "scheduler": status,
        "worker": {
            "state": collection.get("worker_state", "UNKNOWN")
            if collection else "UNKNOWN"
        },
        "process": {
            "state": collection.get("process_state", "UNKNOWN")
            if collection else "UNKNOWN"
        },
        "model": collection,
    }


def decide_run(
    campaign: dict[str, Any], run: dict[str, Any], *, attempt_id: str,
) -> dict[str, Any]:
    """Evaluate retry and research policy from the latest durable observation."""
    local_dir = run_root_dir(campaign, run)
    status = unsubmitted_status(campaign, run)
    collection_path = local_dir / "attempts" / attempt_id / "collection.json"
    collection = (
        json.loads(collection_path.read_text(encoding="utf-8"))
        if collection_path.is_file() else {}
    )
    attempts = [
        path for path in (local_dir / "attempts").glob("attempt-*")
        if path.is_dir()
    ]
    retry = run.get("retry", {"max_infra_retries": 0})
    diagnostic = json.dumps(collection, ensure_ascii=False)
    decision_status = status_for_decision(status, collection)
    decision = decide_next_action(
        decision_status, retries_used=max(0, len(attempts) - 1),
        max_infra_retries=int(retry.get("max_infra_retries", 0)),
        diagnostic_text=diagnostic,
        completed_checkpoint=collection.get("latest_completed_checkpoint"),
    )
    payload = decision.to_dict()
    payload["attempt_id"] = attempt_id
    contract = campaign.get("research_contract")
    if contract is not None:
        role = str(run["research_role"])
        research = evaluate_research_run(
            status=status, collection=collection,
            contract=contract, role=role,
        )
        payload.update(research)
        block = campaign_research_block(
            campaign, current_run_id=str(run["run_id"]),
            current_decision=payload, attempt_id=attempt_id,
        )
        payload.update(block)
        if decision.action == "OBSERVE" and research["research_action"] == "STOP_RECOMMENDED":
            payload["action"] = "STOP_RECOMMENDED"
        elif decision.action == "VERIFY_RESULTS":
            if research["research_outcome"] == "PASS":
                payload["action"] = block["block_action"]
            else:
                payload["action"] = research["research_action"]
    decision_path = local_dir / "attempts" / attempt_id / "decision.json"
    atomic_write(decision_path, payload)
    current = ExperimentStateStore(local_dir).load_backend()
    if current and current.get("attempt_id") == attempt_id:
        atomic_write(local_dir / "decision.json", payload)
    return {"run_id": run["run_id"], **payload}


def has_model_metric(model: dict[str, Any]) -> bool:
    """Return whether a collection contains an actual model metric record."""
    return any(
        model.get(field) is not None
        for field in ("step", "optimizer_step", "latest_metric")
    ) or bool(model.get("evaluation_metrics_by_variant"))


def watch_runs(
    campaign: dict[str, Any], runs: list[dict[str, Any]], *, attempt_id: str,
    interval_seconds: float, timeout_seconds: float, poll_timeout_seconds: float,
    until: str,
) -> int:
    """Stream JSONL observations until every selected run reaches a stop gate.

    Terminal runs are collected by ``observe_run`` and immediately evaluated
    by ``decide_run``. This command is read-only with respect to schedulers: a
    STOP_RECOMMENDED decision is reported but never converted into cancellation.
    """
    if interval_seconds <= 0:
        raise ValueError("--interval-seconds must be greater than zero")
    if timeout_seconds < 0:
        raise ValueError("--timeout-seconds must be zero or greater")
    if poll_timeout_seconds <= 0:
        raise ValueError("--poll-timeout-seconds must be greater than zero")
    if until not in {"terminal", "first-metric"}:
        raise ValueError("--until must be terminal or first-metric")
    started = time.monotonic()
    pending = {str(run["run_id"]): run for run in runs}
    failed_gate_run_ids: list[str] = []
    polls = 0
    while pending:
        elapsed = time.monotonic() - started
        if timeout_seconds and elapsed >= timeout_seconds:
            print(json.dumps({
                "event": "watch_timeout",
                "elapsed_seconds": elapsed,
                "pending_run_ids": sorted(pending),
                "polls": polls,
            }, ensure_ascii=False, sort_keys=True), flush=True)
            return 1
        polls += 1
        for run_id, run in list(pending.items()):
            backend_adapter = BACKENDS.get(str(run["backend"]["kind"]))
            elapsed_before_poll = time.monotonic() - started
            poll_budget = poll_timeout_seconds
            if timeout_seconds:
                poll_budget = min(
                    poll_budget,
                    max(0.001, timeout_seconds - elapsed_before_poll),
                )
            try:
                with command_deadline(poll_budget):
                    observation = observe_run(
                        campaign, run, backend_adapter, attempt_id=attempt_id
                    )
            except subprocess.TimeoutExpired:
                print(json.dumps({
                    "event": "watch_poll_timeout",
                    "poll": polls,
                    "run_id": run_id,
                    "poll_timeout_seconds": poll_budget,
                    "action": "RETRY_OBSERVATION",
                }, ensure_ascii=False, sort_keys=True), flush=True)
                continue
            scheduler_state = str(observation["scheduler"].get("state", "UNKNOWN"))
            model = observation.get("model") or {}
            metric_observed = has_model_metric(model)
            terminal = scheduler_state in _TERMINAL_STATES
            print(json.dumps({
                "event": "watch_observation",
                "poll": polls,
                "run_id": run_id,
                "scheduler_state": scheduler_state,
                "worker_state": observation["worker"]["state"],
                "process_state": observation["process"]["state"],
                "model_state": model.get("model_state", "NOT_OBSERVED"),
                "step": model.get("step"),
                "optimizer_step": model.get("optimizer_step"),
            }, ensure_ascii=False, sort_keys=True), flush=True)
            reached = terminal or (until == "first-metric" and metric_observed)
            if reached:
                decision = decide_run(campaign, run, attempt_id=attempt_id)
                gate_passed = not (
                    until == "first-metric" and terminal and not metric_observed
                )
                reason = (
                    "terminal-without-first-metric" if not gate_passed
                    else "terminal" if terminal
                    else "first-metric"
                )
                if not gate_passed:
                    failed_gate_run_ids.append(run_id)
                print(json.dumps({
                    "event": "watch_run_complete",
                    "run_id": run_id,
                    "reason": reason,
                    "gate_passed": gate_passed,
                    "scheduler_state": scheduler_state,
                    "worker_state": observation["worker"]["state"],
                    "process_state": observation["process"]["state"],
                    "model_state": model.get("model_state", "NOT_OBSERVED"),
                    "step": model.get("step"),
                    "optimizer_step": model.get("optimizer_step"),
                    "decision": decision,
                }, ensure_ascii=False, sort_keys=True), flush=True)
                del pending[run_id]
        if not pending:
            break
        elapsed = time.monotonic() - started
        if timeout_seconds and elapsed >= timeout_seconds:
            print(json.dumps({
                "event": "watch_timeout",
                "elapsed_seconds": elapsed,
                "pending_run_ids": sorted(pending),
                "polls": polls,
            }, ensure_ascii=False, sort_keys=True), flush=True)
            return 1
        sleep_for = interval_seconds
        if timeout_seconds:
            sleep_for = min(sleep_for, max(0.0, timeout_seconds - elapsed))
        time.sleep(sleep_for)
    print(json.dumps({
        "event": "watch_complete",
        "failed_gate_run_ids": sorted(failed_gate_run_ids),
        "gate_passed": not failed_gate_run_ids,
        "polls": polls,
        "run_ids": sorted(str(run["run_id"]) for run in runs),
        "until": until,
    }, ensure_ascii=False, sort_keys=True), flush=True)
    return 1 if failed_gate_run_ids else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse campaign path, operation, optional run filters, and dry-run policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "command", choices=(
            "prepare", "preflight", "stage", "submit", "status", "collect", "cancel",
            "observe", "watch", "logs", "decide", "assets-plan", "assets-verify", "check-identity",
            "refresh-evidence-local",
        )
    )
    parser.add_argument("--run", action="append", default=[], help="limit to this run ID; repeatable")
    parser.add_argument("--attempt-id", default="attempt-001")
    parser.add_argument(
        "--campaign-id",
        help="daemon-reviewed authored Campaign revision for an operational execution copy",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--local-root", type=Path,
        help="original already-local campaign data root (local evidence rebuild only)",
    )
    parser.add_argument(
        "--identity-root", type=Path,
        help="reviewed private campaign/run/Attempt identity root",
    )
    parser.add_argument(
        "--expected-input-digest",
        help="reviewed local evidence digest required before atomic replacement",
    )
    parser.add_argument(
        "--expected-current-collection-digest",
        help=(
            "reviewed live collection digest when the private collection input "
            "is an operational recovery baseline"
        ),
    )
    parser.add_argument(
        "--scope", choices=("stage", "submit", "observe"), default="submit",
        help="readiness scope used by the preflight command",
    )
    parser.add_argument("--tail", type=int, default=100, help="maximum log lines per stream")
    parser.add_argument(
        "--interval-seconds", type=float, default=60.0,
        help="watch polling interval; must be greater than zero",
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=0.0,
        help="watch timeout; zero waits without a deadline",
    )
    parser.add_argument(
        "--poll-timeout-seconds", type=float, default=60.0,
        help="hard deadline for one watch observation poll",
    )
    parser.add_argument(
        "--until", choices=("terminal", "first-metric"), default="terminal",
        help="watch completion gate",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute one deterministic controller operation for selected campaign runs."""
    args = parse_args(argv)
    if args.command == "refresh-evidence-local":
        output = rebuild_local_evidence(args)
        print(json.dumps([output], ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    campaign = load_campaign(args.campaign)
    campaign.update(provenance_identity(args.campaign, campaign_id=args.campaign_id))
    default_identity = source_identity(campaign)
    selected = selected_runs(campaign, args.run)
    runs_with_identity = []
    for run in selected:
        identity = str(run.get("source_id", default_identity))
        if args.command in {"status", "collect", "cancel", "observe", "watch", "logs", "decide"}:
            identity = frozen_source_identity(campaign, run, identity)
        materialized = materialize_run(campaign, run, identity)
        if args.command in {"status", "collect", "cancel", "observe", "watch", "logs", "decide"}:
            materialized = select_attempt(materialized, args.attempt_id)
        runs_with_identity.append((materialized, identity))
    if args.command == "watch":
        return watch_runs(
            campaign, [run for run, _identity in runs_with_identity],
            attempt_id=args.attempt_id,
            interval_seconds=args.interval_seconds,
            timeout_seconds=args.timeout_seconds,
            poll_timeout_seconds=args.poll_timeout_seconds,
            until=args.until,
        )
    outputs: list[dict[str, Any]] = []
    for run, identity in runs_with_identity:
        backend_kind = run["backend"]["kind"]
        backend_adapter = BACKENDS.get(backend_kind)
        manifest = None
        if args.command == "prepare" or (args.command == "submit" and args.dry_run):
            manifest = prepare_run(campaign, run, identity, attempt_id=args.attempt_id)
        if args.command == "prepare":
            outputs.append({"run_id": run["run_id"], "state": "CREATED"})
        elif args.command == "preflight":
            report = backend_adapter.preflight(run, scope=args.scope)
            outputs.append({"run_id": run["run_id"], **report.to_dict()})
        elif args.command == "check-identity":
            backend_adapter.preflight(run, scope="observe").require_ready()
            outputs.append(identity_report(campaign, run, args.attempt_id))
        elif args.command == "stage":
            backend_adapter.preflight(run, scope="stage").require_ready()
            require_identity_available(campaign, run, args.attempt_id)
            prepare_run(campaign, run, identity, attempt_id=args.attempt_id)
            bundle = PROJECTS.get(str(campaign["project"])).source_bundle(REPO_ROOT)
            staged = backend_adapter.stage(campaign, run, identity, bundle)
            outputs.append({"run_id": run["run_id"], "staged": staged})
        elif args.command == "submit":
            intent = None
            if not args.dry_run:
                backend_adapter.preflight(run, scope="submit").require_ready()
                recovered = reconcile_submission(campaign, run, args.attempt_id)
                if recovered:
                    outputs.append({"run_id": run["run_id"], "backend_job_id": recovered, "reconciled": True})
                    continue
                require_identity_available(campaign, run, args.attempt_id)
                manifest = prepare_run(
                    campaign, run, identity, attempt_id=args.attempt_id
                )
                ensure_attempt_not_submitted(campaign, run, args.attempt_id)
                intent = record_submission_intent(campaign, run, args.attempt_id)
            assert manifest is not None
            job_id = backend_adapter.submit(
                campaign, run, manifest, dry_run=args.dry_run, intent=intent
            )
            if args.dry_run:
                root = run_root_dir(campaign, run)
                attempt_dir = root / "attempts" / args.attempt_id
                preview = attempt_dir / "submission.preview"
                preview.write_text(
                    backend_adapter.render(manifest) + "\n", encoding="utf-8"
                )
                outputs.append({
                    "run_id": run["run_id"],
                    "attempt_id": args.attempt_id,
                    "backend_job_id": job_id,
                    "state": "CREATED",
                    "scheduler_mutated": False,
                    "local_run_dir": str(root),
                    "manifest_path": str(root / "manifest.yaml"),
                    "attempt_path": str(attempt_dir / "attempt.yaml"),
                    "submission_preview_path": str(preview),
                    "next_gates": [
                        "check-identity", "assets-verify", "stage", "submit",
                    ],
                })
            else:
                record_submission(campaign, run, args.attempt_id, job_id)
                outputs.append({"run_id": run["run_id"], "backend_job_id": job_id})
        elif args.command == "status":
            reconcile_submission(campaign, run, args.attempt_id)
            selected_record = ExperimentStateStore(
                run_root_dir(campaign, run)
            ).load_backend(args.attempt_id) or {}
            submitted = bool(selected_record.get("backend_job_id"))
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
            summary = write_local_collection(campaign, run, summary)
            outputs.append(summary)
        elif args.command == "logs":
            if args.tail < 1 or args.tail > 10000:
                raise ValueError("--tail must be between 1 and 10000")
            outputs.append(read_logs(
                campaign, run, backend_adapter, tail=args.tail
            ))
        elif args.command == "cancel":
            status = cancel_with_intent(campaign, run, backend_adapter)
            update_observed_status(campaign, run, status)
            outputs.append(status)
        elif args.command == "observe":
            outputs.append(observe_run(
                campaign, run, backend_adapter, attempt_id=args.attempt_id
            ))
        elif args.command == "decide":
            outputs.append(decide_run(
                campaign, run, attempt_id=args.attempt_id
            ))
        elif args.command in {"assets-plan", "assets-verify"}:
            project = PROJECTS.get(str(campaign["project"]))
            requirements = project.plan_assets(
                str(run["config"]), list(map(str, run.get("config_overrides", [])))
            )
            result: dict[str, Any] = {
                "run_id": run["run_id"],
                "requirements": [asdict(item) for item in requirements],
            }
            if args.command == "assets-verify":
                env = command_environment(campaign, run, identity, args.attempt_id)
                probes = project.asset_probes(requirements, env)
                result.update(backend_adapter.verify_assets(run, probes))
            outputs.append(result)
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "preflight" and any(not output.get("ready", False) for output in outputs):
        return 1
    if args.command == "check-identity" and any(not output.get("available", False) for output in outputs):
        return 1
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Render expected operational failures without a Python traceback."""
    try:
        return main(argv)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(json.dumps({
            "error": type(error).__name__,
            "message": str(error),
        }, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
