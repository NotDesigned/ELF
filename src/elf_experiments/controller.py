#!/usr/bin/env python
"""Prepare, submit, inspect, and collect project campaigns through registered backends."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

from .manifest import (  # noqa: E402
    IDENTITY_RE,
    SECRET_KEY_RE,
    URL_USERINFO_RE,
    ExperimentStateStore,
    RunState,
    append_event,
    atomic_create,
    atomic_write,
    sanitize_command,
    utc_now,
)
from .campaign import load_and_resolve_campaign  # noqa: E402
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
from experiment_control.runner import (  # noqa: E402
    CommandResult,
    CommandRunner,
    SubprocessRunner,
)
from experiment_control.states import FailureClass, classify_failure  # noqa: E402


_COMMAND_RUNNER: CommandRunner = SubprocessRunner()
_ATTEMPT_SELECTOR = "__experiment_attempt_id"
_TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED"})
_CAMPAIGN_SECRET_KEY_RE = re.compile(
    r"(?:^|[_-])(?:secret|password|credential|access[_-]?key(?:[_-](?:id|secret))?"
    r"|api[_-]?key|authorization|cookie|token)(?:$|[_-])",
    re.IGNORECASE,
)


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
    if isinstance(value, str) and URL_USERINFO_RE.search(value):
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
        validate_run(run, project=str(payload["project"]))
        if run["run_id"] in seen:
            raise ValueError(f"duplicate run_id: {run['run_id']}")
        seen.add(run["run_id"])
    _reject_embedded_credentials(payload, path="campaign")
    validate_research_contract(payload)
    return payload


def validate_run(run: Any, *, project: str | None = None) -> None:
    """Validate one backend-neutral run and reject secret-bearing settings."""
    if not isinstance(run, dict):
        raise ValueError("each campaign run must be a mapping")
    for key in ("run_id", "config", "backend", "storage", "image_id"):
        if not run.get(key):
            raise ValueError(f"run is missing {key}")
    if not IDENTITY_RE.fullmatch(str(run["run_id"])):
        raise ValueError(f"invalid run_id: {run['run_id']!r}")
    backend = run["backend"]
    if not isinstance(backend, dict) or backend.get("kind") not in BACKENDS.kinds:
        raise ValueError(
            f"run {run['run_id']} backend.kind must be one of {sorted(BACKENDS.kinds)}"
        )
    env = run.get("env", {})
    if not isinstance(env, dict):
        raise ValueError(f"run {run['run_id']} env must be a mapping")
    allowed_env = PROJECTS.get(project).safe_env_keys if project is not None else frozenset()
    forbidden = [key for key in env if key not in allowed_env or SECRET_KEY_RE.search(key)]
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


def provenance_identity(campaign_path: Path) -> dict[str, str]:
    """Keep Git and campaign provenance separate from runtime artifact identity."""
    try:
        commit = run_command(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).stdout.strip()
    except subprocess.CalledProcessError as error:
        commit = os.environ.get("ELF_GIT_COMMIT", "")
        if not commit or commit == "unknown":
            raise RuntimeError(
                "Git metadata is unavailable; build the image with GIT_COMMIT"
            ) from error
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
    if not IDENTITY_RE.fullmatch(attempt_id):
        raise ValueError(f"invalid internal attempt selector: {attempt_id!r}")
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
    remote_run_dir = str(run["storage"]["run_dir"])
    project = PROJECTS.get(str(campaign["project"]))
    overrides = resolved_run_overrides(campaign, run, remote_run_dir)
    resolved = project.resolve_config(str(run["config"]), overrides)
    command = launcher_command(campaign, run, source_id, attempt_id)
    bundle = project.source_bundle(REPO_ROOT)
    identity_run = dict(run)
    identity_run["env"] = {
        str(key): value for key, value in run.get("env", {}).items()
        if str(key).upper() not in {"RESUME", "RESUME_FROM", "CHECKPOINT_PATH"}
    }
    command_template = sanitize_command(
        launcher_command(campaign, identity_run, source_id, "{attempt_id}")
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
        git_commit=campaign.get("git_commit"), campaign_id=campaign.get("campaign_id"),
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
    token = f"{campaign['campaign']}/{run['run_id']}/{attempt_id}"
    backend = BACKENDS.get(str(run["backend"]["kind"]))
    request = {"submission_token": token}
    request.update(backend.submission_request(campaign, run, attempt_id))
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
    return run_root_dir(campaign, run) / "attempts" / attempt_id / "cancel_intent.json"


def cancel_with_intent(campaign: dict[str, Any], run: dict[str, Any], backend_adapter) -> dict[str, Any]:
    """Cancel one exact scheduler identity through a durable create-once outbox."""
    record = backend_record(campaign, run)
    attempt_id = str(record["attempt_id"])
    job_id = str(record["backend_job_id"])
    path = cancel_intent_path(campaign, run)
    if path.is_file():
        intent = json.loads(path.read_text(encoding="utf-8"))
        if intent.get("backend_job_id") != job_id or intent.get("attempt_id") != attempt_id:
            raise ValueError("cancel intent conflicts with selected scheduler identity")
        if intent.get("state") == "VERIFIED":
            return dict(intent["result"])
        status = backend_adapter.status(campaign, run)
        if str(status.get("backend_job_id")) != job_id:
            raise RuntimeError("cancel reconciliation returned a different backend job identity")
        if str(status.get("state", "")).upper() in {
            "SUCCEEDED", "FAILED", "CANCELLED", "PREEMPTED",
        }:
            intent.update({"state": "VERIFIED", "verified_at": utc_now(), "result": status})
            atomic_write(path, intent)
            return status
        raise RuntimeError(
            "cancel intent is unresolved and target is still nonterminal; "
            "do not issue a second cancel mutation"
        )
    intent = {
        "schema_version": 1, "state": "REQUESTED", "requested_at": utc_now(),
        "project": campaign["project"], "run_id": run["run_id"],
        "attempt_id": attempt_id, "backend": record.get("backend"),
        "backend_job_id": job_id,
    }
    atomic_create(path, intent)
    append_event(run_root_dir(campaign, run) / "events.jsonl", {
        "timestamp": intent["requested_at"], "run_id": run["run_id"],
        "attempt_id": attempt_id, "backend": record.get("backend"),
        "backend_job_id": job_id, "event": "cancel_requested", "payload": {},
    })
    status = backend_adapter.cancel(campaign, run)
    if str(status.get("backend_job_id")) != job_id:
        raise RuntimeError("cancel returned a different backend job identity")
    intent.update({"state": "VERIFIED", "verified_at": utc_now(), "result": status})
    atomic_write(path, intent)
    append_event(run_root_dir(campaign, run) / "events.jsonl", {
        "timestamp": intent["verified_at"], "run_id": run["run_id"],
        "attempt_id": attempt_id, "backend": record.get("backend"),
        "backend_job_id": job_id, "event": "cancel_verified",
        "payload": {"state": status.get("state")},
    })
    return status


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


def write_local_collection(campaign: dict[str, Any], run: dict[str, Any], summary: dict[str, Any]) -> None:
    """Persist the latest collected scientific/process observation locally."""
    root = run_root_dir(campaign, run)
    attempt_id = selected_attempt_id(run) or str(backend_record(campaign, run)["attempt_id"])
    attempt_path = root / "attempts" / attempt_id / "collection.json"
    atomic_write(attempt_path, summary)
    current = ExperimentStateStore(root).load_backend()
    if current and current.get("attempt_id") == attempt_id:
        atomic_write(root / "collection.json", summary)


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
BACKENDS = build_registry(backend_services())


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
        write_local_collection(campaign, run, collection)
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
    )


def watch_runs(
    campaign: dict[str, Any], runs: list[dict[str, Any]], *, attempt_id: str,
    interval_seconds: float, timeout_seconds: float, until: str,
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
    if until not in {"terminal", "first-metric"}:
        raise ValueError("--until must be terminal or first-metric")
    started = time.monotonic()
    pending = {str(run["run_id"]): run for run in runs}
    failed_gate_run_ids: list[str] = []
    polls = 0
    while pending:
        polls += 1
        for run_id, run in list(pending.items()):
            backend_adapter = BACKENDS.get(str(run["backend"]["kind"]))
            observation = observe_run(
                campaign, run, backend_adapter, attempt_id=attempt_id
            )
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
        )
    )
    parser.add_argument("--run", action="append", default=[], help="limit to this run ID; repeatable")
    parser.add_argument("--attempt-id", default="attempt-001")
    parser.add_argument("--dry-run", action="store_true")
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
        "--until", choices=("terminal", "first-metric"), default="terminal",
        help="watch completion gate",
    )
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
                record_submission_intent(campaign, run, args.attempt_id)
            assert manifest is not None
            job_id = backend_adapter.submit(campaign, run, manifest, dry_run=args.dry_run)
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
            write_local_collection(campaign, run, summary)
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
