#!/usr/bin/env python
"""Prepare, submit, inspect, and collect project campaigns through registered backends."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
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
    sanitize_command,
    utc_now,
)
from experiment_campaign import load_and_resolve_campaign  # noqa: E402
from experiment_policy import decide_next_action  # noqa: E402
from experiment_control.backends import build_registry  # noqa: E402
from experiment_control.backends.services import BackendServices  # noqa: E402
from experiment_control.projects import build_project_registry  # noqa: E402
from experiment_control.runner import (  # noqa: E402
    CommandResult,
    CommandRunner,
    SubprocessRunner,
)


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
        validate_run(run, project=str(payload["project"]))
        if run["run_id"] in seen:
            raise ValueError(f"duplicate run_id: {run['run_id']}")
        seen.add(run["run_id"])
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
    BACKENDS.get(str(backend["kind"])).validate(run)
    if project is not None:
        PROJECTS.get(project).validate_run(run)


def source_identity(campaign: dict[str, Any]) -> str:
    """Resolve the runtime-tree identity, computing it when set to ``auto``."""
    configured = str(campaign.get("source_id", "auto"))
    if configured != "auto":
        return configured
    bundle = PROJECTS.get(str(campaign["project"])).source_bundle(REPO_ROOT)
    if not bundle.identity_command:
        raise ValueError(f"project {campaign['project']} has no source identity command")
    result = run_command(list(bundle.identity_command), cwd=bundle.root)
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
    manifest_path = local_run_dir(campaign, run) / "manifest.yaml"
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
            "DATA_ROOT": str(storage["data_root"]),
            "PROJECT_DATA_ROOT": str(storage["project_data_root"]),
            "HF_HOME": str(storage["hf_home"]),
            "HF_DATASETS_CACHE": str(storage["hf_datasets_cache"]),
        }
    )
    env.update(PROJECTS.get(str(campaign["project"])).environment(campaign, run))
    env.update(backend.environment(campaign, run, source_id, attempt_id))
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
    local_dir = local_run_dir(campaign, run)
    manifest_path = local_dir / "manifest.yaml"
    remote_run_dir = str(run["storage"]["run_dir"])
    project = PROJECTS.get(str(campaign["project"]))
    overrides = resolved_run_overrides(campaign, run, remote_run_dir)
    resolved = project.resolve_config(str(run["config"]), overrides)
    command = launcher_command(campaign, run, source_id, attempt_id)
    bundle = project.source_bundle(REPO_ROOT)
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
        "execution": {
            "source_mount": bundle.container_path,
            "workdir": bundle.container_path,
        },
        "retry": run.get("retry", {"max_infra_retries": 0}),
    }
    if manifest_path.exists():
        existing = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "campaign", "project", "run_id", "source_id", "image_id", "resolved_config",
            "backend",
        )
        conflicts = [key for key in immutable_keys if existing.get(key) != manifest.get(key)]
        if existing.get("execution") not in (None, manifest["execution"]):
            conflicts.append("execution")
        if conflicts:
            raise ValueError(f"existing control manifest conflicts in {conflicts}: {manifest_path}")
        base_manifest = existing
    else:
        atomic_write(manifest_path, manifest, yaml_format=True)
        base_manifest = manifest
    effective = dict(base_manifest)
    effective["attempt_id"] = attempt_id
    effective["command"] = sanitize_command(command)
    effective["execution"] = manifest["execution"]
    attempt_path = local_dir / "attempts" / attempt_id / "attempt.yaml"
    if attempt_path.exists():
        previous_attempt = yaml.safe_load(attempt_path.read_text(encoding="utf-8"))
        comparable_attempt = dict(previous_attempt)
        comparable_attempt.setdefault("execution", effective["execution"])
        if comparable_attempt != effective:
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


def record_submission(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str, backend_job_id: str
) -> None:
    """Atomically record scheduler acceptance in controller-owned metadata."""
    local_dir = local_run_dir(campaign, run)
    store = ExperimentStateStore(local_dir)
    store.reconcile_submission(
        project=str(campaign["project"]), run_id=str(run["run_id"]),
        attempt_id=attempt_id, backend_job_id=backend_job_id, state=RunState.QUEUED,
    )


def record_submission_intent(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str
) -> dict[str, Any]:
    """Durably record scheduler mutation intent before crossing the API boundary."""
    local_dir = local_run_dir(campaign, run)
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
    local_dir = local_run_dir(campaign, run)
    root_record = local_dir / "backend.json"
    if root_record.is_file():
        payload = json.loads(root_record.read_text(encoding="utf-8"))
        if payload.get("backend_job_id") and payload.get("attempt_id") == attempt_id:
            return str(payload["backend_job_id"])
    submission = ExperimentStateStore(local_dir).read_submission(attempt_id)
    if not submission or submission.get("state") != "SUBMITTING":
        return None
    intent = {**submission, **submission.get("request", {})}
    backend = BACKENDS.get(str(run["backend"]["kind"]))
    job_id = backend.recover_submission(run, intent, attempt_id)
    if job_id:
        record_submission(campaign, run, attempt_id, str(job_id))
        return str(job_id)
    return None


def ensure_attempt_not_submitted(
    campaign: dict[str, Any], run: dict[str, Any], attempt_id: str
) -> None:
    """Reject a second scheduler mutation for an already submitted attempt."""
    submission = ExperimentStateStore(local_run_dir(campaign, run)).read_submission(attempt_id)
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
        repo_root=REPO_ROOT, script_dir=SCRIPT_DIR, run_command=run_command, local_run_dir=local_run_dir,
        backend_record=backend_record, summarize_run=summarize_project_run,
        parse_metric=parse_project_metric,
    )


def summarize_project_run(campaign: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Dispatch result interpretation to the campaign's scientific project."""
    return PROJECTS.get(str(campaign["project"])).summarize(run_dir)


def parse_project_metric(campaign: dict[str, Any], line: str) -> dict[str, Any] | None:
    """Dispatch training-log interpretation without teaching a backend project syntax."""
    return PROJECTS.get(str(campaign["project"])).parse_metric(line)


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


PROJECTS = build_project_registry()
BACKENDS = build_registry(backend_services())


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
            bundle = PROJECTS.get(str(campaign["project"])).source_bundle(REPO_ROOT)
            staged = backend_adapter.stage(campaign, run, identity, bundle)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
