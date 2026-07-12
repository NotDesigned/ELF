#!/usr/bin/env python
"""ELF runtime CLI over the package-owned durable experiment state."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import asdict
from typing import Any

import yaml

from experiment_control.manifest import (
    ExperimentStateStore,
    RunState,
    require_immutable,
    sanitize_command,
    utc_now,
    validate_identity,
)
from experiment_control.run_manifest import build_run_manifest, comparable_manifest

def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Create or validate a scientific run and append one immutable attempt.

    A new run writes its resolved manifest. A retry may reuse the run only when
    project, run ID, source, image, and scientific config are identical. Attempt
    directories are never overwritten. The function also initializes backend
    and normalized status files and records ``attempt_created``.

    Returns:
        Paths to the run manifest and newly created attempt manifest.
    """
    validate_identity("run_id", args.run_id)
    validate_identity("attempt_id", args.attempt_id)
    if args.require_immutable_identities:
        require_immutable("source_id", args.source_id)
        require_immutable("image_id", args.image_id)

    store = ExperimentStateStore(args.output_dir)
    run_dir = store.run_dir
    manifest_path = store.manifest_path
    attempt_path = store.attempt_path(args.attempt_id)
    if attempt_path.exists():
        raise FileExistsError(
            f"attempt already exists: {attempt_path}; choose a new ATTEMPT_ID"
        )

    research_contract: dict[str, Any] | None = None
    research_role: str | None = None
    encoded_contract = getattr(args, "research_contract_b64", "")
    requested_role = getattr(args, "research_role", "")
    if encoded_contract or requested_role:
        if not encoded_contract or not requested_role:
            raise ValueError("research contract and role must be supplied together")
        validate_identity("research_role", requested_role)
        try:
            decoded = base64.urlsafe_b64decode(
                encoded_contract.encode("ascii")
            ).decode("utf-8")
            research_contract = json.loads(decoded)
        except (UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("research contract payload is not valid base64 JSON") from error
        if not isinstance(research_contract, dict) or research_contract.get("schema_version") != 1:
            raise ValueError("research contract payload must be a schema-version-1 mapping")
        research_role = requested_role
    elif manifest_path.is_file():
        existing = store.load_manifest()
        if existing.get("research_contract") is not None:
            research_contract = existing["research_contract"]
            research_role = existing.get("research_role")

    from .projects import build_project_registry

    project = build_project_registry().get(args.project)
    config = project.resolve_config(args.config, args.config_override)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("resolved training command must not be empty")
    command = sanitize_command(command)
    asset_requirements = [
        asdict(item) for item in project.plan_assets(args.config, args.config_override)
    ]

    manifest = build_run_manifest(
        project=args.project, run_id=args.run_id, created_at=utc_now(),
        config_path=args.config, resolved_config=config, source_id=args.source_id,
        runtime_tree_id=args.runtime_tree_id or args.source_id,
        git_commit=args.git_commit or None, campaign_id=args.campaign_id or None,
        campaign=args.campaign or None, image_id=args.image_id,
        run_dir=str(run_dir), max_infra_retries=args.max_infra_retries,
        backend={"kind": args.backend},
        resources={
            "gpus": args.gpus, "nodes": args.nodes, "quota": args.quota,
            "resource_spec": args.resource_spec or None,
        },
        storage={"run_dir": str(run_dir), "checkpoint_dir": str(run_dir)},
        command=command, execution={"source_mount": None, "workdir": None},
        config_overrides=list(args.config_override),
        assets=asset_requirements, checkpoint={"save_freq": config.get("save_freq")},
        evaluation={},
        research_contract=research_contract, research_role=research_role,
    )

    if manifest_path.is_file():
        existing = store.load_manifest()
        # A controller-prepared v2 Run manifest is authoritative.  Runtime
        # arguments can reconstruct the scientific core and scheduler class,
        # but not the controller's full rendered command/mount description.
        # Validate every identity the runtime does know and preserve the
        # existing manifest byte-for-byte rather than weakening it.
        for key in (
            "project", "run_id", "source_id", "runtime_tree_id", "git_commit",
            "campaign_id", "campaign", "image_id", "config_path",
        ):
            if existing.get(key) != manifest.get(key):
                raise ValueError(f"existing run manifest conflicts at {key}")
        if comparable_manifest(existing).get("resolved_config") != comparable_manifest(manifest).get("resolved_config"):
            raise ValueError("existing run manifest conflicts at resolved_config")
        existing_backend = existing.get("backend") or {}
        if existing_backend.get("kind") != args.backend:
            raise ValueError("existing run manifest conflicts at backend.kind")
        existing_resources = existing.get("resources") or {}
        for key, requested in (("gpus", args.gpus), ("nodes", args.nodes)):
            if existing_resources.get(key) != requested:
                raise ValueError(f"existing run manifest conflicts at resources.{key}")
        manifest = existing
    else:
        manifest = store.ensure_manifest(manifest)

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
        "research_role": research_role,
    }
    store.create_attempt(attempt)
    store.initialize_attempt_records(args.attempt_id)
    return {"manifest": str(manifest_path), "attempt": str(attempt_path)}


def record(args: argparse.Namespace) -> dict[str, Any]:
    """Atomically update normalized status and append a lifecycle event.

    Recording is allowed only after the run and attempt manifests exist and
    their identities agree with the command arguments.
    """
    validate_identity("run_id", args.run_id)
    validate_identity("attempt_id", args.attempt_id)
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
    parser.add_argument("--research-contract-b64", default="")
    parser.add_argument("--research-role", default="")
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
