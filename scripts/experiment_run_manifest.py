"""One canonical scientific run-manifest constructor shared by control and runtime."""

from __future__ import annotations

from typing import Any


def build_run_manifest(
    *, project: str, run_id: str, created_at: str, config_path: str,
    resolved_config: dict[str, Any], source_id: str, runtime_tree_id: str,
    git_commit: str | None, campaign_id: str | None, campaign: str | None,
    image_id: str, run_dir: str, max_infra_retries: int,
) -> dict[str, Any]:
    """Build the platform-neutral immutable identity written as manifest.yaml."""
    return {
        "schema_version": 1,
        "project": project,
        "run_id": run_id,
        "created_at": created_at,
        "config_path": config_path,
        "resolved_config": resolved_config,
        "source_id": source_id,
        "runtime_tree_id": runtime_tree_id,
        "git_commit": git_commit,
        "campaign_id": campaign_id,
        "campaign": campaign,
        "image_id": image_id,
        "seed": resolved_config.get("seed"),
        "storage": {"run_dir": run_dir, "checkpoint_dir": run_dir},
        "resume_policy": {
            "enabled": True,
            "max_infra_retries": max_infra_retries,
        },
    }


def comparable_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Remove creation time, the only non-identity run-manifest field."""
    return {key: value for key, value in manifest.items() if key != "created_at"}
