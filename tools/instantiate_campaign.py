#!/usr/bin/env python
"""Create one reviewable campaign file with fresh run identities."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elf_experiments.campaign import instantiate_campaign_template  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--register",
        action="store_true",
        help=(
            "write to experiments/campaigns by default and atomically append "
            "the campaign to the project catalog"
        ),
    )
    parser.add_argument(
        "--project-file",
        type=Path,
        default=REPO_ROOT / "experiments/research_project.yaml",
        help="project catalog used by --register",
    )
    parser.add_argument(
        "--local-root", type=Path,
        help="override controller metadata root in the generated definition",
    )
    return parser.parse_args(argv)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _atomic_replace_yaml(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        _write_exclusive_yaml(temporary, payload)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _catalog_file_value(output: Path) -> str:
    try:
        return str(output.relative_to(REPO_ROOT))
    except ValueError:
        return str(output)


def write_and_register_campaign(
    *, campaign: dict, output: Path, project_file: Path
) -> None:
    """Create a campaign and register it while holding a catalog sidecar lock.

    The campaign is never overwritten. Duplicate names and duplicate file
    registrations fail before file creation. If catalog replacement fails, the
    newly created campaign is removed so callers never observe an unregistered
    partial result from this operation.
    """
    project_file = project_file.resolve()
    project_file.parent.mkdir(parents=True, exist_ok=True)
    lock_path = project_file.with_name(f".{project_file.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        project = yaml.safe_load(project_file.read_text(encoding="utf-8"))
        if not isinstance(project, dict):
            raise ValueError(f"project catalog must be a mapping: {project_file}")
        if project.get("project") != campaign.get("project"):
            raise ValueError(
                "campaign project does not match project catalog: "
                f"{campaign.get('project')!r} != {project.get('project')!r}"
            )
        entries = project.setdefault("campaigns", [])
        if not isinstance(entries, list):
            raise ValueError("project catalog campaigns must be a list")
        name = str(campaign["campaign"])
        file_value = _catalog_file_value(output)
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("project catalog campaign entries must be mappings")
            if entry.get("name") == name:
                raise FileExistsError(f"campaign already registered by name: {name}")
            if entry.get("file") == file_value:
                raise FileExistsError(f"campaign file already registered: {file_value}")

        _write_exclusive_yaml(output, campaign)
        try:
            entries.append({"name": name, "file": file_value})
            _atomic_replace_yaml(project_file, project)
        except Exception:
            output.unlink(missing_ok=True)
            _fsync_directory(output.parent)
            raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template = args.template.resolve()
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    campaign = instantiate_campaign_template(payload, args.instance)
    if args.local_root is not None:
        local_root = args.local_root
        if not local_root.is_absolute():
            local_root = (Path.cwd() / local_root).resolve()
        campaign["local_root"] = str(local_root)
    try:
        generated_from = str(template.relative_to(REPO_ROOT))
    except ValueError:
        generated_from = str(template)
    campaign["generated_from"] = generated_from
    output = args.output
    if output is None:
        output_root = (
            REPO_ROOT / "experiments/campaigns"
            if args.register
            else REPO_ROOT / "outputs/experiment_campaigns/definitions"
        )
        output = output_root / f"{campaign['campaign']}.yml"
    elif not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    if args.register:
        write_and_register_campaign(
            campaign=campaign,
            output=output,
            project_file=args.project_file,
        )
    else:
        _write_exclusive_yaml(output, campaign)
    print(output)
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(
            json.dumps(
                {"error": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
