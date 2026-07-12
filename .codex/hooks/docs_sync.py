#!/usr/bin/env python3
"""Require Codex to review documentation when implementation boundaries change."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPOSITORIES = {
    "ELF": {
        "relative_path": "ELF",
        "implementation": (
            "src/elf_experiments/",
            "tools/experiment",
            "tools/instantiate_campaign.py",
            "requirements.txt",
        ),
        "documentation": ("README.md", "docs/"),
    },
    "ml-experiment-control": {
        "relative_path": "ml-experiment-control",
        "implementation": ("src/experiment_control/", "pyproject.toml"),
        "documentation": ("README.md", "docs/"),
    },
    "research-console": {
        "relative_path": "research-console",
        "implementation": ("src/research_console/", "pyproject.toml"),
        "documentation": ("README.md", "README.zh-CN.md", "docs/"),
    },
}

NO_DOCS_MARKER = re.compile(r"Docs impact:\s*none\s*[—-]\s*\S", re.IGNORECASE)


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _repo_root(cwd: Path) -> Path:
    return Path(_git(cwd, "rev-parse", "--show-toplevel"))


def _repositories(root: Path) -> dict[str, Path]:
    parent = root.parent
    repositories: dict[str, Path] = {}
    for name, policy in REPOSITORIES.items():
        candidate = parent / str(policy["relative_path"])
        if candidate.is_dir() and _git(candidate, "rev-parse", "--is-inside-work-tree", check=False) == "true":
            repositories[name] = candidate
    return repositories


def _state_path(root: Path, session_id: str) -> Path:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:20]
    git_dir = Path(_git(root, "rev-parse", "--absolute-git-dir"))
    return git_dir / "codex-hooks" / f"docs-sync-{digest}.json"


def _write_baseline(path: Path, repositories: dict[str, Path]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "repositories": {
            name: {"path": str(repo), "head": _git(repo, "rev-parse", "HEAD")}
            for name, repo in repositories.items()
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _changed_files(repo: Path, baseline: str) -> set[str]:
    changed: set[str] = set()
    for args in (
        ("diff", "--name-only", f"{baseline}..HEAD"),
        ("diff", "--name-only", "HEAD"),
        ("ls-files", "--others", "--exclude-standard"),
    ):
        changed.update(line for line in _git(repo, *args).splitlines() if line)
    return changed


def _matches(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in prefixes)


def _missing_doc_reviews(state: dict[str, Any]) -> list[tuple[str, list[str]]]:
    missing: list[tuple[str, list[str]]] = []
    for name, baseline in state.get("repositories", {}).items():
        policy = REPOSITORIES.get(name)
        repo = Path(str(baseline.get("path", "")))
        if policy is None or not repo.is_dir():
            continue
        changed = _changed_files(repo, str(baseline["head"]))
        implementation = sorted(
            path for path in changed
            if _matches(path, tuple(policy["implementation"]))
        )
        documentation = any(
            _matches(path, tuple(policy["documentation"])) for path in changed
        )
        if implementation and not documentation:
            missing.append((name, implementation))
    return missing


def _stop(payload: dict[str, Any], state_path: Path) -> None:
    if not state_path.is_file():
        print(json.dumps({
            "decision": "block",
            "reason": (
                "Documentation-sync baseline is missing. Continue the turn, review "
                "documentation impact, and retry after the hook records a baseline."
            ),
        }))
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    missing = _missing_doc_reviews(state)
    last_message = str(payload.get("last_assistant_message") or "")
    if missing and not NO_DOCS_MARKER.search(last_message):
        details = "; ".join(
            f"{name}: {', '.join(paths[:5])}" for name, paths in missing
        )
        print(json.dumps({
            "decision": "block",
            "reason": (
                "Implementation changed without a documentation change. Review and "
                "update the relevant README/docs, or if no update is warranted, end "
                "the final response with `Docs impact: none — <specific reason>`. "
                f"Detected: {details}"
            ),
        }))
        return
    print(json.dumps({"continue": True}))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        root = _repo_root(Path(str(payload.get("cwd") or Path.cwd())))
        repositories = _repositories(root)
        state_path = _state_path(root, str(payload.get("session_id") or "unknown"))
        event = payload.get("hook_event_name")
        if event == "SessionStart":
            _write_baseline(state_path, repositories)
        elif event == "Stop":
            _stop(payload, state_path)
        return 0
    except Exception as error:  # Fail closed: a broken guard must be visible.
        print(json.dumps({
            "decision": "block",
            "reason": f"Documentation-sync hook failed: {error}",
        }))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
