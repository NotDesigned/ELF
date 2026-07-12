from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".codex" / "hooks" / "docs_sync.py"
HOOK_CONFIG = ROOT / ".codex" / "hooks.json"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repository(parent: Path) -> Path:
    repo = parent / "ELF"
    (repo / "src" / "elf_experiments").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "src" / "elf_experiments" / "controller.py").write_text("old\n")
    (repo / "docs" / "experiment_workflow.md").write_text("old\n")
    _git(repo, "init")
    _git(repo, "config", "user.email", "codex-hook@example.invalid")
    _git(repo, "config", "user.name", "Codex Hook Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def _run(repo: Path, event: str, *, message: str = "") -> dict[str, object] | None:
    payload = {
        "session_id": "test-session",
        "cwd": str(repo),
        "hook_event_name": event,
        "last_assistant_message": message,
    }
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        cwd=repo,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout) if result.stdout else None


def test_hook_configuration_registers_baseline_and_stop_checks() -> None:
    config = json.loads(HOOK_CONFIG.read_text(encoding="utf-8"))

    assert set(config["hooks"]) == {"SessionStart", "Stop"}
    assert "docs_sync.py" in config["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    assert "docs_sync.py" in config["hooks"]["Stop"][0]["hooks"][0]["command"]


def test_hook_blocks_source_change_without_documentation(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    assert _run(repo, "SessionStart") is None
    (repo / "src" / "elf_experiments" / "controller.py").write_text("changed\n")

    output = _run(repo, "Stop")

    assert output is not None
    assert output["decision"] == "block"
    assert "controller.py" in str(output["reason"])


def test_hook_accepts_matching_documentation_change_after_commit(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    _run(repo, "SessionStart")
    (repo / "src" / "elf_experiments" / "controller.py").write_text("changed\n")
    (repo / "docs" / "experiment_workflow.md").write_text("changed\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "change implementation and docs")

    assert _run(repo, "Stop") == {"continue": True}


def test_hook_accepts_explicit_no_docs_impact_reason(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    _run(repo, "SessionStart")
    (repo / "src" / "elf_experiments" / "controller.py").write_text("changed\n")

    output = _run(
        repo,
        "Stop",
        message="Docs impact: none — internal typo with no behavior change.",
    )

    assert output == {"continue": True}
