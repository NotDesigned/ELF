from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_documented_manifest_command_is_executable() -> None:
    result = subprocess.run(
        [sys.executable, "tools/experiment_manifest.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_experiment_workflow_uses_current_modules_and_storage_profiles() -> None:
    text = _read("docs/experiment_workflow.md")

    assert "python tools/experiment_manifest.py" in text
    assert "python -m elf_experiments.manifest" not in text
    assert "elf_experiments.run_manifest.build_run_manifest" in text
    assert "experiment_run_manifest.build_run_manifest" not in text
    assert "src/elf_experiments/campaign.py" in text
    assert "src/elf_experiments/controller.py" in text
    assert "/data/liangluocheng/elf" in text
    assert "/datapool/liangluocheng/elf" in text
    for stale_table_entry in (
        "| `experiment_campaign.py`",
        "| `instantiate_campaign.py`",
        "| `experiment_manifest.py`",
        "| `experiment_assets.py`",
        "| `experiment_overrides.py`",
        "| `experiment_policy.py`",
    ):
        assert stale_table_entry not in text


def test_research_guides_use_unique_instance_and_durable_state_root() -> None:
    for path in ("docs/agent_research_guide.md", "docs/experiment_workflow.md"):
        text = _read(path)
        assert "date -u +%Y%m%dT%H%M%SZ" in text
        assert '--instance "$INSTANCE"' in text
        assert 'outputs/experiment_campaigns/state/$INSTANCE' in text
        assert "/tmp/elf-controller-state" not in text


def test_fusion_doc_describes_current_loss_and_external_reference() -> None:
    text = _read("docs/fusion_architecture.md")

    assert "CE and L2 token" in text
    assert "plan loss is added after" in text
    assert "not vendored in this repository" in text
    assert "L_sent joins the" not in text


def test_config_reference_describes_compile_and_resolved_state_location() -> None:
    text = _read("docs/config_reference.md")

    assert "training and generation/evaluation" in text
    assert "resolved `$OUTPUT_DIR`" in text
    assert "campaign `storage.run_dir`" in text
    assert "/datapool/liangluocheng/elf" in text
    assert "canonical state under `/data/<project>/runs/<run-id>`" not in text
