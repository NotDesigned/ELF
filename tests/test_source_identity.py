import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/source_identity.sh"


def test_campaign_identity_is_distinct_from_runtime_identity():
    runtime = subprocess.run(["bash", SCRIPT, "--runtime"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    campaign = subprocess.run(
        ["bash", SCRIPT, "--campaign", ROOT / "experiments/templates/backend_smoke_slurm.yml"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    assert runtime.startswith("runtime.")
    assert campaign.startswith("campaign.")
    assert runtime != campaign


def test_runtime_source_changes_runtime_identity(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "stable.sh").write_text("stable\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "scripts/stable.sh"], cwd=tmp_path, check=True)

    before = subprocess.run(
        ["bash", SCRIPT, "--runtime"], cwd=tmp_path,
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    (scripts / "probe.tmp").write_text("package-change\n", encoding="utf-8")
    after = subprocess.run(
        ["bash", SCRIPT, "--runtime"], cwd=tmp_path,
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    assert after != before
