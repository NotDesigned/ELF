import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/source_identity.sh"


def test_campaign_identity_is_distinct_from_runtime_identity():
    runtime = subprocess.run(["bash", SCRIPT, "--runtime"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    campaign = subprocess.run(
        ["bash", SCRIPT, "--campaign", ROOT / "experiments/campaigns/backend_smoke_slurm_20260711.yml"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    assert runtime.startswith("runtime.")
    assert campaign.startswith("campaign.")
    assert runtime != campaign
