import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "launch.sh"


def run_launch(tmp_path: Path, extra_env: dict[str, str]) -> list[str]:
    """Run the shell launcher against a capturing torchrun stub."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "torchrun.args"
    torchrun = bin_dir / "torchrun"
    torchrun.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n', encoding="utf-8")
    torchrun.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "CAPTURE": str(capture),
            "NGPU": "2",
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    env.update(extra_env)
    subprocess.run(
        ["bash", str(SCRIPT), "train", "config.yml"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return capture.read_text(encoding="utf-8").splitlines()


def test_slurm_job_gets_deterministic_nondefault_master_port(tmp_path: Path) -> None:
    """Concurrent Slurm jobs must not all bind torchrun's port 29500."""
    args = run_launch(tmp_path, {"SLURM_JOB_ID": "1737", "MASTER_PORT": ""})
    assert "--master_port=21737" in args


def test_explicit_master_port_is_preserved(tmp_path: Path) -> None:
    """A caller-provided rendezvous port remains authoritative."""
    args = run_launch(tmp_path, {"SLURM_JOB_ID": "1737", "MASTER_PORT": "31000"})
    assert "--master_port=31000" in args
