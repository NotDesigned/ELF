from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "push_registry_image.sh"
IMAGE = "registry.example.test/team/elf:runtime-source123"
DIGEST = "sha256:" + "a" * 64


def write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)


def fake_environment(
    tmp_path: Path, docker_push: str, *, publisher: str = "crane"
) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "calls"
    docker = f"""
printf 'docker %s\\n' "$*" >> {marker!s}
if [[ "$1" == image && "$2" == inspect && "${{3:-}}" != --format ]]; then exit 0; fi
if [[ "$1" == image && "$2" == inspect && "$3" == --format ]]; then printf '1048576\\n'; exit 0; fi
if [[ "$1" == info ]]; then exit 0; fi
if [[ "$1" == push ]]; then {docker_push}; fi
if [[ "$1" == save ]]; then : > "$3"; exit 0; fi
exit 2
"""
    crane = f"""
printf 'crane %s\\n' "$*" >> {marker!s}
if [[ "$1" == push ]]; then exit 0; fi
if [[ "$1" == digest ]]; then printf '{DIGEST}\\n'; exit 0; fi
exit 2
"""
    skopeo = f"""
printf 'skopeo %s\\n' "$*" >> {marker!s}
if [[ "$1" == copy ]]; then exit 0; fi
if [[ "$1" == inspect ]]; then printf '{DIGEST}\\n'; exit 0; fi
exit 2
"""
    write_executable(bin_dir / "docker", docker)
    if publisher == "crane":
        write_executable(bin_dir / "crane", crane)
    elif publisher == "skopeo":
        write_executable(bin_dir / "skopeo", skopeo)
    else:
        raise ValueError(f"unsupported fake publisher: {publisher}")
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    (docker_config / "config.json").write_text(
        '{"auths":{"registry.example.test":{"auth":"not-read-by-check"}}}',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update({
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "TMPDIR": str(tmp_path),
        "DOCKER_PUSH_TIMEOUT_SECONDS": "5",
        "DOCKER_SAVE_TIMEOUT_SECONDS": "5",
        "REGISTRY_OPERATION_TIMEOUT_SECONDS": "5",
        "DOCKER_CONFIG": str(docker_config),
    })
    return env


def run_script(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_does_not_push(tmp_path: Path) -> None:
    env = fake_environment(tmp_path, "exit 99")
    result = run_script(env, "--dry-run", IMAGE)
    assert result.returncode == 0
    assert "publisher=crane" in result.stdout
    calls = (tmp_path / "calls").read_text(encoding="utf-8")
    assert "docker image inspect" in calls
    assert "docker push" not in calls


def test_dry_run_requires_registry_credential_reference(tmp_path: Path) -> None:
    env = fake_environment(tmp_path, "exit 99")
    (Path(env["DOCKER_CONFIG"]) / "config.json").write_text("{}", encoding="utf-8")
    result = run_script(env, "--dry-run", IMAGE)
    assert result.returncode == 2
    assert "credential reference is missing" in result.stderr
    assert "docker push" not in (tmp_path / "calls").read_text(encoding="utf-8")


def test_successful_docker_push_is_digest_verified(tmp_path: Path) -> None:
    env = fake_environment(tmp_path, "exit 0")
    result = run_script(env, IMAGE)
    assert result.returncode == 0
    assert result.stdout.strip() == f"registry.example.test/team/elf@{DIGEST}"
    calls = (tmp_path / "calls").read_text(encoding="utf-8")
    assert "docker push" in calls
    assert "crane digest" in calls
    assert "crane push" not in calls


def test_auth_failure_is_redacted_and_never_falls_back(tmp_path: Path) -> None:
    env = fake_environment(
        tmp_path,
        "printf 'denied: access_key_secret=very-secret\\n' >&2; exit 1",
    )
    result = run_script(env, IMAGE)
    assert result.returncode == 30
    assert "class=auth" in result.stderr
    assert "very-secret" not in result.stderr
    assert "<redacted>" in result.stderr
    calls = (tmp_path / "calls").read_text(encoding="utf-8")
    assert "crane push" not in calls


def test_transport_failure_uses_archive_fallback_and_cleans_up(tmp_path: Path) -> None:
    env = fake_environment(tmp_path, "printf 'received unexpected HTTP status: 502\\n' >&2; exit 1")
    result = run_script(env, IMAGE)
    assert result.returncode == 0
    assert result.stdout.strip() == f"registry.example.test/team/elf@{DIGEST}"
    calls = (tmp_path / "calls").read_text(encoding="utf-8")
    assert "docker save -o" in calls
    assert "crane push" in calls
    assert "crane digest" in calls
    assert not list(tmp_path.glob("elf-registry.*"))


def test_skopeo_is_used_when_crane_is_unavailable(tmp_path: Path) -> None:
    env = fake_environment(
        tmp_path,
        "printf 'unexpected EOF\\n' >&2; exit 1",
        publisher="skopeo",
    )
    result = run_script(env, IMAGE)
    assert result.returncode == 0
    assert result.stdout.strip() == f"registry.example.test/team/elf@{DIGEST}"
    calls = (tmp_path / "calls").read_text(encoding="utf-8")
    assert "skopeo copy docker-archive:" in calls
    assert "skopeo inspect --format" in calls
    assert not list(tmp_path.glob("elf-registry.*"))


def test_mutable_tag_is_rejected_before_docker(tmp_path: Path) -> None:
    env = fake_environment(tmp_path, "exit 0")
    result = run_script(env, "registry.example.test/team/elf:latest")
    assert result.returncode == 2
    assert "mutable tag" in result.stderr
    assert not (tmp_path / "calls").exists()
