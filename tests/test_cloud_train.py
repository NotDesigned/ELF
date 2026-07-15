import os
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/cloud_train.sh"
CONFIG_ROOT = REPO_ROOT / "src/configs/training_configs/ablations/owt_elfb"


def offline_env(tmp_path: Path) -> dict[str, str]:
    """Create an isolated offline cache containing shared ELF dependencies."""
    data_root = tmp_path / "data"
    hf_home = data_root / "hf"
    datasets = hf_home / "datasets"
    (hf_home / "hub/models--t5-small").mkdir(parents=True)
    (hf_home / "hub/models--gpt2-large").mkdir(parents=True)
    (datasets / "embedded-language-flows___openwebtext-t5").mkdir(parents=True)
    env = os.environ.copy()
    env.update(
        {
            "DATA_ROOT": str(data_root),
            "HF_HOME": str(hf_home),
            "HF_DATASETS_CACHE": str(datasets),
            "BAKED_HF_HOME": str(tmp_path / "no-baked-hf"),
            "BAKED_CHECKPOINT_ROOT": str(tmp_path / "no-baked-checkpoints"),
            "REQUIRE_OFFLINE_CACHE": "1",
            "HYDRATE_ONLY": "1",
            "NGPU": "1",
        }
    )
    return env


def test_learned_plan_does_not_require_sentence_t5_cache(tmp_path):
    env = offline_env(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), str(CONFIG_ROOT / "tier0_2_learned_main_len256.yml")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_frozen_plan_requires_config_selected_sentence_t5_cache(tmp_path):
    env = offline_env(tmp_path)
    result = subprocess.run(
        ["bash", str(SCRIPT), str(CONFIG_ROOT / "tier0_1_sentence_t5_len256.yml")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "models--sentence-transformers--sentence-t5-xl" in result.stderr


def test_explicit_config_override_is_frozen_in_manifest(tmp_path):
    env = offline_env(tmp_path)
    env.update(
        {
            "HYDRATE_ONLY": "0",
            "PREPARE_ONLY": "1",
            "REQUIRE_IMMUTABLE_IDENTITIES": "0",
            "RUN_ID": "override-manifest-test",
            "OUTPUT_DIR": str(tmp_path / "run"),
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(CONFIG_ROOT / "tier0_2_learned_main_len256.yml"),
            "--config_override",
            "epochs=1",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = (tmp_path / "run" / "manifest.yaml").read_text(encoding="utf-8")
    assert "epochs: 1" in manifest


def test_local_dataset_override_is_validated_as_a_directory(tmp_path):
    env = offline_env(tmp_path)
    local_dataset = tmp_path / "local-dataset"
    local_dataset.mkdir()
    env["HYDRATE_ONLY"] = "1"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(CONFIG_ROOT / "tier0_2_learned_main_len256.yml"),
            "--config_override",
            f"data_path={local_dataset}",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_launcher_uses_installed_experiment_control_dependency(tmp_path):
    env = offline_env(tmp_path)
    env.update({
        "PYTHONPATH": "/path/that/does/not/exist",
        "REQUIRE_OFFLINE_CACHE": "0",
        "HYDRATE_ONLY": "1",
    })
    result = subprocess.run(
        ["bash", str(SCRIPT), str(CONFIG_ROOT / "tier0_0_pure_elf_len256.yml")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "experiment_control=" in result.stdout
    assert "/packages/experiment-control/" not in result.stdout


def test_evaluation_launcher_freezes_checkpoint_and_mode_in_manifest(tmp_path):
    env = offline_env(tmp_path)
    checkpoint = tmp_path / "checkpoint_42"
    checkpoint.write_bytes(b"checkpoint")
    env.update({
        "ELF_RUN_MODE": "eval",
        "HYDRATE_ONLY": "0",
        "PREPARE_ONLY": "1",
        "REQUIRE_IMMUTABLE_IDENTITIES": "0",
        "RUN_ID": "evaluation-manifest-test",
        "OUTPUT_DIR": str(tmp_path / "run"),
    })

    result = subprocess.run(
        [
            "bash", str(SCRIPT),
            str(CONFIG_ROOT / "tier0_2_learned_main_len256.yml"),
            "--checkpoint_path", str(checkpoint),
            "--seeds", "42,123",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = yaml.safe_load(
        (tmp_path / "run" / "manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["command"][:3] == ["bash", "scripts/launch.sh", "eval"]
    assert str(checkpoint) in manifest["command"]
    assert "42,123" in manifest["command"]


def test_evaluation_launcher_fails_before_manifest_for_missing_checkpoint(tmp_path):
    env = offline_env(tmp_path)
    env.update({
        "ELF_RUN_MODE": "eval",
        "HYDRATE_ONLY": "0",
        "REQUIRE_IMMUTABLE_IDENTITIES": "0",
        "RUN_ID": "evaluation-missing-checkpoint",
        "OUTPUT_DIR": str(tmp_path / "run"),
    })

    result = subprocess.run(
        [
            "bash", str(SCRIPT),
            str(CONFIG_ROOT / "tier0_2_learned_main_len256.yml"),
            "--checkpoint_path", str(tmp_path / "missing"),
            "--seed", "42",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "required file is missing or empty" in result.stderr
    assert not (tmp_path / "run" / "manifest.yaml").exists()
