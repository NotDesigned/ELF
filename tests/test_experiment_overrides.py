import pytest

from elf_experiments.overrides import operational_overrides
from elf_experiments.projects.elf import ElfProjectAdapter


def test_batch_size_has_explicit_precedence_over_global_batch_size():
    overrides = operational_overrides(
        {"RUN_ID": "run", "GLOBAL_BATCH_SIZE": "512", "BATCH_SIZE": "64"}, "/data/run"
    )
    assert overrides[-3:] == ["global_batch_size=512", "global_batch_size=null", "batch_size=64"]


def test_wandb_attempt_identity_defaults_are_deterministic():
    overrides = operational_overrides({"RUN_ID": "run-42"}, "/data/run-42")
    assert "wandb_run_name=run-42" in overrides
    assert "wandb_run_id=run-42" in overrides
    assert "wandb_resume=allow" in overrides


def test_reviewed_resume_and_remote_tracking_controls_are_operational():
    env = {
        "RESUME": "/data/elf/runs/a/checkpoints",
        "WANDB_RESUME": "must",
        "HF_REPO_ID": "team/elf-a",
    }
    assert env.keys() <= ElfProjectAdapter.safe_env_keys
    assert "HF_TOKEN" not in ElfProjectAdapter.safe_env_keys
    overrides = operational_overrides(env, "/data/elf/runs/a")
    assert "resume=/data/elf/runs/a/checkpoints" in overrides
    assert "wandb_resume=must" in overrides
    assert "hf_repo_id=team/elf-a" in overrides


def test_elf_source_bundle_excludes_local_secrets_caches_and_heavy_artifacts(tmp_path):
    excludes = set(ElfProjectAdapter().source_bundle(tmp_path).excludes)
    assert {
        ".git/", ".gitignore", ".dockerignore", ".env", ".env.*", ".netrc",
        ".npmrc", ".pypirc", ".ssh/", ".aws/", ".venv/", "venv/", "env/",
        ".claude/", ".codex/", ".cache/", "hf_cache/", "huggingface/",
        "__pycache__/", "*.py[cod]", "*$py.class", ".pytest_cache/",
        ".mypy_cache/", ".ruff_cache/", ".ipynb_checkpoints/", "*.egg-info/",
        "build/", "dist/", "outputs/", "output_dir/", "saved_models/",
        "checkpoints/", "wandb/", "runs/", "data/", "*.log", "*.tar.gz",
        "*.pt", "*.pth", "*.ckpt", "*.safetensors", ".DS_Store", "Thumbs.db",
    } <= excludes


def test_elf_evaluation_command_freezes_checkpoint_and_seeds():
    adapter = ElfProjectAdapter()
    run = {
        "run_id": "eval-a",
        "config": "config.yml",
        "operation": "evaluate",
        "evaluation": {
            "checkpoint_path": "/data/run/checkpoint_42",
            "seeds": [42, 123],
        },
        "config_overrides": ["num_samples=256"],
    }

    adapter.validate_run(run)
    assert adapter.command(run) == [
        "env", "ELF_RUN_MODE=eval", "bash", "scripts/cloud_train.sh", "config.yml",
        "--config_override", "num_samples=256",
        "--checkpoint_path", "/data/run/checkpoint_42",
        "--seeds", "42,123",
    ]


@pytest.mark.parametrize(
    "evaluation",
    [
        {},
        {"checkpoint_path": "relative", "seeds": [42]},
        {"checkpoint_path": "/data/checkpoint", "seeds": []},
        {"checkpoint_path": "/data/checkpoint", "seeds": [True]},
    ],
)
def test_elf_evaluation_rejects_incomplete_identity(evaluation):
    with pytest.raises(ValueError):
        ElfProjectAdapter().validate_run({
            "run_id": "eval-a",
            "config": "config.yml",
            "operation": "evaluate",
            "evaluation": evaluation,
        })
