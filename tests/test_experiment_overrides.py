from experiment_overrides import operational_overrides
from experiment_projects.elf import ElfProjectAdapter


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
