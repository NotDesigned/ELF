"""ELF-specific configuration, runtime, asset, metric, and summary behavior."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Mapping

from ..assets import cache_path, plan_assets
from ..overrides import operational_overrides
from ..summary import summarize_run

from experiment_control.project import AssetProbe, AssetRequirement, SourceBundle


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs.config import Config, apply_config_overrides, load_config_from_yaml  # noqa: E402


def parse_training_metric_line(line: str) -> dict[str, Any] | None:
    """Parse human-readable metric records emitted by ELF training or eval."""
    eval_patterns = (
        (r"\bgPPL:\s*([-+0-9.eE]+)", "g_ppl"),
        (r"\boracle_plan_ppl:\s*([-+0-9.eE]+)", "oracle_plan_ppl"),
        (r"\bshuffled_plan_ppl:\s*([-+0-9.eE]+)", "shuffled_plan_ppl"),
        (r"\bToken reconstruction PPL:\s*([-+0-9.eE]+)", "token_recon_ppl"),
    )
    for pattern, key in eval_patterns:
        eval_match = re.search(pattern, line)
        if eval_match:
            return {key: float(eval_match.group(1))}

    match = re.search(r"Step\s+(\d+):\s+(.*)$", line)
    if not match:
        return None
    record: dict[str, Any] = {"step": int(match.group(1))}
    key_map = {
        "loss": "train_loss", "l2": "train_l2_loss", "ce": "train_ce_loss",
        "plan": "train_plan_loss", "plan_aux": "train_plan_aux_loss",
        "emb_var": "train_plan_emb_batch_var", "pred_var": "train_plan_pred_batch_var",
        "emb_norm": "train_plan_emb_norm", "pred_norm": "train_plan_pred_norm",
        "p_phase": "train_plan_phase_fraction", "t_phase": "train_token_phase_fraction",
        "lr": "lr", "steps/sec": "steps_per_sec",
    }
    for key, value in re.findall(r"([A-Za-z0-9_/]+)=([-+0-9.eE]+)", match.group(2)):
        if key in key_map:
            record[key_map[key]] = float(value)
    return record


def parse_checkpoint_line(line: str) -> dict[str, Any] | None:
    match = re.search(r"Checkpoint committed to (\S*/checkpoint_(\d+)) \((\d+) bytes\)", line)
    if not match:
        return None
    return {"path": match.group(1), "step": int(match.group(2)), "bytes": int(match.group(3))}


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    annotations = getattr(type(value), "__annotations__", {})
    if annotations:
        return {name: _plain(getattr(value, name)) for name in annotations}
    raise TypeError(f"cannot serialize config value of type {type(value).__name__}")


class ElfProjectAdapter:
    """The sole owner of assumptions about the ELF repository and training API."""

    name = "elf"
    safe_env_keys = frozenset({
        "BATCH_SIZE", "DATA_ROOT", "ELF_EVAL_DATA_ID", "GLOBAL_BATCH_SIZE", "HF_DATASETS_OFFLINE",
        "HF_DATASETS_CACHE", "HF_HOME", "HF_HUB_OFFLINE", "HF_REPO_ID", "LOG_FREQ",
        "MAX_INFRA_RETRIES", "NUM_WORKERS", "PROJECT_DATA_ROOT", "RESUME",
        "REQUIRE_OFFLINE_CACHE", "TRANSFORMERS_OFFLINE", "USE_COMPILE", "USE_WANDB",
        "WANDB_RESUME",
    })

    def validate_run(self, run: dict[str, Any]) -> None:
        if not str(run["config"]).endswith((".yml", ".yaml")):
            raise ValueError(f"run {run['run_id']} ELF config must be YAML")
        operation = str(run.get("operation", "train"))
        if operation not in {"train", "evaluate"}:
            raise ValueError(
                f"run {run['run_id']} ELF operation must be 'train' or 'evaluate'"
            )
        if operation == "evaluate":
            evaluation = run.get("evaluation")
            if not isinstance(evaluation, dict):
                raise ValueError(
                    f"run {run['run_id']} evaluation must be a mapping"
                )
            checkpoint_path = evaluation.get("checkpoint_path")
            if not isinstance(checkpoint_path, str) or not checkpoint_path.startswith("/"):
                raise ValueError(
                    f"run {run['run_id']} evaluation.checkpoint_path must be absolute"
                )
            seeds = evaluation.get("seeds")
            if (
                not isinstance(seeds, list)
                or not seeds
                or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            ):
                raise ValueError(
                    f"run {run['run_id']} evaluation.seeds must be a non-empty integer list"
                )

    def operational_overrides(
        self, env: Mapping[str, str], output_dir: str
    ) -> list[str]:
        return operational_overrides(env, output_dir)

    def resolve_config(self, config_path: str, overrides: list[str]) -> dict[str, Any]:
        config = apply_config_overrides(load_config_from_yaml(config_path), overrides)
        return {name: _plain(getattr(config, name)) for name in Config.__annotations__}

    def environment(self, campaign: dict[str, Any], run: dict[str, Any]) -> dict[str, str]:
        project_root = str(run["storage"]["project_data_root"])
        return {
            "CHECKPOINT_ROOT": f"{project_root}/checkpoints",
            "ELF_B_OWT_CHECKPOINT": (
                f"{project_root}/checkpoints/ELF-B-owt-torch/checkpoint_95085"
            ),
            "SAVE_DIR": f"{project_root}/saved_models",
            "WANDB_CACHE_DIR": f"{project_root}/wandb_cache",
            "WANDB_DIR": f"{project_root}/wandb",
        }

    def command(self, run: dict[str, Any]) -> list[str]:
        operation = str(run.get("operation", "train"))
        command = [
            "env", f"ELF_RUN_MODE={'eval' if operation == 'evaluate' else 'train'}",
            "bash", "scripts/cloud_train.sh", str(run["config"]),
        ]
        for override in run.get("config_overrides", []):
            command.extend(["--config_override", str(override)])
        if operation == "evaluate":
            evaluation = run["evaluation"]
            command.extend([
                "--checkpoint_path", str(evaluation["checkpoint_path"]),
                "--seeds", ",".join(str(seed) for seed in evaluation["seeds"]),
            ])
        return command

    def plan_assets(
        self, config_path: str, overrides: list[str]
    ) -> list[AssetRequirement]:
        return plan_assets(config_path, overrides)

    def asset_probes(
        self, requirements: list[AssetRequirement], environment: Mapping[str, str]
    ) -> list[AssetProbe]:
        hf_home = Path(environment["HF_HOME"])
        datasets_cache = Path(environment["HF_DATASETS_CACHE"])
        return [
            AssetProbe(
                requirement=item,
                path=str(cache_path(item, hf_home, datasets_cache)),
                file=item.kind == "file",
            )
            for item in requirements
        ]

    def parse_metric(self, line: str) -> dict[str, Any] | None:
        return parse_training_metric_line(line)

    def parse_checkpoint(self, line: str) -> dict[str, Any] | None:
        return parse_checkpoint_line(line)

    def summarize(self, run_dir: Path) -> dict[str, Any]:
        return summarize_run(run_dir)

    def source_bundle(self, repo_root: Path) -> SourceBundle:
        return SourceBundle(
            root=repo_root,
            excludes=(
                # Keep the immutable runtime snapshot aligned with the Docker
                # build context and exclude common local credential stores.
                ".git/", ".gitignore", ".dockerignore",
                ".env", ".env.*", ".netrc", ".npmrc", ".pypirc", ".ssh/", ".aws/",
                ".venv/", "venv/", "env/", ".claude/", ".codex/",
                ".cache/", "hf_cache/", "huggingface/",
                "__pycache__/", "*.py[cod]", "*$py.class",
                ".pytest_cache/", ".mypy_cache/", ".ruff_cache/",
                ".ipynb_checkpoints/", "*.egg-info/", "build/", "dist/",
                "outputs/", "output_dir/", "saved_models/", "checkpoints/",
                "wandb/", "runs/", "data/", "papers/",
                "*.log", "*.tar.gz", "*.pt", "*.pth", "*.ckpt", "*.safetensors",
                ".DS_Store", "Thumbs.db",
            ),
            container_path="/app",
            identity_command=("bash", "scripts/source_identity.sh", "--runtime"),
            required_paths=(
                "scripts/cloud_train.sh",
                "src/train.py",
                "src/eval.py",
            ),
        )
