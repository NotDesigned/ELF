#!/usr/bin/env python
"""Plan and verify config-dependent offline assets before GPU allocation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
PACKAGE_SRC = REPO_ROOT / "packages" / "experiment-control" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from configs.config import apply_config_overrides, load_config_from_yaml  # noqa: E402
from experiment_control.project import AssetRequirement  # noqa: E402

def plan_assets(config_path: str, overrides: list[str]) -> list[AssetRequirement]:
    """Load the resolved config once and return its complete offline asset set."""
    config = apply_config_overrides(load_config_from_yaml(config_path), overrides)
    requirements = [
        AssetRequirement("model", config.encoder_model_name, "token encoder"),
        AssetRequirement("dataset", config.data_path, "training dataset"),
    ]
    if config.use_sentence_plan and config.sentence_encoder_type == "sentence_t5":
        requirements.append(AssetRequirement("model", config.sentence_t5_model_name, "frozen sentence plan"))
    if config.online_eval:
        requirements.append(AssetRequirement("model", config.eval_ppl_model, "generation perplexity evaluation"))
    if config.warm_start:
        requirements.append(AssetRequirement("file", config.warm_start, "warm-start checkpoint"))
    unique: dict[tuple[str, str], AssetRequirement] = {}
    for item in requirements:
        unique[(item.kind, item.identity)] = item
    return list(unique.values())


def cache_path(requirement: AssetRequirement, hf_home: Path, datasets_cache: Path) -> Path:
    identity = requirement.identity
    if identity.startswith("/") or identity.startswith("./"):
        return Path(identity)
    if requirement.kind == "model":
        return hf_home / "hub" / f"models--{identity.replace('/', '--')}"
    if requirement.kind == "dataset":
        return datasets_cache / identity.replace("/", "___")
    return Path(identity)


def verify_assets(requirements: list[AssetRequirement], hf_home: Path, datasets_cache: Path) -> list[dict[str, str]]:
    missing = []
    for requirement in requirements:
        path = cache_path(requirement, hf_home, datasets_cache)
        valid = path.is_file() if requirement.kind == "file" else path.is_dir()
        if not valid:
            missing.append({**asdict(requirement), "path": str(path)})
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "verify"))
    parser.add_argument("config")
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--hf-home", default="")
    parser.add_argument("--datasets-cache", default="")
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    args = parser.parse_args(argv)
    requirements = plan_assets(args.config, args.config_override)
    if args.command == "verify":
        if not args.hf_home or not args.datasets_cache:
            parser.error("verify requires --hf-home and --datasets-cache")
        missing = verify_assets(requirements, Path(args.hf_home), Path(args.datasets_cache))
        print(json.dumps({"requirements": [asdict(item) for item in requirements], "missing": missing}, indent=2))
        return 1 if missing else 0
    if args.format == "tsv":
        for item in requirements:
            print(f"{item.kind}\t{item.identity}\t{item.reason}")
    else:
        print(json.dumps([asdict(item) for item in requirements], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
