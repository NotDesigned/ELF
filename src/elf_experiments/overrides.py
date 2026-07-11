#!/usr/bin/env python
"""Single source of truth for launcher environment-to-config overrides."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping


def operational_overrides(env: Mapping[str, str], output_dir: str) -> list[str]:
    """Return config overrides in the exact order applied by the launcher."""
    overrides = [f"output_dir={output_dir}"]
    direct = (
        ("USE_WANDB", "use_wandb"), ("WANDB_PROJECT", "wandb_project"),
        ("WANDB_ENTITY", "wandb_entity"),
    )
    for env_key, config_key in direct:
        if env.get(env_key, ""):
            overrides.append(f"{config_key}={env[env_key]}")
    run_id = env.get("RUN_ID", "")
    overrides.extend(
        [
            f"wandb_run_name={env.get('WANDB_RUN_NAME') or run_id}",
            f"wandb_run_id={env.get('WANDB_RUN_ID') or run_id}",
            f"wandb_resume={env.get('WANDB_RESUME') or 'allow'}",
        ]
    )
    if env.get("GLOBAL_BATCH_SIZE", ""):
        overrides.append(f"global_batch_size={env['GLOBAL_BATCH_SIZE']}")
    if env.get("BATCH_SIZE", ""):
        overrides.extend(["global_batch_size=null", f"batch_size={env['BATCH_SIZE']}"])
    for env_key, config_key in (
        ("NUM_WORKERS", "num_workers"), ("LOG_FREQ", "log_freq"),
        ("USE_COMPILE", "use_compile"), ("WARM_START", "warm_start"),
        ("WARM_START_USE_EMA", "warm_start_use_ema"), ("RESUME", "resume"),
        ("HF_REPO_ID", "hf_repo_id"),
    ):
        if env.get(env_key, ""):
            overrides.append(f"{config_key}={env[env_key]}")
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--format", choices=("json", "lines"), default="json")
    args = parser.parse_args()
    overrides = operational_overrides(os.environ, args.output_dir)
    if args.format == "lines":
        print("\n".join(overrides))
    else:
        print(json.dumps(overrides, indent=2))


if __name__ == "__main__":
    main()
