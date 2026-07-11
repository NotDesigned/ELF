#!/usr/bin/env python
"""Create one reviewable campaign file with fresh run identities."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from elf_experiments.campaign import instantiate_campaign_template  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--local-root", type=Path,
        help="override controller metadata root in the generated definition",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template = args.template.resolve()
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    campaign = instantiate_campaign_template(payload, args.instance)
    if args.local_root is not None:
        local_root = args.local_root
        if not local_root.is_absolute():
            local_root = (Path.cwd() / local_root).resolve()
        campaign["local_root"] = str(local_root)
    try:
        generated_from = str(template.relative_to(REPO_ROOT))
    except ValueError:
        generated_from = str(template)
    campaign["generated_from"] = generated_from
    output = args.output
    if output is None:
        output = (
            REPO_ROOT
            / "outputs/experiment_campaigns/definitions"
            / f"{campaign['campaign']}.yml"
        )
    elif not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(campaign, handle, sort_keys=False, allow_unicode=True)
        handle.flush()
        os.fsync(handle.fileno())
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
