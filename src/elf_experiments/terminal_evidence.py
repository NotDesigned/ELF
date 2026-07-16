"""Emit one compact, identity-bound terminal summary for remote collectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .summary import summarize_run


PREFIX = "EXPERIMENT_EVIDENCE_JSON="


def encode_terminal_evidence(payload: Mapping[str, Any]) -> str:
    """Encode one bounded-line payload without allowing non-finite JSON values."""
    return PREFIX + json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    print(encode_terminal_evidence(summarize_run(args.run_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
