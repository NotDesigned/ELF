#!/usr/bin/env python3
"""Check for a Docker credential reference without reading or printing secrets."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1] or "/" in sys.argv[1]:
        print("registry credential check requires one host", file=sys.stderr)
        return 2
    host = sys.argv[1]
    config_root = Path(os.environ.get("DOCKER_CONFIG", Path.home() / ".docker"))
    config_path = config_root / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("Docker credential configuration is unavailable", file=sys.stderr)
        return 1
    helpers = payload.get("credHelpers", {})
    auths = payload.get("auths", {})
    aliases = {host, f"https://{host}", f"https://{host}/v1/"}
    present = bool(payload.get("credsStore")) or any(key in helpers for key in aliases)
    present = present or any(key in auths for key in aliases)
    present = present or any(key.rstrip("/").removeprefix("https://") == host for key in auths)
    if not present:
        print("Docker credential reference for registry is missing", file=sys.stderr)
        return 1
    print("docker-credential-reference=present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
