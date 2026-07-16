#!/usr/bin/env python3
"""Confirm that a registry token grants push without printing the token."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys


def _decode_payload(token: str) -> dict[str, object] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    encoded = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded))
    except (ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def main() -> int:
    if len(sys.argv) != 2:
        print("registry push-scope check requires one repository", file=sys.stderr)
        return 2
    repository = sys.argv[1]
    if (
        not repository
        or "://" in repository
        or "@" in repository
        or any(char.isspace() for char in repository)
        or "/" not in repository
    ):
        print("registry push-scope check requires host/path", file=sys.stderr)
        return 2
    repository_name = repository.split("/", 1)[1]
    crane = os.environ.get("EXPERIMENTCTL_CRANE_BIN", "crane")
    try:
        result = subprocess.run(
            [crane, "auth", "token", "--push", repository],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        print("registry push-scope token request was inconclusive", file=sys.stderr)
        return 3
    if result.returncode != 0:
        print("registry push-scope token request was rejected", file=sys.stderr)
        return 3
    payload = _decode_payload(result.stdout.strip())
    if payload is None:
        print("registry push-scope token is not an inspectable JWT", file=sys.stderr)
        return 3
    for access in payload.get("access", []):
        if not isinstance(access, dict):
            continue
        actions = access.get("actions", [])
        if (
            access.get("type") == "repository"
            and access.get("name") == repository_name
            and isinstance(actions, list)
            and "push" in actions
        ):
            print("registry-push-scope=granted")
            return 0
    print("registry push scope was not granted", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
