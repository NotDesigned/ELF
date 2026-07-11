"""Artifact identities and reusable staging verification markers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactIdentity:
    digest: str
    size: int
    mtime_ns: int

    @classmethod
    def from_file(cls, path: Path) -> "ArtifactIdentity":
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return cls(f"sha256:{digest.hexdigest()}", stat.st_size, stat.st_mtime_ns)

    def matches_marker(self, marker: Path) -> bool:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return payload == {
            "digest": self.digest, "size": self.size, "mtime_ns": self.mtime_ns
        }

    def write_marker(self, marker: Path) -> None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"digest": self.digest, "size": self.size, "mtime_ns": self.mtime_ns}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
