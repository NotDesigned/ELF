"""Installed scientific-project adapters."""

from .base import AssetProbe, AssetRequirement, ProjectAdapter, ProjectRegistry, SourceBundle


def build_project_registry() -> ProjectRegistry:
    from .elf import ElfProjectAdapter

    return ProjectRegistry(ElfProjectAdapter())


__all__ = [
    "AssetProbe", "AssetRequirement", "ProjectAdapter", "ProjectRegistry", "SourceBundle",
    "build_project_registry",
]
