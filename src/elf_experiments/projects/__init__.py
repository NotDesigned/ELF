"""ELF's installed scientific-project adapters."""

from experiment_control.project import (
    AssetProbe, AssetRequirement, ProjectAdapter, ProjectRegistry, SourceBundle,
)


def build_project_registry() -> ProjectRegistry:
    from .elf import ElfProjectAdapter

    return ProjectRegistry(ElfProjectAdapter())


__all__ = [
    "AssetProbe", "AssetRequirement", "ProjectAdapter", "ProjectRegistry", "SourceBundle",
    "build_project_registry",
]
