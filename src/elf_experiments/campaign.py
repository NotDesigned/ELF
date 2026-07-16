"""Resolve reusable experiment campaign profiles, defaults, and matrices.

This module is deliberately independent from scheduler adapters.  It turns a
compact authoring document into the fully expanded ``runs`` representation
consumed by the experiment controller.
"""

from __future__ import annotations

import copy
import itertools
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


_TOKEN_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.]*)\}")
_EXACT_TOKEN_RE = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_.]*)\}$")
_INSTANCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive, non-mutating merge with ``override`` taking priority.

    Mapping values are merged recursively.  Lists and scalar values are
    replaced as complete values, which keeps ordered config overrides
    predictable and makes it possible for a run to clear a default list.
    """
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _merge_campaign_layer(
    base: Mapping[str, Any], override: Mapping[str, Any], *, layer: str
) -> dict[str, Any]:
    """Merge one authoring layer and explicitly append config overrides.

    Lists normally replace earlier lists.  ``config_overrides_append`` is the
    opt-in exception for the common campaign pattern where defaults freeze
    shared runtime settings and a profile or run adds only its varying fields.
    Keeping this explicit preserves the existing ability to replace or clear a
    default ``config_overrides`` list.
    """
    authored = copy.deepcopy(dict(override))
    appended = authored.pop("config_overrides_append", None)
    if appended is None:
        return deep_merge(base, authored)
    if "config_overrides" in authored:
        raise ValueError(
            f"{layer} cannot define both config_overrides and "
            "config_overrides_append"
        )
    if not isinstance(appended, list) or not all(
        isinstance(item, str) for item in appended
    ):
        raise ValueError(f"{layer} config_overrides_append must be a list of strings")
    merged = deep_merge(base, authored)
    inherited = merged.get("config_overrides", [])
    if not isinstance(inherited, list) or not all(
        isinstance(item, str) for item in inherited
    ):
        raise ValueError(f"{layer} inherited config_overrides must be a list of strings")
    merged["config_overrides"] = [*inherited, *appended]
    return merged


def _lookup(context: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    value: Any = context
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return False, None
        value = value[component]
    return True, value


def _render(value: Any, context: Mapping[str, Any]) -> Any:
    """Render known matrix tokens while preserving controller placeholders."""
    if isinstance(value, str):
        exact = _EXACT_TOKEN_RE.fullmatch(value)
        if exact:
            found, replacement = _lookup(context, exact.group(1))
            if found:
                return copy.deepcopy(replacement)

        def replace(match: re.Match[str]) -> str:
            found, replacement = _lookup(context, match.group(1))
            return str(replacement) if found else match.group(0)

        return _TOKEN_RE.sub(replace, value)
    if isinstance(value, list):
        return [_render(item, context) for item in value]
    if isinstance(value, Mapping):
        return {key: _render(item, context) for key, item in value.items()}
    return copy.deepcopy(value)


def _expand_matrix(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    matrix = spec.get("matrix")
    template = spec.get("template")
    if not isinstance(matrix, Mapping) or not matrix:
        raise ValueError("run matrix must be a non-empty mapping")
    if not isinstance(template, Mapping) or not template:
        raise ValueError("matrix run template must be a non-empty mapping")
    unexpected = set(spec) - {"matrix", "template"}
    if unexpected:
        raise ValueError(f"matrix run has unsupported sibling keys: {sorted(unexpected)}")

    axes: list[tuple[str, list[Any]]] = []
    for name, values in matrix.items():
        if not isinstance(name, str) or not name:
            raise ValueError("matrix axis names must be non-empty strings")
        if not isinstance(values, list) or not values:
            raise ValueError(f"matrix axis {name!r} must be a non-empty list")
        axes.append((name, values))

    expanded: list[dict[str, Any]] = []
    for combination in itertools.product(*(values for _, values in axes)):
        context = dict(zip((name for name, _ in axes), combination, strict=True))
        run = _render(template, context)
        if not isinstance(run, dict):  # pragma: no cover - guarded by template validation
            raise ValueError("rendered matrix template must be a mapping")
        expanded.append(run)
    return expanded


def resolve_campaign(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Expand authoring helpers into backend-complete run mappings.

    Merge order is campaign ``defaults``, each selected profile in order, then
    the run itself.  A run selects one profile with ``profile: name`` or an
    ordered list with ``profile: [base, accelerator]``.  Matrix run entries use
    ``matrix`` plus ``template``; known ``{axis}``/``{axis.field}`` tokens are
    expanded and later controller placeholders such as ``{source_id}`` remain
    untouched.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("campaign must be a mapping")
    defaults = payload.get("defaults", {})
    profiles = payload.get("profiles", {})
    runs = payload.get("runs")
    if not isinstance(defaults, Mapping):
        raise ValueError("campaign defaults must be a mapping")
    if not isinstance(profiles, Mapping):
        raise ValueError("campaign profiles must be a mapping")
    if not isinstance(runs, list):
        raise ValueError("campaign runs must be a list")
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name:
            raise ValueError("profile names must be non-empty strings")
        if not isinstance(profile, Mapping):
            raise ValueError(f"profile {name!r} must be a mapping")

    authored_runs: list[dict[str, Any]] = []
    for entry in runs:
        if not isinstance(entry, Mapping):
            raise ValueError("each campaign run must be a mapping")
        if "matrix" in entry or "template" in entry:
            authored_runs.extend(_expand_matrix(entry))
        else:
            authored_runs.append(copy.deepcopy(dict(entry)))

    resolved_runs: list[dict[str, Any]] = []
    for authored in authored_runs:
        selected = authored.pop("profile", [])
        if isinstance(selected, str):
            selected = [selected]
        if not isinstance(selected, list) or not all(isinstance(name, str) for name in selected):
            raise ValueError("run profile must be a string or list of strings")
        resolved = copy.deepcopy(dict(defaults))
        for name in selected:
            if name not in profiles:
                raise ValueError(f"run references unknown profile: {name}")
            resolved = _merge_campaign_layer(
                resolved, profiles[name], layer=f"profile {name!r}"
            )
        resolved_runs.append(
            _merge_campaign_layer(
                resolved,
                authored,
                layer=f"run {authored.get('run_id', '<unresolved>')!r}",
            )
        )

    result = copy.deepcopy(dict(payload))
    result.pop("defaults", None)
    result.pop("profiles", None)
    result["runs"] = resolved_runs
    return result


def load_and_resolve_campaign(path: Path) -> dict[str, Any]:
    """Load one YAML campaign and resolve its authoring helpers."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"campaign must be a mapping: {path}")
    return resolve_campaign(payload)


def instantiate_campaign_template(
    payload: Mapping[str, Any], instance: str
) -> dict[str, Any]:
    """Render one explicit fresh-instance token without resolving authoring helpers.

    Templates keep ``{run_id}`` and ``{source_id}`` for the normal campaign
    resolver/controller. Only ``{instance}`` is consumed here, making the
    generated campaign reviewable before any controller or backend mutation.
    """
    if not _INSTANCE_RE.fullmatch(instance):
        raise ValueError(
            "instance must use 1-64 letters, digits, '.', '_' or '-'"
        )
    rendered = _render(payload, {"instance": instance})
    if not isinstance(rendered, dict):
        raise ValueError("campaign template must render to a mapping")
    if "{instance}" in yaml.safe_dump(rendered, sort_keys=False):
        raise ValueError("campaign template contains an unresolved {instance} token")
    rendered["instance"] = instance
    return rendered
