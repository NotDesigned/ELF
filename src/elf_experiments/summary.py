#!/usr/bin/env python
"""Summarize durable ELF run directories into machine-readable campaign rows."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO

import yaml

from experiment_control.checkpoints import discover_latest_completed_checkpoint


TRAIN_KEYS = (
    "epoch",
    "step",
    "optimizer_step",
    "train_loss",
    "train_l2_loss",
    "train_ce_loss",
    "train_plan_loss",
    "train_plan_aux_loss",
    "train_plan_emb_batch_var",
    "train_plan_emb_norm",
    "train_plan_pred_batch_var",
    "train_plan_pred_norm",
    "lr",
    "steps_per_sec",
)
EVAL_KEYS = (
    "g_ppl",
    "oracle_plan_ppl",
    "shuffled_plan_ppl",
    "token_recon_ppl",
)
EVAL_PRIMARY_BY_MODE = {
    "clean_token_reconstruction": "token_recon_ppl",
    "generation_refine_decode": "g_ppl",
    "oracle_plan_generation": "oracle_plan_ppl",
    "shuffled_plan_generation": "shuffled_plan_ppl",
}
SAMPLING_MODES = {
    "generation_refine_decode",
    "oracle_plan_generation",
    "shuffled_plan_generation",
}
FAMILY_DIMENSION_KEYS = (
    "sampling_method", "num_sampling_steps", "cfg", "self_cond_cfg_scale",
    "time_schedule", "time_warp_gamma",
)
_SAMPLE_FILE_RE = re.compile(r"_(?P<epoch>\d+)_(?P<step>\d+)\.jsonl$")


def finite_metric_value(value: Any) -> Any:
    """Convert non-finite numeric evidence to JSON-safe missing evidence."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
SCIENTIFIC_CONFIG_KEYS = (
    "seed",
    "max_length",
    "global_batch_size",
    "use_sentence_plan",
    "sentence_encoder_type",
    "sentence_encoder_grad",
    "plan_aux_passes",
    "plan_aux_token_context",
)

# ``summarize_run`` owns only scientific and evaluation projections when it is
# used to refresh an existing exact-Attempt collection.  Lifecycle,
# checkpoint, identity, and collection provenance remain owned by the original
# backend observation and must not be replaced by a local scientific rebuild.
LOCAL_SCIENCE_EVIDENCE_KEYS = frozenset({
    *TRAIN_KEYS,
    *EVAL_KEYS,
    *SCIENTIFIC_CONFIG_KEYS,
    "metric_evidence",
    "evaluation_metrics_by_variant",
    "evaluation_variants",
    "evaluation_family_state",
    "canonical_evaluation_family_id",
    "artifacts",
    "generation_nonempty_fraction",
    "generation_mean_entropy",
    "plan_ppl_gap",
    "nonfinite_metrics",
    "warnings",
    "evidence_conflicts",
})

LOCAL_EVIDENCE_IDENTITY_KEYS = (
    "project",
    "run_id",
    "attempt_id",
    "source_id",
    "image_id",
)


def merge_local_scientific_evidence(
    previous: dict[str, Any] | None,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Replace only summary-owned science on one exact Attempt.

    A local summary is not a scheduler or process observation.  For an
    existing collection, start from the reviewed preimage, remove every
    summary-owned key (including stale conflicts and flat evaluation values),
    and then install only the newly computed summary-owned values.  All other
    fields retain their exact previous values.

    A missing previous collection has no operational evidence to preserve, so
    the complete summary remains the initial collection representation.
    """
    if previous is None:
        return deepcopy(summary)

    for key in LOCAL_EVIDENCE_IDENTITY_KEYS:
        old_value = previous.get(key)
        new_value = summary.get(key)
        if old_value is not None and new_value is not None and old_value != new_value:
            raise ValueError(
                f"local scientific summary {key} conflicts with reviewed collection"
            )

    result = deepcopy(previous)
    for key in LOCAL_SCIENCE_EVIDENCE_KEYS:
        result.pop(key, None)
    for key in LOCAL_SCIENCE_EVIDENCE_KEYS:
        if key in summary:
            result[key] = deepcopy(summary[key])
    return result


def load_mapping(path: Path) -> dict[str, Any]:
    """Load one JSON or YAML mapping and reject missing or non-mapping content.

    Args:
        path: Manifest or status file to read.

    Returns:
        The decoded top-level mapping. An empty YAML document becomes an empty
        mapping so callers can report missing fields consistently.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the decoded top-level value is not a mapping.
        json.JSONDecodeError: If a JSON file is malformed.
        yaml.YAMLError: If a YAML file is malformed.
    """
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}, got {type(payload).__name__}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read structured JSONL records with precise corruption diagnostics.

    Blank lines are ignored. Every non-blank line must decode to a JSON object;
    silently skipping a malformed metric would make campaign comparisons
    irreproducible.

    Args:
        path: JSONL file containing one object per line.

    Returns:
        Records in their original file order.

    Raises:
        ValueError: If a line is invalid JSON or does not contain an object.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected an object in {path}:{line_number}")
            records.append(record)
    return records


def discover_run_dirs(roots: Iterable[Path]) -> list[Path]:
    """Discover unique durable run directories below one or more roots.

    A run directory is identified by ``manifest.yaml``. Passing a run directory
    directly and passing its parent are both supported. Results are resolved,
    deduplicated, and sorted for deterministic reports.

    Args:
        roots: Filesystem roots to inspect recursively.

    Returns:
        Sorted absolute directories containing a manifest.
    """
    discovered: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if (root / "manifest.yaml").is_file():
            discovered.add(root)
        if root.is_dir():
            discovered.update(path.parent.resolve() for path in root.rglob("manifest.yaml"))
    return sorted(discovered)


def latest_record(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select the most advanced metric record by step and then file order.

    Args:
        records: Metric objects, normally from one training JSONL file.

    Returns:
        A shallow copy of the record with the largest numeric ``step``. When
        steps tie, the later record wins. An empty input returns an empty map.
    """
    winner: dict[str, Any] = {}
    winner_key = (float("-inf"), -1)
    for index, record in enumerate(records):
        raw_step = record.get("step", -1)
        try:
            step = float(raw_step)
        except (TypeError, ValueError):
            step = -1.0
        key = (step, index)
        if key >= winner_key:
            winner = dict(record)
            winner_key = key
    return winner


def _checkpoint_value(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _sampling_dimensions(record: dict[str, Any]) -> dict[str, Any] | None:
    raw = record.get("sampling_config")
    if not isinstance(raw, dict):
        raw = record.get("variant_dimensions")
    if not isinstance(raw, dict):
        raw = record.get("sampling_dimensions")
    if not isinstance(raw, dict) or set(raw) != set(FAMILY_DIMENSION_KEYS):
        return None
    dimensions = dict(raw)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (str, int, float))
        or isinstance(value, float) and not math.isfinite(value)
        for value in dimensions.values()
    ):
        return None
    return dimensions


def _family_id(dimensions: dict[str, Any]) -> str:
    encoded = json.dumps(
        dimensions, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _valid_identity_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _exact_observation_errors(observation: dict[str, Any]) -> list[str]:
    """Return fields that make one purported exact observation non-exact."""
    errors: list[str] = []
    for field in ("project", "run_id", "attempt_id", "variant_id", "source"):
        if not _valid_identity_text(observation.get(field)):
            errors.append(field)
    for field in ("epoch", "step"):
        value = observation.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(field)
    family_id = observation.get("family_id")
    binding = observation.get("binding")
    binding = binding if isinstance(binding, dict) else {}
    dimensions = binding.get("sampling_dimensions")
    if (
        not isinstance(family_id, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", family_id) is None
        or not isinstance(dimensions, dict)
        or _sampling_dimensions({"sampling_dimensions": dimensions}) != dimensions
        or _family_id(dimensions) != family_id
    ):
        errors.append("family_id")
    source = observation.get("source")
    if isinstance(source, str):
        source_path = Path(source)
        if source_path.is_absolute() or ".." in source_path.parts:
            errors.append("source")
    return sorted(set(errors))


def _record_binding(record: dict[str, Any]) -> dict[str, Any]:
    mode = record.get("mode")
    if mode == "clean_token_reconstruction":
        carries_sampling_dimensions = any(
            key in record for key in (
                "sampling_config", "variant_dimensions", "sampling_dimensions",
            )
        )
        return {
            "status": "RESOLVED",
            "scope": "FAMILY_INDEPENDENT_RECONSTRUCTION",
            "mode": mode,
            "family_id": None,
        } if not carries_sampling_dimensions else {
            "status": "CONFLICTING",
            "mode": mode,
            "reason": "clean reconstruction carries sampling dimensions",
        }
    dimensions = _sampling_dimensions(record)
    if mode in SAMPLING_MODES and dimensions is not None:
        return {
            "status": "RESOLVED", "scope": "SAMPLING_FAMILY",
            "mode": mode, "family_id": _family_id(dimensions),
            "sampling_dimensions": dimensions,
        }
    return {
        "status": "UNRESOLVED",
        "mode": mode if isinstance(mode, str) else None,
        "family_id": None,
        "reason": (
            "sampling family dimensions are missing"
            if mode in SAMPLING_MODES else "evaluation mode is missing or unrecognized"
        ),
    }


def _metric_candidates(record: dict[str, Any]) -> list[tuple[str, Any]]:
    candidates = [
        (key, finite_metric_value(record[key])) for key in EVAL_KEYS if key in record
    ]
    mode = record.get("mode")
    primary = EVAL_PRIMARY_BY_MODE.get(mode)
    if primary is not None and "ppl" in record:
        candidates.append((primary, finite_metric_value(record["ppl"])))
    if mode == "generation_refine_decode":
        for key in ("generation_mean_entropy", "mean_entropy"):
            if key in record:
                candidates.append((
                    "generation_mean_entropy", finite_metric_value(record[key]),
                ))
    return candidates


def _variant_id(run_dir: Path, path: Path) -> str:
    """Use the durable metrics parent directory, never parse variant labels."""
    del run_dir
    return path.parent.name


def _declared_family_id(manifest: dict[str, Any]) -> str | None:
    candidates = [manifest]
    for key in ("evaluation", "evaluation_contract", "research_contract"):
        value = manifest.get(key)
        if isinstance(value, dict):
            candidates.append(value)
            nested = value.get("evaluation")
            if isinstance(nested, dict):
                candidates.append(nested)
    for value in candidates:
        family_id = value.get("canonical_family_id")
        if isinstance(family_id, str) and family_id:
            return family_id
        dimensions = value.get("canonical_family_dimensions")
        if isinstance(dimensions, dict):
            normalized = _sampling_dimensions({"sampling_dimensions": dimensions})
            if normalized is not None:
                return _family_id(normalized)
    return None


def collect_eval_evidence(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[dict[str, Any]]]:
    """Collect exact variant-bound evidence and publish only coherent flat science."""
    manifest = load_mapping(run_dir / "manifest.yaml")
    status_path = run_dir / "status.json"
    status = load_mapping(status_path) if status_path.is_file() else {}
    project = manifest.get("project")
    run_id = manifest.get("run_id")
    attempt_id = status.get("attempt_id")
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    conflicts: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("metrics.jsonl")):
        source = str(path.relative_to(run_dir))
        variant_id = _variant_id(run_dir, path)
        try:
            source_mtime = path.stat().st_mtime
        except OSError:
            source_mtime = None
        for record in read_jsonl(path):
            epoch = _checkpoint_value(record.get("epoch"))
            step = _checkpoint_value(record.get("step"))
            binding = _record_binding(record)
            observed_at = record.get("observed_at")
            if not isinstance(observed_at, (int, float)) \
                    or isinstance(observed_at, bool) \
                    or not math.isfinite(float(observed_at)):
                observed_at = source_mtime
            for metric, value in _metric_candidates(record):
                exact_binding = {
                    "project": project, "run_id": run_id,
                    "attempt_id": attempt_id, "epoch": epoch, "step": step,
                    "variant_id": variant_id,
                    "family_id": binding.get("family_id"), "metric": metric,
                }
                candidates.append({
                    **exact_binding,
                    "mode": binding.get("mode"), "value": value,
                    "source": source, "observed_at": observed_at,
                    "binding": dict(binding),
                })

    family_dimensions: dict[str, dict[str, Any]] = {}
    for item in candidates:
        binding = item["binding"]
        family_id = binding.get("family_id")
        dimensions = binding.get("sampling_dimensions")
        if (
            binding.get("scope") == "SAMPLING_FAMILY"
            and isinstance(family_id, str)
            and isinstance(dimensions, dict)
        ):
            family_dimensions[family_id] = dimensions

    expanded: list[dict[str, Any]] = []
    for item in candidates:
        binding = item["binding"]
        if binding.get("scope") != "FAMILY_INDEPENDENT_RECONSTRUCTION":
            expanded.append(item)
            continue
        if not family_dimensions:
            expanded.append(item)
            continue
        # Reconstruction is explicitly family-independent, so materialize one
        # exact binding per producer-authored family instead of publishing a
        # family-less observation or inferring a family from a label.
        for family_id, dimensions in sorted(family_dimensions.items()):
            expanded.append({
                **item,
                "family_id": family_id,
                "binding": {
                    **binding,
                    "family_id": family_id,
                    "sampling_dimensions": dimensions,
                },
            })

    observations: list[dict[str, Any]] = []
    observation_fingerprints: set[str] = set()
    diagnostics: list[dict[str, Any]] = []
    for item in expanded:
        errors = _exact_observation_errors(item)
        if not errors:
            fingerprint = json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            )
            if fingerprint in observation_fingerprints:
                continue
            observation_fingerprints.add(fingerprint)
            observations.append(item)
            continue
        diagnostic = {
            "type": "incomplete_exact_evaluation_identity",
            "invalid_fields": errors,
            **{
                key: item.get(key) for key in (
                    "project", "run_id", "attempt_id", "epoch", "step",
                    "variant_id", "family_id", "mode", "metric", "source",
                    "observed_at",
                )
            },
            "binding": dict(item.get("binding") or {}),
        }
        diagnostics.append(diagnostic)
        warnings.append(
            "excluded non-exact evaluation observation for "
            f"variant {item.get('variant_id')!r}: {', '.join(errors)}"
        )

    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in observations:
        key = tuple(item.get(name) for name in (
            "project", "run_id", "attempt_id", "epoch", "step",
            "variant_id", "family_id", "metric",
        ))
        by_key.setdefault(key, []).append(item)
    conflicting_keys: set[tuple[Any, ...]] = set()
    for key, sources in sorted(by_key.items(), key=lambda item: repr(item[0])):
        values = {json.dumps(item.get("value"), sort_keys=True) for item in sources}
        if len(values) <= 1:
            continue
        conflicting_keys.add(key)
        conflict = {
            "type": "metric_value_conflict",
            "project": key[0], "run_id": key[1], "attempt_id": key[2],
            "epoch": key[3], "step": key[4], "variant_id": key[5],
            "family_id": key[6], "metric": key[7],
            "sources": [{
                "source": item.get("source"), "value": item.get("value"),
                "observed_at": item.get("observed_at"),
                "binding": dict(item.get("binding") or {}) | {
                    name: item.get(name) for name in (
                        "project", "run_id", "attempt_id", "epoch", "step",
                        "variant_id", "family_id", "metric",
                    )
                },
            } for item in sources],
        }
        conflicts.append(conflict)
        warnings.append(
            f"{key[7]} has conflicting values for exact variant {key[5]} "
            f"at epoch={key[3]} step={key[4]}"
        )

    variant_bindings: dict[str, dict[str, Any]] = {}
    unresolved: set[str] = set()
    for item in candidates:
        variant_id = str(item["variant_id"])
        binding = item["binding"]
        previous = variant_bindings.setdefault(variant_id, binding)
        if previous != binding or binding.get("status") != "RESOLVED":
            unresolved.add(variant_id)
    unresolved.update(
        str(item.get("variant_id") or "unknown") for item in diagnostics
    )
    reconstruction = sorted(
        variant_id for variant_id, binding in variant_bindings.items()
        if binding.get("scope") == "FAMILY_INDEPENDENT_RECONSTRUCTION"
    )
    families: dict[str, list[str]] = {}
    for variant_id, binding in variant_bindings.items():
        family_id = binding.get("family_id")
        if binding.get("scope") == "SAMPLING_FAMILY" and isinstance(family_id, str):
            families.setdefault(family_id, []).append(variant_id)
        elif variant_id not in reconstruction:
            unresolved.add(variant_id)
    if len(reconstruction) != 1:
        unresolved.update(reconstruction)

    declared = _declared_family_id(manifest)
    selected_family: str | None = None
    if unresolved:
        family_state = "UNRESOLVED"
    elif declared is not None:
        family_state = "DECLARED" if declared in families else "CANONICAL_NOT_FOUND"
        selected_family = declared if declared in families else None
    elif len(families) == 1:
        family_state = "SINGLE_ELIGIBLE_FAMILY"
        selected_family = next(iter(families))
    elif len(families) > 1:
        family_state = "CANONICAL_NOT_DECLARED"
    else:
        family_state = "NOT_OBSERVED"

    by_variant: dict[str, dict[str, Any]] = {}
    for variant_id, binding in sorted(variant_bindings.items()):
        latest_by_metric: dict[str, dict[str, Any]] = {}
        for item in observations:
            if item["variant_id"] != variant_id:
                continue
            exact_key = tuple(item.get(name) for name in (
                "project", "run_id", "attempt_id", "epoch", "step",
                "variant_id", "family_id", "metric",
            ))
            if exact_key in conflicting_keys:
                continue
            current = latest_by_metric.get(item["metric"])
            order = (item.get("epoch") or -1, item.get("step") or -1,
                     str(item.get("source")))
            previous_order = (
                current.get("epoch") or -1, current.get("step") or -1,
                str(current.get("source")),
            ) if current else None
            if current is None or order >= previous_order:
                latest_by_metric[item["metric"]] = item
        by_variant[variant_id] = {
            "binding": binding,
            "metrics": {
                metric: {
                    key: item.get(key) for key in (
                        "epoch", "step", "value", "source", "observed_at",
                    )
                }
                for metric, item in sorted(latest_by_metric.items())
            },
            "diagnostics": [
                item for item in diagnostics if item.get("variant_id") == variant_id
            ],
        }

    metrics: dict[str, Any] = {}
    if selected_family is not None and len(reconstruction) == 1 and not unresolved:
        selected_variants = set(families[selected_family]) | set(reconstruction)
        by_checkpoint: dict[tuple[int | None, int | None], dict[str, list[dict[str, Any]]]] = {}
        for item in observations:
            if (
                item["variant_id"] not in selected_variants
                or item.get("family_id") != selected_family
            ):
                continue
            identity = (item.get("epoch"), item.get("step"))
            key = tuple(item.get(name) for name in (
                "project", "run_id", "attempt_id", "epoch", "step",
                "variant_id", "family_id", "metric",
            ))
            if key in conflicting_keys:
                continue
            by_checkpoint.setdefault(identity, {}).setdefault(
                item["metric"], [],
            ).append(item)
        for _, checkpoint in sorted(
            by_checkpoint.items(),
            key=lambda item: (item[0][0] or -1, item[0][1] or -1),
            reverse=True,
        ):
            primary: dict[str, Any] = {}
            complete = True
            for metric in EVAL_KEYS:
                expected_mode = next(
                    mode for mode, semantic in EVAL_PRIMARY_BY_MODE.items()
                    if semantic == metric
                )
                candidates = [
                    item for item in checkpoint.get(metric, [])
                    if item.get("mode") == expected_mode
                ]
                variants = {item["variant_id"] for item in candidates}
                if len(variants) != 1:
                    complete = False
                    break
                values = {json.dumps(item["value"]) for item in candidates}
                if len(values) != 1:
                    complete = False
                    break
                primary[metric] = candidates[0]["value"]
            if complete:
                entropy = checkpoint.get("generation_mean_entropy", [])
                if entropy:
                    primary["generation_mean_entropy"] = entropy[0]["value"]
                primary["plan_ppl_gap"] = (
                    primary["shuffled_plan_ppl"] - primary["oracle_plan_ppl"]
                )
                metrics = primary
                break

    evidence = {
        "schema_version": 2,
        "family_state": family_state,
        "canonical_family_id": selected_family,
        "unresolved_variant_ids": sorted(unresolved),
        "families": [{
            "family_id": family_id,
            "variant_ids": sorted(variant_ids),
            "sampling_dimensions": variant_bindings[variant_ids[0]].get(
                "sampling_dimensions",
            ),
        } for family_id, variant_ids in sorted(families.items())],
        "observations": observations,
        "diagnostics": diagnostics,
        "by_variant": by_variant,
    }
    return metrics, evidence, warnings, conflicts


def collect_eval_metrics(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Collect the latest canonical evaluation metrics from a run tree.

    Generation writes one ``metrics.jsonl`` per sampling mode, so this function
    searches recursively. For each canonical key it selects the record with the
    greatest step. If two files report different values for the same key and
    step, the lexicographically later path wins and a warning is returned; this
    makes ambiguity visible without preventing the remaining campaign summary.

    Args:
        run_dir: Durable run directory containing generation subdirectories.

    Returns:
        ``(metrics, warnings)`` where metrics contains canonical evaluation
        values and warnings describes same-step conflicts.
    """
    metrics, _, warnings, _ = collect_eval_evidence(run_dir)
    return metrics, warnings


def _jsonl_artifact_summary(
    paths: list[Path], *, nonempty_text_field: str | None = None
) -> dict[str, Any]:
    """Summarize JSONL files without inventing content semantics.

    ``records`` counts valid JSON objects for every JSONL artifact.  A
    ``nonempty_records`` count is included only when the caller names a text
    field whose non-empty content is meaningful (for example generated sample
    text).  Metric records intentionally do not expose that sample-specific
    field.
    """
    records = 0
    nonempty = 0
    for path in paths:
        payloads = read_jsonl(path)
        records += len(payloads)
        if nonempty_text_field is not None:
            nonempty += sum(
                isinstance(item.get(nonempty_text_field), str)
                and bool(item[nonempty_text_field].strip())
                for item in payloads
            )
    summary = {
        "matches": len(paths),
        "records": records,
    }
    if nonempty_text_field is not None:
        summary["nonempty_records"] = nonempty
    return summary


def collect_artifact_evidence(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Expose project-normalized artifact counts to the generic policy layer."""
    train = [run_dir / "train_metrics.jsonl"] if (run_dir / "train_metrics.jsonl").is_file() else []
    evaluation = sorted(run_dir.rglob("metrics.jsonl"))
    generated = sorted(run_dir.rglob("all_generated_*.jsonl"))
    reconstructed = sorted(run_dir.rglob("all_token_reconstructed_*.jsonl"))
    return {
        "train_metrics": _jsonl_artifact_summary(train),
        "evaluation_metrics": _jsonl_artifact_summary(evaluation),
        "generated_samples": _jsonl_artifact_summary(
            generated, nonempty_text_field="generated"
        ),
        "reconstructed_samples": _jsonl_artifact_summary(
            reconstructed, nonempty_text_field="generated"
        ),
    }


def latest_generation_nonempty_fraction(run_dir: Path) -> float | None:
    """Return the latest primary-generation non-empty fraction, if observable."""
    candidates: list[tuple[float, str, Path]] = []
    for metrics_path in sorted(run_dir.rglob("metrics.jsonl")):
        primary_steps = {
            float(record["step"])
            for record in read_jsonl(metrics_path)
            if record.get("mode") == "generation_refine_decode"
            and isinstance(record.get("step"), (int, float))
        }
        for sample_path in metrics_path.parent.glob("all_generated_*.jsonl"):
            match = _SAMPLE_FILE_RE.search(sample_path.name)
            if match and float(match.group("step")) in primary_steps:
                candidates.append((float(match.group("step")), str(sample_path), sample_path))
    if not candidates:
        return None
    records = read_jsonl(max(candidates)[2])
    if not records:
        return 0.0
    nonempty = sum(
        isinstance(item.get("generated"), str) and bool(item["generated"].strip())
        for item in records
    )
    return nonempty / len(records)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """Build one flat scientific/operational summary row for a run.

    Args:
        run_dir: Directory containing ``manifest.yaml`` and optional status and
        metric files.

    Returns:
        A flat mapping suitable for JSON or CSV output. Missing runtime metrics
        remain absent. ``plan_ppl_gap`` is ``shuffled_plan_ppl -
        oracle_plan_ppl`` so positive values mean the correct plan helped.
    """
    manifest = load_mapping(run_dir / "manifest.yaml")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported manifest schema in {run_dir / 'manifest.yaml'}")
    required = {"project", "run_id", "resolved_config", "source_id", "image_id"}
    missing = sorted(key for key in required if key not in manifest)
    if missing:
        raise ValueError(f"manifest is missing {missing} in {run_dir / 'manifest.yaml'}")
    status_path = run_dir / "status.json"
    status = load_mapping(status_path) if status_path.is_file() else {}
    backend_path = run_dir / "backend.json"
    runtime_backend = load_mapping(backend_path) if backend_path.is_file() else {}
    config = manifest["resolved_config"]
    if not isinstance(config, dict):
        raise ValueError(f"resolved_config must be a mapping in {run_dir / 'manifest.yaml'}")

    row: dict[str, Any] = {
        "project": manifest["project"],
        "run_id": manifest["run_id"],
        "state": status.get("state", "UNKNOWN"),
        "attempt_id": status.get("attempt_id"),
        "backend": runtime_backend.get("backend"),
        "source_id": manifest["source_id"],
        "image_id": manifest["image_id"],
        "run_dir": str(run_dir),
    }
    for key in SCIENTIFIC_CONFIG_KEYS:
        row[key] = config.get(key)

    train_path = run_dir / "train_metrics.jsonl"
    if train_path.is_file():
        latest_train = latest_record(read_jsonl(train_path))
        row.update({
            key: finite_metric_value(latest_train[key])
            for key in TRAIN_KEYS if key in latest_train
        })

    eval_metrics, metric_evidence, warnings, conflicts = collect_eval_evidence(run_dir)
    row.update(eval_metrics)
    row["metric_evidence"] = metric_evidence
    row["evaluation_metrics_by_variant"] = metric_evidence["by_variant"]
    row["evaluation_family_state"] = metric_evidence["family_state"]
    row["canonical_evaluation_family_id"] = metric_evidence[
        "canonical_family_id"
    ]
    row["artifacts"] = collect_artifact_evidence(run_dir)
    nonempty_fraction = latest_generation_nonempty_fraction(run_dir)
    if nonempty_fraction is not None:
        row["generation_nonempty_fraction"] = nonempty_fraction
    checkpoint = discover_latest_completed_checkpoint(run_dir)
    if checkpoint:
        row["latest_completed_checkpoint"] = checkpoint["path"]
        row["latest_completed_checkpoint_step"] = checkpoint["step"]
    nonfinite = sorted(
        key for key in (*TRAIN_KEYS, *EVAL_KEYS, "generation_mean_entropy")
        if key in row and row[key] is None
    )
    if nonfinite:
        row["nonfinite_metrics"] = nonfinite
    if warnings:
        row["warnings"] = warnings
    if conflicts:
        row["evidence_conflicts"] = conflicts
    return row


def write_json(rows: list[dict[str, Any]], stream: TextIO) -> None:
    """Write campaign rows as deterministic, human-readable JSON."""
    json.dump(rows, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")


def write_csv(rows: list[dict[str, Any]], stream: TextIO) -> None:
    """Write campaign rows as CSV using the union of all observed columns."""
    fieldnames = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        encoded = dict(row)
        if isinstance(encoded.get("warnings"), list):
            encoded["warnings"] = json.dumps(encoded["warnings"], ensure_ascii=False)
        writer.writerow(encoded)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI roots, output format, and optional destination path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="run or campaign directories")
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--output", type=Path, help="write to this file instead of stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Discover runs, summarize them, emit the report, and return a CLI status."""
    args = parse_args(argv)
    run_dirs = discover_run_dirs(args.roots)
    if not run_dirs:
        print("no manifest.yaml files found", file=sys.stderr)
        return 2
    rows = [summarize_run(run_dir) for run_dir in run_dirs]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as stream:
            write_csv(rows, stream) if args.format == "csv" else write_json(rows, stream)
    else:
        write_csv(rows, sys.stdout) if args.format == "csv" else write_json(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
