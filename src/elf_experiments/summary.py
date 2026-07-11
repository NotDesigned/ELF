#!/usr/bin/env python
"""Summarize durable ELF run directories into machine-readable campaign rows."""

from __future__ import annotations

import argparse
import csv
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


def collect_eval_evidence(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str], list[str]]:
    """Collect canonical values with source/step evidence and hard conflicts."""
    selected: dict[str, tuple[float, str, Any]] = {}
    warnings: list[str] = []
    conflicts: list[str] = []
    for path in sorted(run_dir.rglob("metrics.jsonl")):
        for record in read_jsonl(path):
            try:
                step = float(record.get("step", -1))
            except (TypeError, ValueError):
                step = -1.0
            candidates = [
                (key, finite_metric_value(record[key]))
                for key in EVAL_KEYS if key in record
            ]
            if record.get("mode") == "generation_refine_decode" and "mean_entropy" in record:
                candidates.append((
                    "generation_mean_entropy",
                    finite_metric_value(record["mean_entropy"]),
                ))
            for key, value in candidates:
                candidate = (step, str(path.relative_to(run_dir)), value)
                previous = selected.get(key)
                if previous and previous[0] == step and previous[2] != value:
                    message = (
                        f"{key} has conflicting values at step {step:g}: "
                        f"{previous[1]}={previous[2]!r}, {candidate[1]}={value!r}"
                    )
                    warnings.append(message)
                    conflicts.append(message)
                if previous is None or candidate[:2] >= previous[:2]:
                    selected[key] = candidate
    metrics = {key: value for key, (_, _, value) in selected.items()}
    evidence = {
        key: {"step": step, "path": path, "value": value}
        for key, (step, path, value) in selected.items()
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


def _artifact_summary(paths: list[Path]) -> dict[str, Any]:
    records = 0
    nonempty = 0
    for path in paths:
        payloads = read_jsonl(path)
        records += len(payloads)
        nonempty += sum(
            isinstance(item.get("generated"), str) and bool(item["generated"].strip())
            for item in payloads
        )
    return {
        "matches": len(paths),
        "records": records,
        "nonempty_records": nonempty,
    }


def collect_artifact_evidence(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Expose project-normalized artifact counts to the generic policy layer."""
    train = [run_dir / "train_metrics.jsonl"] if (run_dir / "train_metrics.jsonl").is_file() else []
    evaluation = sorted(run_dir.rglob("metrics.jsonl"))
    generated = sorted(run_dir.rglob("all_generated_*.jsonl"))
    reconstructed = sorted(run_dir.rglob("all_token_reconstructed_*.jsonl"))
    return {
        "train_metrics": _artifact_summary(train),
        "evaluation_metrics": _artifact_summary(evaluation),
        "generated_samples": _artifact_summary(generated),
        "reconstructed_samples": _artifact_summary(reconstructed),
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
    if (
        isinstance(row.get("oracle_plan_ppl"), (int, float))
        and isinstance(row.get("shuffled_plan_ppl"), (int, float))
    ):
        oracle = metric_evidence["oracle_plan_ppl"]
        shuffled = metric_evidence["shuffled_plan_ppl"]
        if oracle["step"] == shuffled["step"]:
            row["plan_ppl_gap"] = row["shuffled_plan_ppl"] - row["oracle_plan_ppl"]
        else:
            message = "oracle_plan_ppl and shuffled_plan_ppl come from different steps"
            warnings.append(message)
            conflicts.append(message)
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
