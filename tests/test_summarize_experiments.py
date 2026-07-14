import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from elf_experiments.summary import (
    _exact_observation_errors,
    _family_id,
    collect_eval_evidence,
    collect_eval_metrics,
    discover_run_dirs,
    merge_local_scientific_evidence,
    read_jsonl,
    summarize_run,
)


def make_run(tmp_path: Path) -> Path:
    """Create the smallest durable run fixture needed by summary tests."""
    run_dir = tmp_path / "campaign" / "run-a"
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "project": "elf",
        "run_id": "run-a",
        "source_id": "source-abc",
        "image_id": "image-def",
        "resolved_config": {
            "seed": 42,
            "max_length": 256,
            "global_batch_size": 512,
            "use_sentence_plan": True,
            "sentence_encoder_type": "learned",
            "sentence_encoder_grad": "none",
            "plan_aux_passes": 1,
            "plan_aux_token_context": "denoiser_z",
        },
    }
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (run_dir / "status.json").write_text(
        json.dumps({"state": "RUNNING", "attempt_id": "attempt-001"}), encoding="utf-8"
    )
    (run_dir / "train_metrics.jsonl").write_text(
        json.dumps({"step": 100, "train_loss": 2.0}) + "\n"
        + json.dumps({"step": 200, "train_loss": 1.5, "train_plan_emb_norm": 27.6})
        + "\n",
        encoding="utf-8",
    )
    dimensions = {
        "sampling_method": "sde", "num_sampling_steps": 32, "cfg": 1.0,
        "self_cond_cfg_scale": 3.0, "time_schedule": "logit_normal",
        "time_warp_gamma": 1.5,
    }
    for variant, mode, metric, value in (
        ("clean", "clean_token_reconstruction", "token_recon_ppl", 40.0),
        ("generation", "generation_refine_decode", "g_ppl", 31.0),
        ("oracle", "oracle_plan_generation", "oracle_plan_ppl", 20.0),
        ("shuffled", "shuffled_plan_generation", "shuffled_plan_ppl", 25.5),
    ):
        eval_dir = run_dir / "train_sampling_eval" / variant
        eval_dir.mkdir(parents=True)
        record = {"epoch": 1, "step": 200, "mode": mode, metric: value}
        if variant != "clean":
            record["sampling_config"] = dimensions
        (eval_dir / "metrics.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8",
        )
    return run_dir


def test_discovers_and_summarizes_run(tmp_path):
    run_dir = make_run(tmp_path)
    assert discover_run_dirs([tmp_path]) == [run_dir.resolve()]

    row = summarize_run(run_dir)
    assert row["run_id"] == "run-a"
    assert row["state"] == "RUNNING"
    assert row["step"] == 200
    assert row["train_loss"] == 1.5
    assert row["g_ppl"] == 31.0
    assert row["plan_ppl_gap"] == 5.5


def test_local_scientific_merge_preserves_exact_operational_preimage():
    previous = {
        "project": "elf",
        "run_id": "run-a",
        "attempt_id": "attempt-001",
        "source_id": "source-abc",
        "image_id": "image-def",
        "state": "SUCCEEDED",
        "backend": "slurm",
        "run_dir": "/remote/run-a",
        "collected_from": "/remote/run-a",
        "scheduler_state": "SUCCEEDED",
        "worker_state": "RELEASED",
        "process_state": "SUCCEEDED",
        "runtime_state": "SUCCEEDED",
        "model_state": "OBSERVED",
        "process_evidence": {
            "observed": True,
            "stdout_tail": ["checkpoint committed"],
            "sources": {"stdout": "/remote/run-a/stdout.log"},
        },
        "evidence_outcome": "OBSERVED",
        "evidence_unavailable_reason": None,
        "latest_completed_checkpoint": "/remote/run-a/checkpoint_38035",
        "latest_completed_checkpoint_step": 38035,
        "collection_provenance": {"collector": "slurm", "job": "123"},
        "train_loss": 9.0,
        "g_ppl": 50.0,
        "oracle_plan_ppl": 40.0,
        "shuffled_plan_ppl": 45.0,
        "plan_ppl_gap": 5.0,
        "metric_evidence": {"g_ppl": {"step": 38035, "value": 50.0}},
        "evidence_conflicts": [{"metric": "g_ppl"}],
        "warnings": ["stale conflict"],
    }
    summary = {
        "project": "elf",
        "run_id": "run-a",
        "attempt_id": "attempt-001",
        "source_id": "source-abc",
        "image_id": "image-def",
        # These are deliberately different; a scientific rebuild does not own
        # any of them and therefore cannot rewrite the exact observation.
        "state": "FAILED",
        "backend": "local",
        "run_dir": "/local/collected_run",
        "train_loss": 1.25,
        "metric_evidence": {
            "family_state": "UNRESOLVED",
            "by_variant": {"generation": {"binding": {"status": "UNRESOLVED"}}},
        },
        "evaluation_metrics_by_variant": {
            "generation": {"binding": {"status": "UNRESOLVED"}},
        },
        "evaluation_family_state": "UNRESOLVED",
        "canonical_evaluation_family_id": None,
        "warnings": ["family identity is incomplete"],
        "artifacts": {"train_metrics": {"records": 1}},
    }

    rebuilt = merge_local_scientific_evidence(previous, summary)

    preserved = (
        "project", "run_id", "attempt_id", "source_id", "image_id",
        "state", "backend", "run_dir", "collected_from",
        "scheduler_state", "worker_state", "process_state", "runtime_state",
        "model_state", "process_evidence", "evidence_outcome",
        "evidence_unavailable_reason", "latest_completed_checkpoint",
        "latest_completed_checkpoint_step", "collection_provenance",
    )
    assert {key: rebuilt[key] for key in preserved} == {
        key: previous[key] for key in preserved
    }
    assert rebuilt["train_loss"] == 1.25
    assert rebuilt["evaluation_family_state"] == "UNRESOLVED"
    assert rebuilt["evaluation_metrics_by_variant"] == summary[
        "evaluation_metrics_by_variant"
    ]
    assert "g_ppl" not in rebuilt
    assert "oracle_plan_ppl" not in rebuilt
    assert "shuffled_plan_ppl" not in rebuilt
    assert "plan_ppl_gap" not in rebuilt
    assert "evidence_conflicts" not in rebuilt
    assert rebuilt["warnings"] == ["family identity is incomplete"]


@pytest.mark.parametrize(
    "key", ["project", "run_id", "attempt_id", "source_id", "image_id"],
)
def test_local_scientific_merge_rejects_identity_rewrite(key):
    previous = {
        "project": "elf", "run_id": "run-a", "attempt_id": "attempt-001",
        "source_id": "source-a", "image_id": "image-a",
    }
    summary = {**previous, key: "different"}

    with pytest.raises(ValueError, match=rf"summary {key} conflicts"):
        merge_local_scientific_evidence(previous, summary)


def test_eval_conflict_is_reported_and_deterministic(tmp_path):
    run_dir = make_run(tmp_path)
    generation = run_dir / "train_sampling_eval" / "generation" / "metrics.jsonl"
    original = json.loads(generation.read_text(encoding="utf-8"))
    generation.write_text(
        json.dumps(original) + "\n"
        + json.dumps({**original, "g_ppl": 99.0}) + "\n",
        encoding="utf-8",
    )

    metrics, warnings = collect_eval_metrics(run_dir)
    assert "g_ppl" not in metrics
    assert warnings and "exact variant generation" in warnings[0]
    _, evidence, _, conflicts = collect_eval_evidence(run_dir)
    assert evidence["family_state"] == "SINGLE_ELIGIBLE_FAMILY"
    assert len(conflicts) == 1
    assert conflicts[0]["variant_id"] == "generation"
    assert conflicts[0]["metric"] == "g_ppl"
    assert {source["value"] for source in conflicts[0]["sources"]} == {
        31.0, 99.0,
    }


def test_summary_exposes_artifacts_entropy_and_nonempty_generation(tmp_path):
    run_dir = make_run(tmp_path)
    generation = run_dir / "train_sampling_eval" / "generation"
    metrics_path = generation / "metrics.jsonl"
    generation_record = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_path.write_text(
        json.dumps({**generation_record, "mean_entropy": 2.5}) + "\n",
        encoding="utf-8",
    )
    (generation / "all_generated_1_200.jsonl").write_text(
        json.dumps({"generated": "text"}) + "\n"
        + json.dumps({"generated": ""}) + "\n",
        encoding="utf-8",
    )
    reconstruction = run_dir / "reconstruction"
    reconstruction.mkdir()
    (reconstruction / "all_token_reconstructed_1_200.jsonl").write_text(
        json.dumps({"generated": "reconstructed"}) + "\n", encoding="utf-8"
    )
    row = summarize_run(run_dir)
    assert row["generation_mean_entropy"] == 2.5
    assert row["generation_nonempty_fraction"] == 0.5
    assert row["artifacts"]["train_metrics"] == {"matches": 1, "records": 2}
    assert "nonempty_records" not in row["artifacts"]["evaluation_metrics"]
    assert row["artifacts"]["generated_samples"]["nonempty_records"] == 1
    assert row["artifacts"]["reconstructed_samples"]["records"] == 1


def test_plan_gap_is_not_assembled_across_different_steps(tmp_path):
    run_dir = make_run(tmp_path)
    oracle_path = run_dir / "train_sampling_eval" / "oracle" / "metrics.jsonl"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    oracle_path.write_text(
        json.dumps({**oracle, "step": 100}) + "\n", encoding="utf-8",
    )
    row = summarize_run(run_dir)
    assert "plan_ppl_gap" not in row
    assert "evidence_conflicts" not in row


def test_missing_sampling_dimensions_are_unresolved_and_never_flat(tmp_path):
    run_dir = make_run(tmp_path)
    generation_path = (
        run_dir / "train_sampling_eval" / "generation" / "metrics.jsonl"
    )
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation.pop("sampling_config")
    generation_path.write_text(json.dumps(generation) + "\n", encoding="utf-8")

    row = summarize_run(run_dir)

    assert row["evaluation_family_state"] == "UNRESOLVED"
    assert row["metric_evidence"]["unresolved_variant_ids"] == ["generation"]
    assert "g_ppl" not in row
    assert row["evaluation_metrics_by_variant"]["generation"]["binding"][
        "status"
    ] == "UNRESOLVED"
    assert row["evaluation_metrics_by_variant"]["generation"]["diagnostics"][0][
        "invalid_fields"
    ] == ["family_id"]


@pytest.mark.parametrize(
    ("mutation", "invalid_field", "variant"),
    [
        ("missing_status", "attempt_id", None),
        ("empty_project", "project", None),
        ("none_run", "run_id", None),
        ("missing_epoch", "epoch", "generation"),
        ("missing_step", "step", "generation"),
        ("invalid_variant", "variant_id", " "),
    ],
)
def test_incomplete_eval_identity_is_diagnostic_only_and_never_flat(
    tmp_path, mutation, invalid_field, variant,
):
    run_dir = make_run(tmp_path)
    if mutation == "missing_status":
        (run_dir / "status.json").unlink()
    elif mutation in {"empty_project", "none_run"}:
        manifest_path = run_dir / "manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest["project" if mutation == "empty_project" else "run_id"] = (
            "" if mutation == "empty_project" else None
        )
        manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    elif mutation in {"missing_epoch", "missing_step"}:
        metrics_path = run_dir / "train_sampling_eval" / "generation" / "metrics.jsonl"
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
        record.pop("epoch" if mutation == "missing_epoch" else "step")
        metrics_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    else:
        source = run_dir / "train_sampling_eval" / "generation"
        source.rename(source.with_name(" "))

    row = summarize_run(run_dir)

    for metric in (
        "token_recon_ppl", "g_ppl", "oracle_plan_ppl",
        "shuffled_plan_ppl", "plan_ppl_gap",
    ):
        assert metric not in row
    diagnostics = row["metric_evidence"]["diagnostics"]
    assert diagnostics
    assert any(invalid_field in item["invalid_fields"] for item in diagnostics)
    observations = row["metric_evidence"]["observations"]
    assert all(not any(
        all(observation.get(key) == item.get(key) for key in (
            "variant_id", "metric", "source",
        ))
        for observation in observations
    ) for item in diagnostics)
    if variant is not None:
        assert variant in row["metric_evidence"]["unresolved_variant_ids"]


def test_exact_observation_source_and_family_identity_are_validated():
    dimensions = {
        "sampling_method": "sde", "num_sampling_steps": 32, "cfg": 1.0,
        "self_cond_cfg_scale": 3.0, "time_schedule": "logit_normal",
        "time_warp_gamma": 1.5,
    }
    observation = {
        "project": "elf", "run_id": "run-a", "attempt_id": "attempt-001",
        "epoch": 1, "step": 200, "variant_id": "generation",
        "family_id": _family_id(dimensions), "source": "eval/metrics.jsonl",
        "binding": {"sampling_dimensions": dimensions},
    }
    assert _exact_observation_errors(observation) == []

    observation["source"] = "../metrics.jsonl"
    assert _exact_observation_errors(observation) == ["source"]

    observation["source"] = "eval/metrics.jsonl"
    observation["family_id"] = "sha256:" + "0" * 64
    assert _exact_observation_errors(observation) == ["family_id"]


def test_two_real_sampling_families_remain_distinct_without_flat_conflicts(tmp_path):
    run_dir = make_run(tmp_path)
    manifest_path = run_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = "elf-aux1-mb64-ga2-h100-20260714-r1"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    sampling_root = run_dir / "train_sampling_eval"
    for path in sampling_root.iterdir():
        if path.name != "clean":
            for child in path.iterdir():
                child.unlink()
            path.rmdir()
    observed_at = 1784029337.1040568
    clean_path = sampling_root / "clean" / "metrics.jsonl"
    clean_record = json.loads(clean_path.read_text(encoding="utf-8"))
    clean_path.write_text(
        json.dumps({**clean_record, "observed_at": observed_at}) + "\n",
        encoding="utf-8",
    )
    families = [
        ({
            "sampling_method": "sde", "num_sampling_steps": 32, "cfg": 1.0,
            "self_cond_cfg_scale": 3.0, "time_schedule": "logit_normal",
            "time_warp_gamma": 1.5,
        }, {
            "g_ppl": 252.03480132729845,
            "generation_mean_entropy": 4.605427395552397,
            "oracle_plan_ppl": 188.26621848150643,
            "shuffled_plan_ppl": 180.62738464568906,
        }),
        ({
            "sampling_method": "sde", "num_sampling_steps": 64, "cfg": 1.0,
            "self_cond_cfg_scale": 3.0, "time_schedule": "logit_normal",
            "time_warp_gamma": 1.0,
        }, {
            "g_ppl": 238.58811625648886,
            "generation_mean_entropy": 4.608551295474172,
            "oracle_plan_ppl": 177.8099580326585,
            "shuffled_plan_ppl": 179.51874195224673,
        }),
    ]
    for dimensions, values in families:
        steps = dimensions["num_sampling_steps"]
        for alias, mode, metric in (
            ("generation", "generation_refine_decode", "g_ppl"),
            ("oracle", "oracle_plan_generation", "oracle_plan_ppl"),
            ("shuffled", "shuffled_plan_generation", "shuffled_plan_ppl"),
        ):
            root = sampling_root / f"steps{steps}-{alias}"
            root.mkdir()
            record = {
                "epoch": 1, "step": 38035, "mode": mode,
                metric: values[metric], "sampling_config": dimensions,
                "observed_at": observed_at,
            }
            if alias == "generation":
                record["mean_entropy"] = values["generation_mean_entropy"]
            (root / "metrics.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8",
            )

    row = summarize_run(run_dir)

    assert row["evaluation_family_state"] == "CANONICAL_NOT_DECLARED"
    assert row["canonical_evaluation_family_id"] is None
    assert len(row["evaluation_metrics_by_variant"]) == 7
    assert "evidence_conflicts" not in row
    for metric in (
        "token_recon_ppl", "g_ppl", "oracle_plan_ppl",
        "shuffled_plan_ppl", "generation_mean_entropy", "plan_ppl_gap",
    ):
        assert metric not in row
    evidence = row["metric_evidence"]
    assert len(evidence["families"]) == 2
    assert len(evidence["observations"]) == 10
    assert {item["observed_at"] for item in evidence["observations"]} == {
        observed_at,
    }
    assert {item["attempt_id"] for item in evidence["observations"]} == {
        "attempt-001",
    }
    assert {item["run_id"] for item in evidence["observations"]} == {
        "elf-aux1-mb64-ga2-h100-20260714-r1",
    }
    assert {
        item["binding"]["sampling_dimensions"]["num_sampling_steps"]
        for item in evidence["observations"]
    } == {32, 64}
    assert all(item["family_id"].startswith("sha256:") for item in evidence["observations"])


def test_nonfinite_metrics_remain_json_safe_and_visible_to_policy(tmp_path):
    run_dir = make_run(tmp_path)
    (run_dir / "train_metrics.jsonl").write_text(
        json.dumps({"step": 300, "train_loss": float("nan")}) + "\n",
        encoding="utf-8",
    )
    row = summarize_run(run_dir)
    assert row["train_loss"] is None
    assert row["nonfinite_metrics"] == ["train_loss"]
    json.dumps(row, allow_nan=False)


def test_malformed_jsonl_reports_path_and_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"ok": 1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.jsonl:2"):
        read_jsonl(path)


def test_legacy_manifest_schema_is_rejected(tmp_path):
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_id": "legacy-run",
                "status": "waiting_for_spot_capacity",
                "source": {"git_commit": "deadbeef"},
                "image": {"uri": "registry/elf:old"},
                "backend": {"kind": "legacy-backend"},
                "scientific_parameters": {
                    "seed": 42,
                    "max_length": 256,
                    "global_batch_size": 512,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported manifest schema"):
        summarize_run(run_dir)
