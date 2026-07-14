import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from elf_experiments.summary import (
    collect_eval_evidence,
    collect_eval_metrics,
    discover_run_dirs,
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
    assert len(evidence["observations"]) == 9
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
        if item["binding"].get("scope") == "SAMPLING_FAMILY"
    } == {32, 64}


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
