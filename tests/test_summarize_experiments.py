import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from elf_experiments.summary import (
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
    eval_dir = run_dir / "sampling"
    eval_dir.mkdir()
    (eval_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 200, "g_ppl": 31.0}) + "\n"
        + json.dumps({"step": 200, "oracle_plan_ppl": 20.0}) + "\n"
        + json.dumps({"step": 200, "shuffled_plan_ppl": 25.5}) + "\n",
        encoding="utf-8",
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
    other = run_dir / "zzz" / "metrics.jsonl"
    other.parent.mkdir()
    other.write_text(json.dumps({"step": 200, "g_ppl": 99.0}) + "\n", encoding="utf-8")

    metrics, warnings = collect_eval_metrics(run_dir)
    assert metrics["g_ppl"] == 99.0
    assert warnings and "conflicting values" in warnings[0]


def test_summary_exposes_artifacts_entropy_and_nonempty_generation(tmp_path):
    run_dir = make_run(tmp_path)
    generation = run_dir / "main-generation"
    generation.mkdir()
    (generation / "metrics.jsonl").write_text(
        json.dumps({
            "step": 200, "mode": "generation_refine_decode",
            "g_ppl": 31.0, "mean_entropy": 2.5,
        }) + "\n",
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
    eval_path = run_dir / "sampling" / "metrics.jsonl"
    eval_path.write_text(
        json.dumps({"step": 100, "oracle_plan_ppl": 20.0}) + "\n"
        + json.dumps({"step": 200, "shuffled_plan_ppl": 25.0}) + "\n",
        encoding="utf-8",
    )
    row = summarize_run(run_dir)
    assert "plan_ppl_gap" not in row
    assert row["evidence_conflicts"]


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
                "backend": {"kind": "sensecore"},
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
