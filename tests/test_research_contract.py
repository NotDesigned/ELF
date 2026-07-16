import copy
import math
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from elf_experiments.campaign import instantiate_campaign_template, resolve_campaign
from elf_experiments.controller import load_campaign, materialize_run, prepare_run
from elf_experiments.research_contract import (
    evaluate_research_block,
    evaluate_research_run,
    validate_research_contract,
)


def load_template() -> dict:
    payload = yaml.safe_load(
        (REPO_ROOT / "experiments/templates/fusion_len256_gate_slurm.yml").read_text()
    )
    return resolve_campaign(instantiate_campaign_template(payload, "contract-test"))


def complete_collection(contract: dict, role: str) -> dict:
    values = {
        "state": "SUCCEEDED",
        "runtime_state": "SUCCEEDED",
        "train_loss": 1.0,
        "steps_per_sec": 2.0,
        "g_ppl": 30.0,
        "mauve": 80.0,
        "generation_mean_entropy": 3.0,
        "generation_nonempty_fraction": 1.0,
        "token_recon_ppl": 20.0,
        "oracle_plan_ppl": 15.0,
        "shuffled_plan_ppl": 18.0,
        "train_plan_emb_batch_var": 0.1,
        "train_plan_emb_norm": 27.0,
        "latest_completed_checkpoint": "/runs/run/checkpoint_10",
        "artifacts": {
            "train_metrics": {"records": 1},
            "evaluation_metrics": {"records": 1},
            "generated_samples": {"nonempty_records": 1},
            "reconstructed_samples": {"nonempty_records": 1},
        },
    }
    for metric in contract["required_metrics"]["by_role"].get(role, []):
        values.setdefault(metric, 1.0)
    return values


def test_fusion_template_expands_to_a_valid_explicit_contract(tmp_path):
    campaign = load_template()
    validate_research_contract(campaign)
    assert [run["research_role"] for run in campaign["runs"]] == ["a0", "a1", "a2", "a3"]
    authored = instantiate_campaign_template(
        yaml.safe_load(
            (REPO_ROOT / "experiments/templates/fusion_len256_gate_slurm.yml").read_text()
        ),
        "load-test",
    )
    path = tmp_path / "campaign.yml"
    path.write_text(yaml.safe_dump(authored, sort_keys=False), encoding="utf-8")
    assert load_campaign(path)["research_contract"]["schema_version"] == 1


def test_contract_rejects_role_drift_and_unknown_predicate():
    campaign = load_template()
    campaign["runs"][3]["research_role"] = "a2"
    with pytest.raises(ValueError, match="research_role values must be unique"):
        validate_research_contract(campaign)

    campaign = load_template()
    campaign["research_contract"]["terminal_checks"][0]["op"] = "python-eval"
    with pytest.raises(ValueError, match="unsupported op"):
        validate_research_contract(campaign)


def test_run_evaluator_distinguishes_missing_failed_and_passing_evidence():
    contract = load_template()["research_contract"]
    missing = complete_collection(contract, "a2")
    missing.pop("latest_completed_checkpoint")
    result = evaluate_research_run(
        status={"state": "SUCCEEDED"}, collection=missing,
        contract=contract, role="a2",
    )
    assert result["research_outcome"] == "INCONCLUSIVE"
    assert result["research_action"] == "VERIFY_RESULTS"

    failed = complete_collection(contract, "a2")
    failed["oracle_plan_ppl"] = 20.0
    result = evaluate_research_run(
        status={"state": "SUCCEEDED"}, collection=failed,
        contract=contract, role="a2",
    )
    assert result["research_outcome"] == "FAIL"
    assert result["research_action"] == "DO_NOT_EXTEND"

    passed = evaluate_research_run(
        status={"state": "SUCCEEDED"},
        collection=complete_collection(contract, "a2"),
        contract=contract, role="a2",
    )
    assert passed["research_outcome"] == "PASS"


def test_nonfinite_live_metric_only_recommends_stop():
    contract = load_template()["research_contract"]
    result = evaluate_research_run(
        status={"state": "RUNNING"}, collection={"train_loss": math.nan},
        contract=contract, role="a0",
    )
    assert result["research_outcome"] == "FAIL"
    assert result["research_action"] == "STOP_RECOMMENDED"


def test_block_requires_all_roles_and_matched_fields():
    campaign = load_template()
    contract = campaign["research_contract"]
    common_manifest = {
        "source_id": "source",
        "image_id": "sha256:" + "a" * 64,
        "resolved_config": {
            "seed": 42, "max_length": 256, "global_batch_size": 512,
            "epochs": 1, "grad_accum_steps": 2, "save_freq": 0.1,
            "eval_freq": 1, "num_samples": 256,
            "reconstruction_num_samples": 256,
        },
    }
    records = {
        run["research_role"]: {
            "research_outcome": "PASS",
            "manifest": copy.deepcopy(common_manifest),
            "run": run,
        }
        for run in campaign["runs"]
    }
    assert evaluate_research_block(
        contract=contract, role_records=records
    )["block_action"] == "EXTEND"
    records["a3"]["manifest"]["resolved_config"]["seed"] = 43
    mismatch = evaluate_research_block(contract=contract, role_records=records)
    assert mismatch["block_outcome"] == "INCOMPARABLE"
    assert mismatch["block_action"] == "DO_NOT_EXTEND"


def test_prepare_freezes_contract_and_role_in_scientific_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    campaign = load_template()
    campaign["local_root"] = str(tmp_path / "local")
    run = materialize_run(campaign, campaign["runs"][0], "source-fixed")
    manifest = prepare_run(campaign, run, "source-fixed", attempt_id="attempt-001")
    assert manifest["research_role"] == "a0"
    assert manifest["research_contract"] == campaign["research_contract"]
    attempt = yaml.safe_load(
        (
            tmp_path / "local" / campaign["campaign"] / run["run_id"]
            / "attempts" / "attempt-001" / "attempt.yaml"
        ).read_text()
    )
    assert attempt["research_role"] == "a0"
