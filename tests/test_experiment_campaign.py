import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_campaign import deep_merge, load_and_resolve_campaign, resolve_campaign
from experimentctl import materialize_run, validate_run


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_deep_merge_is_recursive_replaces_lists_and_does_not_mutate_inputs():
    base = {"backend": {"kind": "slurm", "time": "1:00"}, "overrides": ["epochs=1"]}
    override = {"backend": {"time": "2:00"}, "overrides": []}
    original = copy.deepcopy(base)

    result = deep_merge(base, override)

    assert result == {"backend": {"kind": "slurm", "time": "2:00"}, "overrides": []}
    assert base == original


def test_resolve_campaign_merges_defaults_profiles_and_run_in_order():
    payload = {
        "schema_version": 1,
        "campaign": "test",
        "project": "elf",
        "defaults": {
            "resources": {"gpus": 4, "cpus": 16},
            "storage": {"run_dir": "/data/runs/{run_id}"},
        },
        "profiles": {
            "slurm": {"backend": {"kind": "slurm", "time": "24:00:00"}},
            "h100": {"backend": {"partition": "h100", "gres": "gpu:h100:4"}},
        },
        "runs": [
            {
                "profile": ["slurm", "h100"],
                "run_id": "a0",
                "config": "a0.yml",
                "resources": {"cpus": 32},
            }
        ],
    }

    resolved = resolve_campaign(payload)

    assert "defaults" not in resolved
    assert "profiles" not in resolved
    assert resolved["runs"] == [
        {
            "run_id": "a0",
            "config": "a0.yml",
            "resources": {"gpus": 4, "cpus": 32},
            "storage": {"run_dir": "/data/runs/{run_id}"},
            "backend": {
                "kind": "slurm",
                "time": "24:00:00",
                "partition": "h100",
                "gres": "gpu:h100:4",
            },
        }
    ]


def test_matrix_expands_cartesian_product_and_preserves_runtime_placeholders():
    payload = {
        "runs": [
            {
                "matrix": {
                    "variant": [
                        {"name": "a0", "config": "pure.yml"},
                        {"name": "a1", "config": "frozen.yml"},
                    ],
                    "seed": [42, 43],
                },
                "template": {
                    "run_id": "elf-{variant.name}-s{seed}",
                    "config": "{variant.config}",
                    "seed": "{seed}",
                    "backend": {"source_dir": "/data/sources/{source_id}"},
                },
            }
        ]
    }

    runs = resolve_campaign(payload)["runs"]

    assert [run["run_id"] for run in runs] == [
        "elf-a0-s42",
        "elf-a0-s43",
        "elf-a1-s42",
        "elf-a1-s43",
    ]
    assert runs[0]["config"] == "pure.yml"
    assert runs[0]["seed"] == 42
    assert runs[0]["backend"]["source_dir"] == "/data/sources/{source_id}"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda campaign: campaign.update(defaults=[]), "defaults must be a mapping"),
        (lambda campaign: campaign["runs"][0].update(profile="missing"), "unknown profile"),
        (
            lambda campaign: campaign["runs"][0].update(
                matrix={"seed": []}, template={"run_id": "a-{seed}"}
            ),
            "non-empty list",
        ),
    ],
)
def test_rejects_invalid_authoring_helpers(mutation, message):
    campaign = {"profiles": {}, "runs": [{}]}
    mutation(campaign)
    with pytest.raises(ValueError, match=message):
        resolve_campaign(campaign)


def test_load_and_resolve_campaign(tmp_path):
    path = tmp_path / "campaign.yml"
    path.write_text(
        yaml.safe_dump({"defaults": {"resources": {"gpus": 1}}, "runs": [{"run_id": "a0"}]}),
        encoding="utf-8",
    )
    assert load_and_resolve_campaign(path)["runs"][0]["resources"] == {"gpus": 1}


@pytest.mark.parametrize(
    ("filename", "expected_runs"),
    [
        ("backend_smoke_sensecore_20260711.yml", 1),
        ("backend_smoke_slurm_20260711.yml", 4),
        ("fusion_len256_gate_h100_20260711.yml", 4),
        ("fusion_len256_gate_slurm_20260711.yml", 4),
        ("fusion_len256_gate_slurm_v2_20260711.yml", 4),
    ],
)
def test_repository_campaigns_expand_to_valid_independent_runs(filename, expected_runs):
    campaign = load_and_resolve_campaign(REPO_ROOT / "experiments" / "campaigns" / filename)

    assert len(campaign["runs"]) == expected_runs
    assert "defaults" not in campaign
    assert "profiles" not in campaign
    for run in campaign["runs"]:
        materialized = materialize_run(campaign, run, "test-source")
        validate_run(materialized)
        assert "profile" not in materialized
        assert "{run_id}" not in materialized["storage"]["run_dir"]
