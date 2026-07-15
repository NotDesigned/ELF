import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from elf_experiments.campaign import (
    deep_merge,
    instantiate_campaign_template,
    load_and_resolve_campaign,
    resolve_campaign,
)
from elf_experiments.controller import materialize_run, validate_run
from configs.config import Config, load_config_from_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hierarchical_plan_campaign_is_a_matched_three_arm_ablation():
    campaign = load_and_resolve_campaign(
        REPO_ROOT
        / "experiments/campaigns/fusion_hierarchical_plan_lead_h200_20260715.yml"
    )
    runs = {run["research_role"]: run for run in campaign["runs"]}
    assert set(runs) == {
        "joint_aligned",
        "triangular_aligned",
        "triangular_lead_g3",
    }

    configs = {
        role: load_config_from_yaml(str(REPO_ROOT / run["config"]))
        for role, run in runs.items()
    }
    assert {config.plan_denoiser_type for config in configs.values()} == {"shared"}
    assert configs["joint_aligned"].plan_attention_topology == "joint"
    assert configs["joint_aligned"].plan_time_schedule == "aligned"
    assert configs["triangular_aligned"].plan_attention_topology == "hierarchical_prefix"
    assert configs["triangular_aligned"].plan_time_schedule == "aligned"
    assert configs["triangular_lead_g3"].plan_attention_topology == "hierarchical_prefix"
    assert configs["triangular_lead_g3"].plan_time_schedule == "noise_power"
    assert configs["triangular_lead_g3"].plan_time_warp_gamma == pytest.approx(3.0)

    matched_fields = (
        "sentence_encoder_type",
        "sentence_emb_dim",
        "num_plan_tokens",
        "plan_loss_weight",
        "max_length",
        "eval_ppl_max_length",
    )
    for field in matched_fields:
        assert len({getattr(config, field) for config in configs.values()}) == 1
    assert len({run["image_id"] for run in runs.values()}) == 1
    assert len({run["backend"]["gres"] for run in runs.values()}) == 1


def test_prefix128_plan_campaign_exercises_prefix_conditioning_in_all_arms():
    campaign = load_and_resolve_campaign(
        REPO_ROOT
        / "experiments/campaigns/fusion_hierarchical_prefix128_plan_lead_h200_20260715.yml"
    )
    runs = {run["research_role"]: run for run in campaign["runs"]}
    assert set(runs) == {
        "joint_aligned",
        "triangular_aligned",
        "triangular_lead_g3",
    }

    configs = {
        role: load_config_from_yaml(str(REPO_ROOT / run["config"]))
        for role, run in runs.items()
    }
    assert {cfg.split_input_as_prefix for cfg in configs.values()} == {True}
    assert {cfg.max_input_length for cfg in configs.values()} == {128}
    assert {cfg.max_length for cfg in configs.values()} == {256}
    assert {cfg.plan_denoiser_type for cfg in configs.values()} == {"shared"}
    assert configs["joint_aligned"].plan_attention_topology == "joint"
    assert configs["joint_aligned"].plan_time_schedule == "aligned"
    assert configs["triangular_aligned"].plan_attention_topology == "hierarchical_prefix"
    assert configs["triangular_aligned"].plan_time_schedule == "aligned"
    assert configs["triangular_lead_g3"].plan_attention_topology == "hierarchical_prefix"
    assert configs["triangular_lead_g3"].plan_time_schedule == "noise_power"
    assert configs["triangular_lead_g3"].plan_time_warp_gamma == pytest.approx(3.0)
    assert len({run["image_id"] for run in runs.values()}) == 1
    assert len({run["backend"]["gres"] for run in runs.values()}) == 1

    allowed_config_differences = {
        "output_dir",
        "wandb_run_name",
        "wandb_tag",
        "plan_attention_topology",
        "plan_time_schedule",
        "plan_time_warp_gamma",
    }
    baseline = configs["joint_aligned"]

    def comparable(value):
        if isinstance(value, list):
            return [vars(item) if hasattr(item, "__dict__") else item for item in value]
        return value

    for role, config in configs.items():
        differences = {
            key for key in Config.__annotations__
            if comparable(getattr(config, key)) != comparable(getattr(baseline, key))
        }
        assert differences <= allowed_config_differences, (role, differences)


def test_research_project_campaign_catalog_matches_campaign_files():
    project = yaml.safe_load(
        (REPO_ROOT / "experiments/research_project.yaml").read_text(encoding="utf-8")
    )

    for entry in project["campaigns"]:
        campaign_path = REPO_ROOT / entry["file"]
        campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
        assert entry["name"] == campaign["campaign"], campaign_path


def test_stage_a_formal_matrix_retains_all_common_overrides():
    campaign = load_and_resolve_campaign(
        REPO_ROOT
        / "experiments/campaigns/fusion_clean_vs_t5_plan_quality_stage_a_h200_20260715.yml"
    )
    required = {
        "global_batch_size=null",
        "batch_size=16",
        "num_samples=256",
        "reconstruction_eval=true",
        "reconstruction_num_samples=256",
        "eval_mauve=true",
        "eval_mauve_model=gpt2-large",
        "data_path=/datapool/liangluocheng/elf/eval_data/openwebtext-t5-head256-v1",
    }

    assert len(campaign["runs"]) == 12
    for run in campaign["runs"]:
        overrides = set(run["config_overrides"])
        assert required <= overrides
        assert sum(
            value.startswith("sampling_configs_path=") for value in overrides
        ) == 1


def test_deep_merge_is_recursive_replaces_lists_and_does_not_mutate_inputs():
    base = {"backend": {"kind": "test-backend", "time": "1:00"}, "overrides": ["epochs=1"]}
    override = {"backend": {"time": "2:00"}, "overrides": []}
    original = copy.deepcopy(base)

    result = deep_merge(base, override)

    assert result == {"backend": {"kind": "test-backend", "time": "2:00"}, "overrides": []}
    assert base == original


def test_resolve_campaign_merges_defaults_profiles_and_run_in_order():
    payload = {
        "schema_version": 1,
        "campaign": "test",
        "project": "elf",
        "defaults": {
            "resources": {"gpus": 4, "cpus": 16},
            "storage": {"run_dir": "/runs/{run_id}"},
        },
        "profiles": {
            "scheduler": {"backend": {"kind": "test-backend", "time": "24:00:00"}},
            "accelerator": {"backend": {"partition": "accelerator", "gres": "gpu:accelerator:4"}},
        },
        "runs": [
            {
                "profile": ["scheduler", "accelerator"],
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
            "storage": {"run_dir": "/runs/{run_id}"},
            "backend": {
                "kind": "test-backend",
                "time": "24:00:00",
                "partition": "accelerator",
                "gres": "gpu:accelerator:4",
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
                    "backend": {"source_dir": "/sources/{source_id}"},
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
    assert runs[0]["backend"]["source_dir"] == "/sources/{source_id}"


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


def test_instantiate_campaign_template_creates_fresh_reviewable_identities():
    template = yaml.safe_load(
        (REPO_ROOT / "experiments/templates/backend_smoke_slurm.yml").read_text()
    )
    authored = instantiate_campaign_template(template, "20260712-a")
    assert authored["campaign"] == "backend-smoke-slurm-20260712-a"
    assert authored["runs"][0]["run_id"] == "elf-smoke-slurm-l40s-20260712-a"
    assert authored["defaults"]["image_id"] == (
        "sha256:318b80ae8c1d188b1cf1bca2972cc27ffd8148faddd72c25104499bce04b4b1e"
    )
    assert authored["profiles"]["wyd-l40s"]["backend"]["sif_path"].endswith(
        "/318b80ae8c1d188b1cf1bca2972cc27ffd8148faddd72c25104499bce04b4b1e.sif"
    )
    assert authored["instance"] == "20260712-a"
    resolved = resolve_campaign(authored)
    assert resolved["runs"][0]["storage"]["run_dir"].endswith("/{run_id}")
    materialized = materialize_run(resolved, resolved["runs"][0], "fresh-source")
    validate_run(materialized, project="elf")
    assert materialized["storage"]["run_dir"].endswith(
        "/elf-smoke-slurm-l40s-20260712-a"
    )


def test_instantiate_campaign_template_rejects_unsafe_instance():
    with pytest.raises(ValueError, match="instance must use"):
        instantiate_campaign_template({"campaign": "x-{instance}"}, "bad/value")


def test_instantiate_campaign_cli_refuses_overwrite(tmp_path):
    output = tmp_path / "fresh.yml"
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/instantiate_campaign.py"),
        str(REPO_ROOT / "experiments/templates/backend_smoke_slurm.yml"),
        "--instance", "fresh-a", "--output", str(output),
    ]
    first = subprocess.run(command, text=True, capture_output=True, check=False)
    second = subprocess.run(command, text=True, capture_output=True, check=False)
    assert first.returncode == 0
    assert second.returncode != 0
    assert load_and_resolve_campaign(output)["runs"][0]["run_id"].endswith("fresh-a")


def test_instantiate_campaign_cli_can_isolate_controller_state(tmp_path):
    output = tmp_path / "fresh.yml"
    local_root = tmp_path / "controller-state"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/instantiate_campaign.py"),
            str(REPO_ROOT / "experiments/templates/backend_smoke_slurm.yml"),
            "--instance", "isolated-a", "--output", str(output),
            "--local-root", str(local_root),
        ],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0
    assert load_and_resolve_campaign(output)["local_root"] == str(local_root)


def test_instantiate_campaign_cli_registers_atomically_without_duplicates(tmp_path):
    project_file = tmp_path / "research_project.yaml"
    project_file.write_text(
        yaml.safe_dump({"schema_version": 1, "project": "elf", "campaigns": []}),
        encoding="utf-8",
    )
    output = tmp_path / "campaigns" / "fresh.yml"
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/instantiate_campaign.py"),
        str(REPO_ROOT / "experiments/templates/backend_smoke_slurm.yml"),
        "--instance", "registered-a",
        "--output", str(output),
        "--register",
        "--project-file", str(project_file),
    ]
    first = subprocess.run(command, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    project = yaml.safe_load(project_file.read_text(encoding="utf-8"))
    assert project["campaigns"] == [{
        "name": "backend-smoke-slurm-registered-a",
        "file": str(output),
    }]
    assert output.is_file()

    duplicate_output = tmp_path / "campaigns" / "duplicate.yml"
    duplicate = subprocess.run(
        [*command[:6], str(duplicate_output), *command[7:]],
        text=True,
        capture_output=True,
        check=False,
    )
    assert duplicate.returncode != 0
    assert "already registered by name" in duplicate.stderr
    assert not duplicate_output.exists()


CAMPAIGN_FILES = sorted(
    path.name for path in (REPO_ROOT / "experiments" / "campaigns").glob("*.yml")
)


@pytest.mark.parametrize("filename", CAMPAIGN_FILES)
def test_repository_campaigns_expand_to_valid_independent_runs(filename):
    campaign = load_and_resolve_campaign(REPO_ROOT / "experiments" / "campaigns" / filename)

    assert campaign["runs"]
    assert len({run["run_id"] for run in campaign["runs"]}) == len(campaign["runs"])
    assert "defaults" not in campaign
    assert "profiles" not in campaign
    for run in campaign["runs"]:
        materialized = materialize_run(campaign, run, "test-source")
        validate_run(materialized, project=campaign["project"])
        assert "profile" not in materialized
        assert "{run_id}" not in materialized["storage"]["run_dir"]
