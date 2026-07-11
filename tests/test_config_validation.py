from pathlib import Path

import pytest
import torch
import yaml

from configs.config import (
    Config,
    apply_config_overrides,
    load_config_from_yaml,
    load_sampling_configs,
    resolve_batch_sizes,
    validate_config,
)
from utils.sampling_utils import plan_time_from_token_time


def write_yaml(path: Path, payload: dict):
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_missing_config_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config_from_yaml(str(tmp_path / "missing.yml"))


def test_unknown_config_key_raises(tmp_path):
    cfg_path = write_yaml(
        tmp_path / "bad.yml",
        {
            "model": "ELF-B",
            "sentence_encoder_gradd": "none",
        },
    )

    with pytest.raises(ValueError, match="Unknown config field"):
        load_config_from_yaml(str(cfg_path))


def test_overrides_accept_null_and_reject_bad_bool():
    cfg = Config()
    cfg.resume = "outputs/old"

    cfg = apply_config_overrides(cfg, ["resume=null"])
    assert cfg.resume is None

    with pytest.raises(ValueError, match="Invalid boolean override"):
        apply_config_overrides(cfg, ["use_wandb=maybe"])


def test_wandb_run_name_does_not_imply_stable_run_id():
    cfg = Config()
    cfg.wandb_run_name = "shared-display-name"

    assert cfg.wandb_run_id is None
    assert cfg.wandb_resume is None

    cfg = apply_config_overrides(cfg, ["wandb_run_id=fixed-id", "wandb_resume=allow"])
    assert cfg.wandb_run_id == "fixed-id"
    assert cfg.wandb_resume == "allow"


def test_plan_time_scheduler_noise_power_leads_and_validates():
    cfg = Config()
    cfg.plan_time_schedule = "noise_power"
    cfg.plan_time_warp_gamma = 2.0

    t = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(plan_time_from_token_time(t, cfg), torch.tensor([0.0, 0.75, 1.0]))

    cfg.plan_time_warp_gamma = 0.5
    with pytest.raises(ValueError, match="plan_time_warp_gamma"):
        validate_config(cfg)


def test_encoder_checkpoint_is_not_silently_ignored():
    cfg = Config()
    cfg.encoder_checkpoint = "some/encoder/checkpoint"

    with pytest.raises(ValueError, match="encoder_checkpoint is set"):
        validate_config(cfg)


def test_sampling_config_validation_fails_early(tmp_path):
    sampling_path = write_yaml(
        tmp_path / "sampling.yml",
        [
            {
                "sampling_method": "ode",
                "num_sampling_steps": [64],
                "cfgs": [1],
                "self_cond_cfg_scales": [1],
                "time_schedule": "logit_normals",
            }
        ],
    )

    with pytest.raises(ValueError, match="time_schedule"):
        load_sampling_configs(str(sampling_path))


def test_empty_sampling_config_file_raises(tmp_path):
    sampling_path = write_yaml(tmp_path / "empty_sampling.yml", [])

    with pytest.raises(ValueError, match="must not be empty"):
        load_sampling_configs(str(sampling_path))


def test_resolve_batch_sizes_rejects_ambiguous_or_indivisible():
    cfg = Config()
    cfg.global_batch_size = 512
    cfg.batch_size = 8
    with pytest.raises(ValueError, match="Specify only one"):
        resolve_batch_sizes(cfg, world_size=8)

    cfg = Config()
    cfg.global_batch_size = 512
    cfg.batch_size = None
    with pytest.raises(ValueError, match="must be divisible"):
        resolve_batch_sizes(cfg, world_size=3)

    cfg = Config()
    cfg.global_batch_size = 512
    cfg.batch_size = None
    total, local = resolve_batch_sizes(cfg, world_size=8)
    assert total == 512
    assert local == 64
    assert cfg.batch_size == 64


def test_owt_elfb_ablation_configs_are_unique_and_expected():
    root = Path("src/configs/training_configs/ablations/owt_elfb")
    expected = {
        "tier0_0_pure_elf.yml",
        "tier0_0_pure_elf_len256.yml",
        "tier0_1_sentence_t5.yml",
        "tier0_1_sentence_t5_len256.yml",
        "tier0_2_learned_main.yml",
        "tier0_2_learned_main_len256.yml",
        "tier2_grad_detached_target.yml",
        "tier2_grad_full.yml",
        "tier3_aux0.yml",
        "tier3_aux2.yml",
        "tier3_aux4.yml",
    }
    paths = sorted(root.glob("*.yml"))
    assert {p.name for p in paths} == expected

    seen = {}
    for path in paths:
        cfg = load_config_from_yaml(str(path))
        key = (
            bool(cfg.use_sentence_plan),
            cfg.sentence_encoder_type,
            cfg.sentence_encoder_grad,
            int(cfg.plan_aux_passes),
            cfg.plan_aux_token_context,
            cfg.plan_adapter_type,
            int(cfg.num_plan_tokens),
            bool(cfg.plan_learned_encoder_norm),
            float(cfg.plan_loss_weight),
            float(cfg.plan_noise_scale),
            int(cfg.max_length),
            int(cfg.eval_ppl_max_length),
        )
        assert key not in seen, f"{path.name} duplicates {seen.get(key)}"
        seen[key] = path.name

    detached = load_config_from_yaml(str(root / "tier2_grad_detached_target.yml"))
    full = load_config_from_yaml(str(root / "tier2_grad_full.yml"))
    assert detached.plan_aux_passes == 1
    assert full.plan_aux_passes == 1
