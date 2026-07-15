import re
from pathlib import Path

import pytest
import torch
import yaml

from configs.config import (
    Config,
    SamplingConfig,
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


def test_mauve_config_override_and_seed_validation():
    cfg = apply_config_overrides(Config(), [
        "eval_mauve=false", "eval_mauve_model=gpt2", "eval_mauve_seed=7",
    ])
    assert cfg.eval_mauve is False
    assert cfg.eval_mauve_model == "gpt2"
    assert cfg.eval_mauve_seed == 7

    cfg.eval_mauve_seed = -1
    with pytest.raises(ValueError, match="eval_mauve_seed"):
        validate_config(cfg)

    cfg.eval_mauve_seed = 0
    cfg.eval_mauve = True
    cfg.eval_mauve_model = ""
    with pytest.raises(ValueError, match="eval_mauve_model"):
        validate_config(cfg)


def test_independent_plan_denoiser_config_validation():
    cfg = Config()
    cfg.use_sentence_plan = True
    cfg.plan_denoiser_type = "independent"
    cfg.plan_denoiser_depth = 3
    assert validate_config(cfg).plan_denoiser_depth == 3

    cfg.plan_denoiser_type = "coupled"
    with pytest.raises(ValueError, match="plan_denoiser_type"):
        validate_config(cfg)

    cfg.plan_denoiser_type = "independent"
    cfg.plan_denoiser_depth = 0
    with pytest.raises(ValueError, match="plan_denoiser_depth"):
        validate_config(cfg)


def test_float_override_uses_declared_type_after_integer_yaml_value():
    cfg = Config()
    cfg.save_freq = 1  # Mirrors YAML parsing of ``save_freq: 1``.
    cfg = apply_config_overrides(cfg, ["save_freq=0.1"])
    assert cfg.save_freq == pytest.approx(0.1)
    assert isinstance(cfg.save_freq, float)


def test_sampling_path_override_reloads_resolved_sampling_family(tmp_path):
    original_path = write_yaml(
        tmp_path / "original.yml",
        [{"sampling_method": "sde", "num_sampling_steps": [64]}],
    )
    override_path = write_yaml(
        tmp_path / "override.yml",
        [{"sampling_method": "sde", "num_sampling_steps": [32]}],
    )
    cfg = Config()
    cfg.sampling_configs_path = str(original_path)
    cfg.sampling_configs = load_sampling_configs(str(original_path))

    cfg = apply_config_overrides(
        cfg, [f"sampling_configs_path={override_path}"],
    )

    assert cfg.sampling_configs_path == str(override_path)
    assert len(cfg.sampling_configs) == 1
    assert cfg.sampling_configs[0].num_sampling_steps == [32]


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

    cfg = Config()
    cfg.global_batch_size = 512
    cfg.batch_size = None
    total, local = resolve_batch_sizes(cfg, world_size=8, grad_accum_steps=4)
    assert total == 512
    assert local == 16
    assert cfg.batch_size == 16


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
        "tier2_grad_detached_target_len256.yml",
        "tier2_grad_full.yml",
        "tier2_grad_full_len256.yml",
        "tier3_aux0.yml",
        "tier3_aux0_len256.yml",
        "tier3_aux2.yml",
        "tier3_aux2_len256.yml",
        "tier3_aux4.yml",
        "tier3_aux4_len256.yml",
        "tier4_independent_plan_denoiser.yml",
        "tier4_independent_plan_denoiser_len256.yml",
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
            cfg.plan_denoiser_type,
            int(cfg.plan_denoiser_depth),
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

    for path in root.glob("tier[23]*_len256.yml"):
        cfg = load_config_from_yaml(str(path))
        assert cfg.max_length == 256
        assert cfg.eval_ppl_max_length == 256

    independent = load_config_from_yaml(str(root / "tier4_independent_plan_denoiser_len256.yml"))
    assert independent.sentence_encoder_type == "sentence_t5"
    assert independent.plan_denoiser_type == "independent"
    assert independent.plan_denoiser_depth == 12
    assert independent.max_length == 256


def test_config_reference_covers_config_sampling_cli_and_launcher_flags():
    reference = Path("docs/config_reference.md").read_text(encoding="utf-8")

    declared_fields = set(Config.__annotations__) | set(SamplingConfig.__annotations__)
    missing_fields = sorted(name for name in declared_fields if f"`{name}`" not in reference)
    assert not missing_fields, f"config reference is missing fields: {missing_fields}"

    shell_sources = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in ("scripts/cloud_train.sh", "scripts/launch.sh")
    )
    env_names = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", shell_sources))
    env_names -= {"BASH_SOURCE", "EXTRA", "PYTHONPATH"}  # Internal shell plumbing.
    missing_env = sorted(name for name in env_names if f"`{name}`" not in reference)
    assert not missing_env, f"config reference is missing launcher variables: {missing_env}"

    for cli_flag in (
        "--config", "--config_override", "--use_cpu", "--seed", "--seeds", "--checkpoint_path",
    ):
        assert f"`{cli_flag}`" in reference
