from types import SimpleNamespace
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.modules.setdefault(
    "sacrebleu",
    types.SimpleNamespace(corpus_bleu=lambda *args, **kwargs: types.SimpleNamespace(score=0.0)),
)

from generation import (
    _IndexedSubset,
    _capture_rng_state,
    _evaluation_sampling_dimensions,
    _restore_rng_state,
)
from configs.config import SamplingConfig
from elf_experiments.summary import _exact_observation_errors, _record_binding
from utils.generation_utils import _dlm_decode_batch, _generate_samples_single_batch


class DecodeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.seen_plan_z = None

    def forward(
        self,
        x,
        t,
        deterministic=True,
        self_cond_cfg_scale=None,
        decoder_step_active=None,
        plan_z=None,
        plan_t=None,
        return_plan=False,
    ):
        self.seen_plan_z = plan_z
        logits = torch.zeros(x.shape[0], x.shape[1], 5, dtype=x.dtype, device=x.device)
        logits[..., 3] = 1.0
        return torch.zeros_like(x), logits


class SamplingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x,
        t,
        deterministic=True,
        self_cond_cfg_scale=None,
        decoder_step_active=None,
        plan_z=None,
        plan_t=None,
        return_plan=False,
    ):
        if return_plan:
            return torch.zeros_like(x), None, torch.zeros_like(plan_z)
        return torch.zeros_like(x), None


def generation_config(**overrides):
    cfg = SimpleNamespace(
        use_sentence_plan=True,
        sentence_emb_dim=3,
        plan_noise_scale=1.0,
        denoiser_noise_scale=1.0,
        use_bf16=False,
        t_eps=5e-2,
        self_cond_prob=0.0,
        num_self_cond_cfg_tokens=0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_indexed_subset_preserves_original_indices():
    dataset = [
        {"text": "zero"},
        {"text": "one", "index": 99},
        {"text": "two"},
        {"text": "three"},
    ]
    subset = _IndexedSubset(dataset, [3, 1, 0])

    assert len(subset) == 3
    assert subset[0] == {"text": "three", "index": 3}
    assert subset[1] == {"text": "one", "index": 99}
    assert subset[2] == {"text": "zero", "index": 0}


def test_evaluation_sampling_dimensions_are_exact_normalized_scalars():
    sampling_config = SamplingConfig(
        sampling_method="sde",
        num_sampling_steps=[32],
        cfgs=[1],
        self_cond_cfg_scales=[3],
        time_schedule="logit_normal",
        sde_gamma=1.5,
    )

    dimensions = _evaluation_sampling_dimensions(
        sampling_config,
        num_sampling_steps=32,
        cfg_scale=1,
        self_cond_cfg_scale=3,
    )

    assert dimensions == {
        "sampling_method": "sde",
        "num_sampling_steps": 32,
        "cfg": 1.0,
        "self_cond_cfg_scale": 3.0,
        "time_schedule": "logit_normal",
        "time_warp_gamma": 1.5,
    }

    binding = _record_binding({
        "mode": "generation_refine_decode",
        "sampling_dimensions": dimensions,
    })
    observation = {
        "project": "elf",
        "run_id": "run-a",
        "attempt_id": "attempt-001",
        "epoch": 1,
        "step": 38035,
        "variant_id": "steps32-generation",
        "family_id": binding["family_id"],
        "source": "steps32-generation/metrics.jsonl",
        "binding": binding,
    }
    assert binding["status"] == "RESOLVED"
    assert _exact_observation_errors(observation) == []


def test_rng_state_restore_replays_paired_cpu_noise():
    generator = torch.Generator(device="cpu").manual_seed(123)
    torch.manual_seed(456)
    state = _capture_rng_state(generator)

    explicit_first = torch.randn(8, generator=generator)
    global_first = torch.randn(8)
    _restore_rng_state(generator, state)
    explicit_second = torch.randn(8, generator=generator)
    global_second = torch.randn(8)

    assert torch.equal(explicit_first, explicit_second)
    assert torch.equal(global_first, global_second)


def test_dlm_decode_requires_plan_z_when_sentence_plan_enabled():
    z = torch.zeros(2, 4, 3)

    with pytest.raises(ValueError, match="plan_z is required"):
        _dlm_decode_batch(
            z=z,
            model=DecodeModel(),
            t_final_val=1.0,
            config=generation_config(),
            self_cond_cfg_scale=1.0,
            plan_z=None,
        )


def test_dlm_decode_passes_plan_z_to_model():
    model = DecodeModel()
    z = torch.zeros(2, 4, 3)
    plan_z = torch.ones(2, 3)

    decoded = _dlm_decode_batch(
        z=z,
        model=model,
        t_final_val=1.0,
        config=generation_config(),
        self_cond_cfg_scale=1.0,
        plan_z=plan_z,
    )

    assert torch.equal(decoded, torch.full((2, 4), 3))
    assert model.seen_plan_z is plan_z


def test_generate_samples_returns_plan_latent_when_enabled():
    z = torch.randn(2, 4, 3)
    t_steps = torch.tensor([0.0, 1.0])
    out = _generate_samples_single_batch(
        model=SamplingModel(),
        generator=torch.Generator(device="cpu").manual_seed(11),
        z=z,
        t_steps=t_steps,
        cond_seq=None,
        cond_seq_mask=None,
        config=generation_config(),
        sampling_config=SimpleNamespace(sampling_method="ode", sde_gamma=0.0),
        cfg_scale=1.0,
        self_cond_cfg_scale=1.0,
    )

    assert isinstance(out, tuple)
    token_z, plan_z = out
    assert token_z.shape == z.shape
    assert plan_z.shape == (2, 3)
