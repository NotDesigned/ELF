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
    _evaluation_generators,
    _teacher_forced_token_stats,
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
        self.seen_cond_seq_mask = None

    def forward(
        self,
        x,
        t,
        deterministic=True,
        self_cond_cfg_scale=None,
        attention_mask=None,
        cond_seq_mask=None,
        decoder_step_active=None,
        plan_z=None,
        plan_t=None,
        return_plan=False,
    ):
        self.seen_plan_z = plan_z
        self.seen_cond_seq_mask = cond_seq_mask
        logits = torch.zeros(x.shape[0], x.shape[1], 5, dtype=x.dtype, device=x.device)
        logits[..., 3] = 1.0
        return torch.zeros_like(x), logits


class SamplingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.seen_cond_seq_mask = None

    def forward(
        self,
        x,
        t,
        deterministic=True,
        self_cond_cfg_scale=None,
        decoder_step_active=None,
        cond_seq_mask=None,
        plan_z=None,
        plan_t=None,
        return_plan=False,
    ):
        self.seen_cond_seq_mask = cond_seq_mask
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


def test_token_rng_is_independent_from_plan_rng_consumption():
    base = torch.Generator(device="cpu").manual_seed(123)
    token_a, plan_a = _evaluation_generators(base, torch.device("cpu"))
    token_b, _ = _evaluation_generators(base, torch.device("cpu"))

    first_a = torch.randn(16, generator=token_a)
    torch.randn(4096, generator=plan_a)
    second_a = torch.randn(16, generator=token_a)
    first_b = torch.randn(16, generator=token_b)
    second_b = torch.randn(16, generator=token_b)

    assert torch.equal(first_a, first_b)
    assert torch.equal(second_a, second_b)


def test_sde_token_path_is_paired_with_and_without_plan_latent():
    base = torch.Generator(device="cpu").manual_seed(19)
    token_a, _ = _evaluation_generators(base, torch.device("cpu"))
    token_b, plan_b = _evaluation_generators(base, torch.device("cpu"))
    z = torch.randn(2, 4, 3, generator=torch.Generator().manual_seed(5))
    sampling = SimpleNamespace(sampling_method="sde", sde_gamma=1.0)

    token_only = _generate_samples_single_batch(
        model=SamplingModel(), generator=token_a, z=z.clone(),
        t_steps=torch.tensor([0.0, 0.5, 1.0]), cond_seq=None,
        cond_seq_mask=None, config=generation_config(use_sentence_plan=False),
        sampling_config=sampling, cfg_scale=1.0, self_cond_cfg_scale=1.0,
    )
    token_with_plan, _ = _generate_samples_single_batch(
        model=SamplingModel(), generator=token_b, plan_generator=plan_b,
        z=z.clone(), t_steps=torch.tensor([0.0, 0.5, 1.0]),
        cond_seq=None, cond_seq_mask=None, config=generation_config(),
        sampling_config=sampling, cfg_scale=1.0, self_cond_cfg_scale=1.0,
    )

    assert torch.equal(token_only, token_with_plan)


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


def test_hierarchical_decode_passes_prefix_mask_to_model():
    model = DecodeModel()
    z = torch.zeros(2, 4, 3)
    plan_z = torch.ones(2, 3)
    cond_seq_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)

    _dlm_decode_batch(
        z=z,
        model=model,
        t_final_val=1.0,
        config=generation_config(plan_attention_topology="hierarchical_prefix"),
        self_cond_cfg_scale=1.0,
        plan_z=plan_z,
        cond_seq_mask=cond_seq_mask,
    )

    assert model.seen_cond_seq_mask is cond_seq_mask


def test_teacher_forced_stats_are_masked_token_nll():
    model = DecodeModel()
    x0 = torch.zeros(2, 3, 4)
    input_ids = torch.tensor([[3, 0, 1], [3, 3, 2]])
    loss_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    plan_z = torch.ones(2, 3)

    nll_sum, token_count = _teacher_forced_token_stats(
        model=model,
        x0=x0,
        input_ids=input_ids,
        attention_mask=torch.ones_like(loss_mask),
        loss_mask=loss_mask,
        config=generation_config(),
        self_cond_cfg_scale=1.0,
        plan_z=plan_z,
    )

    expected = torch.nn.functional.cross_entropy(
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]]), torch.tensor([3]),
    )
    assert token_count.item() == 3
    assert nll_sum.item() == pytest.approx(3 * expected.item())


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


def test_hierarchical_sampling_passes_prefix_mask_to_every_model_step():
    model = SamplingModel()
    z = torch.randn(2, 4, 3)
    cond_seq = torch.randn_like(z)
    cond_seq_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)

    _generate_samples_single_batch(
        model=model,
        generator=torch.Generator(device="cpu").manual_seed(12),
        z=z,
        t_steps=torch.tensor([0.0, 1.0]),
        cond_seq=cond_seq,
        cond_seq_mask=cond_seq_mask,
        config=generation_config(plan_attention_topology="hierarchical_prefix"),
        sampling_config=SimpleNamespace(sampling_method="ode", sde_gamma=0.0),
        cfg_scale=1.0,
        self_cond_cfg_scale=1.0,
    )

    assert model.seen_cond_seq_mask is cond_seq_mask
