import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import modules.model as model_module
from configs.config import Config
from modules.model import ELF
from train_step import _decode_continuation_texts, train_step
from utils.train_utils import TrainState


class TinyEncoder(nn.Module):
    def __init__(self, vocab_size=32, dim=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, input_ids, attention_mask=None, deterministic=True):
        return self.emb(input_ids)


def test_shared_train_eval_model_factory_preserves_attention_topology(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setitem(model_module.ELF_models, "ELF-B", fake_factory)
    config = Config()
    config.use_sentence_plan = True
    config.plan_attention_topology = "hierarchical_prefix"
    config.model_depth = 6
    config.model_active_depth = 6

    built = model_module.build_elf_from_config(
        config, text_encoder_dim=512, vocab_size=32100,
    )

    assert built is sentinel
    assert captured["plan_attention_topology"] == "hierarchical_prefix"
    assert captured["depth"] == 6
    assert captured["active_depth"] == 6
    assert captured["text_encoder_dim"] == 512
    assert captured["vocab_size"] == 32100


def test_active_depth_preserves_checkpoint_schema_and_skips_late_blocks():
    full = ELF(
        text_encoder_dim=4,
        max_length=6,
        hidden_size=32,
        depth=3,
        num_heads=4,
        mlp_ratio=2.0,
        bottleneck_dim=8,
        num_time_tokens=1,
        num_self_cond_cfg_tokens=0,
        vocab_size=32,
    )
    early_exit = ELF(
        text_encoder_dim=4,
        max_length=6,
        hidden_size=32,
        depth=3,
        active_depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        bottleneck_dim=8,
        num_time_tokens=1,
        num_self_cond_cfg_tokens=0,
        vocab_size=32,
    )
    early_exit.load_state_dict(full.state_dict(), strict=True)
    assert early_exit.state_dict().keys() == full.state_dict().keys()

    calls = [0, 0, 0]
    hooks = []
    for index, block in enumerate(early_exit.blocks):
        hooks.append(block.register_forward_hook(
            lambda _module, _args, _output, index=index: calls.__setitem__(index, calls[index] + 1)
        ))
    try:
        early_exit(
            torch.randn(2, 6, 4),
            torch.full((2,), 0.5),
            attention_mask=torch.ones(2, 6),
            cond_seq_mask=torch.zeros(2, 6),
        )
    finally:
        for hook in hooks:
            hook.remove()

    assert calls == [1, 1, 0]


def test_model_depth_instantiates_an_actually_smaller_checkpoint():
    full = model_module.ELF_B(
        text_encoder_dim=4,
        max_length=6,
        num_time_tokens=1,
        num_self_cond_cfg_tokens=0,
        vocab_size=32,
    )
    shallow = model_module.ELF_B(
        text_encoder_dim=4,
        max_length=6,
        depth=3,
        num_time_tokens=1,
        num_self_cond_cfg_tokens=0,
        vocab_size=32,
    )

    assert len(full.blocks) == 12
    assert len(shallow.blocks) == 3
    assert sum(p.numel() for p in shallow.parameters()) < sum(p.numel() for p in full.parameters())
    assert "blocks.2.attn.qkv.weight" in shallow.state_dict()
    assert "blocks.3.attn.qkv.weight" not in shallow.state_dict()


class ToyTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(int(token_id)) for token_id in token_ids)


class ToySentenceEncoder:
    embedding_dim = 8

    def __init__(self):
        self.seen_texts = None

    def encode(self, texts, device, dtype):
        self.seen_texts = list(texts)
        rows = []
        for text in self.seen_texts:
            base = float(sum(ord(ch) for ch in text) % 17)
            rows.append(torch.arange(8, device=device, dtype=dtype) + base)
        return torch.stack(rows, dim=0)


class SyncTrackingWrapper(nn.Module):
    """Minimal DDP-like wrapper that records the sync flag seen by forward."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True
        self.forward_sync_flags = []

    def forward(self, *args, **kwargs):
        self.forward_sync_flags.append(self.require_backward_grad_sync)
        return self.module(*args, **kwargs)


class MainForwardCapture(nn.Module):
    """Capture the loss-bearing mixed forward without changing model behavior."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.main_call = None

    def forward(self, *args, **kwargs):
        if kwargs.get("decoder_step_active") is not None:
            self.main_call = {
                "token_t": args[1].detach().clone(),
                "plan_t": kwargs["plan_t"].detach().clone(),
                "plan_z": kwargs["plan_z"].detach().clone(),
            }
        return self.module(*args, **kwargs)


def tiny_batch():
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 0, 0],
            [5, 6, 7, 8, 9, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = (input_ids != 0).float()
    cond_seq_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    encoder_attention_mask = attention_mask[:, None, :] * attention_mask[:, :, None]
    return {
        "input_ids": input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "cond_seq_mask": cond_seq_mask,
        "attention_mask": attention_mask,
        "label_drop_mask": torch.zeros(input_ids.shape[0], dtype=torch.bool),
    }


def tiny_config(**overrides):
    cfg = SimpleNamespace(
        use_bf16=False,
        t_eps=5e-2,
        self_cond_prob=0.0,
        latent_mean=0.0,
        latent_std=1.0,
        decoder_prob=0.0,
        decoder_noise_scale=1.0,
        pad_token="pad",
        label_drop_prob=0.0,
        num_self_cond_cfg_tokens=1,
        self_cond_cfg_min=0.5,
        self_cond_cfg_max=2.0,
        denoiser_p_mean=0.8,
        denoiser_p_std=0.8,
        denoiser_noise_scale=1.0,
        time_schedule="logit_normal",
        decoder_p_mean=0.8,
        decoder_p_std=0.8,
        use_sentence_plan=True,
        sentence_encoder_type="learned",
        sentence_emb_dim=8,
        plan_noise_scale=1.0,
        plan_loss_weight=1.0,
        sentence_encoder_grad="none",
        plan_aux_passes=1,
        plan_aux_token_context="denoiser_z",
        plan_attention_topology="joint",
        grad_accum_steps=1,
        ema_decay1=0.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def tiny_model(
    sentence_encoder_type="learned",
    use_sentence_plan=True,
    num_self_cond_cfg_tokens=1,
    plan_adapter_type="slot_mlp",
    plan_denoiser_type="shared",
    plan_attention_topology="joint",
    plan_denoiser_conditioning="none",
):
    return ELF(
        text_encoder_dim=4,
        max_length=6,
        hidden_size=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        bottleneck_dim=8,
        num_time_tokens=1,
        num_self_cond_cfg_tokens=num_self_cond_cfg_tokens,
        num_model_mode_tokens=1,
        vocab_size=32,
        use_sentence_plan=use_sentence_plan,
        sentence_encoder_type=sentence_encoder_type,
        sentence_emb_dim=8,
        num_plan_tokens=4,
        plan_adapter_type=plan_adapter_type,
        plan_slot_dit_depth=1,
        plan_denoiser_type=plan_denoiser_type,
        plan_denoiser_depth=1,
        plan_denoiser_conditioning=plan_denoiser_conditioning,
        plan_attention_topology=plan_attention_topology,
    )


def test_hierarchical_attention_mask_blocks_all_upstream_future_reads():
    mask = ELF.build_hierarchical_attention_mask(
        field_attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.float32),
        cond_seq_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        upstream_token_count=2,
    )

    # Internal + observed-prefix queries cannot read the one valid future key.
    assert not mask[0, :4, 4].any()
    # The padded final key is unavailable to every query.
    assert not mask[0, :, 5].any()
    # A future query can read internal, observed-prefix, and future keys.
    assert mask[0, 4, :5].all()


def test_strict_hierarchical_attention_mask_orders_prefix_plan_and_future():
    mask = ELF.build_strict_hierarchical_attention_mask(
        field_attention_mask=torch.tensor([[1, 1, 1, 0]], dtype=torch.float32),
        cond_seq_mask=torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        control_token_count=2,
        plan_token_count=1,
    )

    # Layout: two controls, one plan, two observed-prefix rows, one valid
    # future row, then one padded field row.
    assert mask.shape == (1, 7, 7)
    assert mask[0, 0, :2].all()
    assert not mask[0, 0, 2:].any()
    assert mask[0, 3, [0, 1, 3, 4]].all()
    assert not mask[0, 3, [2, 5, 6]].any()
    assert mask[0, 2, :5].all()
    assert not mask[0, 2, 5:].any()
    assert mask[0, 5, :6].all()
    assert not mask[0, :, 6].any()


def test_hierarchical_shared_plan_depends_on_prefix_not_future_field():
    torch.manual_seed(2031)
    model = tiny_model(
        sentence_encoder_type="sentence_t5",
        plan_attention_topology="hierarchical_prefix",
    ).eval()
    plan_z = torch.randn(2, 8)
    plan_t = torch.tensor([0.2, 0.7])
    token_t = torch.tensor([0.3, 0.6])
    cond_seq_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32,
    )
    attention_mask = torch.ones_like(cond_seq_mask)
    x_base = torch.randn(2, 6, 4)
    x_future_changed = x_base.clone()
    x_future_changed[:, 2:] = torch.randn_like(x_future_changed[:, 2:])
    x_prefix_changed = x_base.clone()
    x_prefix_changed[:, :2] = torch.randn_like(x_prefix_changed[:, :2])

    common = {
        "t": token_t,
        "attention_mask": attention_mask,
        "cond_seq_mask": cond_seq_mask,
        "plan_z": plan_z,
        "plan_t": plan_t,
        "self_cond_cfg_scale": torch.ones(2),
        "return_plan": True,
        "deterministic": True,
    }
    _, _, plan_base = model(x_base, **common)
    _, _, plan_future_changed = model(x_future_changed, **common)
    _, _, plan_prefix_changed = model(x_prefix_changed, **common)

    assert torch.equal(plan_base, plan_future_changed)
    assert not torch.equal(plan_base, plan_prefix_changed)


def test_strict_hierarchical_shared_denoiser_has_no_plan_to_prefix_feedback():
    torch.manual_seed(2032)
    model = tiny_model(
        sentence_encoder_type="sentence_t5",
        plan_attention_topology="strict_hierarchical_prefix",
    ).eval()
    x = torch.randn(2, 6, 4)
    plan_z = torch.randn(2, 8)
    changed_plan_z = torch.randn_like(plan_z)
    common = {
        "t": torch.tensor([0.3, 0.6]),
        "attention_mask": torch.ones(2, 6),
        "cond_seq_mask": torch.tensor(
            [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32,
        ),
        "plan_t": torch.tensor([0.2, 0.7]),
        "self_cond_cfg_scale": torch.ones(2),
        "decoder_step_active": True,
        "return_plan": True,
        "deterministic": True,
    }

    _, logits_base, plan_base = model(x, plan_z=plan_z, **common)
    _, logits_changed, plan_changed = model(x, plan_z=changed_plan_z, **common)

    assert torch.equal(logits_base[:, :2], logits_changed[:, :2])
    assert not torch.equal(plan_base, plan_changed)
    assert not torch.equal(logits_base[:, 2:], logits_changed[:, 2:])


def test_hierarchical_train_step_runs_with_prefix_mask_plumbed():
    model = tiny_model(plan_attention_topology="hierarchical_prefix")
    config = tiny_config(plan_attention_topology="hierarchical_prefix")

    _, metrics = run_tiny_train_step(config, model=model)

    assert torch.isfinite(metrics["loss"])
    assert torch.isfinite(metrics["plan_loss"])


def test_strict_hierarchical_train_step_runs_with_prefix_mask_plumbed():
    topology = "strict_hierarchical_prefix"
    model = tiny_model(plan_attention_topology=topology)
    config = tiny_config(plan_attention_topology=topology)

    _, metrics = run_tiny_train_step(config, model=model)

    assert torch.isfinite(metrics["loss"])
    assert torch.isfinite(metrics["plan_loss"])


def test_plan_first_training_plan_phase_freezes_token_and_masks_token_loss():
    captured = MainForwardCapture(tiny_model(
        sentence_encoder_type="sentence_t5",
        num_self_cond_cfg_tokens=0,
        plan_attention_topology="hierarchical_prefix",
    ))
    config = tiny_config(
        sentence_encoder_type="sentence_t5",
        num_self_cond_cfg_tokens=0,
        plan_attention_topology="hierarchical_prefix",
        plan_training_mode="plan_first",
        plan_first_plan_phase_prob=1.0,
    )

    _, metrics = run_tiny_train_step(
        config,
        model=captured,
        tokenizer=ToyTokenizer(),
        sentence_encoder=ToySentenceEncoder(),
    )

    assert captured.main_call is not None
    assert torch.equal(captured.main_call["token_t"], torch.zeros(2))
    assert torch.all(captured.main_call["plan_t"] > 0)
    assert torch.all(captured.main_call["plan_t"] < 1)
    assert metrics["l2_token_count"].item() == pytest.approx(0.0)
    assert metrics["plan_loss"].item() > 0
    assert metrics["plan_phase_fraction"].item() == pytest.approx(1.0)
    assert metrics["token_phase_fraction"].item() == pytest.approx(0.0)


def test_plan_first_training_token_phase_uses_completed_plan_and_masks_plan_loss():
    captured = MainForwardCapture(tiny_model(
        sentence_encoder_type="sentence_t5",
        num_self_cond_cfg_tokens=0,
        plan_attention_topology="hierarchical_prefix",
    ))
    config = tiny_config(
        sentence_encoder_type="sentence_t5",
        num_self_cond_cfg_tokens=0,
        plan_attention_topology="hierarchical_prefix",
        plan_training_mode="plan_first",
        plan_first_plan_phase_prob=0.0,
    )

    _, metrics = run_tiny_train_step(
        config,
        model=captured,
        tokenizer=ToyTokenizer(),
        sentence_encoder=ToySentenceEncoder(),
    )

    assert captured.main_call is not None
    assert torch.all(captured.main_call["token_t"] > 0)
    assert torch.all(captured.main_call["token_t"] < 1)
    assert torch.equal(captured.main_call["plan_t"], torch.ones(2))
    assert metrics["l2_token_count"].item() > 0
    assert metrics["plan_loss"].item() == pytest.approx(0.0)
    assert metrics["plan_phase_fraction"].item() == pytest.approx(0.0)
    assert metrics["token_phase_fraction"].item() == pytest.approx(1.0)


def test_oracle_training_always_uses_clean_plan_and_trains_all_denoiser_rows():
    captured = MainForwardCapture(tiny_model(
        sentence_encoder_type="sentence_t5",
        num_self_cond_cfg_tokens=0,
        plan_attention_topology="hierarchical_prefix",
    ))
    config = tiny_config(
        sentence_encoder_type="sentence_t5",
        num_self_cond_cfg_tokens=0,
        plan_attention_topology="hierarchical_prefix",
        plan_training_mode="oracle",
        plan_loss_weight=0.0,
    )

    _, metrics = run_tiny_train_step(
        config,
        model=captured,
        tokenizer=ToyTokenizer(),
        sentence_encoder=ToySentenceEncoder(),
    )

    assert captured.main_call is not None
    assert torch.all(captured.main_call["token_t"] > 0)
    assert torch.all(captured.main_call["token_t"] < 1)
    assert torch.equal(captured.main_call["plan_t"], torch.ones(2))
    assert metrics["l2_token_count"].item() > 0
    assert metrics["plan_loss"].item() == pytest.approx(0.0)
    assert metrics["plan_phase_fraction"].item() == pytest.approx(0.0)
    assert metrics["token_phase_fraction"].item() == pytest.approx(1.0)


def train_state(model):
    return TrainState(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        lr_scheduler=None,
        ema_params1=TrainState.init_ema(model),
        dropout_generator=torch.Generator(device="cpu").manual_seed(7),
    )


def run_tiny_train_step(config, model=None, tokenizer=None, sentence_encoder=None):
    torch.manual_seed(123)
    model = model if model is not None else tiny_model(config.sentence_encoder_type)
    state = train_state(model)
    return train_step(
        state,
        encoder=TinyEncoder(),
        batch=tiny_batch(),
        config=config,
        tokenizer=tokenizer,
        sentence_encoder=sentence_encoder,
    )


def plan_encoder_query_grad_norm(model):
    grad = model.plan_encoder_query.grad
    if grad is None:
        return 0.0
    return float(grad.detach().norm().item())


def sentence_plan_mse_encoder_grad_norm(grad_mode):
    torch.manual_seed(2024)
    model = tiny_model(sentence_encoder_type="learned")
    model.zero_grad(set_to_none=True)

    x0 = torch.randn(2, 6, 4)
    attention_mask = torch.ones(2, 6)
    t_encode = torch.ones(2)
    self_cond_cfg_scale = torch.ones(2)

    _, _, s0 = model(
        x0,
        t_encode,
        attention_mask=attention_mask,
        deterministic=True,
        self_cond_cfg_scale=self_cond_cfg_scale,
        learned_plan_encode=True,
        return_plan=True,
    )

    t = torch.tensor([0.25, 0.75])
    plan_noise = torch.randn_like(s0)
    plan_z = t.reshape(-1, 1) * s0 + (1.0 - t.reshape(-1, 1)) * plan_noise
    _, _, plan_pred = model(
        x0,
        t,
        attention_mask=attention_mask,
        deterministic=True,
        self_cond_cfg_scale=self_cond_cfg_scale,
        plan_z=plan_z,
        plan_t=t,
        return_plan=True,
    )

    target = s0 if grad_mode == "full" else s0.detach()
    plan_loss = ((plan_pred - target) ** 2).mean()
    if grad_mode == "none":
        objective = plan_loss.detach() + 0.0 * s0.sum()
    else:
        objective = plan_loss
    objective.backward()
    return plan_encoder_query_grad_norm(model)


def test_decode_continuation_texts_uses_loss_mask_only():
    input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    loss_mask = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.float32)

    texts = _decode_continuation_texts(input_ids, loss_mask, ToyTokenizer())

    assert texts == ["2 3", "4"]


def test_independent_plan_denoiser_output_does_not_depend_on_token_field():
    torch.manual_seed(2026)
    model = tiny_model(
        sentence_encoder_type="sentence_t5",
        plan_denoiser_type="independent",
    ).eval()
    plan_z = torch.randn(2, 8)
    plan_t = torch.tensor([0.2, 0.7])
    token_t = torch.tensor([0.3, 0.6])
    self_cond_cfg_scale = torch.ones(2)
    x_a = torch.randn(2, 6, 4)
    x_b = torch.randn(2, 6, 4)

    _, _, plan_a = model(
        x_a, token_t, plan_z=plan_z, plan_t=plan_t,
        self_cond_cfg_scale=self_cond_cfg_scale,
        return_plan=True, deterministic=True,
    )
    _, _, plan_b = model(
        x_b, token_t, plan_z=plan_z, plan_t=plan_t,
        self_cond_cfg_scale=self_cond_cfg_scale,
        return_plan=True, deterministic=True,
    )

    assert torch.equal(plan_a, plan_b)


def test_prefix_conditioned_plan_denoiser_reads_prefix_but_not_future():
    torch.manual_seed(2028)
    model = tiny_model(
        sentence_encoder_type="sentence_t5",
        plan_denoiser_type="independent",
        plan_denoiser_conditioning="prefix",
        plan_attention_topology="hierarchical_prefix",
    ).eval()
    plan_z = torch.randn(2, 8)
    plan_t = torch.tensor([0.2, 0.7])
    token_t = torch.tensor([0.3, 0.6])
    cond = torch.tensor([[1, 1, 0, 0, 0, 0]] * 2, dtype=torch.float32)
    valid = torch.ones_like(cond)
    x = torch.randn(2, 6, 4)
    changed_prefix = x.clone(); changed_prefix[:, :2] += 3.0
    changed_future = x.clone(); changed_future[:, 2:] += 3.0

    def predict(value):
        return model(
            value, token_t, plan_z=plan_z, plan_t=plan_t,
            attention_mask=valid, cond_seq_mask=cond,
            self_cond_cfg_scale=torch.ones(2), return_plan=True,
            deterministic=True,
        )[2]

    baseline = predict(x)
    assert not torch.equal(baseline, predict(changed_prefix))
    torch.testing.assert_close(baseline, predict(changed_future), rtol=0, atol=0)


def test_independent_plan_loss_does_not_train_shared_token_trunk():
    torch.manual_seed(2027)
    model = tiny_model(
        sentence_encoder_type="sentence_t5",
        plan_denoiser_type="independent",
    )
    plan_z = torch.randn(2, 8)
    target = torch.randn(2, 8)
    _, _, plan_pred = model(
        torch.randn(2, 6, 4),
        torch.tensor([0.3, 0.6]),
        plan_z=plan_z,
        plan_t=torch.tensor([0.2, 0.7]),
        self_cond_cfg_scale=torch.ones(2),
        return_plan=True,
        deterministic=True,
    )

    ((plan_pred - target) ** 2).mean().backward()

    independent_grad = model.independent_plan_denoiser.plan_out.weight.grad
    shared_grad = model.blocks[0].attn.qkv.weight.grad
    assert independent_grad is not None
    assert independent_grad.norm().item() > 0
    assert shared_grad is None or shared_grad.norm().item() == pytest.approx(0.0)


def test_independent_sentence_t5_replaces_and_freezes_shared_plan_readout():
    model = tiny_model(
        sentence_encoder_type="sentence_t5",
        plan_denoiser_type="independent",
    )

    assert all(not parameter.requires_grad for parameter in model.plan_out.parameters())
    assert all(not parameter.requires_grad for parameter in model.plan_norm.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.independent_plan_denoiser.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.blocks.parameters())


@pytest.mark.parametrize(
    ("grad_mode", "should_reach_encoder"),
    [
        ("none", False),
        ("detached_target", True),
        ("full", True),
    ],
)
def test_sentence_plan_mse_gradient_topology(grad_mode, should_reach_encoder):
    grad_norm = sentence_plan_mse_encoder_grad_norm(grad_mode)

    if should_reach_encoder:
        assert grad_norm > 0
    else:
        assert grad_norm == pytest.approx(0.0)


def test_ce_loss_can_train_learned_sentence_encoder_under_grad_none():
    model = tiny_model(sentence_encoder_type="learned")
    cfg = tiny_config(
        decoder_prob=1.0,
        grad_accum_steps=2,
        plan_loss_weight=0.0,
        plan_aux_passes=0,
        sentence_encoder_grad="none",
    )

    state, metrics = run_tiny_train_step(cfg, model=model)

    assert state.step == 1
    assert metrics["ce_loss"].item() > 0
    assert plan_encoder_query_grad_norm(model) > 0


def test_grad_none_aux0_keeps_plan_head_in_backward_with_zero_grad():
    model = tiny_model(sentence_encoder_type="learned")
    _, metrics = run_tiny_train_step(
        tiny_config(
            sentence_encoder_grad="none",
            plan_aux_passes=0,
            grad_accum_steps=2,
        ),
        model=model,
    )

    assert metrics["plan_loss"].item() > 0
    assert metrics["plan_aux_loss"].item() == pytest.approx(0.0)
    for param in model.plan_out.parameters():
        assert param.grad is not None
        assert torch.count_nonzero(param.grad).item() == 0


def test_gradient_accumulation_updates_only_at_window_boundary():
    torch.manual_seed(123)
    model = tiny_model()
    state = train_state(model)
    cfg = tiny_config(grad_accum_steps=2)
    encoder = TinyEncoder()
    initial = {name: param.detach().clone() for name, param in model.named_parameters()}

    state, first_metrics = train_step(state, encoder, tiny_batch(), cfg)
    assert state.accum_step == 1
    assert state.optimizer_step == 0
    assert not bool(first_metrics["did_optimizer_step"])
    assert all(torch.equal(param, initial[name]) for name, param in model.named_parameters())

    state, second_metrics = train_step(state, encoder, tiny_batch(), cfg)
    assert state.accum_step == 0
    assert state.micro_step == 2
    assert state.optimizer_step == 1
    assert bool(second_metrics["did_optimizer_step"])
    assert any(not torch.equal(param, initial[name]) for name, param in model.named_parameters())


def test_final_partial_accumulation_window_is_flushed():
    torch.manual_seed(123)
    model = tiny_model()
    state = train_state(model)
    initial = {name: param.detach().clone() for name, param in model.named_parameters()}

    state, metrics = train_step(
        state,
        TinyEncoder(),
        tiny_batch(),
        tiny_config(grad_accum_steps=4),
        force_optimizer_step=True,
    )

    assert state.accum_step == 0
    assert state.optimizer_step == 1
    assert bool(metrics["did_optimizer_step"])
    assert any(not torch.equal(param, initial[name]) for name, param in model.named_parameters())


def test_partial_window_matches_equivalent_complete_window_normalization():
    torch.manual_seed(321)
    base_model = tiny_model()
    base_encoder = TinyEncoder()
    complete_model = copy.deepcopy(base_model)
    partial_model = copy.deepcopy(base_model)
    complete_encoder = copy.deepcopy(base_encoder)
    partial_encoder = copy.deepcopy(base_encoder)

    torch.manual_seed(777)
    complete_state = train_state(complete_model)
    complete_cfg = tiny_config(grad_accum_steps=2)
    complete_state, _ = train_step(complete_state, complete_encoder, tiny_batch(), complete_cfg)
    complete_state, _ = train_step(complete_state, complete_encoder, tiny_batch(), complete_cfg)

    torch.manual_seed(777)
    partial_state = train_state(partial_model)
    partial_cfg = tiny_config(grad_accum_steps=4)
    partial_state, _ = train_step(partial_state, partial_encoder, tiny_batch(), partial_cfg)
    partial_state, _ = train_step(
        partial_state,
        partial_encoder,
        tiny_batch(),
        partial_cfg,
        force_optimizer_step=True,
    )

    for complete_param, partial_param in zip(complete_model.parameters(), partial_model.parameters()):
        assert torch.allclose(complete_param, partial_param, atol=1e-7, rtol=1e-6)


def test_ddp_sync_decision_is_visible_during_forward():
    torch.manual_seed(123)
    wrapped = SyncTrackingWrapper(tiny_model())
    state = train_state(wrapped)
    cfg = tiny_config(grad_accum_steps=2)
    encoder = TinyEncoder()

    state, _ = train_step(state, encoder, tiny_batch(), cfg)
    assert wrapped.forward_sync_flags
    assert set(wrapped.forward_sync_flags) == {False}
    assert wrapped.require_backward_grad_sync is True

    wrapped.forward_sync_flags.clear()
    state, _ = train_step(state, encoder, tiny_batch(), cfg)
    assert wrapped.forward_sync_flags
    assert set(wrapped.forward_sync_flags) == {True}


@pytest.mark.parametrize("context", ["denoiser_z", "resampled_z", "mixed_z", "clean_x0"])
@pytest.mark.parametrize("adapter", ["slot_mlp", "slot_dit"])
def test_learned_plan_aux_contexts_run_and_log_metrics(context, adapter):
    _, metrics = run_tiny_train_step(
        tiny_config(plan_aux_token_context=context),
        model=tiny_model(plan_adapter_type=adapter),
    )

    assert metrics["plan_loss"].item() > 0
    assert metrics["plan_aux_loss"].item() > 0
    assert metrics["plan_emb_batch_var"].item() >= 0
    assert metrics["plan_emb_norm"].item() > 0
    assert metrics["plan_pred_batch_var"].item() >= 0
    assert metrics["plan_pred_norm"].item() > 0


def test_learned_plan_aux_passes_zero_disables_aux_loss():
    _, metrics = run_tiny_train_step(tiny_config(plan_aux_passes=0))

    assert metrics["plan_loss"].item() > 0
    assert metrics["plan_aux_loss"].item() == pytest.approx(0.0)


def test_invalid_plan_aux_token_context_raises():
    with pytest.raises(ValueError, match="plan_aux_token_context"):
        run_tiny_train_step(tiny_config(plan_aux_token_context="bad_context"))


@pytest.mark.parametrize("plan_denoiser_type", ["shared", "independent"])
def test_sentence_t5_plan_uses_decoded_continuation_texts(plan_denoiser_type):
    sentence_encoder = ToySentenceEncoder()
    _, metrics = run_tiny_train_step(
        tiny_config(sentence_encoder_type="sentence_t5", plan_aux_passes=4),
        model=tiny_model(
            sentence_encoder_type="sentence_t5",
            plan_denoiser_type=plan_denoiser_type,
        ),
        tokenizer=ToyTokenizer(),
        sentence_encoder=sentence_encoder,
    )

    assert sentence_encoder.seen_texts == ["3 4", "6 7 8 9"]
    assert metrics["plan_loss"].item() > 0
    assert metrics["plan_aux_loss"].item() == pytest.approx(0.0)


def test_train_step_without_sentence_plan_keeps_zero_plan_metrics():
    _, metrics = run_tiny_train_step(
        tiny_config(
            use_sentence_plan=False,
            num_self_cond_cfg_tokens=0,
            sentence_encoder_type="learned",
        ),
        model=tiny_model(use_sentence_plan=False, num_self_cond_cfg_tokens=0),
    )

    assert metrics["plan_loss"].item() == pytest.approx(0.0)
    assert metrics["plan_aux_loss"].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("decoder_prob", "active_count", "inactive_count"),
    [
        (1.0, "ce_token_count", "l2_token_count"),
        (0.0, "l2_token_count", "ce_token_count"),
    ],
)
def test_branch_metrics_expose_token_weighted_numerators(decoder_prob, active_count, inactive_count):
    _, metrics = run_tiny_train_step(
        tiny_config(
            decoder_prob=decoder_prob,
            use_sentence_plan=False,
            sentence_encoder_type="learned",
            num_self_cond_cfg_tokens=0,
        ),
        model=tiny_model(use_sentence_plan=False, num_self_cond_cfg_tokens=0),
    )

    assert metrics[active_count].item() > 0
    assert metrics[inactive_count].item() == pytest.approx(0.0)
    assert metrics["plan_emb_batch_var"].item() == pytest.approx(0.0)
