import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from modules.model import ELF
from train_step import _decode_continuation_texts, train_step
from utils.train_utils import TrainState


class TinyEncoder(nn.Module):
    def __init__(self, vocab_size=32, dim=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, input_ids, attention_mask=None, deterministic=True):
        return self.emb(input_ids)


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
    )


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


def test_sentence_t5_plan_uses_decoded_continuation_texts():
    sentence_encoder = ToySentenceEncoder()
    _, metrics = run_tiny_train_step(
        tiny_config(sentence_encoder_type="sentence_t5", plan_aux_passes=4),
        model=tiny_model(sentence_encoder_type="sentence_t5"),
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
    assert metrics["plan_emb_batch_var"].item() == pytest.approx(0.0)
