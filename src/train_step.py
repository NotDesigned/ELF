"""One mini-batch forward/backward for the ELF diffusion language model.

Each example in the batch independently picks the decoder (CE) or denoiser
(L2) branch via a Bernoulli draw at `decoder_prob`. A single forward consumes
a mixed input (decoder_z for decoder rows, denoiser_z for denoiser rows) and
both heads run; the CE / L2 losses are then masked to their respective rows
and combined with a single denominator. Self-conditioning + CFG guidance is
applied on the denoiser branch only.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train_utils import TrainState, ema_update
from utils.encoder_utils import encode_text
from utils.sampling_utils import (
    sample_cfg_scale, add_noise, sample_timesteps,
    net_out_to_v_x, restore_cond, plan_time_from_token_time,
)


def _trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _decode_continuation_texts(input_ids: torch.Tensor, loss_mask: torch.Tensor, tokenizer) -> list:
    """Decode target/continuation tokens selected by loss_mask for Sentence-T5."""
    if tokenizer is None:
        raise ValueError("tokenizer is required for sentence_encoder_type='sentence_t5'")
    ids_cpu = input_ids.detach().cpu()
    mask_cpu = loss_mask.detach().cpu().bool()
    texts = []
    for ids_row, mask_row in zip(ids_cpu, mask_cpu):
        continuation_ids = ids_row[mask_row].tolist()
        texts.append(tokenizer.decode(continuation_ids, skip_special_tokens=True))
    return texts


def train_step(
    state: TrainState,
    encoder: nn.Module,
    batch: Dict[str, torch.Tensor],
    config,
    tokenizer=None,
    sentence_encoder=None,
    force_optimizer_step: bool = False,
) -> Tuple[TrainState, Dict[str, float]]:
    """Perform one microstep and, when due, one optimizer update."""
    model = state.model
    accum_steps = max(int(config.grad_accum_steps), 1)
    accumulated = int(getattr(state, "accum_step", 0))
    is_optimizer_step = accumulated + 1 >= accum_steps or force_optimizer_step
    accumulation_divisor = accumulated + 1 if force_optimizer_step and accumulated + 1 < accum_steps else accum_steps

    # Earlier microsteps were normalized by the full accumulation window. If
    # an epoch ends early, renormalize their resident gradients to the shorter
    # final window before adding its last microbatch.
    if is_optimizer_step and accumulation_divisor < accum_steps and accumulated > 0:
        correction = accum_steps / accumulation_divisor
        for param in _trainable_params(model):
            if param.grad is not None:
                param.grad.mul_(correction)

    # DDP decides whether to install reduction hooks during forward, so this
    # flag must be set before every model forward, not only around backward.
    previous_sync_setting = getattr(model, "require_backward_grad_sync", None)
    if previous_sync_setting is not None:
        model.require_backward_grad_sync = is_optimizer_step

    device = next(state.model.parameters()).device
    dtype = next(state.model.parameters()).dtype
    use_bf16 = bool(getattr(config, "use_bf16", True)) and device.type == "cuda"
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    latent_mean, latent_std = config.latent_mean, config.latent_std
    decoder_prob = config.decoder_prob
    decoder_noise_scale = config.decoder_noise_scale
    use_sentence_plan = bool(getattr(config, "use_sentence_plan", False))
    sentence_encoder_type = getattr(config, "sentence_encoder_type", "sentence_t5")

    gen = state.dropout_generator

    # encoder_attention_mask: cond sees cond, x sees all
    input_ids = batch["input_ids"].to(device, non_blocking=True).long()
    encoder_attention_mask = batch["encoder_attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
    cond_seq_mask = batch["cond_seq_mask"].to(device, dtype=torch.float32, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
    label_drop_mask = batch.get("label_drop_mask",
                                torch.zeros((input_ids.shape[0],), dtype=torch.bool)).to(device, non_blocking=True)

    # Label drop before encoding: prevent target tokens from attending to
    # condition tokens so x0 is truly unconditional for dropped samples.
    if config.label_drop_prob > 0:
        drop = label_drop_mask.to(dtype=torch.float32).reshape(-1, 1, 1)  # (B, 1, 1)
        cond_mask = cond_seq_mask  # (B, S)
        # block_mask is 1 only at (non-cond row, cond col) — leaves cond↔cond unchanged
        block_mask = (1 - cond_mask).unsqueeze(-1) * cond_mask.unsqueeze(1)
        encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)

    x0 = encode_text(
        input_ids=input_ids,
        attention_mask=encoder_attention_mask,
        encoder=encoder,
        latent_mean=latent_mean,
        latent_std=latent_std,
        use_bf16=use_bf16,
    ).to(dtype)

    batch_size, seq_length = x0.shape[0], x0.shape[1]

    t = sample_timesteps(
        batch_size,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
        device=device, dtype=dtype,
    )

    noise = torch.randn(x0.shape, dtype=dtype, device=device)

    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = torch.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - cond_seq_mask)

    plan_attention_kwargs = {}
    if str(getattr(config, "plan_attention_topology", "joint")) == "hierarchical_prefix":
        topology_cond_seq_mask = cond_seq_mask
        if config.label_drop_prob > 0:
            topology_cond_seq_mask = topology_cond_seq_mask * (
                ~label_drop_mask
            ).to(topology_cond_seq_mask.dtype).unsqueeze(1)
        plan_attention_kwargs = {
            "attention_mask": attention_mask,
            "cond_seq_mask": topology_cond_seq_mask,
        }

    cond_seq_mask = cond_seq_mask.unsqueeze(-1)  # (B, S, 1)

    denoiser_z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask)

    drop = label_drop_mask.unsqueeze(1)  # (B, 1)
    if config.label_drop_prob > 0:
        denoiser_z = torch.where(drop.unsqueeze(-1) & (cond_seq_mask > 0), torch.zeros_like(denoiser_z), denoiser_z)
        x0 = torch.where(drop.unsqueeze(-1) & (cond_seq_mask > 0), torch.zeros_like(x0), x0)

    decoder_targets = input_ids  # (B, S)

    # Per-example branching: each example independently picks decoder (CE) vs.
    # denoiser (L2) instead of one scalar bernoulli per step. Smooths training
    decoder_step_active = torch.bernoulli(
        torch.full((batch_size,), decoder_prob, dtype=torch.float32),
        generator=gen,
    ).to(device=device, dtype=dtype)  # (B,) — 1.0 = decoder mode, 0.0 = denoiser
    decoder_mask_B11 = decoder_step_active.view(-1, 1, 1)
    decoder_mask_B1 = decoder_step_active.view(-1, 1)

    # Decoder-branch input: logit-normal-noised latent (decoder_z) at t=1
    decoder_z_vals = (
        torch.randn((batch_size * seq_length,), dtype=dtype, device=device)
        * config.decoder_p_std + config.decoder_p_mean
    )
    decoder_lambda_t = torch.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
    decoder_noise = torch.randn(x0.shape, dtype=dtype, device=device) * decoder_noise_scale
    decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise

    t_expanded = t.reshape(-1, 1, 1)
    v_target = (x0 - denoiser_z) / torch.clamp(1 - t_expanded, min=t_eps)

    if self_cond_prob > 0:
        use_self_cond_mask = (
            (torch.rand((batch_size,), dtype=dtype, device=device) < self_cond_prob)
            .reshape(-1, 1, 1).to(dtype)
        )
    else:
        use_self_cond_mask = None

    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(
            batch_size,
            cfg_min=config.self_cond_cfg_min, cfg_max=config.self_cond_cfg_max,
            dtype=dtype, device=device,
        )
    else:
        self_cond_cfg_scale = None

    plan_z_denoiser = None
    plan_z_mixed = None
    plan_t_denoiser = None
    plan_t_mixed = None
    plan_target = None
    plan_loss = x0.new_tensor(0.0)
    plan_aux_loss = x0.new_tensor(0.0)
    plan_loss_for_backward = x0.new_tensor(0.0)
    plan_emb_batch_var = x0.new_tensor(0.0)
    plan_emb_norm = x0.new_tensor(0.0)
    plan_pred_batch_var = x0.new_tensor(0.0)
    plan_pred_norm = x0.new_tensor(0.0)
    grad_mode = getattr(config, "sentence_encoder_grad", "none")
    plan_aux_passes = int(getattr(config, "plan_aux_passes", 1))
    if plan_aux_passes < 0:
        raise ValueError("plan_aux_passes must be >= 0")
    plan_aux_token_context = str(getattr(config, "plan_aux_token_context", "denoiser_z")).lower()
    valid_aux_contexts = {"denoiser_z", "resampled_z", "mixed_z", "clean_x0"}
    if plan_aux_token_context not in valid_aux_contexts:
        raise ValueError(
            "plan_aux_token_context must be one of "
            f"{sorted(valid_aux_contexts)}, got {plan_aux_token_context!r}"
        )
    if use_sentence_plan:
        if sentence_encoder_type == "sentence_t5":
            if sentence_encoder is None:
                raise ValueError("sentence_encoder is required for sentence_encoder_type='sentence_t5'")
            continuation_texts = _decode_continuation_texts(input_ids, loss_mask, tokenizer)
            s0 = sentence_encoder.encode(continuation_texts, device=device, dtype=dtype)
        elif sentence_encoder_type == "learned":
            encoder_sc_cfg_scale = (
                torch.ones_like(t) if config.num_self_cond_cfg_tokens > 0 else None
            )
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                _, _, s0 = model(
                    x0, torch.ones_like(t),
                    attention_mask=loss_mask,
                    deterministic=True,
                    self_cond_cfg_scale=encoder_sc_cfg_scale,
                    learned_plan_encode=True,
                    return_plan=True,
                )
            s0 = s0.to(dtype)
        else:
            raise ValueError(f"Unknown sentence_encoder_type: {sentence_encoder_type}")

        expected_dim = int(getattr(config, "sentence_emb_dim", s0.shape[-1]))
        if s0.shape[-1] != expected_dim:
            raise ValueError(f"Sentence embedding dim {s0.shape[-1]} does not match sentence_emb_dim={expected_dim}")

        if grad_mode not in {"none", "detached_target", "full"}:
            raise ValueError("sentence_encoder_grad must be 'none', 'detached_target', or 'full'")
        plan_target = s0 if grad_mode == "full" else s0.detach()
        s0_detached_for_metrics = s0.detach().float()
        plan_emb_batch_var = s0_detached_for_metrics.var(dim=0, unbiased=False).mean()
        plan_emb_norm = s0_detached_for_metrics.norm(dim=-1).mean()

        plan_noise = torch.randn_like(s0) * float(getattr(config, "plan_noise_scale", 1.0))
        plan_t_denoiser = plan_time_from_token_time(t, config)
        plan_t_expanded = plan_t_denoiser.reshape(-1, 1)
        plan_z_denoiser = plan_t_expanded * s0 + (1.0 - plan_t_expanded) * plan_noise
        plan_z_mixed = decoder_mask_B1 * s0 + (1.0 - decoder_mask_B1) * plan_z_denoiser

    def compute_shared_uncond(z, t_input, x_tokens, plan_z_input=None, plan_t_input=None):
        """Unconditional forward shared by self-cond-init and sc-cfg-uncond."""
        z_uncond = restore_cond(torch.zeros_like(z), x_tokens, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        plan_kwargs = {}
        if use_sentence_plan:
            plan_kwargs = {"plan_z": plan_z_input, "plan_t": t_input if plan_t_input is None else plan_t_input}
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            net_out_uncond = model(
                z_input_uncond, t_input,
                deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale,
                **plan_attention_kwargs, **plan_kwargs,
            )
        return net_out_uncond

    def get_sc_cond_and_uncond(
        z, t_input, cond_mask, x_tokens, shared_net_out_uncond,
        plan_z_input=None, plan_t_input=None,
    ):
        plan_kwargs = {}
        if use_sentence_plan:
            plan_kwargs = {"plan_z": plan_z_input, "plan_t": t_input if plan_t_input is None else plan_t_input}
        if config.self_cond_prob == 0:
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                net_out_uncond = model(
                    z, t_input,
                    deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale,
                    **plan_attention_kwargs, **plan_kwargs,
                )
            v_uncond, _ = net_out_to_v_x(net_out_uncond, z, t_input, t_eps)
            return v_uncond, v_uncond

        v_uncond, x_uncond = net_out_to_v_x(shared_net_out_uncond, z, t_input, t_eps)
        x_uncond = restore_cond(x_uncond, x_tokens, cond_mask)

        z_input_cond = torch.cat([z, x_uncond], dim=-1)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            net_out_cond = model(
                z_input_cond, t_input,
                deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale,
                **plan_attention_kwargs, **plan_kwargs,
            )
        v_cond, _ = net_out_to_v_x(net_out_cond, z, t_input, t_eps)
        return v_cond, v_uncond

    def get_sc_guided_v(
        z, t_input, base_v_target, x_tokens, shared_net_out_uncond,
        plan_z_input=None, plan_t_input=None,
    ):
        """v target with self-conditioning guidance."""
        v_cond, v_uncond = get_sc_cond_and_uncond(
            z, t_input, cond_mask=cond_seq_mask, x_tokens=x_tokens,
            shared_net_out_uncond=shared_net_out_uncond,
            plan_z_input=plan_z_input, plan_t_input=plan_t_input,
        )
        sc_w = self_cond_cfg_scale.reshape(batch_size, 1, 1)
        sc_guidance = (1 - 1 / sc_w) * (v_cond - v_uncond)
        sc_guidance = torch.where(use_self_cond_mask.bool(), sc_guidance, torch.zeros_like(sc_guidance))
        return (base_v_target + sc_guidance).detach()

    def get_v_target(
        z, t_input, base_v_target, x_tokens, shared_net_out_uncond,
        plan_z_input=None, plan_t_input=None,
    ):
        """Compute final v target with self-conditioning guidance."""
        if config.num_self_cond_cfg_tokens > 0 and config.self_cond_prob > 0:
            return get_sc_guided_v(
                z, t_input, base_v_target=base_v_target, x_tokens=x_tokens,
                shared_net_out_uncond=shared_net_out_uncond,
                plan_z_input=plan_z_input, plan_t_input=plan_t_input,
            )
        return base_v_target

    model.train()

    # Per-example branching: build a mixed input (decoder_z for decoder-mode
    # rows, denoiser_z for denoiser-mode rows). One forward computes both
    # heads; we mask CE / L2 losses to their respective rows. 
    denoiser_t = t
    decoder_t = torch.ones_like(t)
    t_mixed = decoder_step_active * decoder_t + (1.0 - decoder_step_active) * t  # (B,)
    z_mixed = decoder_mask_B11 * decoder_z + (1.0 - decoder_mask_B11) * denoiser_z
    if use_sentence_plan:
        plan_t_mixed = decoder_step_active * decoder_t + (1.0 - decoder_step_active) * plan_t_denoiser

    # Self-cond shared forward (run on denoiser_z / t — only relevant for
    # denoiser-mode rows; decoder-mode rows zero out the self-cond half below).
    if self_cond_prob > 0 or config.num_self_cond_cfg_tokens > 0:
        shared_net_out_uncond = compute_shared_uncond(
            denoiser_z, denoiser_t, x0,
            plan_z_input=plan_z_denoiser, plan_t_input=plan_t_denoiser,
        )
    else:
        shared_net_out_uncond = None

    if config.self_cond_prob > 0:
        _, x_pred_init = net_out_to_v_x(shared_net_out_uncond, denoiser_z, denoiser_t, t_eps)
        x_pred_init = restore_cond(x_pred_init, x0, cond_seq_mask)
        x_pred_cond = x_pred_init * use_self_cond_mask.to(dtype)
        x_pred_cond = restore_cond(x_pred_cond, x0, cond_seq_mask)
        # Zero the self-cond half for decoder-mode rows (matches the old
        # `cat([decoder_z, zeros], -1)` decoder-branch input).
        sc_half = x_pred_cond * (1.0 - decoder_mask_B11)
        model_input = torch.cat([z_mixed, sc_half], dim=-1)
    else:
        model_input = z_mixed

    plan_kwargs = {}
    if use_sentence_plan:
        plan_kwargs = {
            "plan_z": plan_z_mixed,
            "plan_t": plan_t_mixed,
            "return_plan": True,
        }
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        model_out = model(
            model_input, t_mixed,
            deterministic=False,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=decoder_step_active,  # (B,) tensor
            **plan_attention_kwargs, **plan_kwargs,
        )
    if use_sentence_plan:
        net_out, decoder_logits, plan_pred = model_out
        plan_loss = ((plan_pred.float() - plan_target.float()) ** 2).mean(dim=-1).mean()
        plan_pred_detached = plan_pred.detach().float()
        plan_pred_batch_var = plan_pred_detached.var(dim=0, unbiased=False).mean()
        plan_pred_norm = plan_pred_detached.norm(dim=-1).mean()
        plan_loss_for_backward = plan_loss
        if grad_mode == "none" and sentence_encoder_type == "learned" and s0.requires_grad:
            # STAR-LDM topology: the main pass keeps Enc in the field/CE graph,
            # but diffusion MSE is detached so it cannot train Enc through either
            # the target or the noised-input path. A detached aux pass below trains
            # the plan denoiser/head on the same kind of joint input. The zero
            # sink keeps plan-head hooks alive for DDP when plan_aux_passes=0.
            plan_loss_for_backward = plan_loss.detach() + 0.0 * plan_loss
    else:
        net_out, decoder_logits = model_out

    if (
        use_sentence_plan
        and sentence_encoder_type == "learned"
        and grad_mode == "none"
        and s0.requires_grad
        and plan_aux_passes > 0
    ):
        s0_detached = s0.detach()
        for _ in range(plan_aux_passes):
            if plan_aux_token_context == "denoiser_z":
                t_aux = denoiser_t
                plan_t_aux = plan_t_denoiser
                token_z_aux = denoiser_z.detach()
            elif plan_aux_token_context == "mixed_z":
                t_aux = t_mixed
                plan_t_aux = plan_t_mixed
                token_z_aux = z_mixed.detach()
            else:
                t_aux = sample_timesteps(
                    batch_size,
                    P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                    time_schedule=config.time_schedule,
                    device=device, dtype=dtype,
                )
                plan_t_aux = plan_time_from_token_time(t_aux, config)
                if plan_aux_token_context == "clean_x0":
                    token_z_aux = x0.detach()
                else:
                    token_noise_aux = torch.randn_like(x0)
                    token_z_aux = add_noise(x0.detach(), token_noise_aux, t_aux, config, cond_seq_mask=cond_seq_mask)
                    if config.label_drop_prob > 0:
                        token_z_aux = torch.where(
                            drop.unsqueeze(-1) & (cond_seq_mask > 0),
                            torch.zeros_like(token_z_aux),
                            token_z_aux,
                        )

            plan_noise_aux = torch.randn_like(s0_detached) * float(getattr(config, "plan_noise_scale", 1.0))
            plan_t_aux_expanded = plan_t_aux.reshape(-1, 1)
            plan_z_aux = (
                plan_t_aux_expanded * s0_detached
                + (1.0 - plan_t_aux_expanded) * plan_noise_aux
            )
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                _, _, plan_pred_aux = model(
                    token_z_aux.detach(), t_aux,
                    deterministic=False,
                    self_cond_cfg_scale=self_cond_cfg_scale,
                    **plan_attention_kwargs,
                    plan_z=plan_z_aux,
                    plan_t=plan_t_aux,
                    return_plan=True,
                )
            plan_aux_loss = (
                plan_aux_loss
                + ((plan_pred_aux.float() - s0_detached.float()) ** 2).mean(dim=-1).mean()
            )
        plan_loss_for_backward = plan_loss_for_backward + plan_aux_loss

    # CE per-token (used on decoder-mode rows).
    log_probs = F.log_softmax(decoder_logits.to(torch.float32), dim=-1)
    ce_per_token = -log_probs.gather(-1, decoder_targets.unsqueeze(-1)).squeeze(-1)

    # L2 per-token (used on denoiser-mode rows). v_pred is extracted with
    # (denoiser_z, t) — meaningful only for denoiser rows; decoder rows are
    # masked out below.
    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, denoiser_t, t_eps)
    v_final_target = get_v_target(
        denoiser_z, denoiser_t, base_v_target=v_target, x_tokens=x0,
        shared_net_out_uncond=shared_net_out_uncond,
        plan_z_input=plan_z_denoiser, plan_t_input=plan_t_denoiser,
    )
    l2_per_token = ((v_pred - v_final_target) ** 2).mean(dim=-1)

    # Masks: each position is "alive" for exactly one branch.
    loss_mask_f = loss_mask.to(ce_per_token.dtype)
    ce_mask = loss_mask_f * decoder_mask_B1
    l2_mask = loss_mask_f * (1.0 - decoder_mask_B1)

    # Combined loss with a single denominator. In expectation this is
    # decoder_prob * mean_CE + (1 - decoder_prob) * mean_L2.
    total_sum = (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
    loss = total_sum / torch.clamp(loss_mask_f.sum(), min=1.0)
    loss = loss + float(getattr(config, "plan_loss_weight", 1.0)) * plan_loss_for_backward

    # Per-branch metrics: mean per-token within each branch.
    ce_loss_sum = (ce_per_token * ce_mask).sum().detach()
    ce_token_count = ce_mask.sum().detach()
    l2_loss_sum = (l2_per_token * l2_mask).sum().detach()
    l2_token_count = l2_mask.sum().detach()
    ce_loss_val = ce_loss_sum / torch.clamp(ce_token_count, min=1.0)
    l2_loss_val = l2_loss_sum / torch.clamp(l2_token_count, min=1.0)

    (loss / accumulation_divisor).backward()
    state.micro_step = int(getattr(state, "micro_step", state.step)) + 1
    state.step = state.micro_step

    if is_optimizer_step:
        torch.nn.utils.clip_grad_norm_(_trainable_params(model), max_norm=1.0)
        state.optimizer.step()
        if state.lr_scheduler is not None:
            state.lr_scheduler.step()
        ema_update(state.ema_params1, state.model, config.ema_decay1)
        state.optimizer.zero_grad(set_to_none=True)
        state.optimizer_step = int(getattr(state, "optimizer_step", 0)) + 1
        state.accum_step = 0
    else:
        state.accum_step = accumulated + 1

    if previous_sync_setting is not None:
        model.require_backward_grad_sync = previous_sync_setting

    metrics = {
        "loss": loss.detach(),
        "l2_loss": l2_loss_val,
        "ce_loss": ce_loss_val,
        "ce_loss_sum": ce_loss_sum,
        "ce_token_count": ce_token_count,
        "plan_loss": plan_loss.detach(),
        "plan_aux_loss": plan_aux_loss.detach(),
        "plan_emb_batch_var": plan_emb_batch_var.detach(),
        "plan_emb_norm": plan_emb_norm.detach(),
        "plan_pred_batch_var": plan_pred_batch_var.detach(),
        "plan_pred_norm": plan_pred_norm.detach(),
        "l2_loss_sum": l2_loss_sum,
        "l2_token_count": l2_token_count,
        "optimizer_step": torch.tensor(state.optimizer_step, device=loss.device),
        "did_optimizer_step": torch.tensor(is_optimizer_step, device=loss.device),
    }
    return state, metrics
