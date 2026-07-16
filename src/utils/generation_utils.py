from typing import Optional

import torch
import torch.nn as nn

from configs.config import Config, SamplingConfig
from utils.sampling_utils import (
    restore_cond, _ode_step, _sde_step, get_sampling_steps,
    plan_time_from_token_time,
)


# ============================================
# Generation utilities
# ============================================

def mask_after_eos(predicted_ids: torch.Tensor, eos_token_id: int, pad_token_id: int) -> torch.Tensor:
    """Mask everything at/after first EOS token per sequence."""
    eos_mask = (predicted_ids == eos_token_id)
    keep_mask = (eos_mask.to(torch.int32).cumsum(dim=1) == 0)
    return torch.where(keep_mask, predicted_ids, torch.full_like(predicted_ids, pad_token_id))


def shift_left(x: torch.Tensor, shift_per_sample: torch.Tensor, pad_value=0, axis: int = 1) -> torch.Tensor:
    """Shift each sample left along the sequence axis; pad emptied positions."""
    if x.dim() < 2:
        raise ValueError("x must have at least batch and sequence dimensions")
    if axis < 0:
        axis = x.dim() + axis
    if axis == 0:
        raise ValueError("axis=0 is the batch axis and cannot be shifted")
    shift_per_sample = shift_per_sample.to(torch.long)
    if axis != 1:
        x = x.movedim(axis, 1)
    seq_len = x.shape[1]
    base_idx = torch.arange(seq_len, device=x.device)[None, :]
    gather_idx = shift_per_sample[:, None].to(x.device) + base_idx
    valid = gather_idx < seq_len
    gather_idx = gather_idx.clamp(0, seq_len - 1)
    if x.dim() == 2:
        shifted = torch.gather(x, 1, gather_idx)
        shifted = torch.where(valid, shifted, torch.full_like(shifted, pad_value))
    else:
        expand_shape = [-1, -1] + list(x.shape[2:])
        idx = gather_idx.view(*gather_idx.shape, *([1] * (x.dim() - 2))).expand(*expand_shape)
        valid_b = valid.view(*valid.shape, *([1] * (x.dim() - 2))).expand(*expand_shape)
        shifted = torch.gather(x, 1, idx)
        shifted = torch.where(valid_b, shifted, torch.full_like(shifted, pad_value))
    if axis != 1:
        shifted = shifted.movedim(1, axis)
    return shifted


# ============================================
# Single-batch sampling (PyTorch)
# ============================================

@torch.no_grad()
def _generate_samples_single_batch(
    model: nn.Module,
    generator: torch.Generator,
    z: torch.Tensor,
    t_steps: torch.Tensor,
    cond_seq: Optional[torch.Tensor],
    cond_seq_mask: Optional[torch.Tensor],
    config: Config,
    sampling_config: SamplingConfig,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    initial_plan_z: Optional[torch.Tensor] = None,
    fixed_plan_z: bool = False,
    plan_generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate samples for a single batch (PyTorch Euler / SDE rollout).

    ``plan_first`` is a strict two-stage rollout.  The token field, token time,
    and token self-conditioning stay at their initial-noise state while the
    plan advances from noise to clean.  The resulting plan is then frozen at
    ``plan_t=1`` while the token field advances from noise to clean.
    """
    method = sampling_config.sampling_method
    plan_sampling_mode = str(
        getattr(sampling_config, "plan_sampling_mode", "joint")
    ).lower()
    batch_size, max_length, d_model = z.shape
    if cond_seq is None:
        cond_seq = torch.zeros((batch_size, max_length, d_model), dtype=z.dtype, device=z.device)
        cond_seq_mask = torch.zeros((batch_size, max_length), dtype=z.dtype, device=z.device)

    step_kwargs = dict(
        model=model, config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
    plan_z = None
    if bool(getattr(config, "use_sentence_plan", False)):
        plan_shape = (batch_size, int(getattr(config, "sentence_emb_dim", 768)))
        if initial_plan_z is not None:
            if tuple(initial_plan_z.shape) != plan_shape:
                raise ValueError(f"initial_plan_z shape {tuple(initial_plan_z.shape)} does not match {plan_shape}")
            plan_z = initial_plan_z.to(device=z.device, dtype=z.dtype)
        else:
            plan_z = torch.randn(
                plan_shape,
                generator=plan_generator if plan_generator is not None else generator,
                dtype=z.dtype,
                device=z.device,
            )
            plan_z = plan_z * float(getattr(config, "plan_noise_scale", 1.0))
    elif initial_plan_z is not None:
        raise ValueError("initial_plan_z was provided but use_sentence_plan=False")
    if plan_sampling_mode == "plan_first" and plan_z is None:
        raise ValueError("plan_sampling_mode='plan_first' requires use_sentence_plan=True")

    n = t_steps.shape[0]
    sde_gamma = getattr(sampling_config, "sde_gamma", 0.0)
    fixed_plan_t = (
        torch.ones((batch_size,), dtype=z.dtype, device=z.device)
        if (fixed_plan_z or plan_sampling_mode == "plan_first") and plan_z is not None else None
    )

    def _plan_time(value: float) -> Optional[torch.Tensor]:
        if plan_z is None:
            return None
        if fixed_plan_t is not None:
            return fixed_plan_t
        token_t = torch.full((batch_size,), float(value), dtype=z.dtype, device=z.device)
        return plan_time_from_token_time(token_t, config)

    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        if plan_sampling_mode == "plan_first" and not fixed_plan_z:
            plan_num_steps = getattr(sampling_config, "plan_num_sampling_steps", None)
            if plan_num_steps is None:
                plan_num_steps = n - 1
            plan_steps = get_sampling_steps(
                n_steps=int(plan_num_steps),
                time_schedule=sampling_config.time_schedule,
                P_mean=config.denoiser_p_mean,
                P_std=config.denoiser_p_std,
                device=z.device,
                dtype=z.dtype,
                generator=plan_generator if plan_generator is not None else generator,
            )
            token_noise_t = 0.0
            # Do not carry token predictions out of this phase: doing so would
            # let plan-only model calls initialize token self-conditioning.
            for i in range(plan_steps.shape[0] - 2):
                plan_t = torch.full(
                    (batch_size,), float(plan_steps[i].item()),
                    dtype=z.dtype, device=z.device,
                )
                plan_t_next = torch.full(
                    (batch_size,), float(plan_steps[i + 1].item()),
                    dtype=z.dtype, device=z.device,
                )
                if method == "sde":
                    step_out = _sde_step(
                        z=z, t=token_noise_t, t_next=token_noise_t,
                        x_pred_prev=None, gamma=sde_gamma,
                        generator=generator, plan_generator=plan_generator,
                        plan_z=plan_z, plan_t=plan_t,
                        plan_t_next=plan_t_next, freeze_token_z=True,
                        **step_kwargs,
                    )
                elif method == "ode":
                    step_out = _ode_step(
                        z=z, t=token_noise_t, t_next=token_noise_t,
                        x_pred_prev=None, plan_z=plan_z, plan_t=plan_t,
                        plan_t_next=plan_t_next, freeze_token_z=True,
                        **step_kwargs,
                    )
                else:
                    raise ValueError(f"Invalid sampling method: {method}")
                z, _, plan_z, _ = step_out

            # Last plan step is deterministic, matching the token sampler's
            # final-step convention.
            plan_t = torch.full(
                (batch_size,), float(plan_steps[-2].item()),
                dtype=z.dtype, device=z.device,
            )
            plan_t_next = torch.full(
                (batch_size,), float(plan_steps[-1].item()),
                dtype=z.dtype, device=z.device,
            )
            z, _, plan_z, _ = _ode_step(
                z=z, t=token_noise_t, t_next=token_noise_t,
                x_pred_prev=None, plan_z=plan_z, plan_t=plan_t,
                plan_t_next=plan_t_next, freeze_token_z=True,
                **step_kwargs,
            )
            x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

        for i in range(n - 2):
            t = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            plan_t = _plan_time(t)
            plan_t_next = _plan_time(t_next)
            if method == "sde":
                step_out = _sde_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                    gamma=sde_gamma, generator=generator, **step_kwargs,
                    plan_generator=plan_generator,
                    plan_z=plan_z, plan_t=plan_t, plan_t_next=plan_t_next,
                    freeze_plan_z=(fixed_plan_z or plan_sampling_mode == "plan_first"),
                )
            elif method == "ode":
                step_out = _ode_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                    plan_z=plan_z, plan_t=plan_t, plan_t_next=plan_t_next,
                    freeze_plan_z=(fixed_plan_z or plan_sampling_mode == "plan_first"),
                    **step_kwargs,
                )
            else:
                raise ValueError(f"Invalid sampling method: {method}")
            if plan_z is None:
                z, x_pred = step_out
            else:
                z, x_pred, plan_z, _ = step_out

        # Last step always with ODE.
        t = t_steps[-2].item()
        t_next = t_steps[-1].item()
        plan_t = _plan_time(t)
        plan_t_next = _plan_time(t_next)
        step_out = _ode_step(
            z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
            plan_z=plan_z, plan_t=plan_t, plan_t_next=plan_t_next,
            freeze_plan_z=(fixed_plan_z or plan_sampling_mode == "plan_first"),
            **step_kwargs,
        )
        if plan_z is None:
            z, x_pred = step_out
        else:
            z, x_pred, plan_z, _ = step_out
    if plan_z is None:
        return z
    return z, plan_z


@torch.no_grad()
def _dlm_decode_batch(z: torch.Tensor, model: nn.Module, t_final_val,
                      config, self_cond_cfg_scale: float,
                      plan_z: Optional[torch.Tensor] = None,
                      cond_seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Decode z -> tokens with the DLM decoder head."""
    batch_size = z.shape[0]
    if isinstance(t_final_val, torch.Tensor) and t_final_val.dim() == 0:
        t_final = torch.full((batch_size,), t_final_val.item(), dtype=z.dtype, device=z.device)
    else:
        t_final = torch.full((batch_size,), float(t_final_val), dtype=z.dtype, device=z.device)
    sc_batch = (
        torch.full((batch_size,), float(self_cond_cfg_scale), dtype=z.dtype, device=z.device)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    plan_kwargs = {}
    if bool(getattr(config, "use_sentence_plan", False)):
        if plan_z is None:
            raise ValueError("plan_z is required for decoding when use_sentence_plan=True")
        plan_kwargs = {"plan_z": plan_z, "plan_t": t_final}
    topology_kwargs = {}
    if str(getattr(config, "plan_attention_topology", "joint")) in {
        "hierarchical_prefix",
        "strict_hierarchical_prefix",
    }:
        if cond_seq_mask is None:
            cond_seq_mask = torch.zeros(
                z.shape[:2], dtype=z.dtype, device=z.device,
            )
        topology_kwargs = {"cond_seq_mask": cond_seq_mask}
    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        _, decoder_logits = model(
            z_input, t_final, deterministic=True,
            self_cond_cfg_scale=sc_batch,
            decoder_step_active=True,
            **topology_kwargs, **plan_kwargs,
        )
    return decoder_logits.argmax(dim=-1)


def _build_run_name(sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
                    time_schedule, sde_gamma, suffix,
                    plan_sampling_mode="joint", plan_num_sampling_steps=None):
    ts_str = f"-ts_{time_schedule}"
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    sde_str = f"-gamma{sde_gamma}" if sampling_method == "sde" else ""
    plan_str = ""
    if str(plan_sampling_mode).lower() == "plan_first":
        resolved_plan_steps = (
            num_sampling_steps if plan_num_sampling_steps is None
            else int(plan_num_sampling_steps)
        )
        plan_str = f"-planfirst{resolved_plan_steps}"
    return (
        f"{sampling_method}-steps{num_sampling_steps}-cfg{cfg_scale}{sccfg_str}"
        f"{ts_str}{sde_str}{plan_str}-{suffix}"
    )
