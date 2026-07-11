from typing import Optional

import torch


# ============================================
# Noise Schedulers (how to compute z from x0 and noise)
# ============================================

def add_noise(x0, noise, t, config, cond_seq_mask=None):
    """Flow-matching interpolation z = t*x0 + (1-t)*noise*scale, preserving cond tokens."""
    t_expanded = t.reshape(-1, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1 - cond_seq_mask) * z
    return z


# ============================================
# Time Schedulers (how to sample t)
# ============================================

def sample_timesteps(
    batch_size: int,
    P_mean: float = -0.8,
    P_std: float = 0.8,
    time_schedule: str = 'logit_normal',
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """Sample timesteps using various time schedules.

    Args:
        batch_size: Number of samples
        P_mean: Mean for logit-normal distribution
        P_std: Std for logit-normal distribution
        time_schedule: 'logit_normal' or 'uniform'

    Returns:
        Sampled timesteps in [0, 1]
    """
    if time_schedule == 'logit_normal':
        z = torch.randn((batch_size,), dtype=dtype, device=device) * P_std + P_mean
        return torch.sigmoid(z)
    if time_schedule == 'uniform':
        return torch.rand((batch_size,), dtype=dtype, device=device)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def get_sampling_steps(
    n_steps: int, time_schedule: str = "logit_normal",
    P_mean: float = -0.8, P_std: float = 0.8,
    device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a length-(n_steps+1) tensor of t values in [0, 1] for a sampling run.

    - "uniform": evenly-spaced linspace from 0 to 1 (deterministic).
    - "logit_normal": sorted logit-normal samples with 0 / 1 endpoints (random).
    """
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, dtype=dtype, device=device)
    if time_schedule == "logit_normal":
        steps = sample_timesteps(
            batch_size=n_steps - 1,
            P_mean=P_mean, P_std=P_std, time_schedule=time_schedule,
            device=device, dtype=dtype,
        )
        steps = torch.sort(steps).values
        endpoints_lo = torch.zeros((1,), dtype=dtype, device=steps.device)
        endpoints_hi = torch.ones((1,), dtype=dtype, device=steps.device)
        return torch.cat([endpoints_lo, steps, endpoints_hi], dim=0)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def plan_time_from_token_time(token_t: torch.Tensor, config) -> torch.Tensor:
    """Map token diffusion time to sentence-plan diffusion time.

    `aligned` keeps the original behavior: plan_t == token_t.
    `noise_power` makes the plan lead the token field by shrinking plan noise as
    a power of token noise: plan_t = 1 - (1 - token_t) ** gamma.
    """
    schedule = str(getattr(config, "plan_time_schedule", "aligned")).lower()
    if schedule in {"aligned", "identity"}:
        return token_t

    if schedule == "noise_power":
        gamma = float(getattr(config, "plan_time_warp_gamma", 1.0))
        token_t = token_t.clamp(0.0, 1.0)
        plan_noise = (1.0 - token_t).clamp(0.0, 1.0).pow(gamma)
        return 1.0 - plan_noise

    raise ValueError(f"Unknown plan_time_schedule: {schedule}")


# ============================================
# CFG Scale Sampling (how to sample cfg scale)
# ============================================

def sample_cfg_scale(batch_size, cfg_min=0.0, cfg_max=3.0,
                     dtype=torch.float32, device=None):
    """Sample CFG scale from log-uniform distribution in [cfg_min, cfg_max]."""
    u = torch.rand((batch_size,), dtype=dtype, device=device)
    a = float(1.0 + cfg_min)
    b = float(1.0 + cfg_max)
    log_ratio = torch.tensor(b / a, dtype=dtype, device=u.device).log()
    return a * torch.exp(u * log_ratio) - 1.0


# ============================================
# Conditioning helpers (preserve clean tokens during sampling)
# ============================================

def restore_cond(z_updated, cond_seq, cond_seq_mask):
    """Restore clean conditioning tokens in z after a denoising step."""
    mask = cond_seq_mask
    target_ndim = max(z_updated.dim(), cond_seq.dim())
    while mask.dim() < target_ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def restore_vx(v, x, cond_seq, cond_seq_mask):
    """Restore cond positions: x -> clean cond_seq, v -> 0 (cond tokens don't move)."""
    if cond_seq is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)
        v = restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask)
    return v, x


# ============================================
# Flow-matching forward passes (with optional self-cond / CFG)
# ============================================

def net_out_to_v_x(net_out, z, t, t_eps=5e-2):
    """Convert x_pred network output to v and x.

    When the model returns a tuple (denoised_output, decoder_logits),
    decoder logits are discarded here (used separately in training).
    """
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_reshaped = t.reshape(-1, 1, 1)
    x = net_out
    denom = torch.clamp(1.0 - t_reshaped, min=t_eps)
    v = (x - z) / denom
    return v, x


def plan_out_to_v_x(plan_pred, plan_z, t, t_eps=5e-2):
    """Convert sentence-plan x prediction to v and x."""
    denom = torch.clamp(1.0 - t.reshape(-1, 1), min=t_eps)
    plan_x = plan_pred
    plan_v = (plan_x - plan_z) / denom
    return plan_v, plan_x


def _split_model_out(model_out, z, t_batch, t_eps, plan_z=None, plan_t=None):
    field_out = model_out[0] if isinstance(model_out, tuple) else model_out
    v, x = net_out_to_v_x(field_out, z, t_batch, t_eps)
    if plan_z is None:
        return v, x, None, None
    if not isinstance(model_out, tuple) or len(model_out) < 3 or model_out[2] is None:
        raise ValueError("model must return a plan prediction when plan_z is provided")
    plan_t_batch = t_batch if plan_t is None else plan_t
    plan_v, plan_x = plan_out_to_v_x(model_out[2], plan_z, plan_t_batch, t_eps)
    return v, x, plan_v, plan_x


def _forward_sample_self_cond(
    model, z, t_batch, x_pred_prev, config,
    self_cond_cfg_scale, cond_seq, cond_seq_mask,
    plan_z=None, plan_t=None,
):
    """Forward pass with self-conditioning."""
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob

    def _restore(v, x):
        return restore_vx(v, x, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask)

    def _restore_with_plan(v, x, plan_v, plan_x):
        v, x = _restore(v, x)
        return v, x, plan_v, plan_x

    def _model_forward(z_input, self_cond_scale=None):
        plan_kwargs = {}
        if plan_z is not None:
            plan_kwargs = {
                "plan_z": plan_z,
                "plan_t": t_batch if plan_t is None else plan_t,
                "return_plan": True,
            }
        return model(
            z_input, t_batch, deterministic=True,
            self_cond_cfg_scale=self_cond_scale,
            **plan_kwargs,
        )

    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
        self_cond_scale_batch = torch.full((z.shape[0],), float(self_cond_cfg_scale),
                                           dtype=z.dtype, device=z.device)
        net_out_cond = _model_forward(z_input_cond, self_cond_scale_batch)
        v_cond, x_cond, plan_v_cond, plan_x_cond = _split_model_out(
            net_out_cond, z, t_batch, t_eps, plan_z=plan_z, plan_t=plan_t,
        )
        return _restore_with_plan(v_cond, x_cond, plan_v_cond, plan_x_cond)

    # No self-conditioning
    if self_cond_prob == 0:
        net_out = _model_forward(z)
        v, x, plan_v, plan_x = _split_model_out(
            net_out, z, t_batch, t_eps, plan_z=plan_z, plan_t=plan_t,
        )
        return _restore_with_plan(v, x, plan_v, plan_x)

    # Combined unconditional and conditional forward pass
    v_uncond = x_uncond = plan_v_uncond = plan_x_uncond = None
    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        z_uncond = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        net_out_uncond = _model_forward(z_input_uncond)
        v_uncond, x_uncond, plan_v_uncond, plan_x_uncond = _split_model_out(
            net_out_uncond, z, t_batch, t_eps, plan_z=plan_z, plan_t=plan_t,
        )
        v_uncond, x_uncond = _restore(v_uncond, x_uncond)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond, plan_v_uncond, plan_x_uncond

    z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
    net_out_cond = _model_forward(z_input_cond)
    v_cond, x_cond, plan_v_cond, plan_x_cond = _split_model_out(
        net_out_cond, z, t_batch, t_eps, plan_z=plan_z, plan_t=plan_t,
    )
    v_cond, x_cond = _restore(v_cond, x_cond)
    if self_cond_cfg_scale == 1:
        return v_cond, x_cond, plan_v_cond, plan_x_cond

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    if plan_z is not None:
        plan_v_out = plan_v_uncond + self_cond_cfg_scale * (plan_v_cond - plan_v_uncond)
        plan_x_out = plan_x_uncond + self_cond_cfg_scale * (plan_x_cond - plan_x_uncond)
    else:
        plan_v_out = plan_x_out = None
    v_out, x_out = _restore(v_out, x_out)
    return v_out, x_out, plan_v_out, plan_x_out


def _forward_sample(
    model, z, t_batch, x_pred_prev, config,
    cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
    plan_z=None, plan_t=None,
):
    """Forward pass with optional self-conditioning and CFG."""
    v_cond, x_cond, plan_v_cond, plan_x_cond = _forward_sample_self_cond(
        model, z, t_batch, x_pred_prev, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        plan_z=plan_z, plan_t=plan_t,
    )
    if cfg_scale == 1.0:
        return v_cond, x_cond, plan_v_cond, plan_x_cond

    # Unconditional forward: zero out cond prefix, no self-cond state, no restore
    z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)
    x_pred_prev_uncond = (
        None if x_pred_prev is None
        else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask)
    )
    v_uncond, x_uncond, plan_v_uncond, plan_x_uncond = _forward_sample_self_cond(
        model, z_uncond, t_batch, x_pred_prev_uncond, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=torch.zeros_like(cond_seq), cond_seq_mask=cond_seq_mask,
        plan_z=plan_z, plan_t=plan_t,
    )

    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    if plan_z is not None:
        plan_v_out = plan_v_uncond + cfg_scale * (plan_v_cond - plan_v_uncond)
        plan_x_out = plan_x_uncond + cfg_scale * (plan_x_cond - plan_x_uncond)
    else:
        plan_v_out = plan_x_out = None
    v_out, x_out = restore_vx(v_out, x_out, cond_seq, cond_seq_mask)
    return v_out, x_out, plan_v_out, plan_x_out


def _ode_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask, plan_z=None, plan_t=None, plan_t_next=None,
    freeze_plan_z: bool = False,
):
    """Single ODE (Euler) step for sampling."""
    t_batch = torch.full((z.shape[0],), float(t), dtype=z.dtype, device=z.device)
    v_pred, x_pred, plan_v_pred, plan_x_pred = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        plan_z=plan_z, plan_t=plan_t,
    )
    z_next = z + (t_next - t) * v_pred
    if plan_z is None:
        return z_next, x_pred
    if freeze_plan_z:
        plan_z_next = plan_z
    elif plan_t is not None and plan_t_next is not None:
        plan_dt = (plan_t_next - plan_t).reshape(-1, 1)
        plan_z_next = plan_z + plan_dt * plan_v_pred
    else:
        plan_z_next = plan_z + (t_next - t) * plan_v_pred
    return z_next, x_pred, plan_z_next, plan_x_pred


def _sde_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask, gamma, generator, plan_z=None,
    plan_t=None, plan_t_next=None, freeze_plan_z: bool = False,
):
    """Per-step SDE-style sampler with hybrid (t-and-step) noise scaling.

    t_back = t * (1 - gamma * h), where h = t_next - t. alpha = 1 - gamma*h is the
    signal-preservation fraction, constant in t. gamma=0 degenerates to a plain ODE step.
    Uniform-N-step equivalence with old multiplicative gamma_old: gamma_hybrid = gamma_old * N.
    """
    h = float(t_next - t)
    alpha = max(0.0, min(1.0, 1.0 - gamma * h))
    t_back = alpha * float(t)
    if z.is_cuda:
        eps = torch.randn(z.shape, dtype=z.dtype, device=z.device) * config.denoiser_noise_scale
    else:
        eps = torch.randn(z.shape, generator=generator, dtype=z.dtype) * config.denoiser_noise_scale
    z_back = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
    plan_z_back = None
    plan_t_back = None
    if plan_z is not None:
        if freeze_plan_z:
            plan_z_back = plan_z
            plan_t_back = plan_t
        else:
            if z.is_cuda:
                plan_eps = torch.randn(plan_z.shape, dtype=plan_z.dtype, device=plan_z.device)
            else:
                plan_eps = torch.randn(plan_z.shape, generator=generator, dtype=plan_z.dtype, device=plan_z.device)
            plan_eps = plan_eps * float(getattr(config, "plan_noise_scale", 1.0))
            t_back_batch = torch.full((z.shape[0],), t_back, dtype=z.dtype, device=z.device)
            plan_t_back = plan_time_from_token_time(t_back_batch, config)
            if plan_t is not None:
                denom = torch.clamp(plan_t.reshape(-1, 1), min=1e-6)
                plan_alpha = torch.clamp(plan_t_back.reshape(-1, 1) / denom, min=0.0, max=1.0)
            else:
                plan_alpha = alpha
            plan_z_back = plan_alpha * plan_z + (1.0 - plan_alpha) * plan_eps
    t_batch = torch.full((z.shape[0],), t_back, dtype=z.dtype, device=z.device)
    v_pred, x_pred, plan_v_pred, plan_x_pred = _forward_sample(
        model=model, z=z_back, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        plan_z=plan_z_back, plan_t=plan_t_back if plan_t_back is not None else plan_t,
    )
    z_next = z_back + (t_next - t_back) * v_pred
    if plan_z is None:
        return z_next, x_pred
    if freeze_plan_z:
        plan_z_next = plan_z
    elif plan_t_back is not None and plan_t_next is not None:
        plan_dt = (plan_t_next - plan_t_back).reshape(-1, 1)
        plan_z_next = plan_z_back + plan_dt * plan_v_pred
    else:
        plan_z_next = plan_z_back + (t_next - t_back) * plan_v_pred
    return z_next, x_pred, plan_z_next, plan_x_pred
