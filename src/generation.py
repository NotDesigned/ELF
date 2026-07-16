import copy
import itertools
import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from configs.config import Config, SamplingConfig
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import upload_output_dir_to_hf
from utils.train_utils import unwrap_model
from utils.data_utils import get_dataloader, get_pad_token_id
from utils.encoder_utils import encode_text
from utils.metrics_utils import (
    Metrics as PPLMetrics,
    compute_bleu,
    compute_mauve_from_features,
    compute_rouge,
)
from utils.sampling_utils import add_noise, _forward_sample, get_sampling_steps
from utils.generation_utils import (
    mask_after_eos, shift_left,
    _generate_samples_single_batch, _dlm_decode_batch,
    _build_run_name,
)

try:
    import wandb
except ImportError:
    wandb = None


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _evaluation_sampling_dimensions(
    sampling_config: SamplingConfig,
    *,
    num_sampling_steps: int,
    cfg_scale: float,
    self_cond_cfg_scale: float,
) -> dict:
    """Return the exact sampling-family identity written with eval metrics.

    These producer-authored dimensions are consumed by
    ``elf_experiments.summary``.  Keep scalar normalization explicit so the
    derived family digest is stable across YAML values such as ``1`` and
    ``1.0``; never require a collector to reverse-engineer directory labels.
    """
    dimensions = {
        "sampling_method": str(sampling_config.sampling_method),
        "num_sampling_steps": int(num_sampling_steps),
        "cfg": float(cfg_scale),
        "self_cond_cfg_scale": float(self_cond_cfg_scale),
        "time_schedule": str(sampling_config.time_schedule),
        "time_warp_gamma": float(getattr(sampling_config, "sde_gamma", 0.0)),
    }
    plan_sampling_mode = str(
        getattr(sampling_config, "plan_sampling_mode", "joint")
    ).lower()
    if plan_sampling_mode == "plan_first":
        plan_steps = getattr(sampling_config, "plan_num_sampling_steps", None)
        plan_steps = num_sampling_steps if plan_steps is None else int(plan_steps)
        dimensions.update({
            "plan_sampling_mode": "plan_first",
            "plan_num_sampling_steps": plan_steps,
            "total_model_evaluations": int(num_sampling_steps) + plan_steps,
        })
    return dimensions


def _capture_rng_state(generator: torch.Generator) -> tuple:
    """Capture all RNG streams used by evaluation generation."""
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None
    )
    return (
        generator.get_state().clone(),
        torch.random.get_rng_state().clone(),
        cuda_states,
    )


def _restore_rng_state(generator: torch.Generator, state: tuple) -> None:
    """Restore evaluation RNG streams for a paired counterfactual."""
    generator_state, cpu_state, cuda_states = state
    generator.set_state(generator_state)
    torch.random.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


_PLAN_RNG_OFFSET = 1_000_000_007
_RNG_MODULUS = 2**63 - 1


def _evaluation_generators(
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Generator, torch.Generator]:
    """Build deterministic device-local token and plan RNG substreams."""
    base_seed = int(generator.initial_seed())
    token_generator = torch.Generator(device=device)
    plan_generator = torch.Generator(device=device)
    token_generator.manual_seed(base_seed % _RNG_MODULUS)
    plan_generator.manual_seed((base_seed + _PLAN_RNG_OFFSET) % _RNG_MODULUS)
    return token_generator, plan_generator


def _evaluation_rng_metadata(generator: torch.Generator) -> dict:
    return {
        "evaluation_seed": int(generator.initial_seed()),
        "rng_protocol": "split_token_plan_v1",
        "plan_rng_offset": _PLAN_RNG_OFFSET,
    }


def _sampled_plan_diagnostics(
    sampled_plan: torch.Tensor,
    clean_plan: torch.Tensor,
) -> dict[str, float | int]:
    """Measure whether sampled plans preserve per-example clean-plan identity."""
    if sampled_plan.shape != clean_plan.shape:
        raise ValueError(
            "sampled and clean plan shapes differ: "
            f"{tuple(sampled_plan.shape)} != {tuple(clean_plan.shape)}"
        )
    if sampled_plan.ndim < 2 or sampled_plan.shape[0] < 2:
        raise ValueError("sampled plan diagnostics require at least two samples")
    sampled = sampled_plan.detach().float().reshape(sampled_plan.shape[0], -1)
    clean = clean_plan.detach().float().reshape(clean_plan.shape[0], -1)
    if not torch.isfinite(sampled).all() or not torch.isfinite(clean).all():
        raise ValueError("sampled plan diagnostics require finite latents")

    sampled_var = sampled.var(dim=0, unbiased=False).mean()
    clean_var = clean.var(dim=0, unbiased=False).mean()
    per_example_cosine = F.cosine_similarity(sampled, clean, dim=1)
    per_example_mse = (sampled - clean).pow(2).mean(dim=1)
    similarity = F.normalize(sampled, dim=1) @ F.normalize(clean, dim=1).T
    indices = torch.arange(sampled.shape[0], device=similarity.device)
    true_similarity = similarity[indices, indices]
    negative_similarity = similarity.masked_fill(
        torch.eye(
            sampled.shape[0], dtype=torch.bool, device=similarity.device,
        ),
        float("-inf"),
    ).max(dim=1).values

    retrieval_hits = similarity.argmax(dim=1) == indices
    retrieval_top1 = retrieval_hits.float().mean()
    retrieval_chance = 1.0 / float(sampled.shape[0])
    metrics: dict[str, float | int] = {
        "sampled_plan_num_samples": int(sampled.shape[0]),
        "sampled_plan_batch_var": float(sampled_var.item()),
        "clean_plan_batch_var": float(clean_var.item()),
        "sampled_clean_plan_cosine": float(per_example_cosine.mean().item()),
        "sampled_clean_plan_cosine_std": float(
            per_example_cosine.std(unbiased=False).item()
        ),
        "sampled_clean_plan_mse": float(per_example_mse.mean().item()),
        "sampled_clean_plan_mse_std": float(
            per_example_mse.std(unbiased=False).item()
        ),
        "sampled_clean_plan_retrieval_top1": float(retrieval_top1.item()),
        "sampled_clean_plan_retrieval_top1_count": int(
            retrieval_hits.sum().item()
        ),
        "sampled_clean_plan_retrieval_chance": retrieval_chance,
        "sampled_clean_plan_retrieval_lift": float(
            retrieval_top1.item() / retrieval_chance
        ),
        "sampled_clean_plan_retrieval_margin": float(
            (true_similarity - negative_similarity).mean().item()
        ),
    }
    if float(clean_var.item()) > 0.0:
        metrics["sampled_plan_var_ratio"] = float(
            (sampled_var / clean_var).item()
        )
    return metrics


def _paired_nonempty_texts(generated_texts, reference_texts) -> tuple[list[str], list[str]]:
    """Keep equal-size, non-empty generated/reference samples for MAUVE."""
    generated, references = [], []
    for generated_text, reference_text in zip(generated_texts, reference_texts):
        if (
            isinstance(generated_text, str)
            and generated_text.strip()
            and isinstance(reference_text, str)
            and reference_text.strip()
        ):
            generated.append(generated_text)
            references.append(reference_text)
    return generated, references


def _dataset_reference_texts(dataset, tokenizer, count: int, max_length: int) -> list[str]:
    """Decode a deterministic real-text sample for unconditional MAUVE."""
    if dataset is None:
        return []
    references = []
    for idx in range(min(int(count), len(dataset))):
        item = dataset[idx]
        text = item.get("target") or item.get("text")
        if not isinstance(text, str):
            input_ids = item.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "MAUVE reference dataset rows need target, text, or input_ids"
                )
            text = tokenizer.decode(
                np.asarray(input_ids).reshape(-1)[:max_length].tolist(),
                skip_special_tokens=True,
            )
        references.append(text)
    return references


def _compute_mauve_metrics(
    evaluator: PPLMetrics,
    generated_texts: list[str],
    reference_texts: list[str],
    config,
    *,
    generated_features=None,
):
    """Compute reproducible MAUVE metadata, or skip undersized text sets."""
    generated, references = _paired_nonempty_texts(generated_texts, reference_texts)
    if len(generated) < 2:
        log_for_0(
            f"MAUVE eval: need at least 2 non-empty pairs; observed {len(generated)}"
        )
        return None
    if generated_features is None:
        generated_features = evaluator.featurize_texts(
            generated, max_length=config.eval_ppl_max_length,
        )
    elif len(generated_features) != len(generated):
        raise ValueError(
            "Precomputed MAUVE generated features do not match filtered text count: "
            f"{len(generated_features)} != {len(generated)}"
        )
    reference_features = evaluator.featurize_texts(
        references, max_length=config.eval_ppl_max_length,
    )
    results = compute_mauve_from_features(
        generated_features,
        reference_features,
        seed=int(config.eval_mauve_seed),
    )
    results.update({
        "mauve_featurizer": str(getattr(config, "eval_mauve_model", config.eval_ppl_model)),
        "mauve_seed": int(config.eval_mauve_seed),
        "mauve_scale": "percent",
    })
    return results


def _build_mauve_evaluator(config, ppl_evaluator=None):
    """Return the configured MAUVE featurizer, sharing the gPPL model if exact."""
    if not config.online_eval or not bool(config.eval_mauve):
        return None
    mauve_model = str(getattr(config, "eval_mauve_model", config.eval_ppl_model))
    if (
        ppl_evaluator is not None
        and mauve_model == str(config.eval_ppl_model)
    ):
        return ppl_evaluator
    return PPLMetrics(
        gen_ppl_eval_model_name_or_path=mauve_model,
        eval_ppl_batch_size=config.eval_ppl_batch_size,
        eval_context_size=config.eval_ppl_max_length,
    )


class _IndexedSubset:
    """Small map-style subset that preserves original dataset indices."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        item = dict(self.dataset[original_idx])
        item.setdefault("index", original_idx)
        return item


def _build_eval_model(state, use_compile: bool = False) -> nn.Module:
    """Return an eval-mode model copy loaded with EMA params (if available)."""
    model = unwrap_model(state.model)
    eval_model = copy.deepcopy(model)
    if state.ema_params1:
        eval_model.load_state_dict(state.ema_params1)
    eval_model.eval()
    if use_compile:
        log_for_0("Compiling eval model with torch.compile (first batch will be slower)...")
        eval_model = torch.compile(eval_model)
    return eval_model


def _batch_loss_mask(attention_mask: torch.Tensor, cond_seq_mask: torch.Tensor, config) -> torch.Tensor:
    """Return the token positions trained/evaluated as continuation tokens."""
    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = torch.ones_like(attention_mask)
    return loss_mask * (1 - cond_seq_mask)


def _decode_selected_texts(input_ids: torch.Tensor, mask: torch.Tensor, tokenizer) -> list:
    """Decode tokens selected by a boolean mask for clean sentence-plan targets."""
    ids_cpu = input_ids.detach().cpu()
    mask_cpu = mask.detach().cpu().bool()
    texts = []
    for ids_row, mask_row in zip(ids_cpu, mask_cpu):
        texts.append(tokenizer.decode(ids_row[mask_row].tolist(), skip_special_tokens=True))
    return texts


def _mask_after_lengths(predicted_ids: torch.Tensor, lengths: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Pad generated token ids after each sample's expected reconstruction length."""
    lengths = lengths.to(device=predicted_ids.device, dtype=torch.long)
    positions = torch.arange(predicted_ids.shape[1], device=predicted_ids.device).unsqueeze(0)
    keep = positions < lengths.unsqueeze(1)
    return torch.where(keep, predicted_ids, torch.full_like(predicted_ids, pad_token_id))


@torch.no_grad()
def _clean_plan_latent(
    model: nn.Module,
    x0: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    tokenizer,
    sentence_encoder,
    config,
) -> Optional[torch.Tensor]:
    """Build the clean sentence-plan latent used by reconstruction eval."""
    if not bool(getattr(config, "use_sentence_plan", False)):
        return None

    sentence_encoder_type = getattr(config, "sentence_encoder_type", "sentence_t5")
    device = x0.device
    dtype = x0.dtype
    if sentence_encoder_type == "sentence_t5":
        if sentence_encoder is None:
            raise ValueError("sentence_encoder is required for sentence_encoder_type='sentence_t5'")
        continuation_texts = _decode_selected_texts(input_ids, loss_mask, tokenizer)
        return sentence_encoder.encode(continuation_texts, device=device, dtype=dtype)

    if sentence_encoder_type == "learned":
        t_one = torch.ones((x0.shape[0],), dtype=dtype, device=device)
        encoder_sc_cfg_scale = (
            torch.ones_like(t_one) if config.num_self_cond_cfg_tokens > 0 else None
        )
        use_bf16 = bool(getattr(config, "use_bf16", True)) and device.type == "cuda"
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            _, _, plan_z = model(
                x0, t_one,
                attention_mask=loss_mask,
                deterministic=True,
                self_cond_cfg_scale=encoder_sc_cfg_scale,
                learned_plan_encode=True,
                return_plan=True,
            )
        return plan_z.to(dtype)

    raise ValueError(f"Unknown sentence_encoder_type: {sentence_encoder_type}")


@torch.no_grad()
def _teacher_forced_token_stats(
    model: nn.Module,
    x0: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    config,
    self_cond_cfg_scale: float,
    plan_z: Optional[torch.Tensor] = None,
    cond_seq_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked decoder NLL sum/count at clean token and plan latents."""
    batch_size = x0.shape[0]
    t_one = torch.ones((batch_size,), dtype=x0.dtype, device=x0.device)
    sc_batch = (
        torch.full_like(t_one, float(self_cond_cfg_scale))
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    model_input = (
        torch.cat([x0, torch.zeros_like(x0)], dim=-1)
        if config.self_cond_prob > 0 else x0
    )
    plan_kwargs = {}
    if bool(getattr(config, "use_sentence_plan", False)):
        if plan_z is None:
            raise ValueError("plan_z is required for sentence-plan teacher forcing")
        plan_kwargs = {"plan_z": plan_z, "plan_t": t_one}
    topology_kwargs = {}
    if str(getattr(config, "plan_attention_topology", "joint")) in {
        "hierarchical_prefix",
        "strict_hierarchical_prefix",
    }:
        if cond_seq_mask is None:
            cond_seq_mask = torch.zeros_like(attention_mask)
        topology_kwargs = {"cond_seq_mask": cond_seq_mask}
    use_bf16 = bool(getattr(config, "use_bf16", True)) and x0.is_cuda
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        _, decoder_logits = model(
            model_input,
            t_one,
            attention_mask=attention_mask,
            deterministic=True,
            self_cond_cfg_scale=sc_batch,
            decoder_step_active=True,
            **topology_kwargs, **plan_kwargs,
        )
    token_nll = torch.nn.functional.cross_entropy(
        decoder_logits.float().transpose(1, 2), input_ids, reduction="none",
    )
    mask = loss_mask.to(token_nll.dtype)
    return (token_nll * mask).sum(), mask.sum()


@torch.no_grad()
def _token_denoising_l2_stats(
    model: nn.Module,
    x0: torch.Tensor,
    noise: torch.Tensor,
    t_value: float,
    loss_mask: torch.Tensor,
    config,
    self_cond_cfg_scale: float,
    plan_z: Optional[torch.Tensor] = None,
    cond_seq_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked token-field velocity MSE sum/count at one fixed timestep."""
    batch_size = x0.shape[0]
    t_batch = torch.full(
        (batch_size,), float(t_value), dtype=x0.dtype, device=x0.device,
    )
    cond_mask = (
        torch.zeros_like(loss_mask) if cond_seq_mask is None else cond_seq_mask
    )
    cond_seq = x0 if cond_seq_mask is not None else torch.zeros_like(x0)
    z = add_noise(
        x0, noise, t_batch, config, cond_seq_mask=cond_mask.unsqueeze(-1),
    )
    plan_t = torch.ones_like(t_batch) if plan_z is not None else None
    v_pred, _, _, _ = _forward_sample(
        model=model,
        z=z,
        t_batch=t_batch,
        x_pred_prev=torch.zeros_like(x0),
        config=config,
        cfg_scale=1.0,
        self_cond_cfg_scale=float(self_cond_cfg_scale),
        cond_seq=cond_seq,
        cond_seq_mask=cond_mask,
        plan_z=plan_z,
        plan_t=plan_t,
    )
    v_target = x0 - noise * float(config.denoiser_noise_scale)
    squared = (v_pred.float() - v_target.float()).pow(2).mean(dim=-1)
    mask = loss_mask.to(squared.dtype)
    return (squared * mask).sum(), mask.sum()


def _token_reconstruction_run_name(self_cond_cfg_scale: float, suffix: str) -> str:
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    return f"clean-token-reconstruction{sccfg_str}-{suffix}"


def _plan_conditioned_run_name(sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
                               time_schedule, sde_gamma, plan_mode: str, suffix: str,
                               plan_sampling_mode="joint", plan_num_sampling_steps=None):
    return _build_run_name(
        sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
        time_schedule, sde_gamma, suffix=f"{plan_mode}-plan-{suffix}",
        plan_sampling_mode=plan_sampling_mode,
        plan_num_sampling_steps=plan_num_sampling_steps,
    )


# ============================================
# Generation Helper
# ============================================
def run_generation(
    state,
    encoder: nn.Module,
    eval_dataset,
    tokenizer,
    config,
    generator: torch.Generator,
    local_batch_size: int,
    train_dataset=None,
    sentence_encoder=None,
):
    """Run generation, and optionally plan/token reconstruction diagnostics."""
    if bool(getattr(config, "split_input_as_prefix", False)) and eval_dataset is None:
        if train_dataset is None:
            raise ValueError(
                "split_input_as_prefix evaluation requires data_path or eval_data_path"
            )
        # Input-only corpora become conditional evaluation datasets through
        # the deterministic collator split below.
        eval_dataset = train_dataset

    for sc_idx, sc in enumerate(config.sampling_configs):
        if len(config.sampling_configs) > 1:
            log_for_0(f"\n--- Sampling config {sc_idx + 1}/{len(config.sampling_configs)} ---")
        common_kwargs = dict(
            state=state,
            tokenizer=tokenizer,
            generator=generator,
            config=config,
            sampling_config=sc,
            batch_size=local_batch_size,
            num_samples=config.num_samples,
        )
        if eval_dataset is None:
            test_generation_uncond(
                **common_kwargs, reference_dataset=train_dataset,
            )
        else:
            test_generation_cond(
                **common_kwargs, encoder=encoder, dataset=eval_dataset,
                sentence_encoder=sentence_encoder,
            )

    if bool(getattr(config, "reconstruction_eval", False)):
        reconstruction_dataset = eval_dataset if eval_dataset is not None else train_dataset
        if reconstruction_dataset is None:
            log_for_0("Skipping reconstruction/plan eval: no dataset was provided.")
            return

        reconstruction_num_samples = getattr(config, "reconstruction_num_samples", None)
        if reconstruction_num_samples is None:
            reconstruction_num_samples = config.num_samples

        if bool(getattr(config, "use_sentence_plan", False)):
            for sc_idx, sc in enumerate(config.sampling_configs):
                if len(config.sampling_configs) > 1:
                    log_for_0(f"\n--- Oracle/shuffled plan config {sc_idx + 1}/{len(config.sampling_configs)} ---")
                # Oracle and shuffled are a paired intervention: restore the
                # same token-noise streams before each mode so the plan is the
                # only intended difference.
                paired_rng_state = _capture_rng_state(generator)
                for plan_mode in ("oracle", "shuffled"):
                    _restore_rng_state(generator, paired_rng_state)
                    test_plan_conditioned_generation(
                        state=state, encoder=encoder, tokenizer=tokenizer,
                        generator=generator, config=config, sampling_config=sc,
                        dataset=reconstruction_dataset, sentence_encoder=sentence_encoder,
                        is_conditional=eval_dataset is not None,
                        plan_mode=plan_mode,
                        num_samples=int(reconstruction_num_samples),
                        batch_size=local_batch_size,
                    )
        else:
            log_for_0("Skipping oracle/shuffled plan PPL: use_sentence_plan=False.")

        sc_scales = sorted({
            float(scale)
            for sc in config.sampling_configs
            for scale in getattr(sc, "self_cond_cfg_scales", [1.0])
        })
        for self_cond_cfg_scale in sc_scales:
            test_token_reconstruction_clean(
                state=state, encoder=encoder, tokenizer=tokenizer,
                generator=generator,
                config=config, dataset=reconstruction_dataset,
                sentence_encoder=sentence_encoder,
                is_conditional=eval_dataset is not None,
                num_samples=int(reconstruction_num_samples),
                batch_size=local_batch_size,
                self_cond_cfg_scale=self_cond_cfg_scale,
            )


# ============================================
# Unconditional generation
# ============================================
def test_generation_uncond(
    state,
    tokenizer,
    generator: torch.Generator,
    config: Config,
    sampling_config: SamplingConfig,
    reference_dataset=None,
    num_samples: int = 64,
    batch_size: int = 64,
):
    """Test unconditional generation."""
    sampling_method = sampling_config.sampling_method
    time_schedule = sampling_config.time_schedule
    log_for_0(f"Config: {sampling_config}")

    log_for_0("\n" + "=" * 70)
    log_for_0("              UNCONDITIONAL GENERATION EXAMPLES")
    log_for_0("=" * 70)

    model = _build_eval_model(state, use_compile=bool(getattr(config, "use_compile", False)))
    device = next(model.parameters()).device
    d_model = model.text_encoder_dim
    log_for_0(f"Per-device batch size: {batch_size}")

    pad_token_id = get_pad_token_id(tokenizer)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    cfg_list = [1]
    steps_list = sampling_config.num_sampling_steps
    self_cond_cfg_scales_list = sampling_config.self_cond_cfg_scales
    wandb_tables = {}
    ppl_metrics = None
    if config.online_eval:
        ppl_metrics = PPLMetrics(
            gen_ppl_eval_model_name_or_path=config.eval_ppl_model,
            eval_ppl_batch_size=config.eval_ppl_batch_size,
            eval_context_size=config.eval_ppl_max_length,
        )
    mauve_evaluator = _build_mauve_evaluator(config, ppl_metrics)

    world = _world()
    rank = _rank()
    param_dtype = next(model.parameters()).dtype

    eval_specs = list(itertools.product(
        steps_list, cfg_list, self_cond_cfg_scales_list
    ))
    for spec_idx, (num_sampling_steps, cfg_scale, self_cond_cfg_scale) in enumerate(eval_specs):
        log_for_0(f"\n--- Method: {sampling_method}, Steps: {num_sampling_steps}, "
                  f"CFG Scale: {cfg_scale}, SC-CFG: {self_cond_cfg_scale} ---")

        # Shard work across ranks: each rank generates ceil(num_samples/world);
        # the extras are trimmed after the gather on rank 0.
        local_num_samples = (num_samples + world - 1) // world
        local_generated = []
        generation_time = 0.0
        decode_time = 0.0
        num_batches = (local_num_samples + batch_size - 1) // batch_size
        local_processed = 0
        token_generator, plan_generator = _evaluation_generators(generator, device)

        for batch_idx in tqdm(range(num_batches), desc="Generating samples", disable=(rank != 0)):
            if local_processed >= local_num_samples:
                break
            current_batch = min(batch_size, local_num_samples - local_processed)
            z = torch.randn(
                (current_batch, config.max_length, d_model),
                generator=token_generator, dtype=param_dtype, device=device,
            ) * config.denoiser_noise_scale
            # Draw the initial token state before schedule samples so matched
            # joint/plan-first variants share the same initial token noise even
            # when they use different token-step counts.
            t_steps = get_sampling_steps(
                n_steps=num_sampling_steps,
                time_schedule=time_schedule,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, dtype=param_dtype,
                generator=token_generator,
            )

            gen_start = time.time()
            latent_out = _generate_samples_single_batch(
                model=model, generator=token_generator, plan_generator=plan_generator,
                z=z, t_steps=t_steps,
                cond_seq=None, cond_seq_mask=None,
                config=config, sampling_config=sampling_config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
            )
            if isinstance(latent_out, tuple):
                latent, plan_latent = latent_out
            else:
                latent, plan_latent = latent_out, None
            generation_time += time.time() - gen_start

            dec_start = time.time()
            t_final_val = t_steps[-1].item()
            predicted_ids = _dlm_decode_batch(
                z=latent, model=model, t_final_val=t_final_val,
                config=config, self_cond_cfg_scale=self_cond_cfg_scale,
                plan_z=plan_latent,
            )
            decode_time += time.time() - dec_start

            predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)

            for i in range(predicted_ids.shape[0]):
                if local_processed >= local_num_samples:
                    break
                text = tokenizer.decode(predicted_ids[i].detach().cpu().numpy(), skip_special_tokens=True)
                local_generated.append(text)
                local_processed += 1

        # Gather shards to rank 0, then assemble final ID-tagged list.
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, local_generated)
            if rank == 0:
                merged = []
                for shard in gathered:
                    merged.extend(shard)
                all_generated = [(i, txt) for i, txt in enumerate(merged[:num_samples])]
            else:
                all_generated = []
        else:
            all_generated = [(i, txt) for i, txt in enumerate(local_generated[:num_samples])]

        log_for_0(f"Generation: {generation_time:.2f}s ({num_sampling_steps} steps) | Decode: {decode_time:.2f}s")
        log_for_0("-" * 70)

        epoch_val = int(state.epoch)
        step_val = int(state.step)
        name = _build_run_name(
            sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
            time_schedule, getattr(sampling_config, "sde_gamma", 0.0), suffix="uncond",
            plan_sampling_mode=getattr(sampling_config, "plan_sampling_mode", "joint"),
            plan_num_sampling_steps=getattr(sampling_config, "plan_num_sampling_steps", None),
        )

        out_path = os.path.join(config.output_dir, name, f"all_generated_{epoch_val}_{step_val}.jsonl")
        if _rank() == 0:
            os.makedirs(os.path.join(config.output_dir, name), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for tid, gen in all_generated:
                    f.write(json.dumps({"id": tid, "generated": gen}, ensure_ascii=False) + "\n")
            log_for_0(f"Saved {len(all_generated)} generated texts to {out_path}")
            upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation")

        ppl_results = None
        mauve_results = None
        if config.online_eval and _rank() == 0:
            if len(eval_specs) == 1:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            log_for_0("\n" + "=" * 70)
            log_for_0("              GENERATION PPL EVALUATION (gPPL)")
            log_for_0("=" * 70)
            ppl_metrics.reset()
            with open(out_path, "r", encoding="utf-8") as f:
                text_samples = [json.loads(line)["generated"] for line in f]
            nonempty_samples = [s for s in text_samples if isinstance(s, str) and s.strip()]
            reference_texts = _dataset_reference_texts(
                reference_dataset,
                tokenizer,
                count=len(text_samples),
                max_length=config.eval_ppl_max_length,
            )
            mauve_generated, mauve_references = _paired_nonempty_texts(
                text_samples, reference_texts,
            )
            skipped = len(text_samples) - len(nonempty_samples)
            if skipped > 0:
                log_for_0(f"PPL eval: skipped {skipped} empty samples")
            if not nonempty_samples:
                log_for_0("PPL eval: all samples empty; skipping perplexity computation")
            else:
                ppl_results = ppl_metrics.record_generative_perplexity(
                    text_samples=nonempty_samples,
                    max_length=config.eval_ppl_max_length,
                    retokenize=True,
                    return_features=(
                        bool(config.eval_mauve)
                        and nonempty_samples == mauve_generated
                        and len(mauve_generated) >= 2
                    ),
                )
                log_for_0(f"gPPL: {ppl_results['ppl']:.4f}")
                log_for_0(f"Generation Mean Entropy: {ppl_results['mean_entropy']:.4f}")
            if bool(config.eval_mauve):
                if reference_dataset is None:
                    log_for_0("MAUVE eval: no real-text reference dataset; skipping")
                else:
                    generated_features = None
                    if (
                        ppl_results is not None
                        and nonempty_samples == mauve_generated
                    ):
                        generated_features = ppl_results.get("features")
                    mauve_results = _compute_mauve_metrics(
                        mauve_evaluator,
                        mauve_generated,
                        mauve_references,
                        config,
                        generated_features=(
                            generated_features if mauve_evaluator is ppl_metrics else None
                        ),
                    )
                    if mauve_results is not None:
                        log_for_0(f"MAUVE: {mauve_results['mauve']:.2f}")
            log_for_0("=" * 70 + "\n")

        if _rank() == 0:
            if ppl_results is not None or mauve_results is not None:
                metrics_line = {
                    "epoch": epoch_val, "step": step_val,
                    "mode": "generation_refine_decode",
                    "sampling_dimensions": _evaluation_sampling_dimensions(
                        sampling_config,
                        num_sampling_steps=num_sampling_steps,
                        cfg_scale=cfg_scale,
                        self_cond_cfg_scale=self_cond_cfg_scale,
                    ),
                    **_evaluation_rng_metadata(generator),
                }
                if ppl_results is not None:
                    metrics_line.update({
                        "ppl": ppl_results["ppl"], "g_ppl": ppl_results["ppl"],
                        "mean_entropy": ppl_results["mean_entropy"],
                    })
                if mauve_results is not None:
                    metrics_line.update(mauve_results)
                with open(os.path.join(config.output_dir, name, "metrics.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics_line, ensure_ascii=False) + "\n")
                upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation metrics")

            if config.use_wandb and wandb is not None:
                table = wandb.Table(columns=["sample_id", "text"])
                for tid, gen in all_generated[:min(10, len(all_generated))]:
                    table.add_data(tid, gen)
                wandb_tables[f"generated_samples_uncond_steps{num_sampling_steps}_cfg{cfg_scale}"] = table
                if ppl_results is not None:
                    wandb_tables.update({
                        f"generation/{name}/ppl": ppl_results["ppl"],
                        f"generation/{name}/g_ppl": ppl_results["ppl"],
                        f"generation/{name}/mean_entropy": ppl_results["mean_entropy"],
                    })
                if mauve_results is not None:
                    wandb_tables[f"generation/{name}/mauve"] = mauve_results["mauve"]

    if _rank() == 0 and config.use_wandb and wandb_tables and wandb is not None:
        try:
            wandb.log(wandb_tables)
        except Exception as e:
            log_for_0(f"Warning: wandb.log failed: {e}")
    log_for_0("=" * 70 + "\n")


# ============================================
# Conditional generation
# ============================================
def test_generation_cond(
    state,
    encoder: nn.Module,
    tokenizer,
    generator: torch.Generator,
    config: Config,
    sampling_config: SamplingConfig,
    dataset,
    sentence_encoder=None,
    num_samples: int = 64,
    batch_size: int = 64,
):
    """Test conditional generation."""
    sampling_method = sampling_config.sampling_method
    time_schedule = sampling_config.time_schedule
    log_for_0(f"Config: {sampling_config}")

    log_for_0("\n" + "=" * 70)
    log_for_0("              CONDITIONAL GENERATION EXAMPLES")
    log_for_0("=" * 70)

    model = _build_eval_model(state, use_compile=bool(getattr(config, "use_compile", False)))
    device = next(model.parameters()).device
    d_model = model.text_encoder_dim

    encode_latent_mean, encode_latent_std = config.latent_mean, config.latent_std
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id

    world = _world()
    rank = _rank()
    total_samples = min(int(num_samples), len(dataset))
    local_indices = list(range(rank, total_samples, world))
    local_dataset = _IndexedSubset(dataset, local_indices)
    dataloader = get_dataloader(
        local_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        split_input_as_prefix=bool(getattr(config, "split_input_as_prefix", False)),
        distributed=False,
    )
    log_for_0(
        f"Conditional eval samples: total={total_samples}, "
        f"per-rank~={(total_samples + world - 1) // world}, world={world}"
    )

    wandb_tables = {}
    ppl_metrics = None
    if config.online_eval:
        ppl_metrics = PPLMetrics(
            gen_ppl_eval_model_name_or_path=config.eval_ppl_model,
            eval_ppl_batch_size=config.eval_ppl_batch_size,
            eval_context_size=config.eval_ppl_max_length,
        )
    mauve_evaluator = _build_mauve_evaluator(config, ppl_metrics)
    cfg_list = sampling_config.cfgs
    steps_list = sampling_config.num_sampling_steps
    self_cond_cfg_scales_list = sampling_config.self_cond_cfg_scales
    collect_plan_diagnostics = bool(
        getattr(config, "eval_sampled_plan_diagnostics", False)
    )

    for num_sampling_steps, cfg_scale, self_cond_cfg_scale in itertools.product(
        steps_list, cfg_list, self_cond_cfg_scales_list
    ):
        log_for_0(f"\n--- Steps: {num_sampling_steps}, CFG Scale: {cfg_scale}, "
                  f"SC-CFG: {self_cond_cfg_scale} ---")

        local_generated = []
        local_plan_records = []
        generation_time = 0.0
        decode_time = 0.0
        samples_processed = 0
        token_generator, plan_generator = _evaluation_generators(generator, device)

        local_num_samples = len(local_indices)
        local_total_batches = (local_num_samples + batch_size - 1) // batch_size
        pbar = tqdm(total=local_total_batches, desc="Generating samples (cond)", disable=(rank != 0))
        for batch_idx, batch in enumerate(dataloader):
            if samples_processed >= local_num_samples:
                break
            bsz = batch["input_ids"].shape[0]
            input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
            encoder_attention_mask = torch.from_numpy(np.array(batch["encoder_attention_mask"])).to(device).float()
            attention_mask = torch.from_numpy(np.array(batch["attention_mask"])).to(device).float()
            cond_seq_mask_arr = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(device).float()

            cond_seq = encode_text(
                input_ids=input_ids, attention_mask=encoder_attention_mask,
                encoder=encoder, latent_mean=encode_latent_mean, latent_std=encode_latent_std,
            ).to(next(model.parameters()).dtype)

            z = torch.randn(
                (bsz, config.max_length, d_model), generator=token_generator,
                dtype=next(model.parameters()).dtype, device=device,
            ) * config.denoiser_noise_scale
            t_steps = get_sampling_steps(
                n_steps=num_sampling_steps,
                time_schedule=time_schedule,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, dtype=next(model.parameters()).dtype,
                generator=token_generator,
            )

            gen_start = time.time()
            latent_out = _generate_samples_single_batch(
                model=model, generator=token_generator, plan_generator=plan_generator,
                z=z, t_steps=t_steps,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask_arr,
                config=config, sampling_config=sampling_config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
            )
            if isinstance(latent_out, tuple):
                latent, plan_latent = latent_out
            else:
                latent, plan_latent = latent_out, None
            generation_time += time.time() - gen_start

            if (
                plan_latent is not None
                and collect_plan_diagnostics
            ):
                clean_plan_latent = _clean_plan_latent(
                    model=model,
                    x0=cond_seq,
                    input_ids=input_ids,
                    loss_mask=_batch_loss_mask(
                        attention_mask, cond_seq_mask_arr, config,
                    ),
                    tokenizer=tokenizer,
                    sentence_encoder=sentence_encoder,
                    config=config,
                )
                if clean_plan_latent is None:
                    raise ValueError(
                        "sampled plan diagnostics require a clean plan target"
                    )
                sample_ids = [int(i) for i in batch["index"]]
                for sample_id, sampled_item, clean_item in zip(
                    sample_ids, plan_latent, clean_plan_latent,
                ):
                    local_plan_records.append((
                        sample_id,
                        sampled_item.detach().float().cpu(),
                        clean_item.detach().float().cpu(),
                    ))

            gen_length = config.max_length - config.max_input_length
            cond_len_per_sample = cond_seq_mask_arr.to(torch.int32).sum(dim=1)

            dec_start = time.time()
            t_final_val = t_steps[-1].item()
            predicted_ids = _dlm_decode_batch(
                z=latent, model=model, t_final_val=t_final_val,
                config=config, self_cond_cfg_scale=self_cond_cfg_scale,
                plan_z=plan_latent,
                cond_seq_mask=cond_seq_mask_arr,
            )
            predicted_ids = shift_left(
                predicted_ids, cond_len_per_sample, pad_token_id,
            )[:, :gen_length]
            predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
            decode_time += time.time() - dec_start

            if "target" in batch and "input" in batch:
                original_texts = [batch["target"][i] for i in range(bsz)]
                context_texts = [batch["input"][i] for i in range(bsz)]
            else:
                original_texts = _decode_selected_texts(
                    input_ids, attention_mask * (1 - cond_seq_mask_arr), tokenizer,
                )
                context_texts = _decode_selected_texts(
                    input_ids, cond_seq_mask_arr, tokenizer,
                )
            sample_ids = [int(i) for i in batch["index"]]

            for i in range(bsz):
                if samples_processed >= local_num_samples:
                    break
                text = tokenizer.decode(predicted_ids[i].detach().cpu().numpy(), skip_special_tokens=True)
                local_generated.append((sample_ids[i], original_texts[i], text, context_texts[i]))
                samples_processed += 1
            pbar.update(1)
        pbar.close()

        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, local_generated)
            if rank == 0:
                all_generated = []
                for shard in gathered:
                    all_generated.extend(shard)
                all_generated.sort(key=lambda row: row[0])
                all_generated = all_generated[:total_samples]
            else:
                all_generated = []
            if collect_plan_diagnostics:
                gathered_plans = [None] * world
                dist.all_gather_object(gathered_plans, local_plan_records)
                if rank == 0:
                    all_plan_records = []
                    for shard in gathered_plans:
                        all_plan_records.extend(shard)
                    all_plan_records.sort(key=lambda row: row[0])
                    all_plan_records = all_plan_records[:total_samples]
                else:
                    all_plan_records = []
            else:
                all_plan_records = []
        else:
            all_generated = local_generated
            all_plan_records = (
                local_plan_records if collect_plan_diagnostics else []
            )

        log_for_0(f"Generation: {generation_time:.2f}s ({num_sampling_steps} steps) | Decode: {decode_time:.2f}s")
        log_for_0("-" * 70)

        epoch_val = int(state.epoch)
        step_val = int(state.step)
        name = _build_run_name(
            sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
            time_schedule, getattr(sampling_config, "sde_gamma", 0.0), suffix="cond",
            plan_sampling_mode=getattr(sampling_config, "plan_sampling_mode", "joint"),
            plan_num_sampling_steps=getattr(sampling_config, "plan_num_sampling_steps", None),
        )

        if _rank() == 0:
            os.makedirs(os.path.join(config.output_dir, name), exist_ok=True)
            out_path = os.path.join(config.output_dir, name, f"all_generated_{epoch_val}_{step_val}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for tid, orig, gen, ctx in all_generated:
                    f.write(json.dumps({
                        "id": tid,
                        "context": ctx,
                        "reference": orig,
                        "generated": gen,
                    }, ensure_ascii=False) + "\n")
            log_for_0(f"Saved {len(all_generated)} generated texts to {out_path}")
            upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation")

            cond_eval_results = None
            if all_plan_records:
                sampled_plan_metrics = _sampled_plan_diagnostics(
                    torch.stack([row[1] for row in all_plan_records]),
                    torch.stack([row[2] for row in all_plan_records]),
                )
                cond_eval_results = dict(sampled_plan_metrics)
                log_for_0(
                    "Sampled-plan diagnostics: "
                    f"var_ratio={sampled_plan_metrics.get('sampled_plan_var_ratio')} "
                    f"cosine={sampled_plan_metrics['sampled_clean_plan_cosine']:.4f} "
                    f"retrieval_top1="
                    f"{sampled_plan_metrics['sampled_clean_plan_retrieval_top1']:.4f}"
                )
            if config.online_eval and all_generated:
                hypotheses = [gen for _, _, gen, _ in all_generated]
                references = [orig for _, orig, _, _ in all_generated]
                bleu_score = compute_bleu(hypotheses, references)
                rouge_scores = compute_rouge(hypotheses, references)
                cond_eval_results = {
                    **(cond_eval_results or {}),
                    "bleu": bleu_score,
                    **rouge_scores,
                }
                nonempty_hypotheses = [
                    text for text in hypotheses if isinstance(text, str) and text.strip()
                ]
                mauve_generated, mauve_references = _paired_nonempty_texts(
                    hypotheses, references,
                )
                ppl_results = None
                if nonempty_hypotheses:
                    ppl_metrics.reset()
                    ppl_results = ppl_metrics.record_generative_perplexity(
                        text_samples=nonempty_hypotheses,
                        max_length=config.eval_ppl_max_length,
                        retokenize=True,
                        return_features=(
                            bool(config.eval_mauve)
                            and nonempty_hypotheses == mauve_generated
                            and len(mauve_generated) >= 2
                        ),
                    )
                    cond_eval_results.update({
                        "ppl": ppl_results["ppl"],
                        "g_ppl": ppl_results["ppl"],
                        "mean_entropy": ppl_results["mean_entropy"],
                    })
                if mauve_evaluator is not None:
                    mauve_results = _compute_mauve_metrics(
                        mauve_evaluator, mauve_generated, mauve_references, config,
                        generated_features=(
                            ppl_results.get("features")
                            if ppl_results is not None
                            and mauve_evaluator is ppl_metrics
                            and nonempty_hypotheses == mauve_generated
                            else None
                        ),
                    )
                    if mauve_results is not None:
                        cond_eval_results.update(mauve_results)
                log_for_0(
                    f"BLEU: {bleu_score:.2f}  ROUGE-1: {rouge_scores['rouge1']:.2f}  "
                    f"ROUGE-2: {rouge_scores['rouge2']:.2f}  ROUGE-L: {rouge_scores['rougeL']:.2f}"
                )
                if "mauve" in cond_eval_results:
                    log_for_0(f"MAUVE: {cond_eval_results['mauve']:.2f}")
                if "g_ppl" in cond_eval_results:
                    log_for_0(
                        f"Conditional gPPL: {cond_eval_results['g_ppl']:.4f}  "
                        f"Mean entropy: {cond_eval_results['mean_entropy']:.4f}"
                    )

            if config.use_wandb and wandb is not None:
                table = wandb.Table(columns=["sample_id", "context", "original", "generated"])
                for tid, orig, gen, ctx in all_generated[:min(10, len(all_generated))]:
                    table.add_data(tid, ctx, orig, gen)
                wandb_tables[f"generated_samples_cond_steps{num_sampling_steps}_cfg{cfg_scale}"] = table
                if cond_eval_results is not None:
                    if all(
                        key in cond_eval_results
                        for key in ("bleu", "rouge1", "rouge2", "rougeL")
                    ):
                        wandb_tables.update({
                            f"generation/{name}/bleu": cond_eval_results["bleu"],
                            f"generation/{name}/rouge1": cond_eval_results["rouge1"],
                            f"generation/{name}/rouge2": cond_eval_results["rouge2"],
                            f"generation/{name}/rougeL": cond_eval_results["rougeL"],
                        })
                    if "g_ppl" in cond_eval_results:
                        wandb_tables.update({
                            f"generation/{name}/g_ppl": cond_eval_results["g_ppl"],
                            f"generation/{name}/mean_entropy": cond_eval_results["mean_entropy"],
                        })
                    if "mauve" in cond_eval_results:
                        wandb_tables[f"generation/{name}/mauve"] = cond_eval_results["mauve"]
                    for key, value in cond_eval_results.items():
                        if key.startswith("sampled_") or key.startswith("clean_plan_"):
                            wandb_tables[f"generation/{name}/{key}"] = value
            if cond_eval_results is not None:
                metrics_line = {
                    "epoch": epoch_val,
                    "step": step_val,
                    "mode": "generation_refine_decode",
                    "sampling_dimensions": _evaluation_sampling_dimensions(
                        sampling_config,
                        num_sampling_steps=num_sampling_steps,
                        cfg_scale=cfg_scale,
                        self_cond_cfg_scale=self_cond_cfg_scale,
                    ),
                    **cond_eval_results,
                    **_evaluation_rng_metadata(generator),
                }
                with open(os.path.join(config.output_dir, name, "metrics.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics_line, ensure_ascii=False) + "\n")
                upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation metrics")

    if _rank() == 0 and config.use_wandb and wandb_tables and wandb is not None:
        try:
            wandb.log(wandb_tables)
        except Exception as e:
            log_for_0(f"Warning: wandb.log failed: {e}")
    log_for_0("=" * 70 + "\n")


# ============================================
# Oracle / shuffled sentence-plan generation
# ============================================
@torch.no_grad()
def test_plan_conditioned_generation(
    state,
    encoder: nn.Module,
    tokenizer,
    generator: torch.Generator,
    config: Config,
    sampling_config: SamplingConfig,
    dataset,
    sentence_encoder=None,
    is_conditional: bool = False,
    plan_mode: str = "oracle",
    num_samples: int = 64,
    batch_size: int = 64,
):
    """Generate token latents from noise while conditioning on clean/shuffled sentence plans."""
    if plan_mode not in {"oracle", "shuffled"}:
        raise ValueError(f"plan_mode must be 'oracle' or 'shuffled', got {plan_mode!r}")
    if not bool(getattr(config, "use_sentence_plan", False)):
        log_for_0(f"Skipping {plan_mode} plan generation: use_sentence_plan=False")
        return

    suffix = "cond" if is_conditional else "uncond"
    metric_key = f"{plan_mode}_plan_ppl"
    sampling_method = sampling_config.sampling_method
    time_schedule = sampling_config.time_schedule

    log_for_0("\n" + "=" * 70)
    log_for_0(f"              {plan_mode.upper()} SENTENCE-PLAN GENERATION")
    log_for_0("=" * 70)
    log_for_0(f"Config: {sampling_config}")

    model = _build_eval_model(state, use_compile=bool(getattr(config, "use_compile", False)))
    device = next(model.parameters()).device
    d_model = model.text_encoder_dim
    param_dtype = next(model.parameters()).dtype
    use_bf16 = bool(getattr(config, "use_bf16", True)) and device.type == "cuda"

    encode_latent_mean, encode_latent_std = config.latent_mean, config.latent_std
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    ppl_metrics = None
    if config.online_eval:
        ppl_metrics = PPLMetrics(
            gen_ppl_eval_model_name_or_path=config.eval_ppl_model,
            eval_ppl_batch_size=config.eval_ppl_batch_size,
            eval_context_size=config.eval_ppl_max_length,
        )
    mauve_evaluator = _build_mauve_evaluator(config, ppl_metrics)

    world = _world()
    rank = _rank()
    total_samples = min(int(num_samples), len(dataset))
    if plan_mode == "shuffled" and total_samples < 2:
        log_for_0("Skipping shuffled_plan_ppl: need at least 2 samples to build a mismatched plan.")
        return
    local_indices = list(range(rank, total_samples, world))
    local_dataset = _IndexedSubset(dataset, local_indices)
    dataloader = get_dataloader(
        local_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        split_input_as_prefix=bool(getattr(config, "split_input_as_prefix", False)),
        distributed=False,
    )
    plan_dataloader = None
    if plan_mode == "shuffled":
        shuffled_plan_indices = [int((idx + 1) % total_samples) for idx in local_indices]
        plan_dataset = _IndexedSubset(dataset, shuffled_plan_indices)
        plan_dataloader = get_dataloader(
            plan_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, drop_last=False,
            max_seq_length=config.max_length, pad_token_id=pad_token_id,
            max_input_seq_length=config.max_input_length,
            split_input_as_prefix=bool(getattr(config, "split_input_as_prefix", False)),
            distributed=False,
        )
    log_for_0(
        f"{plan_mode} plan samples: total={total_samples}, "
        f"per-rank~={(total_samples + world - 1) // world}, world={world}"
    )

    cfg_list = sampling_config.cfgs
    steps_list = sampling_config.num_sampling_steps
    self_cond_cfg_scales_list = sampling_config.self_cond_cfg_scales
    eval_specs = list(itertools.product(steps_list, cfg_list, self_cond_cfg_scales_list))
    wandb_payload = {}

    for num_sampling_steps, cfg_scale, self_cond_cfg_scale in eval_specs:
        log_for_0(f"\n--- {plan_mode} plan, Steps: {num_sampling_steps}, "
                  f"CFG Scale: {cfg_scale}, SC-CFG: {self_cond_cfg_scale} ---")

        local_generated = []
        encode_time = 0.0
        generation_time = 0.0
        decode_time = 0.0
        samples_processed = 0
        token_generator, plan_generator = _evaluation_generators(generator, device)
        local_num_samples = len(local_indices)
        local_total_batches = (local_num_samples + batch_size - 1) // batch_size
        pbar = tqdm(total=local_total_batches, desc=f"Generating with {plan_mode} plan", disable=(rank != 0))
        batch_iter = (
            zip(dataloader, plan_dataloader)
            if plan_dataloader is not None
            else ((batch, batch) for batch in dataloader)
        )

        for batch, plan_batch in batch_iter:
            if samples_processed >= local_num_samples:
                break

            bsz = batch["input_ids"].shape[0]
            input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
            encoder_attention_mask = torch.from_numpy(np.array(batch["encoder_attention_mask"])).to(device).float()
            attention_mask = torch.from_numpy(np.array(batch["attention_mask"])).to(device).float()
            cond_seq_mask_arr = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(device).float()
            loss_mask = _batch_loss_mask(attention_mask, cond_seq_mask_arr, config)

            enc_start = time.time()
            x0 = encode_text(
                input_ids=input_ids,
                attention_mask=encoder_attention_mask,
                encoder=encoder,
                latent_mean=encode_latent_mean,
                latent_std=encode_latent_std,
                use_bf16=use_bf16,
            ).to(param_dtype)
            if plan_mode == "shuffled":
                plan_input_ids = torch.from_numpy(np.array(plan_batch["input_ids"])).to(device).long()
                plan_encoder_attention_mask = (
                    torch.from_numpy(np.array(plan_batch["encoder_attention_mask"])).to(device).float()
                )
                plan_attention_mask = torch.from_numpy(np.array(plan_batch["attention_mask"])).to(device).float()
                plan_cond_seq_mask_arr = torch.from_numpy(np.array(plan_batch["cond_seq_mask"])).to(device).float()
                plan_loss_mask = _batch_loss_mask(plan_attention_mask, plan_cond_seq_mask_arr, config)
                plan_x0 = encode_text(
                    input_ids=plan_input_ids,
                    attention_mask=plan_encoder_attention_mask,
                    encoder=encoder,
                    latent_mean=encode_latent_mean,
                    latent_std=encode_latent_std,
                    use_bf16=use_bf16,
                ).to(param_dtype)
            else:
                plan_input_ids = input_ids
                plan_loss_mask = loss_mask
                plan_x0 = x0
            plan_latent = _clean_plan_latent(
                model=model, x0=plan_x0, input_ids=plan_input_ids, loss_mask=plan_loss_mask,
                tokenizer=tokenizer, sentence_encoder=sentence_encoder, config=config,
            )
            encode_time += time.time() - enc_start

            z = torch.randn(
                (bsz, config.max_length, d_model), generator=token_generator,
                dtype=param_dtype, device=device,
            ) * config.denoiser_noise_scale
            t_steps = get_sampling_steps(
                n_steps=num_sampling_steps,
                time_schedule=time_schedule,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, dtype=param_dtype,
                generator=token_generator,
            )

            cond_seq = x0 if is_conditional else None
            cond_seq_mask_for_sampling = cond_seq_mask_arr if is_conditional else None

            gen_start = time.time()
            latent_out = _generate_samples_single_batch(
                model=model, generator=token_generator, plan_generator=plan_generator,
                z=z, t_steps=t_steps,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask_for_sampling,
                config=config, sampling_config=sampling_config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
                initial_plan_z=plan_latent, fixed_plan_z=True,
            )
            latent, _ = latent_out
            generation_time += time.time() - gen_start

            dec_start = time.time()
            predicted_ids = _dlm_decode_batch(
                z=latent, model=model, t_final_val=t_steps[-1].item(),
                config=config, self_cond_cfg_scale=self_cond_cfg_scale,
                plan_z=plan_latent,
                cond_seq_mask=(cond_seq_mask_arr if is_conditional else None),
            )
            if is_conditional:
                if config.max_input_length is None:
                    raise ValueError("max_input_length is required for conditional oracle/shuffled plan eval")
                gen_length = config.max_length - config.max_input_length
                cond_len_per_sample = cond_seq_mask_arr.to(torch.int32).sum(dim=1)
                predicted_ids = shift_left(predicted_ids, cond_len_per_sample, pad_token_id)[:, :gen_length]
            predicted_ids = mask_after_eos(
                predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
            )
            decode_time += time.time() - dec_start

            sample_ids = [int(i) for i in batch["index"]]
            if is_conditional:
                if "target" in batch and "input" in batch:
                    reference_texts = [batch["target"][i] for i in range(bsz)]
                    context_texts = [batch["input"][i] for i in range(bsz)]
                else:
                    reference_texts = _decode_selected_texts(
                        input_ids, attention_mask * (1 - cond_seq_mask_arr), tokenizer,
                    )
                    context_texts = _decode_selected_texts(
                        input_ids, cond_seq_mask_arr, tokenizer,
                    )
            else:
                reference_texts = _decode_selected_texts(input_ids, attention_mask, tokenizer)
                context_texts = [""] * bsz

            for i in range(bsz):
                if samples_processed >= local_num_samples:
                    break
                text = tokenizer.decode(predicted_ids[i].detach().cpu().numpy(), skip_special_tokens=True)
                local_generated.append((sample_ids[i], reference_texts[i], text, context_texts[i]))
                samples_processed += 1
            pbar.update(1)
        pbar.close()

        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, local_generated)
            if rank == 0:
                all_generated = []
                for shard in gathered:
                    all_generated.extend(shard)
                all_generated.sort(key=lambda row: row[0])
                all_generated = all_generated[:total_samples]
            else:
                all_generated = []
        else:
            all_generated = local_generated

        log_for_0(
            f"Encode plan: {encode_time:.2f}s | Generation: {generation_time:.2f}s "
            f"({num_sampling_steps} steps) | Decode: {decode_time:.2f}s"
        )
        log_for_0("-" * 70)

        epoch_val = int(state.epoch)
        step_val = int(state.step)
        name = _plan_conditioned_run_name(
            sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
            time_schedule, getattr(sampling_config, "sde_gamma", 0.0),
            plan_mode=plan_mode, suffix=suffix,
            plan_sampling_mode=getattr(sampling_config, "plan_sampling_mode", "joint"),
            plan_num_sampling_steps=getattr(sampling_config, "plan_num_sampling_steps", None),
        )
        out_path = os.path.join(config.output_dir, name, f"all_generated_{epoch_val}_{step_val}.jsonl")

        ppl_results = None
        similarity_results = None
        if _rank() == 0:
            os.makedirs(os.path.join(config.output_dir, name), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for tid, ref, gen, ctx in all_generated:
                    row = {
                        "id": tid,
                        "generated": gen,
                        "reference": ref,
                        "mode": f"{plan_mode}_plan_generation",
                    }
                    if is_conditional:
                        row["context"] = ctx
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_for_0(f"Saved {len(all_generated)} {plan_mode} plan generations to {out_path}")
            upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason=f"{plan_mode} plan generation")

            if config.online_eval:
                if len(eval_specs) == 1:
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                log_for_0("\n" + "=" * 70)
                log_for_0(f"              {plan_mode.upper()} PLAN PPL EVALUATION")
                log_for_0("=" * 70)
                ppl_metrics.reset()
                with open(out_path, "r", encoding="utf-8") as f:
                    text_samples = [json.loads(line)["generated"] for line in f]
                nonempty_samples = [s for s in text_samples if isinstance(s, str) and s.strip()]
                reference_samples = [ref for _, ref, _, _ in all_generated]
                mauve_generated, mauve_references = _paired_nonempty_texts(
                    text_samples, reference_samples,
                )
                skipped = len(text_samples) - len(nonempty_samples)
                if skipped > 0:
                    log_for_0(f"{plan_mode} plan PPL eval: skipped {skipped} empty samples")
                if nonempty_samples:
                    ppl_results = ppl_metrics.record_generative_perplexity(
                        text_samples=nonempty_samples,
                        max_length=config.eval_ppl_max_length,
                        retokenize=True,
                        return_features=(
                            bool(config.eval_mauve)
                            and nonempty_samples == mauve_generated
                            and len(mauve_generated) >= 2
                        ),
                    )
                    log_for_0(f"{metric_key}: {ppl_results['ppl']:.4f}")
                    log_for_0(f"{plan_mode} plan Mean Entropy: {ppl_results['mean_entropy']:.4f}")
                else:
                    log_for_0(f"{plan_mode} plan PPL eval: all samples empty; skipping perplexity computation")
                log_for_0("=" * 70 + "\n")

                if all_generated:
                    hypotheses = [gen for _, _, gen, _ in all_generated]
                    references = [ref for _, ref, _, _ in all_generated]
                    bleu_score = compute_bleu(hypotheses, references)
                    rouge_scores = compute_rouge(hypotheses, references)
                    similarity_results = {"bleu": bleu_score, **rouge_scores}
                    if bool(config.eval_mauve):
                        generated_features = None
                        if (
                            ppl_results is not None
                            and nonempty_samples == mauve_generated
                        ):
                            generated_features = ppl_results.get("features")
                        mauve_results = _compute_mauve_metrics(
                            mauve_evaluator,
                            mauve_generated,
                            mauve_references,
                            config,
                            generated_features=(
                                generated_features if mauve_evaluator is ppl_metrics else None
                            ),
                        )
                        if mauve_results is not None:
                            similarity_results.update(mauve_results)
                    log_for_0(
                        f"{plan_mode} plan BLEU: {bleu_score:.2f}  "
                        f"ROUGE-1: {rouge_scores['rouge1']:.2f}  "
                        f"ROUGE-2: {rouge_scores['rouge2']:.2f}  "
                        f"ROUGE-L: {rouge_scores['rougeL']:.2f}"
                    )
                    if "mauve" in similarity_results:
                        log_for_0(f"{plan_mode} plan MAUVE: {similarity_results['mauve']:.2f}")

            metrics_line = {
                "epoch": epoch_val,
                "step": step_val,
                "mode": f"{plan_mode}_plan_generation",
                "sampling_dimensions": _evaluation_sampling_dimensions(
                    sampling_config,
                    num_sampling_steps=num_sampling_steps,
                    cfg_scale=cfg_scale,
                    self_cond_cfg_scale=self_cond_cfg_scale,
                ),
                **_evaluation_rng_metadata(generator),
            }
            if ppl_results is not None:
                metrics_line.update({
                    "ppl": ppl_results["ppl"],
                    metric_key: ppl_results["ppl"],
                    "mean_entropy": ppl_results["mean_entropy"],
                })
            if similarity_results is not None:
                metrics_line.update(similarity_results)
            if len(metrics_line) > 3:
                with open(os.path.join(config.output_dir, name, "metrics.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics_line, ensure_ascii=False) + "\n")
                upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason=f"{plan_mode} plan metrics")

            if config.use_wandb and wandb is not None:
                table_cols = ["sample_id", "reference", "generated"]
                if is_conditional:
                    table_cols.insert(1, "context")
                table = wandb.Table(columns=table_cols)
                for tid, ref, gen, ctx in all_generated[:min(10, len(all_generated))]:
                    if is_conditional:
                        table.add_data(tid, ctx, ref, gen)
                    else:
                        table.add_data(tid, ref, gen)
                wandb_payload[f"{plan_mode}_plan_samples_{suffix}_steps{num_sampling_steps}_cfg{cfg_scale}"] = table
                if ppl_results is not None:
                    wandb_payload.update({
                        f"{plan_mode}_plan/{name}/ppl": ppl_results["ppl"],
                        f"{plan_mode}_plan/{name}/{metric_key}": ppl_results["ppl"],
                        f"{plan_mode}_plan/{name}/mean_entropy": ppl_results["mean_entropy"],
                    })
                if similarity_results is not None:
                    wandb_payload.update({
                        f"{plan_mode}_plan/{name}/bleu": similarity_results["bleu"],
                        f"{plan_mode}_plan/{name}/rouge1": similarity_results["rouge1"],
                        f"{plan_mode}_plan/{name}/rouge2": similarity_results["rouge2"],
                        f"{plan_mode}_plan/{name}/rougeL": similarity_results["rougeL"],
                    })
                    if "mauve" in similarity_results:
                        wandb_payload[f"{plan_mode}_plan/{name}/mauve"] = similarity_results["mauve"]

    if _rank() == 0 and config.use_wandb and wandb_payload and wandb is not None:
        try:
            wandb.log(wandb_payload)
        except Exception as e:
            log_for_0(f"Warning: wandb.log failed: {e}")
    log_for_0("=" * 70 + "\n")


# ============================================
# Clean-token reconstruction
# ============================================
@torch.no_grad()
def test_token_reconstruction_clean(
    state,
    encoder: nn.Module,
    tokenizer,
    generator: torch.Generator,
    config: Config,
    dataset,
    sentence_encoder=None,
    is_conditional: bool = False,
    num_samples: int = 64,
    batch_size: int = 64,
    self_cond_cfg_scale: float = 1.0,
):
    """Decode clean T5 token latents directly; decoder/token-autoencoding sanity metric."""
    suffix = "cond" if is_conditional else "uncond"
    log_for_0("\n" + "=" * 70)
    log_for_0("              CLEAN-TOKEN RECONSTRUCTION EXAMPLES")
    log_for_0("=" * 70)
    log_for_0(f"Self-cond CFG: {self_cond_cfg_scale}")

    model = _build_eval_model(state, use_compile=bool(getattr(config, "use_compile", False)))
    device = next(model.parameters()).device
    param_dtype = next(model.parameters()).dtype
    use_bf16 = bool(getattr(config, "use_bf16", True)) and device.type == "cuda"

    encode_latent_mean, encode_latent_std = config.latent_mean, config.latent_std
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    world = _world()
    rank = _rank()
    total_samples = min(int(num_samples), len(dataset))
    local_indices = list(range(rank, total_samples, world))
    local_dataset = _IndexedSubset(dataset, local_indices)
    dataloader = get_dataloader(
        local_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        split_input_as_prefix=bool(getattr(config, "split_input_as_prefix", False)),
        distributed=False,
    )
    plan_dataloader = None
    if bool(getattr(config, "use_sentence_plan", False)):
        if total_samples < 2:
            raise ValueError("sentence-plan reconstruction diagnostics require at least 2 samples")
        shuffled_plan_indices = [int((idx + 1) % total_samples) for idx in local_indices]
        plan_dataloader = get_dataloader(
            _IndexedSubset(dataset, shuffled_plan_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            max_seq_length=config.max_length,
            pad_token_id=pad_token_id,
            max_input_seq_length=config.max_input_length,
            split_input_as_prefix=bool(getattr(config, "split_input_as_prefix", False)),
            distributed=False,
        )
    log_for_0(
        f"Clean token reconstruction samples: total={total_samples}, "
        f"per-rank~={(total_samples + world - 1) // world}, world={world}"
    )

    local_reconstructed = []
    encode_time = 0.0
    decode_time = 0.0
    samples_processed = 0
    local_num_samples = len(local_indices)
    local_total_batches = (local_num_samples + batch_size - 1) // batch_size
    pbar = tqdm(total=local_total_batches, desc="Reconstructing clean token latents", disable=(rank != 0))
    token_generator, _ = _evaluation_generators(generator, device)
    timestep_probes = (0.1, 0.3, 0.5, 0.7, 0.9)
    local_diagnostic_stats: dict[str, list[torch.Tensor]] = {}

    def _accumulate(key: str, value_sum: torch.Tensor, value_count: torch.Tensor) -> None:
        if key not in local_diagnostic_stats:
            local_diagnostic_stats[key] = [value_sum.detach(), value_count.detach()]
        else:
            local_diagnostic_stats[key][0] += value_sum.detach()
            local_diagnostic_stats[key][1] += value_count.detach()

    batch_iter = (
        zip(dataloader, plan_dataloader)
        if plan_dataloader is not None
        else ((batch, None) for batch in dataloader)
    )

    for batch, plan_batch in batch_iter:
        if samples_processed >= local_num_samples:
            break

        bsz = batch["input_ids"].shape[0]
        input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
        encoder_attention_mask = torch.from_numpy(np.array(batch["encoder_attention_mask"])).to(device).float()
        attention_mask = torch.from_numpy(np.array(batch["attention_mask"])).to(device).float()
        cond_seq_mask_arr = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(device).float()
        loss_mask = _batch_loss_mask(attention_mask, cond_seq_mask_arr, config)

        enc_start = time.time()
        x0 = encode_text(
            input_ids=input_ids,
            attention_mask=encoder_attention_mask,
            encoder=encoder,
            latent_mean=encode_latent_mean,
            latent_std=encode_latent_std,
            use_bf16=use_bf16,
        ).to(param_dtype)
        plan_latent = _clean_plan_latent(
            model=model, x0=x0, input_ids=input_ids, loss_mask=loss_mask,
            tokenizer=tokenizer, sentence_encoder=sentence_encoder, config=config,
        )
        shuffled_plan_latent = None
        if plan_batch is not None:
            plan_input_ids = torch.from_numpy(np.array(plan_batch["input_ids"])).to(device).long()
            plan_encoder_attention_mask = (
                torch.from_numpy(np.array(plan_batch["encoder_attention_mask"])).to(device).float()
            )
            plan_attention_mask = torch.from_numpy(np.array(plan_batch["attention_mask"])).to(device).float()
            plan_cond_seq_mask = torch.from_numpy(np.array(plan_batch["cond_seq_mask"])).to(device).float()
            plan_loss_mask = _batch_loss_mask(plan_attention_mask, plan_cond_seq_mask, config)
            plan_x0 = encode_text(
                input_ids=plan_input_ids,
                attention_mask=plan_encoder_attention_mask,
                encoder=encoder,
                latent_mean=encode_latent_mean,
                latent_std=encode_latent_std,
                use_bf16=use_bf16,
            ).to(param_dtype)
            shuffled_plan_latent = _clean_plan_latent(
                model=model,
                x0=plan_x0,
                input_ids=plan_input_ids,
                loss_mask=plan_loss_mask,
                tokenizer=tokenizer,
                sentence_encoder=sentence_encoder,
                config=config,
            )
        encode_time += time.time() - enc_start

        plan_variants = (
            {"oracle": plan_latent, "shuffled": shuffled_plan_latent}
            if plan_latent is not None else {"clean": None}
        )
        for plan_mode, diagnostic_plan in plan_variants.items():
            nll_sum, token_count = _teacher_forced_token_stats(
                model=model,
                x0=x0,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                config=config,
                self_cond_cfg_scale=self_cond_cfg_scale,
                plan_z=diagnostic_plan,
                cond_seq_mask=(cond_seq_mask_arr if is_conditional else None),
            )
            _accumulate(f"{plan_mode}:teacher_nll", nll_sum, token_count)

        for t_value in timestep_probes:
            noise = torch.randn(
                x0.shape,
                generator=token_generator,
                dtype=x0.dtype,
                device=x0.device,
            )
            for plan_mode, diagnostic_plan in plan_variants.items():
                l2_sum, l2_count = _token_denoising_l2_stats(
                    model=model,
                    x0=x0,
                    noise=noise,
                    t_value=t_value,
                    loss_mask=loss_mask,
                    config=config,
                    self_cond_cfg_scale=self_cond_cfg_scale,
                    plan_z=diagnostic_plan,
                    cond_seq_mask=(cond_seq_mask_arr if is_conditional else None),
                )
                _accumulate(f"{plan_mode}:l2:t{int(t_value * 100):02d}", l2_sum, l2_count)

        dec_start = time.time()
        predicted_ids = _dlm_decode_batch(
            z=x0, model=model, t_final_val=1.0,
            config=config, self_cond_cfg_scale=self_cond_cfg_scale,
            plan_z=plan_latent,
            cond_seq_mask=(cond_seq_mask_arr if is_conditional else None),
        )
        target_lengths = loss_mask.to(torch.int32).sum(dim=1)
        if is_conditional:
            if config.max_input_length is None:
                raise ValueError("max_input_length is required for conditional token reconstruction eval")
            gen_length = config.max_length - config.max_input_length
            cond_len_per_sample = cond_seq_mask_arr.to(torch.int32).sum(dim=1)
            predicted_ids = shift_left(predicted_ids, cond_len_per_sample, pad_token_id)[:, :gen_length]
        predicted_ids = _mask_after_lengths(predicted_ids, target_lengths, pad_token_id)
        predicted_ids = mask_after_eos(
            predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id,
        )
        decode_time += time.time() - dec_start

        sample_ids = [int(i) for i in batch["index"]]
        if is_conditional:
            if "target" in batch and "input" in batch:
                reference_texts = [batch["target"][i] for i in range(bsz)]
                context_texts = [batch["input"][i] for i in range(bsz)]
            else:
                reference_texts = _decode_selected_texts(
                    input_ids, attention_mask * (1 - cond_seq_mask_arr), tokenizer,
                )
                context_texts = _decode_selected_texts(
                    input_ids, cond_seq_mask_arr, tokenizer,
                )
        else:
            reference_texts = _decode_selected_texts(input_ids, attention_mask, tokenizer)
            context_texts = [""] * bsz

        for i in range(bsz):
            if samples_processed >= local_num_samples:
                break
            text = tokenizer.decode(predicted_ids[i].detach().cpu().numpy(), skip_special_tokens=True)
            local_reconstructed.append((sample_ids[i], reference_texts[i], text, context_texts[i]))
            samples_processed += 1
        pbar.update(1)
    pbar.close()

    for value_sum, value_count in local_diagnostic_stats.values():
        if world > 1:
            dist.all_reduce(value_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(value_count, op=dist.ReduceOp.SUM)

    diagnostic_metrics: dict[str, float] = {}
    diagnostic_modes = ("oracle", "shuffled") if plan_dataloader is not None else ("clean",)
    for plan_mode in diagnostic_modes:
        nll_sum, token_count = local_diagnostic_stats[f"{plan_mode}:teacher_nll"]
        mean_nll = float((nll_sum / token_count.clamp_min(1)).item())
        prefix = "" if plan_mode == "clean" else f"{plan_mode}_plan_"
        diagnostic_metrics[f"{prefix}teacher_forced_token_ppl"] = float(
            torch.exp(torch.tensor(mean_nll, dtype=torch.float64)).item()
        )
        l2_sum_total = 0.0
        l2_count_total = 0.0
        for t_value in timestep_probes:
            bin_name = f"t{int(t_value * 100):02d}"
            l2_sum, l2_count = local_diagnostic_stats[f"{plan_mode}:l2:{bin_name}"]
            value = float((l2_sum / l2_count.clamp_min(1)).item())
            diagnostic_metrics[f"{prefix}token_denoising_l2_{bin_name}"] = value
            l2_sum_total += float(l2_sum.item())
            l2_count_total += float(l2_count.item())
        diagnostic_metrics[f"{prefix}token_denoising_l2"] = l2_sum_total / max(l2_count_total, 1.0)
    if "oracle_plan_teacher_forced_token_ppl" in diagnostic_metrics:
        diagnostic_metrics["teacher_forced_token_ppl"] = diagnostic_metrics[
            "oracle_plan_teacher_forced_token_ppl"
        ]
        diagnostic_metrics["token_denoising_l2"] = diagnostic_metrics[
            "oracle_plan_token_denoising_l2"
        ]
        for t_value in timestep_probes:
            bin_name = f"t{int(t_value * 100):02d}"
            diagnostic_metrics[f"token_denoising_l2_{bin_name}"] = diagnostic_metrics[
                f"oracle_plan_token_denoising_l2_{bin_name}"
            ]

    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, local_reconstructed)
        if rank == 0:
            all_reconstructed = []
            for shard in gathered:
                all_reconstructed.extend(shard)
            all_reconstructed.sort(key=lambda row: row[0])
            all_reconstructed = all_reconstructed[:total_samples]
        else:
            all_reconstructed = []
    else:
        all_reconstructed = local_reconstructed

    log_for_0(f"Encode token latent: {encode_time:.2f}s | Decode: {decode_time:.2f}s")
    log_for_0("-" * 70)

    epoch_val = int(state.epoch)
    step_val = int(state.step)
    name = _token_reconstruction_run_name(self_cond_cfg_scale, suffix)
    out_path = os.path.join(config.output_dir, name, f"all_token_reconstructed_{epoch_val}_{step_val}.jsonl")

    ppl_results = None
    similarity_results = None
    if _rank() == 0:
        os.makedirs(os.path.join(config.output_dir, name), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for tid, ref, recon, ctx in all_reconstructed:
                row = {
                    "id": tid,
                    "generated": recon,
                    "reference": ref,
                    "mode": "clean_token_reconstruction",
                }
                if is_conditional:
                    row["context"] = ctx
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log_for_0(f"Saved {len(all_reconstructed)} clean-token reconstructed texts to {out_path}")
        upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="token reconstruction")

        if config.online_eval:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            log_for_0("\n" + "=" * 70)
            log_for_0("              TOKEN RECONSTRUCTION PPL EVALUATION")
            log_for_0("=" * 70)
            with open(out_path, "r", encoding="utf-8") as f:
                text_samples = [json.loads(line)["generated"] for line in f]
            nonempty_samples = [s for s in text_samples if isinstance(s, str) and s.strip()]
            skipped = len(text_samples) - len(nonempty_samples)
            if skipped > 0:
                log_for_0(f"Token reconstruction PPL eval: skipped {skipped} empty samples")
            if nonempty_samples:
                ppl_metrics = PPLMetrics(
                    gen_ppl_eval_model_name_or_path=config.eval_ppl_model,
                    eval_ppl_batch_size=config.eval_ppl_batch_size,
                    eval_context_size=config.eval_ppl_max_length,
                )
                ppl_metrics.reset()
                ppl_results = ppl_metrics.record_generative_perplexity(
                    text_samples=nonempty_samples,
                    max_length=config.eval_ppl_max_length,
                    retokenize=True,
                )
                log_for_0(f"Token reconstruction PPL: {ppl_results['ppl']:.4f}")
                log_for_0(f"Token reconstruction Mean Entropy: {ppl_results['mean_entropy']:.4f}")
            else:
                log_for_0("Token reconstruction PPL eval: all samples empty; skipping perplexity computation")
            log_for_0("=" * 70 + "\n")

            if all_reconstructed:
                hypotheses = [recon for _, _, recon, _ in all_reconstructed]
                references = [ref for _, ref, _, _ in all_reconstructed]
                bleu_score = compute_bleu(hypotheses, references)
                rouge_scores = compute_rouge(hypotheses, references)
                similarity_results = {"bleu": bleu_score, **rouge_scores}
                log_for_0(
                    f"Token reconstruction BLEU: {bleu_score:.2f}  "
                    f"ROUGE-1: {rouge_scores['rouge1']:.2f}  "
                    f"ROUGE-2: {rouge_scores['rouge2']:.2f}  "
                    f"ROUGE-L: {rouge_scores['rougeL']:.2f}"
                )

        metrics_line = {
            "epoch": epoch_val, "step": step_val,
            "mode": "clean_token_reconstruction",
            "model_active_depth": int(
                getattr(config, "model_active_depth", None)
                or getattr(config, "model_depth", None)
                or {"ELF-B": 12, "ELF-M": 24, "ELF-L": 32}[config.model]
            ),
            **_evaluation_rng_metadata(generator),
            **diagnostic_metrics,
        }
        if ppl_results is not None:
            metrics_line.update({
                "ppl": ppl_results["ppl"],
                "token_recon_ppl": ppl_results["ppl"],
                "mean_entropy": ppl_results["mean_entropy"],
            })
        if similarity_results is not None:
            metrics_line.update(similarity_results)
        if len(metrics_line) > 3:
            with open(os.path.join(config.output_dir, name, "metrics.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics_line, ensure_ascii=False) + "\n")
            upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="token reconstruction metrics")

        if config.use_wandb and wandb is not None:
            wandb_payload = {}
            table_cols = ["sample_id", "reference", "reconstructed"]
            if is_conditional:
                table_cols.insert(1, "context")
            table = wandb.Table(columns=table_cols)
            for tid, ref, recon, ctx in all_reconstructed[:min(10, len(all_reconstructed))]:
                if is_conditional:
                    table.add_data(tid, ctx, ref, recon)
                else:
                    table.add_data(tid, ref, recon)
            wandb_payload[f"token_reconstruction_samples_clean_{suffix}_sccfg{self_cond_cfg_scale}"] = table
            if ppl_results is not None:
                wandb_payload.update({
                    f"token_reconstruction/{name}/ppl": ppl_results["ppl"],
                    f"token_reconstruction/{name}/token_recon_ppl": ppl_results["ppl"],
                    f"token_reconstruction/{name}/mean_entropy": ppl_results["mean_entropy"],
                })
            if similarity_results is not None:
                wandb_payload.update({
                    f"token_reconstruction/{name}/bleu": similarity_results["bleu"],
                    f"token_reconstruction/{name}/rouge1": similarity_results["rouge1"],
                    f"token_reconstruction/{name}/rouge2": similarity_results["rouge2"],
                    f"token_reconstruction/{name}/rougeL": similarity_results["rougeL"],
                })
            try:
                wandb.log(wandb_payload)
            except Exception as e:
                log_for_0(f"Warning: wandb.log failed: {e}")

    log_for_0("=" * 70 + "\n")
