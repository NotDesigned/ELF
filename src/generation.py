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
from tqdm import tqdm

from configs.config import Config, SamplingConfig
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import upload_output_dir_to_hf
from utils.train_utils import unwrap_model
from utils.data_utils import get_dataloader, get_pad_token_id
from utils.encoder_utils import encode_text
from utils.metrics_utils import Metrics as PPLMetrics, compute_bleu, compute_rouge
from utils.sampling_utils import get_sampling_steps
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


def _token_reconstruction_run_name(self_cond_cfg_scale: float, suffix: str) -> str:
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    return f"clean-token-reconstruction{sccfg_str}-{suffix}"


def _plan_conditioned_run_name(sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
                               time_schedule, sde_gamma, plan_mode: str, suffix: str):
    return _build_run_name(
        sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
        time_schedule, sde_gamma, suffix=f"{plan_mode}-plan-{suffix}",
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
            test_generation_uncond(**common_kwargs)
        else:
            test_generation_cond(
                **common_kwargs, encoder=encoder, dataset=eval_dataset,
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
                for plan_mode in ("oracle", "shuffled"):
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

        for batch_idx in tqdm(range(num_batches), desc="Generating samples", disable=(rank != 0)):
            if local_processed >= local_num_samples:
                break
            current_batch = min(batch_size, local_num_samples - local_processed)
            t_steps = get_sampling_steps(
                n_steps=num_sampling_steps,
                time_schedule=time_schedule,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, dtype=param_dtype,
            )
            if device.type == "cuda":
                z = torch.randn(
                    (current_batch, config.max_length, d_model),
                    dtype=param_dtype, device=device,
                ) * config.denoiser_noise_scale
            else:
                z = (torch.randn((current_batch, config.max_length, d_model),
                                 generator=generator, dtype=param_dtype)
                     * config.denoiser_noise_scale).to(device)

            gen_start = time.time()
            latent_out = _generate_samples_single_batch(
                model=model, generator=generator, z=z, t_steps=t_steps,
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
                )
                log_for_0(f"gPPL: {ppl_results['ppl']:.4f}")
                log_for_0(f"Generation Mean Entropy: {ppl_results['mean_entropy']:.4f}")
            log_for_0("=" * 70 + "\n")

        if _rank() == 0:
            if ppl_results is not None:
                metrics_line = {
                    "epoch": epoch_val, "step": step_val,
                    "mode": "generation_refine_decode",
                    "ppl": ppl_results["ppl"], "g_ppl": ppl_results["ppl"],
                    "mean_entropy": ppl_results["mean_entropy"],
                }
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
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    log_for_0(
        f"Conditional eval samples: total={total_samples}, "
        f"per-rank~={(total_samples + world - 1) // world}, world={world}"
    )

    wandb_tables = {}
    cfg_list = sampling_config.cfgs
    steps_list = sampling_config.num_sampling_steps
    self_cond_cfg_scales_list = sampling_config.self_cond_cfg_scales

    for num_sampling_steps, cfg_scale, self_cond_cfg_scale in itertools.product(
        steps_list, cfg_list, self_cond_cfg_scales_list
    ):
        log_for_0(f"\n--- Steps: {num_sampling_steps}, CFG Scale: {cfg_scale}, "
                  f"SC-CFG: {self_cond_cfg_scale} ---")

        local_generated = []
        generation_time = 0.0
        decode_time = 0.0
        samples_processed = 0

        local_num_samples = len(local_indices)
        local_total_batches = (local_num_samples + batch_size - 1) // batch_size
        pbar = tqdm(total=local_total_batches, desc="Generating samples (cond)", disable=(rank != 0))
        for batch_idx, batch in enumerate(dataloader):
            if samples_processed >= local_num_samples:
                break
            bsz = batch["input_ids"].shape[0]
            input_ids = torch.from_numpy(np.array(batch["input_ids"])).to(device).long()
            encoder_attention_mask = torch.from_numpy(np.array(batch["encoder_attention_mask"])).to(device).float()
            cond_seq_mask_arr = torch.from_numpy(np.array(batch["cond_seq_mask"])).to(device).float()

            t_steps = get_sampling_steps(
                n_steps=num_sampling_steps,
                time_schedule=time_schedule,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, dtype=next(model.parameters()).dtype,
            )

            cond_seq = encode_text(
                input_ids=input_ids, attention_mask=encoder_attention_mask,
                encoder=encoder, latent_mean=encode_latent_mean, latent_std=encode_latent_std,
            ).to(next(model.parameters()).dtype)

            z = (torch.randn((bsz, config.max_length, d_model),
                             generator=generator, dtype=next(model.parameters()).dtype)
                 * config.denoiser_noise_scale).to(device)

            gen_start = time.time()
            latent_out = _generate_samples_single_batch(
                model=model, generator=generator, z=z, t_steps=t_steps,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask_arr,
                config=config, sampling_config=sampling_config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
            )
            if isinstance(latent_out, tuple):
                latent, plan_latent = latent_out
            else:
                latent, plan_latent = latent_out, None
            generation_time += time.time() - gen_start

            gen_length = config.max_length - config.max_input_length
            cond_len_per_sample = cond_seq_mask_arr.to(torch.int32).sum(dim=1)

            dec_start = time.time()
            t_final_val = t_steps[-1].item()
            predicted_ids = _dlm_decode_batch(
                z=latent, model=model, t_final_val=t_final_val,
                config=config, self_cond_cfg_scale=self_cond_cfg_scale,
                plan_z=plan_latent,
            )
            predicted_ids = shift_left(predicted_ids, cond_len_per_sample, 0)[:, :gen_length]
            predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
            decode_time += time.time() - dec_start

            original_texts = [batch["target"][i] for i in range(bsz)]
            context_texts = [batch["input"][i] for i in range(bsz)]
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
        else:
            all_generated = local_generated

        log_for_0(f"Generation: {generation_time:.2f}s ({num_sampling_steps} steps) | Decode: {decode_time:.2f}s")
        log_for_0("-" * 70)

        epoch_val = int(state.epoch)
        step_val = int(state.step)
        name = _build_run_name(
            sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
            time_schedule, getattr(sampling_config, "sde_gamma", 0.0), suffix="cond",
        )

        if _rank() == 0:
            os.makedirs(os.path.join(config.output_dir, name), exist_ok=True)
            out_path = os.path.join(config.output_dir, name, f"all_generated_{epoch_val}_{step_val}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for tid, orig, gen, ctx in all_generated:
                    f.write(json.dumps({"id": tid, "generated": gen}, ensure_ascii=False) + "\n")
            log_for_0(f"Saved {len(all_generated)} generated texts to {out_path}")
            upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="generation")

            cond_eval_results = None
            if config.online_eval and all_generated:
                hypotheses = [gen for _, _, gen, _ in all_generated]
                references = [orig for _, orig, _, _ in all_generated]
                bleu_score = compute_bleu(hypotheses, references)
                rouge_scores = compute_rouge(hypotheses, references)
                cond_eval_results = {"bleu": bleu_score, **rouge_scores}
                log_for_0(
                    f"BLEU: {bleu_score:.2f}  ROUGE-1: {rouge_scores['rouge1']:.2f}  "
                    f"ROUGE-2: {rouge_scores['rouge2']:.2f}  ROUGE-L: {rouge_scores['rougeL']:.2f}"
                )

            if config.use_wandb and wandb is not None:
                table = wandb.Table(columns=["sample_id", "context", "original", "generated"])
                for tid, orig, gen, ctx in all_generated[:min(10, len(all_generated))]:
                    table.add_data(tid, ctx, orig, gen)
                wandb_tables[f"generated_samples_cond_steps{num_sampling_steps}_cfg{cfg_scale}"] = table
                if cond_eval_results is not None:
                    wandb_tables.update({
                        f"generation/{name}/bleu": cond_eval_results["bleu"],
                        f"generation/{name}/rouge1": cond_eval_results["rouge1"],
                        f"generation/{name}/rouge2": cond_eval_results["rouge2"],
                        f"generation/{name}/rougeL": cond_eval_results["rougeL"],
                    })
            if cond_eval_results is not None:
                metrics_line = {"epoch": epoch_val, "step": step_val, **cond_eval_results}
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
        max_input_seq_length=config.max_input_length, distributed=False,
    )
    plan_dataloader = None
    if plan_mode == "shuffled":
        shuffled_plan_indices = [int((idx + 1) % total_samples) for idx in local_indices]
        plan_dataset = _IndexedSubset(dataset, shuffled_plan_indices)
        plan_dataloader = get_dataloader(
            plan_dataset, batch_size=batch_size,
            shuffle=False, num_workers=0, drop_last=False,
            max_seq_length=config.max_length, pad_token_id=pad_token_id,
            max_input_seq_length=config.max_input_length, distributed=False,
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

            t_steps = get_sampling_steps(
                n_steps=num_sampling_steps,
                time_schedule=time_schedule,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, dtype=param_dtype,
            )
            if device.type == "cuda":
                z = torch.randn((bsz, config.max_length, d_model), dtype=param_dtype, device=device)
                z = z * config.denoiser_noise_scale
            else:
                z = (
                    torch.randn((bsz, config.max_length, d_model), generator=generator, dtype=param_dtype)
                    * config.denoiser_noise_scale
                ).to(device)

            cond_seq = x0 if is_conditional else None
            cond_seq_mask_for_sampling = cond_seq_mask_arr if is_conditional else None

            gen_start = time.time()
            latent_out = _generate_samples_single_batch(
                model=model, generator=generator, z=z, t_steps=t_steps,
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
                reference_texts = [batch["target"][i] for i in range(bsz)]
                context_texts = [batch["input"][i] for i in range(bsz)]
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
                skipped = len(text_samples) - len(nonempty_samples)
                if skipped > 0:
                    log_for_0(f"{plan_mode} plan PPL eval: skipped {skipped} empty samples")
                if nonempty_samples:
                    ppl_results = ppl_metrics.record_generative_perplexity(
                        text_samples=nonempty_samples,
                        max_length=config.eval_ppl_max_length,
                        retokenize=True,
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
                    log_for_0(
                        f"{plan_mode} plan BLEU: {bleu_score:.2f}  "
                        f"ROUGE-1: {rouge_scores['rouge1']:.2f}  "
                        f"ROUGE-2: {rouge_scores['rouge2']:.2f}  "
                        f"ROUGE-L: {rouge_scores['rougeL']:.2f}"
                    )

            metrics_line = {
                "epoch": epoch_val,
                "step": step_val,
                "mode": f"{plan_mode}_plan_generation",
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
        max_input_seq_length=config.max_input_length, distributed=False,
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

    for batch in dataloader:
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
        encode_time += time.time() - enc_start

        dec_start = time.time()
        predicted_ids = _dlm_decode_batch(
            z=x0, model=model, t_final_val=1.0,
            config=config, self_cond_cfg_scale=self_cond_cfg_scale,
            plan_z=plan_latent,
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
            reference_texts = [batch["target"][i] for i in range(bsz)]
            context_texts = [batch["input"][i] for i in range(bsz)]
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
