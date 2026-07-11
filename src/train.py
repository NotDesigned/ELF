#!/usr/bin/env python
"""Training script for the ELF."""

import argparse
import logging
import os
import sys
import time

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer

from modules.t5_encoder import get_encoder
from modules.sentence_plan import build_sentence_plan_encoder
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import (
    save_checkpoint, load_checkpoint, load_warm_start_checkpoint, find_latest_checkpoint,
)
from utils.train_utils import (
    TrainState, prefetch_to_device, get_optimizer, create_learning_rate_fn,
    attach_lr_scheduler, local_rank_zero_first,
)
from generation import run_generation
from configs.config import (
    load_config_from_yaml,
    apply_config_overrides,
    load_sampling_configs,
    resolve_batch_sizes,
    SamplingConfig,
)
from modules.model import ELF_models
from utils.data_utils import get_dataloader, prepare_batch, load_dataset, get_pad_token_id
from train_step import train_step

try:
    import wandb
except ImportError:
    wandb = None

# Logging: no timestamps; suppress noisy checkpoint loggers; unbuffered stdout
logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)


def _init_distributed():
    """Initialize torch.distributed if launched via torchrun."""
    if "WORLD_SIZE" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="gloo")


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _format_param_count(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    return str(num_params)


def _count_named_params(model, prefixes, *, trainable_only: bool = False) -> int:
    prefixes = tuple(prefixes)
    return sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith(prefixes) and (not trainable_only or p.requires_grad)
    )


def _log_model_parameter_summary(model) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_for_0(
        f"ELF parameters: {total_params:,} ({_format_param_count(total_params)}) | "
        f"trainable={total_trainable:,} ({_format_param_count(total_trainable)})"
    )

    plan_prefixes = (
        "plan_tokens",
        "plan_in.",
        "plan_in_dit.",
        "plan_out_input.",
        "plan_out_dit.",
        "plan_time_embedder.",
        "plan_norm.",
        "plan_out.",
        "plan_encoder_query",
        "plan_encoder_output_norm.",
    )
    plan_params = _count_named_params(model, plan_prefixes)
    if plan_params == 0:
        return

    base_params = total_params - plan_params
    log_for_0(
        f"Base ELF parameters excluding sentence plan: {base_params:,} "
        f"({_format_param_count(base_params)})"
    )
    log_for_0(
        f"Sentence plan parameters: {plan_params:,} "
        f"({_format_param_count(plan_params)}, {plan_params / total_params:.1%} of model)"
    )

    plan_groups = [
        (
            "SlotMLPPlanAdapter",
            ("plan_tokens", "plan_in.", "plan_time_embedder.", "plan_norm.", "plan_out."),
        ),
        ("SlotDiTPlanAdapter", ("plan_in_dit.", "plan_out_input.", "plan_out_dit.")),
        ("LearnedPlanEncoderExtras", ("plan_encoder_query", "plan_encoder_output_norm.")),
    ]
    for label, prefixes in plan_groups:
        count = _count_named_params(model, prefixes)
        if count:
            log_for_0(f"  - {label}: {count:,} ({_format_param_count(count)})")


def _run_train_sampling_eval_if_due(
    *,
    state,
    encoder,
    eval_dataset,
    train_dataset,
    tokenizer,
    config,
    generator,
    local_batch_size: int,
    sentence_encoder,
    global_step: int,
) -> bool:
    """Run lightweight generation+PPL during training when the step schedule asks for it."""
    eval_freq = int(getattr(config, "train_sampling_eval_freq", 0))
    if eval_freq <= 0 or global_step <= 0 or global_step % eval_freq != 0:
        return False

    original_output_dir = config.output_dir
    original_num_samples = config.num_samples
    original_reconstruction_eval = getattr(config, "reconstruction_eval", False)
    original_reconstruction_num_samples = getattr(config, "reconstruction_num_samples", None)
    original_sampling_configs = config.sampling_configs

    eval_num_samples = int(getattr(config, "train_sampling_eval_num_samples", 64))
    eval_batch_size = int(getattr(config, "train_sampling_eval_batch_size", min(local_batch_size, eval_num_samples)))
    max_configs = int(getattr(config, "train_sampling_eval_max_configs", 1))
    metric_names = ["gPPL"]
    if bool(getattr(config, "use_sentence_plan", False)):
        metric_names.extend(["oracle_plan_ppl", "shuffled_plan_ppl"])
    metric_names.append("token_recon_ppl")

    try:
        config.output_dir = os.path.join(original_output_dir, "train_sampling_eval")
        config.num_samples = eval_num_samples
        config.reconstruction_eval = True
        config.reconstruction_num_samples = eval_num_samples
        config.sampling_configs = list(original_sampling_configs[:max_configs])
        if not config.sampling_configs:
            raise ValueError("train sampling eval has no sampling config to run")

        log_for_0(
            "Running train-time sampling eval: "
            f"step={global_step}, samples={eval_num_samples}, "
            f"batch={eval_batch_size}, configs={len(config.sampling_configs)}, "
            f"metrics={'+'.join(metric_names)}"
        )
        run_generation(
            state=state, encoder=encoder, eval_dataset=eval_dataset,
            tokenizer=tokenizer, config=config, generator=generator,
            local_batch_size=eval_batch_size,
            train_dataset=train_dataset,
            sentence_encoder=sentence_encoder,
        )
        return True
    finally:
        config.output_dir = original_output_dir
        config.num_samples = original_num_samples
        config.reconstruction_eval = original_reconstruction_eval
        config.reconstruction_num_samples = original_reconstruction_num_samples
        config.sampling_configs = original_sampling_configs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Train ELF Diffusion Model (PyTorch).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file to override defaults.")
    parser.add_argument(
        "--config_override", action="append", default=[],
        help="Override config values (field_name=value). Can be specified multiple times.",
    )
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def run_training(config, *, force_cpu: bool = False):
    _init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cpu") if force_cpu or not torch.cuda.is_available() else torch.device(f"cuda:{local_rank}")
    rank = _rank()
    world = _world_size()

    log_for_0("=" * 60)
    log_for_0("ELF Diffusion Model Training (PyTorch)")
    log_for_0("=" * 60)
    log_for_0(f"Model: {config.model}")
    log_for_0(f"Encoder Model: {config.encoder_model_name}")
    log_for_0(f"Encoder Checkpoint: {config.encoder_checkpoint}")
    log_for_0(f"Data: {config.data_path}")
    log_for_0(f"Max sequence length: {config.max_length}")
    log_for_0(f"Output dir: {config.output_dir}")
    log_for_0(f"HF Repo ID: {config.hf_repo_id}")
    log_for_0(f"Resume: {config.resume}")
    log_for_0(
        f"Warm start: {getattr(config, 'warm_start', None)} "
        f"(use_ema={bool(getattr(config, 'warm_start_use_ema', False))})"
    )
    log_for_0(f"Batch size per device: {config.batch_size}")
    log_for_0(f"Number of epochs: {config.epochs}")
    log_for_0(
        f"Eval PPL: online={config.online_eval}, model={config.eval_ppl_model}, "
        f"max_length={config.eval_ppl_max_length}, batch={config.eval_ppl_batch_size}"
    )
    log_for_0(
        f"Reconstruction diagnostics: enabled={config.reconstruction_eval}, "
        f"samples={config.reconstruction_num_samples or config.num_samples}"
    )
    log_for_0(f"PyTorch device: {device}, world_size={world}")
    log_for_0(f"BF16 autocast: {bool(getattr(config, 'use_bf16', True)) and device.type == 'cuda'}")
    log_for_0(f"Gradient checkpointing: {bool(getattr(config, 'gradient_checkpointing', True))}")
    if bool(getattr(config, "use_sentence_plan", False)):
        log_for_0(
            "Sentence plan: "
            f"type={config.sentence_encoder_type}, adapter={getattr(config, 'plan_adapter_type', 'slot_mlp')}, "
            f"slots={config.num_plan_tokens}, dim={config.sentence_emb_dim}, "
            f"time={getattr(config, 'plan_time_schedule', 'aligned')}"
            f"(gamma={getattr(config, 'plan_time_warp_gamma', 1.0)}), "
            f"grad={getattr(config, 'sentence_encoder_grad', 'none')}, "
            f"aux_passes={getattr(config, 'plan_aux_passes', 1)}, "
            f"aux_context={getattr(config, 'plan_aux_token_context', 'denoiser_z')}"
        )
    log_for_0("=" * 60)

    if config.use_wandb and rank == 0 and wandb is not None:
        wandb_config = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
        wandb_tags = config.wandb_tag.split(",") if config.wandb_tag else None
        wandb_run_id = getattr(config, "wandb_run_id", None)
        wandb_resume = getattr(config, "wandb_resume", None)
        wandb_kwargs = dict(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            tags=wandb_tags,
            config=wandb_config,
            dir=os.environ.get("WANDB_DIR", "/tmp"),
        )
        if wandb_run_id:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = wandb_resume or "allow"
        elif wandb_resume:
            wandb_kwargs["resume"] = wandb_resume
        wandb.init(**wandb_kwargs)
        run_id = getattr(wandb.run, "id", wandb_run_id)
        resume_suffix = f" (resume={wandb_kwargs.get('resume')}, id={run_id})"
        log_for_0(f"Wandb initialized: {wandb.run.url}{resume_suffix}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    # Per-rank seed so stochastic draws (decoder/denoiser branch coin,
    # timesteps, noise) diverge across ranks. A shared seed would make every
    # rank take the same branch in lockstep, producing spiky decoder gradients
    # instead of an evenly-mixed CE/L2 reduction.
    g = torch.Generator(device="cpu").manual_seed(config.seed + rank)

    # TF32 for fp32 matmuls on Ampere/Hopper (no hyperparameter change).
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    with local_rank_zero_first():
        log_for_0("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
        pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
        log_for_0(f"Using {'EOS' if config.pad_token == 'eos' else 'PAD'} token for padding: {pad_token_id}")

        train_dataset, eval_dataset = load_dataset(config)

        log_for_0(f"Loading Encoder config: {config.encoder_model_name}...")
        encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
        encoder = encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        log_for_0(f"Encoder d_model: {encoder_config.d_model}")

        sentence_encoder = build_sentence_plan_encoder(config, device)
        if sentence_encoder is not None:
            if sentence_encoder.embedding_dim != int(config.sentence_emb_dim):
                raise ValueError(
                    f"Sentence-T5 dim {sentence_encoder.embedding_dim} does not match "
                    f"config.sentence_emb_dim={config.sentence_emb_dim}"
                )
            log_for_0(f"Sentence-T5 encoder loaded: dim={sentence_encoder.embedding_dim}")

    log_for_0(f"Creating {config.model} model...")
    # Use the full tokenizer length for CE heads; tokenizer.vocab_size can exclude
    # added special tokens that still appear in tokenized Qwen targets.
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    log_for_0(f"Tokenizer vocab: CE head={vocab_size}")
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
        gradient_checkpointing=bool(getattr(config, "gradient_checkpointing", True)),
        use_sentence_plan=bool(getattr(config, "use_sentence_plan", False)),
        sentence_encoder_type=getattr(config, "sentence_encoder_type", "sentence_t5"),
        sentence_emb_dim=int(getattr(config, "sentence_emb_dim", 768)),
        num_plan_tokens=int(getattr(config, "num_plan_tokens", 8)),
        plan_adapter_type=getattr(config, "plan_adapter_type", "slot_mlp"),
        plan_slot_dit_depth=int(getattr(config, "plan_slot_dit_depth", 2)),
        plan_learned_encoder_norm=bool(getattr(config, "plan_learned_encoder_norm", True)),
    ).to(device)

    _log_model_parameter_summary(model)

    # Keep initialization identical across ranks, then make runtime stochastic
    # ops (e.g. dropout) rank-specific.
    torch.manual_seed(config.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + rank)

    if config.global_batch_size is not None:
        log_for_0(f"Using global batch size: {config.global_batch_size}")
    elif config.batch_size is not None:
        log_for_0(f"Using batch size per device: {config.batch_size}")
    total_batch_size, local_batch_size = resolve_batch_sizes(config, world, context="training")

    steps_per_epoch = len(train_dataset) // total_batch_size
    if steps_per_epoch <= 0:
        raise ValueError(
            f"Training dataset has {len(train_dataset)} examples, smaller than "
            f"global_batch_size={total_batch_size}; steps_per_epoch would be 0."
        )
    num_train_steps = steps_per_epoch * config.epochs
    if config.warmup_steps >= 0:
        num_warmup_steps = config.warmup_steps
    elif config.warmup_epochs is not None:
        num_warmup_steps = int(config.warmup_epochs * steps_per_epoch)
    else:
        num_warmup_steps = 0

    # Gradient accumulation: LR schedule is parameterized in optimizer steps
    grad_accum_steps = config.grad_accum_steps
    num_optimizer_steps = num_train_steps // grad_accum_steps
    num_warmup_optimizer_steps = num_warmup_steps // grad_accum_steps

    # Effective learning rate (scaled with effective batch size, including grad accum)
    if config.lr is None or config.lr <= 0:
        if config.lr is not None:
            log_for_0(f"Configured lr={config.lr} is non-positive; recomputing from blr={config.blr}")
        config.lr = config.blr * (total_batch_size * grad_accum_steps) / 256

    log_for_0(
        f"World={world} | batch local={local_batch_size}, total={total_batch_size} | "
        f"steps/epoch={steps_per_epoch}, total_train={num_train_steps}, "
        f"warmup={num_warmup_steps}, lr={config.lr:.2e}"
    )
    if grad_accum_steps > 1:
        log_for_0(
            f"Grad accum={grad_accum_steps}, effective batch={total_batch_size * grad_accum_steps}, "
            f"optimizer steps={num_optimizer_steps}"
        )

    lr_fn = create_learning_rate_fn(
        num_train_steps=num_optimizer_steps, num_warmup_steps=num_warmup_optimizer_steps,
        learning_rate=config.lr, schedule=config.lr_schedule, min_lr=config.min_lr,
    )
    optimizer = get_optimizer(model, config, lr=config.lr, grad_accum_steps=grad_accum_steps)
    lr_scheduler = attach_lr_scheduler(optimizer, lr_fn)

    state = TrainState(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
        ema_params1=TrainState.init_ema(model),
        step=0, epoch=0, dropout_generator=g,
    )

    # Auto-resume: if no explicit resume path, check output_dir for existing checkpoints
    if not config.resume:
        auto_ckpt = find_latest_checkpoint(config.output_dir)
        if auto_ckpt:
            config.resume = config.output_dir
            log_for_0(f"Auto-resuming from {auto_ckpt}")

    start_epoch, resume_step = 0, 0
    resume_epoch_fractional = 0.0  # Fractional epoch for save-point tracking
    if config.resume:
        try:
            ckpt_path = config.resume
            if "checkpoint_" not in ckpt_path:
                ckpt_path = find_latest_checkpoint(ckpt_path) or ckpt_path
            state, resume_step = load_checkpoint(ckpt_path, state)
            resume_epoch_fractional = float(state.epoch)
            start_epoch = int(state.epoch)
            log_for_0(f"Resumed from step {resume_step} (epoch {resume_epoch_fractional:.2f})")
        except Exception as e:
            raise RuntimeError(f"Failed to load resume checkpoint from {config.resume!r}: {e}") from e

    if getattr(config, "warm_start", None):
        if resume_step > 0:
            log_for_0("Warm-start skipped because training resumed from an existing checkpoint")
        else:
            try:
                state, warm_stats = load_warm_start_checkpoint(
                    config.warm_start,
                    state,
                    use_ema=bool(getattr(config, "warm_start_use_ema", False)),
                )
                log_for_0(
                    "Warm-start complete: "
                    f"loaded={warm_stats['loaded']}, missing={warm_stats['missing']}, "
                    f"shape_mismatch={warm_stats['shape_mismatch']}"
                )
            except Exception as e:
                log_for_0(f"Error warm-starting checkpoint: {e}")
                raise

    # torch.compile before DDP so only the inner module is compiled and
    # checkpoint I/O (which uses unwrap_model -> _orig_mod) still works.
    if device.type == "cuda" and bool(getattr(config, "use_compile", False)):
        log_for_0("Compiling ELF model with torch.compile (first step will be slower)...")
        state = state.replace(model=torch.compile(state.model))
    elif device.type == "cuda":
        log_for_0("torch.compile disabled")

    if world > 1:
        # find_unused_parameters=False is safe: 0-mult sinks in train_step
        # (`0 * net_out.sum()` for CE, `0 * decoder_logits.sum()` for L2)
        # keep every head in the autograd graph on every step.
        state = state.replace(model=DDP(
            state.model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        ))

    os.makedirs(config.output_dir, exist_ok=True)

    if rank == 0:
        config_dict = {
            k: ([vars(sc) for sc in v] if isinstance(v, list) and v and isinstance(v[0], SamplingConfig) else v)
            for k, v in vars(config).items()
        }
        config_path = os.path.join(config.output_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        log_for_0(f"Config saved to {config_path}")

    train_dataloader = get_dataloader(
        train_dataset, batch_size=local_batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        distributed=(world > 1),
    )

    log_for_0("\n" + "=" * 60)
    log_for_0("Checkpoint and Evaluation Schedule")
    log_for_0("=" * 60)
    log_for_0(
        f"Steps/epoch={steps_per_epoch}, epochs={config.epochs}, total={steps_per_epoch * config.epochs} | "
        f"save every {config.save_freq} epoch(s), eval every {config.eval_freq} epoch(s)"
    )
    if int(getattr(config, "train_sampling_eval_freq", 0)) > 0:
        metric_names = ["gPPL"]
        if bool(getattr(config, "use_sentence_plan", False)):
            metric_names.extend(["oracle_plan_ppl", "shuffled_plan_ppl"])
        metric_names.append("token_recon_ppl")
        log_for_0(
            f"Train-time sampling eval ({' + '.join(metric_names)}): "
            f"every {config.train_sampling_eval_freq} step(s), "
            f"samples={config.train_sampling_eval_num_samples}, "
            f"batch={config.train_sampling_eval_batch_size}, "
            f"max_configs={config.train_sampling_eval_max_configs}"
        )

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)
    log_for_0(f"Sampling configs: {len(config.sampling_configs)} config(s)")

    log_for_0("\n" + "=" * 60)
    log_for_0("Starting Training")
    log_for_0("=" * 60)

    if resume_step > 0:
        global_step = resume_step
        # Skip already-processed batches within the current epoch on resume
        steps_to_skip_in_epoch = resume_step - start_epoch * steps_per_epoch
    else:
        global_step = start_epoch * steps_per_epoch
        steps_to_skip_in_epoch = 0
    state.step = global_step

    last_log_step = global_step
    train_metrics = []
    last_log_time = time.time()

    # Track last save point for fractional save_freq; use fractional epoch from
    # checkpoint to avoid re-saving immediately after resume.
    last_save_epoch = resume_epoch_fractional if resume_step > 0 else float(start_epoch)

    for epoch in range(start_epoch, config.epochs):
        log_for_0(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Free device buffers from previous epoch before allocating new ones, to avoid
        # transient OOM at epoch boundaries.
        if epoch > start_epoch:
            del train_loader, train_iterator
            train_metrics = []
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if world > 1 and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        train_iterator = iter(train_dataloader)
        train_loader = prefetch_to_device(train_iterator, size=4)

        initial_pbar = (resume_step - start_epoch * steps_per_epoch) if (epoch == start_epoch and resume_step > 0) else 0
        epoch_pbar = tqdm(
            total=steps_per_epoch, desc=f"Epoch {epoch + 1}", initial=initial_pbar,
            mininterval=1.0, disable=rank != 0,
        )

        for step_in_epoch, batch in enumerate(train_loader):
            is_first_step = step_in_epoch == 0 and epoch == start_epoch
            if is_first_step:
                log_for_0("Performing initial training step, this may take longer...")
            # Skip already-processed batches when resuming mid-epoch
            if epoch == start_epoch and step_in_epoch < steps_to_skip_in_epoch:
                continue
            batch = prepare_batch(batch, config, generator=g)
            state, metrics = train_step(
                state, encoder=encoder, batch=batch, config=config,
                tokenizer=tokenizer, sentence_encoder=sentence_encoder,
            )

            # Sync only on first step to measure torch.compile time;
            # float() on the loss below already forces a device-to-host sync.
            if is_first_step:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                log_for_0("First training step (torch.compile + execution) completed...")

            global_step += 1
            train_metrics.append(metrics)
            epoch_pbar.update(1)

            if global_step % config.log_freq == 0:
                stacked = torch.stack([
                    torch.stack([m["loss"] for m in train_metrics]).mean(),
                    torch.stack([m["l2_loss"] for m in train_metrics]).mean(),
                    torch.stack([m["ce_loss"] for m in train_metrics]).mean(),
                    torch.stack([m["plan_loss"] for m in train_metrics]).mean(),
                    torch.stack([m["plan_aux_loss"] for m in train_metrics]).mean(),
                    torch.stack([m["plan_emb_batch_var"] for m in train_metrics]).mean(),
                    torch.stack([m["plan_emb_norm"] for m in train_metrics]).mean(),
                    torch.stack([m["plan_pred_batch_var"] for m in train_metrics]).mean(),
                    torch.stack([m["plan_pred_norm"] for m in train_metrics]).mean(),
                ])
                # Average each metric across DDP ranks before logging — done
                # once per log_freq so we never sync on every train step.
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
                    stacked = stacked / dist.get_world_size()
                (
                    avg_loss, avg_l2, avg_ce, avg_plan, avg_plan_aux,
                    avg_plan_emb_var, avg_plan_emb_norm,
                    avg_plan_pred_var, avg_plan_pred_norm,
                ) = (float(x) for x in stacked.tolist())
                now = time.time()
                steps_per_sec = (global_step - last_log_step) / max(now - last_log_time, 1e-8)
                current_lr = state.optimizer.param_groups[0]["lr"]

                postfix_dict = {
                    "step": f"{global_step}", "loss": f"{avg_loss:.4f}",
                    "l2": f"{avg_l2:.4f}", "ce": f"{avg_ce:.4f}",
                    "plan": f"{avg_plan:.4f}", "plan_aux": f"{avg_plan_aux:.4f}",
                    "emb_var": f"{avg_plan_emb_var:.2e}", "pred_var": f"{avg_plan_pred_var:.2e}",
                    "sps": f"{steps_per_sec:.1f}", "lr": f"{current_lr:.2e}",
                }
                log_for_0(postfix_dict)
                epoch_pbar.set_postfix(**postfix_dict)

                if rank == 0:
                    tqdm.write(
                        f"INFO - engine - Step {global_step}: loss={avg_loss:.4f}, "
                        f"l2={avg_l2:.4f}, ce={avg_ce:.4f}, "
                        f"plan={avg_plan:.4f}, plan_aux={avg_plan_aux:.4f}, "
                        f"emb_var={avg_plan_emb_var:.3e}, pred_var={avg_plan_pred_var:.3e}, "
                        f"lr={current_lr:.2e}, steps/sec={steps_per_sec:.2f}"
                    )
                    if config.use_wandb and wandb is not None:
                        current_epoch_progress = epoch + (step_in_epoch + 1) / steps_per_epoch
                        try:
                            wandb.log({
                                "train_loss": avg_loss, "train_l2_loss": avg_l2,
                                "train_ce_loss": avg_ce, "train_plan_loss": avg_plan,
                                "train_plan_aux_loss": avg_plan_aux, "lr": current_lr,
                                "train_plan_emb_batch_var": avg_plan_emb_var,
                                "train_plan_emb_norm": avg_plan_emb_norm,
                                "train_plan_pred_batch_var": avg_plan_pred_var,
                                "train_plan_pred_norm": avg_plan_pred_norm,
                                "epoch": current_epoch_progress, "step": global_step,
                            }, step=global_step)
                        except Exception:
                            pass

                train_metrics = []
                last_log_step = global_step
                last_log_time = now

            # Intra-epoch checkpoint saving (fractional save_freq, e.g., 0.1 epoch)
            if 0 < config.save_freq < 1:
                progress = epoch + (global_step - epoch * steps_per_epoch) / steps_per_epoch
                if progress - last_save_epoch >= config.save_freq:
                    save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
                    log_for_0(f"Saved checkpoint at epoch {progress:.2f} (step {global_step})")
                    last_save_epoch = progress

            if _run_train_sampling_eval_if_due(
                state=state, encoder=encoder, eval_dataset=eval_dataset,
                train_dataset=train_dataset, tokenizer=tokenizer,
                config=config, generator=g, local_batch_size=local_batch_size,
                sentence_encoder=sentence_encoder, global_step=global_step,
            ):
                last_log_step = global_step
                last_log_time = time.time()

        epoch_pbar.close()
        current_epoch = epoch + 1
        state.epoch = current_epoch

        if config.save_freq >= 1 and current_epoch % config.save_freq == 0:
            save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
            log_for_0(f"Saved checkpoint at epoch {current_epoch} (step {global_step})")

        if config.eval_freq >= 1 and current_epoch % config.eval_freq == 0:
            run_generation(
                state=state, encoder=encoder, eval_dataset=eval_dataset,
                tokenizer=tokenizer, config=config, generator=g,
                local_batch_size=local_batch_size,
                train_dataset=train_dataset,
                sentence_encoder=sentence_encoder,
            )
            last_log_step = global_step
            last_log_time = time.time()

    log_for_0("\n" + "=" * 60)
    log_for_0("Final Generation")
    log_for_0("=" * 60)
    save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
    log_for_0(f"Final checkpoint saved to {config.output_dir}")
    if config.use_wandb and rank == 0 and wandb is not None:
        wandb.finish()


def main():
    """CLI entry point: parse args, load config, then run training."""
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
        log_for_0(f"Applied {len(args.config_override)} config override(s)")
    run_training(config, force_cpu=args.use_cpu)


if __name__ == "__main__":
    main()
