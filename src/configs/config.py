from __future__ import annotations

import os
import yaml


class SamplingConfig:
    """Sampling configuration for generation."""
    def __init__(self, **kwargs):
        unknown = sorted(set(kwargs) - set(self.__class__.__annotations__))
        if unknown:
            raise ValueError(f"Unknown sampling config field(s): {', '.join(unknown)}")
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        fields = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        for k in self.__class__.__annotations__:
            if k not in fields:
                fields[k] = getattr(self, k, None)
        items = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"SamplingConfig({items})"

    sampling_method: str = "ode"
    num_sampling_steps: list = [50]
    cfgs: list = [1]
    self_cond_cfg_scales: list = [1.0]
    time_schedule: str = "logit_normal"  # 'logit_normal' or 'uniform'
    sde_gamma: float = 0.0  # Per-step SDE churn fraction; 0.0 -> pure ODE. Used when sampling_method == "sde".
    plan_sampling_mode: str = "joint"  # "joint" or strict two-stage "plan_first".
    plan_num_sampling_steps: int | None = None  # Separate plan-only NFE for plan_first; defaults to token steps.


_CONFIG_FIELDS = set()
_NONE_STRINGS = {"none", "null"}
_TRUE_STRINGS = {"true", "1", "yes", "y", "on"}
_FALSE_STRINGS = {"false", "0", "no", "n", "off"}


def _parse_bool(value_str: str, field_name: str) -> bool:
    lowered = value_str.lower()
    if lowered in _TRUE_STRINGS:
        return True
    if lowered in _FALSE_STRINGS:
        return False
    raise ValueError(
        f"Invalid boolean override for {field_name}: {value_str!r}. "
        f"Use one of {sorted(_TRUE_STRINGS | _FALSE_STRINGS)}."
    )


def _listify(value, field_name: str):
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    if not items:
        raise ValueError(f"{field_name} must not be empty")
    return items


def _validate_probability(value, field_name: str):
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {value}")


def _validate_positive_int(value, field_name: str):
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")


def validate_sampling_config(sampling_config: SamplingConfig, source: str = "sampling config") -> SamplingConfig:
    """Validate and lightly normalize one generation sampling configuration."""
    if sampling_config.sampling_method not in {"ode", "sde"}:
        raise ValueError(
            f"{source}: sampling_method must be 'ode' or 'sde', "
            f"got {sampling_config.sampling_method!r}"
        )
    if sampling_config.time_schedule not in {"logit_normal", "uniform"}:
        raise ValueError(
            f"{source}: time_schedule must be 'logit_normal' or 'uniform', "
            f"got {sampling_config.time_schedule!r}"
        )

    steps = _listify(sampling_config.num_sampling_steps, f"{source}.num_sampling_steps")
    normalized_steps = []
    for step in steps:
        if isinstance(step, bool) or int(step) != step or int(step) <= 0:
            raise ValueError(f"{source}: num_sampling_steps entries must be positive integers, got {step!r}")
        normalized_steps.append(int(step))
    sampling_config.num_sampling_steps = normalized_steps

    cfgs = [float(v) for v in _listify(sampling_config.cfgs, f"{source}.cfgs")]
    if any(v < 0.0 for v in cfgs):
        raise ValueError(f"{source}: cfgs entries must be non-negative, got {cfgs!r}")
    sampling_config.cfgs = cfgs

    sc_cfgs = [float(v) for v in _listify(
        sampling_config.self_cond_cfg_scales, f"{source}.self_cond_cfg_scales"
    )]
    if any(v < 0.0 for v in sc_cfgs):
        raise ValueError(f"{source}: self_cond_cfg_scales entries must be non-negative, got {sc_cfgs!r}")
    sampling_config.self_cond_cfg_scales = sc_cfgs

    sampling_config.sde_gamma = float(getattr(sampling_config, "sde_gamma", 0.0))
    if sampling_config.sde_gamma < 0.0:
        raise ValueError(f"{source}: sde_gamma must be non-negative, got {sampling_config.sde_gamma}")
    sampling_config.plan_sampling_mode = str(
        getattr(sampling_config, "plan_sampling_mode", "joint")
    ).lower()
    if sampling_config.plan_sampling_mode not in {"joint", "plan_first"}:
        raise ValueError(
            f"{source}: plan_sampling_mode must be 'joint' or 'plan_first', "
            f"got {sampling_config.plan_sampling_mode!r}"
        )
    plan_steps = getattr(sampling_config, "plan_num_sampling_steps", None)
    if plan_steps is not None:
        if isinstance(plan_steps, bool) or int(plan_steps) != plan_steps or int(plan_steps) <= 0:
            raise ValueError(
                f"{source}: plan_num_sampling_steps must be a positive integer or null, "
                f"got {plan_steps!r}"
            )
        sampling_config.plan_num_sampling_steps = int(plan_steps)
        if sampling_config.plan_sampling_mode != "plan_first":
            raise ValueError(
                f"{source}: plan_num_sampling_steps is only valid when "
                "plan_sampling_mode='plan_first'"
            )
    return sampling_config


def validate_config(config) -> Config:
    """Fail fast on invalid experiment settings."""
    if config.model not in {"ELF-B", "ELF-M", "ELF-L"}:
        raise ValueError(f"model must be one of ['ELF-B', 'ELF-M', 'ELF-L'], got {config.model!r}")
    preset_depth = {"ELF-B": 12, "ELF-M": 24, "ELF-L": 32}[config.model]
    if config.model_depth is not None:
        _validate_positive_int(config.model_depth, "model_depth")
        if int(config.model_depth) > preset_depth:
            raise ValueError(
                f"model_depth must not exceed the {config.model} preset depth "
                f"of {preset_depth}, got {config.model_depth}"
            )
    model_depth = int(config.model_depth or preset_depth)
    if config.model_active_depth is not None:
        _validate_positive_int(config.model_active_depth, "model_active_depth")
        if int(config.model_active_depth) > model_depth:
            raise ValueError(
                f"model_active_depth must not exceed the instantiated model depth "
                f"of {model_depth}, got {config.model_active_depth}"
            )
    if config.pad_token not in {"pad", "eos"}:
        raise ValueError(f"pad_token must be 'pad' or 'eos', got {config.pad_token!r}")
    if config.encoder_checkpoint is not None:
        raise ValueError(
            "encoder_checkpoint is set, but this code path does not load separate "
            "encoder checkpoints. Use encoder_model_name for HF/local T5 weights."
        )
    if config.optimizer not in {"adamw", "muon"}:
        raise ValueError(f"optimizer must be 'adamw' or 'muon', got {config.optimizer!r}")
    if config.lr_schedule not in {"constant", "cosine"}:
        raise ValueError(f"lr_schedule must be 'constant' or 'cosine', got {config.lr_schedule!r}")
    if config.time_schedule not in {"logit_normal", "uniform"}:
        raise ValueError(f"time_schedule must be 'logit_normal' or 'uniform', got {config.time_schedule!r}")

    if config.sentence_encoder_type not in {"sentence_t5", "learned"}:
        raise ValueError(
            f"sentence_encoder_type must be 'sentence_t5' or 'learned', got {config.sentence_encoder_type!r}"
        )
    if config.sentence_encoder_grad not in {"none", "detached_target", "full"}:
        raise ValueError(
            "sentence_encoder_grad must be 'none', 'detached_target', or 'full', "
            f"got {config.sentence_encoder_grad!r}"
        )
    if config.plan_adapter_type not in {"slot_mlp", "slot_dit"}:
        raise ValueError(f"plan_adapter_type must be 'slot_mlp' or 'slot_dit', got {config.plan_adapter_type!r}")
    if config.plan_denoiser_type not in {"shared", "independent"}:
        raise ValueError(
            "plan_denoiser_type must be 'shared' or 'independent', "
            f"got {config.plan_denoiser_type!r}"
        )
    if config.plan_denoiser_conditioning not in {"none", "prefix"}:
        raise ValueError(
            "plan_denoiser_conditioning must be 'none' or 'prefix', "
            f"got {config.plan_denoiser_conditioning!r}"
        )
    _validate_positive_int(config.plan_denoiser_depth, "plan_denoiser_depth")
    if config.plan_denoiser_hidden_size is not None:
        _validate_positive_int(config.plan_denoiser_hidden_size, "plan_denoiser_hidden_size")
    if config.plan_denoiser_num_heads is not None:
        _validate_positive_int(config.plan_denoiser_num_heads, "plan_denoiser_num_heads")
    plan_hidden = int(config.plan_denoiser_hidden_size or {"ELF-B": 768, "ELF-M": 1056, "ELF-L": 1280}[config.model])
    plan_heads = int(config.plan_denoiser_num_heads or {"ELF-B": 12, "ELF-M": 16, "ELF-L": 16}[config.model])
    if plan_hidden % plan_heads:
        raise ValueError("plan_denoiser_hidden_size must be divisible by plan_denoiser_num_heads")
    if config.plan_attention_topology not in {
        "joint",
        "hierarchical_prefix",
        "strict_hierarchical_prefix",
    }:
        raise ValueError(
            "plan_attention_topology must be 'joint', 'hierarchical_prefix', or "
            "'strict_hierarchical_prefix', "
            f"got {config.plan_attention_topology!r}"
        )
    if config.plan_aux_token_context not in {"denoiser_z", "resampled_z", "mixed_z", "clean_x0"}:
        raise ValueError(
            "plan_aux_token_context must be one of ['clean_x0', 'denoiser_z', 'mixed_z', 'resampled_z'], "
            f"got {config.plan_aux_token_context!r}"
        )
    if config.plan_time_schedule not in {"aligned", "noise_power"}:
        raise ValueError(
            "plan_time_schedule must be 'aligned' or 'noise_power', "
            f"got {config.plan_time_schedule!r}"
        )
    if float(config.plan_time_warp_gamma) < 1.0:
        raise ValueError(f"plan_time_warp_gamma must be >= 1.0, got {config.plan_time_warp_gamma!r}")
    if config.plan_training_mode not in {"joint", "plan_first", "plan_only", "oracle"}:
        raise ValueError(
            "plan_training_mode must be 'joint', 'plan_first', 'plan_only', or 'oracle', "
            f"got {config.plan_training_mode!r}"
        )
    _validate_probability(config.plan_first_plan_phase_prob, "plan_first_plan_phase_prob")
    if config.plan_training_mode == "plan_first":
        if not bool(config.use_sentence_plan):
            raise ValueError("plan_training_mode='plan_first' requires use_sentence_plan=true")
        if config.plan_attention_topology not in {
            "hierarchical_prefix", "strict_hierarchical_prefix",
        }:
            raise ValueError(
                "plan_training_mode='plan_first' requires a hierarchical plan attention topology"
            )
        if config.sentence_encoder_type != "sentence_t5":
            raise ValueError(
                "plan_training_mode='plan_first' currently requires sentence_encoder_type='sentence_t5'"
            )
        if not 0.0 < float(config.plan_first_plan_phase_prob) < 1.0:
            raise ValueError(
                "plan_first_plan_phase_prob must be strictly between 0 and 1 in plan_first mode"
            )
        if float(config.decoder_prob) >= 1.0:
            raise ValueError("plan_training_mode='plan_first' requires decoder_prob < 1")
        if float(config.plan_loss_weight) <= 0.0:
            raise ValueError("plan_training_mode='plan_first' requires plan_loss_weight > 0")
        if config.plan_time_schedule != "aligned" or float(config.plan_time_warp_gamma) != 1.0:
            raise ValueError(
                "plan_first training samples plan time independently; use "
                "plan_time_schedule='aligned' and plan_time_warp_gamma=1.0"
            )
    if config.plan_training_mode == "oracle":
        if not bool(config.use_sentence_plan):
            raise ValueError("plan_training_mode='oracle' requires use_sentence_plan=true")
        if config.sentence_encoder_type != "sentence_t5":
            raise ValueError(
                "plan_training_mode='oracle' requires frozen sentence_encoder_type='sentence_t5'"
            )
        if config.plan_attention_topology not in {
            "hierarchical_prefix", "strict_hierarchical_prefix",
        }:
            raise ValueError(
                "plan_training_mode='oracle' requires a hierarchical plan attention topology"
            )
        if float(config.plan_loss_weight) != 0.0:
            raise ValueError("plan_training_mode='oracle' requires plan_loss_weight=0")
    if config.plan_training_mode == "plan_only":
        if not bool(config.use_sentence_plan):
            raise ValueError("plan_training_mode='plan_only' requires use_sentence_plan=true")
        if config.sentence_encoder_type != "sentence_t5":
            raise ValueError("plan_training_mode='plan_only' requires sentence_encoder_type='sentence_t5'")
        if config.plan_denoiser_type != "independent":
            raise ValueError("plan_training_mode='plan_only' requires plan_denoiser_type='independent'")
        if config.plan_denoiser_conditioning != "prefix":
            raise ValueError("plan_training_mode='plan_only' requires plan_denoiser_conditioning='prefix'")
        if float(config.plan_loss_weight) <= 0:
            raise ValueError("plan_training_mode='plan_only' requires plan_loss_weight > 0")

    for field_name in ("decoder_prob", "label_drop_prob", "self_cond_prob"):
        _validate_probability(getattr(config, field_name), field_name)
    for field_name in (
        "latent_std", "sentence_latent_std", "denoiser_noise_scale",
        "decoder_noise_scale", "plan_noise_scale",
    ):
        value = float(getattr(config, field_name))
        if value <= 0.0:
            raise ValueError(f"{field_name} must be positive, got {value}")
    for field_name in ("max_length", "num_time_tokens", "epochs", "grad_accum_steps", "log_freq", "eval_freq"):
        _validate_positive_int(getattr(config, field_name), field_name)
    for field_name in ("bottleneck_dim", "num_self_cond_cfg_tokens", "num_model_mode_tokens"):
        value = int(getattr(config, field_name))
        if value < 0:
            raise ValueError(f"{field_name} must be >= 0, got {value}")
    if int(config.bottleneck_dim) == 0:
        raise ValueError("bottleneck_dim must be positive")
    if float(config.t_eps) <= 0.0 or float(config.t_eps) >= 1.0:
        raise ValueError(f"t_eps must be in (0, 1), got {config.t_eps!r}")
    for field_name in ("denoiser_p_std", "decoder_p_std"):
        if float(getattr(config, field_name)) < 0.0:
            raise ValueError(f"{field_name} must be >= 0, got {getattr(config, field_name)!r}")
    if float(config.self_cond_cfg_min) < 0.0:
        raise ValueError(f"self_cond_cfg_min must be >= 0, got {config.self_cond_cfg_min!r}")
    if float(config.self_cond_cfg_max) < float(config.self_cond_cfg_min):
        raise ValueError(
            f"self_cond_cfg_max must be >= self_cond_cfg_min, "
            f"got min={config.self_cond_cfg_min!r}, max={config.self_cond_cfg_max!r}"
        )
    if int(config.warmup_steps) < -1:
        raise ValueError(f"warmup_steps must be >= -1, got {config.warmup_steps!r}")
    if config.warmup_epochs is not None and float(config.warmup_epochs) < 0.0:
        raise ValueError(f"warmup_epochs must be >= 0, got {config.warmup_epochs!r}")
    if config.max_input_length is not None:
        _validate_positive_int(config.max_input_length, "max_input_length")
        if config.max_input_length >= config.max_length:
            raise ValueError(
                f"max_input_length must be smaller than max_length for conditional generation, "
                f"got max_input_length={config.max_input_length}, max_length={config.max_length}"
            )
    if bool(config.split_input_as_prefix) and config.max_input_length is None:
        raise ValueError(
            "split_input_as_prefix=true requires max_input_length to define the prefix length"
        )
    if config.batch_size is not None:
        _validate_positive_int(config.batch_size, "batch_size")
    if config.global_batch_size is not None:
        _validate_positive_int(config.global_batch_size, "global_batch_size")
    if int(config.num_workers) < 0:
        raise ValueError(f"num_workers must be >= 0, got {config.num_workers!r}")
    if int(config.num_samples) <= 0:
        raise ValueError(f"num_samples must be positive, got {config.num_samples!r}")
    if int(config.eval_mauve_seed) < 0:
        raise ValueError(f"eval_mauve_seed must be >= 0, got {config.eval_mauve_seed!r}")
    if bool(config.eval_mauve) and (
        not isinstance(config.eval_mauve_model, str) or not config.eval_mauve_model.strip()
    ):
        raise ValueError("eval_mauve_model must be a non-empty model ID/path when eval_mauve=true")
    if config.reconstruction_num_samples is not None:
        _validate_positive_int(config.reconstruction_num_samples, "reconstruction_num_samples")
    if int(config.train_sampling_eval_freq) < 0:
        raise ValueError(f"train_sampling_eval_freq must be >= 0, got {config.train_sampling_eval_freq!r}")
    if int(config.train_sampling_eval_max_configs) <= 0:
        raise ValueError(
            f"train_sampling_eval_max_configs must be positive, got {config.train_sampling_eval_max_configs!r}"
        )
    _validate_positive_int(config.train_sampling_eval_num_samples, "train_sampling_eval_num_samples")
    _validate_positive_int(config.train_sampling_eval_batch_size, "train_sampling_eval_batch_size")
    if float(config.save_freq) <= 0:
        raise ValueError(f"save_freq must be positive, got {config.save_freq!r}")
    if int(config.plan_aux_passes) < 0:
        raise ValueError(f"plan_aux_passes must be >= 0, got {config.plan_aux_passes!r}")
    if float(config.plan_loss_weight) < 0:
        raise ValueError(f"plan_loss_weight must be >= 0, got {config.plan_loss_weight!r}")
    if bool(config.use_sentence_plan):
        _validate_positive_int(config.num_plan_tokens, "num_plan_tokens")
        _validate_positive_int(config.sentence_emb_dim, "sentence_emb_dim")
    if bool(config.eval_sampled_plan_diagnostics):
        if not bool(config.use_sentence_plan):
            raise ValueError(
                "eval_sampled_plan_diagnostics=true requires use_sentence_plan=true"
            )
        if not bool(config.split_input_as_prefix) and config.eval_data_path is None:
            raise ValueError(
                "eval_sampled_plan_diagnostics=true requires conditional evaluation "
                "through split_input_as_prefix or eval_data_path"
            )
    if config.plan_adapter_type == "slot_dit":
        _validate_positive_int(config.plan_slot_dit_depth, "plan_slot_dit_depth")
    if config.plan_denoiser_type == "independent":
        _validate_positive_int(config.plan_denoiser_depth, "plan_denoiser_depth")

    config.sampling_configs = [
        validate_sampling_config(sc, source=f"sampling_configs[{idx}]")
        for idx, sc in enumerate(config.sampling_configs)
    ]
    return config


def resolve_batch_sizes(
    config: Config,
    world_size: int,
    *,
    context: str = "training",
    grad_accum_steps: int = 1,
):
    """Resolve effective-global and per-rank microbatch sizes.

    ``global_batch_size`` is the effective batch for one optimizer update and
    therefore includes all ranks and gradient accumulation. ``batch_size`` is
    the per-rank microbatch size.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if grad_accum_steps <= 0:
        raise ValueError(f"grad_accum_steps must be positive, got {grad_accum_steps}")
    has_global = config.global_batch_size is not None
    has_local = config.batch_size is not None
    if has_global and has_local:
        raise ValueError(
            "Specify only one of global_batch_size or batch_size. "
            "global_batch_size is effective across ranks and accumulation; "
            "batch_size is the per-rank microbatch."
        )
    if has_global:
        total_batch_size = int(config.global_batch_size)
        batch_divisor = world_size * grad_accum_steps
        if total_batch_size % batch_divisor != 0:
            raise ValueError(
                f"global_batch_size={total_batch_size} must be divisible by "
                f"world_size*grad_accum_steps={batch_divisor}"
            )
        local_batch_size = total_batch_size // batch_divisor
        config.batch_size = local_batch_size
    elif has_local:
        local_batch_size = int(config.batch_size)
        total_batch_size = local_batch_size * world_size * grad_accum_steps
        config.global_batch_size = total_batch_size
    else:
        raise ValueError(f"Either global_batch_size or batch_size must be specified for {context}")
    return total_batch_size, local_batch_size


# ============================================
# Configuration
# ============================================
class Config:
    # Dataset
    data_path: str = None
    eval_data_path: str = None
    max_length: int = 128
    max_input_length: int = None  # Max length for conditioning input (e.g., prompt or encoder input); None = no limit
    split_input_as_prefix: bool = False  # Split input-only corpora into bounded prefix + future halves.
    pad_token: str = "pad"  # "pad" or "eos" - which token to use for padding

    # Tokenizer
    tokenizer_name: str = None  # Defaults to encoder_model_name if not set

    # Encoder
    encoder_model_name: str = "t5-small"
    encoder_checkpoint: str = None
    latent_mean: float = 0.0
    latent_std: float = 1.0

    # Model architecture
    model: str = "ELF-B"
    # Instantiate fewer transformer blocks while retaining the preset width.
    # Warm-start loading copies the matching prefix of a deeper checkpoint.
    model_depth: int = None
    # Eval-time early exit for capacity probes. All blocks remain instantiated
    # so full-depth checkpoints keep exactly the same state-dict schema.
    model_active_depth: int = None
    bottleneck_dim: int = 128  # Bottleneck dimension for text projection
    num_time_tokens: int = 4  # Number of in-context time conditioning tokens
    num_self_cond_cfg_tokens: int = 4  # Number of in-context self-cond CFG tokens
    num_model_mode_tokens: int = 4  # If > 0, prepend learnable model-mode tokens that signal decoding mode
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    # Sentence-level planning / STAR-LDM style fusion
    use_sentence_plan: bool = False
    sentence_encoder_type: str = "sentence_t5"  # "sentence_t5" or "learned"
    sentence_t5_model_name: str = "sentence-transformers/sentence-t5-xl"
    sentence_emb_dim: int = 768
    sentence_latent_mean: float = 0.0
    sentence_latent_std: float = 1.0
    num_plan_tokens: int = 8
    plan_adapter_type: str = "slot_mlp"  # "slot_mlp" or "slot_dit"
    plan_slot_dit_depth: int = 2
    plan_denoiser_type: str = "shared"  # "shared" or plan-only "independent"
    plan_denoiser_depth: int = 12
    plan_denoiser_hidden_size: int = None
    plan_denoiser_num_heads: int = None
    plan_denoiser_conditioning: str = "none"  # "none" or observed-prefix-only
    # joint; two-block (prefix+plan)->future; or strict control->prefix->plan->future.
    plan_attention_topology: str = "joint"
    plan_learned_encoder_norm: bool = True
    plan_loss_weight: float = 1.0
    plan_noise_scale: float = 1.0
    plan_time_schedule: str = "aligned"  # "aligned" or "noise_power"; maps token t -> plan t.
    plan_time_warp_gamma: float = 1.0  # For noise_power: plan_t = 1 - (1 - token_t) ** gamma.
    plan_training_mode: str = "joint"  # "joint", "plan_first", "plan_only", or clean-plan "oracle".
    plan_first_plan_phase_prob: float = 0.5  # Among non-decoder rows, allocate this fraction to plan-only denoising.
    plan_aux_passes: int = 1  # Extra detached plan-denoiser passes for learned+none topology.
    plan_aux_token_context: str = "denoiser_z"  # "denoiser_z", "resampled_z", "mixed_z", or "clean_x0"
    sentence_encoder_grad: str = "none"  # "none", "detached_target", or "full" 

    # Denoiser objective
    denoiser_p_mean: float = 0.8
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 1.0
    t_eps: float = 5e-2
    time_schedule: str = "logit_normal"  # 'logit_normal' or 'uniform'

    # Decoder objective
    decoder_prob: float = 0.5  # Probability of decoder (CE) step vs denoiser (L2) step
    decoder_noise_scale: float = 1.0  # Scale of noise in logit-normal-noised latent for CE branch
    decoder_p_mean: float = 0.8  # Mean for logit-normal noise schedule in decoder objective
    decoder_p_std: float = 0.8  # Std for logit-normal noise schedule in decoder objective

    # Conditioning / CFG
    label_drop_prob: float = 0.0
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0

    # Training (optimizer + schedule)
    epochs: int = 200
    warmup_epochs: float = None
    warmup_steps: int = 5000
    batch_size: int = None  # Per-rank microbatch; effective batch also multiplies world size and grad_accum_steps.
    global_batch_size: int = 512  # Effective batch per optimizer update, including ranks and accumulation.
    lr: float = None
    blr: float = 5e-5
    min_lr: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 0.0
    optimizer: str = "muon"  # "adamw" or "muon"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    grad_accum_steps: int = 1  # Microsteps per optimizer update; global_batch_size semantics stay fixed.
    use_bf16: bool = True  # Use CUDA BF16 autocast for training/eval forward passes.
    use_compile: bool = False  # Wrap the eval/sampling model in torch.compile.
    gradient_checkpointing: bool = False  # Save activation memory by recomputing ELF blocks during backward.

    # EMA
    ema_decay1: float = 0.9999

    # Sampling
    sampling_configs_path: str = None
    # Sampling configs sweep (list of SamplingConfig objects, loaded from YAML)
    sampling_configs: list = [SamplingConfig()]
    num_samples: int = 100

    # Online generation evaluation
    online_eval: bool = True  # Enable generation metrics for generated samples.
    eval_ppl_model: str = "gpt2-large"  # Model for PPL evaluation
    eval_ppl_batch_size: int = 64  # Batch size for PPL evaluation (adjusted to be divisible by device count)
    eval_ppl_max_length: int = 1024  # Max sequence length for PPL evaluation
    eval_mauve: bool = True  # Compare generated/reference distributions in a fixed LM feature space.
    eval_mauve_model: str = "gpt2-large"  # Feature model for MAUVE; may differ from eval_ppl_model.
    eval_mauve_seed: int = 25  # Deterministic PCA/k-means seed used by mauve-text.
    eval_sampled_plan_diagnostics: bool = False  # Compare sampled plans with per-example clean targets.
    reconstruction_eval: bool = False  # Run oracle/shuffled plan PPL and clean-token reconstruction diagnostics.
    reconstruction_num_samples: int = None  # None = reuse num_samples.
    train_sampling_eval_freq: int = 0  # Step interval for lightweight gPPL/plan/token-recon eval. 0 disables.
    train_sampling_eval_num_samples: int = 64
    train_sampling_eval_batch_size: int = 16
    train_sampling_eval_max_configs: int = 1  # Use the first N sampling configs for train-time monitoring.

    # Logging & Checkpointing
    log_freq: int = 100
    eval_freq: int = 10
    save_freq: float = 100  # Can be fractional (e.g., 0.1 for saving every 0.1 epoch)

    # Output
    output_dir: str = "./output_dir"
    hf_repo_id: str = None  # Optional HF repo id to mirror local outputs/checkpoints.
    resume: str = None
    warm_start: str = None  # Optional checkpoint path for model-only partial initialization.
    warm_start_use_ema: bool = False  # If true, warm-start from ema_params1 instead of params.

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "ELF"
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_run_id: str = None  # Optional stable W&B id; leave unset to create a fresh run.
    wandb_tag: str = None
    wandb_resume: str = None

    # Misc
    seed: int = 0
    num_workers: int = 8


_CONFIG_FIELDS = set(Config.__annotations__)


def _make_sampling_configs(entries, source: str):
    if not isinstance(entries, list):
        raise ValueError(f"{source}: sampling_configs must be a list of mappings")
    if not entries:
        raise ValueError(f"{source}: sampling_configs must not be empty")
    configs = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: sampling_configs[{idx}] must be a mapping")
        configs.append(validate_sampling_config(SamplingConfig(**entry), source=f"{source}[{idx}]"))
    return configs


def load_config_from_yaml(path: str) -> Config:
    """Load a YAML config and override defaults in Config."""
    if not path:
        return validate_config(Config())
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")

    base_config_path = cfg_dict.pop("base_config", None)
    unknown = sorted(set(cfg_dict) - _CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"Unknown config field(s) in {path}: {', '.join(unknown)}")
    inline_sampling_configs = cfg_dict.pop("sampling_configs", None)

    if base_config_path:
        if not os.path.isabs(base_config_path):
            base_config_path = os.path.join(os.path.dirname(path), base_config_path)
        config = load_config_from_yaml(os.path.normpath(base_config_path))
    else:
        config = Config()

    for key, value in cfg_dict.items():
        setattr(config, key, value)

    if inline_sampling_configs is not None:
        config.sampling_configs = _make_sampling_configs(inline_sampling_configs, f"{path}:sampling_configs")
    elif config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    return validate_config(config)


def apply_config_overrides(config: Config, overrides: list) -> Config:
    """Apply command-line config overrides to a Config object.

    Args:
        config: Config object to modify
        overrides: List of strings in format "field_name=value"

    Returns:
        Modified config object
    """
    if not overrides:
        return config

    sampling_configs_path_overridden = False
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'field_name=value'")

        field_name, value_str = override.split("=", 1)
        field_name = field_name.strip()
        value_str = value_str.strip()
        if field_name == "sampling_configs_path":
            sampling_configs_path_overridden = True

        if field_name not in _CONFIG_FIELDS:
            raise ValueError(f"Config has no field named '{field_name}'")

        original_value = getattr(config, field_name)
        original_type = type(original_value)
        annotated_type = config.__annotations__.get(field_name)

        # Allow setting a field back to None
        if value_str.lower() in _NONE_STRINGS:
            setattr(config, field_name, None)
            continue

        # Prefer the declared field type over YAML's runtime scalar type. YAML
        # parses ``save_freq: 1`` as int even though Config declares float, and
        # an override such as ``save_freq=0.1`` must remain valid.
        if annotated_type in (bool, "bool"):
            converted_value = _parse_bool(value_str, field_name)
        elif annotated_type in (int, "int"):
            converted_value = int(value_str)
        elif annotated_type in (float, "float"):
            converted_value = float(value_str)
        elif annotated_type in (str, "str"):
            converted_value = value_str
        elif original_value is None:
            converted_value = value_str
        elif original_type == bool:
            converted_value = _parse_bool(value_str, field_name)
        elif original_type == int:
            converted_value = int(value_str)
        elif original_type == float:
            converted_value = float(value_str)
        elif original_type == str:
            converted_value = value_str
        else:
            converted_value = value_str

        setattr(config, field_name, converted_value)

    if sampling_configs_path_overridden and config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    return validate_config(config)


def load_sampling_configs(sampling_configs_path: str):
    """Return sampling configs, loading from sampling_configs_path if set."""
    if not os.path.isfile(sampling_configs_path):
        raise FileNotFoundError(f"Sampling config file not found: {sampling_configs_path}")
    with open(sampling_configs_path, "r") as f:
        entries = yaml.safe_load(f)
    return _make_sampling_configs(entries or [], sampling_configs_path)
