# ELF Configuration and Launcher Reference

This is the maintained reference for experiment YAML fields, sampling sweep
fields, command-line flags, and container launcher environment variables.
Defaults below come from `src/configs/config.py`; an experiment manifest stores
the fully resolved values used by a run.

For an autonomous experiment, first use
[`agent_research_guide.md`](agent_research_guide.md) to define the comparison
and lifecycle. Use this document to verify the exact meaning of every field;
use [`experiment_workflow.md`](experiment_workflow.md) for campaign and backend
schema rather than treating training flags as scheduler configuration.

## Configuration rules

- YAML keys and `--config_override field=value` must name a declared `Config`
  field. Unknown names fail validation.
- Override values `none` and `null` become `None`. Boolean overrides accept
  `true/false`, `1/0`, `yes/no`, `y/n`, and `on/off`.
- Specify exactly one of `global_batch_size` and `batch_size` after overrides.
  `global_batch_size` is the effective batch per optimizer update, including
  every rank and `grad_accum_steps`. `batch_size` is the per-rank microbatch.
- Warm start loads compatible model tensors and starts fresh optimizer state.
  Resume continues the same run from an atomically completed checkpoint.

## Dataset, tokenizer, and encoder

| Field | Default | Meaning |
| --- | --- | --- |
| `data_path` | `None` | Training dataset path or Hugging Face dataset ID. |
| `eval_data_path` | `None` | Optional separate evaluation dataset. |
| `max_length` | `128` | Total padded/truncated token length. |
| `max_input_length` | `None` | Maximum conditioning-input length; when set, smaller than `max_length`. |
| `pad_token` | `pad` | Padding policy: `pad` or `eos`. |
| `tokenizer_name` | `None` | Tokenizer ID/path; falls back to `encoder_model_name`. |
| `encoder_model_name` | `t5-small` | Frozen text encoder ID/path. |
| `encoder_checkpoint` | `None` | Reserved and currently rejected; use `encoder_model_name`. |
| `latent_mean` | `0.0` | Token-latent normalization mean. |
| `latent_std` | `1.0` | Positive token-latent normalization standard deviation. |

## Model architecture

| Field | Default | Meaning |
| --- | --- | --- |
| `model` | `ELF-B` | Architecture preset: `ELF-B`, `ELF-M`, or `ELF-L`. |
| `bottleneck_dim` | `128` | Positive text projection bottleneck dimension. |
| `num_time_tokens` | `4` | In-context time-conditioning token count. |
| `num_self_cond_cfg_tokens` | `4` | Self-conditioning CFG token count; zero disables them. |
| `num_model_mode_tokens` | `4` | Learnable decoder/denoiser mode-token count. |
| `attn_dropout` | `0.0` | Attention dropout probability. |
| `proj_dropout` | `0.0` | Projection dropout probability. |

## Sentence-plan fusion

| Field | Default | Meaning |
| --- | --- | --- |
| `use_sentence_plan` | `false` | Enable sentence-level planning. |
| `sentence_encoder_type` | `sentence_t5` | Plan encoder: `sentence_t5` or `learned`. |
| `sentence_t5_model_name` | `sentence-transformers/sentence-t5-xl` | Sentence-T5 model ID/path. |
| `sentence_emb_dim` | `768` | Positive sentence embedding dimension. |
| `sentence_latent_mean` | `0.0` | Sentence-latent normalization mean. |
| `sentence_latent_std` | `1.0` | Positive sentence-latent normalization standard deviation. |
| `num_plan_tokens` | `8` | Number of sentence-plan slots. |
| `plan_adapter_type` | `slot_mlp` | Plan adapter: `slot_mlp` or `slot_dit`. |
| `plan_slot_dit_depth` | `2` | DiT depth when using `slot_dit`. |
| `plan_learned_encoder_norm` | `true` | Normalize learned plan-encoder output. |
| `plan_loss_weight` | `1.0` | Non-negative multiplier for plan loss. |
| `plan_noise_scale` | `1.0` | Positive plan diffusion noise scale. |
| `plan_time_schedule` | `aligned` | Plan time: `aligned` or `noise_power`. |
| `plan_time_warp_gamma` | `1.0` | For `noise_power`, uses `1-(1-t)^gamma`; must be at least one. |
| `plan_aux_passes` | `1` | Detached auxiliary plan-denoiser passes. |
| `plan_aux_token_context` | `denoiser_z` | `denoiser_z`, `resampled_z`, `mixed_z`, or `clean_x0`. |
| `sentence_encoder_grad` | `none` | Gradient topology: `none`, `detached_target`, or `full`. |

## Objectives and conditioning

| Field | Default | Meaning |
| --- | --- | --- |
| `denoiser_p_mean` | `0.8` | Logit-normal denoiser time mean. |
| `denoiser_p_std` | `0.8` | Non-negative denoiser time standard deviation. |
| `denoiser_noise_scale` | `1.0` | Positive denoiser noise scale. |
| `t_eps` | `0.05` | Numerical time epsilon in `(0,1)`. |
| `time_schedule` | `logit_normal` | Training time distribution: `logit_normal` or `uniform`. |
| `decoder_prob` | `0.5` | Per-example probability of selecting the CE decoder branch. |
| `decoder_noise_scale` | `1.0` | Positive decoder-branch noise scale. |
| `decoder_p_mean` | `0.8` | Logit-normal decoder time mean. |
| `decoder_p_std` | `0.8` | Non-negative decoder time standard deviation. |
| `label_drop_prob` | `0.0` | Probability of dropping conditioning labels. |
| `self_cond_prob` | `0.5` | Probability of self-conditioning. |
| `self_cond_cfg_min` | `0.5` | Minimum non-negative self-conditioning CFG scale. |
| `self_cond_cfg_max` | `5.0` | Maximum CFG scale, not smaller than the minimum. |

## Optimization and runtime

| Field | Default | Meaning |
| --- | --- | --- |
| `epochs` | `200` | Number of dataset epochs. |
| `warmup_epochs` | `None` | Warmup duration in epochs when `warmup_steps < 0`. |
| `warmup_steps` | `5000` | Warmup microsteps; `-1` selects `warmup_epochs`. |
| `batch_size` | `None` | Per-rank microbatch; mutually exclusive with `global_batch_size`. |
| `global_batch_size` | `512` | Effective batch per optimizer update across ranks and accumulation. |
| `lr` | `None` | Explicit learning rate; non-positive/`None` derives it from `blr`. |
| `blr` | `5e-5` | Base LR scaled by effective batch divided by 256. |
| `min_lr` | `0.0` | Minimum cosine-schedule LR. |
| `lr_schedule` | `constant` | `constant` or `cosine`. |
| `weight_decay` | `0.0` | Optimizer weight decay. |
| `optimizer` | `muon` | `muon` or `adamw`. |
| `adam_b1` | `0.9` | AdamW beta 1. |
| `adam_b2` | `0.95` | AdamW beta 2. |
| `grad_accum_steps` | `1` | Microsteps per optimizer update; tail windows are flushed. |
| `use_bf16` | `true` | Enable CUDA bfloat16 autocast. |
| `use_compile` | `false` | Enable `torch.compile` for eligible model execution in training and generation/evaluation. |
| `gradient_checkpointing` | `false` | Recompute block activations during backward to save memory. |
| `ema_decay1` | `0.9999` | EMA decay applied at optimizer boundaries. |

## Sampling and evaluation

| Field | Default | Meaning |
| --- | --- | --- |
| `sampling_configs_path` | `None` | YAML file containing a sampling sweep. |
| `sampling_configs` | one default entry | Inline list of sampling sweep mappings. |
| `num_samples` | `100` | Number of generated samples. |
| `online_eval` | `true` | Compute PPL for generated samples. |
| `eval_ppl_model` | `gpt2-large` | Evaluation language model ID/path. |
| `eval_ppl_batch_size` | `64` | PPL evaluation batch size. |
| `eval_ppl_max_length` | `1024` | Maximum PPL evaluation sequence length. |
| `reconstruction_eval` | `false` | Enable oracle/shuffled-plan and clean-token reconstruction diagnostics. |
| `reconstruction_num_samples` | `None` | Reconstruction sample count; `None` reuses `num_samples`. |
| `train_sampling_eval_freq` | `0` | Microstep interval for lightweight generation diagnostics; zero disables. |
| `train_sampling_eval_num_samples` | `64` | Samples per train-time diagnostic. |
| `train_sampling_eval_batch_size` | `16` | Batch size for train-time diagnostics. |
| `train_sampling_eval_max_configs` | `1` | Maximum sampling sweep entries used during training. |

Each `SamplingConfig` mapping supports:

| Field | Default | Meaning |
| --- | --- | --- |
| `sampling_method` | `ode` | `ode` or `sde`. |
| `num_sampling_steps` | `[50]` | Positive step-count sweep. |
| `cfgs` | `[1]` | Non-negative classifier-free guidance sweep. |
| `self_cond_cfg_scales` | `[1.0]` | Non-negative self-conditioning CFG sweep. |
| `time_schedule` | `logit_normal` | Sampling schedule: `logit_normal` or `uniform`. |
| `sde_gamma` | `0.0` | Non-negative SDE churn fraction; zero is pure ODE. |

## Logging, checkpoints, output, and W&B

| Field | Default | Meaning |
| --- | --- | --- |
| `log_freq` | `100` | Microstep interval for `train_metrics.jsonl` and W&B. |
| `eval_freq` | `10` | Evaluation interval in epochs. |
| `save_freq` | `100` | Checkpoint interval in epochs; fractions enable intra-epoch saves at optimizer boundaries. |
| `output_dir` | `./output_dir` | Run artifact and checkpoint directory. Cloud launcher overrides it uniquely. |
| `hf_repo_id` | `None` | Optional Hugging Face repository used as a non-canonical mirror. |
| `resume` | `None` | Same-run directory or completed checkpoint; restores full training state. |
| `warm_start` | `None` | Checkpoint used for compatible model tensors only. |
| `warm_start_use_ema` | `false` | Warm-start from EMA tensors instead of raw parameters. |
| `use_wandb` | `false` | Mirror metrics to W&B. |
| `wandb_project` | `ELF` | W&B project. |
| `wandb_entity` | `None` | Optional W&B entity. |
| `wandb_run_name` | `None` | Display name; cloud defaults it to scientific run ID. |
| `wandb_run_id` | `None` | Stable W&B identity; cloud defaults it to scientific run ID. |
| `wandb_tag` | `None` | Comma-separated W&B tags. |
| `wandb_resume` | `None` | W&B resume policy; cloud uses `allow`. |
| `seed` | `0` | Scientific random seed. |
| `num_workers` | `8` | DataLoader worker count per process. |

Checkpoint files are committed as `checkpoint_<microstep>` followed by
`checkpoint_<microstep>.complete`. Directory resume ignores incomplete files,
validates marker size, and falls back from a corrupt newest checkpoint.

## Python command-line flags

Training:

```text
python src/train.py [--config PATH] [--config_override FIELD=VALUE ...] [--use_cpu]
```

- `--config` loads experiment YAML.
- `--config_override` applies a repeatable typed override.
- `--use_cpu` forces CPU execution.

Evaluation:

```text
python src/eval.py --config PATH --checkpoint_path PATH
  [--config_override FIELD=VALUE ...] [--seed N | --seeds CSV] [--use_cpu]
```

- `--checkpoint_path` selects evaluation weights.
- `--seed` selects one generation seed; `--seeds` accepts comma-separated seeds.

The generic launcher is:

```text
bash scripts/launch.sh <train|eval> <config.yml> [Python flags]
```

`NGPU=1` and `NNODES=1` use Python directly. Larger values use `torchrun` with
`NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`. GPU detection also recognizes
`NPROC_PER_NODE`, `LOCAL_WORLD_SIZE`, `NUM_GPUS`, `GPU_COUNT`,
`CUDA_VISIBLE_DEVICES`, and `NVIDIA_VISIBLE_DEVICES`.
When `MASTER_PORT` is unset in Slurm, the launcher derives a deterministic port
from `SLURM_JOB_ID` so multiple distributed jobs can share one node safely.

## Cloud launcher environment

Run identity and backend:

- `CONFIG`, `PROJECT_NAME`, `RUN_ID`, `ATTEMPT_ID`, `BACKEND`,
  `BACKEND_JOB_ID`, `SOURCE_ID`, `ELF_SOURCE_ID`,
  `RUNTIME_TREE_ID`, `GIT_COMMIT`, `CAMPAIGN_ID`, `CAMPAIGN_NAME`, `IMAGE_ID`,
  `ELF_IMAGE_ID`, `QUOTA_TYPE`, `RESOURCE_SPEC`, and `MAX_INFRA_RETRIES`.
  Runtime, Git, campaign, and image identities are recorded separately so a
  campaign-only edit does not imply a new training image.
- Direct local launches default to `BACKEND=local`, `QUOTA_TYPE=unknown`, and
  a repository-local `.runtime` data root. Registered backends always inject
  their explicit values and persistent storage paths.
- `REQUIRE_IMMUTABLE_IDENTITIES` defaults to true for real runs. Set it false
  only for unrecorded local smoke tests.

Storage and offline assets:

- `DATA_ROOT`, `PROJECT_DATA_ROOT`, `OUTPUT_ROOT`, `OUTPUT_DIR`,
  `CHECKPOINT_ROOT`, `SAVE_DIR`, `HF_HOME`, `HF_DATASETS_CACHE`, `WANDB_DIR`,
  and `WANDB_CACHE_DIR`.
- `BAKED_HF_HOME`, `BAKED_CHECKPOINT_ROOT`, `DATASET_ID`, `ENCODER_MODEL`,
  `ELF_B_CHECKPOINT_FILE`, and `ELF_B_OWT_CHECKPOINT` locate baked or hydrated
  resources.
- `HF_ENDPOINT`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`,
  `HF_DATASETS_OFFLINE`, `REQUIRE_OFFLINE_CACHE`,
  `REQUIRE_ELF_B_CHECKPOINT`, and `HYDRATE_LOCK_TIMEOUT_SECONDS` control
  offline validation and hydration.

When offline validation is enabled, the launcher resolves the inherited YAML
and checks the token encoder, dataset, config-selected Sentence-T5 model (only
for frozen-plan runs), config-selected PPL evaluator (when online evaluation is
enabled), and requested warm-start checkpoint before starting distributed
workers.

Training overrides:

- `USE_WANDB`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME`,
  `WANDB_RUN_ID`, `WANDB_RESUME`, `GLOBAL_BATCH_SIZE`, `BATCH_SIZE`,
  `NUM_WORKERS`, `LOG_FREQ`, `USE_COMPILE`, `WARM_START`,
  `WARM_START_USE_EMA`, `USE_ELF_B_WARM_START`, `RESUME`, and `HF_REPO_ID`.

Execution controls:

- `NGPU`, `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` control
  distributed execution.
- `RESEARCH_CONTRACT_B64` and `RESEARCH_ROLE` are controller-generated values
  that carry the reviewed campaign contract into the runtime manifest. Do not
  author or override them as ad-hoc launcher settings.
- `DRY_RUN` set to `1` prints the resolved command without filesystem mutation.
- `HYDRATE_ONLY` set to `1` hydrates baked assets and exits.
- `PREPARE_ONLY` set to `1` writes/validates manifest records and exits before training.

The cloud launcher writes canonical state under resolved `$OUTPUT_DIR`, which
corresponds to campaign `storage.run_dir`. Its shared root is backend-specific:
SenseCore commonly uses `/data/elf`, L40S uses `/data/liangluocheng/elf`, and
H100 uses `/datapool/liangluocheng/elf`. The state includes:
`manifest.yaml`, `status.json`, `backend.json`, `events.jsonl`,
`train_metrics.jsonl`, completed checkpoints, and attempt-specific process
logs. W&B and Hugging Face uploads are mirrors, not the source of truth.
