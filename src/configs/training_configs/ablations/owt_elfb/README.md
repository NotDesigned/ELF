# OWT ELF-B Sentence-Plan Ablations

These configs use `base_config` inheritance from `../../train_owt_ELF-B.yml`.
Each file should change one scientific axis, not just rename an existing run.

Run with the normal training entrypoint:

```bash
python src/train.py --config src/configs/training_configs/ablations/owt_elfb/tier0_2_learned_main.yml
```

## Required Anchors

Run the length-aligned anchors first. They are the cleanest Tier 1
learned-vs-frozen comparison because Sentence-T5 itself truncates at 256 tokens.

| Config | Purpose |
|---|---|
| `tier0_0_pure_elf_len256.yml` | Pure ELF baseline with `max_length=256`. |
| `tier0_1_sentence_t5_len256.yml` | Frozen Sentence-T5 teacher with token block length aligned to Sentence-T5. |
| `tier0_2_learned_main_len256.yml` | Learned-plan main hypothesis under the same length-aligned setting. |

### Full-Length Tier 0

The default OWT configs inherit `max_length=1024`, while the frozen
`sentence-transformers/sentence-t5-xl` plan encoder has `max_seq_length=256`.
That means the frozen sentence plan can be a truncated summary of the token
continuation. The 1024-token variants are still useful for full-length ELF
comparisons after the length-aligned sanity pass:

| Config | Purpose |
|---|---|
| `tier0_0_pure_elf.yml` | Original ELF baseline, `use_sentence_plan=false`. |
| `tier0_1_sentence_t5.yml` | Frozen Sentence-T5 plan sanity / teacher baseline; truncated teacher at 1024-token blocks. |
| `tier0_2_learned_main.yml` | Main hypothesis at 1024-token blocks: learned sentence plan with `sentence_encoder_grad=none` and `plan_aux_passes=1`. |

The `*_len256.yml` configs set both `max_length=256` and
`eval_ppl_max_length=256`, so generation / reconstruction and sliding-window PPL
are evaluated at the same span.

Core question: compare `tier0_1_sentence_t5_len256.yml` against
`tier0_2_learned_main_len256.yml` to test whether the sentence embedding can be
learned while the word/token T5 field stays fixed, with
`tier0_0_pure_elf_len256.yml` as the no-plan control.

Primary metrics:

- `plan_emb_batch_var`: collapse probe.
- `plan_emb_norm`: with RMSNorm over 768 dims, expect about `sqrt(768) = 27.7`.
- `plan_loss`, `plan_aux_loss`.
- Common token metrics versus pure ELF: `train_ce_loss`, `train_l2_loss`.
- Generation PPL, and BLEU/ROUGE on conditional tasks.

## Gradient Topology

These configs are the ELF-fusion analogue of STAR-LDM's
`encoder_diffusion_grad` overlays. They inherit from
`tier0_2_learned_main.yml`, so `plan_aux_passes=1` is held fixed. Because ELF's
token denoiser and decoder are fused, token-field `L2` and decoder `CE` remain
allowed to train the learned encoder through plan slots in all three settings;
this axis only controls sentence-plan MSE gradients.

| Config | Delta | Expected behavior |
|---|---|---|
| `tier2_grad_detached_target.yml` | `sentence_encoder_grad=detached_target` | Target detached, but the noised input path still leaks gradients to the encoder. |
| `tier2_grad_full.yml` | `sentence_encoder_grad=full` | Strongest coupling; collapse / instability baseline. |

For the primary length-aligned campaign, use
`tier2_grad_detached_target_len256.yml` and `tier2_grad_full_len256.yml`.
These inherit their corresponding topology configs and change only
`max_length`, `eval_ppl_max_length`, and operational run metadata. Compare them
against `tier0_2_learned_main_len256.yml`, not the 1024-token learned config.

Compare both against `tier0_2_learned_main.yml`, whose `none` topology detaches
main sentence-plan MSE and trains extra plan-denoiser passes against `sg(s0)`.

## UNITE Ratio

These configs match STAR-LDM-style `n_mse_passes` / UNITE denoiser-pass ratio.
They inherit from `tier0_2_learned_main.yml`, so
`sentence_encoder_type=learned` and `sentence_encoder_grad=none` are held fixed.

| Config | Delta |
|---|---|
| `tier3_aux0.yml` | `plan_aux_passes=0` |
| `tier3_aux2.yml` | `plan_aux_passes=2` |
| `tier3_aux4.yml` | `plan_aux_passes=4` |

Compare these against `tier0_2_learned_main.yml`, which is the default
`plan_aux_passes=1` run. Question: do extra detached plan-denoiser passes
improve sampling / plan refinement without damaging the learned encoder
representation?

For the primary length-aligned campaign, use `tier3_aux0_len256.yml`,
`tier3_aux2_len256.yml`, and `tier3_aux4_len256.yml`, with
`tier0_2_learned_main_len256.yml` as the `plan_aux_passes=1` reference.

## Independent Plan Denoiser

This axis keeps the frozen Sentence-T5 target, plan-slot conditioning path,
token model, and objectives fixed while changing who predicts the clean plan:

| Config | Plan predictor |
|---|---|
| `tier0_1_sentence_t5_len256.yml` | `shared`: read plan slots after the shared ELF trunk. |
| `tier4_independent_plan_denoiser_len256.yml` | `independent`: separate 12-block plan-only ELF stack over noisy plan slots and plan time. |

The 12-block depth matches ELF-B's shared trunk. The independent predictor
cannot read the token field. The token ELF trunk
still receives the current plan slots, so this tests whether the plan prior
should have separate parameters without changing how tokens are conditioned.
Use the same pure-ELF warm start, source, image, seed, batch settings, training
budget, hardware, and sampling variants for both arms.

## Hierarchical Prefix -> Plan -> Future Topology

This axis keeps the shared ELF-B parameter count fixed and separates activation
topology from relative diffusion time:

| Config | Attention topology | Plan time |
|---|---|---|
| `tier0_1_sentence_t5_len256.yml` | joint plan/future | aligned |
| `tier5_hierarchical_prefix_len256.yml` | prefix/plan -> future block-triangular | aligned |
| `tier5_hierarchical_prefix_lead_g3_len256.yml` | prefix/plan -> future block-triangular | `noise_power`, gamma=3 |

For `hierarchical_prefix`, time/self-cond/mode/plan/observed-prefix queries are
all upstream and cannot read future-token keys. Future-token queries may read
every valid upstream and future key. Blocking the whole upstream group in every
layer prevents future information from leaking back to plan slots through an
intermediate special or prefix token. On unconditional OWT, the observed prefix
is empty, so this is a plan-prior -> future test; conditional datasets are still
required to measure prefix-conditioned plan adherence.

The gamma-3 arm uses the same mapping in training and sampling:
`plan_t = 1 - (1 - token_t)^3`. At token `t=0.5`, the plan has reached
`t=0.875`, approximately 90% of its denoising path. It is a soft plan-first
schedule, not a separate
two-stage sampler.

The prefix-conditioned counterpart uses a deterministic 128/128 split for full
256-token OWT windows and balanced halves for shorter rows:

| Config | Observed prefix | Attention topology | Plan time |
|---|---:|---|---|
| `tier5_prefix128_joint_aligned_len256.yml` | first 128 valid tokens | joint plan/future | aligned |
| `tier5_prefix128_hierarchical_aligned_len256.yml` | first 128 valid tokens | `(prefix + plan) -> future` two-block | aligned |
| `tier5_prefix128_strict_hierarchical_aligned_len256.yml` | first 128 valid tokens | `prefix -> plan -> future` strict | aligned |
| `tier5_prefix128_hierarchical_plan_first_len256.yml` | first 128 valid tokens | `(prefix + plan) -> future` two-block | strict two-phase training |

The strict variant changes only the noisy shared denoiser. Clean token target
encoding still uses the dataset encoder mask (future rows may read the full
valid clean field), and frozen Sentence-T5 still embeds the complete clean
continuation. Inside every denoiser block, control queries read controls only,
prefix queries read controls plus prefix, plan queries read controls, prefix,
and plan, and future queries read all valid keys. This removes the plan ->
prefix feedback edge retained by the original two-block topology.

The matched plan-first variant keeps the better-performing two-block attention
topology and changes the training state distribution. Among non-decoder rows,
half are plan-phase rows with `token_t=0` and an independently sampled
`plan_t`; only plan reconstruction is active. The other half are token-phase
rows with sampled `token_t` and a completed clean plan at `plan_t=1`; only
token denoising is active. Decoder rows also receive a completed clean plan.
This separates phase allocation from `plan_loss_weight` and reports the actual
plan/token phase fractions in `train_metrics.jsonl`.
| `tier5_prefix128_hierarchical_aligned_len256.yml` | first 128 valid tokens | prefix/plan -> future | aligned |
| `tier5_prefix128_hierarchical_lead_g3_len256.yml` | first 128 valid tokens | prefix/plan -> future | `noise_power`, gamma=3 |

The continuation-only Sentence-T5 embedding is the clean plan target. Prefix
T5 rows attend only prefix keys, so the hierarchical plan prior cannot receive
future information through contextual token embeddings. Conditional free
generation, oracle/shuffled plan interventions, teacher-forced PPL, and token
denoising diagnostics all use the same prefix mask.

## Warm Start

For warm-starting plan runs from a trained pure ELF checkpoint, add overrides:

```bash
--config_override warm_start=outputs/ablations/owt_elfb/tier0_0_pure_elf/checkpoint_X
--config_override resume=null
```
