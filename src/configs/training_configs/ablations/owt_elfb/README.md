# OWT ELF-B Sentence-Plan Ablations

These configs use `base_config` inheritance from `../../train_owt_ELF-B.yml`.
Each file should change one scientific axis, not just rename an existing run.

Run with the normal training entrypoint:

```bash
python src/train.py --config src/configs/training_configs/ablations/owt_elfb/tier0_2_learned_main.yml
```

## Required Anchors

Run these first. They are also the Tier 1 learned-vs-frozen comparison.

| Config | Purpose |
|---|---|
| `tier0_0_pure_elf.yml` | Original ELF baseline, `use_sentence_plan=false`. |
| `tier0_1_sentence_t5.yml` | Frozen Sentence-T5 plan sanity / teacher baseline. |
| `tier0_2_learned_main.yml` | Main hypothesis: learned sentence plan with `sentence_encoder_grad=none` and `plan_aux_passes=1`. |

Core question: compare `tier0_1_sentence_t5.yml` against
`tier0_2_learned_main.yml` to test whether the sentence embedding can be
learned while the word/token T5 field stays fixed.

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

## Warm Start

For warm-starting plan runs from a trained pure ELF checkpoint, add overrides:

```bash
--config_override warm_start=outputs/ablations/owt_elfb/tier0_0_pure_elf/checkpoint_X
--config_override resume=null
```
