# PyTorch ELF

PyTorch version of [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938).

## Installation

Create a conda environment named `elf` and install the dependencies:

```bash
conda create -n elf python=3.10 -y
conda activate elf
pip install -r requirements.txt
pip install -e packages/experiment-control
```

The experiment backend package has its own API and integration guide in
[`packages/experiment-control/README.md`](packages/experiment-control/README.md).

Then log in to WandB to track your experiments if needed:

```bash
wandb login YOUR_WANDB_API_KEY
```

## Converted Checkpoints

We provide PyTorch-converted versions of the official JAX checkpoints on HuggingFace:

| Model | Task | Params | HuggingFace Repo |
| --- | --- | --- | --- |
| ELF-B | OpenWebText (unconditional) | 105M | [embedded-language-flows/ELF-B-owt-torch](https://huggingface.co/embedded-language-flows/ELF-B-owt-torch) |
| ELF-M | OpenWebText (unconditional) | 342M | [embedded-language-flows/ELF-M-owt-torch](https://huggingface.co/embedded-language-flows/ELF-M-owt-torch) |
| ELF-L | OpenWebText (unconditional) | 652M | [embedded-language-flows/ELF-L-owt-torch](https://huggingface.co/embedded-language-flows/ELF-L-owt-torch) |
| ELF-B | XSum (summarization) | 105M | [embedded-language-flows/ELF-B-xsum-torch](https://huggingface.co/embedded-language-flows/ELF-B-xsum-torch) |
| ELF-B | WMT14 De-En (translation) | 105M | [embedded-language-flows/ELF-B-de-en-torch](https://huggingface.co/embedded-language-flows/ELF-B-de-en-torch) |

These are pulled automatically via `--checkpoint_path <hf-repo-id>` — no manual download needed.

## Reference Results

The PyTorch port targets parity with the JAX reference numbers from the
paper. Small differences (≲1 PPL, ≲0.5 BLEU/ROUGE) are expected due to bf16
vs. JAX TPU numerics and sampling stochasticity.

**Unconditional generation (OpenWebText), expected:**

| Model | Sampling | Gen. PPL ↓ | Entropy ↑ |
| --- | --- | --- | --- |
| ELF-B (105M) | 32-step SDE | 24.1 | 5.15 |
| ELF-M (342M) | 64-step SDE | 21.7 | 5.18 |
| ELF-L (652M) | 64-step SDE | 23.3 | 5.28 |

Gen. PPL is computed under a frozen GPT-2 Large; entropy is unigram entropy
over the generated tokens. Default sampling configs
(`src/configs/sampling_configs/uncond_sampling_configs.yml`) use SC-CFG=3 and
γ=1.5 (32-step) or γ=1.0 (64-step).

**Conditional generation (ELF-B), expected on the validation set:**

| Task | Metric | Reference (paper, test) | Validation |
| --- | --- | --- | --- |
| WMT14 De-En | BLEU ↑ | 26.4 | ≈ 26.7 |
| XSum | ROUGE-1 ↑ | 36.0 | ≈ 36.3 |
| XSum | ROUGE-2 ↑ | 12.2 | ≈ 12.5 |
| XSum | ROUGE-L ↑ | 27.8 | ≈ 28.1 |

Default conditional sampling
(`src/configs/sampling_configs/cond_sampling_configs.yml`): 64-step ODE,
CFG=2, SC-CFG=1.

The paper numbers were computed on TPU v5p-64; numbers from this PyTorch port
on 8× L40S / H200 should land within sampling noise (typically <1 PPL or
<0.5 metric points).

## Training

Launch single-GPU training:

```bash
bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
```

Launch multi-GPU (single-host) training:

```bash
NGPU=8 bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
```

Available training configs:

- `src/configs/training_configs/train_owt_ELF-B.yml` — ELF-B on OpenWebText
- `src/configs/training_configs/train_owt_ELF-M.yml` — ELF-M on OpenWebText
- `src/configs/training_configs/train_owt_ELF-L.yml` — ELF-L on OpenWebText
- `src/configs/training_configs/train_de-en_ELF-B.yml` — WMT14 De-En machine translation
- `src/configs/training_configs/train_xsum_ELF-B.yml` — XSum abstractive summarization

See [`docs/config_reference.md`](docs/config_reference.md) for every training,
sampling, CLI, and cloud-launcher flag, including gradient accumulation and
resume semantics. See [`docs/experiment_workflow.md`](docs/experiment_workflow.md)
for durable manifests, per-function automation contracts, offline preflight,
campaign summaries, and the first fusion experiment decision set.

**Estimated wall-clock:** ~4 h per epoch on 8× H200 (OpenWebText, ELF-B,
global batch size 512, bf16). The default ELF-B OWT run is 5 epochs.

## Quick demo

Generate a few OpenWebText-style samples from a checkpoint with a short 8-step
demo sampler:

```bash
bash scripts/demo.sh embedded-language-flows/ELF-B-owt-torch
```

Useful overrides:

```bash
CONDA_ENV=elf NUM_SAMPLES=8 BATCH_SIZE=2 bash scripts/demo.sh /path/to/checkpoint_95085
PPL=1 RECONSTRUCTION=1 NUM_SAMPLES=8 BATCH_SIZE=2 bash scripts/demo.sh embedded-language-flows/ELF-B-owt-torch
```

The script writes outputs under `outputs/demo/elf_b_owt` by default and prints
the first few generated samples in the terminal.

## Evaluation

Run evaluation against the converted checkpoints on HuggingFace. We recommend
passing `use_bf16=true` (matches the bf16 autocast used at training time) and
`use_compile=true` (wraps the eval model in `torch.compile`) for a ~3–4×
speedup on consumer GPUs:

**Unconditional generation (OpenWebText):**

```bash
# ELF-B (105M)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-owt-torch \
    --config_override use_bf16=true --config_override use_compile=true

# ELF-M (342M)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-M.yml \
    --checkpoint_path embedded-language-flows/ELF-M-owt-torch \
    --config_override use_bf16=true --config_override use_compile=true

# ELF-L (652M)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-L.yml \
    --checkpoint_path embedded-language-flows/ELF-L-owt-torch \
    --config_override use_bf16=true --config_override use_compile=true
```

**Conditional generation (XSum / WMT14 De-En):**

```bash
# XSum (ROUGE)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_xsum_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-xsum-torch \
    --config_override use_bf16=true --config_override use_compile=true

# WMT14 De-En (BLEU)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_de-en_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-de-en-torch \
    --config_override use_bf16=true --config_override use_compile=true
```

### Eval config flags

| Flag | Default | What it does |
| --- | --- | --- |
| `use_bf16` | `true` | Wraps the sampling forward in `torch.amp.autocast('cuda', dtype=bfloat16)`. Mirrors the training-time precision; output heads stay fp32. |
| `use_compile` | `false` | Wraps the eval model in `torch.compile`. First batch is slower due to tracing; subsequent batches run materially faster. |

Both flags are also editable in the YAML config under the same names. You can also run the standalone
PPL script afterwards:

```bash
python scripts/eval_ppl.py \
    --input outputs/<run>/<sampling_dir>/all_generated_*.jsonl \
    --batch_size 16
```
