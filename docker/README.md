# ELF Cloud Docker

This image follows the STAR-LDM container layout:

- code lives in `/app`
- persistent data, HF cache, wandb logs, and checkpoints live in `/data`
- training uses `scripts/cloud_train.sh`, which forwards to `scripts/launch.sh`

## Build

```bash
docker build -f docker/Dockerfile -t elf:cloud .
```

To override the PyTorch base image:

```bash
docker build \
  --build-arg PYTORCH_IMAGE=pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime \
  -f docker/Dockerfile -t elf:cloud .
```

## Run Tier 0

```bash
docker run --gpus all --rm -it \
  -v /path/to/persistent-data:/data \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  elf:cloud \
  bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf.yml
```

Other Tier 0 configs:

```bash
bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_1_sentence_t5.yml
bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_2_learned_main.yml
```

For multi-GPU single-node runs:

```bash
NGPU=8 bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_2_learned_main.yml
```

Useful runtime overrides:

```bash
USE_WANDB=false \
GLOBAL_BATCH_SIZE=64 \
LOG_FREQ=10 \
OUTPUT_ROOT=/data/outputs \
bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_2_learned_main.yml
```

Set `DRY_RUN=1` to print the final launch command without starting training.

The image defaults to `HF_ENDPOINT=https://hf-mirror.com`, matching the network
setup where the official Hugging Face endpoint is not reachable. If a cloud
provider exposes a different reachable mirror, override it with
`-e HF_ENDPOINT=...`. If downloads stall on a provider, `-e HF_HUB_DISABLE_XET=1`
is also useful.

In multi-GPU runs, each node's local rank 0 populates the Hugging Face cache
before peer ranks read it. If a previous interrupted run left stale dataset
cache entries, the loader prunes matching `*.incomplete` directories under
`$HF_HOME/datasets` before retrying.
