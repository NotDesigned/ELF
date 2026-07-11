# syntax=docker/dockerfile:1.7
ARG PYTORCH_IMAGE=pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --break-system-packages --upgrade pip setuptools wheel \
    && python -m pip install --break-system-packages -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY tools/ tools/
COPY tests/ tests/
COPY docs/ docs/
COPY experiments/ experiments/
COPY README.md ./

ARG PROJECT_NAME=elf
ARG BAKED_HF_HOME=/opt/elf/hf-cache
ARG BAKED_CHECKPOINT_ROOT=/opt/elf/checkpoints
ARG ELF_B_CHECKPOINT_FILE=checkpoint_95085
ARG SOURCE_ID=unknown

ENV PROJECT_NAME=${PROJECT_NAME} \
    PYTHONPATH=/app/src \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/data/.cache/huggingface \
    HF_DATASETS_CACHE=/data/.cache/huggingface/datasets \
    BAKED_HF_HOME=${BAKED_HF_HOME} \
    BAKED_CHECKPOINT_ROOT=${BAKED_CHECKPOINT_ROOT} \
    ELF_SOURCE_ID=${SOURCE_ID} \
    ELF_B_OWT_CHECKPOINT=/data/${PROJECT_NAME}/checkpoints/ELF-B-owt-torch/${ELF_B_CHECKPOINT_FILE} \
    WANDB_DIR=/data/${PROJECT_NAME}/wandb \
    WANDB_CACHE_DIR=/data/${PROJECT_NAME}/wandb_cache \
    SAVE_DIR=/data/${PROJECT_NAME}/saved_models \
    NCCL_DEBUG=WARN \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

RUN chmod +x scripts/launch.sh scripts/cloud_train.sh scripts/source_identity.sh tools/push_registry_image.sh \
    && mkdir -p /data/.cache/huggingface /data/${PROJECT_NAME}/wandb \
        /data/${PROJECT_NAME}/wandb_cache /data/${PROJECT_NAME}/outputs \
        /data/${PROJECT_NAME}/saved_models /data/${PROJECT_NAME}/checkpoints

CMD ["bash"]
