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

# Install deps first for layer cache.
COPY requirements.txt .
RUN python -m pip install --break-system-packages --upgrade pip setuptools wheel \
    && python -m pip install --break-system-packages -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY docs/ docs/
COPY README.md ./

ENV PYTHONPATH=/app/src \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/data/.cache/huggingface \
    WANDB_DIR=/data/wandb \
    SAVE_DIR=/data/saved_models \
    NCCL_DEBUG=WARN

RUN chmod +x scripts/launch.sh scripts/cloud_train.sh \
    && mkdir -p /data/.cache/huggingface /data/wandb /data/outputs /data/saved_models

CMD ["bash"]
