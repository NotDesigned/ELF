#!/usr/bin/env bash
# Launcher for the PyTorch ELF port.
#
# Single GPU / CPU:
#     bash scripts/launch.sh train  src/configs/training_configs/train_owt_ELF-B.yml
#     bash scripts/launch.sh eval   src/configs/training_configs/train_owt_ELF-B.yml  embedded-language-flows/ELF-B-owt
#
# Multi-GPU (single-host):
#     NGPU=8 bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
#
# Multi-host (torchrun rendezvous):
#     NGPU=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=node-0 MASTER_PORT=29500 \
#         bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: bash scripts/launch.sh <train|eval> <config.yml> [extra args...]"
    exit 1
fi

MODE=$1
CONFIG=$2
shift 2
EXTRA=("$@")

# Pick up the script we want to run.
case "$MODE" in
    train) ENTRY=src/train.py ;;
    eval)  ENTRY=src/eval.py ;;
    *) echo "Unknown mode: $MODE (expected 'train' or 'eval')"; exit 1 ;;
esac

detect_ngpu() {
    if [[ -n "${NGPU:-}" ]]; then
        printf '%s\n' "$NGPU"
        return
    fi
    for var_name in NPROC_PER_NODE LOCAL_WORLD_SIZE NUM_GPUS GPU_COUNT; do
        var_value="${!var_name:-}"
        if [[ -n "$var_value" ]]; then
            printf '%s\n' "$var_value"
            return
        fi
    done
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "all" ]]; then
        case "$CUDA_VISIBLE_DEVICES" in
            ""|"none"|"void") ;;
            *) printf '%s\n' "$(( $(tr -cd ',' <<<"$CUDA_VISIBLE_DEVICES" | wc -c) + 1 ))"; return ;;
        esac
    fi
    if [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" && "${NVIDIA_VISIBLE_DEVICES}" != "all" ]]; then
        case "$NVIDIA_VISIBLE_DEVICES" in
            ""|"none"|"void") ;;
            *) printf '%s\n' "$(( $(tr -cd ',' <<<"$NVIDIA_VISIBLE_DEVICES" | wc -c) + 1 ))"; return ;;
        esac
    fi
    python - <<'PY' 2>/dev/null || printf '1\n'
import torch
print(torch.cuda.device_count() or 1)
PY
}

NGPU=$(detect_ngpu)
if ! [[ "$NGPU" =~ ^[0-9]+$ ]] || [[ "$NGPU" -lt 1 ]]; then
    echo "Invalid NGPU value: $NGPU" >&2
    exit 1
fi
export NGPU
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

if [[ "$NGPU" == "1" && "$NNODES" == "1" ]]; then
    echo "[launch] single-process: python $ENTRY --config $CONFIG ${EXTRA[*]}"
    exec python "$ENTRY" --config "$CONFIG" "${EXTRA[@]}"
else
    echo "[launch] torchrun nproc_per_node=$NGPU nnodes=$NNODES node_rank=$NODE_RANK $ENTRY"
    exec torchrun \
        --nproc_per_node="$NGPU" \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        "$ENTRY" --config "$CONFIG" "${EXTRA[@]}"
fi
