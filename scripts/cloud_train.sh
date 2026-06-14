#!/usr/bin/env bash
# Cloud-container launcher for ELF ablation runs.
#
# Examples:
#   bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf.yml
#   CONFIG=src/configs/training_configs/ablations/owt_elfb/tier0_2_learned_main.yml \
#       NGPU=8 OUTPUT_ROOT=/data/outputs bash scripts/cloud_train.sh
set -euo pipefail

DEFAULT_CONFIG="src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf.yml"

if [[ $# -gt 0 ]]; then
    CONFIG="$1"
    shift
else
    CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
fi

DATA_ROOT="${DATA_ROOT:-/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/outputs}"

run_path="${CONFIG#src/configs/training_configs/}"
run_path="${run_path%.yml}"
run_path="${run_path%.yaml}"

OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$run_path}"

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

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/.cache/huggingface}"
export WANDB_DIR="${WANDB_DIR:-$DATA_ROOT/wandb}"
export SAVE_DIR="${SAVE_DIR:-$DATA_ROOT/saved_models}"
export PYTHONPATH="/app/src:${PYTHONPATH:-}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$WANDB_DIR" "$SAVE_DIR"
fi

overrides=(
    --config_override "output_dir=$OUTPUT_DIR"
)

if [[ -n "${USE_WANDB:-}" ]]; then
    overrides+=(--config_override "use_wandb=$USE_WANDB")
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    overrides+=(--config_override "wandb_project=$WANDB_PROJECT")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
    overrides+=(--config_override "wandb_entity=$WANDB_ENTITY")
fi
if [[ -n "${GLOBAL_BATCH_SIZE:-}" ]]; then
    overrides+=(--config_override "global_batch_size=$GLOBAL_BATCH_SIZE")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
    overrides+=(--config_override "global_batch_size=null")
    overrides+=(--config_override "batch_size=$BATCH_SIZE")
fi
if [[ -n "${NUM_WORKERS:-}" ]]; then
    overrides+=(--config_override "num_workers=$NUM_WORKERS")
fi
if [[ -n "${LOG_FREQ:-}" ]]; then
    overrides+=(--config_override "log_freq=$LOG_FREQ")
fi
if [[ -n "${USE_COMPILE:-}" ]]; then
    overrides+=(--config_override "use_compile=$USE_COMPILE")
fi
if [[ -n "${WARM_START:-}" ]]; then
    overrides+=(--config_override "warm_start=$WARM_START")
fi
if [[ -n "${WARM_START_USE_EMA:-}" ]]; then
    overrides+=(--config_override "warm_start_use_ema=$WARM_START_USE_EMA")
fi
if [[ -n "${HF_REPO_ID:-}" ]]; then
    overrides+=(--config_override "hf_repo_id=$HF_REPO_ID")
fi

echo "[cloud_train] config=$CONFIG"
echo "[cloud_train] output_dir=$OUTPUT_DIR"
echo "[cloud_train] data_root=$DATA_ROOT"
echo "[cloud_train] hf_home=$HF_HOME hf_endpoint=$HF_ENDPOINT"
echo "[cloud_train] NGPU=$NGPU NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0}"

cmd=(bash scripts/launch.sh train "$CONFIG" "${overrides[@]}" "$@")
printf '[cloud_train] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

exec "${cmd[@]}"
