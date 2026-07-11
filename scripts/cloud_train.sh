#!/usr/bin/env bash
# Cloud-container launcher for ELF ablation runs.
#
# Examples:
#   bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf.yml
#   CONFIG=src/configs/training_configs/ablations/owt_elfb/tier0_2_learned_main.yml \
#       NGPU=8 PROJECT_NAME=elf bash scripts/cloud_train.sh
#   HYDRATE_ONLY=1 bash scripts/cloud_train.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DEFAULT_CONFIG="src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf.yml"

# Project code may be mounted over the immutable image. The scheduler package
# remains the independently installed, commit-pinned image dependency.
export PYTHONPATH="$REPO_ROOT/src:/app/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python - <<'PY'
import experiment_control
from experiment_control.project import AssetRequirement
print(f"[cloud_train] experiment_control={experiment_control.__file__}")
PY

if [[ $# -gt 0 ]]; then
    CONFIG="$1"
    shift
else
    CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
fi
extra_args=("$@")
preflight_config_overrides=()
for ((preflight_index = 0; preflight_index < ${#extra_args[@]}; preflight_index++)); do
    preflight_arg="${extra_args[$preflight_index]}"
    if [[ "$preflight_arg" == "--config_override" ]]; then
        if ((preflight_index + 1 >= ${#extra_args[@]})); then
            echo "[cloud_train] --config_override requires FIELD=VALUE" >&2
            exit 1
        fi
        preflight_index=$((preflight_index + 1))
        preflight_config_overrides+=("${extra_args[$preflight_index]}")
    elif [[ "$preflight_arg" == --config_override=* ]]; then
        preflight_config_overrides+=("${preflight_arg#--config_override=}")
    fi
done

PROJECT_NAME="${PROJECT_NAME:-elf}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/.runtime}"
PROJECT_DATA_ROOT="${PROJECT_DATA_ROOT:-$DATA_ROOT/$PROJECT_NAME}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DATA_ROOT/runs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_DATA_ROOT/checkpoints}"

run_path="${CONFIG#src/configs/training_configs/}"
run_path="${run_path%.yml}"
run_path="${run_path%.yaml}"
run_slug="$(printf '%s' "$run_path" | tr '/_' '--' | tr -cd '[:alnum:].-')"
if [[ -z "${RUN_ID:-}" ]]; then
    run_nonce="$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
    RUN_ID="${run_slug}-$(date -u +%Y%m%dT%H%M%SZ)-${run_nonce}"
fi
ATTEMPT_ID="${ATTEMPT_ID:-attempt-001}"
BACKEND="${BACKEND:-local}"
BACKEND_JOB_ID="${BACKEND_JOB_ID:-}"
QUOTA_TYPE="${QUOTA_TYPE:-unknown}"

OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_ID}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-$DATA_ROOT/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
WANDB_DIR="${WANDB_DIR:-$PROJECT_DATA_ROOT/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PROJECT_DATA_ROOT/wandb_cache}"
SAVE_DIR="${SAVE_DIR:-$PROJECT_DATA_ROOT/saved_models}"
BAKED_HF_HOME="${BAKED_HF_HOME:-/opt/$PROJECT_NAME/hf-cache}"
BAKED_CHECKPOINT_ROOT="${BAKED_CHECKPOINT_ROOT:-/opt/$PROJECT_NAME/checkpoints}"
DATASET_ID="${DATASET_ID:-embedded-language-flows/openwebtext-t5}"
ENCODER_MODEL="${ENCODER_MODEL:-t5-small}"
ELF_B_CHECKPOINT_FILE="${ELF_B_CHECKPOINT_FILE:-checkpoint_95085}"
ELF_B_OWT_CHECKPOINT="${ELF_B_OWT_CHECKPOINT:-$CHECKPOINT_ROOT/ELF-B-owt-torch/$ELF_B_CHECKPOINT_FILE}"
SOURCE_ID="${SOURCE_ID:-${ELF_SOURCE_ID:-unknown}}"
RUNTIME_TREE_ID="${RUNTIME_TREE_ID:-$SOURCE_ID}"
GIT_COMMIT="${GIT_COMMIT:-unknown}"
CAMPAIGN_ID="${CAMPAIGN_ID:-unknown}"
CAMPAIGN_NAME="${CAMPAIGN_NAME:-unknown}"
IMAGE_ID="${IMAGE_ID:-${ELF_IMAGE_ID:-unknown}}"

# Return success when a launcher value uses a supported true spelling.
truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

# Convert a Hugging Face model ID to its on-disk hub cache directory name.
cache_key_model() {
    printf 'models--%s\n' "$(printf '%s' "$1" | sed 's#/#--#g')"
}

# Convert a Hugging Face dataset ID to its datasets-cache directory name.
cache_key_dataset() {
    printf '%s\n' "$(printf '%s' "$1" | sed 's#/#___#g')"
}

# Return a baked-directory content identity from its marker or file metadata.
dir_id() {
    local source_dir="$1"
    local id_file="$2"

    if [[ -f "$source_dir/$id_file" ]]; then
        tr -d '[:space:]' < "$source_dir/$id_file"
        return
    fi

    find "$source_dir" -type f ! -name "$id_file" -printf '%P %s %T@\n' \
        | LC_ALL=C sort \
        | sha256sum \
        | awk '{print $1}'
}

# Return success when two paths resolve to the same filesystem directory.
same_dir() {
    local left="$1"
    local right="$2"
    [[ "$(readlink -f "$left" 2>/dev/null || printf '%s' "$left")" == \
       "$(readlink -f "$right" 2>/dev/null || printf '%s' "$right")" ]]
}

# Hydrate one baked asset tree exactly once across concurrent workers.
copy_baked_dir_once() {
    local source_dir="$1"
    local dest_dir="$2"
    local id_file="$3"
    local marker_subdir="$4"
    local label="$5"

    if [[ ! -d "$source_dir" ]]; then
        echo "[cloud_train] no baked $label at $source_dir; skip hydrate"
        return
    fi

    local content_id marker_dir marker_file lock_dir waited
    content_id="$(dir_id "$source_dir" "$id_file")"
    if [[ -z "$content_id" ]]; then
        echo "[cloud_train] failed to compute baked $label id from $source_dir" >&2
        exit 1
    fi

    marker_dir="$dest_dir/$marker_subdir"
    marker_file="$marker_dir/$content_id"
    lock_dir="$marker_dir/$content_id.lock"

    mkdir -p "$dest_dir" "$marker_dir"
    if [[ -f "$marker_file" ]]; then
        echo "[cloud_train] baked $label already hydrated: $content_id"
        return
    fi

    if mkdir "$lock_dir" 2>/dev/null; then
        (
            trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT
            echo "[cloud_train] hydrating baked $label $content_id -> $dest_dir"
            if ! same_dir "$source_dir" "$dest_dir"; then
                cp -a --no-preserve=ownership "$source_dir"/. "$dest_dir"/
            fi
            touch "$marker_file"
        )
        return
    fi

    echo "[cloud_train] waiting for another process to hydrate baked $label: $content_id"
    waited=0
    while [[ -d "$lock_dir" && ! -f "$marker_file" ]]; do
        sleep 5
        waited=$((waited + 5))
        if [[ "$waited" -ge "${HYDRATE_LOCK_TIMEOUT_SECONDS:-3600}" ]]; then
            echo "[cloud_train] timed out waiting for hydrate lock: $lock_dir" >&2
            exit 1
        fi
    done

    if [[ -f "$marker_file" ]]; then
        echo "[cloud_train] baked $label hydrated by peer: $content_id"
        return
    fi

    copy_baked_dir_once "$source_dir" "$dest_dir" "$id_file" "$marker_subdir" "$label"
}

# Hydrate the image's HF cache and checkpoints into persistent shared storage.
hydrate_baked_assets() {
    copy_baked_dir_once \
        "$BAKED_HF_HOME" \
        "$HF_HOME" \
        ".baked-cache-id" \
        ".baked-cache-markers" \
        "hf-cache"

    copy_baked_dir_once \
        "$BAKED_CHECKPOINT_ROOT" \
        "$CHECKPOINT_ROOT" \
        ".baked-checkpoints-id" \
        ".baked-checkpoint-markers" \
        "checkpoints"
}

# Fail with an actionable message unless a required directory exists.
require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "[cloud_train] required directory is missing: $1" >&2
        exit 1
    fi
}

# Fail with an actionable message unless a required non-empty file exists.
require_file() {
    if [[ ! -s "$1" ]]; then
        echo "[cloud_train] required file is missing or empty: $1" >&2
        exit 1
    fi
}

# Require either an explicit local model directory or the corresponding
# Hugging Face hub cache directory. Model IDs are converted using the hub's
# models--organization--name convention.
require_model_cache() {
    local model_name="$1"
    if [[ "$model_name" == /* || "$model_name" == ./* ]]; then
        require_dir "$model_name"
    else
        require_dir "$HF_HOME/hub/$(cache_key_model "$model_name")"
    fi
}

# Require an explicit local dataset directory or its Hugging Face cache root.
require_dataset_cache() {
    local dataset_name="$1"
    if [[ "$dataset_name" == /* || "$dataset_name" == ./* ]]; then
        require_dir "$dataset_name"
    else
        require_dir "$HF_DATASETS_CACHE/$(cache_key_dataset "$dataset_name")"
    fi
}

# Resolve the selected config and verify every offline asset needed at runtime.
fail_fast_for_offline_cache() {
    local require_offline_cache="${REQUIRE_OFFLINE_CACHE:-}"
    if [[ -z "$require_offline_cache" ]]; then
        require_offline_cache=0
        if truthy "${HF_HUB_OFFLINE:-0}" || truthy "${TRANSFORMERS_OFFLINE:-0}" || truthy "${HF_DATASETS_OFFLINE:-0}"; then
            require_offline_cache=1
        fi
    fi

    if ! truthy "$require_offline_cache"; then
        return
    fi

    local asset_output asset_kind asset_identity asset_reason
    local asset_cmd=(python -m elf_experiments.assets plan "$CONFIG" --format tsv)
    for preflight_override in "${preflight_config_overrides[@]}"; do
        asset_cmd+=(--config-override "$preflight_override")
    done
    asset_output="$("${asset_cmd[@]}")"
    while IFS=$'\t' read -r asset_kind asset_identity asset_reason; do
        [[ -n "$asset_kind" ]] || continue
        case "$asset_kind" in
            model) require_model_cache "$asset_identity" ;;
            dataset) require_dataset_cache "$asset_identity" ;;
            file) require_file "$asset_identity" ;;
            *) echo "[cloud_train] unknown asset kind: $asset_kind" >&2; exit 1 ;;
        esac
    done <<< "$asset_output"

    if truthy "${USE_ELF_B_WARM_START:-0}" || truthy "${REQUIRE_ELF_B_CHECKPOINT:-0}"; then
        require_file "$ELF_B_OWT_CHECKPOINT"
    fi
    if [[ -n "${WARM_START:-}" ]]; then
        require_file "$WARM_START"
    fi
}

# Resolve per-node GPU count from explicit launcher/scheduler signals or PyTorch.
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

export PROJECT_NAME
export PROJECT_DATA_ROOT
export HF_ENDPOINT
export HF_HOME
export HF_DATASETS_CACHE
export WANDB_DIR
export WANDB_CACHE_DIR
export SAVE_DIR
export ELF_B_OWT_CHECKPOINT
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$WANDB_DIR" \
        "$WANDB_CACHE_DIR" "$SAVE_DIR" "$CHECKPOINT_ROOT"
    hydrate_baked_assets
    fail_fast_for_offline_cache
fi

if truthy "${USE_ELF_B_WARM_START:-0}" && [[ -z "${WARM_START:-}" ]]; then
    WARM_START="$ELF_B_OWT_CHECKPOINT"
fi

overrides=()
manifest_overrides=()

# Add one typed override to both the training command and frozen manifest config.
add_override() {
    overrides+=(--config_override "$1")
    manifest_overrides+=(--config-override "$1")
}

# Mirror explicit Python config overrides into the resolved manifest. Other
# Python flags remain recorded in the sanitized command but do not alter Config.
for ((extra_index = 0; extra_index < ${#extra_args[@]}; extra_index++)); do
    extra_arg="${extra_args[$extra_index]}"
    if [[ "$extra_arg" == "--config_override" ]]; then
        if ((extra_index + 1 >= ${#extra_args[@]})); then
            echo "[cloud_train] --config_override requires FIELD=VALUE" >&2
            exit 1
        fi
        extra_index=$((extra_index + 1))
        manifest_overrides+=(--config-override "${extra_args[$extra_index]}")
    elif [[ "$extra_arg" == --config_override=* ]]; then
        manifest_overrides+=(--config-override "${extra_arg#--config_override=}")
    fi
done

WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_ID}"
WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_ID}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
export RUN_ID WANDB_RUN_NAME WANDB_RUN_ID WANDB_RESUME
export USE_WANDB WANDB_PROJECT WANDB_ENTITY GLOBAL_BATCH_SIZE BATCH_SIZE
export NUM_WORKERS LOG_FREQ USE_COMPILE WARM_START WARM_START_USE_EMA RESUME HF_REPO_ID
override_output="$(python -m elf_experiments.overrides --output-dir "$OUTPUT_DIR" --format lines)"
while IFS= read -r planned_override; do
    [[ -n "$planned_override" ]] && add_override "$planned_override"
done <<< "$override_output"

echo "[cloud_train] config=$CONFIG"
echo "[cloud_train] run_id=$RUN_ID attempt_id=$ATTEMPT_ID backend=$BACKEND backend_job_id=${BACKEND_JOB_ID:-<pending>}"
echo "[cloud_train] output_dir=$OUTPUT_DIR"
echo "[cloud_train] data_root=$DATA_ROOT"
echo "[cloud_train] project_data_root=$PROJECT_DATA_ROOT"
echo "[cloud_train] hf_home=$HF_HOME hf_datasets_cache=$HF_DATASETS_CACHE"
echo "[cloud_train] hf_endpoint=$HF_ENDPOINT offline=${HF_HUB_OFFLINE:-0}/${TRANSFORMERS_OFFLINE:-0}/${HF_DATASETS_OFFLINE:-0}"
echo "[cloud_train] wandb_dir=$WANDB_DIR wandb_run_id=$WANDB_RUN_ID save_dir=$SAVE_DIR"
echo "[cloud_train] checkpoint_root=$CHECKPOINT_ROOT"
echo "[cloud_train] baked_hf_home=$BAKED_HF_HOME baked_checkpoint_root=$BAKED_CHECKPOINT_ROOT"
echo "[cloud_train] NGPU=$NGPU NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0}"

cmd=(bash scripts/launch.sh train "$CONFIG" "${overrides[@]}" "${extra_args[@]}")
printf '[cloud_train] command:'
for command_arg in "${cmd[@]}"; do
    command_key="${command_arg%%=*}"
    case "${command_key,,}" in
        *secret*|*token*|*password*|*credential*|*access_key*|*access-key*|*api_key*|*api-key*|*proxy*|*authorization*|*cookie*)
            printf ' %q' "$command_key=<redacted>"
            ;;
        *)
            printf ' %q' "$command_arg"
            ;;
    esac
done
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

if truthy "${HYDRATE_ONLY:-0}"; then
    echo "[cloud_train] HYDRATE_ONLY=1, exiting after cache/checkpoint hydration"
    exit 0
fi

manifest_cmd=(
    python -m elf_experiments.manifest
    --project "$PROJECT_NAME"
    --run-id "$RUN_ID"
    --attempt-id "$ATTEMPT_ID"
    --backend "$BACKEND"
    --backend-job-id "$BACKEND_JOB_ID"
    --config "$CONFIG"
    --output-dir "$OUTPUT_DIR"
    --source-id "$SOURCE_ID"
    --runtime-tree-id "$RUNTIME_TREE_ID"
    --git-commit "$GIT_COMMIT"
    --campaign-id "$CAMPAIGN_ID"
    --campaign "$CAMPAIGN_NAME"
    --image-id "$IMAGE_ID"
    --gpus "$NGPU"
    --nodes "${NNODES:-1}"
    --quota "$QUOTA_TYPE"
    --resource-spec "${RESOURCE_SPEC:-}"
    --max-infra-retries "${MAX_INFRA_RETRIES:-2}"
    "${manifest_overrides[@]}"
)
if [[ -n "${RESEARCH_CONTRACT_B64:-}" || -n "${RESEARCH_ROLE:-}" ]]; then
    [[ -n "${RESEARCH_CONTRACT_B64:-}" && -n "${RESEARCH_ROLE:-}" ]] || {
        echo "[cloud_train] research contract and role must be supplied together" >&2
        exit 2
    }
    manifest_cmd+=(
        --research-contract-b64 "$RESEARCH_CONTRACT_B64"
        --research-role "$RESEARCH_ROLE"
    )
fi
if truthy "${REQUIRE_IMMUTABLE_IDENTITIES:-1}"; then
    manifest_cmd+=(--require-immutable-identities)
fi
manifest_cmd+=(-- "${cmd[@]}")
"${manifest_cmd[@]}"

if truthy "${PREPARE_ONLY:-0}"; then
    echo "[cloud_train] PREPARE_ONLY=1, exiting after manifest creation"
    exit 0
fi

# Persist one normalized process state transition for the current attempt.
record_lifecycle() {
    local state="$1"
    local event="$2"
    shift 2
    python -m elf_experiments.manifest record \
        --project "$PROJECT_NAME" \
        --run-id "$RUN_ID" \
        --attempt-id "$ATTEMPT_ID" \
        --output-dir "$OUTPUT_DIR" \
        --state "$state" \
        --event "$event" \
        "$@"
}

attempt_log_dir="$OUTPUT_DIR/attempts/$ATTEMPT_ID"
record_lifecycle RUNNING process_started
set +e
"${cmd[@]}" \
    > >(tee -a "$attempt_log_dir/stdout.log") \
    2> >(tee -a "$attempt_log_dir/stderr.log" >&2)
exit_code=$?
set -e

if [[ "$exit_code" -eq 0 ]]; then
    record_lifecycle SUCCEEDED process_exited --exit-code "$exit_code"
else
    record_lifecycle FAILED process_exited --exit-code "$exit_code" --reason training_process_nonzero_exit
fi
exit "$exit_code"
