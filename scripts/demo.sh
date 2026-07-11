#!/usr/bin/env bash
# Short ELF generation demo, similar in spirit to STAR-LDM's basic generation.
#
# Examples:
#   bash scripts/demo.sh embedded-language-flows/ELF-B-owt-torch
#   CONDA_ENV=elf CHECKPOINT_PATH=/data/elf/checkpoints/ELF-B-owt-torch/checkpoint_95085 bash scripts/demo.sh
#   PPL=1 RECONSTRUCTION=1 NUM_SAMPLES=8 BATCH_SIZE=2 bash scripts/demo.sh
#     RECONSTRUCTION=1 enables oracle/shuffled plan PPL when available plus token_recon_ppl.
set -euo pipefail

if [[ $# -gt 0 ]]; then
    CHECKPOINT_PATH="$1"
    shift
else
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-embedded-language-flows/ELF-B-owt-torch}"
fi

CONFIG="${CONFIG:-src/configs/training_configs/train_owt_ELF-B.yml}"
SAMPLING_CONFIGS_PATH="${SAMPLING_CONFIGS_PATH:-src/configs/sampling_configs/demo_uncond.yml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/demo/elf_b_owt}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PPL="${PPL:-0}"
RECONSTRUCTION="${RECONSTRUCTION:-0}"
EVAL_PPL_BATCH_SIZE="${EVAL_PPL_BATCH_SIZE:-2}"
PRINT_SAMPLES="${PRINT_SAMPLES:-4}"
SEED="${SEED:-42}"
USE_BF16="${USE_BF16:-true}"
USE_COMPILE="${USE_COMPILE:-false}"

if [[ -n "${CONDA_ENV:-}" ]]; then
    PY_CMD=(conda run --no-capture-output -n "$CONDA_ENV" python)
else
    PY_CMD=("${PYTHON:-python}")
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "[demo] config=$CONFIG"
echo "[demo] checkpoint=$CHECKPOINT_PATH"
echo "[demo] samples=$NUM_SAMPLES batch=$BATCH_SIZE sampling=$SAMPLING_CONFIGS_PATH"
echo "[demo] output_dir=$OUTPUT_DIR"

"${PY_CMD[@]}" src/eval.py \
    --config "$CONFIG" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --seed "$SEED" \
    --config_override "output_dir=$OUTPUT_DIR" \
    --config_override "sampling_configs_path=$SAMPLING_CONFIGS_PATH" \
    --config_override "num_samples=$NUM_SAMPLES" \
    --config_override "global_batch_size=$BATCH_SIZE" \
    --config_override "use_wandb=false" \
    --config_override "online_eval=$PPL" \
    --config_override "reconstruction_eval=$RECONSTRUCTION" \
    --config_override "reconstruction_num_samples=$NUM_SAMPLES" \
    --config_override "eval_ppl_batch_size=$EVAL_PPL_BATCH_SIZE" \
    --config_override "use_bf16=$USE_BF16" \
    --config_override "use_compile=$USE_COMPILE" \
    "$@"

generated_file="$(
    find "$OUTPUT_DIR" -type f -name 'all_generated_*.jsonl' 2>/dev/null \
        | LC_ALL=C sort \
        | tail -n 1
)"

if [[ -z "$generated_file" ]]; then
    echo "[demo] no generated JSONL found under $OUTPUT_DIR"
    exit 0
fi

echo
echo "[demo] generated file: $generated_file"
"${PY_CMD[@]}" - "$generated_file" "$PRINT_SAMPLES" <<'PY'
import json
import sys
import textwrap

path = sys.argv[1]
limit = int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= limit:
            break
        obj = json.loads(line)
        text = " ".join(str(obj.get("generated", "")).split())
        print(f"\n--- sample {idx + 1} ---")
        print(textwrap.shorten(text, width=1000, placeholder=" ..."))
PY
