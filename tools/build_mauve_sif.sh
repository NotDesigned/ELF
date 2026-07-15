#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 BASE.sif OUTPUT.sif WORK_DIR" >&2
    exit 2
fi

BASE_SIF=$1
OUTPUT_SIF=$2
WORK_DIR=$3
SANDBOX="$WORK_DIR/rootfs"
TMP_SIF="$WORK_DIR/output.sif"
TMP_SQUASHFS="$WORK_DIR/rootfs.squashfs"

[[ -s "$BASE_SIF" ]] || { echo "missing base SIF: $BASE_SIF" >&2; exit 2; }
[[ ! -e "$OUTPUT_SIF" ]] || { echo "refusing to overwrite: $OUTPUT_SIF" >&2; exit 2; }

mkdir -p -- "$WORK_DIR"

if [[ ! -d "$SANDBOX" ]]; then
    SQUASHFS_OFFSET=$(
        apptainer sif list "$BASE_SIF" \
            | awk '/FS \(Squashfs/ {gsub(/[^0-9-]/, "", $4); split($4, range, "-"); print range[1]; exit}'
    )
    [[ "$SQUASHFS_OFFSET" =~ ^[0-9]+$ ]] \
        || { echo "could not discover SquashFS offset" >&2; exit 2; }

    unsquashfs -processors "${SLURM_CPUS_PER_TASK:-1}" \
        -f -o "$SQUASHFS_OFFSET" -d "$SANDBOX" "$BASE_SIF"
    apptainer exec --writable "$SANDBOX" python -m pip install --no-cache-dir \
        "mauve-text==0.4.0" \
        "faiss-cpu==1.14.3"
else
    echo "reusing existing sandbox: $SANDBOX"
fi

apptainer exec "$SANDBOX" python -m pip check
apptainer exec "$SANDBOX" python - <<'PY'
import importlib.metadata as metadata
import faiss
import mauve

assert metadata.version("mauve-text") == "0.4.0"
assert metadata.version("faiss-cpu") == "1.14.3"
print("validated mauve-text=0.4.0 faiss-cpu=1.14.3")
PY

rm -f -- "$TMP_SIF"
if [[ ! -s "$TMP_SQUASHFS" ]]; then
    /usr/bin/mksquashfs "$SANDBOX" "$TMP_SQUASHFS" \
        -noappend -all-root -processors "${SLURM_CPUS_PER_TASK:-1}"
else
    echo "reusing existing SquashFS: $TMP_SQUASHFS"
fi
apptainer sif new "$TMP_SIF"
apptainer sif add "$TMP_SIF" "$TMP_SQUASHFS" \
    --datatype 4 --parttype 2 --partfs 1 --partarch 2 --groupid 1
apptainer exec "$TMP_SIF" python - <<'PY'
import importlib.metadata as metadata
import faiss
import mauve

assert metadata.version("mauve-text") == "0.4.0"
assert metadata.version("faiss-cpu") == "1.14.3"
print("validated assembled SIF")
PY
mv -- "$TMP_SIF" "$OUTPUT_SIF"
sha256sum -- "$OUTPUT_SIF"
rm -f -- "$TMP_SQUASHFS"
