#!/usr/bin/env bash
set -euo pipefail

mode=${1:---full}

if [[ "$mode" == "--runtime" ]]; then
    # Hash only files that can affect the built image or training runtime. This
    # deliberately excludes campaign YAML and prose so scheduling edits do not
    # force a byte-identical image/SIF rebuild.
    digest="$({
        git ls-files -co --exclude-standard -- .dockerignore docker requirements.txt scripts src \
          | LC_ALL=C sort \
          | while IFS= read -r path; do
                [[ -f "$path" ]] || continue
                printf '%s\0' "$path"
                sha256sum -- "$path"
            done
    } | sha256sum | awk '{print $1}')"
    printf 'runtime.%s\n' "$digest"
    exit 0
fi

if [[ "$mode" == "--campaign" ]]; then
    [[ $# -eq 2 && -f "$2" ]] || { echo 'usage: source_identity.sh --campaign FILE' >&2; exit 2; }
    digest="$(sha256sum -- "$2" | awk '{print $1}')"
    printf 'campaign.%s\n' "$digest"
    exit 0
fi

[[ "$mode" == "--full" ]] || { echo 'usage: source_identity.sh [--full|--runtime|--campaign FILE]' >&2; exit 2; }

commit="$(git rev-parse --verify HEAD)"
if git diff --quiet --ignore-submodules HEAD -- && \
   [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
    printf '%s\n' "$commit"
    exit 0
fi

diff_id="$({ git diff --binary HEAD --; git ls-files --others --exclude-standard -z | sort -z | xargs -0 -r sha256sum; } | sha256sum | awk '{print $1}')"
printf '%s-dirty.%s\n' "$commit" "${diff_id:0:16}"
