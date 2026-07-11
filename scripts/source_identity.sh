#!/usr/bin/env bash
set -euo pipefail

commit="$(git rev-parse --verify HEAD)"
if git diff --quiet --ignore-submodules HEAD -- && \
   [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
    printf '%s\n' "$commit"
    exit 0
fi

diff_id="$({ git diff --binary HEAD --; git ls-files --others --exclude-standard -z | sort -z | xargs -0 -r sha256sum; } | sha256sum | awk '{print $1}')"
printf '%s-dirty.%s\n' "$commit" "${diff_id:0:16}"
