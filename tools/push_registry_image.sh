#!/usr/bin/env bash
# Publish one immutable Docker image and verify its remote registry digest.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/push_registry_image.sh [--dry-run] IMAGE:IMMUTABLE_TAG

Try a bounded Docker push first. Authentication failures stop immediately;
transport failures may fall back to a native crane or skopeo using a temporary
Docker archive. The remote digest is always verified before success.
EOF
}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ $# -ne 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  [[ $# -eq 1 ]] && exit 0 || exit 2
fi

IMAGE=$1
DOCKER_PUSH_TIMEOUT_SECONDS=${DOCKER_PUSH_TIMEOUT_SECONDS:-900}
DOCKER_SAVE_TIMEOUT_SECONDS=${DOCKER_SAVE_TIMEOUT_SECONDS:-900}
REGISTRY_OPERATION_TIMEOUT_SECONDS=${REGISTRY_OPERATION_TIMEOUT_SECONDS:-300}
DOCKER_BIN=${EXPERIMENTCTL_DOCKER_BIN:-docker}
CRANE_BIN=${EXPERIMENTCTL_CRANE_BIN:-crane}
SKOPEO_BIN=${EXPERIMENTCTL_SKOPEO_BIN:-skopeo}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AUTH_CHECK="$SCRIPT_DIR/check_docker_registry_auth.py"
WORK_DIR=""

cleanup() {
  if [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]]; then
    rm -rf -- "$WORK_DIR"
  fi
}
trap cleanup EXIT INT TERM HUP

die() {
  printf 'push_registry_image: %s\n' "$*" >&2
  exit 2
}

for value in "$DOCKER_PUSH_TIMEOUT_SECONDS" "$DOCKER_SAVE_TIMEOUT_SECONDS" "$REGISTRY_OPERATION_TIMEOUT_SECONDS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "timeout values must be positive integers"
done
[[ "$IMAGE" != *[[:space:]]* && "$IMAGE" != *://* && "$IMAGE" != *@* ]] \
  || die "IMAGE must be a credential-free tagged registry reference"
IMAGE_NAME=${IMAGE##*/}
[[ "$IMAGE_NAME" == *:* ]] || die "IMAGE must have an explicit immutable tag"
IMAGE_TAG=${IMAGE_NAME##*:}
case "$IMAGE_TAG" in
  latest|runtime|seed) die "mutable tag '$IMAGE_TAG' is not accepted for recorded experiments" ;;
esac

command -v "$DOCKER_BIN" >/dev/null || die "docker is required"
command -v timeout >/dev/null || die "timeout is required"
[[ -f "$AUTH_CHECK" ]] || die "missing Docker credential checker: $AUTH_CHECK"
python3 -c 'import experiment_control.safe_sco' >/dev/null 2>&1 \
  || die "ml-experiment-control package is not installed"
"$DOCKER_BIN" info >/dev/null 2>&1 || die "Docker daemon is unavailable"
"$DOCKER_BIN" image inspect "$IMAGE" >/dev/null 2>&1 || die "local image does not exist: $IMAGE"
REGISTRY_HOST=${IMAGE%%/*}
python3 "$AUTH_CHECK" "$REGISTRY_HOST" >/dev/null \
  || die "Docker credential reference is missing for $REGISTRY_HOST"

PUBLISHER=""
if command -v "$CRANE_BIN" >/dev/null; then
  PUBLISHER=crane
  PUBLISHER_BIN=$CRANE_BIN
elif command -v "$SKOPEO_BIN" >/dev/null; then
  PUBLISHER=skopeo
  PUBLISHER_BIN=$SKOPEO_BIN
fi
[[ -n "$PUBLISHER" ]] || die "native crane or skopeo is required for remote digest verification"

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'dry-run: image=%s publisher=%s push_timeout=%ss registry_timeout=%ss\n' \
    "$IMAGE" "$PUBLISHER" "$DOCKER_PUSH_TIMEOUT_SECONDS" "$REGISTRY_OPERATION_TIMEOUT_SECONDS"
  exit 0
fi

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/elf-registry.XXXXXXXX")
PUSH_LOG="$WORK_DIR/docker-push.log"

set +e
timeout "${DOCKER_PUSH_TIMEOUT_SECONDS}s" "$DOCKER_BIN" push "$IMAGE" >"$PUSH_LOG" 2>&1
PUSH_RC=$?
set -e

classify_push_failure() {
  local rc=$1 log=$2
  if [[ "$rc" -eq 124 ]] || grep -Eiq \
    '(^|[^0-9])502([^0-9]|$)|TLS handshake timeout|unexpected EOF|(^|[^A-Za-z])EOF([^A-Za-z]|$)|connection reset|connection refused|i/o timeout|dial tcp|network is unreachable|temporary failure' \
    "$log"; then
    printf 'transport'
  elif grep -Eiq \
    'unauthorized|authentication required|requested access .* denied|denied:|insufficient[_ -]?scope|forbidden|status code: 401|status code: 403' \
    "$log"; then
    printf 'auth'
  else
    printf 'unknown'
  fi
}

print_redacted_tail() {
  tail -n 80 "$1" | python3 -m experiment_control.safe_sco redact-lines >&2
}

if [[ "$PUSH_RC" -ne 0 ]]; then
  FAILURE_CLASS=$(classify_push_failure "$PUSH_RC" "$PUSH_LOG")
  printf 'docker push failed (class=%s, exit=%s)\n' "$FAILURE_CLASS" "$PUSH_RC" >&2
  print_redacted_tail "$PUSH_LOG"
  if [[ "$FAILURE_CLASS" == auth ]]; then
    printf 'authentication/authorization failures are not eligible for transport fallback\n' >&2
    exit 30
  fi
  if [[ "$FAILURE_CLASS" != transport ]]; then
    printf 'unclassified failures require diagnosis before retrying with another client\n' >&2
    exit 31
  fi

  IMAGE_SIZE=$("$DOCKER_BIN" image inspect --format '{{.Size}}' "$IMAGE" 2>/dev/null || true)
  [[ "$IMAGE_SIZE" =~ ^[0-9]+$ ]] || die "could not determine local image size"
  FREE_KIB=$(df -Pk "$WORK_DIR" | awk 'NR == 2 {print $4}')
  REQUIRED_KIB=$(( (IMAGE_SIZE * 12 / 10 + 536870912 + 1023) / 1024 ))
  if [[ ! "$FREE_KIB" =~ ^[0-9]+$ || "$FREE_KIB" -lt "$REQUIRED_KIB" ]]; then
    die "insufficient temporary disk for fallback archive"
  fi

  ARCHIVE="$WORK_DIR/image.tar"
  timeout "${DOCKER_SAVE_TIMEOUT_SECONDS}s" "$DOCKER_BIN" save -o "$ARCHIVE" "$IMAGE"
  FALLBACK_LOG="$WORK_DIR/fallback.log"
  set +e
  if [[ "$PUBLISHER" == crane ]]; then
    env -u http_proxy -u https_proxy -u all_proxy \
      -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      timeout "${REGISTRY_OPERATION_TIMEOUT_SECONDS}s" "$PUBLISHER_BIN" push "$ARCHIVE" "$IMAGE" \
      >"$FALLBACK_LOG" 2>&1
  else
    env -u http_proxy -u https_proxy -u all_proxy \
      -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
      timeout "${REGISTRY_OPERATION_TIMEOUT_SECONDS}s" \
      "$PUBLISHER_BIN" copy "docker-archive:$ARCHIVE" "docker://$IMAGE" >"$FALLBACK_LOG" 2>&1
  fi
  FALLBACK_RC=$?
  set -e
  if [[ "$FALLBACK_RC" -ne 0 ]]; then
    printf '%s fallback failed (exit=%s)\n' "$PUBLISHER" "$FALLBACK_RC" >&2
    print_redacted_tail "$FALLBACK_LOG"
    exit 32
  fi
fi

set +e
if [[ "$PUBLISHER" == crane ]]; then
  REMOTE_DIGEST=$(env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    timeout "${REGISTRY_OPERATION_TIMEOUT_SECONDS}s" "$PUBLISHER_BIN" digest "$IMAGE" 2>"$WORK_DIR/digest.log")
  DIGEST_RC=$?
else
  REMOTE_DIGEST=$(env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    timeout "${REGISTRY_OPERATION_TIMEOUT_SECONDS}s" \
    "$PUBLISHER_BIN" inspect --format '{{.Digest}}' "docker://$IMAGE" 2>"$WORK_DIR/digest.log")
  DIGEST_RC=$?
fi
set -e
if [[ "$DIGEST_RC" -ne 0 ]]; then
  printf 'remote digest verification failed (exit=%s)\n' "$DIGEST_RC" >&2
  print_redacted_tail "$WORK_DIR/digest.log"
  exit 33
fi
REMOTE_DIGEST=${REMOTE_DIGEST//$'\r'/}
REMOTE_DIGEST=${REMOTE_DIGEST//$'\n'/}
[[ "$REMOTE_DIGEST" =~ ^sha256:[0-9a-fA-F]{64}$ ]] || die "remote verifier returned an invalid digest"
printf '%s@%s\n' "${IMAGE%:*}" "${REMOTE_DIGEST,,}"
