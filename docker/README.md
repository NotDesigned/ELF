# ELF Docker

Two images are used on SenseCore:

- `Dockerfile.seed`: large one-time image that bakes HF cache and checkpoints
  into `/opt/elf/...`, then `scripts/cloud_train.sh` hydrates them to `/data`.
- `Dockerfile`: small runtime image. It never references large build contexts
  and reads cache/checkpoints from `/data`.

Both images install the exact `ml-experiment-control` commit pinned in
`requirements.txt`. The reusable scheduler package is no longer copied from the
ELF source tree. Startup prints the resolved installed package path and fails
before asset planning if the package cannot be imported. Changing the package
pin changes ELF's runtime source identity and requires a new image/SIF.

Set the registry tag:

```bash
export IMAGE=registry.cn-sh-01.sensecore.cn/ccr-zhicheng-02/elf
```

## Seed Once

Prepare a slim HF cache context:

```bash
rm -rf /home/proton/.cache/elf/docker-hf-cache
mkdir -p /home/proton/.cache/elf/docker-hf-cache/hub
mkdir -p /home/proton/.cache/elf/docker-hf-cache/datasets

cp -al ~/.cache/huggingface/hub/models--t5-small \
  /home/proton/.cache/elf/docker-hf-cache/hub/

cp -al ~/.cache/huggingface/datasets/embedded-language-flows___openwebtext-t5 \
  /home/proton/.cache/elf/docker-hf-cache/datasets/
```

Build and push:

```bash
docker build . \
  -f docker/Dockerfile.seed \
  --build-arg SOURCE_ID="$(bash scripts/source_identity.sh --runtime)" \
  --build-context hf_cache=/home/proton/.cache/elf/docker-hf-cache \
  --build-context elf_b_ckpt=/home/proton/.cache/elf/checkpoints/ELF-B-owt-torch \
  -t "$IMAGE:seed"

docker push "$IMAGE:seed"
```

On SenseCore, run the seed image once:

```bash
HYDRATE_ONLY=1 \
bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf.yml
```

Hydration copies to `/data/.cache/huggingface` and `/data/elf/checkpoints`.
The copy intentionally uses `--no-preserve=ownership` because SenseCore `/data`
mounts may reject `chown`.

For `tier0_1_sentence_t5.yml`, add `models--sentence-transformers--sentence-t5-xl`
to the slim context and build seed with `--build-arg PRELOAD_SENTENCE_T5=true`.

## Runtime

After `/data` is hydrated:

```bash
docker build . -f docker/Dockerfile \
  --build-arg SOURCE_ID="$(bash scripts/source_identity.sh --runtime)" \
  -t "$IMAGE:runtime"
docker push "$IMAGE:runtime"
```

For a recorded run, publish a source-qualified immutable tag through the
bounded repository helper and record the digest it prints:

```bash
SOURCE_ID=$(bash scripts/source_identity.sh --runtime)
IMMUTABLE_IMAGE="$IMAGE:runtime-$SOURCE_ID"
docker tag "$IMAGE:runtime" "$IMMUTABLE_IMAGE"
scripts/push_registry_image.sh --dry-run "$IMMUTABLE_IMAGE"
scripts/push_registry_image.sh "$IMMUTABLE_IMAGE"
```

The helper tries Docker first, stops on authentication/authorization errors,
and only uses a temporary `docker save` plus native `crane`/`skopeo` fallback
for classified transport failures such as TLS EOF or HTTP 502. It removes the
archive on every exit and succeeds only after verifying the remote digest.

The dry run is a local registry preflight: it requires a reachable Docker
daemon, the exact local immutable image, a `crane` or `skopeo` verifier, and a
credential reference for the target registry in Docker's configured store or
helper. It never prints or copies the credential. A real push remains the only
authoritative test of push authorization.

The short `seed` and `runtime` tags above are convenient staging aliases. A
recorded research run must use an immutable source-qualified tag or image
digest; the experiment manifest workflow records that identity separately.

Run examples:

```bash
RUN_ID=tier0-pure-elf-s0-$(date -u +%Y%m%dT%H%M%SZ) \
IMAGE_ID="$IMAGE@sha256:<registry-digest>" \
NGPU=8 \
bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf.yml
```

```bash
USE_ELF_B_WARM_START=1 \
NGPU=8 \
bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_2_learned_main.yml
```

Use `DRY_RUN=1` to print the final launch command without starting training.
Use `PREPARE_ONLY=1` to create and validate the durable experiment records but
stop before the training process; unlike dry-run, this intentionally writes to
the selected run directory.

## Experiment identity

Every real launch creates a unique `/data/elf/runs/<run-id>` directory. When
`RUN_ID` is omitted, the launcher generates one from the config path, UTC time,
and random bytes. A recorded SenseCore run requires immutable `SOURCE_ID` and
`IMAGE_ID` values; `SOURCE_ID` is normally baked into the image, while
`IMAGE_ID` should be the registry digest or a source-qualified immutable tag.

Before training starts, the launcher atomically writes:

- `manifest.yaml`: immutable scientific config and source/image identities.
- `attempts/<attempt-id>/attempt.yaml`: command, backend, resources, and resume source.
- `events.jsonl`, `backend.json`, and `status.json`: durable Agent-facing control records.
- `attempts/<attempt-id>/stdout.log` and `stderr.log`: persistent process logs.

The launcher transitions `status.json` through `CREATED`, `RUNNING`, and then
`SUCCEEDED` or `FAILED`, appending the matching lifecycle events and exit code.
Scheduler-side spot eviction still has to be normalized to `PREEMPTED` by the
external SenseCore observer when the container cannot execute its exit path.

Spot recovery keeps the same `RUN_ID`, uses a new `ATTEMPT_ID`, and explicitly
sets `RESUME` to the run directory or a completed checkpoint. Reusing an
existing attempt ID fails instead of overwriting metadata:

```bash
RUN_ID=<original-run-id> ATTEMPT_ID=attempt-002 \
RESUME=/data/elf/runs/<original-run-id> \
IMAGE_ID="$IMAGE@sha256:<same-registry-digest>" \
bash scripts/cloud_train.sh <config>
```

`Wandb` uses the scientific `RUN_ID` by default, so a new generated run creates
a fresh tracker run and later attempts resume the same tracker identity.
`REQUIRE_IMMUTABLE_IDENTITIES=0` is reserved for local smoke tests; do not use
it for recorded research runs.

## Notes

- Default project storage is `/data/elf/...`.
- Shared HF cache is `/data/.cache/huggingface`.
- Default endpoint is `HF_ENDPOINT=https://hf-mirror.com`.
- Runtime defaults to HF offline mode and fails fast if cache is missing.
- If a previous hydrate was interrupted, remove stale
  `/data/.cache/huggingface/.baked-cache-markers/*.lock` before retrying.
