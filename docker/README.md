# ELF Docker

Two images are used on SenseCore:

- `Dockerfile.seed`: large one-time image that bakes HF cache and checkpoints
  into `/opt/elf/...`, then `scripts/cloud_train.sh` hydrates them to `/data`.
- `Dockerfile`: small runtime image. It never references large build contexts
  and reads cache/checkpoints from `/data`.

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
docker build . -f docker/Dockerfile -t "$IMAGE:runtime"
docker push "$IMAGE:runtime"
```

The short `seed` and `runtime` tags above are convenient staging aliases. A
recorded research run must use an immutable source-qualified tag or image
digest; the experiment manifest workflow records that identity separately.

Run examples:

```bash
NGPU=8 \
bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf.yml
```

```bash
USE_ELF_B_WARM_START=1 \
NGPU=8 \
bash scripts/cloud_train.sh src/configs/training_configs/ablations/owt_elfb/tier0_2_learned_main.yml
```

Use `DRY_RUN=1` to print the final launch command without starting training.

## Notes

- Default project storage is `/data/elf/...`.
- Shared HF cache is `/data/.cache/huggingface`.
- Default endpoint is `HF_ENDPOINT=https://hf-mirror.com`.
- Runtime defaults to HF offline mode and fails fast if cache is missing.
- If a previous hydrate was interrupted, remove stale
  `/data/.cache/huggingface/.baked-cache-markers/*.lock` before retrying.
