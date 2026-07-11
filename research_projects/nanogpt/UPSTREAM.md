# Upstream provenance

- Repository: https://github.com/karpathy/nanoGPT
- Commit: `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`
- Retrieved: 2026-07-12
- License: MIT; see [`LICENSE`](LICENSE).

The source is vendored so cluster runs do not depend on outbound network
access. The only local source change is the addition of output-directory
ignore rules. Generated Shakespeare inputs (`input.txt`, `*.bin`, `*.pkl`) are
kept on backend storage and are not committed.
