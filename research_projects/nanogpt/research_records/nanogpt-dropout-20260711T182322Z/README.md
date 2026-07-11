# Dropout short-budget record

This is the small, version-controlled evidence subset collected from the
canonical Wyd run directory:

```text
/data/liangluocheng/nanogpt/runs/nanogpt-dropout-20260711T182322Z
```

The hypothesis and decision were frozen before submission. Jobs `1789`
(control, dropout 0.0) and `1790` (treatment, dropout 0.2) each used one L40S,
seed 1337, the same 0.80M-parameter architecture, and 300 optimizer steps.
Both completed with exit `0:0`. The matched step-300 validation losses were
2.2797 and 2.4025, respectively, for a treatment-minus-control delta of
`+0.1228`; this exceeds the preregistered `+0.020` support threshold.

Checkpoint payloads remain only on `/data`. The included marker and evidence
files record their sizes, steps, readability checks, and SHA-256 digests. This
single-seed, short-budget result does not establish a general long-training
effect.
