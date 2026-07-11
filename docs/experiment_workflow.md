# ELF Experiment Workflow

This document is the operational contract for preparing, observing, and
comparing ELF experiments. Architecture motivation lives in
`fusion_architecture.md`; configuration fields live in `config_reference.md`.

## Canonical run layout

Every scientific run has a unique `run_id` and persistent directory:

```text
manifest.yaml                 resolved scientific identity
backend.json                  current scheduler backend/job identity
status.json                   normalized current state
events.jsonl                  append-only lifecycle events
train_metrics.jsonl           rank-zero structured training metrics
attempts/attempt-NNN/         immutable attempt metadata and process logs
checkpoint_<step>             checkpoint payload
checkpoint_<step>.complete    checkpoint completion marker
*/metrics.jsonl               generation/reconstruction evaluation metrics
```

Do not infer scientific success from scheduler success. A completed run must
also contain its required evaluation records and readable artifacts.

## Manifest helper: `scripts/experiment_manifest.py`

The helper has two commands. The default command prepares a run/attempt; the
`record` command appends a lifecycle transition.

### Function contract

| Function | Input | Output / side effect | Failure meaning |
| --- | --- | --- | --- |
| `resolved_config` | YAML path and typed overrides | Fully inherited `Config` mapping | Invalid config or override |
| `scientific_config` | Resolved mapping | Mapping without attempt-only fields | Pure transformation |
| `sanitize_command` | Argument vector | Secret-redacted argument vector | Pure transformation |
| `atomic_write` | Path and JSON/YAML payload | fsync + atomic replacement | Durable state was not committed |
| `append_event` | Event path and mapping | One fsynced JSONL event | Lifecycle record was not committed |
| `prepare` | Parsed prepare arguments | Run manifest, attempt manifest, initial state/event | Identity conflict, reused attempt, or mutable source/image identity |
| `record` | Parsed transition arguments | New `status.json` and appended event | Transition does not belong to a prepared run/attempt |

Detailed argument, return, and exception semantics are also kept beside each
Python function as docstrings, so `help()` and IDE hover remain authoritative.

Prepare example:

```bash
python scripts/experiment_manifest.py \
  --project elf \
  --run-id fusion-l256-learned-none-aux1-s42-SOURCE \
  --attempt-id attempt-001 \
  --backend slurm \
  --config src/configs/training_configs/ablations/owt_elfb/tier0_2_learned_main_len256.yml \
  --output-dir /data/liangluocheng/elf/runs/RUN_ID \
  --source-id SOURCE_ID \
  --image-id SIF_SHA256 \
  --gpus 4 --nodes 1 --quota normal \
  --require-immutable-identities \
  -- bash scripts/launch.sh train CONFIG
```

## Campaign summary: `scripts/summarize_experiments.py`

The summary command is read-only. It discovers every `manifest.yaml` under the
given roots, combines run identity/status with the latest training and
evaluation metrics, and emits JSON or CSV.

```bash
python scripts/summarize_experiments.py /data/elf/runs --format json
python scripts/summarize_experiments.py /data/liangluocheng/elf/runs \
  --format csv --output fusion-len256-v1.csv
```

### Function contract

| Function | Responsibility |
| --- | --- |
| `load_mapping` | Strictly load one JSON/YAML object. |
| `read_jsonl` | Strictly load JSON objects and report exact corrupt line. |
| `discover_run_dirs` | Resolve, deduplicate, and sort run directories. |
| `latest_record` | Select greatest-step training record, breaking ties by file order. |
| `collect_eval_metrics` | Search nested evaluation outputs and surface same-step conflicts. |
| `summarize_run` | Produce one flat run row and compute `plan_ppl_gap`. |
| `write_json` / `write_csv` | Serialize deterministic campaign output. |

`plan_ppl_gap = shuffled_plan_ppl - oracle_plan_ppl`; positive is better because
the correct plan produced lower perplexity than a mismatched plan.

## Cross-platform controller: `scripts/experimentctl.py`

The controller freezes campaign metadata locally before scheduler submission
and exposes the same operations for SenseCore and WYD Slurm:

```bash
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml prepare
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml stage
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml render
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml submit --run RUN_ID
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml status --run RUN_ID
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml collect --run RUN_ID
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml cancel --run RUN_ID
```

`--run` is repeatable; omitting it selects every campaign run. `--dry-run` on
`submit` renders and validates without mutating a scheduler.

### Controller function contract

| Function | Responsibility |
| --- | --- |
| `load_campaign` / `validate_run` | Validate schema, backend fields, safe environment allowlist, explicit Slurm GRES, and SenseCore spot policy. |
| `source_identity` / `materialize_run` | Compute commit+diff identity and expand immutable path placeholders. |
| `prepare_run` | Resolve config/overrides and atomically freeze a local control manifest before submission. |
| `stage_slurm` | Rsync one immutable source path and verify the configured digest-addressed SIF. |
| `render_slurm_script` | Render explicit partition/account/QOS/GRES/time and Apptainer bindings. |
| `submit_slurm` | Copy the frozen sbatch file, call `sbatch --parsable`, and return the numeric job ID. |
| `submit_sensecore` | Check exact-name uniqueness, submit one spot job, and verify it through sanitized describe. |
| `status_slurm` / `status_sensecore` | Normalize scheduler state without claiming model progress. |
| `collect_slurm` | Execute the canonical run summarizer on shared `/data`. |
| `collect_sensecore_logs` | Fetch a bounded, redacted metric-bearing log snapshot. |
| `cancel_slurm` / `cancel_sensecore` | Cancel only the recorded nonterminal scheduler identity, then re-observe it. |

Every Python function also carries its detailed input/output/failure semantics
as an IDE-visible docstring.

### Campaign schema

The Slurm SSH alias is only a login entry point. Compute placement is explicit
per run and may target any current WYD partition:

```yaml
schema_version: 1
campaign: fusion-smoke
project: elf
source_id: auto
local_root: outputs/experiment_campaigns

runs:
  - run_id: smoke-slurm-h100
    config: src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf_len256.yml
    config_overrides: [epochs=1]
    image_id: sha256:<SIF_SHA256>
    resources: {gpus: 1, cpus: 8}
    env: {BATCH_SIZE: "4", LOG_FREQ: "10"}
    storage:
      run_dir: /data/liangluocheng/elf/runs/smoke-slurm-h100
    backend:
      kind: slurm
      ssh_alias: wyd-l40s
      partition: h100
      account: lab
      qos: normal
      gres: gpu:h100:1
      time: "00:10:00"
      source_dir: /data/liangluocheng/elf/sources/{source_id}
      sif_path: /data/liangluocheng/elf/images/<SIF_SHA256>.sif
      data_root: /data/liangluocheng
      project_data_root: /data/liangluocheng/elf
      hf_home: /data/liangluocheng/elf/cache/huggingface
      hf_datasets_cache: /data/liangluocheng/elf/cache/huggingface/datasets
```

Valid current partition/GRES pairs must still be rediscovered before submit:
`l40s/gpu:l40s:N`, `h100/gpu:h100:N`, `h200/gpu:h200:N`, and
`rtx5880/gpu:rtx5880:N`. The controller never derives partition from
`ssh_alias` and never silently changes it after a failure.

SenseCore runs instead declare workspace, AEC2, exact job/display name,
immutable image tag+digest, worker spec, spot quota, and AFS mount. Campaign
environment fields use a strict non-secret allowlist; credentials must not be
placed in YAML or startup commands.

### State ownership

The controller owns local pre-submit metadata and scheduler observations. The
container launcher owns process lifecycle and model artifacts under shared
storage. A scheduler `RUNNING` state and a first `train_metrics.jsonl` record
remain separate gates. SenseCore abrupt eviction may prevent the process from
updating its own status, so external scheduler observation takes precedence
for `PREEMPTED` classification.

### WYD site-specific execution notes

- `ssh_alias` is a login route only; `partition` and typed GPU `gres` remain
  explicit and independently validated before every submission.
- Set Slurm storage roots to `/data/liangluocheng/elf`, including HF cache,
  checkpoints, W&B, and saved-model directories. The container image's
  SenseCore-oriented `/data/elf` defaults are not writable on WYD.
- On the currently observed Slurm installation, batch jobs failed before the
  shell when `#SBATCH --output/--error` named normal shared files. Generated
  scripts therefore point Slurm's pre-shell streams at `/dev/null`, then the
  bash process immediately tees stdout/stderr into the attempt directory under
  `/data`. This preserves early application errors without relying on slurmd
  to create the shared log file.
- Collection rsyncs only manifests, status, training JSONL, and nested
  evaluation `metrics.jsonl` to the controller, then summarizes locally.
  Checkpoints and generated sample payloads remain on shared storage.

## Offline preflight

When offline mode or `REQUIRE_OFFLINE_CACHE=1` is active,
`scripts/cloud_train.sh` resolves the inherited config before launching GPUs and
requires:

- the token encoder/tokenizer model;
- the dataset cache;
- Sentence-T5 for a frozen sentence-plan run;
- the PPL evaluator model when `online_eval=true`;
- requested warm-start checkpoints.

This check is deliberately config-aware. Learned-plan and pure-ELF runs do not
require Sentence-T5, while frozen-plan runs fail before scheduler resources are
wasted if it is absent.

## First length-256 campaign

All comparisons below use seed 42, global batch 512, and train from scratch
unless the campaign declares a separate warm-start question.

### Four-run decision set

| Run | Config | Question answered |
| --- | --- | --- |
| A0 | `tier0_0_pure_elf_len256.yml` | Does the unchanged ELF control train normally? |
| A1 | `tier0_1_sentence_t5_len256.yml` | Can this architecture use a meaningful frozen plan? |
| A2 | `tier0_2_learned_main_len256.yml` | Can the plan be learned without collapse? |
| A3 | `tier3_aux0_len256.yml` | Is the detached auxiliary pass necessary to train/refine the learned plan? |

A0–A2 identify whether failure comes from the fusion mechanism or the learned
encoder. A3 is the cheapest causal test of the main new mechanism because it
changes only `plan_aux_passes: 1 -> 0` relative to A2.

If only three runs fit, run A0–A2. Do not substitute a gradient-topology run:
without a frozen teacher anchor, a failed learned run cannot distinguish “plan
fusion is useless” from “the learned encoder collapsed.”

### One-epoch gate

Continue a run beyond one epoch only if:

1. process/checkpoint/evaluation artifacts are complete and finite;
2. learned-plan embedding norm and variance do not show collapse;
3. `oracle_plan_ppl < shuffled_plan_ppl` for plan-enabled models;
4. generation is non-empty and entropy is not degenerate;
5. token reconstruction does not catastrophically regress from A0.

The first gate is diagnostic, not a final scientific claim. Final comparisons
must use matched training budgets and additional seeds.

## Backend blocking

- Keep A0–A2 on one backend whenever possible.
- A3 must share a backend with A2 because their difference is the scientific
  treatment.
- Use a separately named platform-control replica before comparing treatments
  that were split across SenseCore and Slurm.
- Compare quality within seed/backend blocks. Compare throughput only on the
  same GPU type/count and checkpoint/evaluation policy.

SenseCore spot attempts use `/data/elf/runs`; Slurm attempts use
`/data/liangluocheng/elf/runs`. A preemption creates a new `attempt_id` but
keeps the same scientific `run_id` and resumes only a completed checkpoint.
