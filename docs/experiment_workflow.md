# ELF Experiment Workflow

This document is the operational contract for preparing, observing, and
comparing ELF experiments. Architecture motivation lives in
[`fusion_architecture.md`](fusion_architecture.md); configuration fields live
in [`config_reference.md`](config_reference.md). An autonomous operator should
start with [`agent_research_guide.md`](agent_research_guide.md), which turns
these contracts into a guarded research loop and report format.

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

## Manifest and state helper: `scripts/experiment_manifest.py`

The helper has two commands. The default command prepares a run/attempt; the
`record` command appends a lifecycle transition.

### Function contract

| Function | Input | Output / side effect | Failure meaning |
| --- | --- | --- | --- |
| `sanitize_command` | Argument vector | Secret-redacted argument vector | Pure transformation |
| `atomic_write` / `atomic_create` | Path and JSON/YAML payload | fsync + atomic replacement, or exclusive creation | Durable state was not committed or immutable state already exists |
| `append_event` | Event path and mapping | One fsynced JSONL event | Lifecycle record was not committed |
| `ExperimentStateStore` | One local run directory | Run/attempt creation, status transitions, submission intent, and reconciliation | Identity conflict, invalid transition, or duplicate scheduler identity |
| `prepare` | Parsed prepare arguments | Canonical run manifest, attempt manifest, initial state/event | Identity conflict, reused attempt, or mutable source/image identity |
| `record` | Parsed transition arguments | New `status.json` and appended event | Transition does not belong to a prepared run/attempt |

ELF config resolution and scientific-field selection belong to
`scripts/experiment_projects/elf.py`, not this backend-neutral state helper.
Controller and runtime construct the shared manifest schema through
`experiment_run_manifest.build_run_manifest`.

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
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml preflight --scope submit
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml stage
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml render
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml submit --run RUN_ID
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml status --run RUN_ID
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml collect --run RUN_ID
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml observe --run RUN_ID
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml logs --run RUN_ID --tail 100
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml decide --run RUN_ID
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml assets-plan --run RUN_ID
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml assets-verify --run RUN_ID
python scripts/experimentctl.py experiments/campaigns/CAMPAIGN.yml cancel --run RUN_ID
```

`--run` is repeatable; omitting it selects every campaign run. `--dry-run` on
`submit` renders and validates without mutating a scheduler.

`preflight` performs sanitized, read-only checks against the selected compute
backend. `stage` and non-dry-run `submit` require the corresponding preflight
to pass before remote mutation. Local-only `prepare`, `render`, and
`assets-plan` do not require an active platform login.

The controller accepts only non-secret tool overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `EXPERIMENTCTL_SCO_BIN` | `sco` | SenseCore CLI executable. |
| `EXPERIMENTCTL_SSH_BIN` | `ssh` | WYD SSH executable. |
| `EXPERIMENTCTL_RSYNC_BIN` | `rsync` | WYD transfer executable. |
| `EXPERIMENTCTL_DOCKER_BIN` | `docker` | Local registry publisher engine. |
| `EXPERIMENTCTL_CRANE_BIN` | `crane` | Preferred registry client. |
| `EXPERIMENTCTL_SKOPEO_BIN` | `skopeo` | Registry client fallback. |

These select tools, not credentials. SCO uses its own profile, Docker uses its
credential store/helper, and WYD uses SSH config or the agent. Credential
material must never be copied into campaign environment fields.

### Controller function contract

| Function | Responsibility |
| --- | --- |
| `load_campaign` / `validate_run` | Validate schema, backend fields, safe environment allowlist, explicit Slurm GRES, and SenseCore spot policy. |
| `source_identity` / `materialize_run` | Compute runtime-tree identity and expand immutable path placeholders. |
| `prepare_run` | Resolve config/overrides and atomically freeze the canonical run manifest and local attempt before submission. |
| `ProjectAdapter` | Own project config semantics, launcher command, runtime environment, assets, metrics, summary, and source-bundle policy. |
| `WydSlurmBackend` | Stage immutable source/SIF artifacts and implement Slurm render, submit, status, collect, and cancel. |
| `SenseCoreBackend` | Implement exact-name SCO submit/status/log collection/cancel through immediate sanitization. |
| `BackendRegistry` | Dispatch the common backend contract without platform branches in the CLI. |

Every Python function also carries its detailed input/output/failure semantics
as an IDE-visible docstring.

### Controller module boundaries

`experimentctl.py` is the thin CLI and backend-neutral orchestration loop.
Platform and pure-policy code are separated so the core can be exercised
without network access:

| Module | Responsibility |
| --- | --- |
| `experiment_campaign.py` | Resolve defaults, profiles, matrices, and authored runs. |
| `experiment_manifest.py` | Atomic run/attempt store, lifecycle states, submission outbox, reconciliation. |
| `experiment_assets.py` | ELF asset discovery and Hugging Face cache layout used by the ELF project adapter. |
| `experiment_overrides.py` | ELF's ordered environment-to-config override sequence. |
| `experiment_policy.py` | Classify failures and recommend bounded next actions without mutating a scheduler. |
| `packages/experiment-control/.../runner.py` | Injectable command boundary for production subprocesses and hermetic fakes. |
| `packages/experiment-control/.../preflight.py` | Sanitized backend readiness checks and fail-closed reports. |
| `packages/experiment-control` | Independently installable backend, preflight, runner, state, sanitizer, and project-protocol package. |
| `scripts/experiment_projects/elf.py` | The only adapter that knows ELF Config, `cloud_train.sh`, ELF checkpoints, metrics, and summaries. |
| `packages/experiment-control/.../backends/wyd.py` | SSH, rsync, Slurm, Apptainer staging/status/collection/cancellation. |
| `packages/experiment-control/.../backends/sensecore.py` | SCO submission/status/logging/cancellation through packaged sanitization. |

The backend registry dispatches validation, platform environment resolution,
asset verification, submission recovery, `stage`, `render`, `submit`,
`status`, `collect`, `logs`, and `cancel`. Adding a backend does not add a new
backend-kind branch or scheduler field to the CLI loop. Shared storage and
launcher paths live in the backend profile's `storage` mapping; scheduler-only
fields remain inside `backend`.

The controller composes two independent adapters. A project adapter answers
*what* is executed and interpreted; a backend answers *where and how* it is
staged, scheduled, observed, and cancelled. Backends receive filesystem asset
probes instead of model/dataset names, and receive a `SourceBundle` instead of
assuming a repository root, rsync exclusion list, container mount, or working
directory. `tests/test_project_adapter_contract.py` prepares and renders a
dummy project with no import from ELF's `src/configs`, guarding this boundary.

### Submission recovery

Before calling `sbatch` or `sco ... create`, the controller atomically writes a
`SUBMITTING` record under the attempt. Scheduler acceptance is then reconciled
to one job ID. Repeating reconciliation with the same ID repairs derived
`backend.json`, `status.json`, and lifecycle events; a different job ID is
rejected.

SenseCore recovery queries the exact unique resource name. Slurm scripts carry
`campaign/run/attempt` in `#SBATCH --comment`, allowing a controller that
crashed after `sbatch` acceptance to recover the job from the queue. If an
intent remains unresolved, the controller refuses to create a second job.

### Identity separation

The ELF project adapter selects `source_identity.sh --runtime`, which hashes
only Docker/training runtime inputs.
Campaign YAML and documentation changes therefore do not force an identical
image or SIF rebuild. Manifests record four separate provenance values:

- full Git commit;
- runtime tree identity;
- campaign-file identity;
- registry image digest or SIF SHA-256.

The default `source_identity.sh --full` remains available for complete dirty
working-tree provenance.

### Offline assets and decisions

`assets-plan` shows config-dependent requirements without accessing a backend.
For Slurm, `assets-verify` checks the resolved cache paths through the declared
SSH alias on the remote persistent filesystem. SenseCore reports
`requires-running-sensecore-worker` until verification can run inside a mounted
worker; it never treats the controller's local `/data` as SenseCore AFS. Asset
transfer/hydration is a separate future transport interface; verification
never downloads implicitly inside a GPU job.

`observe` records scheduler evidence and collects process/model evidence as
separate objects. `decide` writes `decision.json`. It permits a retry only for
classified transport, scheduler/node, or preemption failures within the
declared `max_infra_retries`; OOM, timeout, configuration, model, and
evaluation failures are never silently retried or resource-adjusted.

`logs` returns a bounded, redacted snapshot. Slurm reads attempt-local
`stdout.log` and `stderr.log` from persistent storage; carriage-return progress
bars are normalized before the final line bound is applied. SenseCore uses a
20-second live stream and reports `expired: true` when the terminal job's log
token has expired instead of retrying indefinitely.

Collection records both `scheduler_state` and `runtime_state`. They may differ
after preemption or cancellation because a killed process cannot always update
its own status file; scheduler truth must not overwrite the historical runtime
observation or vice versa.

### Registry publication

SenseCore output sanitization lives in the installed `experiment_control`
package, so the repository does not depend on a particular user's `~/.codex`
directory. Publish immutable
images with `scripts/push_registry_image.sh`; it distinguishes authorization
from transport failures, bounds every operation, uses archive plus native
crane/skopeo only as a transport fallback, verifies the remote digest, and
removes temporary archives on exit.

### Campaign schema

Campaign authoring supports recursive `defaults`, named `profiles`, and
cartesian `matrix` entries. Resolution order is defaults, selected profiles in
the listed order, then the individual run. Mapping fields merge recursively;
lists such as `config_overrides` are replaced as a whole. The controller
validates only the fully expanded runs, and freezes that expanded form in the
canonical manifest.

Known matrix tokens use `{axis}` or `{axis.field}`. Controller-time tokens such
as `{source_id}`, `{run_id}`, `{project}`, and `{campaign}` remain untouched
until the immutable source identity and run are materialized. For example:

```yaml
schema_version: 1
campaign: fusion-smoke
project: elf
source_id: auto
local_root: outputs/experiment_campaigns

defaults:
  image_id: sha256:<SIF_SHA256>
  resources: {gpus: 1, cpus: 8}
  config_overrides: [epochs=1]
  storage:
    run_dir: /datapool/liangluocheng/elf/runs/{run_id}
    data_root: /datapool/liangluocheng
    project_data_root: /datapool/liangluocheng/elf
    hf_home: /datapool/liangluocheng/.cache/huggingface
    hf_datasets_cache: /datapool/liangluocheng/.cache/huggingface/datasets

profiles:
  wyd-h100:
    backend:
      kind: slurm
      ssh_alias: wyd-l40s
      partition: h100
      account: lab
      qos: normal
      gres: gpu:h100:1
      time: "00:10:00"
      mount_root: /datapool
      source_dir: /datapool/liangluocheng/elf/sources/{source_id}
      sif_path: /datapool/liangluocheng/elf/images/<SIF_SHA256>.sif

runs:
  - matrix:
      variant:
        - {name: a0, config: configs/a0.yml}
        - {name: a1, config: configs/a1.yml}
      seed: [42, 43]
    template:
      profile: wyd-h100
      run_id: "fusion-{variant.name}-s{seed}"
      config: "{variant.config}"
```

`profile` may also be an ordered list, for example
`[wyd-common, l40s]`, so shared login/storage settings and accelerator
placement remain independently reusable. Lists are intentionally not appended:
a run-level override list is the complete ordered override set for that run.

The Slurm SSH alias is only a login entry point. Compute placement is explicit
in the resolved backend and may target any current WYD partition.

For nodes whose persistent filesystem is mounted elsewhere, declare
`mount_root`, `apptainer_cache_dir`, and `apptainer_tmp_dir` explicitly. For
example, WYD H100 uses `/datapool`; the renderer binds that same absolute path
into the container instead of assuming `/data`.

Valid current partition/GRES pairs must still be rediscovered before submit:
`l40s/gpu:l40s:N`, `h100/gpu:h100:N`, `h200/gpu:h200:N`, and
`rtx5880/gpu:rtx5880:N`. The controller never derives partition from
`ssh_alias` and never silently changes it after a failure.

SenseCore runs instead declare workspace, AEC2, exact job/display name,
immutable image tag+digest, worker spec, spot quota, and AFS mount. Campaign
environment fields use a strict non-secret allowlist; credentials must not be
placed in YAML or startup commands.

SenseCore preflight checks the SCO executable and an exact-name workspace query
through `safe_sco.py`; malformed/non-JSON responses fail closed. WYD observe
preflight checks only SSH and Slurm control access; stage adds rsync/storage;
submit adds live partition/GRES, account/QOS, Apptainer, and mount checks.
Reports contain fixed messages rather than raw platform responses.

### State ownership

The controller owns local pre-submit metadata and scheduler observations. The
container launcher owns process lifecycle and model artifacts under shared
storage. A scheduler `RUNNING` state and a first `train_metrics.jsonl` record
remain separate gates. SenseCore abrupt eviction may prevent the process from
updating its own status, so external scheduler observation takes precedence
for `PREEMPTED` classification.

Controller and runtime use `experiment_run_manifest.build_run_manifest` for
the same canonical `manifest.yaml` schema. Slurm stages that manifest before
the job script; the runtime validates it before creating attempt/process
records. Checkpoint collection accepts only `checkpoint_<step>` payloads whose
`.complete` JSON marker has the same step and byte count. WYD probes those
markers remotely without copying checkpoint payloads; SenseCore extracts the
same committed path from sanitized launcher logs while they remain available.

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
