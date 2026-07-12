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
manifest.yaml                 immutable Run identity v2
backend.json                  current scheduler backend/job identity
status.json                   normalized current state
events.jsonl                  append-only lifecycle events
train_metrics.jsonl           rank-zero structured training metrics
attempts/attempt-NNN/         canonical attempt state/decision and process logs
checkpoint_<step>             checkpoint payload
checkpoint_<step>.complete    checkpoint completion marker
*/metrics.jsonl               generation/reconstruction evaluation metrics
```

Do not infer scientific success from scheduler success. A completed run must
also contain its required evaluation records and readable artifacts.
The root `backend.json`, `status.json`, `collection.json`, and `decision.json`
are read-model mirrors of the current attempt. Attempt-local files are the
canonical history; observing an older attempt cannot move the root mirror.

## Manifest and state helper: `elf_experiments.manifest`

The helper has two commands. The default command prepares a run/attempt; the
`record` command appends a lifecycle transition.

ELF config resolution and scientific-field selection belong to
`src/elf_experiments/projects/elf.py`, not this backend-neutral state helper.
Controller and runtime construct the shared manifest schema through
`elf_experiments.run_manifest.build_run_manifest`.

New Run manifests set `identity_version: 2` and freeze resolved scientific
config, source/runtime/campaign/image identities, backend, resources, full
storage mapping, redacted command template, execution mount/workdir, declared
assets, checkpoint policy, and any evaluation-as-run identity. A new Attempt
may change only attempt/job timestamps and a same-Run completed resume
checkpoint. Changing GPU count/type, backend, walltime, command, storage, or
assets requires a new `run_id`; older v1 manifests are not eligible for
automatic retry.

Argument, return, and failure semantics live beside each Python function as
tested docstrings; this document records only cross-module contracts.

Prepare example:

```bash
python tools/experiment_manifest.py \
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

## Campaign summary: `tools/summarize_experiments.py`

The summary command is read-only. It discovers every `manifest.yaml` under the
given roots, combines run identity/status with the latest training and
evaluation metrics, and emits JSON or CSV.

```bash
python tools/summarize_experiments.py /data/elf/runs --format json
python tools/summarize_experiments.py /data/liangluocheng/elf/runs \
  --format csv --output fusion-len256-v1.csv
```

`plan_ppl_gap = shuffled_plan_ppl - oracle_plan_ppl`; positive is better because
the correct plan produced lower perplexity than a mismatched plan.

## Cross-platform controller: `tools/experimentctl.py`

The controller freezes campaign metadata locally before scheduler submission
and exposes the same operations for SenseCore and WYD Slurm:

```bash
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml check-identity --run RUN_ID
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml assets-plan
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml prepare
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml preflight --scope submit
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml submit --run RUN_ID --dry-run
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml assets-verify --run RUN_ID
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml stage
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml submit --run RUN_ID
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml status --run RUN_ID
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml collect --run RUN_ID
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml observe --run RUN_ID
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml watch --run RUN_ID --until terminal
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml logs --run RUN_ID --tail 100
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml decide --run RUN_ID
python tools/experimentctl.py experiments/campaigns/CAMPAIGN.yml cancel --run RUN_ID
```

`--run` is repeatable; omitting it selects every campaign run. `--dry-run` on
`submit` renders and validates without mutating a scheduler.

`watch` emits one compact JSON object per line. It periodically performs the
same durable four-layer refresh as `observe`; terminal runs are collected and
passed through `decide` automatically. `--until first-metric` is the initial
health gate, while the default `--until terminal` waits for completion.
`--interval-seconds` controls polling and `--timeout-seconds 0` means no
deadline. Watch never cancels or retries a job: `STOP_RECOMMENDED` and retry
recommendations remain explicit decisions for a separate authorized action.

`preflight` performs sanitized, read-only checks against the selected compute
backend. `stage` and non-dry-run `submit` require the corresponding preflight
to pass before remote mutation. Local-only `prepare`, `assets-plan`, and
dry-run submission do not require an active platform login.

`check-identity` combines local durable history with the backend package's
read-only scheduler/storage probe. It rejects a consumed identity and reports
all matching job IDs when history is ambiguous. Non-dry-run submission repeats
this gate immediately before recording its submission intent.

The controller accepts only non-secret tool overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `EXPERIMENTCTL_SCO_BIN` | `sco` | SenseCore CLI executable. |
| `EXPERIMENTCTL_SCO_CREATE_TIMEOUT_SECONDS` | `120` | Bound SCO create before exact-name reconciliation. |
| `EXPERIMENTCTL_SSH_BIN` | `ssh` | WYD SSH executable. |
| `EXPERIMENTCTL_RSYNC_BIN` | `rsync` | WYD transfer executable. |
| `EXPERIMENTCTL_DOCKER_BIN` | `docker` | Local registry publisher engine. |
| `EXPERIMENTCTL_CRANE_BIN` | `crane` | Preferred registry client. |
| `EXPERIMENTCTL_SKOPEO_BIN` | `skopeo` | Registry client fallback. |

These select tools, not credentials. SCO uses its own profile, Docker uses its
credential store/helper, and WYD uses SSH config or the agent. Credential
material must never be copied into campaign environment fields.

### Controller module boundaries

`experimentctl.py` is the thin CLI and backend-neutral orchestration loop.
Platform and pure-policy code are separated so the core can be exercised
without network access:

| Module | Responsibility |
| --- | --- |
| `src/elf_experiments/campaign.py` | Resolve defaults, profiles, matrices, and authored runs. |
| `tools/instantiate_campaign.py` | Render one explicit fresh ELF campaign instance without overwriting history. |
| `src/elf_experiments/manifest.py` and `tools/experiment_manifest.py` | Atomic run/attempt store and its thin state-helper CLI. |
| `src/elf_experiments/run_manifest.py` | Construct the canonical manifest schema shared by controller and runtime. |
| `src/elf_experiments/assets.py` | ELF asset discovery and Hugging Face cache layout used by the ELF project adapter. |
| `src/elf_experiments/overrides.py` | ELF's ordered environment-to-config override sequence. |
| `src/elf_experiments/policy.py` | Classify failures and recommend bounded next actions without mutating a scheduler. |
| `src/elf_experiments/controller.py` and `tools/experimentctl.py` | Repository integration layer and its thin controller CLI. |
| Installed `experiment_control/runner.py` | Independently versioned injectable command boundary for production subprocesses and hermetic fakes. |
| Installed `experiment_control/preflight.py` | Independently versioned sanitized readiness checks and fail-closed reports. |
| [`ml-experiment-control`](https://github.com/NotDesigned/ml-experiment-control) | Commit-pinned backend, preflight, runner, state, sanitizer, project protocol, and adapter template package. |
| `src/elf_experiments/projects/elf.py` | The only adapter that knows ELF Config, `cloud_train.sh`, ELF checkpoints, metrics, and summaries. |
| `experiment_control/backends/wyd.py` | SSH, rsync, Slurm, identity probes, Apptainer staging/status/collection/cancellation. |
| `experiment_control/backends/sensecore.py` | Sanitized SCO identity/submission/status/logging/cancellation. |

The backend registry dispatches validation, platform environment resolution,
identity and asset verification, submission recovery, `stage`, `render`, `submit`,
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

The installed package owns backend `identity()` probes and refuses multiple
scheduler matches. ELF owns campaign YAML, fresh-instance generation, local
event reconciliation, and scientific health gates. This keeps site mechanics
reusable without teaching the package ELF run names or configs.

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
Existing legacy events are also audited: two recorded job IDs for one attempt
are an ambiguity error, never a “latest job wins” choice.

Cancellation uses a separate create-once `cancel_intent.json` bound to the
exact attempt and backend job ID. A repeated cancel returns an already verified
receipt, or performs status-only reconciliation. If the target is still
nonterminal after an unresolved request, the controller refuses a second
cancel mutation.

SenseCore resource names are attempt-qualified and actual creates use the
manifest's immutable `repository@sha256:...` reference while retaining the
authored source-qualified tag as provenance. WYD permits a new attempt in an
existing run directory only after the remote `manifest.yaml` digest exactly
matches the controller's frozen manifest.

### Identity separation

The ELF project adapter selects `source_identity.sh --runtime`, which hashes
only Docker/training runtime inputs.
Campaign YAML and documentation changes therefore do not force an identical
image or SIF rebuild. Manifests record four separate provenance values:

- full Git commit;
- runtime tree identity;
- campaign-file identity;
- registry image digest or SIF SHA-256.

These fields are required for newly prepared runs. Historical runs created by
older controller versions may lack some fields and must be reported as legacy
evidence rather than silently attributed to the current source.

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

ELF pins `ml-experiment-control` by commit in `requirements.txt`; the pin is
part of runtime source identity and the package is installed in the image/SIF.
WYD verifies ELF's project entrypoint and training module in the staged source,
while the container launcher imports the installed scheduler package before
any training work and prints its resolved module path. Login nodes may not have
`/dev/fuse` permission to mount the SIF for a true container-side preflight.

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

When a transient live log probe fails but the selected attempt already has
sanitized process evidence persisted by `observe` or `collect`, `logs` returns
that bounded cache with `live: false` and
`evidence_source: cached_collection`. It still fails closed when neither live
nor cached evidence exists.

SenseCore CLI v1.2 may print the exact non-JSON sentinel `No jobs found` on a
successful exact-name query requested as JSON. The bundled sanitizer maps only
that exact sentinel (or empty stdout) to `[]`; every other non-JSON response
still fails closed. Expired live logs produce an explicit
`evidence_unavailable_reason` and `INCONCLUSIVE` evidence outcome rather than a
claim that model artifacts do not exist.

Likewise, accessible logs that contain no project-parsed process or model
evidence are not proof that the model never executed. A cancelled terminal run
is reported as `INCONCLUSIVE/cancelled_before_observation`; other terminal runs
without either layer use `terminal_without_process_or_model_evidence`.

Collection records both `scheduler_state` and `runtime_state`. They may differ
after preemption or cancellation because a killed process cannot always update
its own status file; scheduler truth must not overwrite the historical runtime
observation or vice versa. Once the scheduler is terminal, current
`worker_state` is `RELEASED`; when process/runtime evidence exists, current
`process_state` follows that terminal state. Without such evidence it remains
`UNKNOWN`. The stale process-authored value remains available only as
`runtime_state`.

### Registry publication

SenseCore output sanitization lives in the installed `experiment_control`
package, so the repository does not depend on a particular user's `~/.codex`
directory. Publish immutable
images with `tools/push_registry_image.sh`; it distinguishes authorization
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
until the immutable source identity and run are materialized. Executable
examples live under [`experiments/campaigns/`](../experiments/campaigns/), so
schema examples cannot drift independently from validated campaigns.

An executable research campaign may declare one top-level
`research_contract` and exactly one `research_role` per required run. The
contract uses only fixed predicates (`finite`, ordered numeric comparisons,
bounded ranges, and non-finite early-stop detection); arbitrary expressions or
code are rejected. It declares normalized metrics and artifact counts, a
completed-checkpoint requirement, and fields that must match across roles.
The project adapter owns the mapping from repository files to normalized
evidence; scheduler backends do not know ELF metric names.

`decide` distinguishes `PENDING`, `PASS`, `FAIL`, and `INCONCLUSIVE`. Missing,
conflicting, expired, or inaccessible evidence is always `INCONCLUSIVE`, never
a failed hypothesis. A live non-finite metric produces `STOP_RECOMMENDED` but
never cancels automatically. `EXTEND` requires every role to pass and every
declared match field to agree. The reviewable four-run example is
[`experiments/templates/fusion_len256_gate_slurm.yml`](../experiments/templates/fusion_len256_gate_slurm.yml).

Files under `experiments/campaigns/` are immutable authored/history records.
Fresh executable identities come from templates. For example:

```bash
INSTANCE=$(date -u +%Y%m%dT%H%M%SZ)
python tools/instantiate_campaign.py \
  experiments/templates/backend_smoke_slurm.yml \
  --instance "$INSTANCE" \
  --local-root "outputs/experiment_campaigns/state/$INSTANCE"
```

The generator renders only `{instance}`, preserves controller placeholders
such as `{run_id}` and `{source_id}`, writes exclusively, and fails rather than
overwriting a previously generated campaign.

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

The SenseCore resource name is validated locally against the live API rule:
1–63 characters, starting with a lowercase letter, ending with a lowercase
letter or digit, and containing only lowercase letters, digits, and hyphens.
Attempt IDs used on SenseCore follow the same lowercase/internal-hyphen
discipline. Display names are not assumed to share this constraint.

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

`observe` reports scheduler, worker, process, and model layers separately.
Unavailable worker evidence is explicitly `UNKNOWN`; it is never inferred from
scheduler success. A terminal scheduler observation marks its allocation
`RELEASED` without inventing a process result. Collection records
`scheduler_state`, `runtime_state`,
`worker_state`, and `model_state` alongside the underlying metrics/artifacts.
For SenseCore, collection obtains the exact job's worker table through a
schema-checked sanitizer that drops host/pod IP columns and retains only worker
identity and phase.

Controller and runtime use `elf_experiments.run_manifest.build_run_manifest` for
the same canonical `manifest.yaml` schema. Slurm stages that manifest before
the job script; the runtime validates it before creating attempt/process
records. Checkpoint collection accepts only `checkpoint_<step>` payloads whose
`.complete` JSON marker has the same step and byte count. WYD probes those
markers remotely without copying checkpoint payloads; SenseCore extracts the
same committed path from sanitized launcher logs while they remain available.

### WYD site-specific execution notes

- `ssh_alias` is a login route only; `partition` and typed GPU `gres` remain
  explicit and independently validated before every submission.
- Resolve Slurm storage roots from the selected backend profile. L40S uses
  `/data/liangluocheng/elf`; H100 uses `/datapool/liangluocheng/elf`. Keep HF
  cache, checkpoints, W&B, and saved-model directories on that profile's shared
  filesystem. The container image's SenseCore-oriented `/data/elf` defaults are
  not writable on WYD.
- On the currently observed Slurm installation, batch jobs failed before the
  shell when `#SBATCH --output/--error` named normal shared files. Generated
  scripts therefore point Slurm's pre-shell streams at `/dev/null`, then the
  bash process immediately tees stdout/stderr into the attempt directory under
  `/data`. This preserves early application errors without relying on slurmd
  to create the shared log file.
- Collection rsyncs manifests, status, training/evaluation JSONL, and generated
  or reconstructed sample JSONL to the controller, then summarizes locally.
  Checkpoint payloads remain on shared storage.
- Log observation checks exact canonical stream paths and then exact
  `slurm-<job-id>.out/.err` paths; bounded redacted tails are retained as
  `process_evidence` for failures that occur before metrics/status exist.

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

This four-rank scientific gate is distinct from the one-GPU backend smoke. The
backend smoke validates one process through its first finite metric and
completed checkpoint; it makes no DDP/NCCL claim.

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

SenseCore spot attempts use `/data/elf/runs`. Slurm attempts use the selected
backend profile: L40S uses `/data/liangluocheng/elf/runs`, while H100 uses
`/datapool/liangluocheng/elf/runs`. A preemption creates a new `attempt_id` but
keeps the same scientific `run_id` and resumes only a completed checkpoint.
