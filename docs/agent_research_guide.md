# Automated Research Agent Guide

This is the entry point for an agent that must turn an ELF hypothesis into
reproducible experiments, observe them on either supported backend, and return
an evidence-backed next action. It describes the decision loop; the linked
references own the architecture, field, and controller details.

## Read in this order

1. Read [`fusion_architecture.md`](fusion_architecture.md) to understand the
   current hypothesis, treatment axes, ablation ladder, and scientific gates.
2. Read [`config_reference.md`](config_reference.md) before changing a YAML
   field or CLI override. It is the maintained flag and resume-semantics
   reference.
3. Read the selected file under [`experiments/campaigns/`](../experiments/campaigns/)
   to review the exact run IDs, comparison block, resources, immutable image,
   storage, and retry budget.
4. Use [`experiment_workflow.md`](experiment_workflow.md) for the durable state
   model, campaign schema, backend behavior, artifact contracts, and failure
   policy.
5. Read [`packages/experiment-control/README.md`](../packages/experiment-control/README.md)
   only when changing, testing, or reusing the backend package itself.

Repository documents define project intent and stable interfaces. Mutable site
facts belong to the operator environment. When running under Codex, load
`orchestrate-ml-experiments` for the cross-platform lifecycle and exactly one
of `operate-sensecore` or `operate-wyd-slurm` for current platform discovery
and recovery. Those runbooks supplement this guide; they do not override the
campaign's frozen scientific identity or mutation safeguards.

Campaigns in the repository are versioned records and examples, not proof that
their image, partition, quota, credentials, or run IDs are suitable now. Query
live backend state before submission and create new run IDs for new scientific
runs.

## Non-negotiable invariants

- Define one falsifiable question and its matched control before allocating a
  GPU. Keep seed, data, training budget, backend, GPU type/count, evaluation,
  and checkpoint policy matched unless one of them is the treatment.
- A `run_id` is scientific identity. Infrastructure retries use a new
  `attempt_id` under the same run; a changed hypothesis or scientific config
  uses a new run.
- Freeze and report the Git commit, runtime-tree identity, campaign identity,
  and immutable image digest or SIF SHA-256. Do not compare ambiguous builds.
- Never put credentials in campaign YAML, commands, manifests, logs, or agent
  reports. Backends obtain them from SCO, Docker, and SSH native stores.
- Do not infer model health from scheduler state. Scheduler, worker/process,
  checkpoint, training metric, and evaluation evidence are separate gates.
- Resume only a checkpoint with a valid `.complete` marker. Never silently
  change backend, resources, batch semantics, timeout, or scientific flags to
  make a failed run pass.

## Research loop

### 1. Orient and state the hypothesis

Inspect the working tree and recent commits, then state:

- the causal question;
- control and treatment run IDs;
- the single intended scientific difference;
- the metric and direction that would support the hypothesis;
- the early-stop gate and the evidence required to continue.

For the first length-256 fusion gate, the intended A0–A3 comparisons and their
interpretation are maintained in
[`experiment_workflow.md`](experiment_workflow.md#first-length-256-campaign).

### 2. Validate locally and freeze identity

Run the relevant unit tests and config validation before platform access. Then
review the fully expanded campaign rather than only its defaults or profile.
Use unique run IDs and an immutable image/SIF identity.

```bash
python scripts/experimentctl.py CAMPAIGN.yml prepare --run RUN_ID
python scripts/experimentctl.py CAMPAIGN.yml assets-plan --run RUN_ID
python scripts/experimentctl.py CAMPAIGN.yml render --run RUN_ID
python scripts/experimentctl.py CAMPAIGN.yml submit --run RUN_ID --dry-run
```

These commands can create or update controller-owned local metadata. The
dry-run renders and validates but does not submit to a scheduler.

### 3. Prove backend and asset readiness

Choose the backend explicitly in the campaign. The WYD SSH alias is only a
login route; Slurm `partition` and typed `gres` select the compute resource.
SenseCore jobs must declare the approved spot quota and persistent AFS mount.

```bash
python scripts/experimentctl.py CAMPAIGN.yml preflight --scope stage --run RUN_ID
python scripts/experimentctl.py CAMPAIGN.yml stage --run RUN_ID
python scripts/experimentctl.py CAMPAIGN.yml preflight --scope submit --run RUN_ID
python scripts/experimentctl.py CAMPAIGN.yml assets-verify --run RUN_ID
```

Preflight and asset verification inspect remote state without changing the
scheduler. `stage` mutates remote storage. WYD can verify assets directly on
its persistent filesystem; SenseCore may report that verification requires a
running worker with the AFS mount, which must remain an explicit unresolved
gate rather than being mistaken for success. Resolve missing credentials,
storage, image, cache, partition/GRES, account/QOS, or mount failures before
submission; do not repair them implicitly inside the training job.

Before a long run, estimate the time to the first useful metric and completed
checkpoint using a current smoke measurement. Historical throughput is only a
planning hint because hardware, data cache, image, and logging policy change.

### 4. Submit intentionally

Submit one small comparison block at a time. The non-dry-run command is a
scheduler mutation:

```bash
python scripts/experimentctl.py CAMPAIGN.yml submit --run RUN_ID
```

The controller records a submission intent before calling the scheduler and
reconciles an interrupted submission rather than creating a duplicate job.
Do not submit the same scientific run concurrently to two backends.

### 5. Observe all evidence layers

```bash
python scripts/experimentctl.py CAMPAIGN.yml status --run RUN_ID
python scripts/experimentctl.py CAMPAIGN.yml logs --run RUN_ID --tail 200
python scripts/experimentctl.py CAMPAIGN.yml observe --run RUN_ID
python scripts/experimentctl.py CAMPAIGN.yml collect --run RUN_ID
python scripts/experimentctl.py CAMPAIGN.yml decide --run RUN_ID
```

`status`, `observe`, `collect`, and `decide` persist normalized local records;
`logs` returns a bounded, redacted snapshot. Inspect, in order:

1. scheduler placement and terminal reason;
2. worker/process start and exit state;
3. latest completed checkpoint;
4. finite loss, step, throughput, and plan-collapse diagnostics;
5. evaluation artifacts and the predeclared comparison metric.

W&B is an optional visualization surface, not the source of truth. Automation
must remain functional with `USE_WANDB=false` by using persistent JSONL,
manifests, status, checkpoint markers, and collected evaluation records.

### 6. Decide within policy

Retry automatically only when `decision.json` classifies a transport,
scheduler/node, or preemption failure, the campaign's infrastructure retry
budget remains, and a completed checkpoint is available when resume is needed.
OOM, timeout, configuration, model, and evaluation failures require analysis
or an explicit human-approved change.

Cancel only with deliberate operator intent:

```bash
python scripts/experimentctl.py CAMPAIGN.yml cancel --run RUN_ID
```

After collection, compare treatments only within matched seed/backend blocks.
Record inconclusive outcomes as such; absence of logs or metrics is not a
negative scientific result.

## Stop and ask for a human

Pause before mutating remote state when any of these is true:

- login, registry authorization, or credentials are missing or expired;
- the requested submission target, quota, cost, or run set is ambiguous;
- the proposed recovery changes scientific config, backend, GPU resources,
  global-batch semantics, timeout, image identity, or storage ownership;
- an immutable image/SIF or required offline asset cannot be verified;
- OOM, timeout, model divergence, corrupt checkpoint, or evaluation failure is
  the current diagnosis;
- a previous submission intent cannot be reconciled to exactly one scheduler
  job.

## Agent report contract

Every progress or final report should contain enough evidence for another
agent to continue without guessing:

```text
hypothesis / comparison:
campaign, run_id, attempt_id:
backend and scheduler job ID:
Git commit, runtime identity, image/SIF identity:
scheduler state and reason:
process/runtime state:
latest completed checkpoint:
latest finite training metrics:
evaluation metrics and artifact paths:
failure classification, if any:
next action allowed by policy:
open uncertainty or human decision required:
```

Never report “running,” “failed,” or “successful” without naming which evidence
layer supports that conclusion.
