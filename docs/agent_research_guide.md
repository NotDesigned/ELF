# Automated Research Agent Guide

Use this entry point to turn an ELF hypothesis into reproducible experiments
and an evidence-backed next action. Detailed schemas and implementation
contracts stay in the linked references.

## Read before acting

1. [`fusion_architecture.md`](fusion_architecture.md): hypothesis, treatment
   axes, ablation ladder, and scientific gates.
2. [`config_reference.md`](config_reference.md): exact training-field and
   override semantics.
3. The selected file in
   [`experiments/campaigns/`](../experiments/campaigns/) or a fresh file created
   from [`experiments/templates/`](../experiments/templates/): run IDs,
   comparison block, resources, immutable image, storage, and retry budget.
4. [`experiment_workflow.md`](experiment_workflow.md): durable state, campaign
   schema, backend behavior, artifacts, and failure policy.
5. [`packages/experiment-control/README.md`](../packages/experiment-control/README.md):
   only when modifying or reusing the backend package.

Repository documents define stable project intent. Discover mutable platform
facts through the current operator runbook. Under Codex, use
`orchestrate-ml-experiments` plus exactly one of `operate-sensecore` or
`operate-wyd-slurm`. A historical campaign is not evidence that its image,
quota, partition, credentials, or run IDs are usable now.

## Invariants

- State one falsifiable question, its matched control, decision metric, and
  early-stop gate before allocating GPUs.
- A `run_id` is scientific identity. Infrastructure retries use a new
  `attempt_id`; changed scientific config requires a new run.
- Freeze Git, runtime-tree, campaign, and image/SIF identities.
- Keep credentials in SCO, Docker, or SSH native stores, never campaign data.
- Treat scheduler, process, checkpoint, training metric, and evaluation states
  as separate evidence layers. W&B is an optional mirror, not canonical state.
- Resume only a checkpoint with a valid `.complete` marker. Never silently
  change backend, resources, batch semantics, timeout, or scientific flags.

## Standard loop

Historical campaign files are records, not reusable allocations. For the
standard ELF Slurm smoke, create a reviewable fresh definition with an explicit
unique instance (do not overwrite an existing output):

```bash
CAMPAIGN=$(python scripts/instantiate_campaign.py \
  experiments/templates/backend_smoke_slurm.yml \
  --instance 20260712T120000)
```

Review the generated YAML and validate the relevant code/config. Then use one
run at a time while establishing a new execution path:

```bash
CTL="python scripts/experimentctl.py $CAMPAIGN"
$CTL check-identity --run RUN_ID
$CTL assets-plan --run RUN_ID
$CTL submit --run RUN_ID --dry-run
$CTL assets-verify --run RUN_ID
$CTL stage --run RUN_ID
$CTL submit --run RUN_ID
$CTL observe --run RUN_ID
$CTL decide --run RUN_ID
```

`check-identity` is the first live read-only gate and exits nonzero for a
consumed or ambiguous run/attempt. Dry-run writes controller-local metadata;
`stage` changes remote storage; non-dry-run `submit` changes the scheduler.
Both mutations enforce their own scoped preflight. Real submit repeats the
identity gate, so bypassing the documented order cannot silently duplicate a
job.

For WYD, `stage` also verifies the project-declared required files in the
staged source tree. The ELF launcher performs the actual import check as its
first container-side action. Login nodes may lack permission to mount SIF
images through `/dev/fuse`, so do not treat a login-side `apptainer exec` as a
portable pre-submit gate.

WYD can verify assets on persistent storage. SenseCore may report that asset
verification requires a running AFS-mounted worker; preserve that as an open
gate. Before a long preemptible run, estimate time to the first useful metric
and completed checkpoint from a current smoke measurement.

Use these narrower commands only when diagnosing or operating explicitly:

```bash
$CTL status --run RUN_ID                 # scheduler only
$CTL logs --run RUN_ID --tail 200        # bounded redacted snapshot
$CTL collect --run RUN_ID                # refresh collected artifacts
$CTL cancel --run RUN_ID                 # scheduler mutation
```

Inspect observation evidence in this order:

1. scheduler placement and terminal reason;
2. worker/process start and exit state;
3. latest completed checkpoint;
4. finite loss, throughput, and collapse diagnostics;
5. required evaluation artifacts and the predeclared comparison metric.

A one-GPU backend smoke proves launcher, storage, image, cache, one training
step, structured metrics, and checkpoint commit. It does **not** prove DDP/NCCL
health. A distributed fusion gate separately requires the campaign-declared
rank count (four ranks in the current A0–A3 gate) to initialize and train.

Retry only a classified transport, scheduler/node, or preemption failure when
the declared retry budget remains. OOM, timeout, configuration, model, and
evaluation failures require explicit analysis or approval. Compare treatments
only within matched seed/backend blocks; missing evidence is inconclusive, not
a negative scientific result.

## Stop before remote mutation

Ask for a human decision when credentials are unavailable; target, quota,
cost, or run set is ambiguous; recovery would change scientific identity or
resources; immutable inputs cannot be verified; the failure is OOM, timeout,
model divergence, checkpoint corruption, or evaluation failure; or one
submission intent cannot be reconciled to exactly one scheduler job.

## Report contract

Every handoff must identify:

```text
hypothesis and comparison
campaign / run_id / attempt_id
backend / scheduler job ID / artifact root
Git / runtime / campaign / image identities
scheduler / process / model states
latest completed checkpoint and checkpoint exposure
latest finite training and evaluation metrics
failure class and next policy-permitted action
open uncertainty or required human decision
```

Never report “running,” “failed,” or “successful” without naming the evidence
layer that supports it.
