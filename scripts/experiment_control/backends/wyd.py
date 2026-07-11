"""WYD Slurm/Apptainer side-effect adapter."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from .services import BackendServices
from .slurm import parse_accounting, render_job, shell_join


class WydSlurmBackend:
    kind = "slurm"

    def __init__(self, services: BackendServices):
        self.s = services

    def stage(self, campaign: dict[str, Any], run: dict[str, Any], source_id: str) -> bool:
        backend = run["backend"]
        expected_suffix = f"/sources/{source_id}"
        if not str(backend["source_dir"]).endswith(expected_suffix):
            raise ValueError(f"source_dir must end with {expected_suffix}")
        source_marker = f"{backend['source_dir']}/.source-complete"
        self.s.remote_exec(
            backend["ssh_alias"],
            shell_join(["mkdir", "-p", backend["source_dir"], str(Path(run["storage"]["run_dir"]).parent)]),
        )
        staged = self.s.remote_exec(
            backend["ssh_alias"], f"test -f {shlex.quote(source_marker)}", check=False
        ).returncode == 0
        if not staged:
            transport = f"ssh -o ControlMaster=auto -o ControlPersist=900 -o ControlPath={self.s.ssh_control_path}"
            self.s.run_command(
                ["rsync", "-a", "--delete", "-e", transport,
                 "--exclude=.git/", "--exclude=outputs/", "--exclude=runs/",
                 "--exclude=checkpoints/", "--exclude=wandb/", "--exclude=*.log",
                 f"{self.s.repo_root}/", f"{backend['ssh_alias']}:{backend['source_dir']}/"]
            )
            self.s.remote_exec(backend["ssh_alias"], shell_join(["touch", source_marker]))
        expected_image = str(run["image_id"])
        expected_sha = expected_image.removeprefix("sha256:")
        marker = f"{backend['sif_path']}.sha256-{expected_sha}.verified"
        valid = self.s.remote_exec(
            backend["ssh_alias"],
            f"test -s {shlex.quote(backend['sif_path'])} -a -f {shlex.quote(marker)}",
            check=False,
        ).returncode == 0
        if not valid:
            verify = self.s.remote_exec(
                backend["ssh_alias"],
                f"test -s {shlex.quote(backend['sif_path'])} && sha256sum {shlex.quote(backend['sif_path'])}",
            )
            actual_sha = verify.stdout.split()[0]
            if expected_image.startswith("sha256:") and actual_sha != expected_sha:
                raise ValueError(f"SIF checksum mismatch: expected {expected_image}, got sha256:{actual_sha}")
            self.s.remote_exec(backend["ssh_alias"], shell_join(["touch", marker]))
        return True

    def render(self, manifest: dict[str, Any]) -> str:
        return render_job(manifest)

    def validate_live(self, run: dict[str, Any]) -> dict[str, str]:
        backend = run["backend"]
        partition = backend["partition"]
        expected_gpu = backend["gres"].split(":", 2)[1]
        query = (
            f"sinfo -h -p {shlex.quote(partition)} -o '%P|%a|%l|%G'; "
            "sacctmgr -n -P show assoc where user=$(id -un) format=User,Account,Partition,QOS,DefaultQOS"
        )
        result = self.s.remote_exec(backend["ssh_alias"], query)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        partition_lines = [line for line in lines if line.split("|", 1)[0].rstrip("*") == partition]
        if not partition_lines:
            raise RuntimeError(f"Slurm partition is not currently visible: {partition}")
        fields = partition_lines[0].split("|")
        if len(fields) < 4 or fields[1] != "up" or f"gpu:{expected_gpu}:" not in fields[3]:
            raise RuntimeError(f"Slurm partition/GRES is not currently usable: {partition}/{backend['gres']}")
        associations = [line for line in lines if line not in partition_lines and "|" in line]
        if not any(backend["account"] in line.split("|") and backend["qos"] in line.split("|") for line in associations):
            raise RuntimeError(f"Slurm association does not expose account={backend['account']} qos={backend['qos']}")
        return {"partition": partition, "availability": fields[1], "gres": fields[3]}

    def submit(self, campaign, run, manifest, *, dry_run: bool) -> str:
        local_dir = self.s.local_run_dir(campaign, run)
        script_path = local_dir / "attempts" / manifest["attempt_id"] / "job.sbatch"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(self.render(manifest), encoding="utf-8")
        if dry_run:
            return "DRY_RUN"
        backend = run["backend"]
        self.validate_live(run)
        remote_script = f"{run['storage']['run_dir']}/controller-{manifest['attempt_id']}.sbatch"
        self.s.remote_exec(backend["ssh_alias"], shell_join(["mkdir", "-p", run["storage"]["run_dir"]]))
        transport = f"ssh -o ControlMaster=auto -o ControlPersist=900 -o ControlPath={self.s.ssh_control_path}"
        self.s.run_command(["rsync", "-a", "-e", transport, str(script_path), f"{backend['ssh_alias']}:{remote_script}"])
        result = self.s.remote_exec(backend["ssh_alias"], f"sbatch --parsable {shlex.quote(remote_script)}")
        job_id = result.stdout.strip().split(";", 1)[0]
        if not re.fullmatch(r"\d+", job_id):
            raise ValueError(f"unexpected sbatch response: {result.stdout!r}")
        return job_id

    def status(self, campaign, run) -> dict[str, Any]:
        record = self.s.backend_record(campaign, run)
        backend, job_id = run["backend"], str(record["backend_job_id"])
        result = self.s.remote_exec(
            backend["ssh_alias"],
            f"sacct -j {shlex.quote(job_id)} -X -n -P -o JobID,JobName,Partition,State,Elapsed,ExitCode",
            check=False,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            queue = self.s.remote_exec(
                backend["ssh_alias"], f"squeue -j {shlex.quote(job_id)} -h -o '%i|%j|%P|%T|%M|0:0'", check=False
            )
            lines = [line for line in queue.stdout.splitlines() if line.strip()]
        return parse_accounting("\n".join(lines), job_id=job_id, run_id=run["run_id"], partition=backend["partition"])

    def cancel(self, campaign, run) -> dict[str, Any]:
        current = self.status(campaign, run)
        if current["state"] in {"SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED"}:
            return current
        self.s.remote_exec(run["backend"]["ssh_alias"], f"scancel {shlex.quote(str(current['backend_job_id']))}")
        return self.status(campaign, run)

    def collect(self, campaign, run) -> dict[str, Any]:
        backend = run["backend"]
        mirror = self.s.local_run_dir(campaign, run) / "collected_run"
        mirror.mkdir(parents=True, exist_ok=True)
        transport = f"ssh -o ControlMaster=auto -o ControlPersist=900 -o ControlPath={self.s.ssh_control_path}"
        self.s.run_command(
            ["rsync", "-a", "--delete", "-e", transport,
             "--include=*/", "--include=manifest.yaml", "--include=status.json",
             "--include=backend.json", "--include=train_metrics.jsonl",
             "--include=metrics.jsonl", "--exclude=*",
             f"{backend['ssh_alias']}:{run['storage']['run_dir']}/", f"{mirror}/"]
        )
        summary = self.s.summarize_run(mirror)
        summary["collected_from"] = run["storage"]["run_dir"]
        summary["run_dir"] = run["storage"]["run_dir"]
        return summary
