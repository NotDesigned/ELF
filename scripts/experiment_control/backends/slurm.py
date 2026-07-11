"""Pure Slurm rendering and accounting parsers."""

from __future__ import annotations

import shlex
from typing import Any, Iterable

from ..states import normalize_slurm_state


def shell_join(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(argument)) for argument in command)


def render_job(manifest: dict[str, Any]) -> str:
    backend = manifest["backend"]
    resources = manifest.get("resources", {})
    run_dir = manifest["storage"]["run_dir"]
    source_dir = backend["source_dir"]
    sif_path = backend["sif_path"]
    mount_root = str(backend.get("mount_root", "/data"))
    cache = str(backend.get("apptainer_cache_dir", f"{mount_root.rstrip('/')}/apptainer/cache/liangluocheng"))
    temp = str(backend.get("apptainer_tmp_dir", f"{mount_root.rstrip('/')}/apptainer/tmp/liangluocheng"))
    command = shell_join(manifest["command"])
    cpus = int(resources.get("cpus", 8))
    comment = shlex.quote(f"{manifest.get('campaign', 'campaign')}/{manifest['run_id']}/{manifest['attempt_id']}")
    return f"""#!/usr/bin/env bash
#SBATCH --partition={backend['partition']}
#SBATCH --account={backend['account']}
#SBATCH --qos={backend['qos']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres={backend['gres']}
#SBATCH --time={backend['time']}
#SBATCH --job-name={manifest['run_id']}
#SBATCH --comment={comment}
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
export APPTAINER_CACHEDIR={shlex.quote(cache)}
export APPTAINER_TMPDIR={shlex.quote(temp)}
export BACKEND_JOB_ID="$SLURM_JOB_ID"
mkdir -p {shlex.quote(run_dir)}
attempt_log_dir={shlex.quote(f"{run_dir}/attempts/{manifest['attempt_id']}")}
mkdir -p "$attempt_log_dir"
exec > >(tee -a "$attempt_log_dir/slurm-$SLURM_JOB_ID.out") \\
     2> >(tee -a "$attempt_log_dir/slurm-$SLURM_JOB_ID.err" >&2)
test -d {shlex.quote(source_dir)}
test -s {shlex.quote(sif_path)}
srun apptainer exec --nv \\
  --bind {shlex.quote(mount_root)}:{shlex.quote(mount_root)} \\
  --bind {shlex.quote(source_dir)}:/app \\
  --pwd /app \\
  {shlex.quote(sif_path)} \\
  {command}
"""


def parse_accounting(output: str, *, job_id: str, run_id: str, partition: str) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return {
            "run_id": run_id, "backend": "slurm", "backend_job_id": job_id,
            "state": "UNKNOWN", "raw_state": "UNKNOWN", "partition": partition,
            "elapsed": None, "exit_code": None,
        }
    fields = lines[0].split("|")
    fields += [""] * (6 - len(fields))
    raw = fields[3].split()[0].rstrip("+")
    return {
        "run_id": run_id, "backend": "slurm", "backend_job_id": job_id,
        "state": normalize_slurm_state(raw, fields[5]), "raw_state": raw,
        "partition": fields[2] or partition, "elapsed": fields[4] or None,
        "exit_code": fields[5] or None,
    }
