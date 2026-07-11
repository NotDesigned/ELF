"""SenseCore SCO side-effect adapter with immediate output sanitization."""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Callable

from experiment_manifest import atomic_write, utc_now

from .services import BackendServices


def normalize_state(raw_state: str, *, cancellation_requested: bool = False) -> str:
    raw = raw_state.upper()
    if cancellation_requested and raw in {"SUSPENDING", "SUSPENDED", "DELETING", "DELETED"}:
        return "CANCELLED"
    if raw in {"WAITING", "INIT", "QUEUEING", "PENDING", "CREATING"}:
        return "QUEUED"
    if raw in {"STARTING", "RECOVERING"}:
        return "STARTING"
    if raw in {"RUNNING", "RESTARTING"}:
        return "RUNNING"
    if raw in {"SUCCEEDED", "COMPLETED"}:
        return "SUCCEEDED"
    if raw in {"SUSPENDING", "SUSPENDED"}:
        return "PREEMPTED"
    if raw in {"FAILED", "ERROR"}:
        return "FAILED"
    if raw in {"DELETING", "DELETED", "CANCELLED", "CANCELED"}:
        return "CANCELLED"
    return "UNKNOWN"


class SenseCoreBackend:
    kind = "sensecore"

    def __init__(self, services: BackendServices, metric_parser: Callable[[str], dict[str, Any] | None]):
        self.s = services
        self.metric_parser = metric_parser

    def validate(self, run: dict[str, Any]) -> None:
        backend = run["backend"]
        required = {"workspace", "aec2", "worker_spec", "image", "storage_mount", "quota_type", "job_name"}
        missing = sorted(key for key in required if not backend.get(key))
        if missing:
            raise ValueError(f"run {run['run_id']} backend is missing: {missing}")
        if backend["quota_type"] != "spot":
            raise ValueError("SenseCore runs for this account must use spot quota")
        if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", str(run["image_id"])):
            raise ValueError(f"run {run['run_id']} SenseCore image_id must be a registry digest")
        image = str(backend["image"])
        tag = image.rsplit(":", 1)[-1]
        if tag in {"latest", "runtime", "seed"} or ":" not in image:
            raise ValueError(f"run {run['run_id']} SenseCore image must use an immutable source-qualified tag")
        mount_path = Path(str(backend["storage_mount"]).rsplit(":", 1)[-1])
        if not mount_path.is_absolute():
            raise ValueError(f"run {run['run_id']} SenseCore storage mount path must be absolute")
        for field, value in run["storage"].items():
            if field == "run_dir" or field.endswith(("_root", "_home", "_cache")):
                path = Path(str(value))
                if not path.is_relative_to(mount_path):
                    raise ValueError(
                        f"run {run['run_id']} storage.{field} must be under mounted path {mount_path}"
                    )

    def environment(self, campaign, run, source_id, attempt_id) -> dict[str, str]:
        backend = run["backend"]
        return {
            "BACKEND_JOB_ID": str(backend["job_name"]),
            "QUOTA_TYPE": str(backend["quota_type"]),
            "RESOURCE_SPEC": str(backend["worker_spec"]),
        }

    def submission_request(self, campaign, run, attempt_id) -> dict[str, Any]:
        return {"scheduler_name": str(run["backend"]["job_name"])}

    def recover_submission(self, run, intent, attempt_id) -> str | None:
        return str(run["backend"]["job_name"]) if self.find(run) else None

    def verify_assets(self, run, requirements, environment) -> dict[str, Any]:
        return {
            "missing": None,
            "verification": "requires-running-sensecore-worker",
            "verified_on": None,
        }

    def safe_command(self, arguments: list[str], mode: str) -> list[str]:
        safe_script = self.s.script_dir / "safe_sco.py"
        sco = shlex.join(["env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
                          "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY", *arguments])
        return ["bash", "-o", "pipefail", "-c", f"{sco} | {shlex.join([sys.executable, str(safe_script), mode])}"]

    def describe(self, run: dict[str, Any]) -> dict[str, Any]:
        backend = run["backend"]
        result = self.s.run_command(
            self.safe_command(["sco", "acp", "jobs", "describe", backend["job_name"],
                               "--workspace-name", backend["workspace"], "-o", "json"], "job-summary"),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "sanitized SenseCore describe failed")
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise ValueError("SenseCore describe sanitizer returned a non-object")
        return payload

    def find(self, run: dict[str, Any]) -> list[dict[str, Any]]:
        backend = run["backend"]
        result = self.s.run_command(
            self.safe_command(["sco", "acp", "jobs", "list", "--workspace-name", backend["workspace"],
                               "--name", backend["job_name"], "--page-size", "5", "-o", "json"], "job-list"),
            check=False,
        )
        if result.returncode != 0:
            if not result.stdout.strip() and result.stderr.strip() == "safe_sco: input was not valid JSON; raw response suppressed":
                return []
            raise RuntimeError(result.stderr.strip() or "sanitized SenseCore list failed")
        payload = json.loads(result.stdout)
        if not isinstance(payload, list):
            raise ValueError("SenseCore list sanitizer returned a non-list")
        return [item for item in payload if item.get("name") == backend["job_name"]]

    def stage(self, campaign, run, source_id) -> bool:
        return False

    def render(self, manifest) -> str:
        return shlex.join(manifest["command"])

    def _redact_error(self, text: str) -> str:
        result = self.s.run_command(
            [sys.executable, str(self.s.script_dir / "safe_sco.py"), "redact-lines"],
            input_text=text, check=False,
        )
        return result.stdout.strip()

    def submit(self, campaign, run, manifest, *, dry_run: bool) -> str:
        backend = run["backend"]
        if dry_run:
            return "DRY_RUN"
        if self.find(run):
            raise FileExistsError(f"SenseCore job already exists: {backend['job_name']}")
        create = [
            "env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
            "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY",
            "sco", "acp", "jobs", "create", "--workspace-name", backend["workspace"],
            "--aec2-name", backend["aec2"], "--name", backend["job_name"],
            "--job-name", backend["display_name"], "--container-image-url", backend["image"],
            "--training-framework", "pytorch", "--worker-spec", backend["worker_spec"],
            "--worker-nodes", str(backend.get("worker_nodes", 1)),
            "--priority", str(backend.get("priority", "NORMAL")),
            "--quota-type", backend["quota_type"], "--storage-mount", backend["storage_mount"],
            "--wait", "--command", shlex.join(manifest["command"]),
        ]
        result = self.s.run_command(create, check=False)
        if result.returncode != 0:
            raise RuntimeError(self._redact_error(result.stderr) or "SenseCore create failed")
        summary = self.describe(run)
        if summary.get("name") != backend["job_name"]:
            raise RuntimeError("SenseCore accepted create but exact job was not observable")
        return backend["job_name"]

    def status(self, campaign, run) -> dict[str, Any]:
        record = self.s.backend_record(campaign, run)
        summary = self.describe(run)
        marker = self.s.local_run_dir(campaign, run) / "cancel_requested.json"
        state = normalize_state(str(summary.get("state", "")), cancellation_requested=marker.is_file())
        return {
            "run_id": run["run_id"], "backend": "sensecore",
            "backend_job_id": record["backend_job_id"],
            "state": state,
            "raw_state": summary.get("state"), "pool": summary.get("pool"), "spec": summary.get("spec"),
            "failure_class": "preemption" if state == "PREEMPTED" else None,
        }

    def cancel(self, campaign, run) -> dict[str, Any]:
        current = self.status(campaign, run)
        marker = self.s.local_run_dir(campaign, run) / "cancel_requested.json"
        atomic_write(marker, {"run_id": run["run_id"], "backend_job_id": current["backend_job_id"], "requested_at": utc_now()})
        if current.get("raw_state") in {"SUSPENDING", "SUSPENDED", "DELETING", "DELETED"}:
            current["state"] = "CANCELLED"
            return current
        if current["state"] in {"SUCCEEDED", "FAILED", "PREEMPTED", "CANCELLED"}:
            return current
        backend = run["backend"]
        result = self.s.run_command(
            ["env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
             "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY",
             "sco", "acp", "jobs", "stop", backend["job_name"], "--workspace-name", backend["workspace"]],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(self._redact_error(result.stderr) or "SenseCore stop failed")
        return self.status(campaign, run)

    def collect(self, campaign, run) -> dict[str, Any]:
        snapshot = self.logs(campaign, run, tail=200)
        lines = snapshot["lines"]
        metrics = [metric for line in lines if (metric := self.metric_parser(line))]
        metric_lines = [line for line in lines if "Step " in line or "gPPL:" in line or ("plan" in line.lower() and "ppl" in line.lower())]
        return {"run_id": run["run_id"], "backend": "sensecore", "model_observed": bool(metrics),
                "latest_metric": metrics[-1] if metrics else None, "metric_log_lines": metric_lines[-20:],
                "live_logs_expired": snapshot["expired"]}

    def logs(self, campaign, run, *, tail: int) -> dict[str, Any]:
        backend = run["backend"]
        result = self.s.run_command(
            ["timeout", "20s", "env", "-u", "http_proxy", "-u", "https_proxy", "-u", "all_proxy",
             "-u", "HTTP_PROXY", "-u", "HTTPS_PROXY", "-u", "ALL_PROXY", "sco", "acp", "jobs",
             "stream-logs", backend["job_name"], "--workspace-name", backend["workspace"]],
            check=False,
        )
        redacted = self._redact_error(result.stdout + "\n" + result.stderr)
        lines = redacted.splitlines()[-tail:]
        expired = result.returncode != 0 and any(
            token in redacted.lower() for token in ("expired", "403", "offline log")
        )
        return {
            "run_id": run["run_id"], "backend": "sensecore",
            "backend_job_id": backend["job_name"], "tail": tail,
            "lines": lines, "expired": expired, "stream_exit_code": result.returncode,
        }
