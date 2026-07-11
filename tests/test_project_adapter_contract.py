"""Proof that controller preparation does not require ELF's config package."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import experimentctl
from experiment_control.backends.base import BackendRegistry
from experiment_control.project import (
    AssetProbe,
    AssetRequirement,
    ProjectRegistry,
    SourceBundle,
)


class DummyProject:
    name = "dummy"
    safe_env_keys = frozenset()

    def validate_run(self, run):
        if Path(run["config"]).suffix != ".yaml":
            raise ValueError("dummy configs are YAML")

    def operational_overrides(self, env, output_dir):
        return [f"output={output_dir}"]

    def resolve_config(self, config_path, overrides):
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        return {**payload, "applied": list(overrides)}

    def environment(self, campaign, run):
        return {"DUMMY_OUTPUT": run["storage"]["run_dir"]}

    def command(self, run):
        return ["python", "train.py", "--config", str(run["config"])]

    def plan_assets(self, config_path, overrides):
        return [AssetRequirement("file", "weights.bin", "dummy weights")]

    def asset_probes(self, requirements, environment):
        return [AssetProbe(requirements[0], "/persistent/weights.bin", file=True)]

    def parse_metric(self, line):
        if line.startswith("metric="):
            return {"score": float(line.split("=", 1)[1])}
        return None

    def parse_checkpoint(self, line):
        return None

    def summarize(self, run_dir):
        return {"run_dir": str(run_dir), "project": self.name}

    def source_bundle(self, repo_root):
        return SourceBundle(repo_root, excludes=("artifacts/",), container_path="/work")


class DummyBackend:
    kind = "dummy-backend"

    def validate(self, run):
        return None

    def environment(self, campaign, run, source_id, attempt_id):
        return {"BACKEND_JOB_ID": "dummy-job"}

    def render(self, manifest):
        return " ".join(manifest["command"])


def test_prepare_render_assets_and_metrics_without_elf_config(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("seed: 7\nmodel: tiny\n", encoding="utf-8")
    campaign = {
        "schema_version": 1,
        "campaign": "dummy-campaign",
        "project": "dummy",
        "local_root": str(tmp_path / "control"),
    }
    run = {
        "run_id": "dummy-s0",
        "config": str(config),
        "image_id": "sha256:" + "a" * 64,
        "backend": {"kind": "dummy-backend"},
        "storage": {
            "run_dir": "/persistent/runs/dummy-s0",
            "data_root": "/persistent",
            "project_data_root": "/persistent/dummy",
            "hf_home": "/persistent/cache",
            "hf_datasets_cache": "/persistent/cache/datasets",
        },
        "env": {},
    }
    project = DummyProject()
    monkeypatch.setattr(experimentctl, "PROJECTS", ProjectRegistry(project))
    monkeypatch.setattr(experimentctl, "BACKENDS", BackendRegistry(DummyBackend()))

    experimentctl.validate_run(run, project="dummy")
    manifest = experimentctl.prepare_run(
        campaign, run, "source-dummy", attempt_id="attempt-001"
    )

    assert manifest["resolved_config"]["model"] == "tiny"
    assert manifest["execution"] == {"source_mount": "/work", "workdir": "/work"}
    assert "python train.py" in DummyBackend().render(manifest)
    assert project.plan_assets(str(config), [])[0].identity == "weights.bin"
    assert project.parse_metric("metric=0.75") == {"score": 0.75}
    assert not any("configs.config" in argument for argument in manifest["command"])
