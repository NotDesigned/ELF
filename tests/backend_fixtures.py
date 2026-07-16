from pathlib import Path


CONFIG = "src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf_len256.yml"


def slurm_campaign(tmp_path: Path) -> dict:
    """Return a concrete Slurm campaign for backend adapter tests."""
    return {
        "schema_version": 1,
        "campaign": "controller-test",
        "project": "elf",
        "source_id": "source-fixed",
        "local_root": str(tmp_path / "local"),
        "runs": [
            {
                "run_id": "smoke-h100",
                "config": CONFIG,
                "config_overrides": ["epochs=1", "save_freq=0.1"],
                "image_id": "sha256:" + "a" * 64,
                "resources": {"gpus": 1, "cpus": 8},
                "storage": {
                    "run_dir": "/mnt/test-project/runs/smoke-h100",
                    "data_root": "/mnt",
                    "project_data_root": "/mnt/test-project",
                    "hf_home": "/mnt/test-project/cache/huggingface",
                    "hf_datasets_cache": "/mnt/test-project/cache/huggingface/datasets",
                },
                "env": {"BATCH_SIZE": "4", "LOG_FREQ": "10"},
                "backend": {
                    "kind": "slurm",
                    "ssh_alias": "test-login",
                    "partition": "h100",
                    "account": "lab",
                    "qos": "normal",
                    "gres": "gpu:h100:1",
                    "time": "00:10:00",
                    "mount_root": "/mnt",
                    "source_dir": "/mnt/test-project/sources/{source_id}",
                    "sif_path": "/mnt/test-project/images/test.sif",
                },
            }
        ],
    }
