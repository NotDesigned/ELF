from pathlib import Path

from experiment_assets import cache_path, plan_assets
from experiment_control.project import AssetRequirement


ROOT = Path(__file__).resolve().parents[1]
PURE = "src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf_len256.yml"
FROZEN = "src/configs/training_configs/ablations/owt_elfb/tier0_1_sentence_t5_len256.yml"


def test_asset_plan_is_config_aware(monkeypatch):
    monkeypatch.chdir(ROOT)
    pure = plan_assets(PURE, [])
    frozen = plan_assets(FROZEN, [])
    assert not any(item.identity == "sentence-transformers/sentence-t5-xl" for item in pure)
    assert any(item.identity == "sentence-transformers/sentence-t5-xl" for item in frozen)


def test_asset_cache_path_maps_remote_identities(tmp_path):
    hf_home = tmp_path / "hf"
    datasets = tmp_path / "datasets"
    model = AssetRequirement("model", "org/model", "encoder")
    dataset = AssetRequirement("dataset", "org/data", "training")
    checkpoint = AssetRequirement("file", "/data/checkpoint", "warm start")
    assert cache_path(model, hf_home, datasets) == hf_home / "hub/models--org--model"
    assert cache_path(dataset, hf_home, datasets) == datasets / "org___data"
    assert cache_path(checkpoint, hf_home, datasets) == Path("/data/checkpoint")
