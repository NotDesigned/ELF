from pathlib import Path

from experiment_assets import plan_assets, verify_assets


ROOT = Path(__file__).resolve().parents[1]
PURE = "src/configs/training_configs/ablations/owt_elfb/tier0_0_pure_elf_len256.yml"
FROZEN = "src/configs/training_configs/ablations/owt_elfb/tier0_1_sentence_t5_len256.yml"


def test_asset_plan_is_config_aware(monkeypatch):
    monkeypatch.chdir(ROOT)
    pure = plan_assets(PURE, [])
    frozen = plan_assets(FROZEN, [])
    assert not any(item.identity == "sentence-transformers/sentence-t5-xl" for item in pure)
    assert any(item.identity == "sentence-transformers/sentence-t5-xl" for item in frozen)


def test_asset_verify_reports_resolved_cache_path(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    requirements = plan_assets(PURE, ["online_eval=false"])
    missing = verify_assets(requirements, tmp_path / "hf", tmp_path / "datasets")
    assert {item["kind"] for item in missing} == {"model", "dataset"}
    assert all(Path(item["path"]).is_absolute() for item in missing)
