from utils.data_utils import _prune_incomplete_dataset_cache


def test_prune_incomplete_dataset_cache_removes_matching_entries(tmp_path):
    cache_root = tmp_path / "datasets"
    target_root = cache_root / "org___dataset"
    stale_dir = target_root / "default" / "0.0.0" / "abc.incomplete"
    stale_dir.mkdir(parents=True)
    (stale_dir / "partial").write_text("unfinished", encoding="utf-8")

    other_dir = cache_root / "org___other" / "default" / "0.0.0" / "abc.incomplete"
    other_dir.mkdir(parents=True)

    _prune_incomplete_dataset_cache("org/dataset", dataset_cache_dir=str(cache_root))

    assert not stale_dir.exists()
    assert other_dir.exists()
