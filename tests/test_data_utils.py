import numpy as np
import pytest

from utils.data_utils import _prune_incomplete_dataset_cache, get_dataloader


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


def test_input_only_corpus_can_be_split_into_prefix_and_future():
    dataset = [
        {"input_ids": np.array([10, 11, 12, 13], dtype=np.int64)},
        {"input_ids": np.array([20, 21], dtype=np.int64)},
        {"input_ids": np.array([30, 31, 32, 33, 34], dtype=np.int64)},
    ]

    batch = next(iter(get_dataloader(
        dataset,
        batch_size=3,
        shuffle=False,
        drop_last=False,
        max_seq_length=6,
        max_input_seq_length=2,
        split_input_as_prefix=True,
        distributed=False,
    )))

    np.testing.assert_array_equal(
        batch["cond_seq_mask"],
        np.array([
            [1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ], dtype=np.float32),
    )
    # Prefix contextual embeddings cannot read future keys, while future rows
    # can read the entire valid prefix+future span.
    np.testing.assert_array_equal(
        batch["encoder_attention_mask"][0, 0],
        np.array([1, 1, 0, 0, 0, 0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batch["encoder_attention_mask"][0, 2],
        np.array([1, 1, 1, 1, 0, 0], dtype=np.float32),
    )


def test_input_prefix_split_requires_a_configured_boundary():
    loader = get_dataloader(
        [{"input_ids": np.array([1, 2], dtype=np.int64)}],
        batch_size=1,
        shuffle=False,
        drop_last=False,
        split_input_as_prefix=True,
        distributed=False,
    )

    with pytest.raises(ValueError, match="max_input_seq_length"):
        next(iter(loader))
