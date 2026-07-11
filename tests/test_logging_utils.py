import json

import pytest

from utils.logging_utils import append_jsonl_for_0


def test_append_jsonl_for_rank_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    path = tmp_path / "nested" / "train_metrics.jsonl"

    assert append_jsonl_for_0(path, {"step": 10, "train_loss": 1.25})
    assert json.loads(path.read_text()) == {"step": 10, "train_loss": 1.25}


def test_append_jsonl_skips_nonzero_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    path = tmp_path / "train_metrics.jsonl"

    assert not append_jsonl_for_0(path, {"step": 10})
    assert not path.exists()


def test_append_jsonl_rejects_non_finite_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(ValueError, match="Out of range float values"):
        append_jsonl_for_0(tmp_path / "train_metrics.jsonl", {"train_loss": float("nan")})
