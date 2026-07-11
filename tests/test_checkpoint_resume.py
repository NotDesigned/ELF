import json
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.checkpoint_utils import find_latest_checkpoint, load_checkpoint, save_checkpoint
from utils.train_utils import TrainState


def make_state(seed=7):
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return TrainState(
        model=model,
        optimizer=optimizer,
        ema_params1=TrainState.init_ema(model),
        step=8,
        micro_step=8,
        optimizer_step=4,
        accum_step=0,
        epoch=1.5,
        dropout_generator=torch.Generator(device="cpu").manual_seed(seed),
    )


def test_checkpoint_is_atomic_complete_and_restores_counters(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    state = make_state()
    expected_weight = state.model.weight.detach().clone()
    save_checkpoint(state, str(tmp_path), step=8)

    checkpoint = tmp_path / "checkpoint_8"
    marker = tmp_path / "checkpoint_8.complete"
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_payload["bytes"] == checkpoint.stat().st_size
    assert find_latest_checkpoint(str(tmp_path)) == str(checkpoint)
    assert not list(tmp_path.glob("*.tmp"))

    with torch.no_grad():
        state.model.weight.zero_()
    state.step = state.micro_step = state.optimizer_step = 0
    state.epoch = 0
    state, step = load_checkpoint(str(tmp_path), state)

    assert step == 8
    assert state.micro_step == 8
    assert state.optimizer_step == 4
    assert state.accum_step == 0
    assert state.epoch == pytest.approx(1.5)
    assert torch.equal(state.model.weight, expected_weight)


def test_incomplete_checkpoint_is_ignored_and_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    state = make_state()
    save_checkpoint(state, str(tmp_path), step=8)
    torch.save({"params": state.model.state_dict(), "opt_state": {}, "step": 9, "epoch": 2}, tmp_path / "checkpoint_9")

    assert find_latest_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint_8")
    with pytest.raises(ValueError, match="completion marker"):
        load_checkpoint(str(tmp_path / "checkpoint_9"), state)


def test_directory_resume_falls_back_from_corrupt_completed_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    state = make_state()
    save_checkpoint(state, str(tmp_path), step=8)
    corrupt = tmp_path / "checkpoint_9"
    corrupt.write_bytes(b"not a torch checkpoint")
    (tmp_path / "checkpoint_9.complete").write_text(
        json.dumps({"bytes": corrupt.stat().st_size, "step": 9}), encoding="utf-8"
    )

    restored, step = load_checkpoint(str(tmp_path), state)
    assert restored is state
    assert step == 8


def test_eval_directory_ignores_uncommitted_or_size_mismatched_checkpoints(tmp_path):
    state = make_state()
    payload = {
        "params": state.model.state_dict(),
        "step": 8,
        "epoch": 1,
    }
    committed = tmp_path / "checkpoint_8"
    torch.save(payload, committed)
    (tmp_path / "checkpoint_8.complete").write_text(
        json.dumps({"bytes": committed.stat().st_size, "step": 8}), encoding="utf-8"
    )

    uncommitted = tmp_path / "checkpoint_9"
    torch.save({**payload, "step": 9}, uncommitted)
    mismatched = tmp_path / "checkpoint_10"
    torch.save({**payload, "step": 10}, mismatched)
    (tmp_path / "checkpoint_10.complete").write_text(
        json.dumps({"bytes": mismatched.stat().st_size + 1, "step": 10}), encoding="utf-8"
    )
    wrong_step = tmp_path / "checkpoint_11"
    torch.save({**payload, "step": 11}, wrong_step)
    (tmp_path / "checkpoint_11.complete").write_text(
        json.dumps({"bytes": wrong_step.stat().st_size, "step": 99}), encoding="utf-8"
    )

    assert find_latest_checkpoint(str(tmp_path)) == str(committed)
    _, step = load_checkpoint(str(tmp_path), state, load_optimizer=False)
    assert step == 8


def test_eval_explicit_legacy_checkpoint_does_not_require_marker(tmp_path):
    state = make_state()
    checkpoint = tmp_path / "legacy_checkpoint"
    torch.save(
        {"params": state.model.state_dict(), "step": 7, "epoch": 1}, checkpoint
    )

    _, step = load_checkpoint(str(checkpoint), state, load_optimizer=False)
    assert step == 7


def test_checkpoint_rejects_partial_accumulation_state(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    state = make_state()
    state.accum_step = 1
    with pytest.raises(RuntimeError, match="accumulation window"):
        save_checkpoint(state, str(tmp_path), step=8)


def test_checkpoint_restores_process_rng_state(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    state = make_state(seed=11)
    random.seed(13)
    np.random.seed(17)
    torch.manual_seed(19)
    save_checkpoint(state, str(tmp_path), step=8)

    expected = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=state.dropout_generator))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    state.dropout_generator.manual_seed(99)

    load_checkpoint(str(tmp_path), state)
    actual = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=state.dropout_generator))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])
