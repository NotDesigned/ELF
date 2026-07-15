import json

import pytest

from elf_experiments.terminal_evidence import PREFIX, encode_terminal_evidence


def test_terminal_evidence_is_one_compact_identity_bound_json_line():
    line = encode_terminal_evidence({
        "run_id": "run-a",
        "attempt_id": "attempt-001",
        "image_id": "sha256:" + "a" * 64,
        "state": "SUCCEEDED",
        "train_loss": 1.25,
        "artifacts": {"train_metrics": {"records": 3}},
    })

    assert line.startswith(PREFIX)
    assert "\n" not in line
    payload = json.loads(line[len(PREFIX):])
    assert payload["run_id"] == "run-a"
    assert payload["attempt_id"] == "attempt-001"
    assert payload["state"] == "SUCCEEDED"
    assert payload["artifacts"]["train_metrics"]["records"] == 3


def test_terminal_evidence_rejects_nonfinite_metrics():
    with pytest.raises(ValueError, match="Out of range float values"):
        encode_terminal_evidence({"train_loss": float("nan")})
