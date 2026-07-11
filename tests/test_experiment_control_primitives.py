from pathlib import Path

import pytest

from experiment_control.artifacts import ArtifactIdentity
from experiment_control.runner import CommandResult
from experiment_control.states import FailureClass, classify_failure, normalize_slurm_state


def test_command_result_preserves_subprocess_failure_contract():
    result = CommandResult(("false",), 7, stderr="failed")
    with pytest.raises(Exception) as error:
        result.check_returncode()
    assert error.value.returncode == 7


def test_slurm_success_requires_zero_exit():
    assert normalize_slurm_state("COMPLETED", "0:0") == "SUCCEEDED"
    assert normalize_slurm_state("COMPLETED", "1:0") == "FAILED"
    assert normalize_slurm_state("OUT_OF_MEMORY", "0:125") == "FAILED"


def test_failure_classifier_does_not_hide_resource_or_model_failures():
    assert classify_failure("OUT_OF_MEMORY") is FailureClass.RESOURCE
    assert classify_failure(text="loss became NaN") is FailureClass.MODEL
    assert classify_failure(text="TLS EOF") is FailureClass.TRANSPORT


def test_artifact_marker_is_invalidated_by_content_change(tmp_path: Path):
    artifact = tmp_path / "image.sif"
    marker = tmp_path / ".verified" / "image.json"
    artifact.write_bytes(b"one")
    identity = ArtifactIdentity.from_file(artifact)
    identity.write_marker(marker)
    assert identity.matches_marker(marker)
    artifact.write_bytes(b"two")
    assert not ArtifactIdentity.from_file(artifact).matches_marker(marker)
