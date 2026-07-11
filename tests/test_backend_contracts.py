from experiment_control.backends.base import BackendRegistry
from experiment_control.backends.slurm import parse_accounting


def test_slurm_accounting_contract_normalizes_exit_code():
    result = parse_accounting(
        "42|run|h100|COMPLETED|00:10:00|1:0\n",
        job_id="42", run_id="run", partition="h100",
    )
    assert result["state"] == "FAILED"
    assert result["exit_code"] == "1:0"


def test_backend_registry_rejects_unknown_kind():
    registry = BackendRegistry()
    try:
        registry.get("other")
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("unknown backend was accepted")
