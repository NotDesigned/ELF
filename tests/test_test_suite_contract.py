from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
DEPLOYMENT_LITERAL_ALLOWLIST = {
    # These files explicitly verify backend or checked-in configuration/docs.
    "backend_fixtures.py",
    "test_experiment_backend_adapters.py",
    "test_experiment_campaign.py",
    "test_documentation_contract.py",
}
FORBIDDEN_DEPLOYMENT_LITERALS = (
    "wyd-",
    "/data/liangluocheng",
    "/datapool/liangluocheng",
)


def test_non_backend_tests_do_not_hardcode_deployment_topology():
    violations: list[str] = []
    for path in sorted(TEST_ROOT.glob("*.py")):
        if path.name in DEPLOYMENT_LITERAL_ALLOWLIST or path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        for literal in FORBIDDEN_DEPLOYMENT_LITERALS:
            if literal in text:
                violations.append(f"{path.name}: {literal}")
    assert violations == []
