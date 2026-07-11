"""Validate and evaluate small declarative research evidence contracts."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


ROLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
CHECK_OPS = frozenset({"finite", "gt", "gte", "lt", "lte", "between"})
EARLY_STOP_OPS = frozenset({"any_nonfinite", "lte", "gte", "outside"})


def _string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise ValueError(f"research_contract.{label} must be a {qualifier}list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"research_contract.{label} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"research_contract.{label} must not contain duplicates")
    return value


def _validate_role_map(value: Any, roles: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"research_contract.{label} must be a mapping")
    unknown = set(map(str, value)) - roles
    if unknown:
        raise ValueError(f"research_contract.{label} has unknown roles: {sorted(unknown)}")


def _validate_check(check: Any, roles: set[str], *, early: bool) -> None:
    label = "early_stop" if early else "terminal_checks"
    if not isinstance(check, Mapping):
        raise ValueError(f"research_contract.{label} entries must be mappings")
    op = str(check.get("op", ""))
    allowed = EARLY_STOP_OPS if early else CHECK_OPS
    if op not in allowed:
        raise ValueError(f"research_contract.{label} has unsupported op: {op!r}")
    selected = check.get("roles", [])
    if selected:
        selected_roles = set(_string_list(selected, f"{label}.roles", nonempty=True))
        unknown = selected_roles - roles
        if unknown:
            raise ValueError(f"research_contract.{label} has unknown roles: {sorted(unknown)}")
    if op == "any_nonfinite":
        _string_list(check.get("metrics"), f"{label}.metrics", nonempty=True)
        return
    if early and "metric" not in check:
        raise ValueError(f"research_contract.{label} check needs metric")
    if "metric" not in check and not ({"left", "right"} <= set(check)):
        raise ValueError(
            f"research_contract.{label} check needs metric or left/right"
        )
    if op in {"gt", "gte", "lt", "lte"} and "right" not in check:
        value = check.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"research_contract.{label} numeric value must be finite")
    if op in {"between", "outside"}:
        bounds = (check.get("min"), check.get("max"))
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in bounds):
            raise ValueError(f"research_contract.{label} bounds must be finite")
        if bounds[0] > bounds[1]:
            raise ValueError(f"research_contract.{label} min must not exceed max")


def validate_research_contract(campaign: Mapping[str, Any]) -> None:
    """Validate one optional fixed-predicate contract and its run-role binding."""
    contract = campaign.get("research_contract")
    runs = campaign.get("runs", [])
    bound_roles = [run.get("research_role") for run in runs if run.get("research_role")]
    if contract is None:
        if bound_roles:
            raise ValueError("run research_role requires a campaign research_contract")
        return
    if not isinstance(contract, Mapping) or contract.get("schema_version") != 1:
        raise ValueError("research_contract.schema_version must be 1")
    if not isinstance(contract.get("question"), str) or not contract["question"].strip():
        raise ValueError("research_contract.question must be non-empty text")
    required_roles = _string_list(
        contract.get("required_roles"), "required_roles", nonempty=True
    )
    if any(not ROLE_RE.fullmatch(role) for role in required_roles):
        raise ValueError("research_contract.required_roles contains an invalid role")
    if len(bound_roles) != len(set(bound_roles)):
        raise ValueError("campaign research_role values must be unique")
    if set(bound_roles) != set(required_roles):
        raise ValueError(
            "campaign research roles must exactly match research_contract.required_roles"
        )
    roles = set(required_roles)

    metrics = contract.get("required_metrics", {})
    if not isinstance(metrics, Mapping):
        raise ValueError("research_contract.required_metrics must be a mapping")
    _string_list(metrics.get("common", []), "required_metrics.common")
    by_role = metrics.get("by_role", {})
    _validate_role_map(by_role, roles, "required_metrics.by_role")
    for role, names in by_role.items():
        _string_list(names, f"required_metrics.by_role.{role}")

    artifacts = contract.get("required_artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("research_contract.required_artifacts must be a mapping")
    for scope, declarations in artifacts.items():
        if scope != "common" and scope not in roles:
            raise ValueError(f"research_contract.required_artifacts has unknown role: {scope}")
        if not isinstance(declarations, Mapping):
            raise ValueError("research_contract.required_artifacts entries must be mappings")
        for name, requirement in declarations.items():
            if not isinstance(name, str) or not name or not isinstance(requirement, Mapping):
                raise ValueError("research_contract artifact requirements must be named mappings")
            unexpected = set(requirement) - {
                "min_records", "min_matches", "min_nonempty_records"
            }
            if unexpected:
                raise ValueError(
                    f"research_contract artifact requirement has unsupported fields: {sorted(unexpected)}"
                )
            for field in ("min_records", "min_matches", "min_nonempty_records"):
                if field in requirement and (
                    isinstance(requirement[field], bool)
                    or not isinstance(requirement[field], int)
                    or requirement[field] < 0
                ):
                    raise ValueError(f"research_contract artifact {field} must be non-negative")

    for check in contract.get("terminal_checks", []):
        _validate_check(check, roles, early=False)
    for check in contract.get("early_stop", []):
        _validate_check(check, roles, early=True)

    checkpoint = contract.get("checkpoint", {})
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("required_on_success", False), bool
    ):
        raise ValueError("research_contract.checkpoint must declare required_on_success")
    comparison = contract.get("comparison", {})
    if not isinstance(comparison, Mapping):
        raise ValueError("research_contract.comparison must be a mapping")
    _string_list(comparison.get("match_fields", []), "comparison.match_fields")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _applies(check: Mapping[str, Any], role: str) -> bool:
    return not check.get("roles") or role in check["roles"]


def _check_result(check: Mapping[str, Any], evidence: Mapping[str, Any]) -> tuple[str, str]:
    op = str(check["op"])
    if "metric" in check:
        label = str(check["metric"])
        left = _number(evidence.get(label))
    else:
        label = f"{check['left']} {op} {check['right']}"
        left = _number(evidence.get(str(check["left"])))
    if left is None:
        return "MISSING", label
    if op == "finite":
        return "PASS", label
    right = _number(evidence.get(str(check["right"]))) if "right" in check else _number(check.get("value"))
    if op in {"gt", "gte", "lt", "lte"} and right is None:
        return "MISSING", label
    if op == "gt":
        passed = left > right
    elif op == "gte":
        passed = left >= right
    elif op == "lt":
        passed = left < right
    elif op == "lte":
        passed = left <= right
    else:
        passed = float(check["min"]) <= left <= float(check["max"])
    return ("PASS" if passed else "FAIL"), label


def _early_stop_hit(check: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    op = str(check["op"])
    if op == "any_nonfinite":
        return any(
            name in evidence and _number(evidence.get(name)) is None
            for name in check["metrics"]
        )
    value = _number(evidence.get(str(check["metric"])))
    if value is None:
        return False
    if op == "lte":
        return value <= float(check["value"])
    if op == "gte":
        return value >= float(check["value"])
    return value < float(check["min"]) or value > float(check["max"])


def evaluate_research_run(
    *, status: Mapping[str, Any], collection: Mapping[str, Any],
    contract: Mapping[str, Any], role: str,
) -> dict[str, Any]:
    """Evaluate one run without confusing absent evidence with a failed gate."""
    state = str(status.get("state", "UNKNOWN"))
    checks: list[dict[str, str]] = []
    if state in {"CREATED", "NOT_SUBMITTED", "SUBMITTING", "QUEUED", "STARTING", "RUNNING", "EVALUATING"}:
        hits = [
            check for check in contract.get("early_stop", [])
            if _applies(check, role) and _early_stop_hit(check, collection)
        ]
        if hits:
            return {
                "research_outcome": "FAIL", "research_action": "STOP_RECOMMENDED",
                "research_checks": [{"status": "FAIL", "check": str(item)} for item in hits],
            }
        return {
            "research_outcome": "PENDING", "research_action": "OBSERVE",
            "research_checks": [],
        }
    if state != "SUCCEEDED":
        return {
            "research_outcome": "INCONCLUSIVE", "research_action": "DO_NOT_EXTEND",
            "research_checks": [],
        }

    runtime_state = collection.get("runtime_state") or collection.get("state")
    if runtime_state != "SUCCEEDED":
        checks.append({"status": "MISSING", "check": "runtime_state=SUCCEEDED"})
    conflicts = collection.get("evidence_conflicts", [])
    if conflicts:
        checks.append({"status": "MISSING", "check": "unambiguous metric evidence"})

    required = list(contract.get("required_metrics", {}).get("common", []))
    required.extend(contract.get("required_metrics", {}).get("by_role", {}).get(role, []))
    for metric in required:
        status_name = "PASS" if _number(collection.get(metric)) is not None else "MISSING"
        checks.append({"status": status_name, "check": f"finite metric {metric}"})

    artifacts = collection.get("artifacts", {})
    declarations: dict[str, Any] = {}
    declarations.update(contract.get("required_artifacts", {}).get("common", {}))
    declarations.update(contract.get("required_artifacts", {}).get(role, {}))
    for name, requirement in declarations.items():
        observed = artifacts.get(name, {}) if isinstance(artifacts, Mapping) else {}
        observed_fields = {
            "min_records": "records",
            "min_matches": "matches",
            "min_nonempty_records": "nonempty_records",
        }
        passed = isinstance(observed, Mapping) and all(
            int(observed.get(observed_fields[field], 0)) >= int(minimum)
            for field, minimum in requirement.items()
        )
        checks.append({"status": "PASS" if passed else "MISSING", "check": f"artifact {name}"})

    if contract.get("checkpoint", {}).get("required_on_success"):
        checks.append({
            "status": "PASS" if collection.get("latest_completed_checkpoint") else "MISSING",
            "check": "completed checkpoint",
        })
    for declaration in contract.get("terminal_checks", []):
        if _applies(declaration, role):
            result, label = _check_result(declaration, collection)
            checks.append({"status": result, "check": label})

    if any(item["status"] == "MISSING" for item in checks):
        outcome, action = "INCONCLUSIVE", "VERIFY_RESULTS"
    elif any(item["status"] == "FAIL" for item in checks):
        outcome, action = "FAIL", "DO_NOT_EXTEND"
    else:
        outcome, action = "PASS", "WAIT_FOR_BLOCK"
    return {
        "research_outcome": outcome,
        "research_action": action,
        "research_checks": checks,
    }


def _lookup_path(record: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    value: Any = record
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return False, None
        value = value[component]
    return True, value


def evaluate_research_block(
    *, contract: Mapping[str, Any], role_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Gate extension on complete, matched, passing role evidence."""
    required = list(contract["required_roles"])
    missing = [role for role in required if role not in role_records]
    if missing:
        return {
            "block_outcome": "INCONCLUSIVE", "block_action": "WAIT_FOR_BLOCK",
            "block_missing_roles": missing, "block_mismatches": [],
        }
    outcomes = {
        role: str(role_records[role].get("research_outcome", "INCONCLUSIVE"))
        for role in required
    }
    if any(value == "FAIL" for value in outcomes.values()):
        return {
            "block_outcome": "FAIL", "block_action": "DO_NOT_EXTEND",
            "block_role_outcomes": outcomes, "block_mismatches": [],
        }
    if any(value != "PASS" for value in outcomes.values()):
        return {
            "block_outcome": "INCONCLUSIVE", "block_action": "WAIT_FOR_BLOCK",
            "block_role_outcomes": outcomes, "block_mismatches": [],
        }

    mismatches: list[dict[str, Any]] = []
    for field in contract.get("comparison", {}).get("match_fields", []):
        values: dict[str, Any] = {}
        unavailable: list[str] = []
        for role in required:
            record = role_records[role]
            found, value = _lookup_path(record.get("manifest", {}), field)
            if not found:
                found, value = _lookup_path(record.get("run", {}), field)
            if found:
                values[role] = value
            else:
                unavailable.append(role)
        if unavailable or len({repr(value) for value in values.values()}) != 1:
            mismatches.append({"field": field, "values": values, "missing": unavailable})
    if mismatches:
        return {
            "block_outcome": "INCOMPARABLE", "block_action": "DO_NOT_EXTEND",
            "block_role_outcomes": outcomes, "block_mismatches": mismatches,
        }
    return {
        "block_outcome": "PASS", "block_action": "EXTEND",
        "block_role_outcomes": outcomes, "block_mismatches": [],
    }
