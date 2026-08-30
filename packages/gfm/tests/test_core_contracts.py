import json
from pathlib import Path
import tomllib

import pytest
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core_contracts import (
    load_core_readiness,
    load_core_task_contract,
)


PROJECT = Path(__file__).resolve().parents[1]


def _payload(name: str) -> dict:
    return json.loads((PROJECT / "contracts" / name).read_text(encoding="utf-8"))


def test_core_task_contract_is_strict_hash_bound_and_safety_disabled():
    contract = load_core_task_contract()
    assert {task.task_id for task in contract.tasks} == {
        "core.community_resilience_review",
        "core.risk_and_trust_review",
        "core.collaboration_completion",
    }
    assert all(
        task.enabled is False and task.human_review_required is True for task in contract.tasks
    )
    assert contract.model_or_serving_readiness_implied is False

    forged = _payload("core-task-contract.json")
    forged["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        type(contract).model_validate(forged)
    forged = _payload("core-task-contract.json")
    forged["tasks"][0]["enabled"] = True
    with pytest.raises(ValidationError):
        type(contract).model_validate(forged)
    forged = _payload("core-task-contract.json")
    forged["tasks"][0]["description"] = "tampered"
    with pytest.raises(ValueError, match="contentHash"):
        type(contract).model_validate(forged)


def _rehash_readiness(payload: dict) -> dict:
    payload["contentHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "contentHash"}
    )
    return payload


def test_core_readiness_v2_is_strict_hash_bound_generation_zero_evidence():
    readiness = load_core_readiness()
    assert readiness.schema_version == "socialgraph-fm.core-readiness/2.0"
    assert readiness.identity.control_generation == 0
    assert readiness.identity.registry_generation == 0
    assert readiness.gates.corpus_ready.ready is False
    assert readiness.gates.model_validated.ready is False
    assert readiness.gates.accepted.ready is False
    assert readiness.gates.core_serving_ready.ready is False
    assert all(value is None for value in readiness.evidence.model_dump(mode="python").values())

    forged = _payload("core-readiness.json")
    forged["gates"]["corpusReady"]["ready"] = "false"
    _rehash_readiness(forged)
    with pytest.raises(ValidationError, match="boolean"):
        type(readiness).model_validate(forged)
    forged = _payload("core-readiness.json")
    forged["gates"]["unexpected"] = {"ready": False, "reason": "forged"}
    with pytest.raises(ValidationError, match="Extra inputs"):
        type(readiness).model_validate(forged)


def test_core_readiness_v2_rejects_gate_escalation_without_prior_evidence():
    readiness = load_core_readiness()
    forged = _payload("core-readiness.json")
    forged["gates"]["accepted"]["ready"] = True
    forged["gates"]["accepted"]["reasonCode"] = "ACCEPTED_CANDIDATE_VERIFIED"
    forged["evidence"]["acceptedCandidateHash"] = "a" * 64
    _rehash_readiness(forged)

    with pytest.raises(ValueError, match="accepted readiness requires model validation"):
        type(readiness).model_validate(forged)

    forged = _payload("core-readiness.json")
    forged["gates"]["coreServingReady"]["ready"] = True
    forged["gates"]["coreServingReady"]["reasonCode"] = "LIVE_SERVING_EVIDENCE_VERIFIED"
    forged["evidence"]["servingSmokeHash"] = "b" * 64
    forged["evidence"]["servingModelHash"] = "c" * 64
    _rehash_readiness(forged)
    with pytest.raises(ValueError, match="serving readiness requires accepted"):
        type(readiness).model_validate(forged)


def test_core_readiness_v2_is_immutable_after_validation():
    readiness = load_core_readiness()

    with pytest.raises(ValidationError, match="frozen"):
        readiness.gates.corpus_ready.ready = True


def test_core_readiness_v2_rejects_unregistered_reason_codes():
    readiness = load_core_readiness()
    forged = _payload("core-readiness.json")
    forged["gates"]["corpusReady"]["reasonCode"] = "CALLER_SAYS_READY"
    _rehash_readiness(forged)

    with pytest.raises(ValidationError, match="reasonCode"):
        type(readiness).model_validate(forged)


def test_core_readiness_v2_rejects_reason_that_contradicts_gate_state():
    readiness = load_core_readiness()
    forged = _payload("core-readiness.json")
    forged["gates"]["corpusReady"]["reasonCode"] = "FORMAL_PREFLIGHT_READY"
    _rehash_readiness(forged)

    with pytest.raises(ValueError, match="reason code contradicts"):
        type(readiness).model_validate(forged)


def test_hash_bound_contracts_are_available_to_installed_wheel_consumers():
    pyproject = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include["contracts/core-readiness.json"] == (
        "socialgraph_gfm/resources/core-readiness.json"
    )
