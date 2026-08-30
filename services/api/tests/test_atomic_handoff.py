from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.config import Settings
from app.dataset_imports import DatasetImportService, graph_fact_hash_v1
from app.dataset_schemas import (
    DatasetPreparationSpec,
    GraphDatasetHandoffRequest,
    GraphHandoffReserveRequest,
    GraphVersionTargetDomainEnvelope,
)
from app.dataset_storage import DatasetArtifactStore

from .test_dataset_imports import graph_version_handoff


def _request(service: DatasetImportService, graph_id: str = "graph-v1") -> tuple[
    GraphDatasetHandoffRequest,
    str,
    str,
]:
    envelope = GraphVersionTargetDomainEnvelope.model_validate_json(
        graph_version_handoff(graphVersionId=graph_id)
    )
    fact_hash = graph_fact_hash_v1(envelope)
    reservation = service.reserve_graph_handoff(
        GraphHandoffReserveRequest(
            graphVersionId=graph_id,
            graphFactHash=fact_hash,
        )
    )
    preparation = DatasetPreparationSpec(
        graphVersionId=graph_id,
        featureAttributes=[],
        taskKind="none",
        splitStrategy="none",
        excludedAttributes=["email", "phone"],
        governance={
            "containsPersonalData": True,
            "deidentified": True,
            "attributeAllowlist": [],
            "excludedAttributes": ["email", "phone"],
            "retention": "project",
            "userDataTrainingOptIn": False,
        },
    )
    request = GraphDatasetHandoffRequest(
        token=reservation.token,
        envelope=envelope,
        preparation=preparation,
    )
    return request, fact_hash, hashlib.sha256(reservation.token.encode()).hexdigest()


def test_handoff_database_failure_rolls_back_active_artifact_and_token(
    unconfigured_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = DatasetImportService(unconfigured_settings)
    request, fact_hash, token_hash = _request(service)

    def fail_insert(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("injected metadata failure")

    with monkeypatch.context() as context:
        context.setattr(
            DatasetArtifactStore,
            "_insert_artifact_rows",
            staticmethod(fail_insert),
        )
        with pytest.raises(sqlite3.OperationalError, match="injected metadata failure"):
            service.commit_graph_handoff(request)

    assert service.list_artifacts() == []
    assert list(service.store.artifacts_root.iterdir()) == []
    assert list(service.store.staging_root.iterdir()) == []
    with sqlite3.connect(service.store.database_path) as connection:
        token = connection.execute(
            "SELECT consumed_at FROM graph_handoff_tokens WHERE token_hash = ?",
            (token_hash,),
        ).fetchone()
        binding_count = connection.execute(
            "SELECT COUNT(*) FROM graph_dataset_bindings"
        ).fetchone()[0]
    assert token == (None,)
    assert binding_count == 0

    retried = service.commit_graph_handoff(request)
    assert retried.reused is False
    assert retried.binding.graph_fact_hash == fact_hash
    assert len(service.list_artifacts()) == 1


def test_handoff_concurrent_commit_publishes_exactly_one_artifact(
    unconfigured_settings: Settings,
) -> None:
    first_service = DatasetImportService(unconfigured_settings)
    second_service = DatasetImportService(unconfigured_settings)
    request, _, _ = _request(first_service, "concurrent-graph")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(first_service.commit_graph_handoff, request),
            executor.submit(second_service.commit_graph_handoff, request),
        ]
        results = [future.result() for future in futures]

    assert sorted(result.reused for result in results) == [False, True]
    assert results[0].binding.id == results[1].binding.id
    assert results[0].artifact.id == results[1].artifact.id
    assert len(first_service.list_artifacts()) == 1
    assert len(list(first_service.store.artifacts_root.iterdir())) == 1
    assert list(first_service.store.staging_root.iterdir()) == []


def test_reused_binding_consumes_a_fresh_one_time_token(
    unconfigured_settings: Settings,
) -> None:
    service = DatasetImportService(unconfigured_settings)
    first_request, fact_hash, _ = _request(service, "reused-graph")
    first = service.commit_graph_handoff(first_request)
    second_reservation = service.reserve_graph_handoff(
        GraphHandoffReserveRequest(
            graphVersionId="reused-graph",
            graphFactHash=fact_hash,
        )
    )
    second_request = first_request.model_copy(
        update={"token": second_reservation.token}
    )

    second = service.commit_graph_handoff(second_request)

    assert first.binding.id == second.binding.id
    assert second.reused is True
    second_token_hash = hashlib.sha256(second_reservation.token.encode()).hexdigest()
    with sqlite3.connect(service.store.database_path) as connection:
        consumed_at = connection.execute(
            "SELECT consumed_at FROM graph_handoff_tokens WHERE token_hash = ?",
            (second_token_hash,),
        ).fetchone()[0]
    assert consumed_at is not None


def test_expired_handoff_token_never_creates_or_activates_artifact(
    unconfigured_settings: Settings,
) -> None:
    service = DatasetImportService(unconfigured_settings)
    request, _, token_hash = _request(service, "expired-graph")
    expired = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(service.store.database_path) as connection:
        connection.execute(
            "UPDATE graph_handoff_tokens SET expires_at = ? WHERE token_hash = ?",
            (expired, token_hash),
        )

    with pytest.raises(HTTPException) as raised:
        service.commit_graph_handoff(request)

    assert raised.value.status_code == 409
    assert raised.value.detail["code"] == "HANDOFF_TOKEN_EXPIRED"
    assert service.list_artifacts() == []
    assert list(service.store.artifacts_root.iterdir()) == []
    assert list(service.store.staging_root.iterdir()) == []


def test_graph_fact_hash_matches_cross_runtime_unicode_golden_vector() -> None:
    envelope = GraphVersionTargetDomainEnvelope.model_validate(
        {
            "schemaVersion": "socialgraph-fm-graph/1.0",
            "graphVersionId": "graph-v1",
            "contentHash": "a" * 64,
            "buildSpecHash": "b" * 64,
            "sourceFile": "治理关系.csv",
            "directedness": "directed",
            "nodes": [
                {
                    "id": "node-b",
                    "label": "机构乙",
                    "type": "机构",
                    "attributes": {},
                },
                {
                    "id": "node-a",
                    "label": "社区甲",
                    "type": "社区",
                    "attributes": {"district": "东区"},
                },
            ],
            "edges": [
                {
                    "id": "edge-1",
                    "source": "node-a",
                    "target": "node-b",
                    "type": "协作",
                    "weight": 0.75,
                    "timestamp": "2024-08-01",
                    "directed": True,
                    "attributes": {"evidence": "公开记录"},
                }
            ],
        }
    )

    assert graph_fact_hash_v1(envelope) == (
        "3938b6dffcdbbea96d483347bc2e4578fca0c60bff14da2e1331fbe1a583f50c"
    )


def test_graph_fact_hash_is_locale_independent_and_preserves_unicode_composition() -> None:
    base = json.loads(graph_version_handoff())
    base["nodes"][0]["attributes"] = {
        "😀": "astral",
        "中": "中文",
        "é": "composed",
        "e\u0301": "decomposed",
    }
    first = GraphVersionTargetDomainEnvelope.model_validate(base)
    base["nodes"].reverse()
    base["nodes"][1]["attributes"] = dict(
        reversed(list(base["nodes"][1]["attributes"].items()))
    )
    second = GraphVersionTargetDomainEnvelope.model_validate(base)

    assert graph_fact_hash_v1(first) == graph_fact_hash_v1(second)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_graph_fact_contract_rejects_non_finite_numbers(non_finite: float) -> None:
    value = json.loads(graph_version_handoff())
    value["nodes"][0]["attributes"]["unsafe"] = non_finite

    with pytest.raises(ValidationError, match="NaN 或无穷值"):
        GraphVersionTargetDomainEnvelope.model_validate(value)
