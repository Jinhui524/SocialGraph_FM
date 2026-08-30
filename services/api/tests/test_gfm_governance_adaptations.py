from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.gfm_client import GfmProxyError
from app.gfm_hashing import canonical_sha256
from app.gfm_governance import GovernanceGateway
from app.gfm_governance_schemas import (
    AdaptationComparisonV2,
    AdaptationComparisonPage,
    AdaptationHandoffCreateRequest,
    AdaptationOverlayActivationRequest,
    GovernancePreviewNodeEdgeBudgets,
    GovernancePreviewQuery,
    TargetLabelSet,
    TargetLabelSetCreateRequest,
    TargetReviewPolicy,
    TargetReviewPolicyFitRequest,
    TargetReviewPolicyV2,
)
from app.gfm_governance_store import GovernanceStore
from app.main import create_app


def _binding() -> dict[str, Any]:
    return {
        "artifactId": "governance-artifact-" + "a" * 32,
        "datasetContentHash": "1" * 64,
        "graphVersionHash": "2" * 64,
        "runId": "governance-" + "b" * 32,
        "requestHash": "3" * 64,
        "resultHash": "4" * 64,
        "runArtifactHash": "5" * 64,
        "modelVersionId": "socialgraph-fm-global/" + "1" * 16,
        "modelVersionHash": "6" * 64,
        "modelStateHash": "7" * 64,
        "recipeHash": "8" * 64,
        "codeHash": "9" * 64,
        "seed": 1729,
    }


def test_target_policy_fit_request_requires_the_complete_run_identity() -> None:
    payload = {
        "schemaVersion": "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
        "targetTaskRegistrationId": "target-task-" + "1" * 32,
        "runId": "governance-" + "2" * 32,
        "resultHash": "3" * 64,
    }
    assert TargetReviewPolicyFitRequest.model_validate(payload).result_hash == "3" * 64
    for invalid in (
        {key: value for key, value in payload.items() if key != "resultHash"},
        {**payload, "runId": "governance-" + "4" * 32, "unexpected": True},
        {**payload, "targetTaskRegistrationId": "target-task-cross-lane"},
    ):
        with pytest.raises(ValidationError):
            TargetReviewPolicyFitRequest.model_validate(invalid)


def _sidecar_request() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for label, offset in (("io", 0), ("control", 2)):
        for stratum in range(4):
            for within in range(2):
                index = stratum * 32 + offset + within
                rows.append(
                    {
                        "nodeId": f"node-{index:03d}",
                        "label": label,
                        "structuralStratum": stratum,
                        "fusedDegree": index + 1,
                    }
                )
    rows.sort(key=lambda item: item["nodeId"])
    label_selection = {
        "version": "graph-fused-degree-quartile-stable-hash-v2",
        "stratification": "graph-fused-degree-rank-quartile",
        "structuralStrata": 4,
        "labelsPerClass": 8,
        "labelsPerClassPerStratum": 2,
        "scoreInputs": [],
    }
    label_document = {
        "schemaVersion": "socialgraph-fm.governance-target-label-recipe/1.1",
        "datasetId": "thailand-authorized",
        "bundleSha256": "a" * 64,
        "selectionRecipe": label_selection,
        "labels": rows,
    }
    labels_bytes = json.dumps(
        label_document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    receipt_logical: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.governance-target-package-receipt/1.1",
        "datasetId": "thailand-authorized",
        "sourceSchemaVersion": "socialgraph-fm.anonymized-posts/1.0",
        "sourceSha256": "b" * 64,
        "authorizationReference": "fixture-approval-2026-08-20",
        "bundleSha256": "a" * 64,
        "labelsSha256": hashlib.sha256(labels_bytes).hexdigest(),
        "encoder": {
            "modelId": "fixture/deterministic-encoder",
            "revision": "fixture-v1",
            "cacheSha256": "1" * 64,
            "compatibility": "dimension-only-unverified",
            "dimension": 768,
        },
        "selectionRecipe": {
            "version": "connected-structural-hash-v2",
            "nodeCount": 128,
            "requiredIo": 16,
            "requiredControls": 64,
            "minimumNonemptyModalities": 4,
            "scoreInputs": [],
            "groupRelations": {"maxGroupAccounts": 256, "totalPotentialPairBudget": 50_000},
            "fastRT": {"windowSeconds": 10, "pairBudget": 50_000, "algorithm": "sorted-sliding-window-v1"},
            "tweetSim": {"mutualTopK": 5, "cosineThreshold": 0.8, "pairBudget": 10_000},
        },
        "labelSelectionRecipe": label_selection,
        "coverage": {
            "nodeCount": 128,
            "ioCount": 32,
            "controlCount": 96,
            "nonemptyModalities": ["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"],
            "connected": True,
        },
    }
    receipt = {**receipt_logical, "receiptHash": canonical_sha256(receipt_logical)}
    sources = []
    for row in rows:
        source_hash = canonical_sha256(
            {
                "schemaVersion": label_document["schemaVersion"],
                "datasetId": label_document["datasetId"],
                "bundleSha256": label_document["bundleSha256"],
                "labelsSha256": receipt["labelsSha256"],
                "receiptHash": receipt["receiptHash"],
                **row,
            }
        )
        sources.append(
            {
                "sourceType": "imported_sidecar",
                "sourceRecordId": f"thailand-authorized:{row['nodeId']}",
                "sourceRecordHash": source_hash,
                "nodeId": row["nodeId"],
                "cohort": row["label"],
                "structuralStratum": row["structuralStratum"],
                "fusedDegree": row["fusedDegree"],
                "labelsSha256": receipt["labelsSha256"],
                "receiptHash": receipt["receiptHash"],
            }
        )
    return {
        "schemaVersion": "socialgraph-fm.governance-target-label-set/1.1",
        "runId": _binding()["runId"],
        "resultHash": _binding()["resultHash"],
        "sidecarReceipt": receipt,
        "sources": sources,
    }


def test_adaptation_overview_query_accepts_the_visible_product_ceiling() -> None:
    query = GovernancePreviewQuery.model_validate(
        {"preset": "overview", "nodeBudget": 128, "edgeBudget": 12_000}
    )
    assert (query.node_budget, query.edge_budget) == (128, 12_000)


def test_adaptation_overview_response_echoes_the_same_visible_product_ceiling() -> None:
    budgets = GovernancePreviewNodeEdgeBudgets.model_validate(
        {"nodes": 128, "edges": 12_000}
    )
    assert (budgets.nodes, budgets.edges) == (128, 12_000)


def test_sidecar_request_requires_hashed_receipt_exact_strata_and_declared_degrees() -> None:
    request = TargetLabelSetCreateRequest.model_validate(_sidecar_request())
    assert request.schema_version == "socialgraph-fm.governance-target-label-set/1.1"
    assert len(request.sources) == 16
    assert request.sidecar_receipt is not None
    for cohort in ("io", "control"):
        assert {
            stratum: sum(
                source.cohort == cohort and source.structural_stratum == stratum
                for source in request.sources
                if hasattr(source, "cohort")
            )
            for stratum in range(4)
        } == {0: 2, 1: 2, 2: 2, 3: 2}


@pytest.mark.parametrize("tamper", ["source_hash", "receipt_hash", "labels_digest", "degree", "stratum"])
def test_sidecar_request_rejects_tampered_provenance_before_proxy(tamper: str) -> None:
    payload = copy.deepcopy(_sidecar_request())
    if tamper == "source_hash":
        payload["sources"][0]["sourceRecordHash"] = "f" * 64
    elif tamper == "receipt_hash":
        payload["sidecarReceipt"]["receiptHash"] = "f" * 64
    elif tamper == "labels_digest":
        payload["sidecarReceipt"]["labelsSha256"] = "e" * 64
    elif tamper == "degree":
        payload["sources"][0]["fusedDegree"] += 1
    else:
        payload["sources"][0]["structuralStratum"] = 3
    with pytest.raises(ValidationError):
        TargetLabelSetCreateRequest.model_validate(payload)


def _label_set_payload() -> dict[str, Any]:
    binding = _binding()
    labels = [
        {
            "nodeId": f"node-{index}",
            "label": "positive" if index < 4 else "negative",
            "sourceType": "imported_sidecar",
            "sourceRecordId": f"source-{index}",
            "sourceRecordHash": f"{index + 10:064x}",
            "reviewEventHash": None,
            "binding": binding,
        }
        for index in range(8)
    ]
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.governance-target-label-set/1.0",
        "binding": binding,
        "sourceRecords": [
            {
                "sourceType": row["sourceType"],
                "sourceRecordId": row["sourceRecordId"],
                "sourceRecordHash": row["sourceRecordHash"],
                "reviewEventHash": None,
            }
            for row in labels
        ],
        "reviewEventHashes": [],
        "labels": labels,
        "conflicts": [],
        "positiveCount": 4,
        "negativeCount": 4,
    }
    payload["labelSetHash"] = canonical_sha256(payload)
    return payload


def _policy_payload(label_set_hash: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.governance-target-review-policy/1.0",
        "binding": _binding(),
        "labelSetHash": label_set_hash,
        "status": "ready",
        "selectedLambda": 0.5,
        "lambdaCandidates": [0.0, 0.25, 0.5, 1.0],
        "validationLosses": {"0": 0.7, "0.25": 0.6, "0.5": 0.5, "1": 0.55},
        "eligibleLabelCount": 8,
        "positiveCount": 4,
        "negativeCount": 4,
        "embeddingDimension": 256,
        "positiveCentroidHash": "a" * 64,
        "negativeCentroidHash": "b" * 64,
        "normalizationEpsilon": 1e-8,
        "fittingRecipe": "l2-centroids+run-zscore+loo-balanced-log-loss-v1",
        "readyPolicyHash": "c" * 64,
    }
    payload["policyHash"] = canonical_sha256(payload)
    return payload


def _sidecar_label_set_payload() -> dict[str, Any]:
    request = _sidecar_request()
    binding = _binding()
    labels = [
        {
            "nodeId": source["nodeId"],
            "label": "positive" if source["cohort"] == "io" else "negative",
            "sourceType": "imported_sidecar",
            "sourceRecordId": source["sourceRecordId"],
            "sourceRecordHash": source["sourceRecordHash"],
            "reviewEventHash": None,
            "structuralStratum": source["structuralStratum"],
            "fusedDegree": source["fusedDegree"],
            "labelsSha256": source["labelsSha256"],
            "receiptHash": source["receiptHash"],
            "binding": binding,
        }
        for source in request["sources"]
    ]
    payload: dict[str, Any] = {
        "schemaVersion": request["schemaVersion"],
        "binding": binding,
        "sidecarReceipt": request["sidecarReceipt"],
        "sourceRecords": [
            {
                "sourceType": row["sourceType"],
                "sourceRecordId": row["sourceRecordId"],
                "sourceRecordHash": row["sourceRecordHash"],
                "reviewEventHash": None,
            }
            for row in labels
        ],
        "reviewEventHashes": [],
        "labels": labels,
        "conflicts": [],
        "positiveCount": 8,
        "negativeCount": 8,
    }
    payload["labelSetHash"] = canonical_sha256(payload)
    return payload


def _comparison_payload(policy_hash: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.governance-adaptation-comparison/1.0",
        "binding": _binding(),
        "policyHash": policy_hash,
        "total": 2,
        "offset": 0,
        "limit": 50,
        "rows": [
            {
                "nodeId": "node-0",
                "baseScore": 0.9,
                "baseRank": 1,
                "adaptedReviewPriority": 0.8,
                "adaptedRank": 2,
                "rankDelta": 1,
            },
            {
                "nodeId": "node-1",
                "baseScore": 0.7,
                "baseRank": 2,
                "adaptedReviewPriority": 0.95,
                "adaptedRank": 1,
                "rankDelta": -1,
            },
        ],
        "comparisonHash": "d" * 64,
    }
    payload["pageHash"] = canonical_sha256(payload)
    return payload


def test_adaptation_contracts_reject_vectors_bad_delta_and_hash_tampering() -> None:
    # Catches raw 256-d data crossing the API boundary and unverified immutable identities.
    label_set = _label_set_payload()
    assert TargetLabelSet.model_validate(label_set).positive_count == 4
    label_set["embeddings"] = [[0.0] * 256]
    with pytest.raises(ValidationError):
        TargetLabelSet.model_validate(label_set)

    policy = _policy_payload(_label_set_payload()["labelSetHash"])
    assert TargetReviewPolicy.model_validate(policy).status == "ready"
    policy["selectedLambda"] = 1.0
    with pytest.raises(ValidationError, match="policyHash"):
        TargetReviewPolicy.model_validate(policy)

    comparison = _comparison_payload("e" * 64)
    comparison["rows"][0]["rankDelta"] = 0
    comparison["pageHash"] = canonical_sha256(
        {key: value for key, value in comparison.items() if key != "pageHash"}
    )
    with pytest.raises(ValidationError, match="rankDelta"):
        AdaptationComparisonPage.model_validate(comparison)


@pytest.mark.parametrize(
    ("status", "selected_lambda"),
    (("ready", 0.0), ("insufficient_signal", 1.0), ("ready", 0.75)),
)
def test_v2_policy_contract_rejects_status_lambda_disagreement(
    status: str, selected_lambda: float
) -> None:
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.governance-target-review-policy/2.0",
        "binding": _binding(),
        "labelSetHash": "a" * 64,
        "status": status,
        "selectedLambda": selected_lambda,
        "eligibleLabelCount": 16,
        "positiveCount": 8,
        "negativeCount": 8,
        "fittingRecipe": "l2-centroids+run-zscore+loo-balanced-log-loss-v1",
        "baseOutputsImmutable": True,
        "adaptedOutputFields": ["adaptedReviewPriority", "adaptedRank"],
    }
    payload["policyHash"] = canonical_sha256(payload)
    with pytest.raises(ValidationError, match="lambda|readiness|signal"):
        TargetReviewPolicyV2.model_validate(payload)


@pytest.mark.anyio
async def test_v2_insufficient_policy_cannot_publish_comparison_activation_or_handoff() -> None:
    """All live v2 publication boundaries fail before returning or persisting identities."""
    policy_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.governance-target-review-policy/2.0",
        "binding": _binding(),
        "labelSetHash": "a" * 64,
        "status": "insufficient_signal",
        "selectedLambda": 0.0,
        "eligibleLabelCount": 16,
        "positiveCount": 8,
        "negativeCount": 8,
        "fittingRecipe": "l2-centroids+run-zscore+loo-balanced-log-loss-v1",
        "baseOutputsImmutable": True,
        "adaptedOutputFields": ["adaptedReviewPriority", "adaptedRank"],
    }
    policy_payload["policyHash"] = canonical_sha256(policy_payload)
    policy = TargetReviewPolicyV2.model_validate(policy_payload)
    comparison_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.governance-adaptation-comparison/2.0",
        "binding": _binding(),
        "policyHash": policy.policy_hash,
        "total": 1,
        "baseOutputsImmutable": True,
        "rows": [
            {
                "nodeId": "node-0",
                "baseScore": 0.5,
                "baseRank": 1,
                "adaptedReviewPriority": 0.5,
                "adaptedRank": 1,
                "rankDelta": 0,
            }
        ],
    }
    comparison_payload["comparisonHash"] = canonical_sha256(comparison_payload)
    comparison = AdaptationComparisonV2.model_validate(comparison_payload)
    target_registration_id = "target-task-" + "1" * 32
    target_receipt_hash = "2" * 64
    source = {
        "schemaVersion": "socialgraph-fm.governance-target-label-set/2.0",
        "sourceType": "imported_sidecar",
        "targetTaskRegistrationId": target_registration_id,
        "runId": policy.binding.run_id,
        "resultHash": policy.binding.result_hash,
    }
    handoff_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.governance-adaptation-handoff/1.0",
        "targetTaskRegistrationId": target_registration_id,
        "targetReceiptHash": target_receipt_hash,
        "labelSetHash": policy.label_set_hash,
        "binding": _binding(),
        "policyHash": policy.policy_hash,
        "comparisonHash": comparison.comparison_hash,
        "decision": "pending_governance_review",
        "baseModelMutation": False,
    }
    handoff_payload["handoffHash"] = canonical_sha256(handoff_payload)
    writes: list[dict[str, Any]] = []
    remote_comparison = AsyncMock(
        return_value=comparison.model_dump(mode="json", by_alias=True)
    )
    governance = SimpleNamespace(
        get_target_label_set=lambda _label_set_hash: (
            SimpleNamespace(label_set_hash=policy.label_set_hash),
            source,
        ),
        get_target_adaptation_metadata=lambda **_kwargs: handoff_payload,
        put_target_adaptation_metadata=lambda **kwargs: writes.append(kwargs),
    )
    gateway = GovernanceGateway(
        SimpleNamespace(get_governance_adaptation_comparison=remote_comparison),
        inbox=SimpleNamespace(),
        governance=governance,
    )
    gateway.adaptation_policy = AsyncMock(return_value=policy)  # type: ignore[method-assign]
    gateway._live_target_task = lambda *_args, **_kwargs: (  # type: ignore[method-assign]
        SimpleNamespace(
            registration_id=target_registration_id,
            target_receipt=SimpleNamespace(receipt_hash=target_receipt_hash),
        ),
        SimpleNamespace(),
    )
    create_handoff = AdaptationHandoffCreateRequest.model_validate(
        {
            "schemaVersion": "socialgraph-fm.governance-adaptation-handoff/1.0",
            "targetTaskRegistrationId": target_registration_id,
            "policyHash": policy.policy_hash,
            "decision": "pending_governance_review",
        }
    )
    activate = AdaptationOverlayActivationRequest.model_validate(
        {
            "schemaVersion": "socialgraph-fm.governance-adaptation-overlay/1.0",
            "targetTaskRegistrationId": target_registration_id,
        }
    )
    operations = (
        lambda: gateway.adaptation_comparison(
            policy.binding.run_id, policy.policy_hash, offset=0, limit=500
        ),
        lambda: gateway.adaptation_comparison(
            "governance-" + "c" * 32, policy.policy_hash, offset=0, limit=500
        ),
        lambda: gateway.create_adaptation_handoff(create_handoff),
        lambda: gateway.adaptation_handoff(str(handoff_payload["handoffHash"])),
        lambda: gateway.activate_adaptation_overlay(policy.policy_hash, activate),
    )

    for operation in operations:
        with pytest.raises(GfmProxyError) as rejected:
            await operation()
        assert rejected.value.status_code == 409
        assert rejected.value.code == "GOVERNANCE_ADAPTATION_POLICY_NOT_READY"
        assert policy.policy_hash not in str(rejected.value)
        assert str(handoff_payload["handoffHash"]) not in str(rejected.value)
    assert remote_comparison.await_count == 0
    assert writes == []


def test_adaptation_metadata_store_is_immutable_audited_and_reopenable(
    tmp_path: Path,
) -> None:
    # Catches mutable/replaced API metadata and unaudited persistence.
    store = GovernanceStore(tmp_path / "governance")
    label_set = TargetLabelSet.model_validate(_label_set_payload())
    policy = TargetReviewPolicy.model_validate(_policy_payload(label_set.label_set_hash))
    store.put_adaptation_label_set(label_set)
    store.put_adaptation_policy(policy)
    store.put_adaptation_label_set(label_set)  # same immutable content is idempotent

    reopened = GovernanceStore(tmp_path / "governance")
    assert reopened.get_adaptation_label_set(label_set.label_set_hash) == label_set
    assert reopened.get_adaptation_policy(policy.policy_hash) == policy
    with sqlite3.connect(tmp_path / "governance" / "governance.sqlite3") as connection:
        rows = connection.execute(
            "SELECT kind, record_hash, payload_json, audit_hash FROM adaptation_metadata "
            "ORDER BY kind"
        ).fetchall()
    assert len(rows) == 2
    for kind, record_hash, payload_json, audit_hash in rows:
        assert audit_hash == canonical_sha256(
            {"kind": kind, "recordHash": record_hash, "payload": json.loads(payload_json)}
        )


class FakeAdaptationClient:
    def __init__(self) -> None:
        self.label_set = _sidecar_label_set_payload()
        self.policy = _policy_payload(self.label_set["labelSetHash"])
        self.created_payload: dict[str, Any] | None = None
        self.fitted_hash: str | None = None

    async def create_governance_label_set(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.created_payload = payload
        return self.label_set

    async def fit_governance_policy(self, label_set_hash: str) -> dict[str, Any]:
        self.fitted_hash = label_set_hash
        return self.policy

    async def get_governance_policy(self, policy_hash: str) -> dict[str, Any]:
        assert policy_hash == self.policy["policyHash"]
        return self.policy

    async def get_governance_adaptation_comparison(
        self, run_id: str, policy_hash: str, offset: int, limit: int
    ) -> dict[str, Any]:
        assert run_id == _binding()["runId"]
        assert policy_hash == self.policy["policyHash"]
        assert (offset, limit) == (0, 50)
        return _comparison_payload(policy_hash)


@pytest.mark.anyio
async def test_api_revalidates_current_graph_degree_and_stratum_before_proxy_or_persistence(
    tmp_path: Path,
) -> None:
    request = TargetLabelSetCreateRequest.model_validate(_sidecar_request())
    fake = FakeAdaptationClient()
    governance = GovernanceStore(tmp_path / "governance")
    receipt = SimpleNamespace(
        artifact_id=_binding()["artifactId"],
        dataset_content_hash=_binding()["datasetContentHash"],
        graph_version_hash=_binding()["graphVersionHash"],
        bundle_sha256="a" * 64,
        dataset_id="thailand-authorized",
        node_count=128,
        relation_row_count=127,
    )
    gateway = GovernanceGateway(
        fake,
        inbox=SimpleNamespace(get=lambda _artifact_id: receipt),
        governance=governance,
    )
    gateway.get_run = AsyncMock(
        return_value=SimpleNamespace(
            status="succeeded",
            run_id=_binding()["runId"],
            request_hash=_binding()["requestHash"],
            artifact_id=_binding()["artifactId"],
            dataset_content_hash=_binding()["datasetContentHash"],
            graph_version_hash=_binding()["graphVersionHash"],
            model_version_id=_binding()["modelVersionId"],
            model_version_hash=_binding()["modelVersionHash"],
            model_state_hash=_binding()["modelStateHash"],
        )
    )
    gateway.result = AsyncMock(
        return_value=SimpleNamespace(
            result_hash=_binding()["resultHash"],
            run_id=_binding()["runId"],
            request_hash=_binding()["requestHash"],
            artifact_id=_binding()["artifactId"],
            dataset_content_hash=_binding()["datasetContentHash"],
            graph_version_hash=_binding()["graphVersionHash"],
            model_version_id=_binding()["modelVersionId"],
            model_version_hash=_binding()["modelVersionHash"],
            model_state_hash=_binding()["modelStateHash"],
        )
    )
    nodes = [SimpleNamespace(id=f"node-{index:03d}", degree=index + 1) for index in range(128)]
    nodes[0] = SimpleNamespace(id="node-000", degree=999)
    gateway.run_preview = AsyncMock(
        return_value=SimpleNamespace(
            run_id=_binding()["runId"],
            result_hash=_binding()["resultHash"],
            artifact_id=_binding()["artifactId"],
            dataset_content_hash=_binding()["datasetContentHash"],
            graph_version_hash=_binding()["graphVersionHash"],
            node_count=128,
            edge_count=127,
            nodes=nodes,
        )
    )

    with pytest.raises(GfmProxyError, match="DEGREE|STRATUM|PROVENANCE"):
        await gateway.create_adaptation_label_set(request)
    assert fake.created_payload is None
    with sqlite3.connect(tmp_path / "governance" / "governance.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM adaptation_metadata").fetchone()[0] == 0


def _imported_sources(count: int = 8) -> list[dict[str, Any]]:
    return [
        {
            "sourceType": "imported_sidecar",
            "sourceRecordId": f"source-{index}",
            "sourceRecordHash": f"{index + 10:064x}",
            "nodeId": f"node-{index}",
            "cohort": "io" if index < count // 2 else "control",
        }
        for index in range(count)
    ]


def _review_sources(count: int) -> list[dict[str, Any]]:
    return [
        {
            "sourceType": "concluded_review",
            "caseId": f"case-{index:032x}",
            "eventHash": f"{index + 10:064x}",
        }
        for index in range(count)
    ]


def test_adaptation_request_accepts_256_sources_and_rejects_257() -> None:
    # Catches an API contract wider than its bounded metadata persistence envelope.
    common = {
        "schemaVersion": "socialgraph-fm.governance-target-label-set/1.0",
        "runId": _binding()["runId"],
        "resultHash": _binding()["resultHash"],
    }
    accepted = TargetLabelSetCreateRequest.model_validate(
        {**common, "sources": _review_sources(256)}
    )
    assert len(accepted.sources) == 256
    with pytest.raises(ValidationError):
        TargetLabelSetCreateRequest.model_validate(
            {**common, "sources": _review_sources(257)}
        )


@pytest.mark.anyio
async def test_adaptation_routes_proxy_only_bounded_evidence_and_persist_metadata(
    unconfigured_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Catches missing proxy routes, vector leakage, weak binding checks, and unbounded pages.
    root = tmp_path / "governance"
    settings = unconfigured_settings.model_copy(update={"gfm_governance_root": str(root)})
    fake = FakeAdaptationClient()
    async def current_context(_self: object, _run_id: str, _result_hash: str):  # type: ignore[no-untyped-def]
        status = SimpleNamespace(**{
            "run_id": _binding()["runId"], "request_hash": _binding()["requestHash"], "artifact_id": _binding()["artifactId"],
            "dataset_content_hash": _binding()["datasetContentHash"], "graph_version_hash": _binding()["graphVersionHash"],
            "model_version_id": _binding()["modelVersionId"], "model_version_hash": _binding()["modelVersionHash"], "model_state_hash": _binding()["modelStateHash"], "status": "succeeded",
        })
        result = SimpleNamespace(result_hash=_binding()["resultHash"])
        receipt = SimpleNamespace(bundle_sha256="a" * 64, dataset_id="thailand-authorized", node_count=128)
        preview = SimpleNamespace(nodes=[SimpleNamespace(id=f"node-{index:03d}", degree=index + 1) for index in range(128)], node_count=128)
        return status, result, receipt, preview
    monkeypatch.setattr(GovernanceGateway, "_current_adaptation_context", current_context)
    app = create_app(settings, gfm_governance_client=fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        created = await client.post(
            "/api/v2/gfm/governance/adaptations/label-sets",
            json=_sidecar_request(),
        )
        assert created.status_code == 201, created.text
        label_set_hash = created.json()["labelSetHash"]
        fitted = await client.post(
            f"/api/v2/gfm/governance/adaptations/label-sets/{label_set_hash}/policies"
        )
        assert fitted.status_code == 201, fitted.text
        policy_hash = fitted.json()["policyHash"]
        read = await client.get(
            f"/api/v2/gfm/governance/adaptations/policies/{policy_hash}"
        )
        assert read.status_code == 200
        comparison = await client.get(
            f"/api/v2/gfm/governance/adaptations/runs/{_binding()['runId']}"
            f"/policies/{policy_hash}/comparison?offset=0&limit=50"
        )
        assert comparison.status_code == 200, comparison.text
        too_large = await client.get(
            f"/api/v2/gfm/governance/adaptations/runs/{_binding()['runId']}"
            f"/policies/{policy_hash}/comparison?limit=501"
        )
        assert too_large.status_code == 422

    assert fake.created_payload is not None
    encoded_proxy = json.dumps(fake.created_payload)
    encoded_response = json.dumps(created.json())
    assert "embedding" not in encoded_proxy.lower()
    assert "embedding" not in encoded_response.lower()
    assert sum(row["label"] == "positive" for row in fake.created_payload["labels"]) == 8
    assert sum(row["label"] == "negative" for row in fake.created_payload["labels"]) == 8
    assert fake.fitted_hash == label_set_hash
    with sqlite3.connect(root / "governance.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM adaptation_metadata").fetchone()[0] == 2


@pytest.mark.anyio
async def test_adaptation_route_rejects_source_disagreement_before_proxy(
    unconfigured_settings: Settings, tmp_path: Path
) -> None:
    # Catches contradictory imported labels for the same node reaching the GFM process.
    settings = unconfigured_settings.model_copy(
        update={"gfm_governance_root": str(tmp_path / "governance")}
    )
    fake = FakeAdaptationClient()
    payload = copy.deepcopy(_sidecar_request())
    payload["sources"].append({ **payload["sources"][0], "sourceRecordId": "contradiction", "cohort": "control" })
    app = create_app(settings, gfm_governance_client=fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/v2/gfm/governance/adaptations/label-sets",
            json=payload,
        )
    assert response.status_code == 422
    assert fake.created_payload is None


@pytest.mark.anyio
async def test_adaptation_route_rejects_257_sources_before_proxy_or_persistence(
    unconfigured_settings: Settings, tmp_path: Path
) -> None:
    # Catches oversized label evidence reaching the GFM proxy or immutable API store.
    root = tmp_path / "governance"
    settings = unconfigured_settings.model_copy(update={"gfm_governance_root": str(root)})
    fake = FakeAdaptationClient()
    app = create_app(settings, gfm_governance_client=fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/v2/gfm/governance/adaptations/label-sets",
            json={
                "schemaVersion": "socialgraph-fm.governance-target-label-set/1.0",
                "runId": _binding()["runId"],
                "resultHash": _binding()["resultHash"],
                "sources": _review_sources(257),
            },
        )
    assert response.status_code == 422
    assert fake.created_payload is None
    with sqlite3.connect(root / "governance.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM adaptation_metadata").fetchone()[0] == 0
