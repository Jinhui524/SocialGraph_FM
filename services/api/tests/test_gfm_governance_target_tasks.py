from __future__ import annotations

import sqlite3
import hashlib
import io
import json
import zipfile
from types import MethodType, SimpleNamespace
from pathlib import Path

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError

from app.gfm_governance_schemas import (
    AdaptationLabelSetCreateRequestV2,
    ReviewCollectionCreateRequest,
)
from app.gfm_governance_store import GovernanceStore
from app.gfm_hashing import canonical_sha256
from app.gfm_client import GfmProxyError
from app.gfm_governance_artifacts import inspect_governance_bundle
from app.gfm_governance_target_tasks import inspect_target_task_bundle
from app.main import create_app
from .test_gfm_governance_api import _bundle

LOCAL_GOVERNANCE_ROOT = (
    Path(__file__).resolve().parents[3] / "var" / "governance" / "adaptation-inputs"
)


def _hash(token: str) -> str:
    return (token * 64)[:64]


def _assert_stale(response: httpx.Response) -> None:
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == (
        "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH"
    )
    assert "policyHash" not in response.text


def _review_collection() -> dict[str, object]:
    return {
        "schemaVersion": "socialgraph-fm.governance-review-collection/1.0",
        "idempotencyKey": "target-b-initial-review",
        "targetTaskRegistrationId": "target-task-" + "1" * 32,
        "runId": "governance-" + "2" * 32,
        "resultHash": "3" * 64,
        "title": "Initial few-shot collection",
        "description": "Review without creating labels.",
        "items": [
            {"targetType": "node", "targetId": f"node-{index}", "note": ""}
            for index in range(8)
        ],
    }


def test_v2_label_creation_is_a_top_level_discriminated_union() -> None:
    adapter = TypeAdapter(AdaptationLabelSetCreateRequestV2)
    common = {
        "schemaVersion": "socialgraph-fm.governance-target-label-set/2.0",
        "targetTaskRegistrationId": "target-task-" + "1" * 32,
        "runId": "governance-" + "2" * 32,
        "resultHash": "3" * 64,
    }
    imported = adapter.validate_python({**common, "sourceType": "imported_sidecar"})
    concluded = adapter.validate_python(
        {
            **common,
            "sourceType": "concluded_review",
            "reviews": [
                {"caseId": f"case-{index:032x}", "eventHash": f"{index + 4:064x}"}
                for index in range(8)
            ],
        }
    )
    assert imported.source_type == "imported_sidecar"
    assert concluded.source_type == "concluded_review"
    with pytest.raises(ValidationError):
        adapter.validate_python({**common, "sourceType": "unknown"})


def test_v2_label_creation_openapi_publishes_the_source_type_discriminator(
    unconfigured_settings,
) -> None:
    app = create_app(unconfigured_settings)
    request_schema = app.openapi()["paths"][
        "/api/v2/gfm/governance/adaptations/label-sets"
    ]["post"]["requestBody"]["content"]["application/json"]["schema"]
    discriminated = [
        option
        for option in request_schema["anyOf"]
        if "discriminator" in option
    ]

    assert discriminated == [
        {
            "discriminator": {
                "mapping": {
                    "concluded_review": "#/components/schemas/ConcludedReviewLabelSetCreateRequestV2",
                    "imported_sidecar": "#/components/schemas/ImportedSidecarLabelSetCreateRequestV2",
                },
                "propertyName": "sourceType",
            },
            "oneOf": [
                {"$ref": "#/components/schemas/ImportedSidecarLabelSetCreateRequestV2"},
                {"$ref": "#/components/schemas/ConcludedReviewLabelSetCreateRequestV2"},
            ],
        }
    ]


def test_review_collection_store_is_atomic_idempotent_and_creates_no_review_signal(
    tmp_path: Path,
) -> None:
    store = GovernanceStore(tmp_path / "governance")
    request = ReviewCollectionCreateRequest.model_validate(_review_collection())

    first = store.create_review_collection(request)
    second = store.create_review_collection(request)

    assert first == second
    assert first.result_hash == request.result_hash
    assert first.case.state == "active"
    assert len(first.case.items) == 8
    assert first.case.review_events == ()
    assert first.case.current_decisions == {}
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM governance_cases").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM case_items").fetchone()[0] == 8
        assert connection.execute("SELECT COUNT(*) FROM case_state_events").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM review_events").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM adaptation_metadata").fetchone()[0] == 0
        assert connection.execute("SELECT result_hash FROM review_collections").fetchone()[0] == request.result_hash

    conflicting = request.model_copy(update={"title": "Different title"})
    with pytest.raises(Exception, match="IDEMPOTENCY"):
        store.create_review_collection(conflicting)


def _target_task_bundle() -> tuple[bytes, bytes]:
    inference = _bundle()
    _, inspected = inspect_governance_bundle(
        inference, clean_self_loops=False, max_expanded_bytes=1024 * 1024
    )
    receipt: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-target-domain-receipt/2.0",
        "taskId": "generic-target",
        "countryId": "fixture-country",
        "sourceContentHash": "4" * 64,
        "sourceManifestSha256": "5" * 64,
        "graphPopulation": "full",
        "graphPopulationMaskSha256": None,
        "labelEligibility": "none",
        "labelEligibilityMaskSha256": None,
        "inferenceSha256": hashlib.sha256(inference).hexdigest(),
        "nodeSetSha256": canonical_sha256(["a", "b"]),
        "nodeCount": 2,
        "fusedEdgeCount": 1,
        "modalities": ["coRT"],
        "connected": True,
        "selectionRecipe": {"version": "fixture-v1", "scoreInputs": []},
    }
    receipt["receiptHash"] = canonical_sha256(receipt)
    receipt_bytes = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    task = {
        "schemaVersion": "socialgraph-fm.governance-target-task-bundle/1.0",
        "taskId": "generic-target",
        "displayName": "Generic target",
        "mode": "zero_shot",
        "nodeCount": 2,
        "fusedEdgeCount": 1,
        "modalities": ["coRT"],
        "inference": {
            "name": "inference.zip",
            "sha256": hashlib.sha256(inference).hexdigest(),
            "bytes": len(inference),
        },
        "targetReceipt": {
            "name": "target-receipt.json",
            "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "bytes": len(receipt_bytes),
        },
        "labels": None,
        "labelReceipt": None,
    }
    task_bytes = json.dumps(task, sort_keys=True, separators=(",", ":")).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("task.json", task_bytes)
        archive.writestr("inference.zip", inference)
        archive.writestr("target-receipt.json", receipt_bytes)
    return output.getvalue(), inference


class _TargetTaskClient:
    async def validate_governance_artifact(
        self, artifact_id: str, payload: dict[str, object]
    ) -> dict[str, object]:
        response = {
            "schemaVersion": "socialgraph-fm.gfm-governance/2.0",
            "artifactId": artifact_id,
            "datasetContentHash": payload["datasetContentHash"],
            "graphVersionHash": payload["graphVersionHash"],
            "nodeCount": 2,
            "relationRowCount": 1,
            "selfLoopsRemoved": 0,
            "modalities": ["coRT"],
            "createdAt": "2026-08-21T00:00:00Z",
            "compatibility": "compatible",
        }
        response["artifactHash"] = canonical_sha256(response)
        return response


@pytest.mark.anyio
async def test_target_task_upload_registers_inner_artifact_and_is_immutable(
    tmp_path: Path, unconfigured_settings
) -> None:
    root = tmp_path / "governance"
    settings = unconfigured_settings.model_copy(
        update={"gfm_governance_root": str(root)}
    )
    outer, inference = _target_task_bundle()
    app = create_app(settings, gfm_governance_client=_TargetTaskClient())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        created = await client.post(
            "/api/v2/gfm/governance/target-tasks",
            files={"file": ("target.sgtask.zip", outer, "application/zip")},
        )
        assert created.status_code == 201, created.text
        registration = created.json()
        assert registration["task"]["taskId"] == "generic-target"
        assert registration["artifact"]["bundleSha256"] == hashlib.sha256(inference).hexdigest()
        repeated = await client.post(
            "/api/v2/gfm/governance/target-tasks",
            files={"file": ("target.sgtask.zip", outer, "application/zip")},
        )
        assert repeated.json() == registration
        read = await client.get(
            f"/api/v2/gfm/governance/target-tasks/{registration['registrationId']}"
        )
        assert read.json() == registration

    with sqlite3.connect(root / "governance.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM target_task_registrations").fetchone()[0] == 1
    assert (root / "target-tasks" / f"{registration['registrationId']}.sgtask.zip").read_bytes() == outer


@pytest.mark.anyio
async def test_regenerated_real_governance_b_passes_api_registration(
    tmp_path: Path, unconfigured_settings
) -> None:
    catalog_root = LOCAL_GOVERNANCE_ROOT
    catalog_path = catalog_root / "governance-target-tasks.catalog.json"
    if not catalog_path.is_file():
        pytest.skip("regenerated governance B is unavailable")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    target_b = next(
        entry for entry in catalog["targets"] if entry["role"] == "few_shot"
    )
    source = (catalog_root / target_b["path"]).resolve(strict=True)
    assert source.is_relative_to(catalog_root.resolve(strict=True))
    assert source.name == "target-domain-b-few.sgtask.zip"
    outer = source.read_bytes()
    inspected = inspect_target_task_bundle(outer, max_expanded_bytes=1024**3)
    _, inner = inspect_governance_bundle(
        inspected.inference,
        clean_self_loops=False,
        max_expanded_bytes=1024**3,
    )

    class RealBundleClient:
        async def validate_governance_artifact(self, artifact_id, payload):
            response = {
                "schemaVersion": "socialgraph-fm.gfm-governance/2.0",
                "artifactId": artifact_id,
                "datasetContentHash": payload["datasetContentHash"],
                "graphVersionHash": payload["graphVersionHash"],
                "nodeCount": inner["nodeCount"],
                "relationRowCount": inner["relationRowCount"],
                "selfLoopsRemoved": inner["selfLoopsRemoved"],
                "modalities": inner["modalities"],
                "createdAt": "2026-08-21T00:00:00Z",
                "compatibility": "compatible",
            }
            response["artifactHash"] = canonical_sha256(response)
            return response

    root = tmp_path / "governance"
    settings = unconfigured_settings.model_copy(
        update={"gfm_governance_root": str(root)}
    )
    app = create_app(settings, gfm_governance_client=RealBundleClient())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/v2/gfm/governance/target-tasks",
            files={"file": (source.name, outer, "application/zip")},
        )
        assert response.status_code == 201, response.text
        assert response.json()["task"]["displayName"] == "目标域网络 B"
        assert response.json()["artifact"]["displayName"] == "匿名目标数据源 B"
        visible_names = " ".join(
            (
                response.json()["task"]["displayName"],
                response.json()["artifact"]["displayName"],
            )
        )
        assert all(value not in visible_names for value in ("Cuba", "UAE", "Thailand"))
        assert response.json()["targetReceipt"]["countryId"] == "UAE"
    assert response.json()["task"]["taskId"] == "target-b"
    assert response.json()["labels"]["positiveCount"] == 8
    assert response.json()["labels"]["negativeCount"] == 8


def _few_shot_target_task_bundle() -> tuple[bytes, bytes]:
    node_ids = [f"node-{index}" for index in range(8)]
    nodes = ("node_id,display_name\n" + "".join(f"{node},{node}\n" for node in node_ids)).encode()
    relations = (
        "source,target,modality,weight\n"
        + "".join(
            f"{node_ids[index]},{node_ids[(index + 1) % 8]},coRT,1\n"
            for index in range(8)
        )
    ).encode()
    import numpy as np

    features_stream = io.BytesIO()
    np.savez_compressed(
        features_stream,
        node_ids=np.asarray(node_ids),
        text_features=np.arange(8 * 768, dtype=np.float32).reshape(8, 768),
    )
    features = features_stream.getvalue()
    files = {
        name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
        for name, value in {
            "nodes.csv": nodes,
            "relations.csv": relations,
            "features.npz": features,
        }.items()
    }
    manifest = {
        "schemaVersion": "socialgraph-fm.governance-input/2.0",
        "datasetId": "fixture:few-shot",
        "displayName": "Few-shot ring",
        "nodeCount": 8,
        "relationRowCount": 8,
        "featureDimension": 768,
        "modalities": ["coRT"],
        "files": files,
    }
    inference_stream = io.BytesIO()
    with zipfile.ZipFile(inference_stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True, separators=(",", ":")))
        archive.writestr("nodes.csv", nodes)
        archive.writestr("relations.csv", relations)
        archive.writestr("features.npz", features)
    inference = inference_stream.getvalue()
    receipt: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-target-domain-receipt/2.0",
        "taskId": "few-shot-ring",
        "countryId": "fixture-country",
        "sourceContentHash": "4" * 64,
        "sourceManifestSha256": "5" * 64,
        "graphPopulation": "full",
        "graphPopulationMaskSha256": None,
        "labelEligibility": "fixture",
        "labelEligibilityMaskSha256": "6" * 64,
        "inferenceSha256": hashlib.sha256(inference).hexdigest(),
        "nodeSetSha256": canonical_sha256(sorted(node_ids)),
        "nodeCount": 8,
        "fusedEdgeCount": 8,
        "modalities": ["coRT"],
        "connected": True,
        "selectionRecipe": {"version": "fixture-v1", "scoreInputs": []},
    }
    receipt["receiptHash"] = canonical_sha256(receipt)
    labels: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-target-label-set/2.0",
        "taskId": "few-shot-ring",
        "inferenceSha256": hashlib.sha256(inference).hexdigest(),
        "labels": [
            {
                "nodeId": node_id,
                "label": "positive" if index < 4 else "negative",
                "structuralStratum": index // 2,
                "fusedDegree": 2,
            }
            for index, node_id in enumerate(node_ids)
        ],
        "positiveCount": 4,
        "negativeCount": 4,
    }
    labels["labelSetHash"] = canonical_sha256(labels)
    labels_bytes = json.dumps(labels, sort_keys=True, separators=(",", ":")).encode()
    label_receipt: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-target-label-receipt/2.0",
        "taskId": "few-shot-ring",
        "targetReceiptHash": receipt["receiptHash"],
        "labelsSha256": hashlib.sha256(labels_bytes).hexdigest(),
        "sourceLabelsSha256": "7" * 64,
        "eligibilityMaskSha256": "6" * 64,
        "eligibleNodeIds": node_ids,
        "selectionRecipe": {"version": "fixture-v1", "scoreInputs": []},
    }
    label_receipt["receiptHash"] = canonical_sha256(label_receipt)
    entry_bytes = {
        "inference.zip": inference,
        "target-receipt.json": json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(),
        "labels.json": labels_bytes,
        "label-receipt.json": json.dumps(label_receipt, sort_keys=True, separators=(",", ":")).encode(),
    }
    task = {
        "schemaVersion": "socialgraph-fm.governance-target-task-bundle/1.0",
        "taskId": "few-shot-ring",
        "displayName": "Few-shot ring",
        "mode": "few_shot",
        "nodeCount": 8,
        "fusedEdgeCount": 8,
        "modalities": ["coRT"],
        "inference": {"name": "inference.zip", "sha256": hashlib.sha256(inference).hexdigest(), "bytes": len(inference)},
        "targetReceipt": {"name": "target-receipt.json", "sha256": hashlib.sha256(entry_bytes["target-receipt.json"]).hexdigest(), "bytes": len(entry_bytes["target-receipt.json"])},
        "labels": {"name": "labels.json", "sha256": hashlib.sha256(labels_bytes).hexdigest(), "bytes": len(labels_bytes)},
        "labelReceipt": {"name": "label-receipt.json", "sha256": hashlib.sha256(entry_bytes["label-receipt.json"]).hexdigest(), "bytes": len(entry_bytes["label-receipt.json"])},
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("task.json", json.dumps(task, sort_keys=True, separators=(",", ":")))
        for name, value in entry_bytes.items():
            archive.writestr(name, value)
    return output.getvalue(), inference


def _rebind_false_label_facts(outer: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(outer)) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    labels = json.loads(entries["labels.json"])
    labels["labels"][0]["fusedDegree"] += 1
    labels["labels"][0]["structuralStratum"] = 3
    labels.pop("labelSetHash")
    labels["labelSetHash"] = canonical_sha256(labels)
    entries["labels.json"] = json.dumps(
        labels, sort_keys=True, separators=(",", ":")
    ).encode()
    label_receipt = json.loads(entries["label-receipt.json"])
    label_receipt["labelsSha256"] = hashlib.sha256(entries["labels.json"]).hexdigest()
    label_receipt.pop("receiptHash")
    label_receipt["receiptHash"] = canonical_sha256(label_receipt)
    entries["label-receipt.json"] = json.dumps(
        label_receipt, sort_keys=True, separators=(",", ":")
    ).encode()
    task = json.loads(entries["task.json"])
    for field, name in (("labels", "labels.json"), ("labelReceipt", "label-receipt.json")):
        task[field]["sha256"] = hashlib.sha256(entries[name]).hexdigest()
        task[field]["bytes"] = len(entries[name])
    entries["task.json"] = json.dumps(
        task, sort_keys=True, separators=(",", ":")
    ).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "task.json",
            "inference.zip",
            "target-receipt.json",
            "labels.json",
            "label-receipt.json",
        ):
            archive.writestr(name, entries[name])
    return output.getvalue()


def test_rehashed_false_label_degree_and_stratum_are_rejected() -> None:
    outer, _ = _few_shot_target_task_bundle()
    rebound = _rebind_false_label_facts(outer)
    with pytest.raises(GfmProxyError) as raised:
        inspect_target_task_bundle(rebound, max_expanded_bytes=1024 * 1024)
    assert raised.value.status_code == 400


class _V2AdaptationClient(_TargetTaskClient):
    def __init__(self) -> None:
        self.binding: dict[str, object] | None = None
        self.label_set: dict[str, object] | None = None
        self.policy: dict[str, object] | None = None
        self.comparison: dict[str, object] | None = None
        self.stale_policy = False
        self.fit_request: dict[str, object] | None = None

    async def validate_governance_artifact(self, artifact_id, payload):
        response = {
            "schemaVersion": "socialgraph-fm.gfm-governance/2.0", "artifactId": artifact_id,
            "datasetContentHash": payload["datasetContentHash"], "graphVersionHash": payload["graphVersionHash"],
            "nodeCount": 8, "relationRowCount": 8, "selfLoopsRemoved": 0, "modalities": ["coRT"],
            "createdAt": "2026-08-21T00:00:00Z", "compatibility": "compatible",
        }
        response["artifactHash"] = canonical_sha256(response)
        return response

    async def create_governance_label_set(self, payload):
        logical = {key: payload[key] for key in ("schemaVersion", "taskId", "inferenceSha256", "labels")}
        logical["positiveCount"] = 4
        logical["negativeCount"] = 4
        logical["labelSetHash"] = canonical_sha256(logical)
        self.label_set = logical
        return logical

    async def fit_governance_policy(
        self, label_set_hash: str, payload: dict[str, object] | None = None
    ):
        assert self.binding is not None
        self.fit_request = payload
        logical = {
            "schemaVersion": "socialgraph-fm.governance-target-review-policy/2.0",
            "binding": self.binding,
            "labelSetHash": label_set_hash,
            "status": "ready",
            "selectedLambda": 0.5,
            "eligibleLabelCount": 8,
            "positiveCount": 4,
            "negativeCount": 4,
            "fittingRecipe": "l2-centroids+run-zscore+loo-balanced-log-loss-v1",
            "baseOutputsImmutable": True,
            "adaptedOutputFields": ["adaptedReviewPriority", "adaptedRank"],
        }
        logical["policyHash"] = canonical_sha256(logical)
        self.policy = logical
        rows = [
            {"nodeId": f"node-{index}", "baseScore": 0.1 + index / 10, "baseRank": index + 1, "adaptedReviewPriority": 0.9 - index / 10, "adaptedRank": 8 - index, "rankDelta": (8 - index) - (index + 1)}
            for index in range(8)
        ]
        comparison = {"schemaVersion": "socialgraph-fm.governance-adaptation-comparison/2.0", "binding": self.binding, "policyHash": logical["policyHash"], "total": 8, "baseOutputsImmutable": True, "rows": rows}
        comparison["comparisonHash"] = canonical_sha256(comparison)
        self.comparison = comparison
        return logical

    async def get_governance_policy(self, policy_hash: str):
        if self.stale_policy:
            raise GfmProxyError(409, "GFM_GOVERNANCE_CONFLICT")
        assert self.policy is not None and self.policy["policyHash"] == policy_hash
        return self.policy

    async def get_governance_adaptation_comparison(self, run_id, policy_hash, offset, limit):
        assert self.comparison is not None
        return self.comparison


class _RepeatedV2AdaptationClient(_V2AdaptationClient):
    def __init__(self) -> None:
        super().__init__()
        self.bindings: dict[str, dict[str, object]] = {}
        self.fit_requests: list[dict[str, object]] = []

    async def fit_governance_policy(
        self, label_set_hash: str, payload: dict[str, object] | None = None
    ):
        assert payload is not None
        binding = self.bindings[str(payload["runId"])]
        assert payload == {
            "schemaVersion": "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
            "targetTaskRegistrationId": payload["targetTaskRegistrationId"],
            "runId": binding["runId"],
            "resultHash": binding["resultHash"],
        }
        self.fit_requests.append(payload)
        logical = {
            "schemaVersion": "socialgraph-fm.governance-target-review-policy/2.0",
            "binding": binding,
            "labelSetHash": label_set_hash,
            "status": "ready",
            "selectedLambda": 0.5,
            "eligibleLabelCount": 8,
            "positiveCount": 4,
            "negativeCount": 4,
            "fittingRecipe": "l2-centroids+run-zscore+loo-balanced-log-loss-v1",
            "baseOutputsImmutable": True,
            "adaptedOutputFields": ["adaptedReviewPriority", "adaptedRank"],
        }
        logical["policyHash"] = canonical_sha256(logical)
        return logical


@pytest.mark.anyio
async def test_v2_import_fit_read_review_collection_handoff_and_stale_fail_closed(
    tmp_path: Path, unconfigured_settings
) -> None:
    root = tmp_path / "governance"
    settings = unconfigured_settings.model_copy(update={"gfm_governance_root": str(root)})
    outer, _ = _few_shot_target_task_bundle()
    fake = _V2AdaptationClient()
    app = create_app(settings, gfm_governance_client=fake)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        registered = (await client.post("/api/v2/gfm/governance/target-tasks", files={"file": ("few.sgtask.zip", outer, "application/zip")})).json()
        binding = {
            "artifactId": registered["artifact"]["artifactId"], "datasetContentHash": registered["artifact"]["datasetContentHash"], "graphVersionHash": registered["artifact"]["graphVersionHash"],
            "runId": "governance-" + "8" * 32, "requestHash": "9" * 64, "resultHash": "a" * 64, "runArtifactHash": "b" * 64,
            "modelVersionId": "socialgraph-fm-global/" + "1" * 16, "modelVersionHash": "c" * 64, "modelStateHash": "d" * 64, "recipeHash": "e" * 64, "codeHash": "f" * 64, "seed": 1729,
        }
        fake.binding = binding
        gateway = app.state.gfm_governance_gateway

        async def context(_self, run_id, result_hash):
            status = SimpleNamespace(status="succeeded", run_id=run_id, request_hash=binding["requestHash"], artifact_id=binding["artifactId"], dataset_content_hash=binding["datasetContentHash"], graph_version_hash=binding["graphVersionHash"], model_version_id=binding["modelVersionId"], model_version_hash=binding["modelVersionHash"], model_state_hash=binding["modelStateHash"])
            result = SimpleNamespace(result_hash=result_hash)
            return status, result, gateway.inbox.get(binding["artifactId"]), None

        gateway._current_adaptation_context = MethodType(context, gateway)
        label_request = {"schemaVersion": "socialgraph-fm.governance-target-label-set/2.0", "sourceType": "imported_sidecar", "targetTaskRegistrationId": registered["registrationId"], "runId": binding["runId"], "resultHash": binding["resultHash"]}
        created = await client.post("/api/v2/gfm/governance/adaptations/label-sets", json=label_request)
        assert created.status_code == 201, created.text
        fitted = await client.post(f"/api/v2/gfm/governance/adaptations/label-sets/{created.json()['labelSetHash']}/policies")
        assert fitted.status_code == 201, fitted.text
        policy_hash = fitted.json()["policyHash"]
        comparison = await client.get(f"/api/v2/gfm/governance/adaptations/runs/{binding['runId']}/policies/{policy_hash}/comparison")
        assert comparison.status_code == 200, comparison.text
        collection_payload = {"schemaVersion": "socialgraph-fm.governance-review-collection/1.0", "idempotencyKey": "initial", "targetTaskRegistrationId": registered["registrationId"], "runId": binding["runId"], "resultHash": binding["resultHash"], "title": "Initial", "description": "", "items": [{"targetType": "node", "targetId": f"node-{index}", "note": ""} for index in range(8)]}
        collection = await client.post("/api/v2/gfm/governance/adaptations/review-collections", json=collection_payload)
        assert collection.status_code == 201 and collection.json()["case"]["reviewEvents"] == []
        assert collection.json()["resultHash"] == binding["resultHash"]
        repeated_collection = await client.post("/api/v2/gfm/governance/adaptations/review-collections", json=collection_payload)
        assert repeated_collection.json() == collection.json()
        case_id = collection.json()["case"]["caseId"]
        review_hashes = []
        for index in range(8):
            reviewed = await client.post(
                f"/api/v2/gfm/governance/cases/{case_id}/review-events",
                json={"schemaVersion": "socialgraph-fm.gfm-governance/2.0", "targetType": "node", "targetId": f"node-{index}", "decision": "rejected" if index < 4 else "confirmed", "reason": "fixture conclusion", "actor": "analyst"},
            )
            assert reviewed.status_code == 201, reviewed.text
            review_hashes.append(reviewed.json()["reviewEvents"][-1]["eventHash"])
        concluded = await client.post(
            f"/api/v2/gfm/governance/cases/{case_id}/transitions",
            json={"schemaVersion": "socialgraph-fm.gfm-governance/2.0", "state": "concluded", "reason": "review complete"},
        )
        assert concluded.status_code == 200
        review_label_set = await client.post(
            "/api/v2/gfm/governance/adaptations/label-sets",
            json={"schemaVersion": "socialgraph-fm.governance-target-label-set/2.0", "sourceType": "concluded_review", "targetTaskRegistrationId": registered["registrationId"], "runId": binding["runId"], "resultHash": binding["resultHash"], "reviews": [{"caseId": case_id, "eventHash": event_hash} for event_hash in review_hashes]},
        )
        assert review_label_set.status_code == 201, review_label_set.text
        assert review_label_set.json()["labelSetHash"] != created.json()["labelSetHash"]
        handoff = await client.post("/api/v2/gfm/governance/adaptations/handoffs", json={"schemaVersion": "socialgraph-fm.governance-adaptation-handoff/1.0", "targetTaskRegistrationId": registered["registrationId"], "policyHash": policy_hash, "decision": "pending_governance_review"})
        assert handoff.status_code == 201, handoff.text
        activation = await client.post(
            f"/api/v2/gfm/governance/adaptations/policies/{policy_hash}/activate",
            json={"schemaVersion": "socialgraph-fm.governance-adaptation-overlay/1.0", "targetTaskRegistrationId": registered["registrationId"]},
        )
        assert activation.status_code == 201, activation.text
        assert activation.json()["active"] is True
        outer_path = root / "target-tasks" / f"{registered['registrationId']}.sgtask.zip"
        outer_path.write_bytes(_rebind_false_label_facts(outer))
        stale = await client.get(f"/api/v2/gfm/governance/adaptations/policies/{policy_hash}")
        assert stale.status_code == 409
        assert "policyHash" not in stale.text
        stale_comparison = await client.get(f"/api/v2/gfm/governance/adaptations/runs/{binding['runId']}/policies/{policy_hash}/comparison")
        stale_handoff = await client.get(f"/api/v2/gfm/governance/adaptations/handoffs/{handoff.json()['handoffHash']}")
        stale_activation = await client.post(f"/api/v2/gfm/governance/adaptations/policies/{policy_hash}/activate", json={"schemaVersion": "socialgraph-fm.governance-adaptation-overlay/1.0", "targetTaskRegistrationId": registered["registrationId"]})
        assert {stale_comparison.status_code, stale_handoff.status_code, stale_activation.status_code} == {409}
        outer_path.write_bytes(outer)

        receipt_path = root / "incoming" / binding["artifactId"] / "receipt.json"
        receipt_bytes = receipt_path.read_bytes()
        receipt_path.unlink()
        missing_receipt = await client.get(f"/api/v2/gfm/governance/adaptations/policies/{policy_hash}")
        _assert_stale(missing_receipt)
        receipt_path.write_bytes(b"not-json")
        corrupt_receipt = await client.get(f"/api/v2/gfm/governance/adaptations/policies/{policy_hash}")
        _assert_stale(corrupt_receipt)
        receipt_path.write_bytes(receipt_bytes)

        inner_bundle_path = receipt_path.parent / "bundle.zip"
        inner_bundle_bytes = inner_bundle_path.read_bytes()
        inner_bundle_path.write_bytes(b"corrupt-bundle")
        corrupt_bundle = await client.get(f"/api/v2/gfm/governance/adaptations/policies/{policy_hash}")
        _assert_stale(corrupt_bundle)
        inner_bundle_path.unlink()
        missing_bundle = await client.get(f"/api/v2/gfm/governance/adaptations/policies/{policy_hash}")
        _assert_stale(missing_bundle)
        inner_bundle_path.write_bytes(inner_bundle_bytes)

        async def drifted_model(_self, run_id, result_hash):
            status, result, receipt, preview = await context(_self, run_id, result_hash)
            status.model_state_hash = "0" * 64
            return status, result, receipt, preview

        gateway._current_adaptation_context = MethodType(drifted_model, gateway)
        model_drift = await client.get(f"/api/v2/gfm/governance/adaptations/policies/{policy_hash}")
        _assert_stale(model_drift)

        async def drifted_result(_self, run_id, result_hash):
            status, result, receipt, preview = await context(_self, run_id, result_hash)
            result.result_hash = "0" * 64
            return status, result, receipt, preview

        gateway._current_adaptation_context = MethodType(drifted_result, gateway)
        result_drift = await client.get(f"/api/v2/gfm/governance/adaptations/policies/{policy_hash}")
        _assert_stale(result_drift)
        gateway._current_adaptation_context = MethodType(context, gateway)
        fake.stale_policy = True
        gfm_stale = await client.get(f"/api/v2/gfm/governance/adaptations/policies/{policy_hash}")
        _assert_stale(gfm_stale)


@pytest.mark.anyio
async def test_v2_same_few_shot_labels_bind_two_runs_and_repeat_idempotently(
    tmp_path: Path, unconfigured_settings
) -> None:
    root = tmp_path / "governance"
    settings = unconfigured_settings.model_copy(
        update={"gfm_governance_root": str(root)}
    )
    outer, _ = _few_shot_target_task_bundle()
    fake = _RepeatedV2AdaptationClient()
    app = create_app(settings, gfm_governance_client=fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        registered = (
            await client.post(
                "/api/v2/gfm/governance/target-tasks",
                files={"file": ("few.sgtask.zip", outer, "application/zip")},
            )
        ).json()
        bindings = []
        for token in ("8", "7"):
            binding = {
                "artifactId": registered["artifact"]["artifactId"],
                "datasetContentHash": registered["artifact"]["datasetContentHash"],
                "graphVersionHash": registered["artifact"]["graphVersionHash"],
                "runId": "governance-" + token * 32,
                "requestHash": hashlib.sha256(f"request-{token}".encode()).hexdigest(),
                "resultHash": hashlib.sha256(f"result-{token}".encode()).hexdigest(),
                "runArtifactHash": hashlib.sha256(f"artifact-{token}".encode()).hexdigest(),
                "modelVersionId": "socialgraph-fm-global/" + "1" * 16,
                "modelVersionHash": "c" * 64,
                "modelStateHash": "d" * 64,
                "recipeHash": "e" * 64,
                "codeHash": "f" * 64,
                "seed": 1729,
            }
            fake.bindings[binding["runId"]] = binding
            bindings.append(binding)
        gateway = app.state.gfm_governance_gateway

        async def context(_self, run_id, result_hash):
            binding = fake.bindings[run_id]
            assert result_hash == binding["resultHash"]
            status = SimpleNamespace(
                status="succeeded",
                run_id=run_id,
                request_hash=binding["requestHash"],
                artifact_id=binding["artifactId"],
                dataset_content_hash=binding["datasetContentHash"],
                graph_version_hash=binding["graphVersionHash"],
                model_version_id=binding["modelVersionId"],
                model_version_hash=binding["modelVersionHash"],
                model_state_hash=binding["modelStateHash"],
            )
            result = SimpleNamespace(result_hash=result_hash)
            return status, result, gateway.inbox.get(binding["artifactId"]), None

        gateway._current_adaptation_context = MethodType(context, gateway)
        label_hashes: list[str] = []
        policy_hashes: list[str] = []
        for binding in bindings:
            create_body = {
                "schemaVersion": "socialgraph-fm.governance-target-label-set/2.0",
                "sourceType": "imported_sidecar",
                "targetTaskRegistrationId": registered["registrationId"],
                "runId": binding["runId"],
                "resultHash": binding["resultHash"],
            }
            created = await client.post(
                "/api/v2/gfm/governance/adaptations/label-sets", json=create_body
            )
            assert created.status_code == 201, created.text
            repeated = await client.post(
                "/api/v2/gfm/governance/adaptations/label-sets", json=create_body
            )
            assert repeated.status_code == 201
            assert repeated.json() == created.json()
            label_hashes.append(created.json()["labelSetHash"])
            fit_body = {
                "schemaVersion": "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
                "targetTaskRegistrationId": registered["registrationId"],
                "runId": binding["runId"],
                "resultHash": binding["resultHash"],
            }
            path = f"/api/v2/gfm/governance/adaptations/label-sets/{label_hashes[-1]}/policies"
            fitted = await client.post(path, json=fit_body)
            assert fitted.status_code == 201, fitted.text
            repeated_fit = await client.post(path, json=fit_body)
            assert repeated_fit.status_code == 201
            assert repeated_fit.json() == fitted.json()
            policy_hashes.append(fitted.json()["policyHash"])

        assert label_hashes[0] == label_hashes[1]
        assert policy_hashes[0] != policy_hashes[1]
        ambiguous = await client.post(
            f"/api/v2/gfm/governance/adaptations/label-sets/{label_hashes[0]}/policies"
        )
        assert ambiguous.status_code == 409
        wrong_identity = await client.post(
            f"/api/v2/gfm/governance/adaptations/label-sets/{label_hashes[0]}/policies",
            json={
                "schemaVersion": "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
                "targetTaskRegistrationId": registered["registrationId"],
                "runId": bindings[0]["runId"],
                "resultHash": bindings[1]["resultHash"],
            },
        )
        assert wrong_identity.status_code in {404, 409}
        cross_lane = await client.post(
            f"/api/v2/gfm/governance/adaptations/label-sets/{label_hashes[0]}/policies",
            json={
                "schemaVersion": "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
                "targetTaskRegistrationId": "target-task-" + "f" * 32,
                "runId": bindings[0]["runId"],
                "resultHash": bindings[0]["resultHash"],
            },
        )
        assert cross_lane.status_code in {404, 409}

    with sqlite3.connect(root / "governance.sqlite3") as connection:
        counts = dict(
            connection.execute(
                "SELECT kind, COUNT(*) FROM target_adaptation_metadata "
                "WHERE kind LIKE 'label_set_%' GROUP BY kind"
            ).fetchall()
        )
    assert counts == {"label_set_binding": 2, "label_set_content": 1}
