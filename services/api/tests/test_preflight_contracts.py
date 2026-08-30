from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest

from app.config import Settings
from app.dataset_imports import (
    DatasetImportService,
    _array_descriptors,
    _canonical_graph_json,
    _license_policy,
)
from app.dataset_storage import DatasetArtifactStore
from app.runtime_fingerprint import converter_environment_details

from .test_dataset_imports import graph_version_handoff


def _zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _npz(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def test_converter_fingerprint_records_runtime_packages_source_and_dependencies(
    unconfigured_settings: Settings,
) -> None:
    details = converter_environment_details(unconfigured_settings)

    assert details["schemaVersion"] == "converter-environment/1.0"
    assert details["sourceCommit"]
    assert len(details["sourceTreeHash"]) == 64
    assert len(details["dependencyHash"]) == 64
    assert "pyproject.toml" in details["dependencies"]
    assert details["interpreter"]["python"]
    assert set(details["interpreter"]["packages"]) == {
        "torch",
        "torch-geometric",
        "ogb",
        "numpy",
    }
    assert "cuda" in details["interpreter"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.0, "0"),
        (1.0, "1"),
        (0.001, "0.001"),
        (1e-6, "0.000001"),
        (1e-7, "1e-7"),
        (1e20, "100000000000000000000"),
        (1e21, "1e+21"),
    ],
)
def test_graph_fact_numbers_match_ecmascript_json(value: float, expected: str) -> None:
    assert _canonical_graph_json(value) == expected


def test_graph_fact_canonical_json_matches_shared_cross_runtime_golden() -> None:
    golden_path = (
        Path(__file__).resolve().parents[3]
        / "packages"
        / "gfm"
        / "tests"
        / "golden"
        / "canonical-vectors.json"
    )
    vectors = json.loads(golden_path.read_text(encoding="utf-8"))["vectors"]
    for vector in vectors:
        serialized = _canonical_graph_json(vector["value"])
        assert serialized == vector["canonical"], vector["name"]
        assert hashlib.sha256(serialized.encode("utf-8")).hexdigest() == vector["sha256"]


@pytest.mark.anyio
async def test_runtime_capability_contract_is_explicit(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.get("/api/v1/capabilities")
    assert response.status_code == 200
    runtime = response.json()["runtime"]
    assert runtime["apiContract"] == "socialgraph-fm-api/1.1"
    assert runtime["storageSchema"] == "dataset-store/2"
    assert runtime["datasetArtifactSchemas"] == ["1.0", "2.0", "2.1", "2.2"]
    assert runtime["trainingRefSchemas"] == ["1.0", "1.1"]
    assert runtime["graphFactHash"] == "graph-fact-hash/1"
    assert len(runtime["converterEnvironmentFingerprint"]) == 64
    public_environment = runtime["converterEnvironment"]
    serialized = json.dumps(public_environment, ensure_ascii=False).casefold()
    assert public_environment["schemaVersion"] == "converter-environment-public/1.0"
    assert "executable" not in serialized
    assert "devices" not in serialized
    assert "\\users\\" not in serialized
    assert ":/users/" not in serialized


@pytest.mark.anyio
async def test_graph_fact_hash_reservation_commit_and_idempotency(
    api_client: httpx.AsyncClient,
) -> None:
    legacy = json.loads(graph_version_handoff())
    inspected = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={
            "file": (
                "graph-v1.sgfm-graph.json",
                json.dumps(legacy).encode(),
                "application/json",
            )
        },
    )
    fact_hash = inspected.json()["serverGraphFactHash"]
    assert len(fact_hash) == 64
    reservation = await api_client.post(
        "/api/v1/graph-dataset-handoffs/reserve",
        json={"graphVersionId": "graph-v1", "graphFactHash": fact_hash},
    )
    assert reservation.status_code == 200
    request = {
        "token": reservation.json()["token"],
        "envelope": legacy,
        "preparation": {
            "schemaVersion": "1.0",
            "graphVersionId": "graph-v1",
            "featureAttributes": [],
            "taskKind": "none",
            "splitStrategy": "none",
            "excludedAttributes": ["email", "phone"],
            "deidentify": True,
            "governance": {
                "containsPersonalData": True,
                "deidentified": True,
                "attributeAllowlist": [],
                "excludedAttributes": ["email", "phone"],
                "retention": "project",
                "userDataTrainingOptIn": False,
            },
        },
    }
    committed = await api_client.post("/api/v1/graph-dataset-handoffs/commit", json=request)
    assert committed.status_code == 200
    body = committed.json()
    assert body["reused"] is False
    assert body["binding"]["graphFactHash"] == fact_hash
    assert body["artifact"]["schemaVersion"] == "2.2"
    assert body["artifact"]["trainingRefs"] == []
    assert body["artifact"]["datasetRole"] == "target_domain"

    repeated = await api_client.post("/api/v1/graph-dataset-handoffs/commit", json=request)
    assert repeated.status_code == 200
    assert repeated.json()["reused"] is True
    assert repeated.json()["binding"]["id"] == body["binding"]["id"]
    assert repeated.json()["artifact"]["id"] == body["artifact"]["id"]


@pytest.mark.anyio
async def test_graph_handoff_v11_rejects_client_hash_mismatch(
    api_client: httpx.AsyncClient,
) -> None:
    payload = json.loads(graph_version_handoff())
    payload["schemaVersion"] = "socialgraph-fm-graph/1.1"
    payload["graphFactHash"] = "0" * 64
    response = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={"file": ("bad.sgfm-graph.json", json.dumps(payload).encode(), "application/json")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert response.json()["issues"][0]["code"] == "GRAPH_FACT_HASH_MISMATCH"


def test_artifact_list_isolates_invalid_legacy_row(tmp_path: Path) -> None:
    store = DatasetArtifactStore(tmp_path / "store")
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            """
            INSERT INTO dataset_artifacts
            (id, dataset_name, checksum, canonical_graph_hash, scope,
             created_at, artifact_json, tensor_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "broken-row",
                "broken",
                "0" * 64,
                "0" * 64,
                "complete",
                "2026-08-11T00:00:00+00:00",
                "{}",
                "artifacts/broken-row/graph.npz",
            ),
        )
    assert store.list_artifacts() == []
    assert store.last_list_issues == [
        {"artifactId": "unknown", "code": "ARTIFACT_ROW_INVALID"}
    ]


def test_embedded_verified_license_claim_is_downgraded() -> None:
    policy = _license_policy(
        {
            "licensePolicy": {
                "status": "verified",
                "identifier": "claimed-license",
                "allowedUses": ["pretraining"],
            }
        },
        trusted_generated=False,
    )
    assert policy.status == "unknown"
    assert policy.allowed_uses == []


def test_pytest_refuses_formal_dataset_store() -> None:
    with pytest.raises(RuntimeError, match="系统临时目录"):
        DatasetArtifactStore(Path.cwd() / "artifacts" / "must-not-be-opened-by-tests")
    assert not (Path.cwd() / "artifacts" / "must-not-be-opened-by-tests").exists()


def test_trusted_ogbl_contract_is_ready_and_materializable(tmp_path: Path) -> None:
    graph = _npz(
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        x=np.eye(6, dtype=np.float32),
        num_nodes=np.asarray(6, dtype=np.int64),
        node_id_map=np.asarray([str(index) for index in range(6)]),
        directed=np.asarray(False),
        edge_weight=np.asarray([1.0, 2.0], dtype=np.float32),
        edge_timestamp=np.asarray([2016, 2017], dtype=np.int16),
        variant_train_positive=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        variant_validation_positive=np.asarray([[2], [3]], dtype=np.int64),
        variant_test_positive=np.asarray([[3], [4]], dtype=np.int64),
        variant_validation_negative=np.asarray([[0], [5]], dtype=np.int64),
        variant_test_negative=np.asarray([[1], [5]], dtype=np.int64),
    )
    protocol = {
        "messagePassingEdgeArray": "edge_index",
        "trainPositiveArray": "variant_train_positive",
        "validationPositiveArray": "variant_validation_positive",
        "testPositiveArray": "variant_test_positive",
        "validationNegativeArray": "variant_validation_negative",
        "testNegativeArray": "variant_test_negative",
        "edgeYearArray": "edge_timestamp",
        "edgeWeightArray": "edge_weight",
        "trainYearMax": 2017,
        "validationYear": 2018,
        "testYear": 2019,
        "negativeSampler": "stored",
        "evaluator": "ogb.linkproppred.Evaluator(ogbl-collab)",
        "evaluatorVersion": "1.3.6",
    }
    item = {
        "name": "ogbl-collab",
        "path": "datasets/ogbl-collab/graph.npz",
        "sourceFormat": "trusted_local_ogb",
        "datasetRole": "benchmark",
        "licensePolicy": {
            "status": "verified",
            "identifier": "ODC-BY-1.0",
            "sourceUrl": "https://ogb.stanford.edu/docs/linkprop/#ogbl-collab",
            "allowedUses": ["evaluation", "adaptation", "inference", "pretraining"],
        },
        "licenseEvidence": [
            {
                "id": "ogbl-official",
                "kind": "official_metadata",
                "sourceUrl": "https://ogb.stanford.edu/docs/linkprop/#ogbl-collab",
                "recordedAt": "2026-08-11T00:00:00Z",
                "recordedBy": "socialgraph-fm-ogb-adapter",
            }
        ],
        "dataGovernance": {
            "containsPersonalData": False,
            "deidentified": True,
            "attributeAllowlist": [],
            "excludedAttributes": [],
            "retention": "research_archive",
            "userDataTrainingOptIn": False,
        },
        "transformRecipes": [
            {
                "id": "ogb-identity-v1",
                "graphVariant": "raw",
                "featureTransform": "identity",
            }
        ],
        "linkPredictionProtocol": protocol,
    }
    manifest = {
        "schemaVersion": "socialgraph-fm-dataset-package/1.0",
        "datasets": [item],
        "skipped": [],
    }
    package = tmp_path / "ogbl.sgfm.zip"
    package.write_bytes(
        _zip(
            {
                "manifest.json": json.dumps(manifest).encode(),
                "datasets/ogbl-collab/graph.npz": graph,
            }
        )
    )
    service = DatasetImportService(Settings(dataset_storage_root=str(tmp_path / "store")))
    artifact = service.import_trusted_package(
        str(package), job_id="job-ogbl", source_path=str(tmp_path / "trusted-ogb")
    )[0]
    assert artifact.schema_version == "2.2"
    assert artifact.training_ref is not None
    assert artifact.training_ref.schema_version == "1.1"
    assert artifact.training_ref.split_fold == 0
    assert artifact.task_specs[0].kind == "link_prediction"
    assert artifact.license_policy is not None
    assert artifact.license_policy.status == "verified"
    readiness = service.readiness(
        artifact.id, training_ref_hash=artifact.training_ref.ref_hash
    )
    assert readiness.status == "ready", readiness.blockers
    bundle = service.materialize_contract(
        artifact.id, training_ref_hash=artifact.training_ref.ref_hash
    )
    assert bundle.task_kind == "link_prediction"
    assert bundle.feature_shape == [6, 6]
    assert bundle.split_sizes == {"train": 2, "validation": 1, "test": 1}

    arrays = service.store.load_arrays(artifact.id)
    arrays["edge_timestamp"] = np.asarray([2019, 2017], dtype=np.int16)
    tensor = service.store.artifact_directory(artifact.id) / "graph.npz"
    with tensor.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    artifact.arrays, _attachments = _array_descriptors(arrays, {})
    with sqlite3.connect(service.store.database_path) as connection:
        connection.execute(
            "UPDATE dataset_artifacts SET artifact_json = ? WHERE id = ?",
            (artifact.model_dump_json(by_alias=True), artifact.id),
        )
    leaked = service.readiness(
        artifact.id, training_ref_hash=artifact.training_ref.ref_hash
    )
    assert leaked.status == "corrupt"
    assert "TEMPORAL_TRAINING_LEAKAGE" in leaked.blockers[0].message
