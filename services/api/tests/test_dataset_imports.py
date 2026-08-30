from __future__ import annotations

import io
import json
import pickle
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest

from app import dataset_tools
from app.config import Settings
from app.dataset_imports import (
    GraphPayload,
    _build_view,
    _canonical_graph_hash,
    _content_hash,
    _graph_from_arrays,
)
from app.dataset_schemas import DatasetArtifact
from app.dataset_tools import main as dataset_tools_main
from app.main import create_app


def npz_bytes(**arrays: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def zip_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return output.getvalue()


@pytest.mark.anyio
async def test_safe_graph_npz_inspect_commit_and_get(api_client: httpx.AsyncClient) -> None:
    payload = npz_bytes(
        x=np.eye(4, dtype=np.float32),
        edge_index=np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        y=np.asarray([0, 0, 1, 1], dtype=np.int64),
        train_mask=np.asarray([1, 1, 0, 0], dtype=np.uint8),
    )
    inspected = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={"file": ("graph.npz", payload, "application/octet-stream")},
    )
    assert inspected.status_code == 200
    inspection = inspected.json()
    assert inspection["detectedFormat"] == "graph_npz"
    assert inspection["status"] == "accepted"
    assert inspection["profile"] == {
        "nodeCount": 4,
        "edgeCount": 3,
        "featureDimension": 4,
        "labelCount": 2,
        "splitNames": ["train_mask"],
        "directed": False,
    }

    committed = await api_client.post(
        f"/api/v1/dataset-imports/{inspection['id']}/commit"
    )
    assert committed.status_code == 200
    artifact = committed.json()
    assert artifact["graphView"]["summary"]["nodeCount"] == 4
    assert artifact["graphView"]["summary"]["edgeCount"] == 3
    assert artifact["graphView"]["summary"]["connectedComponents"] == 1
    assert artifact["graphView"]["summary"]["partialPreview"] is False
    assert len(artifact["checksum"]) == 64
    assert artifact["schemaVersion"] == "2.2"
    assert len(artifact["contentHash"]) == 64
    assert len(artifact["manifestHash"]) == 64
    assert artifact["datasetRole"] == "target_domain"
    assert artifact["nodeIdentity"] == {
        "id": "node-identity-v1",
        "arrayName": "node_id_map",
        "kind": "row_index",
        "count": 4,
        "unique": True,
    }
    assert artifact["trainingRef"] == artifact["trainingRefs"][0]
    assert artifact["trainingRef"]["artifactId"] == artifact["id"]
    assert artifact["trainingRef"]["contentHash"] == artifact["contentHash"]
    assert artifact["trainingRef"]["splitSetId"] == "source-splits"
    assert artifact["trainingRef"]["taskSpecId"] == "node-classification-v1"
    assert len(artifact["trainingRef"]["refHash"]) == 64
    assert artifact["derivedManifest"]["contentHash"] == artifact["contentHash"]

    fetched = await api_client.get(f"/api/v1/dataset-artifacts/{artifact['id']}")
    assert fetched.status_code == 200
    assert fetched.json() == artifact


def graph_version_handoff(**overrides: object) -> bytes:
    value: dict[str, object] = {
        "schemaVersion": "socialgraph-fm-graph/1.0",
        "graphVersionId": "graph-v1",
        "contentHash": "a" * 64,
        "buildSpecHash": "b" * 64,
        "sourceFile": "治理关系.csv",
        "directedness": "directed",
        "nodes": [
            {
                "id": "actor-a",
                "label": "社区甲",
                "type": "社区",
                "attributes": {"district": "东区"},
            },
            {
                "id": "actor-b",
                "label": "机构乙",
                "type": "机构",
                "attributes": {},
            },
        ],
        "edges": [
            {
                "id": "relation-1",
                "source": "actor-a",
                "target": "actor-b",
                "type": "协作",
                "weight": 0.75,
                "timestamp": "2024-08-01",
                "directed": True,
                "attributes": {"evidence": "公开记录"},
            }
        ],
    }
    value.update(overrides)
    return json.dumps(value, ensure_ascii=False).encode()


@pytest.mark.anyio
async def test_graph_version_text_handoff_is_target_domain_and_preserves_facts(
    api_client: httpx.AsyncClient,
) -> None:
    inspected = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={
            "file": (
                "graph-v1.sgfm-graph.json",
                graph_version_handoff(),
                "application/vnd.socialgraph-fm.graph+json",
            )
        },
    )
    assert inspected.status_code == 200
    inspection = inspected.json()
    assert inspection["status"] == "accepted"
    assert inspection["detectedFormat"] == "graph_version_target_domain"

    committed = await api_client.post(
        f"/api/v1/dataset-imports/{inspection['id']}/commit"
    )
    assert committed.status_code == 200
    artifact = committed.json()
    assert artifact["schemaVersion"] == "2.2"
    assert artifact["datasetRole"] == "target_domain"
    assert artifact["nodeIdentity"]["kind"] == "source"
    assert artifact["rawManifest"]["graphVersionHandoff"] == {
        "schemaVersion": "socialgraph-fm-graph/1.0",
        "graphVersionId": "graph-v1",
        "contentHash": "a" * 64,
        "buildSpecHash": "b" * 64,
            "sourceFile": "治理关系.csv",
            "directedness": "directed",
            "graphFactHash": inspection["serverGraphFactHash"],
        }
    assert artifact["graphView"]["nodes"][0] == {
        "id": "actor-a",
        "label": "社区甲",
        "nodeType": "社区",
        "attributes": {"district": "东区"},
    }
    assert artifact["graphView"]["edges"][0] == {
        "id": "relation-1",
        "source": "actor-a",
        "target": "actor-b",
        "edgeType": "协作",
        "weight": 0.75,
        "timestamp": "2024-08-01",
        "directed": True,
        "attributes": {"evidence": "公开记录"},
    }
    array_names = {item["name"] for item in artifact["arrays"]}
    assert {
        "node_id_map",
        "node_label",
        "node_type",
        "node_attributes_json",
        "edge_id_map",
        "edge_type",
        "edge_weight",
        "edge_timestamp",
        "edge_directed",
        "edge_attributes_json",
    }.issubset(array_names)
    assert artifact["featureSchemas"] == []
    assert artifact["labelSchemas"] == []
    assert artifact["splitSets"] == []
    assert artifact["taskSpecs"] == []

    readiness = await api_client.get(
        f"/api/v1/dataset-artifacts/{artifact['id']}/readiness"
    )
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "blocked"
    assert readiness.json()["blockers"][0]["code"] == "TRAINING_REFERENCE_MISSING"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (graph_version_handoff(datasetRole="pretraining_candidate"), "GRAPH_VERSION_SCHEMA_INVALID"),
        (
            graph_version_handoff(
                edges=[
                    {
                        "id": "relation-1",
                        "source": "actor-a",
                        "target": "missing",
                        "directed": True,
                        "attributes": {},
                    }
                ]
            ),
            "GRAPH_VERSION_DANGLING_ENDPOINT",
        ),
        (
            graph_version_handoff(
                edges=[
                    {
                        "id": "relation-1",
                        "source": "actor-a",
                        "target": "actor-b",
                        "directed": False,
                        "attributes": {},
                    }
                ]
            ),
            "GRAPH_VERSION_DIRECTEDNESS_MISMATCH",
        ),
    ],
)
async def test_graph_version_text_handoff_rejects_privilege_and_invalid_facts(
    api_client: httpx.AsyncClient,
    payload: bytes,
    code: str,
) -> None:
    response = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={"file": ("invalid.sgfm-graph.json", payload, "application/json")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert response.json()["issues"][0]["code"] == code


def test_content_hash_covers_features_labels_splits_and_recipe() -> None:
    base = GraphPayload(
        node_count=3,
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        features=np.eye(3, dtype=np.float32),
        labels=np.asarray([0, 1, 1], dtype=np.int64),
        split_names=["train_mask"],
        splits={"train_mask": np.asarray([1, 0, 0], dtype=np.uint8)},
    )
    recipe = [{"id": "identity-v1", "graphVariant": "raw"}]
    expected = _content_hash(base, recipe)

    changed_features = GraphPayload(**{**base.__dict__, "features": base.features.copy()})
    assert changed_features.features is not None
    changed_features.features[0, 0] = 2
    changed_labels = GraphPayload(**{**base.__dict__, "labels": np.asarray([1, 1, 1])})
    changed_splits = GraphPayload(
        **{
            **base.__dict__,
            "splits": {"train_mask": np.asarray([0, 1, 0], dtype=np.uint8)},
        }
    )

    assert _content_hash(changed_features, recipe) != expected
    assert _content_hash(changed_labels, recipe) != expected
    assert _content_hash(changed_splits, recipe) != expected
    assert _content_hash(base, [{"id": "pca-50", "graphVariant": "raw"}]) != expected


def test_content_hash_always_covers_directed_structural_semantics() -> None:
    edges = np.asarray([[0], [1]], dtype=np.int64)
    recipe = [{"id": "identity-v1", "graphVariant": "raw"}]

    undirected = GraphPayload(node_count=2, edge_index=edges, directed=False)
    directed = GraphPayload(node_count=2, edge_index=edges, directed=True)

    assert _content_hash(undirected, recipe) != _content_hash(directed, recipe)


def test_split_masks_are_binary_fold_aligned_and_disjoint() -> None:
    base = {
        "edge_index": np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        "num_nodes": np.asarray(3, dtype=np.int64),
    }
    with pytest.raises(ValueError, match="0/1"):
        _graph_from_arrays(
            {**base, "train_mask": np.asarray([-1, 0, 2], dtype=np.int64)}
        )
    with pytest.raises(ValueError, match="折数"):
        _graph_from_arrays(
            {
                **base,
                "train_mask": np.asarray([[1, 0], [0, 1], [0, 0]], dtype=np.uint8),
                "val_mask": np.asarray([0, 0, 1], dtype=np.uint8),
            }
        )
    with pytest.raises(ValueError, match="存在交叉"):
        _graph_from_arrays(
            {
                **base,
                "train_mask": np.asarray([1, 0, 0], dtype=np.uint8),
                "val_mask": np.asarray([1, 0, 0], dtype=np.uint8),
            }
        )


@pytest.mark.anyio
async def test_external_split_cannot_override_embedded_official_mask(
    api_client: httpx.AsyncClient,
) -> None:
    graph = npz_bytes(
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        num_nodes=np.asarray(3, dtype=np.int64),
        train_mask=np.asarray([1, 0, 0], dtype=np.uint8),
    )
    conflicting = npz_bytes(train_mask=np.asarray([0, 1, 0], dtype=np.uint8))

    response = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files=[
            ("files", ("graph.npz", graph, "application/octet-stream")),
            ("files", ("split.npz", conflicting, "application/octet-stream")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "rejected"
    assert "内嵌官方划分冲突" in body["issues"][0]["message"]


def test_topology_aware_preview_does_not_create_false_isolates() -> None:
    dense_prefix = [(source, target) for _ in range(130) for source in range(10) for target in range(10)]
    connecting_tail = [(index - 1, index) for index in range(10, 100)]
    edges = np.asarray([*dense_prefix, *connecting_tail], dtype=np.int64).T
    payload = GraphPayload(node_count=100, edge_index=edges)

    view = _build_view(payload, "preview-test")

    incident = {edge.source for edge in view.edges} | {edge.target for edge in view.edges}
    assert view.summary.connected_components == 1
    assert view.summary.visible_node_count == 100
    assert view.summary.visible_edge_count == 145
    assert view.summary.edge_count == 145
    assert view.summary.partial_preview is False
    assert incident == {str(index) for index in range(100)}


def test_truncated_preview_never_fills_one_slot_with_half_of_a_dyad() -> None:
    chain_source = np.arange(0, 2998, dtype=np.int64)
    chain_target = np.arange(1, 2999, dtype=np.int64)
    edges = np.stack(
        (
            np.concatenate((chain_source, np.asarray([2999], dtype=np.int64))),
            np.concatenate((chain_target, np.asarray([3000], dtype=np.int64))),
        )
    )
    payload = GraphPayload(node_count=3002, edge_index=edges, directed=True)

    view = _build_view(payload, "single-slot")

    visible_nodes = {int(node.id) for node in view.nodes}
    incident = {int(edge.source) for edge in view.edges} | {
        int(edge.target) for edge in view.edges
    }
    full_incident = set(edges.reshape(-1).tolist())
    assert all(node not in full_incident or node in incident for node in visible_nodes)
    assert 2999 not in visible_nodes
    assert 3001 in visible_nodes


def test_undirected_preview_deduplicates_symmetric_coo_and_density() -> None:
    payload = GraphPayload(
        node_count=3,
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        directed=False,
    )

    view = _build_view(payload, "symmetric")

    assert view.summary.edge_count == 1
    assert view.summary.visible_edge_count == 1
    assert view.summary.density == pytest.approx(1 / 3)
    assert view.summary.partial_preview is False
    assert len(view.edges) == 1


def test_undirected_canonical_hash_normalizes_orientation_and_duplicates() -> None:
    forward = GraphPayload(2, np.asarray([[0], [1]], dtype=np.int64), directed=False)
    reverse = GraphPayload(2, np.asarray([[1], [0]], dtype=np.int64), directed=False)
    symmetric = GraphPayload(
        2,
        np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
        directed=False,
    )

    assert _canonical_graph_hash(forward) == _canonical_graph_hash(reverse)
    assert _canonical_graph_hash(forward) == _canonical_graph_hash(symmetric)
    directed_forward = GraphPayload(2, forward.edge_index, directed=True)
    directed_reverse = GraphPayload(2, reverse.edge_index, directed=True)
    assert _canonical_graph_hash(directed_forward) != _canonical_graph_hash(directed_reverse)


def test_pyg_field_extraction_preserves_official_masks() -> None:
    value = {
        "x": np.eye(4, dtype=np.float32),
        "edge_index": np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        "y": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "train_mask": np.asarray([1, 1, 0, 0], dtype=np.bool_),
        "val_mask": np.asarray([0, 0, 1, 0], dtype=np.bool_),
        "test_mask": np.asarray([0, 0, 0, 1], dtype=np.bool_),
    }

    _features, _edges, _labels, splits = dataset_tools._extract_pyg_fields(value)

    assert set(splits) == {"train_mask", "val_mask", "test_mask"}
    assert int(splits["train_mask"].sum()) == 2


def test_weighted_structure_edge_recipe_is_undirected_loop_free_and_sorted() -> None:
    raw = np.asarray([[0, 1, 1, 2, 2], [1, 0, 1, 0, 2]], dtype=np.int64)
    actual = dataset_tools._canonicalize_weighted_structure_edges(raw, 3)
    pairs = list(zip(actual[0].tolist(), actual[1].tolist(), strict=True))

    assert pairs == [(0, 1), (0, 2), (1, 0), (2, 0)]


def test_weighted_structure_dimension_match_is_identity_and_seed_is_hashed() -> None:
    features = np.arange(60 * 50, dtype=np.float32).reshape(60, 50)
    edges = np.asarray([[0, 1], [1, 2]], dtype=np.int64)

    arrays_0, recipes_0 = dataset_tools._weighted_structure_variant_arrays(
        features,
        edges,
        pca_seed=0,
    )
    _arrays_7, recipes_7 = dataset_tools._weighted_structure_variant_arrays(
        features,
        edges,
        pca_seed=7,
    )

    np.testing.assert_array_equal(arrays_0["variant_weighted_structure_x"], features)
    assert recipes_0[0]["featureTransform"] == "identity_dimension_match"
    assert recipes_0[0]["parameters"] == {"nComponents": 50, "randomState": 0}
    assert recipes_7[0]["parameters"] == {"nComponents": 50, "randomState": 7}
    payload = GraphPayload(
        node_count=60,
        edge_index=edges,
        features=features,
        feature_dimension=50,
        variant_arrays=arrays_0,
    )
    assert _content_hash(payload, recipes_0) != _content_hash(payload, recipes_7)


def test_converter_split_files_are_source_aligned(tmp_path: Path) -> None:
    np.savez_compressed(
        tmp_path / "split_0.npz",
        train_mask=np.asarray([1, 0, 0], dtype=np.uint8),
        val_mask=np.asarray([0, 1, 0], dtype=np.uint8),
        test_mask=np.asarray([0, 0, 1], dtype=np.uint8),
    )
    np.savez_compressed(
        tmp_path / "split_1.npz",
        train_mask=np.asarray([0, 1, 0], dtype=np.uint8),
        val_mask=np.asarray([1, 0, 0], dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="字段与其他折不对齐"):
        dataset_tools._safe_split_arrays(tmp_path, 3)


def test_converter_rejects_multifile_ragged_idx_instead_of_dropping_it(
    tmp_path: Path,
) -> None:
    np.savez_compressed(
        tmp_path / "split_0.npz",
        train_idx=np.asarray([0], dtype=np.int64),
    )
    np.savez_compressed(
        tmp_path / "split_1.npz",
        train_idx=np.asarray([1, 2], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="变长 idx split 无法无损合并"):
        dataset_tools._safe_split_arrays(tmp_path, 3)


def test_dataset_artifact_v2_requires_hash_and_matching_training_ref() -> None:
    graph_view = {
        "id": "view",
        "nodes": [],
        "edges": [],
        "summary": {
            "nodeCount": 0,
            "edgeCount": 0,
            "density": 0,
            "connectedComponents": 0,
            "visibleNodeCount": 0,
            "visibleEdgeCount": 0,
            "partialPreview": False,
        },
    }
    base = {
        "schemaVersion": "2.0",
        "id": "artifact",
        "inspectionId": "inspection",
        "sourceFormat": "graph_npz",
        "sourceFiles": ["graph.npz"],
        "checksum": "source",
        "profile": {},
        "graphView": graph_view,
        "createdAt": datetime.now(UTC),
    }
    legacy = DatasetArtifact.model_validate({**base, "schemaVersion": "1.0"})
    assert legacy.content_hash == ""
    with pytest.raises(ValueError, match="contentHash"):
        DatasetArtifact.model_validate(base)

    content_hash = "a" * 64
    with pytest.raises(ValueError, match="trainingRef"):
        DatasetArtifact.model_validate({**base, "contentHash": content_hash})

    mismatched_ref = {
        "artifactId": "artifact",
        "contentHash": "b" * 64,
        "graphVariant": "raw",
        "featureRecipeId": "identity-v1",
    }
    with pytest.raises(ValueError, match="必须与 DatasetArtifact 一致"):
        DatasetArtifact.model_validate(
            {**base, "contentHash": content_hash, "trainingRef": mismatched_ref}
        )

    valid_ref = {**mismatched_ref, "contentHash": content_hash}
    artifact = DatasetArtifact.model_validate(
        {**base, "contentHash": content_hash, "trainingRef": valid_ref}
    )
    assert artifact.content_hash == content_hash


@pytest.mark.anyio
async def test_geom_gcn_pair_and_multifold_split_are_accepted(
    api_client: httpx.AsyncClient,
) -> None:
    nodes = (
        b"node_id\tfeature\tlabel\n"
        b"a\t1,0,0\t0\n"
        b"b\t0,1,0\t1\n"
        b"c\t0,0,1\t1\n"
        b"d\t1,1,0\t0\n"
    )
    edges = b"node_id\tnode_id\na\tb\nb\tc\nc\td\n"
    split = npz_bytes(
        train_mask=np.asarray(
            [[1, 0], [1, 0], [0, 1], [0, 1]],
            dtype=np.uint8,
        ),
        val_mask=np.zeros((4, 2), dtype=np.bool_),
        test_mask=np.zeros((4, 2), dtype=np.bool_),
    )
    response = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files=[
            ("files", ("out1_node_feature_label.txt", nodes, "text/plain")),
            ("files", ("out1_graph_edges.txt", edges, "text/plain")),
            ("files", ("dataset_split_0.npz", split, "application/octet-stream")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "accepted"
    assert body["detectedFormat"] == "geom_gcn_text"
    assert body["profile"]["featureDimension"] == 3
    assert body["profile"]["splitNames"] == ["train_mask", "val_mask", "test_mask"]


@pytest.mark.anyio
async def test_split_only_needs_graph_and_cannot_commit(api_client: httpx.AsyncClient) -> None:
    split = npz_bytes(
        train_mask=np.asarray([1, 0], dtype=np.uint8),
        test_mask=np.asarray([0, 1], dtype=np.uint8),
    )
    response = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={"file": ("split.npz", split, "application/octet-stream")},
    )
    body = response.json()
    assert body["status"] == "mapping_required"
    assert body["issues"][0]["code"] == "GRAPH_FILE_REQUIRED"
    commit = await api_client.post(f"/api/v1/dataset-imports/{body['id']}/commit")
    assert commit.status_code == 409
    assert commit.json()["detail"]["code"] == "DATASET_NOT_COMMITTABLE"


@pytest.mark.anyio
async def test_fewshot_manifest_references_safe_graph_and_splits(
    api_client: httpx.AsyncClient,
) -> None:
    graph = npz_bytes(
        x=np.ones((3, 2), dtype=np.float32),
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        y=np.asarray([0, 1, 1], dtype=np.int64),
    )
    split = npz_bytes(train_idx=np.asarray([0, 1], dtype=np.int64))
    manifest = json.dumps({"graph": "graph.npz", "splits": ["shot-0.npz"]}).encode()
    response = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files=[
            ("files", ("manifest.json", manifest, "application/json")),
            ("files", ("graph.npz", graph, "application/octet-stream")),
            ("files", ("shot-0.npz", split, "application/octet-stream")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["detectedFormat"] == "fewshot_json_npz"
    assert body["status"] == "accepted"
    assert body["profile"]["splitNames"] == ["train_idx"]


@pytest.mark.anyio
async def test_socialgraph_batch_package_requires_and_accepts_dataset_selection(
    api_client: httpx.AsyncClient,
) -> None:
    first = npz_bytes(
        x=np.ones((2, 2), dtype=np.float32),
        edge_index=np.asarray([[0], [1]], dtype=np.int64),
    )
    second = npz_bytes(
        x=np.ones((3, 2), dtype=np.float32),
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
    )
    manifest = {
        "schemaVersion": "socialgraph-fm-dataset-package/1.0",
        "datasets": [
            {"name": "alpha", "path": "datasets/alpha/graph.npz"},
            {"name": "beta", "path": "datasets/beta/graph.npz"},
        ],
    }
    package = zip_bytes(
        {
            "manifest.json": json.dumps(manifest).encode(),
            "datasets/alpha/graph.npz": first,
            "datasets/beta/graph.npz": second,
        }
    )
    needs_selection = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={"file": ("batch.sgfm.zip", package, "application/zip")},
    )
    assert needs_selection.status_code == 200
    assert needs_selection.json()["status"] == "mapping_required"
    assert needs_selection.json()["issues"][0]["code"] == "DATASET_SELECTION_REQUIRED"
    assert needs_selection.json()["datasetCandidates"] == ["alpha", "beta"]

    selected = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        data={"dataset": "beta"},
        files={"file": ("batch.sgfm.zip", package, "application/zip")},
    )
    assert selected.status_code == 200
    assert selected.json()["status"] == "accepted"
    assert selected.json()["detectedFormat"] == "socialgraph_dataset_package"
    assert selected.json()["profile"]["nodeCount"] == 3
    assert selected.json()["datasetCandidates"] == ["alpha", "beta"]

    committed = await api_client.post(
        f"/api/v1/dataset-imports/{selected.json()['id']}/commit"
    )
    assert committed.status_code == 200
    artifact = committed.json()
    assert artifact["datasetName"] == "beta"
    assert artifact["rawManifest"]["selectedDatasetManifest"]["name"] == "beta"
    assert artifact["rawManifest"]["packageManifest"]["datasets"][0]["name"] == "alpha"


@pytest.mark.anyio
async def test_portable_package_commit_preserves_manifest_and_episode_attachments(
    tmp_path: Path,
) -> None:
    graph = npz_bytes(
        x=np.ones((3, 2), dtype=np.float32),
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        y=np.asarray([0, 1, 1], dtype=np.int64),
    )
    episode = npz_bytes(
        train_idx=np.asarray([0], dtype=np.int64),
        test_idx=np.asarray([1, 2], dtype=np.int64),
    )
    manifest = {
        "schemaVersion": "socialgraph-fm-dataset-package/1.0",
        "sourceFingerprint": "source-fingerprint",
        "datasets": [
            {
                "name": "Cora",
                "path": "datasets/cora/graph.npz",
                "sourceFormat": "trusted_torch_pyg",
                "license": "research-only",
                "transforms": ["preserve_source_topology"],
                "transformRecipes": [
                    {"id": "identity-v1", "graphVariant": "raw"}
                ],
                "fewShotEpisodes": [
                    {
                        "shot": "5-shot",
                        "episode": "0",
                        "path": "datasets/cora/episodes/5-shot-0.npz",
                    }
                ],
            }
        ],
    }
    package = zip_bytes(
        {
            "manifest.json": json.dumps(manifest).encode(),
            "datasets/cora/graph.npz": graph,
            "datasets/cora/episodes/5-shot-0.npz": episode,
        }
    )
    settings = Settings(dataset_storage_root=str(tmp_path / "store"))
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        inspected = await client.post(
            "/api/v1/dataset-imports/inspect",
            files={"file": ("cora.sgfm.zip", package, "application/zip")},
        )
        assert inspected.status_code == 200
        assert inspected.json()["status"] == "accepted"
        assert inspected.json()["datasetCandidates"] == ["Cora"]

        committed = await client.post(
            f"/api/v1/dataset-imports/{inspected.json()['id']}/commit"
        )

    assert committed.status_code == 200
    artifact = committed.json()
    assert artifact["datasetName"] == "Cora"
    assert artifact["rawManifest"]["license"] == "research-only"
    assert artifact["rawManifest"]["sourceFingerprint"] == "source-fingerprint"
    episodes = artifact["rawManifest"]["fewShotEpisodes"]
    assert episodes == [
        {
            "shot": "5-shot",
            "episode": "0",
            "path": "datasets/cora/episodes/5-shot-0.npz",
            "artifactPath": "episodes/5-shot-0.npz",
        }
    ]
    attachment = tmp_path / "store" / "artifacts" / artifact["id"] / "episodes" / "5-shot-0.npz"
    assert attachment.read_bytes() == episode
    assert artifact["splitSets"][0] == {
        "id": "fewshot-5-shot-0",
        "kind": "few_shot",
        "target": "node",
        "representation": "index",
        "arrays": {
            "train": "episodes/5-shot-0.npz#train_idx",
            "test": "episodes/5-shot-0.npz#test_idx",
        },
        "foldCount": 1,
        "foldCounts": [{"train": 1, "validation": 0, "test": 2}],
        "seed": None,
        "source": "episodes/5-shot-0.npz",
    }


@pytest.mark.anyio
async def test_artifact_21_readiness_resolve_and_tamper_detection(tmp_path: Path) -> None:
    graph = npz_bytes(
        x=np.eye(4, dtype=np.float32),
        edge_index=np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        y=np.asarray([0, 0, 1, 1], dtype=np.int64),
        node_id_map=np.asarray(["paper-a", "paper-b", "paper-c", "paper-d"]),
        train_mask=np.asarray([1, 1, 0, 0], dtype=np.uint8),
        val_mask=np.asarray([0, 0, 1, 0], dtype=np.uint8),
        test_mask=np.asarray([0, 0, 0, 1], dtype=np.uint8),
    )
    manifest = {
        "schemaVersion": "socialgraph-fm-dataset-package/1.0",
        "datasets": [
            {
                "name": "AuditableGraph",
                "path": "datasets/a/graph.npz",
                "datasetRole": "benchmark",
                "splitKind": "official",
                    "licensePolicy": {
                    "status": "user_attested",
                    "identifier": "private-evaluation",
                        "allowedUses": ["evaluation"],
                    },
                    "licenseEvidence": [
                        {
                            "id": "private-evaluation-attestation",
                            "kind": "user_attestation",
                            "recordedAt": "2026-08-11T00:00:00Z",
                            "recordedBy": "test-user",
                        }
                    ],
                "transformRecipes": [
                    {
                        "id": "identity-v1",
                        "graphVariant": "raw",
                        "featureTransform": "identity",
                    }
                ],
            }
        ],
    }
    package = zip_bytes(
        {
            "manifest.json": json.dumps(manifest).encode(),
            "datasets/a/graph.npz": graph,
        }
    )
    settings = Settings(dataset_storage_root=str(tmp_path / "store"))
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        inspected = await client.post(
            "/api/v1/dataset-imports/inspect",
            files={"file": ("ready.sgfm.zip", package, "application/zip")},
        )
        committed = await client.post(
            f"/api/v1/dataset-imports/{inspected.json()['id']}/commit"
        )
        assert committed.status_code == 200
        artifact = committed.json()
        assert artifact["schemaVersion"] == "2.2"
        assert artifact["datasetRole"] == "benchmark"
        assert artifact["nodeIdentity"]["kind"] == "source"
        assert artifact["splitSets"][0]["kind"] == "official"
        assert artifact["sourceFileDigests"]

        readiness = await client.get(
            f"/api/v1/dataset-artifacts/{artifact['id']}/readiness",
            params={"trainingRefHash": artifact["trainingRef"]["refHash"]},
        )
        assert readiness.status_code == 200
        assert readiness.json()["status"] == "ready"

        resolved = await client.post(
            "/api/v1/training-dataset-refs/resolve",
            json={
                "artifactId": artifact["id"],
                "contentHash": artifact["contentHash"],
                "graphVariant": "raw",
                "splitSetId": "source-splits",
                "featureRecipeId": "identity-v1",
                "taskSpecId": "node-classification-v1",
                "intendedUse": "evaluation",
            },
        )
        assert resolved.status_code == 200
        assert resolved.json()["readiness"]["status"] == "ready"
        assert len(resolved.json()["reference"]["refHash"]) == 64

        tensor = tmp_path / "store" / "artifacts" / artifact["id"] / "graph.npz"
        with np.load(tensor, allow_pickle=False) as archive:
            tampered_arrays = {name: np.asarray(archive[name]) for name in archive.files}
        tampered_arrays["x"] = tampered_arrays["x"].copy()
        tampered_arrays["x"][0, 0] = 99
        np.savez_compressed(tensor, **tampered_arrays)
        corrupt = await client.get(
            f"/api/v1/dataset-artifacts/{artifact['id']}/readiness"
        )
        assert corrupt.status_code == 200
        assert corrupt.json()["status"] == "corrupt"


@pytest.mark.anyio
async def test_legacy_artifact_is_read_only_and_requires_reimport(tmp_path: Path) -> None:
    settings = Settings(dataset_storage_root=str(tmp_path / "legacy-store"))
    app = create_app(settings)
    legacy = DatasetArtifact.model_validate(
        {
            "schemaVersion": "1.0",
            "id": "legacy-artifact",
            "inspectionId": "legacy-inspection",
            "sourceFormat": "legacy",
            "sourceFiles": ["legacy.npz"],
            "checksum": "legacy-source",
            "profile": {"nodeCount": 1, "edgeCount": 0, "splitNames": [], "directed": False},
            "graphView": {
                "id": "legacy-view",
                "nodes": [{"id": "0", "label": "0"}],
                "edges": [],
                "summary": {
                    "nodeCount": 1,
                    "edgeCount": 0,
                    "density": 0,
                    "connectedComponents": 1,
                    "visibleNodeCount": 1,
                    "visibleEdgeCount": 0,
                    "partialPreview": False,
                },
            },
            "createdAt": datetime.now(UTC),
        }
    )
    app.state.dataset_imports.store.save_artifact(
        legacy,
        {
            "edge_index": np.empty((2, 0), dtype=np.int64),
            "num_nodes": np.asarray(1, dtype=np.int64),
        },
    )
    with pytest.raises(ValueError, match="不可覆盖"):
        app.state.dataset_imports.store.save_artifact(
            legacy,
            {
                "edge_index": np.empty((2, 0), dtype=np.int64),
                "num_nodes": np.asarray(1, dtype=np.int64),
            },
        )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        readiness = await client.get(
            "/api/v1/dataset-artifacts/legacy-artifact/readiness"
        )
        listing = await client.get("/api/v1/dataset-artifacts")

    assert readiness.json()["status"] == "legacy"
    assert readiness.json()["blockers"][0]["code"] == "LEGACY_ARTIFACT_REIMPORT_REQUIRED"
    assert listing.json()[0]["readinessStatus"] == "legacy"


@pytest.mark.anyio
async def test_pt_and_planetoid_pickle_are_detected_without_deserialization(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"pickle": False}

    def fail_pickle(*_args: object, **_kwargs: object) -> object:
        called["pickle"] = True
        raise AssertionError("pickle.load must not run in an HTTP request")

    monkeypatch.setattr(pickle, "load", fail_pickle)
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None:
        monkeypatch.setattr(
            torch,
            "load",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("torch.load must not run in an HTTP request")
            ),
        )

    pt_response = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={"file": ("processed/data.pt", b"not-a-real-pickle", "application/octet-stream")},
    )
    assert pt_response.status_code == 200
    assert pt_response.json()["status"] == "conversion_required"
    assert pt_response.json()["detectedFormat"] == "torch_pyg_archive"

    legacy_response = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={"file": ("raw/ind.cora.graph", b"unsafe", "application/octet-stream")},
    )
    assert legacy_response.status_code == 200
    assert legacy_response.json()["status"] == "conversion_required"
    assert legacy_response.json()["detectedFormat"] == "legacy_planetoid_pickle"
    assert called["pickle"] is False


@pytest.mark.anyio
async def test_zip_path_traversal_and_unsafe_object_npz_are_rejected(
    api_client: httpx.AsyncClient,
) -> None:
    traversal = zip_bytes({"../escape.npz": b"x"})
    bad_zip = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={"file": ("dataset.zip", traversal, "application/zip")},
    )
    assert bad_zip.status_code == 400

    unsafe = npz_bytes(
        edge_index=np.asarray([[0], [1]], dtype=np.int64),
        metadata=np.asarray([{"secret": True}], dtype=object),
    )
    bad_npz = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={"file": ("unsafe.npz", unsafe, "application/octet-stream")},
    )
    assert bad_npz.status_code == 200
    assert bad_npz.json()["status"] == "rejected"
    assert bad_npz.json()["issues"][0]["code"] == "INVALID_SAFE_NPZ"


@pytest.mark.anyio
async def test_upload_limit_is_enforced_before_adapter_runs() -> None:
    settings = Settings(dataset_upload_max_bytes=1024, dataset_archive_max_bytes=2048)
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/dataset-imports/inspect",
            files={"file": ("large.npz", b"x" * 1025, "application/octet-stream")},
        )
    assert response.status_code == 413


def test_cli_refuses_all_input_without_explicit_trust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "data"
    source.mkdir()
    monkeypatch.setattr(
        pickle,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not load")),
    )
    exit_code = dataset_tools_main(
        [
            "convert-pyg",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "result.sgfm.zip"),
        ]
    )
    assert exit_code == 2
    assert not (tmp_path / "result.sgfm.zip").exists()


def test_cli_converts_safe_geom_source_into_importable_package(tmp_path: Path) -> None:
    source = tmp_path / "data"
    raw = source / "toy" / "raw"
    raw.mkdir(parents=True)
    (raw / "out1_node_feature_label.txt").write_bytes(
        b"node_id\tfeature\tlabel\n0\t1,0\t0\n1\t0,1\t1\n"
    )
    (raw / "out1_graph_edges.txt").write_bytes(b"node_id\tnode_id\n0\t1\n")
    output = tmp_path / "result.sgfm.zip"

    exit_code = dataset_tools_main(
        [
            "convert-pyg",
            "--input",
            str(source),
            "--output",
            str(output),
            "--trust-pickle",
        ]
    )

    assert exit_code == 0
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["schemaVersion"] == "socialgraph-fm-dataset-package/1.0"
        assert manifest["datasets"][0]["name"] == "toy"
        graph_payload = archive.read("datasets/toy/graph.npz")
        assert graph_payload
        with np.load(io.BytesIO(graph_payload), allow_pickle=False) as arrays:
            assert arrays["node_id_map"].tolist() == ["0", "1"]
            assert bool(arrays["directed"]) is False
        assert len(manifest["datasets"][0]["sourceFileDigests"]) == 2


def test_cli_packages_all_cora_fewshot_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "data"
    raw = source / "Cora" / "raw"
    raw.mkdir(parents=True)
    (raw / "out1_node_feature_label.txt").write_bytes(
        b"node_id\tfeature\tlabel\n0\t1,0\t0\n1\t0,1\t1\n"
    )
    (raw / "out1_graph_edges.txt").write_bytes(b"node_id\tnode_id\n0\t1\n")
    for episode in range(51):
        episode_dir = source / "fewshot_cora" / "5-shot_cora" / str(episode)
        episode_dir.mkdir(parents=True)
        for name in ("idx.pt", "labels.pt", "split.pt"):
            (episode_dir / name).write_bytes(b"trusted-test-placeholder")

    monkeypatch.setattr(
        dataset_tools,
        "_trusted_torch_arrays",
        lambda path: {path.stem: np.asarray([0, 1], dtype=np.int64)},
    )
    output = tmp_path / "episodes.sgfm.zip"
    manifest = dataset_tools.convert_pyg_dataset(source, output, trust_pickle=True)
    cora = manifest["datasets"][0]
    assert cora["name"] == "Cora"
    assert len(cora["fewShotEpisodes"]) == 51
    assert cora["fewShotEpisodes"][0]["splitSetId"] == "fewshot-5-shot_cora-0"
    assert len(cora["fewShotEpisodes"][0]["sha256"]) == 64
    with zipfile.ZipFile(output) as archive:
        episode_files = [name for name in archive.namelist() if "/episodes/" in name]
        assert len(episode_files) == 51
