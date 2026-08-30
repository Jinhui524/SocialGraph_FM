from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from socialgraph_gfm.core.adapters import (
    BundleInputAdapter,
    MultiHotFieldSchema,
    derive_training_selection,
    fit_adapter_schema,
)
from socialgraph_gfm.core.bundle import CoreGraphBundle, calculate_graph_version_hash
from socialgraph_gfm.canonical import canonical_sha256
import socialgraph_gfm.core.serving_control as serving_control
from socialgraph_gfm.core.serving_control import ServingControlStore
from socialgraph_gfm.core.inference_contracts import (
    InternalCreateRunReceipt,
    InternalCreateRunRequest,
)
from socialgraph_gfm.core.inference_service import RunStore, ServingControlStaleError
from tests.test_core_inference_fix_round1 import (
    _catalog,
    _serving_registry,
    _wait_terminal,
)
from _core_inference_test_support import _make_test_internal_create_request


def _publish_control(
    path: Path,
    *,
    root: Path,
    registry_path: Path,
    catalog_path: Path,
    generation: int,
) -> dict[str, object]:
    registry_bytes = registry_path.read_bytes()
    catalog_bytes = catalog_path.read_bytes()
    registry = json.loads(registry_bytes)
    catalog = json.loads(catalog_bytes)
    versions = root / "serving-control-versions"
    versions.mkdir(exist_ok=True)
    versioned_registry = versions / (
        f"registry.g{registry['generation']}.{hashlib.sha256(registry_bytes).hexdigest()[:16]}.json"
    )
    versioned_catalog = versions / (
        f"catalog.g{catalog['generation']}.{hashlib.sha256(catalog_bytes).hexdigest()[:16]}.json"
    )
    if versioned_registry.exists():
        assert versioned_registry.read_bytes() == registry_bytes
    else:
        versioned_registry.write_bytes(registry_bytes)
    if versioned_catalog.exists():
        assert versioned_catalog.read_bytes() == catalog_bytes
    else:
        versioned_catalog.write_bytes(catalog_bytes)
    payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-serving-control/1.0",
        "generation": generation,
        "registry": {
            "relativePath": versioned_registry.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(registry_bytes).hexdigest(),
            "semanticHash": canonical_sha256(registry),
            "generation": registry["generation"],
        },
        "catalog": {
            "relativePath": versioned_catalog.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "semanticHash": canonical_sha256(catalog),
            "generation": catalog["generation"],
        },
    }
    payload["controlHash"] = canonical_sha256(payload)
    replacement = path.with_suffix(".replacement")
    replacement.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(replacement, path)
    return payload


def _bundle(*, assignments: list[dict[str, str]], strategy: str = "official") -> CoreGraphBundle:
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [
            {"id": "a", "index": 0},
            {"id": "b", "index": 1},
            {"id": "c", "index": 2},
            {"id": "d", "index": 3},
        ],
        "edges": [
            {"sourceId": "a", "targetId": "b", "edgeType": "link", "weight": 1.0},
            {"sourceId": "b", "targetId": "c", "edgeType": "link", "weight": 1.0},
            {"sourceId": "c", "targetId": "d", "edgeType": "link", "weight": 1.0},
        ],
        "nodeFeatures": [
            {"kind": "numeric", "name": "value", "values": [1.0, 3.0, 100.0, 200.0]},
            {
                "kind": "multiHot",
                "name": "tags",
                "rowOffsets": [0, 1, 2, 3, 4],
                "values": ["known-a", "known-b", "held-validation", "held-test"],
            },
        ],
        "structuralFeatures": {
            "names": ["degree"],
            "values": [[1.0], [2.0], [2.0], [1.0]],
        },
        "source": {"sourceName": "test", "sourceSha256": "a" * 64},
        "splitManifest": {"strategy": strategy, "assignments": assignments},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _node_split() -> CoreGraphBundle:
    return _bundle(
        assignments=[
            {"entityId": "a", "role": "train"},
            {"entityId": "b", "role": "train"},
            {"entityId": "c", "role": "validation"},
            {"entityId": "d", "role": "test"},
        ]
    )


def _wide_multi_hot_bundle(token_count: int) -> CoreGraphBundle:
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [{"id": "a", "index": 0}],
        "edges": [],
        "nodeFeatures": [
            {
                "kind": "multiHot",
                "name": "sharedAttributes",
                "rowOffsets": [0, token_count],
                "values": [f"attribute-{index:04d}" for index in range(token_count)],
            }
        ],
        "source": {"sourceName": "test", "sourceSha256": "a" * 64},
        "splitManifest": {
            "strategy": "official",
            "assignments": [{"entityId": "a", "role": "train"}],
        },
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def test_adapter_fit_derives_node_rows_visible_topology_and_known_multihot_tokens() -> None:
    bundle = _node_split()
    selection = derive_training_selection(bundle)
    assert selection.fit_row_ids == ("a", "b")
    assert selection.visible_edge_indices == (0,)

    schema = fit_adapter_schema(bundle, multi_hot_buckets=16)
    assert schema.schema_version == "socialgraph-fm.core-adapter-schema/1.1"
    assert schema.fit_row_count == 2
    assert schema.visible_topology_edge_count == 1
    multi_hot = next(field for field in schema.fields if isinstance(field, MultiHotFieldSchema))
    assert len(multi_hot.known_token_digests) == 2

    with pytest.raises(ValueError, match="authoritative split"):
        fit_adapter_schema(bundle, train_row_ids=("a", "c"), multi_hot_buckets=16)
    with pytest.raises(ValueError, match="authoritative split"):
        fit_adapter_schema(bundle, visible_edge_indices=(0, 1), multi_hot_buckets=16)


def test_adapter_schema_accepts_real_twitch_multihot_cardinality() -> None:
    schema = fit_adapter_schema(_wide_multi_hot_bundle(2_545), multi_hot_buckets=256)
    multi_hot = next(field for field in schema.fields if isinstance(field, MultiHotFieldSchema))

    assert len(multi_hot.known_token_digests) == 2_545
    assert len(schema.model_dump_json(by_alias=True).encode("utf-8")) < 256 * 1024


def test_adapter_fit_derives_edge_split_and_rejects_ambiguous_or_partial_splits() -> None:
    edge_bundle = _bundle(
        strategy="spanning-forest-80-10-10",
        assignments=[
            {"entityId": "edge:a:b", "role": "train"},
            {"entityId": "edge:b:c", "role": "validation"},
            {"entityId": "edge:c:d", "role": "test"},
        ],
    )
    selection = derive_training_selection(edge_bundle)
    assert selection.fit_row_ids == ("a", "b")
    assert selection.visible_edge_indices == (0,)

    partial = _bundle(assignments=[{"entityId": "a", "role": "train"}])
    with pytest.raises(ValueError, match="complete node or edge inventory"):
        derive_training_selection(partial)

    unsupported = _bundle(assignments=[], strategy="graph-disjoint")
    with pytest.raises(ValueError, match="unsupported"):
        derive_training_selection(unsupported)


def test_empty_split_is_full_training_but_empty_train_role_fails_closed() -> None:
    full = _bundle(assignments=[])
    selection = derive_training_selection(full)
    assert selection.fit_row_ids == ("a", "b", "c", "d")
    assert selection.visible_edge_indices == (0, 1, 2)

    no_train = _bundle(
        assignments=[
            {"entityId": "a", "role": "validation"},
            {"entityId": "b", "role": "validation"},
            {"entityId": "c", "role": "test"},
            {"entityId": "d", "role": "test"},
        ]
    )
    with pytest.raises(ValueError, match="at least one training row"):
        derive_training_selection(no_train)


def test_multihot_validation_and_test_tokens_are_oov_zero_in_target_mode() -> None:
    source = _node_split()
    schema = fit_adapter_schema(source, multi_hot_buckets=16)
    training = BundleInputAdapter(source, schema=schema, mode="training")
    training_indices = training._field_1_indices.tolist()
    assert training_indices[0] != 0 and training_indices[1] != 0
    assert training_indices[2:] == [0, 0]

    target_payload = source.model_dump(mode="python", by_alias=True)
    target_payload["nodeFeatures"][1]["values"] = [
        "known-a",
        "new-target",
        "held-validation",
        "held-test",
    ]
    target_payload["graphVersionHash"] = calculate_graph_version_hash(target_payload)
    target = CoreGraphBundle.model_validate(target_payload)
    inference = BundleInputAdapter(target, schema=schema, mode="inference")
    assert inference._field_1_indices.tolist()[0] != 0
    assert inference._field_1_indices.tolist()[1:] == [0, 0, 0]


def test_training_resume_revalidates_source_provenance_but_inference_accepts_target_graph() -> None:
    source = _node_split()
    schema = fit_adapter_schema(source, multi_hot_buckets=16)
    BundleInputAdapter(source, schema=schema, mode="training")

    changed_payload = copy.deepcopy(source.model_dump(mode="json", by_alias=True))
    changed_payload["nodeFeatures"][0]["values"][0] = 9.0
    changed_payload["graphVersionHash"] = calculate_graph_version_hash(changed_payload)
    changed = CoreGraphBundle.model_validate(changed_payload)
    with pytest.raises(ValueError, match="source training provenance"):
        BundleInputAdapter(changed, schema=schema, mode="training")
    BundleInputAdapter(changed, schema=schema, mode="inference")


@pytest.mark.parametrize(
    "invalid_state",
    [
        "missing-adapter",
        "unexpected-adapter",
        "bad-adapter-shape",
        "legacy-row-buffer",
        "missing-model",
    ],
)
def test_capabilities_require_strict_model_and_adapter_state_load(
    tmp_path, invalid_state: str
) -> None:
    catalog, _reference, _bundle_path = _catalog(tmp_path)
    registry = _serving_registry(
        tmp_path, catalog=catalog, invalid_state=invalid_state
    )
    with pytest.raises(ValueError, match="serving checkpoint.*state"):
        registry.capabilities()


def test_serving_control_acquires_one_bound_registry_catalog_snapshot(tmp_path: Path) -> None:
    catalog, _reference, _bundle_path = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control_path = tmp_path / "serving-control.json"
    expected = _publish_control(
        control_path,
        root=tmp_path,
        registry_path=registry.path,
        catalog_path=catalog.path,
        generation=1,
    )
    snapshot = ServingControlStore.load(
        control_path, high_water_root=tmp_path / "high-water"
    ).acquire()
    assert snapshot.document.control_hash == expected["controlHash"]
    assert snapshot.registry_document.models[0].model_version_id == "socialgraph-fm-core/review"
    assert snapshot.catalog_document.artifacts[0].artifact_id == "artifact-v1"


def test_serving_control_rejects_duplicate_model_ids_before_capability_use(
    tmp_path: Path,
) -> None:
    catalog, _reference, _bundle_path = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    duplicate_path = tmp_path / "serving-registry.duplicate.json"
    duplicate = json.loads(registry.path.read_bytes())
    duplicate["models"].append(copy.deepcopy(duplicate["models"][0]))
    duplicate_path.write_text(
        json.dumps(duplicate, separators=(",", ":")), encoding="utf-8"
    )
    control_path = tmp_path / "serving-control.json"
    _publish_control(
        control_path,
        root=tmp_path,
        registry_path=duplicate_path,
        catalog_path=catalog.path,
        generation=1,
    )

    with pytest.raises(ValueError, match="modelVersionId values must be unique"):
        ServingControlStore.load(
            control_path, high_water_root=tmp_path / "high-water"
        ).acquire()


def test_serving_control_rejects_rollback_and_same_generation_catalog_fork(
    tmp_path: Path,
) -> None:
    catalog, _reference, _bundle_path = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control_path = tmp_path / "serving-control.json"
    _publish_control(
        control_path,
        root=tmp_path,
        registry_path=registry.path,
        catalog_path=catalog.path,
        generation=2,
    )
    store = ServingControlStore.load(control_path, high_water_root=tmp_path / "high-water")
    store.acquire()

    _publish_control(
        control_path,
        root=tmp_path,
        registry_path=registry.path,
        catalog_path=catalog.path,
        generation=1,
    )
    with pytest.raises(ValueError, match="rollback"):
        store.acquire()

    fork_path = tmp_path / "artifact-catalog.fork.json"
    fork = json.loads(catalog.path.read_bytes())
    fork["artifacts"] = []
    fork_path.write_text(json.dumps(fork, separators=(",", ":")), encoding="utf-8")
    _publish_control(
        control_path,
        root=tmp_path,
        registry_path=registry.path,
        catalog_path=fork_path,
        generation=2,
    )
    with pytest.raises(ValueError, match="same-generation fork"):
        store.acquire()


def test_serving_control_retries_to_one_new_control_after_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog, _reference, _bundle_path = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control_path = tmp_path / "serving-control.json"
    _publish_control(
        control_path,
        root=tmp_path,
        registry_path=registry.path,
        catalog_path=catalog.path,
        generation=1,
    )
    replaced = False

    def replace_once(_stage: str) -> None:
        nonlocal replaced
        if not replaced:
            replaced = True
            _publish_control(
                control_path,
                root=tmp_path,
                registry_path=registry.path,
                catalog_path=catalog.path,
                generation=2,
            )

    monkeypatch.setattr(serving_control, "_CONTROL_ACQUIRE_SEAM", replace_once)
    snapshot = ServingControlStore.load(
        control_path, high_water_root=tmp_path / "high-water"
    ).acquire()
    assert snapshot.document.generation == 2


def test_serving_control_is_not_accepted_when_durable_high_water_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog, _reference, _bundle_path = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control_path = tmp_path / "serving-control.json"
    _publish_control(
        control_path,
        root=tmp_path,
        registry_path=registry.path,
        catalog_path=catalog.path,
        generation=1,
    )
    store = ServingControlStore.load(control_path, high_water_root=tmp_path / "high-water")

    def fail_write(_path: Path, _payload: dict[str, object]) -> None:
        raise OSError("injected durable publication failure")

    monkeypatch.setattr(serving_control, "_atomic_private_json", fail_write)
    with pytest.raises(OSError, match="durable publication"):
        store.acquire()
    assert not store.high_water_path.exists()


def test_expected_control_create_returns_bound_receipt_and_stale_creates_no_run(
    tmp_path: Path,
) -> None:
    catalog, reference, _bundle_path = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control_path = tmp_path / "serving-control.json"
    _publish_control(
        control_path,
        root=tmp_path,
        registry_path=registry.path,
        catalog_path=catalog.path,
        generation=1,
    )
    control_store = ServingControlStore.load(
        control_path, high_water_root=tmp_path / "high-water"
    )
    control = control_store.acquire()
    legacy = _make_test_internal_create_request(reference, control_store).model_dump(
        mode="json", by_alias=True
    )
    envelope = InternalCreateRunRequest.model_validate(legacy)
    store = RunStore(
        tmp_path / "inference",
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control_store,
    )
    created = store.create(envelope)
    assert isinstance(created, InternalCreateRunReceipt)
    assert created.execution_snapshot.control_hash == control.document.control_hash
    assert created.execution_snapshot.artifact_catalog_hash == control.catalog_hash
    _wait_terminal(store, created.status.run_id)

    stale = copy.deepcopy(legacy)
    stale["expectedServingControl"]["controlHash"] = "f" * 64
    before = tuple((tmp_path / "inference" / "runs").iterdir())
    with pytest.raises(ServingControlStaleError):
        store.create(InternalCreateRunRequest.model_validate(stale))
    assert tuple((tmp_path / "inference" / "runs").iterdir()) == before
