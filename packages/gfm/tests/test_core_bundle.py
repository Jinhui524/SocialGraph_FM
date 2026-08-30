import json
import math

import pytest
from pydantic import ValidationError

from socialgraph_gfm.core.bundle import (
    CoreGraphBundle,
    calculate_graph_version_hash,
    load_core_graph_bundle_json,
)


def _payload() -> dict:
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [
            {"id": "alice", "index": 0},
            {"id": "bob", "index": 1},
            {"id": "carol", "index": 2},
        ],
        "edges": [
            {"sourceId": "bob", "targetId": "carol", "edgeType": "collaborates", "weight": 2.0},
            {"sourceId": "alice", "targetId": "bob", "edgeType": "collaborates", "weight": 1.0},
        ],
        "nodeFeatures": [
            {"kind": "numeric", "name": "activity", "values": [1.0, 2.0, 3.0]},
            {"kind": "categorical", "name": "team", "values": ["red", "blue", None]},
            {
                "kind": "multiHot",
                "name": "skills",
                "rowOffsets": [0, 2, 3, 3],
                "values": ["python", "graphs", "graphs"],
            },
        ],
        "structuralFeatures": {
            "names": ["degree", "clustering"],
            "values": [[1.0, 0.0], [2.0, 0.5], [1.0, 0.0]],
        },
        "source": {
            "sourceName": "tiny-fixture",
            "sourceUri": "https://example.invalid/tiny",
            "sourceSha256": "1" * 64,
        },
        "splitManifest": {
            "strategy": "official",
            "assignments": [
                {"entityId": "alice", "role": "train"},
                {"entityId": "bob", "role": "validation"},
                {"entityId": "carol", "role": "test"},
            ],
        },
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return payload


def test_bundle_hash_is_independent_of_edge_input_order_but_binds_edge_meaning():
    first = _payload()
    reordered = _payload()
    reordered["edges"] = list(reversed(reordered["edges"]))
    reordered["graphVersionHash"] = calculate_graph_version_hash(reordered)
    assert reordered["graphVersionHash"] == first["graphVersionHash"]
    assert CoreGraphBundle.model_validate(reordered).graph_version_hash == first["graphVersionHash"]

    changed_direction = _payload()
    changed_direction["directed"] = True
    assert calculate_graph_version_hash(changed_direction) != first["graphVersionHash"]
    changed_weight = _payload()
    changed_weight["edges"][0]["weight"] = 2.5
    assert calculate_graph_version_hash(changed_weight) != first["graphVersionHash"]
    reversed_directed_edge = _payload()
    reversed_directed_edge["directed"] = True
    reversed_directed_edge["edges"][0].update(sourceId="carol", targetId="bob")
    assert calculate_graph_version_hash(reversed_directed_edge) != calculate_graph_version_hash(
        changed_direction
    )
    changed_provenance = _payload()
    changed_provenance["source"]["sourceSha256"] = "2" * 64
    assert calculate_graph_version_hash(changed_provenance) == first["graphVersionHash"]


def test_omitted_and_explicit_default_edge_weight_have_identical_accepted_hashes():
    explicit = _payload()
    omitted = _payload()
    del omitted["edges"][1]["weight"]
    omitted["graphVersionHash"] = calculate_graph_version_hash(omitted)

    assert omitted["graphVersionHash"] == explicit["graphVersionHash"]
    explicit_bundle = load_core_graph_bundle_json(json.dumps(explicit))
    omitted_bundle = load_core_graph_bundle_json(json.dumps(omitted))
    assert omitted_bundle.edges[1].weight == 1.0
    assert omitted_bundle.graph_version_hash == explicit_bundle.graph_version_hash


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(directed="false"),
        lambda value: value["edges"][0].update(weight="2.0"),
        lambda value: value["nodeFeatures"][0].update(values=["1.0", 2.0, 3.0]),
        lambda value: value["nodes"][1].update(index="1"),
        lambda value: value["nodeFeatures"][2].update(rowOffsets=[0, "2", 3, 3]),
    ],
)
def test_json_loader_rejects_coercive_scalar_types(mutate):
    payload = _payload()
    mutate(payload)
    with pytest.raises(ValidationError, match="boolean|number|integer"):
        load_core_graph_bundle_json(json.dumps(payload))


def test_strict_json_loader_accepts_integer_json_numbers_for_float_fields():
    payload = _payload()
    payload["edges"][0]["weight"] = 2
    payload["nodeFeatures"][0]["values"] = [1, 2, 3]
    payload["structuralFeatures"]["values"] = [[1, 0], [2, 0.5], [1, 0]]
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    bundle = load_core_graph_bundle_json(json.dumps(payload))
    assert bundle.edges[0].weight == 2.0
    assert bundle.node_features[0].values == (1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value["nodes"].append({"id": "alice", "index": 3}), "unique"),
        (lambda value: value["nodes"][1].update(index=2), "indices"),
        (lambda value: value["edges"][0].update(targetId="missing"), "endpoint"),
        (lambda value: value["nodeFeatures"][0].update(values=[1.0]), "row count"),
        (lambda value: value["nodeFeatures"][0].update(values=[math.inf, 2.0, 3.0]), "finite"),
        (lambda value: value["splitManifest"]["assignments"][0].update(role="future"), "role"),
        (lambda value: value["edges"][0].update(sourceId="carol", targetId="bob"), "canonical"),
    ],
)
def test_bundle_rejects_malformed_semantics(mutate, match):
    payload = _payload()
    mutate(payload)
    if match != "finite":
        payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    with pytest.raises((ValidationError, ValueError), match=match):
        CoreGraphBundle.model_validate(payload)


def test_bundle_rejects_unknown_fields_unsafe_objects_and_forged_hash():
    unknown = _payload()
    unknown["picklePayload"] = "gASVunsafe"
    with pytest.raises(ValidationError, match="Extra inputs"):
        CoreGraphBundle.model_validate(unknown)

    unsafe = _payload()
    unsafe["nodes"][0]["id"] = object()
    with pytest.raises((ValidationError, TypeError, ValueError)):
        CoreGraphBundle.model_validate(unsafe)

    forged = _payload()
    forged["graphVersionHash"] = "0" * 64
    with pytest.raises(ValidationError, match="graphVersionHash"):
        CoreGraphBundle.model_validate(forged)


def test_json_loader_is_strict_and_preserves_sparse_multi_hot_rows():
    bundle = load_core_graph_bundle_json(json.dumps(_payload()))
    skills = next(feature for feature in bundle.node_features if feature.name == "skills")
    assert skills.kind == "multiHot"
    assert skills.row_offsets == (0, 2, 3, 3)
    assert not hasattr(skills, "dense_values")
