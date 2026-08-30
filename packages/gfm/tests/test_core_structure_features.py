from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.core.adapters import BundleInputAdapter, fit_adapter_schema
from socialgraph_gfm.core.bundle import CoreGraphBundle, calculate_graph_version_hash
from socialgraph_gfm.core.structure_features import (
    STRUCTURE_FEATURE_NAMES,
    StructureAlgorithmConfig,
    build_structure_cache,
    compute_structure_rows,
    enrich_bundle_with_structure,
    load_structure_cache,
    verify_adapter_structure_binding,
)


def _bundle(
    *,
    directed: bool,
    edges: list[tuple[str, str]],
    assignments: list[dict[str, str]] | None = None,
    strategy: str = "official",
) -> CoreGraphBundle:
    node_ids = sorted({endpoint for edge in edges for endpoint in edge})
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": directed,
        "nodes": [
            {"id": identifier, "index": index}
            for index, identifier in enumerate(node_ids)
        ],
        "edges": [
            {
                "sourceId": left,
                "targetId": right,
                "edgeType": "relation",
                "weight": 1.0,
            }
            for left, right in edges
        ],
        "nodeFeatures": [],
        "structuralFeatures": None,
        "source": {"sourceName": "literal", "sourceSha256": "a" * 64},
        "splitManifest": {
            "strategy": strategy,
            "assignments": assignments or [],
        },
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _directed_triangle_tail() -> CoreGraphBundle:
    return _bundle(
        directed=True,
        edges=[
            ("a", "b"),
            ("b", "a"),
            ("a", "c"),
            ("c", "a"),
            ("b", "c"),
            ("c", "b"),
            ("c", "d"),
        ],
    )


def test_fixed_structure_kernel_has_hand_derived_directed_semantics() -> None:
    bundle = _directed_triangle_tail()
    rows = compute_structure_rows(
        bundle,
        visible_edge_indices=tuple(range(len(bundle.edges))),
        config=StructureAlgorithmConfig.fixed(),
    )
    by_name = {name: rows[:, index] for index, name in enumerate(STRUCTURE_FEATURE_NAMES)}

    np.testing.assert_allclose(by_name["degree"], [2, 2, 3, 1])
    np.testing.assert_allclose(by_name["in-degree"], [2, 2, 2, 1])
    np.testing.assert_allclose(by_name["out-degree"], [2, 2, 3, 0])
    np.testing.assert_allclose(by_name["triangle-count"], [1, 1, 1, 0])
    np.testing.assert_allclose(by_name["clustering"], [1, 1, 1 / 3, 0])
    np.testing.assert_allclose(by_name["ego-density"], [1, 1, 2 / 3, 1])
    np.testing.assert_allclose(by_name["k-core"], [2, 2, 2, 1])
    np.testing.assert_allclose(by_name["reciprocity"], [1, 1, 2 / 3, 0])
    np.testing.assert_allclose(by_name["component-size-fraction"], [1, 1, 1, 1])
    np.testing.assert_allclose(by_name["two-hop-size"], [1, 1, 0, 2])
    np.testing.assert_allclose(by_name["mean-neighbor-degree"], [2.5, 2.5, 5 / 3, 3])
    assert np.all(np.isfinite(rows))
    assert np.isclose(by_name["pagerank"].sum(), 1.0)


def test_rwse_is_counter_deterministic_and_obvious_on_two_node_graph() -> None:
    bundle = _bundle(directed=False, edges=[("a", "b")])
    first = compute_structure_rows(
        bundle, visible_edge_indices=(0,), config=StructureAlgorithmConfig.fixed()
    )
    second = compute_structure_rows(
        bundle, visible_edge_indices=(0,), config=StructureAlgorithmConfig.fixed()
    )
    assert first.dtype == np.dtype("<f4")
    assert first.flags.c_contiguous
    assert first.tobytes() == second.tobytes()
    columns = {name: index for index, name in enumerate(STRUCTURE_FEATURE_NAMES)}
    np.testing.assert_array_equal(first[:, columns["rwse-1"]], [0, 0])
    np.testing.assert_array_equal(first[:, columns["rwse-2"]], [1, 1])
    np.testing.assert_array_equal(first[:, columns["rwse-4"]], [1, 1])
    np.testing.assert_array_equal(first[:, columns["rwse-8"]], [1, 1])


def test_training_cache_uses_only_authoritative_visible_topology(tmp_path: Path) -> None:
    assignments = [
        {"entityId": "a", "role": "train"},
        {"entityId": "b", "role": "train"},
        {"entityId": "c", "role": "train"},
        {"entityId": "d", "role": "validation"},
    ]
    first_bundle = _bundle(
        directed=False,
        edges=[("a", "b"), ("b", "c"), ("c", "d")],
        assignments=assignments,
    )
    second_bundle = _bundle(
        directed=False,
        edges=[("a", "b"), ("b", "c"), ("a", "d")],
        assignments=assignments,
    )
    first = build_structure_cache(
        first_bundle, cache_root=tmp_path / "one", role="training"
    )
    second = build_structure_cache(
        second_bundle, cache_root=tmp_path / "two", role="training"
    )

    assert first.manifest.visible_topology_edge_count == 2
    assert first.manifest.visible_topology_hash == second.manifest.visible_topology_hash
    assert first.manifest.fit_row_ids_hash == second.manifest.fit_row_ids_hash
    assert first.rows.tobytes() == second.rows.tobytes()
    assert first.manifest.transform_hash == second.manifest.transform_hash
    assert first.manifest.base_graph_version_hash != second.manifest.base_graph_version_hash


def test_cache_is_exact_reusable_and_tampering_fails_closed(tmp_path: Path) -> None:
    bundle = _directed_triangle_tail()
    artifact = build_structure_cache(bundle, cache_root=tmp_path, role="training")
    before_manifest = artifact.manifest_path.read_bytes()
    before_npz = artifact.npz_path.read_bytes()
    repeated = build_structure_cache(bundle, cache_root=tmp_path, role="training")

    assert repeated.manifest == artifact.manifest
    assert repeated.manifest_path.read_bytes() == before_manifest
    assert repeated.npz_path.read_bytes() == before_npz
    assert artifact.manifest.tensor_digest == canonical_sha256(
        {
            "dtype": "float32-le",
            "shape": list(artifact.rows.shape),
            "sha256": hashlib.sha256(artifact.rows.tobytes()).hexdigest(),
        }
    )

    artifact.npz_path.write_bytes(before_npz + b"tamper")
    with pytest.raises(ValueError, match="NPZ byte hash"):
        load_structure_cache(bundle, cache_root=tmp_path, role="training")
    with pytest.raises(ValueError, match="NPZ byte hash"):
        build_structure_cache(bundle, cache_root=tmp_path, role="training")


def test_manifest_mutation_and_wrong_shape_are_rejected(tmp_path: Path) -> None:
    bundle = _directed_triangle_tail()
    artifact = build_structure_cache(bundle, cache_root=tmp_path, role="training")
    raw = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    raw["visibleTopologyEdgeCount"] += 1
    raw["manifestHash"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "manifestHash"}
    )
    artifact.manifest_path.write_text(canonical_json(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cache binding"):
        load_structure_cache(bundle, cache_root=tmp_path, role="training")

    second_root = tmp_path / "wrong-shape"
    artifact = build_structure_cache(bundle, cache_root=second_root, role="training")
    with np.load(artifact.npz_path, allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    arrays["rows"] = arrays["rows"][:-1]
    replacement = io.BytesIO()
    np.savez(replacement, **arrays)
    npz_bytes = replacement.getvalue()
    artifact.npz_path.write_bytes(npz_bytes)
    raw = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    raw["npzSha256"] = hashlib.sha256(npz_bytes).hexdigest()
    raw["manifestHash"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "manifestHash"}
    )
    artifact.manifest_path.write_text(canonical_json(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row shape"):
        load_structure_cache(bundle, cache_root=second_root, role="training")


def test_email_style_structure_only_bundle_fits_train_only_adapter(tmp_path: Path) -> None:
    bundle = _bundle(
        directed=False,
        edges=[("a", "b"), ("b", "c")],
        strategy="spanning-forest-80-10-10",
        assignments=[
            {"entityId": "edge:a:b", "role": "train"},
            {"entityId": "edge:b:c", "role": "validation"},
        ],
    )
    cache = build_structure_cache(bundle, cache_root=tmp_path, role="training")
    enriched = enrich_bundle_with_structure(bundle, cache)
    schema = fit_adapter_schema(enriched)
    verify_adapter_structure_binding(cache, schema)
    adapter = BundleInputAdapter(enriched, schema=schema, mode="training")

    assert adapter.field_names == ("structure-view",)
    assert adapter().shape == (3, 128)
    structure = schema.fields[0]
    assert structure.model_dump(mode="json")["means"] == pytest.approx(
        cache.manifest.train_means
    )
    assert structure.model_dump(mode="json")["scales"] == pytest.approx(
        cache.manifest.train_scales
    )


def test_inference_uses_source_transform_over_complete_target_topology(
    tmp_path: Path,
) -> None:
    source = _bundle(directed=False, edges=[("a", "b")])
    source_cache = build_structure_cache(
        source, cache_root=tmp_path / "source", role="training"
    )
    source_enriched = enrich_bundle_with_structure(source, source_cache)
    schema = fit_adapter_schema(source_enriched)
    target = _bundle(
        directed=False, edges=[("w", "x"), ("x", "y"), ("y", "z")]
    )
    target_cache = build_structure_cache(
        target, cache_root=tmp_path / "target", role="inference"
    )
    target_enriched = enrich_bundle_with_structure(target, target_cache)
    adapter = BundleInputAdapter(target_enriched, schema=schema, mode="inference")
    structure_schema = schema.fields[0]
    raw = target_cache.rows
    expected = (raw - np.asarray(structure_schema.means)) / np.asarray(
        structure_schema.scales
    )

    np.testing.assert_allclose(adapter._field_0_values.numpy(), expected, rtol=1e-6)
    assert target_cache.manifest.transform_hash != source_cache.manifest.transform_hash
