from __future__ import annotations

import copy
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import networkx as nx
import numpy as np
import pytest

from socialgraph_gfm.canonical import canonical_json, canonical_sha256, file_sha256
from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.global_model import converter
from socialgraph_gfm.global_model.contracts import (
    COUNTRY_IDS,
    GRAPH_STAT_NAMES,
    TRACE_ARRAY_TOKENS,
    GlobalCountryManifest,
    GlobalSplitDescriptor,
    atomic_write_contract,
)
from socialgraph_gfm.global_model.converter import (
    build_unlabeled_graph_stats,
    build_worker_contract,
    convert_country_in_worker,
    convert_trusted_country,
    factual_graph_fingerprint,
    graph_to_arrays,
    inspect_global_model_pickle,
    load_trusted_global_model_pickle,
    publish_corpus_manifest,
    relation_graph_to_csr,
    validate_undersampling_inventory,
    validate_undersampling_splits,
    validate_variant_factual_graph,
    validate_worker_contract,
    write_array_artifact,
)
from socialgraph_gfm.global_model.corpus import load_corpus_index, load_country_corpus


def _source_payload() -> dict[str, object]:
    graph = nx.Graph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from(((0, 1), (1, 2), (2, 3)))
    traces = {}
    for offset, name in enumerate(("coRT", "coURL", "hashSeq", "fastRT", "tweetSim")):
        trace = nx.Graph()
        trace.add_edge(offset, (offset + 1) % 6, weight=offset + 0.5)
        traces[name] = trace
    return {
        "graph": graph,
        **traces,
        "labels": np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
        "splits": {
            0: {
                "train": np.asarray([1, 1, 0, 0, 0, 0], dtype=np.bool_),
                "val": np.asarray([0, 0, 1, 1, 0, 0], dtype=np.bool_),
                "test": np.asarray([0, 0, 0, 0, 1, 1], dtype=np.bool_),
            }
        },
    }


def _write_country(root: Path, country: str) -> Path:
    root.mkdir(parents=True)
    arrays = {
        "edge_index": np.asarray(
            [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=np.int64
        ),
        "fused_indptr": np.asarray([0, 1, 3, 5, 7, 8, 8], dtype=np.int64),
        "fused_indices": np.asarray([1, 0, 2, 1, 3, 2, 4, 3], dtype=np.int64),
        "text_features": np.zeros((6, 768), dtype=np.float32),
        "degree_bucket": np.arange(6, dtype=np.uint8),
        "structure_missing": np.asarray([0, 0, 0, 0, 0, 1], dtype=np.bool_),
        "graph_stats": np.zeros(len(GRAPH_STAT_NAMES), dtype=np.float32),
        "labels": np.asarray([0, 1, 0, 1, 0, 1], dtype=np.uint8),
        "trace_membership": np.zeros((6, 5), dtype=np.bool_),
        "split_full_0_train": np.asarray([1, 1, 0, 0, 0, 0], dtype=np.bool_),
        "split_full_0_validation": np.asarray([0, 0, 1, 1, 0, 0], dtype=np.bool_),
        "split_full_0_test": np.asarray([0, 0, 0, 0, 1, 1], dtype=np.bool_),
    }
    for token in TRACE_ARRAY_TOKENS.values():
        arrays[f"relation_{token}_indptr"] = np.zeros(7, dtype=np.int64)
        arrays[f"relation_{token}_indices"] = np.empty(0, dtype=np.int64)
        arrays[f"relation_{token}_weights"] = np.empty(0, dtype=np.float64)
    descriptors = [
        write_array_artifact(root, name=name, array=value) for name, value in arrays.items()
    ]
    split = GlobalSplitDescriptor.create(
        split_id="full-fold-0",
        regime="full",
        fold=0,
        train_array="split_full_0_train",
        validation_array="split_full_0_validation",
        test_array="split_full_0_test",
    )
    manifest = GlobalCountryManifest.create(
        country_id=country,  # type: ignore[arg-type]
        node_count=6,
        edge_count=8,
        arrays=descriptors,
        splits=(split,),
        source_hashes={"pickle:full": "1" * 64, "textTensor": "2" * 64},
        relation_edge_counts={name: 0 for name in TRACE_ARRAY_TOKENS},
        preprocessing={"test": "synthetic"},
    )
    path = root / "manifest.json"
    atomic_write_contract(path, manifest)
    return path


def test_pickle_boundary_requires_trust_and_rejects_an_executable_global(tmp_path: Path) -> None:
    source = tmp_path / "official.pkl"
    source.write_bytes(pickle.dumps(_source_payload(), protocol=4))
    inspection = inspect_global_model_pickle(source)
    assert inspection.protocol == 4
    assert ("networkx.classes.graph", "Graph") in inspection.globals
    with pytest.raises(ContractViolation, match="trusted_source=True"):
        load_trusted_global_model_pickle(source)
    assert set(load_trusted_global_model_pickle(source, trusted_source=True)) == set(_source_payload())

    class Executable:
        def __reduce__(self):
            return eval, ("40 + 2",)

    malicious = tmp_path / "malicious.pkl"
    malicious.write_bytes(pickle.dumps(Executable(), protocol=4))
    with pytest.raises(ContractViolation, match="outside the exact allowlist"):
        inspect_global_model_pickle(malicious)


def test_graph_conversion_is_bidirectional_deterministic_and_retains_isolates() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from(((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (7, 7)))
    first_edges, first_buckets, first_missing = graph_to_arrays(graph)
    second_edges, second_buckets, second_missing = graph_to_arrays(graph)
    assert np.array_equal(first_edges, second_edges)
    assert np.array_equal(first_buckets, second_buckets)
    assert np.array_equal(first_missing, second_missing)
    pairs = set(map(tuple, first_edges.T.tolist()))
    assert all((target, source) in pairs for source, target in pairs)
    assert all(source != target for source, target in pairs)
    assert set(first_edges.reshape(-1)) == set(range(5))
    assert first_missing.tolist() == [False, False, False, False, False, True, True, True]
    assert first_buckets.dtype == np.uint8
    assert int(first_buckets.max()) < 128
    counts = np.bincount(first_edges[0], minlength=8)
    indptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    relation_counts = {name: 0 for name in TRACE_ARRAY_TOKENS}
    relation_counts["coRT"] = 2
    relation_counts["coURL"] = 6
    stats = build_unlabeled_graph_stats(
        graph, fused_indptr=indptr, relation_edge_counts=relation_counts
    )
    assert stats.shape == (13,)
    assert stats[2] == pytest.approx(0.5)
    assert stats[3] == pytest.approx(3 / 8)
    assert stats[8:].tolist() == pytest.approx([0.25, 0.75, 0.0, 0.0, 0.0])


def test_relation_csr_preserves_factual_weights_and_removes_self_loops() -> None:
    relation = nx.Graph()
    relation.add_nodes_from((0, 1, 2))
    relation.add_edge(0, 1, weight=2.75)
    relation.add_edge(2, 2, weight=99.0)
    indptr, indices, weights = relation_graph_to_csr(
        relation, node_count=4, trace_name="coRT"
    )
    assert indptr.tolist() == [0, 1, 2, 2, 2]
    assert indices.tolist() == [1, 0]
    assert weights.dtype == np.float64
    assert weights.tolist() == [2.75, 2.75]


def test_factual_fingerprint_rejects_changed_fused_edges_or_relation_weights() -> None:
    full = _source_payload()
    expected = factual_graph_fingerprint(full, node_count=6)

    changed_edge = copy.deepcopy(full)
    cast(nx.Graph, changed_edge["graph"]).add_edge(0, 5)
    with pytest.raises(ContractViolation, match="weighted relations differ from full"):
        validate_variant_factual_graph(
            changed_edge,
            node_count=6,
            expected_fingerprint=expected,
            regime="0.95",
        )

    changed_weight = copy.deepcopy(full)
    cast(nx.Graph, changed_weight["coRT"])[0][1]["weight"] = 9.25
    with pytest.raises(ContractViolation, match="weighted relations differ from full"):
        validate_variant_factual_graph(
            changed_weight,
            node_count=6,
            expected_fingerprint=expected,
            regime="0.95",
        )


def _split_payload(train: tuple[int, ...], validation: int, test: int) -> dict[str, Any]:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
    splits = {}
    for fold in range(5):
        train_mask = np.zeros(8, dtype=np.bool_)
        validation_mask = np.zeros(8, dtype=np.bool_)
        test_mask = np.zeros(8, dtype=np.bool_)
        train_mask[list(train)] = True
        validation_mask[validation] = True
        test_mask[test] = True
        splits[fold] = {
            "train": train_mask,
            "val": validation_mask,
            "test": test_mask,
        }
    return {"labels": labels, "splits": splits}


def test_undersampling_folds_are_bound_to_full_and_classwise_official_rounding() -> None:
    full = _split_payload((0, 1, 2, 3), validation=4, test=5)
    variant = _split_payload((0, 1), validation=4, test=5)
    records = validate_undersampling_splits(
        full, variant, node_count=8, regime="0.5"
    )
    assert len(records) == 5

    changed_validation = _split_payload((0, 1), validation=6, test=5)
    with pytest.raises(ContractViolation, match="changed validation membership"):
        validate_undersampling_splits(
            full, changed_validation, node_count=8, regime="0.5"
        )

    non_subset = _split_payload((0, 7), validation=4, test=5)
    with pytest.raises(ContractViolation, match="not a full-train subset"):
        validate_undersampling_splits(full, non_subset, node_count=8, regime="0.5")


def test_legacy_global_undersampling_is_exact_but_not_assumed_classwise() -> None:
    full = _split_payload((0, 1, 2, 3, 4, 5), validation=6, test=7)
    legacy_global = _split_payload((0, 1, 2), validation=6, test=7)
    records = validate_undersampling_splits(
        full, legacy_global, node_count=8, regime="0.5"
    )
    assert len(records) == 5

    invalid_count = _split_payload((0, 1, 2, 4), validation=6, test=7)
    with pytest.raises(ContractViolation, match="expected per-class .* legacy-global total"):
        validate_undersampling_splits(
            full, invalid_count, node_count=8, regime="0.5"
        )


def test_uae_official_half_removal_class_counts_regression() -> None:
    node_count = 5636
    labels = np.ones(node_count, dtype=np.uint8)
    labels[:3625] = 0
    full_splits = {}
    variant_splits = {}
    observed_counts = ((1829, 988), (1794, 1023), (1804, 1013), (1825, 992), (1812, 1005))
    for fold, (zero_count, one_count) in enumerate(observed_counts):
        full_train = np.zeros(node_count, dtype=np.bool_)
        full_train[:5634] = True
        variant_train = np.zeros(node_count, dtype=np.bool_)
        variant_train[:zero_count] = True
        variant_train[3625 : 3625 + one_count] = True
        validation = np.zeros(node_count, dtype=np.bool_)
        test = np.zeros(node_count, dtype=np.bool_)
        validation[5634] = True
        test[5635] = True
        full_splits[fold] = {
            "train": full_train,
            "val": validation,
            "test": test,
        }
        variant_splits[fold] = {
            "train": variant_train,
            "val": validation.copy(),
            "test": test.copy(),
        }
    records = validate_undersampling_splits(
        {"labels": labels, "splits": full_splits},
        {"labels": labels.copy(), "splits": variant_splits},
        node_count=node_count,
        regime="0.5",
    )
    assert len(records) == 5


def test_undersampling_strength_must_be_class_count_monotonic() -> None:
    full = _split_payload((0, 1, 2, 3, 4, 5), validation=6, test=7)
    half = _split_payload((0, 1, 3), validation=6, test=7)
    quarter = _split_payload((0, 2), validation=6, test=7)
    records = converter._split_masks(full, node_count=8, regime="full")
    records.extend(validate_undersampling_splits(full, half, node_count=8, regime="0.5"))
    records.extend(
        validate_undersampling_splits(full, quarter, node_count=8, regime="0.75")
    )
    with pytest.raises(ContractViolation, match="not monotonically stronger"):
        validate_undersampling_inventory(records, labels=cast(np.ndarray, full["labels"]))


def test_country_reader_verifies_hashes_shapes_splits_and_six_country_index(
    tmp_path: Path,
) -> None:
    corpus_root = tmp_path / "corpus"
    country_paths = {
        country: _write_country(corpus_root / "countries" / country, country)
        for country in COUNTRY_IDS
    }
    manifest = publish_corpus_manifest(corpus_root, country_manifest_paths=country_paths)
    index = load_corpus_index(corpus_root)
    assert index.manifest.content_hash == manifest.content_hash
    china = index.load_country("china")
    assert china.text_features.shape == (6, 768)
    assert china.edge_index.dtype == np.int64
    assert china.structure_missing.tolist() == [False, False, False, False, False, True]
    assert china.fused_csr.indices.shape == (8,)
    assert china.relation("coRT").weights.dtype == np.float64
    assert china.split("full-fold-0").validation_mask.tolist() == [0, 0, 1, 1, 0, 0]

    labels_path = country_paths["china"].parent / "arrays" / "labels.npy"
    with labels_path.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original[0] ^ 1]))
    with pytest.raises(ContractViolation, match="SHA-256"):
        load_country_corpus(country_paths["china"].parent)


def test_manifest_paths_reject_traversal() -> None:
    from socialgraph_gfm.global_model.contracts import GlobalArrayDescriptor

    with pytest.raises(ValueError, match="relative POSIX"):
        GlobalArrayDescriptor(
            name="labels",
            path="..\\labels.npy",
            sha256="0" * 64,
            dtype="|u1",
            shape=(1,),
            byteLength=1,
        )


def test_conversion_fails_before_creating_country_when_volume_has_under_20_gib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        converter.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=20 * 1024**3 - 1),
    )
    destination = tmp_path / "corpus" / "countries" / "china"
    with pytest.raises(ContractViolation, match="insufficient free space"):
        convert_trusted_country(
            country_id="china",
            pickle_sources={"full": tmp_path / "missing.pkl"},
            text_tensor_path=tmp_path / "missing.pt",
            destination=destination,
            trusted_source=True,
        )
    assert not destination.exists()
    assert not destination.parent.exists()


def _worker_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source_root = tmp_path / "source"
    country_root = source_root / "china"
    country_root.mkdir(parents=True)
    pickle_path = country_root / "0.7_datasets.pkl"
    text_path = country_root / "sbert_nodeattributes_mostPop5.pt"
    pickle_path.write_bytes(b"pickle-placeholder")
    text_path.write_bytes(b"tensor-placeholder")
    destination_root = tmp_path / "destination"
    destination_root.mkdir()
    destination = destination_root / "countries" / "china"
    return source_root, pickle_path, text_path, destination


def test_worker_contract_rejects_unknown_keys_and_source_path_escape(tmp_path: Path) -> None:
    source_root, pickle_path, text_path, destination = _worker_sources(tmp_path)
    contract = build_worker_contract(
        country_id="china",
        source_root=source_root,
        destination_root=destination.parents[1],
        pickle_sources={"full": pickle_path},
        text_tensor_path=text_path,
        destination=destination,
    )
    validate_worker_contract(contract)

    unknown = dict(contract)
    unknown["unexpected"] = True
    with pytest.raises(ContractViolation, match="unknown or missing keys"):
        validate_worker_contract(unknown)

    outside = tmp_path / "outside.pkl"
    outside.write_bytes(b"outside")
    escaped = copy.deepcopy(contract)
    escaped_sources = cast(dict[str, dict[str, object]], escaped["pickleSources"])
    escaped_sources["full"] = {
        "path": str(outside.resolve()),
        "sha256": file_sha256(outside),
        "byteLength": outside.stat().st_size,
    }
    escaped["contractHash"] = canonical_sha256(
        {key: value for key, value in escaped.items() if key != "contractHash"}
    )
    with pytest.raises(ContractViolation, match="escapes its declared root"):
        validate_worker_contract(escaped)


def test_parent_conversion_only_dispatches_worker_and_never_loads_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, pickle_path, text_path, destination = _worker_sources(tmp_path)
    monkeypatch.setattr(
        converter,
        "require_conversion_disk_space",
        lambda selected: converter.DiskSpaceInspection(
            volume_path=selected,
            free_bytes=converter.MINIMUM_CONVERSION_FREE_BYTES,
            required_bytes=converter.MINIMUM_CONVERSION_FREE_BYTES,
        ),
    )

    def forbidden_loader(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the parent process must not load pickle")

    monkeypatch.setattr(converter, "load_trusted_global_model_pickle", forbidden_loader)

    def fake_run(command: tuple[str, ...], **_kwargs: object) -> SimpleNamespace:
        from socialgraph_gfm.gfm.corpus.common import read_json_object

        request = read_json_object(Path(command[-1]))
        manifest_path = _write_country(Path(cast(str, request["destination"])), "china")
        manifest = GlobalCountryManifest.model_validate(
            __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        )
        receipt = {
            "schemaVersion": converter.WORKER_RECEIPT_SCHEMA,
            "countryId": "china",
            "contractHash": request["contractHash"],
            "manifestPath": str(manifest_path.resolve()),
            "manifestHash": manifest.content_hash,
        }
        receipt["receiptHash"] = canonical_sha256(receipt)
        return SimpleNamespace(
            returncode=0,
            stdout=canonical_json(receipt) + "\n",
            stderr="",
        )

    monkeypatch.setattr(converter.subprocess, "run", fake_run)
    receipt = convert_country_in_worker(
        country_id="china",
        source_root=source_root,
        destination_root=destination.parents[1],
        pickle_sources={"full": pickle_path},
        text_tensor_path=text_path,
        destination=destination,
        trusted_source=True,
    )
    assert receipt.manifest_path == destination / "manifest.json"
    assert receipt.country_id == "china"
