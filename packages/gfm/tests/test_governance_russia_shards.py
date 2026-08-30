from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from socialgraph_gfm.governance.materialize import load_materialized_artifact, materialize_bundle
from socialgraph_gfm.governance.russia_shards import (
    EXPECTED_FUSED_EDGE_COUNTS,
    EXPECTED_NODE_COUNTS,
    RussiaShardCatalog,
    generate_russia_shards,
    partition_russia_components,
    verify_russia_shard_catalog,
)


def _source_bundle() -> Path:
    repository = Path(__file__).resolve().parents[3]
    source = repository / "var" / "gfm" / "governance" / "samples" / "russia-replay.zip"
    if not source.is_file():
        pytest.skip("the ignored canonical Russia replay bundle is not installed")
    return source


def test_component_partition_is_deterministic_and_keeps_components_whole() -> None:
    rows = (
        (0, 1, "coRT", "1"),
        (1, 2, "coRT", "1"),
        (3, 4, "coURL", "1"),
        (5, 6, "hashSeq", "1"),
    )
    first = partition_russia_components(10, rows)
    second = partition_russia_components(10, tuple(reversed(rows)))
    assert first == second
    assert first[0] == (0, 1, 2)
    assert sorted(node for shard in first for node in shard) == list(range(10))
    owner = {node: shard for shard, nodes in enumerate(first) for node in nodes}
    assert all(owner[source] == owner[target] for source, target, _kind, _weight in rows)


def test_canonical_russia_shards_are_byte_deterministic_and_lossless(tmp_path: Path) -> None:
    source = _source_bundle()
    first_path = generate_russia_shards(source, tmp_path / "first")
    second_path = generate_russia_shards(source, tmp_path / "second")
    first = RussiaShardCatalog.model_validate_json(first_path.read_bytes())
    second = RussiaShardCatalog.model_validate_json(second_path.read_bytes())
    assert first.catalog_hash == second.catalog_hash
    assert tuple(item.node_count for item in first.shards) == EXPECTED_NODE_COUNTS
    assert tuple(item.fused_undirected_edge_count for item in first.shards) == (
        EXPECTED_FUSED_EDGE_COUNTS
    )
    assert sum(item.relation_row_count for item in first.shards) == 10_968
    assert first.full.sha256 == first.source_sha256
    assert (first_path.parent / first.full.file_name).read_bytes() == source.read_bytes()
    for left, right in zip(first.shards, second.shards, strict=True):
        assert (first_path.parent / left.file_name).read_bytes() == (
            second_path.parent / right.file_name
        ).read_bytes()
        assert set(left.relation_edge_counts) == {
            "coRT",
            "coURL",
            "hashSeq",
            "fastRT",
            "tweetSim",
        }
        with zipfile.ZipFile(first_path.parent / left.file_name) as archive:
            assert archive.namelist() == [
                "manifest.json",
                "nodes.csv",
                "relations.csv",
                "features.npz",
            ]
            assert b"label" not in archive.read("manifest.json")
            assert b"split" not in archive.read("manifest.json")
            assert b"score" not in archive.read("manifest.json")
            with np.load(archive.open("features.npz"), allow_pickle=False) as features:
                assert set(features.files) == {"node_ids", "text_features"}
                assert features["text_features"].shape == (left.node_count, 768)
                assert features["text_features"].dtype == np.dtype(np.float32)
    verified = verify_russia_shard_catalog(source, first_path)
    assert verified.catalog_hash == first.catalog_hash


def test_each_canonical_shard_satisfies_the_v2_materializer(tmp_path: Path) -> None:
    source = _source_bundle()
    catalog_path = generate_russia_shards(source, tmp_path / "shards")
    catalog = RussiaShardCatalog.model_validate_json(catalog_path.read_bytes())
    runtime = tmp_path / "runtime"
    for descriptor in catalog.shards:
        artifact_id = f"governance-artifact-{descriptor.dataset_content_hash[:32]}"
        incoming = runtime / "incoming" / artifact_id
        incoming.mkdir(parents=True)
        shutil.copyfile(catalog_path.parent / descriptor.file_name, incoming / "bundle.zip")
        artifact = materialize_bundle(
            runtime,
            artifact_id,
            expected_dataset_content_hash=descriptor.dataset_content_hash,
            expected_graph_version_hash=descriptor.graph_version_hash,
            clean_self_loops=False,
        )
        loaded = load_materialized_artifact(artifact.root)
        assert loaded.text_features.shape == (descriptor.node_count, 768)
        assert loaded.artifact.document["fusedUndirectedEdgeCount"] == (
            descriptor.fused_undirected_edge_count
        )
        for modality in ("cort", "courl", "hashseq", "fastrt", "tweetsim"):
            assert f"relation_{modality}_indptr" in loaded.arrays
