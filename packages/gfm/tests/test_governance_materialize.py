from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from socialgraph_gfm.governance.bundle import create_tiny_contract_bundle
from socialgraph_gfm.governance.materialize import (
    BundleValidationError,
    load_materialized_artifact,
    materialize_bundle,
)

TINY_DATASET_HASH = "0cbac59a3d09773fbfb72df6b6f8732b1c888b958d38f4f8cbca4dff64687337"
TINY_GRAPH_HASH = "06c4f3ce8b09fef5e16ff8d7827bbe5d202235027d14394451de5b74eb768298"
TINY_ARTIFACT_ID = f"governance-artifact-{TINY_DATASET_HASH[:32]}"


def _install_bundle(root: Path, bundle: Path, artifact_id: str = TINY_ARTIFACT_ID) -> None:
    destination = root / "incoming" / artifact_id
    destination.mkdir(parents=True)
    shutil.copyfile(bundle, destination / "bundle.zip")


def test_tiny_bundle_is_byte_deterministic_and_materializes_with_known_hashes(
    tmp_path: Path,
) -> None:
    first = create_tiny_contract_bundle(tmp_path / "first.zip")
    second = create_tiny_contract_bundle(tmp_path / "second.zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as outer:
        assert all(info.create_system == 0 for info in outer.infolist())
        features_path = tmp_path / "features.npz"
        features_path.write_bytes(outer.read("features.npz"))
    with zipfile.ZipFile(features_path) as features:
        assert all(info.create_system == 0 for info in features.infolist())
    root = tmp_path / "runtime"
    _install_bundle(root, first)

    artifact = materialize_bundle(
        root,
        TINY_ARTIFACT_ID,
        expected_dataset_content_hash=TINY_DATASET_HASH,
        expected_graph_version_hash=TINY_GRAPH_HASH,
        clean_self_loops=False,
    )
    loaded = load_materialized_artifact(artifact.root)

    assert loaded.artifact.dataset_content_hash == TINY_DATASET_HASH
    assert loaded.artifact.graph_version_hash == TINY_GRAPH_HASH
    assert loaded.text_features.shape == (6, 768)
    assert loaded.edge_index.shape == (2, 12)
    assert loaded.graph_stats.shape == (13,)
    assert not np.any(loaded.edge_index[0] == loaded.edge_index[1])
    assert not bool(loaded.structure_missing.any())


def test_bundle_tampering_and_unconfirmed_self_loop_are_rejected(tmp_path: Path) -> None:
    source = create_tiny_contract_bundle(tmp_path / "source.zip")
    root = tmp_path / "tampered-runtime"
    _install_bundle(root, source)
    bundle = root / "incoming" / TINY_ARTIFACT_ID / "bundle.zip"
    with zipfile.ZipFile(bundle) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    manifest["files"]["nodes.csv"]["sha256"] = "0" * 64
    entries["manifest.json"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    bundle.unlink()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    with pytest.raises(BundleValidationError, match="nodes.csv does not match"):
        materialize_bundle(
            root,
            TINY_ARTIFACT_ID,
            expected_dataset_content_hash=TINY_DATASET_HASH,
            expected_graph_version_hash=TINY_GRAPH_HASH,
            clean_self_loops=False,
        )

    loop_root = tmp_path / "loop-runtime"
    loop_bundle = create_tiny_contract_bundle(tmp_path / "loop-source.zip")
    _install_bundle(loop_root, loop_bundle)
    installed = loop_root / "incoming" / TINY_ARTIFACT_ID / "bundle.zip"
    with zipfile.ZipFile(installed) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    relations = entries["relations.csv"].decode().replace(
        "synthetic:0,synthetic:1,coRT", "synthetic:0,synthetic:0,coRT", 1
    ).encode()
    import hashlib

    manifest = json.loads(entries["manifest.json"])
    manifest["files"]["relations.csv"] = {
        "sha256": hashlib.sha256(relations).hexdigest(),
        "bytes": len(relations),
    }
    entries["relations.csv"] = relations
    entries["manifest.json"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    installed.unlink()
    with zipfile.ZipFile(installed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    with pytest.raises(BundleValidationError, match="cleanSelfLoops=true"):
        materialize_bundle(
            loop_root,
            TINY_ARTIFACT_ID,
            expected_dataset_content_hash="0" * 64,
            expected_graph_version_hash="0" * 64,
            clean_self_loops=False,
        )
