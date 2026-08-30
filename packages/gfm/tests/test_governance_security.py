from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.governance import materialize as materialize_module
from socialgraph_gfm.governance.bundle import create_tiny_contract_bundle
from socialgraph_gfm.governance.contracts import INPUT_SCHEMA_VERSION
from socialgraph_gfm.governance.materialize import (
    BundleValidationError,
    load_materialized_artifact,
    materialize_bundle,
)

_ZERO_ARTIFACT_ID = f"governance-artifact-{'0' * 32}"
_TINY_DATASET_HASH = "0cbac59a3d09773fbfb72df6b6f8732b1c888b958d38f4f8cbca4dff64687337"
_TINY_GRAPH_HASH = "06c4f3ce8b09fef5e16ff8d7827bbe5d202235027d14394451de5b74eb768298"


def _fixture_entries(tmp_path: Path) -> dict[str, bytes]:
    source = create_tiny_contract_bundle(tmp_path / "source.zip")
    with zipfile.ZipFile(source) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _reseal_manifest(entries: dict[str, bytes], **overrides: Any) -> None:
    manifest = json.loads(entries["manifest.json"])
    for name in ("nodes.csv", "relations.csv", "features.npz"):
        value = entries[name]
        manifest["files"][name] = {
            "sha256": hashlib.sha256(value).hexdigest(),
            "bytes": len(value),
        }
    manifest.update(overrides)
    entries["manifest.json"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_archive(
    path: Path, members: Sequence[tuple[str | zipfile.ZipInfo, bytes]]
) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members:
            archive.writestr(name, value)
    return path


def _archive_from_entries(path: Path, entries: dict[str, bytes]) -> Path:
    return _write_archive(path, list(entries.items()))


def _install(root: Path, bundle: Path, artifact_id: str = _ZERO_ARTIFACT_ID) -> None:
    destination = root / "incoming" / artifact_id
    destination.mkdir(parents=True)
    shutil.copyfile(bundle, destination / "bundle.zip")


def _rejects(root: Path, *, clean: bool = False) -> None:
    with pytest.raises(BundleValidationError):
        materialize_bundle(
            root,
            _ZERO_ARTIFACT_ID,
            expected_dataset_content_hash="0" * 64,
            expected_graph_version_hash="0" * 64,
            clean_self_loops=clean,
        )


@pytest.mark.parametrize("unsafe_name", ["../manifest.json", "/manifest.json", "C:\\manifest.json"])
def test_outer_zip_rejects_traversal_and_absolute_members(
    tmp_path: Path, unsafe_name: str
) -> None:
    entries = _fixture_entries(tmp_path)
    members: list[tuple[str | zipfile.ZipInfo, bytes]] = [
        (unsafe_name if name == "manifest.json" else name, value)
        for name, value in entries.items()
    ]
    root = tmp_path / "runtime"
    _install(root, _write_archive(tmp_path / "unsafe.zip", members))
    _rejects(root)


def test_outer_zip_rejects_duplicate_symlink_and_high_ratio_members(tmp_path: Path) -> None:
    entries = _fixture_entries(tmp_path)

    duplicate = [
        ("manifest.json", entries["manifest.json"]),
        ("manifest.json", entries["manifest.json"]),
        ("nodes.csv", entries["nodes.csv"]),
        ("relations.csv", entries["relations.csv"]),
    ]
    duplicate_root = tmp_path / "duplicate-runtime"
    with pytest.warns(UserWarning, match="Duplicate name"):
        duplicate_bundle = _write_archive(tmp_path / "duplicate.zip", duplicate)
    _install(duplicate_root, duplicate_bundle)
    _rejects(duplicate_root)

    link_info = zipfile.ZipInfo("manifest.json")
    link_info.create_system = 3
    link_info.external_attr = 0o120777 << 16
    link_info.compress_type = zipfile.ZIP_DEFLATED
    symlink_members: list[tuple[str | zipfile.ZipInfo, bytes]] = [
        (link_info if name == "manifest.json" else name, value)
        for name, value in entries.items()
    ]
    symlink_root = tmp_path / "symlink-runtime"
    _install(symlink_root, _write_archive(tmp_path / "symlink.zip", symlink_members))
    _rejects(symlink_root)

    entries["manifest.json"] = b"A" * 100_000
    ratio_root = tmp_path / "ratio-runtime"
    _install(ratio_root, _archive_from_entries(tmp_path / "ratio.zip", entries))
    with pytest.raises(BundleValidationError, match="compression ratio"):
        materialize_bundle(
            ratio_root,
            _ZERO_ARTIFACT_ID,
            expected_dataset_content_hash="0" * 64,
            expected_graph_version_hash="0" * 64,
            clean_self_loops=False,
        )


def _replace_features(
    entries: dict[str, bytes], *, node_ids: np.ndarray, features: np.ndarray
) -> None:
    stream = io.BytesIO()
    np.savez_compressed(stream, node_ids=node_ids, text_features=features)
    entries["features.npz"] = stream.getvalue()
    _reseal_manifest(entries)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_features_reject_nonfinite_values(tmp_path: Path, nonfinite: float) -> None:
    entries = _fixture_entries(tmp_path)
    features = np.zeros((6, 768), dtype=np.float32)
    features[0, 0] = nonfinite
    _replace_features(
        entries,
        node_ids=np.asarray([f"synthetic:{index}" for index in range(6)]),
        features=features,
    )
    root = tmp_path / "runtime"
    _install(root, _archive_from_entries(tmp_path / "nonfinite.zip", entries))
    _rejects(root)


@pytest.mark.parametrize(
    ("node_ids", "features"),
    [
        (
            np.asarray([f"synthetic:{index}" for index in reversed(range(6))]),
            np.zeros((6, 768), dtype=np.float32),
        ),
        (
            np.asarray([f"synthetic:{index}" for index in range(6)]),
            np.zeros((6, 767), dtype=np.float32),
        ),
        (
            np.asarray([f"synthetic:{index}" for index in range(6)], dtype=object),
            np.zeros((6, 768), dtype=np.float32),
        ),
    ],
)
def test_features_reject_misalignment_wrong_dimensions_and_pickle_arrays(
    tmp_path: Path, node_ids: np.ndarray, features: np.ndarray
) -> None:
    entries = _fixture_entries(tmp_path)
    _replace_features(entries, node_ids=node_ids, features=features)
    root = tmp_path / "runtime"
    _install(root, _archive_from_entries(tmp_path / "invalid-features.zip", entries))
    _rejects(root)


@pytest.mark.parametrize(
    "relations",
    [
        "source,target,modality,weight\nsynthetic:0,synthetic:1,unknown,1\n",
        (
            "source,target,modality,weight\n"
            "synthetic:0,synthetic:1,coRT,1\n"
            "synthetic:1,synthetic:0,coRT,2\n"
        ),
        "source,target,modality,weight\nmissing,synthetic:1,coRT,1\n",
        "source,target,modality,weight\nsynthetic:0,synthetic:1,coRT,nan\n",
    ],
)
def test_relations_reject_unknown_duplicate_dangling_and_nonfinite_rows(
    tmp_path: Path, relations: str
) -> None:
    entries = _fixture_entries(tmp_path)
    entries["relations.csv"] = relations.encode("utf-8")
    _reseal_manifest(
        entries,
        relationRowCount=len(relations.strip().splitlines()) - 1,
        modalities=["coRT"],
    )
    root = tmp_path / "runtime"
    _install(root, _archive_from_entries(tmp_path / "invalid-relations.zip", entries))
    _rejects(root)


def test_self_loop_cleaning_is_explicit_deterministic_and_topology_preserving(
    tmp_path: Path,
) -> None:
    entries = _fixture_entries(tmp_path)
    relations = entries["relations.csv"] + b"synthetic:0,synthetic:0,coRT,1\n"
    entries["relations.csv"] = relations
    _reseal_manifest(entries, relationRowCount=7)
    raw_manifest = entries["manifest.json"]
    file_digests = {
        name: hashlib.sha256(entries[name]).hexdigest()
        for name in ("nodes.csv", "relations.csv", "features.npz")
    }
    dataset_hash = canonical_sha256(
        {
            "schemaVersion": INPUT_SCHEMA_VERSION,
            "manifestHash": hashlib.sha256(raw_manifest).hexdigest(),
            "fileDigests": file_digests,
            "cleanSelfLoops": True,
            "selfLoopsRemoved": 1,
        }
    )
    artifact_id = f"governance-artifact-{dataset_hash[:32]}"
    bundle = _archive_from_entries(tmp_path / "self-loop.zip", entries)

    blocked_root = tmp_path / "blocked-runtime"
    _install(blocked_root, bundle, artifact_id)
    with pytest.raises(BundleValidationError, match="cleanSelfLoops=true"):
        materialize_bundle(
            blocked_root,
            artifact_id,
            expected_dataset_content_hash=dataset_hash,
            expected_graph_version_hash=_TINY_GRAPH_HASH,
            clean_self_loops=False,
        )

    artifacts = []
    for suffix in ("first", "second"):
        root = tmp_path / f"{suffix}-runtime"
        _install(root, bundle, artifact_id)
        artifacts.append(
            materialize_bundle(
                root,
                artifact_id,
                expected_dataset_content_hash=dataset_hash,
                expected_graph_version_hash=_TINY_GRAPH_HASH,
                clean_self_loops=True,
            )
        )
    assert artifacts[0].document["selfLoopsRemoved"] == 1
    assert artifacts[0].document["relationRowCount"] == 6
    assert artifacts[0].dataset_content_hash == artifacts[1].dataset_content_hash
    first = load_materialized_artifact(artifacts[0].root)
    second = load_materialized_artifact(artifacts[1].root)
    assert np.array_equal(first.edge_index, second.edge_index)
    assert not bool(np.any(first.edge_index[0] == first.edge_index[1]))


def test_manifest_and_frozen_array_tampering_fail_closed(tmp_path: Path) -> None:
    entries = _fixture_entries(tmp_path)
    manifest = json.loads(entries["manifest.json"])
    manifest["files"]["nodes.csv"]["sha256"] = "0" * 64
    entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
    tamper_root = tmp_path / "manifest-runtime"
    _install(tamper_root, _archive_from_entries(tmp_path / "manifest-tamper.zip", entries))
    _rejects(tamper_root)

    valid_bundle = create_tiny_contract_bundle(tmp_path / "valid.zip")
    valid_root = tmp_path / "valid-runtime"
    artifact_id = f"governance-artifact-{_TINY_DATASET_HASH[:32]}"
    _install(valid_root, valid_bundle, artifact_id)
    artifact = materialize_bundle(
        valid_root,
        artifact_id,
        expected_dataset_content_hash=_TINY_DATASET_HASH,
        expected_graph_version_hash=_TINY_GRAPH_HASH,
        clean_self_loops=False,
    )
    feature_path = artifact.root / "text_features.npy"
    with feature_path.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 0x01]))
    with pytest.raises(BundleValidationError, match="hash or length mismatch"):
        load_materialized_artifact(artifact.root)


def test_declared_and_expanded_limits_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = _fixture_entries(tmp_path)
    manifest = json.loads(entries["manifest.json"])
    manifest["nodeCount"] = 10_001
    entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
    declared_root = tmp_path / "declared-runtime"
    _install(declared_root, _archive_from_entries(tmp_path / "declared.zip", entries))
    _rejects(declared_root)

    limit_bundle = create_tiny_contract_bundle(tmp_path / "limit.zip")
    limit_root = tmp_path / "limit-runtime"
    _install(limit_root, limit_bundle)
    monkeypatch.setattr(materialize_module, "_MAX_EXPANDED_BYTES", 100)
    with pytest.raises(BundleValidationError, match="configured limit"):
        materialize_bundle(
            limit_root,
            _ZERO_ARTIFACT_ID,
            expected_dataset_content_hash="0" * 64,
            expected_graph_version_hash="0" * 64,
            clean_self_loops=False,
        )
