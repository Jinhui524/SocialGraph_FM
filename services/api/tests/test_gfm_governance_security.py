from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.gfm_client import GfmProxyError
from app.gfm_governance_artifacts import GovernanceArtifactInbox, inspect_governance_bundle


def _entries(
    *,
    node_ids: np.ndarray | None = None,
    features: np.ndarray | None = None,
    relations: bytes = b"source,target,modality,weight\na,b,coRT,1\n",
) -> dict[str, bytes]:
    nodes = b"node_id,display_name\na,Account A\nb,Account B\n"
    feature_stream = io.BytesIO()
    np.savez_compressed(
        feature_stream,
        node_ids=np.asarray(("a", "b")) if node_ids is None else node_ids,
        text_features=(
            np.arange(2 * 768, dtype=np.float32).reshape(2, 768)
            if features is None
            else features
        ),
    )
    entries = {
        "nodes.csv": nodes,
        "relations.csv": relations,
        "features.npz": feature_stream.getvalue(),
    }
    _reseal(entries)
    return entries


def _reseal(entries: dict[str, bytes], **overrides: Any) -> None:
    previous = json.loads(entries["manifest.json"]) if "manifest.json" in entries else {}
    relation_count = max(entries["relations.csv"].count(b"\n") - 1, 0)
    manifest: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.governance-input/2.0",
        "datasetId": "test:security",
        "displayName": "Security contract graph",
        "nodeCount": 2,
        "relationRowCount": relation_count,
        "featureDimension": 768,
        "modalities": ["coRT"],
        "files": {
            name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
            for name, value in entries.items()
            if name != "manifest.json"
        },
    }
    manifest.update(previous)
    manifest.update(overrides)
    manifest["files"] = {
        name: {"sha256": hashlib.sha256(entries[name]).hexdigest(), "bytes": len(entries[name])}
        for name in ("nodes.csv", "relations.csv", "features.npz")
    }
    entries["manifest.json"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _archive(members: Sequence[tuple[str | zipfile.ZipInfo, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members:
            archive.writestr(name, value)
    return output.getvalue()


def _payload(entries: dict[str, bytes]) -> bytes:
    return _archive(list(entries.items()))


def _inspect(payload: bytes, *, clean: bool = False, maximum: int = 1024**3):
    return inspect_governance_bundle(
        payload, clean_self_loops=clean, max_expanded_bytes=maximum
    )


@pytest.mark.parametrize("unsafe_name", ["../manifest.json", "/manifest.json", "C:\\manifest.json"])
def test_outer_zip_rejects_traversal_and_absolute_members(unsafe_name: str) -> None:
    entries = _entries()
    payload = _archive(
        [
            (unsafe_name if name == "manifest.json" else name, value)
            for name, value in entries.items()
        ]
    )
    with pytest.raises(GfmProxyError) as raised:
        _inspect(payload)
    assert raised.value.code in {
        "GOVERNANCE_ZIP_PATH_UNSAFE",
        "GOVERNANCE_ZIP_MEMBERS_INVALID",
    }


def test_outer_zip_rejects_duplicate_symlink_and_compression_bomb() -> None:
    entries = _entries()
    with pytest.warns(UserWarning, match="Duplicate name"):
        duplicate = _archive(
            [
                ("manifest.json", entries["manifest.json"]),
                ("manifest.json", entries["manifest.json"]),
                ("nodes.csv", entries["nodes.csv"]),
                ("relations.csv", entries["relations.csv"]),
            ]
        )
    with pytest.raises(GfmProxyError) as duplicate_error:
        _inspect(duplicate)
    assert duplicate_error.value.code == "GOVERNANCE_ZIP_MEMBERS_INVALID"

    link_info = zipfile.ZipInfo("manifest.json")
    link_info.create_system = 3
    link_info.external_attr = 0o120777 << 16
    link_info.compress_type = zipfile.ZIP_DEFLATED
    symlink = _archive(
        [
            (link_info if name == "manifest.json" else name, value)
            for name, value in entries.items()
        ]
    )
    with pytest.raises(GfmProxyError) as symlink_error:
        _inspect(symlink)
    assert symlink_error.value.code == "GOVERNANCE_ZIP_LINK_REJECTED"

    entries["manifest.json"] = b"A" * 100_000
    bomb = _payload(entries)
    with pytest.raises(GfmProxyError) as ratio_error:
        _inspect(bomb)
    assert ratio_error.value.code == "GOVERNANCE_ZIP_RATIO_INVALID"

    with pytest.raises(GfmProxyError) as expanded_error:
        _inspect(_payload(_entries()), maximum=100)
    assert expanded_error.value.code == "GOVERNANCE_ZIP_EXPANDED_TOO_LARGE"


@pytest.mark.parametrize(
    ("node_ids", "features", "expected_code"),
    [
        (
            np.asarray(("b", "a")),
            np.zeros((2, 768), dtype=np.float32),
            "GOVERNANCE_FEATURE_NODE_ALIGNMENT_MISMATCH",
        ),
        (
            np.asarray(("a", "b")),
            np.zeros((2, 767), dtype=np.float32),
            "GOVERNANCE_FEATURE_SHAPE_INVALID",
        ),
        (
            np.asarray(("a", "b"), dtype=object),
            np.zeros((2, 768), dtype=np.float32),
            "GOVERNANCE_FEATURES_NPZ_INVALID",
        ),
    ],
)
def test_features_reject_misalignment_wrong_dimensions_and_pickle_arrays(
    node_ids: np.ndarray, features: np.ndarray, expected_code: str
) -> None:
    with pytest.raises(GfmProxyError) as raised:
        _inspect(_payload(_entries(node_ids=node_ids, features=features)))
    assert raised.value.code == expected_code


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_features_reject_nan_and_infinity(nonfinite: float) -> None:
    features = np.zeros((2, 768), dtype=np.float32)
    features[0, 0] = nonfinite
    with pytest.raises(GfmProxyError) as raised:
        _inspect(_payload(_entries(features=features)))
    assert raised.value.code == "GOVERNANCE_FEATURE_NONFINITE"


@pytest.mark.parametrize(
    ("relations", "expected_code"),
    [
        (
            b"source,target,modality,weight\na,b,unknown,1\n",
            "GOVERNANCE_RELATION_MODALITY_INVALID",
        ),
        (
            b"source,target,modality,weight\na,b,coRT,1\nb,a,coRT,2\n",
            "GOVERNANCE_RELATION_DUPLICATE",
        ),
        (
            b"source,target,modality,weight\nmissing,b,coRT,1\n",
            "GOVERNANCE_RELATION_NODE_MISSING",
        ),
        (
            b"source,target,modality,weight\na,b,coRT,nan\n",
            "GOVERNANCE_RELATION_WEIGHT_INVALID",
        ),
    ],
)
def test_relations_reject_unknown_duplicate_dangling_and_nonfinite_rows(
    relations: bytes, expected_code: str
) -> None:
    with pytest.raises(GfmProxyError) as raised:
        _inspect(_payload(_entries(relations=relations)))
    assert raised.value.code == expected_code


def test_self_loop_cleaning_is_confirmed_deterministic_and_changes_only_content_identity() -> None:
    base = _inspect(_payload(_entries()))[1]
    relations = b"source,target,modality,weight\na,a,coRT,0\na,b,coRT,1\n"
    payload = _payload(_entries(relations=relations))
    with pytest.raises(GfmProxyError) as blocked:
        _inspect(payload, clean=False)
    assert blocked.value.code == "GOVERNANCE_SELF_LOOP_CONFIRMATION_REQUIRED"

    first = _inspect(payload, clean=True)[1]
    second = _inspect(payload, clean=True)[1]
    assert first == second
    assert first["selfLoopsRemoved"] == 1
    assert first["relationRowCount"] == 1
    assert first["graphVersionHash"] == base["graphVersionHash"]
    assert first["datasetContentHash"] != base["datasetContentHash"]


def test_manifest_limits_and_hash_tampering_are_rejected() -> None:
    entries = _entries()
    manifest = json.loads(entries["manifest.json"])
    manifest["nodeCount"] = 10_001
    entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
    with pytest.raises(GfmProxyError) as limit_error:
        _inspect(_payload(entries))
    assert limit_error.value.code == "GOVERNANCE_MANIFEST_INVALID"

    entries = _entries()
    manifest = json.loads(entries["manifest.json"])
    manifest["files"]["nodes.csv"]["sha256"] = "0" * 64
    entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
    with pytest.raises(GfmProxyError) as hash_error:
        _inspect(_payload(entries))
    assert hash_error.value.code == "GOVERNANCE_MANIFEST_FILE_HASH_MISMATCH"


def test_inbox_receipt_and_bundle_tampering_fail_closed(tmp_path: Path) -> None:
    inbox = GovernanceArtifactInbox(tmp_path / "governance")
    payload = _payload(_entries())
    receipt = inbox.commit(payload, clean_self_loops=False, max_expanded_bytes=1024**3)
    directory = tmp_path / "governance" / "incoming" / receipt.artifact_id

    receipt_path = directory / "receipt.json"
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    document["nodeCount"] = 3
    receipt_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(GfmProxyError) as receipt_error:
        inbox.get(receipt.artifact_id)
    assert receipt_error.value.code == "GOVERNANCE_ARTIFACT_RECEIPT_INVALID"

    receipt_path.write_text(receipt.model_dump_json(by_alias=True), encoding="utf-8")
    bundle_path = directory / "bundle.zip"
    bundle_path.write_bytes(payload + b"tamper")
    with pytest.raises(GfmProxyError) as bundle_error:
        inbox.commit(payload, clean_self_loops=False, max_expanded_bytes=1024**3)
    assert bundle_error.value.code == "GOVERNANCE_ARTIFACT_ID_CONFLICT"
