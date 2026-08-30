from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from socialgraph_gfm.corpus import ogbl_collab
from socialgraph_gfm.errors import ContractViolation


def _npy(value: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, value, allow_pickle=True)
    return output.getvalue()


def test_safe_npz_round_trip_and_object_array_rejection(tmp_path: Path) -> None:
    safe = tmp_path / "safe.npz"
    np.savez_compressed(safe, x=np.asarray([[1.0, 2.0]], dtype=np.float32))
    arrays = ogbl_collab._load_npz_safely(safe, expected_keys=frozenset({"x"}))
    assert arrays["x"].dtype == np.float32
    assert arrays["x"].shape == (1, 2)

    unsafe = tmp_path / "unsafe.npz"
    np.savez_compressed(unsafe, x=np.asarray([{"payload": "pickle"}], dtype=object))
    with pytest.raises(ContractViolation, match="pickle/object"):
        ogbl_collab._load_npz_safely(unsafe, expected_keys=frozenset({"x"}))


@pytest.mark.parametrize("entry", ["../x.npy", "/x.npy", "folder\\x.npy", "C:x.npy"])
def test_npz_path_traversal_and_noncanonical_members_are_rejected(
    tmp_path: Path,
    entry: str,
) -> None:
    package = tmp_path / "traversal.npz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(entry, _npy(np.asarray([1], dtype=np.int64)))
    with pytest.raises(
        ContractViolation,
        match="unsafe archive member|path traversal|whitelist mismatch",
    ):
        ogbl_collab._load_npz_safely(package, expected_keys=frozenset({"x"}))


def test_npz_duplicate_and_extra_members_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.npz"
    with zipfile.ZipFile(duplicate, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("x.npy", _npy(np.asarray([1], dtype=np.int64)))
        archive.writestr("x.npy", _npy(np.asarray([2], dtype=np.int64)))
    with pytest.raises(ContractViolation, match="duplicate member|too many entries"):
        ogbl_collab._load_npz_safely(duplicate, expected_keys=frozenset({"x"}))

    extra = tmp_path / "extra.npz"
    np.savez_compressed(extra, x=np.asarray([1]), surprise=np.asarray([2]))
    with pytest.raises(ContractViolation, match="whitelist mismatch|too many entries"):
        ogbl_collab._load_npz_safely(extra, expected_keys=frozenset({"x"}))


def test_manifest_duplicate_keys_and_nonfinite_constants_are_rejected() -> None:
    with pytest.raises(ContractViolation, match="duplicate JSON key"):
        ogbl_collab._json_no_duplicates(b'{"a":1,"a":2}', label="test")
    with pytest.raises(ContractViolation, match="forbidden JSON constant"):
        ogbl_collab._json_no_duplicates(b'{"a":NaN}', label="test")


def test_edge_shape_dtype_bounds_and_self_loop_fail_closed() -> None:
    with pytest.raises(ContractViolation, match="out-of-bounds"):
        ogbl_collab._validate_edge_array(
            np.asarray([[0], [3]], dtype=np.int64),
            size=1,
            node_count=3,
            name="edge",
            canonical=True,
        )
    with pytest.raises(ContractViolation, match="self-loop"):
        ogbl_collab._validate_edge_array(
            np.asarray([[1], [1]], dtype=np.int64),
            size=1,
            node_count=3,
            name="edge",
            canonical=True,
        )
    with pytest.raises(ContractViolation, match="dtype"):
        ogbl_collab._validate_edge_array(
            np.asarray([[0], [1]], dtype=np.int32),
            size=1,
            node_count=3,
            name="edge",
            canonical=True,
        )


def test_trusted_ogb_pickle_scope_is_local_and_restores_torch_load() -> None:
    torch = pytest.importorskip("torch")
    original = torch.load
    with ogbl_collab._trusted_ogb_pickle_scope():
        assert torch.load is not original
    assert torch.load is original


def test_package_manifest_requires_license_and_temporal_protocol() -> None:
    manifest = ogbl_collab._package_manifest("a" * 64)
    assert ogbl_collab._validate_package_manifest(manifest) == "a" * 64
    tampered = json.loads(json.dumps(manifest))
    tampered["datasets"][0]["licensePolicy"]["identifier"] = "unknown"
    with pytest.raises(ContractViolation, match="license"):
        ogbl_collab._validate_package_manifest(tampered)
