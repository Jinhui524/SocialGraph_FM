from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.gfm.corpus import common
from socialgraph_gfm.gfm.corpus.common import (
    NumericShardWriter,
    atomic_write_bytes,
    atomic_write_npz,
    atomic_write_json,
    atomic_write_jsonl,
    build_manifest,
    exclusive_file_lock,
    load_npz_safe,
    resolve_within,
    verify_manifest,
)


def _windows_replace_error(code: int) -> PermissionError:
    error = PermissionError(13, "synthetic Windows sharing violation")
    error.winerror = code
    return error


@pytest.mark.parametrize("writer", ["bytes", "jsonl"])
@pytest.mark.parametrize("winerror", sorted(common.WINDOWS_SHARING_RETRY_ERRORS))
def test_atomic_writes_retry_short_windows_replace_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str, winerror: int
) -> None:
    path = tmp_path / f"artifact.{writer}"
    path.write_text("old", encoding="utf-8")
    real_replace = common.os.replace
    calls = 0
    delays: list[float] = []

    def flaky_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise _windows_replace_error(winerror)
        real_replace(source, destination)

    monkeypatch.setattr(common.os, "replace", flaky_replace)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    if writer == "bytes":
        atomic_write_bytes(path, b"new")
        assert path.read_bytes() == b"new"
    else:
        assert atomic_write_jsonl(path, [{"row": 1}, {"row": 2}]) == 2
        assert path.read_text(encoding="utf-8").splitlines() == [
            '{"row":1}',
            '{"row":2}',
        ]
    assert calls == 3
    assert delays == list(common.WINDOWS_SHARING_RETRY_DELAYS[:2])
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


@pytest.mark.parametrize("winerror", sorted(common.WINDOWS_SHARING_RETRY_ERRORS))
def test_atomic_write_fails_after_bounded_windows_sharing_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winerror: int
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"committed")
    calls = 0
    delays: list[float] = []

    def locked_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        raise _windows_replace_error(winerror)

    monkeypatch.setattr(common.os, "replace", locked_replace)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    with pytest.raises(PermissionError):
        atomic_write_bytes(path, b"candidate")
    assert calls == len(common.WINDOWS_SHARING_RETRY_DELAYS) + 1
    assert delays == list(common.WINDOWS_SHARING_RETRY_DELAYS)
    assert path.read_bytes() == b"committed"
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_atomic_write_gives_access_denied_only_two_writable_target_micro_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"committed")
    calls = 0
    delays: list[float] = []

    def locked_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        raise _windows_replace_error(5)

    monkeypatch.setattr(common.os, "replace", locked_replace)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    with pytest.raises(PermissionError):
        atomic_write_bytes(path, b"candidate")
    assert calls == len(common.WINDOWS_ACCESS_DENIED_RETRY_DELAYS) + 1
    assert delays == list(common.WINDOWS_ACCESS_DENIED_RETRY_DELAYS)
    assert path.read_bytes() == b"committed"
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_atomic_write_does_not_retry_access_denied_for_missing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "missing.bin"
    calls = 0
    delays: list[float] = []

    def denied_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        raise _windows_replace_error(5)

    monkeypatch.setattr(common.os, "replace", denied_replace)
    monkeypatch.setattr(common.time, "sleep", delays.append)
    with pytest.raises(PermissionError):
        atomic_write_bytes(path, b"candidate")
    assert calls == 1
    assert delays == []
    assert not path.exists()
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_exclusive_file_lock_rejects_a_concurrent_holder_and_is_reusable(
    tmp_path: Path,
) -> None:
    path = tmp_path / ".operation.lock"
    with exclusive_file_lock(path):
        with pytest.raises(ContractViolation, match="already running"):
            with exclusive_file_lock(path):
                raise AssertionError("a second holder must never enter")
    with exclusive_file_lock(path):
        pass
    assert path.read_bytes() == b"\0"


def test_numeric_npz_roundtrip_and_object_rejection(tmp_path: Path) -> None:
    path = tmp_path / "safe.npz"
    atomic_write_npz(path, {"x": np.asarray([[1.0, 2.0]], dtype=np.float32)})
    loaded = load_npz_safe(path, expected={"x": (np.dtype(np.float32).str, 2)})
    assert np.array_equal(loaded["x"], [[1.0, 2.0]])
    with pytest.raises(ContractViolation, match="fixed numeric"):
        atomic_write_npz(tmp_path / "bad.npz", {"x": np.asarray([{"x": 1}], dtype=object)})


def test_npz_path_traversal_and_symlink_escape_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    payload = io.BytesIO()
    np.save(payload, np.asarray([1], dtype=np.int64), allow_pickle=False)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../x.npy", payload.getvalue())
    with pytest.raises(ContractViolation):
        load_npz_safe(path, expected={"x": (np.dtype(np.int64).str, 1)})
    with pytest.raises(ContractViolation, match="traverses"):
        resolve_within(tmp_path, "../escape")


def test_shard_writer_is_immutable_and_bounded(tmp_path: Path) -> None:
    writer = NumericShardWriter(tmp_path, prefix="events", rows_per_shard=2)
    record = writer.write({"x": np.asarray([1, 2], dtype=np.int64)})
    assert record.rows == 2
    assert record.path == "events-00000.npz"
    with pytest.raises(ContractViolation, match="capacity"):
        writer.write({"x": np.asarray([1, 2, 3], dtype=np.int64)})


def test_manifest_rejects_undeclared_files(tmp_path: Path) -> None:
    writer = NumericShardWriter(tmp_path, prefix="events", rows_per_shard=1)
    record = writer.write({"x": np.asarray([1], dtype=np.int64)})
    manifest = build_manifest(
        schema_version="fixture/1.0",
        corpus_id="fixture",
        license_id="CC0-1.0",
        source={"fixture": True},
        shards=(record,),
        splits={"train": 1},
        privacy={"safe": True},
    )
    atomic_write_json(tmp_path / "manifest.json", manifest)
    verify_manifest(tmp_path, manifest)
    (tmp_path / "undeclared.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ContractViolation, match="whitelist mismatch"):
        verify_manifest(tmp_path, manifest)
