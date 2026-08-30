from __future__ import annotations

import ctypes
import multiprocessing
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import socialgraph_gfm.governance.reviewed_cases as reviewed_cases_module
from socialgraph_gfm.governance.reviewed_cases import CaseVectors, ReviewedCaseIndex


def _hold_reviewed_case_root_lock(root: str, ready: str, release: str) -> None:
    index = ReviewedCaseIndex(Path(root))
    with index._locked():
        Path(ready).write_text("ready", encoding="utf-8")
        deadline = time.monotonic() + 15
        while not Path(release).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("parent did not release reviewed-case root lock")
            time.sleep(0.01)


def _vectors(seed: int) -> CaseVectors:
    generator = np.random.default_rng(seed)
    return CaseVectors(
        embedding=generator.normal(size=256).astype(np.float32),
        structure=generator.random(6).astype(np.float32),
        modality=generator.random(5).astype(np.float32),
    )


def _metadata(case_id: str, *, graph_hash: str, model_hash: str = "2" * 64) -> dict[str, object]:
    return {
        "caseId": case_id,
        "caseHash": "a" * 64,
        "runId": "governance-" + "1" * 32,
        "resultHash": "b" * 64,
        "kindKey": "node+relation",
        "kindEntries": [
            {"kind": "node", "targetIds": ["synthetic:0"]},
            {"kind": "relation", "targetIds": ["relation-0-1"]},
        ],
        "concludedAt": "2026-08-18T00:00:00.000000Z",
        "reviewHash": "c" * 64,
        "reviewStatus": "concluded",
        "artifactId": "governance-artifact-" + "d" * 32,
        "datasetContentHash": "e" * 64,
        "graphVersionHash": graph_hash,
        "modelVersionId": "socialgraph-fm-global/test",
        "modelStateHash": model_hash,
    }


@pytest.mark.skipif(os.name != "nt", reason="Windows directory durability contract")
def test_windows_directory_fsync_opens_shared_write_handle_and_flushes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []

    def create_file(*arguments: object) -> int:
        events.append(("create", *arguments))
        return 37

    def flush_file_buffers(handle: int) -> int:
        events.append(("flush", handle))
        return 1

    def close_handle(handle: int) -> int:
        events.append(("close", handle))
        return 1

    monkeypatch.setattr(reviewed_cases_module, "_CreateFileW", create_file, raising=False)
    monkeypatch.setattr(
        reviewed_cases_module,
        "_FlushFileBuffers",
        flush_file_buffers,
        raising=False,
    )
    monkeypatch.setattr(reviewed_cases_module, "_CloseHandle", close_handle, raising=False)

    reviewed_cases_module._fsync_directory(tmp_path)

    assert events == [
        (
            "create",
            str(tmp_path),
            0x40000000,
            0x0001 | 0x0002 | 0x0004,
            None,
            3,
            0x02000000,
            None,
        ),
        ("flush", 37),
        ("close", 37),
    ]


@pytest.mark.skipif(os.name != "nt", reason="Windows directory durability contract")
def test_windows_directory_fsync_fails_closed_when_flush_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[int] = []
    monkeypatch.setattr(reviewed_cases_module, "_CreateFileW", lambda *_arguments: 41, raising=False)
    monkeypatch.setattr(reviewed_cases_module, "_FlushFileBuffers", lambda _handle: 0, raising=False)
    monkeypatch.setattr(
        reviewed_cases_module,
        "_CloseHandle",
        lambda handle: closed.append(handle) or 1,
        raising=False,
    )
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 1117)

    with pytest.raises(OSError, match="failed to flush directory metadata") as error:
        reviewed_cases_module._fsync_directory(tmp_path)

    assert error.value.errno == 1117
    assert closed == [41]


@pytest.mark.skipif(os.name != "nt", reason="Windows directory durability contract")
def test_windows_directory_fsync_fails_closed_when_directory_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flush_called = False

    def unexpected_flush(_handle: int) -> int:
        nonlocal flush_called
        flush_called = True
        return 1

    monkeypatch.setattr(
        reviewed_cases_module,
        "_CreateFileW",
        lambda *_arguments: ctypes.c_void_p(-1).value,
        raising=False,
    )
    monkeypatch.setattr(
        reviewed_cases_module,
        "_FlushFileBuffers",
        unexpected_flush,
        raising=False,
    )
    monkeypatch.setattr(reviewed_cases_module, "_CloseHandle", lambda _handle: 1, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5)

    with pytest.raises(OSError, match="failed to open directory durability handle") as error:
        reviewed_cases_module._fsync_directory(tmp_path)

    assert error.value.errno == 5
    assert flush_called is False


@pytest.mark.skipif(os.name != "nt", reason="Windows directory durability contract")
def test_windows_directory_fsync_fails_closed_when_handle_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reviewed_cases_module, "_CreateFileW", lambda *_arguments: 43)
    monkeypatch.setattr(reviewed_cases_module, "_FlushFileBuffers", lambda _handle: 1)
    monkeypatch.setattr(reviewed_cases_module, "_CloseHandle", lambda _handle: 0)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 6)

    with pytest.raises(OSError, match="failed to close directory durability handle") as error:
        reviewed_cases_module._fsync_directory(tmp_path)

    assert error.value.errno == 6


def test_atomic_bytes_fails_closed_after_replacement_when_directory_barrier_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "manifest.json"
    target.write_bytes(b"previous")

    def fail_after_replace(path: Path) -> None:
        assert path == tmp_path
        assert target.read_bytes() == b"following"
        raise OSError("simulated directory barrier failure")

    monkeypatch.setattr(reviewed_cases_module, "_fsync_directory", fail_after_replace)

    with pytest.raises(OSError, match="directory barrier failure"):
        reviewed_cases_module._atomic_bytes(target, b"following")

    assert target.read_bytes() == b"following"


def test_durable_replace_flushes_both_directories_and_propagates_barrier_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_parent = tmp_path / "staging"
    destination_parent = tmp_path / "vectors"
    source_parent.mkdir()
    destination_parent.mkdir()
    source = source_parent / "case.npz"
    destination = destination_parent / "case.npz"
    source.write_bytes(b"vectors")
    barriers: list[Path] = []

    def observe_after_replace(path: Path) -> None:
        assert destination.read_bytes() == b"vectors"
        assert not source.exists()
        barriers.append(path)
        if path == source_parent:
            raise OSError("simulated source directory barrier failure")

    monkeypatch.setattr(reviewed_cases_module, "_fsync_directory", observe_after_replace)

    with pytest.raises(OSError, match="source directory barrier failure"):
        reviewed_cases_module._durable_replace(source, destination)

    assert barriers == [destination_parent, source_parent]


def test_durable_unlink_fails_closed_after_removal_when_directory_barrier_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = tmp_path / ".pending-transition.json"
    pending.write_bytes(b"pending")

    def fail_after_unlink(path: Path) -> None:
        assert path == tmp_path
        assert not pending.exists()
        raise OSError("simulated directory barrier failure")

    monkeypatch.setattr(reviewed_cases_module, "_fsync_directory", fail_after_unlink)

    with pytest.raises(OSError, match="directory barrier failure"):
        reviewed_cases_module._durable_unlink(pending)

    assert not pending.exists()


def test_staging_removal_fails_closed_after_directory_barrier_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = ReviewedCaseIndex(tmp_path / "cases")
    index._staging.mkdir()
    (index._staging / "index.sqlite3").write_bytes(b"staged")

    def fail_after_removal(path: Path) -> None:
        assert path == index.root
        assert not index._staging.exists()
        raise OSError("simulated directory barrier failure")

    monkeypatch.setattr(reviewed_cases_module, "_fsync_directory", fail_after_removal)

    with pytest.raises(OSError, match="directory barrier failure"):
        index._remove_staging()

    assert not index._staging.exists()


def test_reviewed_case_index_is_idempotent_and_cross_graph_same_kind(tmp_path: Path) -> None:
    index = ReviewedCaseIndex(tmp_path / "cases")
    first, idempotent, first_hash = index.index(
        _metadata("case-1", graph_hash="3" * 64),
        _vectors(1),
        indexed_at="2026-08-18T01:00:00.000000Z",
        source_request_hash="4" * 64,
    )
    assert idempotent is False
    repeated, idempotent, repeated_hash = index.index(
        _metadata("case-1", graph_hash="3" * 64),
        _vectors(1),
        indexed_at="2026-08-18T02:00:00.000000Z",
        source_request_hash="4" * 64,
    )
    assert idempotent is True
    assert repeated == first
    assert repeated_hash == first_hash
    second, _, second_hash = index.index(
        _metadata("case-2", graph_hash="5" * 64),
        _vectors(2),
        indexed_at="2026-08-18T03:00:00.000000Z",
        source_request_hash="6" * 64,
    )
    assert second.graph_version_hash != first.graph_version_hash
    assert second_hash != first_hash
    other_model, _, _ = index.index(
        _metadata("case-3", graph_hash="7" * 64, model_hash="8" * 64),
        _vectors(3),
        indexed_at="2026-08-18T04:00:00.000000Z",
        source_request_hash="9" * 64,
    )
    assert other_model.model_state_hash != first.model_state_hash
    matches = index.query(
        _vectors(1),
        model_state_hash="2" * 64,
        kind_key="node+relation",
        limit=10,
        exclude_case_id="case-1",
    )
    assert [item.record.case_id for item in matches] == ["case-2"]
    assert all(item.record.model_state_hash == "2" * 64 for item in matches)
    assert matches[0].score == pytest.approx(
        0.7 * matches[0].embedding_score
        + 0.2 * matches[0].structure_score
        + 0.1 * matches[0].modality_score
    )
    assert 0 <= matches[0].score <= 1


def test_reviewed_case_index_rejects_changed_idempotent_payload(tmp_path: Path) -> None:
    index = ReviewedCaseIndex(tmp_path / "cases")
    index.index(
        _metadata("case-1", graph_hash="3" * 64),
        _vectors(1),
        indexed_at="2026-08-18T01:00:00.000000Z",
        source_request_hash="4" * 64,
    )
    with pytest.raises(ValueError, match="already bound"):
        index.index(
            _metadata("case-1", graph_hash="3" * 64),
            _vectors(2),
            indexed_at="2026-08-18T01:00:00.000000Z",
            source_request_hash="4" * 64,
        )


def test_reviewed_case_vector_and_sqlite_tampering_fail_closed(tmp_path: Path) -> None:
    index = ReviewedCaseIndex(tmp_path / "vectors")
    record, _, _ = index.index(
        _metadata("case-1", graph_hash="3" * 64),
        _vectors(1),
        indexed_at="2026-08-18T01:00:00.000000Z",
        source_request_hash="4" * 64,
    )
    vector_path = index.vector_root / record.vector_file
    raw = bytearray(vector_path.read_bytes())
    raw[-1] ^= 1
    vector_path.write_bytes(raw)
    with pytest.raises(ValueError, match="vector artifact identity"):
        _ = index.index_hash

    sqlite_index = ReviewedCaseIndex(tmp_path / "sqlite")
    sqlite_index.index(
        _metadata("case-2", graph_hash="5" * 64),
        _vectors(2),
        indexed_at="2026-08-18T03:00:00.000000Z",
        source_request_hash="6" * 64,
    )
    database = sqlite_index.database
    database.write_bytes(database.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SQLite identity"):
        sqlite_index.record("case-2")


def test_reviewed_case_contract_rejects_noncanonical_kind_entries(tmp_path: Path) -> None:
    index = ReviewedCaseIndex(tmp_path / "cases")
    metadata = _metadata("case-1", graph_hash="3" * 64)
    metadata["kindEntries"] = list(reversed(metadata["kindEntries"]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="canonical order"):
        index.index(
            metadata,
            _vectors(1),
            indexed_at="2026-08-18T01:00:00.000000Z",
            source_request_hash="4" * 64,
        )


def test_manifest_publish_failure_after_sqlite_commit_recovers_exact_next_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cases"
    index = ReviewedCaseIndex(root)
    previous_index_hash = index.index_hash
    original_atomic_json = reviewed_cases_module._atomic_json
    observed_committed_row = False

    def fail_next_manifest(path: Path, value: object) -> None:
        nonlocal observed_committed_row
        if path == index.manifest and isinstance(value, dict) and value.get("recordHashes"):
            with sqlite3.connect(index.database) as connection:
                rows = connection.execute("SELECT case_id FROM cases ORDER BY case_id").fetchall()
            assert rows == [("case-1",)]
            observed_committed_row = True
            raise OSError("simulated manifest publication failure")
        original_atomic_json(path, value)  # type: ignore[arg-type]

    monkeypatch.setattr(reviewed_cases_module, "_atomic_json", fail_next_manifest)
    with pytest.raises(OSError, match="manifest publication"):
        index.index(
            _metadata("case-1", graph_hash="3" * 64),
            _vectors(1),
            indexed_at="2026-08-18T01:00:00.000000Z",
            source_request_hash="4" * 64,
        )
    assert observed_committed_row is True

    monkeypatch.setattr(reviewed_cases_module, "_atomic_json", original_atomic_json)
    recovered = ReviewedCaseIndex(root)
    recovered_record = recovered.record("case-1")
    assert recovered.index_hash != previous_index_hash
    repeated, idempotent, repeated_index_hash = recovered.index(
        _metadata("case-1", graph_hash="3" * 64),
        _vectors(1),
        indexed_at="2026-08-18T02:00:00.000000Z",
        source_request_hash="4" * 64,
    )
    assert idempotent is True
    assert repeated == recovered_record
    assert repeated_index_hash == recovered.index_hash


def test_interruption_before_sqlite_replace_rolls_back_to_exact_previous_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cases"
    index = ReviewedCaseIndex(root)
    previous_index_hash = index.index_hash
    original_replace = reviewed_cases_module._durable_replace

    def interrupt_database_replace(source: Path, destination: Path) -> None:
        if destination == index.database:
            assert tuple(index.vector_root.iterdir())
            raise KeyboardInterrupt("simulated interruption before SQLite replace")
        original_replace(source, destination)

    monkeypatch.setattr(reviewed_cases_module, "_durable_replace", interrupt_database_replace)
    with pytest.raises(KeyboardInterrupt, match="before SQLite replace"):
        index.index(
            _metadata("case-1", graph_hash="3" * 64),
            _vectors(1),
            indexed_at="2026-08-18T01:00:00.000000Z",
            source_request_hash="4" * 64,
        )

    monkeypatch.setattr(reviewed_cases_module, "_durable_replace", original_replace)
    recovered = ReviewedCaseIndex(root)
    assert recovered.index_hash == previous_index_hash
    assert tuple(recovered.vector_root.iterdir()) == ()
    with pytest.raises(KeyError):
        recovered.record("case-1")
    _, idempotent, _ = recovered.index(
        _metadata("case-1", graph_hash="3" * 64),
        _vectors(1),
        indexed_at="2026-08-18T02:00:00.000000Z",
        source_request_hash="4" * 64,
    )
    assert idempotent is False


def test_pending_transition_does_not_authorize_tampered_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cases"
    index = ReviewedCaseIndex(root)
    original_atomic_json = reviewed_cases_module._atomic_json

    def fail_next_manifest(path: Path, value: object) -> None:
        if path == index.manifest and isinstance(value, dict) and value.get("recordHashes"):
            raise OSError("simulated manifest publication failure")
        original_atomic_json(path, value)  # type: ignore[arg-type]

    monkeypatch.setattr(reviewed_cases_module, "_atomic_json", fail_next_manifest)
    with pytest.raises(OSError, match="manifest publication"):
        index.index(
            _metadata("case-1", graph_hash="3" * 64),
            _vectors(1),
            indexed_at="2026-08-18T01:00:00.000000Z",
            source_request_hash="4" * 64,
        )

    monkeypatch.setattr(reviewed_cases_module, "_atomic_json", original_atomic_json)
    raw_database = bytearray(index.database.read_bytes())
    raw_database[0] ^= 1
    index.database.write_bytes(raw_database)
    with pytest.raises(ValueError, match="SQLite is outside the pending transition"):
        ReviewedCaseIndex(root)


def test_concurrent_writers_serialize_complete_transitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cases"
    first = ReviewedCaseIndex(root)
    second = ReviewedCaseIndex(root)
    roles = threading.local()
    first_paused = threading.Event()
    release_first = threading.Event()
    contender_started = threading.Event()
    contender_recovery = threading.Event()
    contender_recovery_done = threading.Event()
    original_replace = reviewed_cases_module._durable_replace
    original_recover = ReviewedCaseIndex._recover_transition

    def pause_first_after_vector_replace(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        if getattr(roles, "value", None) == "first" and destination.parent == first.vector_root:
            first_paused.set()
            assert release_first.wait(timeout=10)

    def observe_contender_recovery(self: ReviewedCaseIndex):  # type: ignore[no-untyped-def]
        if getattr(roles, "value", None) == "contender":
            contender_recovery.set()
        try:
            return original_recover(self)
        finally:
            if getattr(roles, "value", None) == "contender":
                contender_recovery_done.set()

    def write(index: ReviewedCaseIndex, case_id: str, role: str, seed: int):
        roles.value = role
        if role == "contender":
            contender_started.set()
        return index.index(
            _metadata(case_id, graph_hash=f"{seed + 2:x}" * 64),
            _vectors(seed),
            indexed_at=f"2026-08-18T0{seed}:00:00.000000Z",
            source_request_hash=f"{seed + 3:x}" * 64,
        )

    monkeypatch.setattr(reviewed_cases_module, "_durable_replace", pause_first_after_vector_replace)
    monkeypatch.setattr(ReviewedCaseIndex, "_recover_transition", observe_contender_recovery)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(write, first, "case-1", "first", 1)
        assert first_paused.wait(timeout=10)
        second_future = pool.submit(write, second, "case-2", "contender", 2)
        assert contender_started.wait(timeout=10)
        if contender_recovery.wait(timeout=1):
            assert contender_recovery_done.wait(timeout=10)
        release_first.set()
        first_result = first_future.result(timeout=10)
        second_result = second_future.result(timeout=10)

    assert first_result[0].case_id == "case-1"
    assert second_result[0].case_id == "case-2"
    recovered = ReviewedCaseIndex(root)
    assert recovered.record("case-1").case_id == "case-1"
    assert recovered.record("case-2").case_id == "case-2"


def test_reader_waits_for_complete_writer_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "cases"
    writer = ReviewedCaseIndex(root)
    reader = ReviewedCaseIndex(root)
    roles = threading.local()
    writer_paused = threading.Event()
    release_writer = threading.Event()
    reader_started = threading.Event()
    reader_recovery = threading.Event()
    reader_recovery_done = threading.Event()
    original_replace = reviewed_cases_module._durable_replace
    original_recover = ReviewedCaseIndex._recover_transition

    def pause_writer_after_vector_replace(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        if getattr(roles, "value", None) == "writer" and destination.parent == writer.vector_root:
            writer_paused.set()
            assert release_writer.wait(timeout=10)

    def observe_reader_recovery(self: ReviewedCaseIndex):  # type: ignore[no-untyped-def]
        if getattr(roles, "value", None) == "reader":
            reader_recovery.set()
        try:
            return original_recover(self)
        finally:
            if getattr(roles, "value", None) == "reader":
                reader_recovery_done.set()

    def write():  # type: ignore[no-untyped-def]
        roles.value = "writer"
        return writer.index(
            _metadata("case-1", graph_hash="3" * 64),
            _vectors(1),
            indexed_at="2026-08-18T01:00:00.000000Z",
            source_request_hash="4" * 64,
        )

    def read():  # type: ignore[no-untyped-def]
        roles.value = "reader"
        reader_started.set()
        return reader.record("case-1")

    monkeypatch.setattr(
        reviewed_cases_module, "_durable_replace", pause_writer_after_vector_replace
    )
    monkeypatch.setattr(ReviewedCaseIndex, "_recover_transition", observe_reader_recovery)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer_future = pool.submit(write)
        assert writer_paused.wait(timeout=10)
        reader_future = pool.submit(read)
        assert reader_started.wait(timeout=10)
        if reader_recovery.wait(timeout=1):
            assert reader_recovery_done.wait(timeout=10)
        release_writer.set()
        writer_result = writer_future.result(timeout=10)
        reader_result = reader_future.result(timeout=10)

    assert writer_result[0].case_id == "case-1"
    assert reader_result.case_id == "case-1"


def test_root_lock_serializes_a_separate_process(tmp_path: Path) -> None:
    index = ReviewedCaseIndex(tmp_path / "cases")
    ready = tmp_path / "child-ready"
    release = tmp_path / "child-release"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_hold_reviewed_case_root_lock,
        args=(str(index.root), str(ready), str(release)),
    )
    process.start()
    try:
        deadline = time.monotonic() + 15
        while not ready.exists():
            if not process.is_alive() or time.monotonic() >= deadline:
                raise AssertionError("child did not acquire reviewed-case root lock")
            time.sleep(0.01)
        with ThreadPoolExecutor(max_workers=1) as pool:
            blocked_read = pool.submit(lambda: index.index_hash)
            time.sleep(0.2)
            assert blocked_read.done() is False
            release.write_text("release", encoding="utf-8")
            assert blocked_read.result(timeout=10) == index.index_hash
    finally:
        release.write_text("release", encoding="utf-8")
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
    assert process.exitcode == 0


def test_root_lock_is_reentrant_for_public_reads(tmp_path: Path) -> None:
    index = ReviewedCaseIndex(tmp_path / "cases")
    with index._locked():
        assert index.index_hash == index._read_manifest()["indexHash"]


def test_manifest_and_transition_require_canonical_unique_key_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_index = ReviewedCaseIndex(tmp_path / "manifest")
    manifest_index.manifest.write_bytes(manifest_index.manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        ReviewedCaseIndex(manifest_index.root)

    transition_index = ReviewedCaseIndex(tmp_path / "transition")
    original_atomic_json = reviewed_cases_module._atomic_json

    def fail_next_manifest(path: Path, value: object) -> None:
        if (
            path == transition_index.manifest
            and isinstance(value, dict)
            and value.get("recordHashes")
        ):
            raise OSError("simulated manifest publication failure")
        original_atomic_json(path, value)  # type: ignore[arg-type]

    monkeypatch.setattr(reviewed_cases_module, "_atomic_json", fail_next_manifest)
    with pytest.raises(OSError, match="manifest publication"):
        transition_index.index(
            _metadata("case-1", graph_hash="3" * 64),
            _vectors(1),
            indexed_at="2026-08-18T01:00:00.000000Z",
            source_request_hash="4" * 64,
        )
    monkeypatch.setattr(reviewed_cases_module, "_atomic_json", original_atomic_json)
    raw_transition = transition_index._transition.read_bytes()
    duplicate_schema = (
        b'{"schemaVersion":"socialgraph-fm.governance-reviewed-case-transition/1.0",'
        + raw_transition[1:]
    )
    transition_index._transition.write_bytes(duplicate_schema)
    with pytest.raises(ValueError, match="duplicate"):
        ReviewedCaseIndex(transition_index.root)


def test_sqlite_authentication_and_consumption_use_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = ReviewedCaseIndex(tmp_path / "source")
    index.index(
        _metadata("case-1", graph_hash="3" * 64),
        _vectors(1),
        indexed_at="2026-08-18T01:00:00.000000Z",
        source_request_hash="4" * 64,
    )
    replacement = ReviewedCaseIndex(tmp_path / "replacement")
    replacement.index(
        _metadata("case-2", graph_hash="5" * 64),
        _vectors(2),
        indexed_at="2026-08-18T02:00:00.000000Z",
        source_request_hash="6" * 64,
    )
    replacement_bytes = replacement.database.read_bytes()
    original_records = ReviewedCaseIndex._records
    swapped = False

    def swap_live_path_before_consumption(
        self: ReviewedCaseIndex,
        database: Path | None = None,
    ):
        nonlocal swapped
        if self is index and not swapped:
            index.database.write_bytes(replacement_bytes)
            swapped = True
        return original_records(self, database)

    monkeypatch.setattr(ReviewedCaseIndex, "_records", swap_live_path_before_consumption)
    assert index.record("case-1").case_id == "case-1"
    assert swapped is True


def test_staged_sqlite_must_copy_the_authenticated_previous_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = ReviewedCaseIndex(tmp_path / "cases")
    injected_database = tmp_path / "injected.sqlite3"
    injected_database.write_bytes(index.database.read_bytes())
    with sqlite3.connect(injected_database) as connection:
        connection.execute("CREATE TABLE injected_unverified_data (value TEXT)")
        connection.commit()
    injected_bytes = injected_database.read_bytes()
    original_remove_staging = index._remove_staging
    swapped = False

    def swap_after_verification_before_staging() -> None:
        nonlocal swapped
        original_remove_staging()
        if not swapped:
            index.database.write_bytes(injected_bytes)
            swapped = True

    monkeypatch.setattr(index, "_remove_staging", swap_after_verification_before_staging)
    with pytest.raises(ValueError, match="SQLite identity"):
        index.index(
            _metadata("case-1", graph_hash="3" * 64),
            _vectors(1),
            indexed_at="2026-08-18T01:00:00.000000Z",
            source_request_hash="4" * 64,
        )
    assert swapped is True


def test_vector_authentication_and_consumption_use_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = ReviewedCaseIndex(tmp_path / "source")
    record, _, _ = index.index(
        _metadata("case-1", graph_hash="3" * 64),
        _vectors(1),
        indexed_at="2026-08-18T01:00:00.000000Z",
        source_request_hash="4" * 64,
    )
    replacement = ReviewedCaseIndex(tmp_path / "replacement")
    replacement_record, _, _ = replacement.index(
        _metadata("case-2", graph_hash="5" * 64),
        _vectors(2),
        indexed_at="2026-08-18T02:00:00.000000Z",
        source_request_hash="6" * 64,
    )
    vector_path = index.vector_root / record.vector_file
    replacement_bytes = (replacement.vector_root / replacement_record.vector_file).read_bytes()
    original_zip_file = reviewed_cases_module.zipfile.ZipFile
    swapped = False

    def swap_live_path_before_zip_parse(file: object, *args: object, **kwargs: object):
        nonlocal swapped
        if not swapped:
            vector_path.write_bytes(replacement_bytes)
            swapped = True
        return original_zip_file(file, *args, **kwargs)

    monkeypatch.setattr(reviewed_cases_module.zipfile, "ZipFile", swap_live_path_before_zip_parse)
    loaded = reviewed_cases_module._load_vector(vector_path, record)
    assert swapped is True
    np.testing.assert_allclose(loaded.embedding, _vectors(1).embedding)
