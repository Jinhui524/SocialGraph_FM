from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from socialgraph_gfm.canonical import canonical_sha256, file_sha256
from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.gfm.corpus import text_embeddings
from socialgraph_gfm.gfm.corpus.common import (
    array_inventory,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_npz,
    load_npz_safe,
    portable_id_hash,
    read_json_object,
)
from socialgraph_gfm.gfm.corpus.text_embeddings import (
    EmbeddingConfig,
    _model_inventory,
    build_bge_m3_embeddings,
    iter_embedding_shards,
    lookup_embedding_rows,
    open_embedding_artifact,
    open_embedding_artifact_view,
    verify_embedding_artifact,
)


def test_model_inventory_allows_only_cache_contained_snapshot_links(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "models--BAAI--bge-m3"
    blob = repository / "blobs" / "config"
    blob.parent.mkdir(parents=True)
    blob.write_text("{}", encoding="utf-8")
    snapshot = repository / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    link = snapshot / "config.json"
    try:
        link.symlink_to(blob)
    except OSError:
        pytest.skip("symbolic-link creation is unavailable on this Windows host")
    inventory, digest = _model_inventory(snapshot)
    assert inventory[0]["path"] == "config.json"
    assert len(digest) == 64

    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    escaping = snapshot / "escaping.json"
    escaping.symlink_to(outside)
    with pytest.raises(ContractViolation, match="escaping symbolic link"):
        _model_inventory(snapshot)


def test_model_inventory_uses_portable_case_sensitive_relative_order(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    for name in ("z.json", "README.md", "config.json", "A.json"):
        (model / name).write_text(name, encoding="utf-8")

    inventory, digest = _model_inventory(model)

    assert [item["path"] for item in inventory] == sorted(
        ["z.json", "README.md", "config.json", "A.json"]
    )
    assert digest == canonical_sha256(inventory)


def _model(tmp_path: Path) -> Path:
    model = tmp_path / "5617a9f61b028005a4858fdac845db406aefb181"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    return model


def _unit_encoder(values: Sequence[str]) -> np.ndarray:
    result = np.zeros((len(values), 1024), dtype=np.float32)
    result[:, 0] = 1.0
    return result


def test_formal_encoder_forbids_cpu_and_has_no_wrapper_progress_or_shrink_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    fake_flag_embedding = types.ModuleType("FlagEmbedding")
    fake_flag_embedding.BGEM3FlagModel = object
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_flag_embedding)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ContractViolation, match="requires CUDA; CPU fallback is forbidden"):
        text_embeddings._offline_encoder(
            EmbeddingConfig(corpus_id="cuda-required"), _model(tmp_path)
        )
    source = inspect.getsource(text_embeddings._offline_encoder)
    assert "model.encode(" not in source
    assert "while flag" not in source


def test_pinned_offline_embedding_artifact_and_hash_verification(tmp_path: Path) -> None:
    text = tmp_path / "text.jsonl"
    atomic_write_jsonl(
        text,
        [
            {"id": "W1", "text": "graph governance", "timestamp": 1},
            {"id": "W2", "text": "social networks", "timestamp": 2},
        ],
    )
    config = EmbeddingConfig(corpus_id="fixture", batch_size=1)
    manifest = build_bge_m3_embeddings(
        text, tmp_path, config=config, model_dir=_model(tmp_path), encoder=_unit_encoder
    )
    assert manifest["dimension"] == 1024
    assert manifest["privacy"]["trainingTimeEncoding"] is False
    assert manifest["producer"] == {
        "implementation": "injected-test-encoder",
        "distribution": None,
        "version": None,
        "formalEligible": False,
    }
    output = tmp_path / "embeddings/fixture-bge-m3-v1"
    assert verify_embedding_artifact(output)["rows"] == 2
    shard = output / "embeddings-00000.npz"
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(ContractViolation, match="hash mismatch"):
        verify_embedding_artifact(output)


def test_online_embedding_is_forbidden(tmp_path: Path) -> None:
    text = tmp_path / "text.jsonl"
    atomic_write_jsonl(text, [{"id": "1", "text": "x", "timestamp": 1}])
    with pytest.raises(ContractViolation, match="online"):
        build_bge_m3_embeddings(
            text,
            tmp_path,
            config=EmbeddingConfig(corpus_id="fixture"),
            model_dir=_model(tmp_path),
            encoder=lambda _: np.zeros((1, 1024), dtype=np.float32),
            offline=False,
        )


def test_embedding_role_view_does_not_open_future_shards(tmp_path: Path) -> None:
    text = tmp_path / "text.jsonl"
    atomic_write_jsonl(
        text,
        [
            {"id": "W1", "text": "train", "timestamp": 1},
            {"id": "W2", "text": "validation", "timestamp": 2},
            {"id": "W3", "text": "test", "timestamp": 3},
            {"id": "W4", "text": "shadow", "timestamp": 4},
        ],
    )
    roles = {1: "train", 2: "validation", 3: "test", 4: "shadow"}
    manifest = build_bge_m3_embeddings(
        text,
        tmp_path,
        config=EmbeddingConfig(corpus_id="role-fixture", batch_size=2),
        model_dir=_model(tmp_path),
        encoder=_unit_encoder,
        role_resolver=lambda _identifier, timestamp: roles[timestamp],
    )
    output = tmp_path / "embeddings/role-fixture-bge-m3-v1"
    test_path = Path(manifest["splits"]["physicalAccess"]["roleShards"]["test"][0])
    (output / test_path).write_bytes(b"corrupt future embedding")

    validation = open_embedding_artifact_view(output, maximum_role="validation")
    assert validation.rows == 2
    assert [int(item.timestamp[0]) for item in validation.iter_shards()] == [1, 2]
    with pytest.raises(ContractViolation, match="hash mismatch"):
        open_embedding_artifact_view(output, maximum_role="test")

    validation_path = Path(
        manifest["splits"]["physicalAccess"]["roleShards"]["validation"][0]
    )
    (output / validation_path).write_bytes(b"corrupt validation embedding")
    with pytest.raises(ContractViolation, match="hash mismatch"):
        open_embedding_artifact_view(output, maximum_role="validation")


def _role_view_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    text = tmp_path / "text.jsonl"
    atomic_write_jsonl(
        text,
        [
            {"id": "W1", "text": "train", "timestamp": 1},
            {"id": "W2", "text": "validation", "timestamp": 2},
            {"id": "W3", "text": "test", "timestamp": 3},
            {"id": "W4", "text": "shadow", "timestamp": 4},
        ],
    )
    roles = {1: "train", 2: "validation", 3: "test", 4: "shadow"}
    manifest = build_bge_m3_embeddings(
        text,
        tmp_path,
        config=EmbeddingConfig(corpus_id="role-security-fixture", batch_size=2),
        model_dir=_model(tmp_path),
        encoder=_unit_encoder,
        role_resolver=lambda _identifier, timestamp: roles[timestamp],
    )
    return tmp_path / "embeddings/role-security-fixture-bge-m3-v1", manifest


def _reseal_embedding_manifest(manifest: dict[str, object]) -> None:
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"logicalHash", "createdAt"}
    }
    manifest["logicalHash"] = canonical_sha256(payload)


def test_embedding_role_view_rejects_resealed_path_traversal(tmp_path: Path) -> None:
    output, raw_manifest = _role_view_fixture(tmp_path)
    manifest = dict(raw_manifest)
    manifest["splits"] = dict(manifest["splits"])  # type: ignore[arg-type]
    physical = dict(manifest["splits"]["physicalAccess"])  # type: ignore[index]
    role_shards = {
        role: list(paths)
        for role, paths in physical["roleShards"].items()  # type: ignore[union-attr]
    }
    physical["roleShards"] = role_shards
    manifest["splits"]["physicalAccess"] = physical  # type: ignore[index]
    original = role_shards["train"][0]
    outside = tmp_path / "outside.npz"
    outside.write_bytes((output / original).read_bytes())
    role_shards["train"][0] = "../outside.npz"
    shards = [dict(record) for record in manifest["shards"]]  # type: ignore[arg-type]
    next(record for record in shards if record["path"] == original)["path"] = "../outside.npz"
    manifest["shards"] = shards
    _reseal_embedding_manifest(manifest)
    atomic_write_json(output / "manifest.json", manifest)

    with pytest.raises(ContractViolation, match="role shard ordering|unsafe|traverses"):
        open_embedding_artifact_view(output, maximum_role="validation")


@pytest.mark.parametrize("mutation", ["schema", "rows", "extra-field"])
def test_embedding_role_view_rejects_resealed_malformed_manifest(
    tmp_path: Path, mutation: str
) -> None:
    output, raw_manifest = _role_view_fixture(tmp_path)
    manifest = dict(raw_manifest)
    if mutation == "schema":
        manifest["schemaVersion"] = "gfm.text-embeddings/9.9"
    else:
        shards = [dict(record) for record in manifest["shards"]]  # type: ignore[arg-type]
        role_record = next(
            record for record in shards if str(record["path"]).startswith("role-train-")
        )
        if mutation == "rows":
            role_record["rows"] = 8193
        else:
            role_record["unexpected"] = True
        manifest["shards"] = shards
    _reseal_embedding_manifest(manifest)
    atomic_write_json(output / "manifest.json", manifest)

    with pytest.raises(ContractViolation, match="manifest|role shard"):
        open_embedding_artifact_view(output, maximum_role="validation")


def test_streams_bounded_encoder_batches_and_multiple_bounded_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(text_embeddings, "EMBEDDING_ROWS_PER_SHARD", 4)
    text = tmp_path / "text.jsonl"
    atomic_write_jsonl(
        text,
        ({"id": f"W{index}", "text": f"row {index}", "timestamp": index} for index in range(10)),
    )
    calls: list[int] = []

    def encoder(values: Sequence[str]) -> np.ndarray:
        calls.append(len(values))
        return _unit_encoder(values)

    manifest = build_bge_m3_embeddings(
        text,
        tmp_path,
        config=EmbeddingConfig(corpus_id="streaming", batch_size=3),
        model_dir=_model(tmp_path),
        encoder=encoder,
    )
    assert calls == [3, 3, 3, 1]
    assert max(calls) == 3
    assert manifest["rows"] == 10
    assert manifest["rowsPerShard"] == 4
    assert [record["rows"] for record in manifest["shards"]] == [4, 4, 2]
    output = tmp_path / "embeddings/streaming-bge-m3-v1"
    shards = list(iter_embedding_shards(output))
    assert [item.embedding.shape for item in shards] == [(4, 1024), (4, 1024), (2, 1024)]
    assert all(not item.embedding.flags.writeable for item in shards)
    handle = open_embedding_artifact(output)
    assert handle.shard_count == 3
    assert handle.load_shard(1).id_hash.shape == (4,)
    with pytest.raises(ContractViolation, match="index"):
        handle.load_shard(3)

    first = int(portable_id_hash("W1"))
    last = int(portable_id_hash("W9"))
    selected = lookup_embedding_rows(output, [first, last])
    assert set(selected) == {first, last}
    assert selected[first][1] == 1
    assert selected[last][1] == 9
    assert np.linalg.norm(selected[last][0]) == pytest.approx(1.0)


def test_encoder_failure_removes_staging_and_never_publishes(
    tmp_path: Path,
) -> None:
    text = tmp_path / "text.jsonl"
    atomic_write_jsonl(
        text,
        [
            {"id": "W1", "text": "first", "timestamp": 1},
            {"id": "W2", "text": "second", "timestamp": 2},
        ],
    )
    calls = 0

    def encoder(values: Sequence[str]) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("encoder failed")
        return _unit_encoder(values)

    with pytest.raises(RuntimeError, match="encoder failed"):
        build_bge_m3_embeddings(
            text,
            tmp_path,
            config=EmbeddingConfig(corpus_id="failed", batch_size=1),
            model_dir=_model(tmp_path),
            encoder=encoder,
        )
    embeddings = tmp_path / "embeddings"
    assert not (embeddings / "failed-bge-m3-v1").exists()
    assert not list(embeddings.glob(".failed-bge-m3-v1.*.tmp"))


def test_completed_canonical_shard_is_hash_verified_and_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(text_embeddings, "EMBEDDING_ROWS_PER_SHARD", 2)
    text = tmp_path / "text.jsonl"
    atomic_write_jsonl(
        text,
        ({"id": f"W{index}", "text": f"row {index}", "timestamp": index} for index in range(5)),
    )
    first_calls = 0
    progress: list[dict[str, object]] = []

    def interrupted(values: Sequence[str]) -> np.ndarray:
        nonlocal first_calls
        first_calls += 1
        if first_calls == 2:
            raise RuntimeError("planned interruption")
        return _unit_encoder(values)

    with pytest.raises(RuntimeError, match="planned interruption"):
        build_bge_m3_embeddings(
            text,
            tmp_path,
            config=EmbeddingConfig(corpus_id="resumable", batch_size=2),
            model_dir=_model(tmp_path),
            encoder=interrupted,
            progress_callback=progress.append,
        )
    staging = next((tmp_path / "embeddings").glob(".resumable-bge-m3-v1.*.resume"))
    completed = staging / "embeddings-00000.npz"
    completed_hash = file_sha256(completed)
    resumed_calls: list[list[str]] = []

    def resumed(values: Sequence[str]) -> np.ndarray:
        resumed_calls.append(list(values))
        return _unit_encoder(values)

    manifest = build_bge_m3_embeddings(
        text,
        tmp_path,
        config=EmbeddingConfig(corpus_id="resumable", batch_size=2),
        model_dir=staging.parent.parent / "5617a9f61b028005a4858fdac845db406aefb181",
        encoder=resumed,
        progress_callback=progress.append,
    )
    output = tmp_path / "embeddings/resumable-bge-m3-v1"
    assert manifest["rows"] == 5
    assert resumed_calls == [["row 2", "row 3"], ["row 4"]]
    assert file_sha256(output / "embeddings-00000.npz") == completed_hash
    assert not staging.exists()
    assert not list((tmp_path / "embeddings").glob(".resumable-*.resume.json"))
    canonical = [item for item in progress if item["phase"] == "canonical"]
    assert [item["rowsCompleted"] for item in canonical] == [2, 4, 5]
    assert canonical[-1]["schemaVersion"] == "gfm.text-embedding-progress/1.0"
    assert canonical[-1]["resumedRows"] == 2
    assert canonical[-1]["etaSeconds"] == 0.0


def test_tampered_completed_resume_shard_is_rejected_before_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(text_embeddings, "EMBEDDING_ROWS_PER_SHARD", 2)
    text = tmp_path / "text.jsonl"
    atomic_write_jsonl(
        text,
        [
            {"id": "W1", "text": "one", "timestamp": 1},
            {"id": "W2", "text": "two", "timestamp": 2},
            {"id": "W3", "text": "three", "timestamp": 3},
        ],
    )
    calls = 0

    def interrupted(values: Sequence[str]) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("stop")
        return _unit_encoder(values)

    with pytest.raises(RuntimeError, match="stop"):
        build_bge_m3_embeddings(
            text,
            tmp_path,
            config=EmbeddingConfig(corpus_id="resume-tamper", batch_size=2),
            model_dir=_model(tmp_path),
            encoder=interrupted,
        )
    staging = next((tmp_path / "embeddings").glob(".resume-tamper-*.resume"))
    shard = staging / "embeddings-00000.npz"
    shard.write_bytes(shard.read_bytes() + b"tamper")

    def forbidden(_: Sequence[str]) -> np.ndarray:
        raise AssertionError("tampered resume state must fail before encoding")

    with pytest.raises(ContractViolation, match="resumable embedding shard hash mismatch"):
        build_bge_m3_embeddings(
            text,
            tmp_path,
            config=EmbeddingConfig(corpus_id="resume-tamper", batch_size=2),
            model_dir=staging.parent.parent / "5617a9f61b028005a4858fdac845db406aefb181",
            encoder=forbidden,
        )


def test_role_partition_resumes_without_reencoding_canonical_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(text_embeddings, "EMBEDDING_ROWS_PER_SHARD", 2)
    text = tmp_path / "text.jsonl"
    atomic_write_jsonl(
        text,
        ({"id": f"W{index}", "text": f"row {index}", "timestamp": index} for index in range(5)),
    )

    def stop_after_first_role(value: dict[str, object]) -> None:
        if value["phase"] == "roles":
            raise RuntimeError("role interruption")

    with pytest.raises(RuntimeError, match="role interruption"):
        build_bge_m3_embeddings(
            text,
            tmp_path,
            config=EmbeddingConfig(corpus_id="role-resume", batch_size=2),
            model_dir=_model(tmp_path),
            encoder=_unit_encoder,
            role_resolver=lambda _identifier, _timestamp: "train",
            progress_callback=stop_after_first_role,
        )

    def forbidden(_: Sequence[str]) -> np.ndarray:
        raise AssertionError("role resume must not re-encode canonical rows")

    manifest = build_bge_m3_embeddings(
        text,
        tmp_path,
        config=EmbeddingConfig(corpus_id="role-resume", batch_size=2),
        model_dir=tmp_path / "5617a9f61b028005a4858fdac845db406aefb181",
        encoder=forbidden,
        role_resolver=lambda _identifier, _timestamp: "train",
    )
    access = manifest["splits"]["physicalAccess"]
    assert (
        sum(
            int(record["rows"])
            for record in manifest["shards"]
            if record["path"].startswith("role-train-")
        )
        == 5
    )
    assert access["roleShards"]["validation"] == ["role-validation-embeddings-00000.npz"]
    assert not list((tmp_path / "embeddings").glob(".embedding-audit-*.sqlite3"))


def test_duplicate_identifier_across_future_shards_fails_closed_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(text_embeddings, "EMBEDDING_ROWS_PER_SHARD", 2)
    text = tmp_path / "text.jsonl"
    atomic_write_jsonl(
        text,
        [
            {"id": "W1", "text": "one", "timestamp": 1},
            {"id": "W2", "text": "two", "timestamp": 2},
            {"id": "W1", "text": "duplicate", "timestamp": 3},
        ],
    )
    with pytest.raises(ContractViolation, match="duplicate identifier or uint64 hash collision"):
        build_bge_m3_embeddings(
            text,
            tmp_path,
            config=EmbeddingConfig(corpus_id="duplicate", batch_size=1),
            model_dir=_model(tmp_path),
            encoder=_unit_encoder,
        )
    assert not (tmp_path / "embeddings/duplicate-bge-m3-v1").exists()


def test_semantic_shard_tamper_is_rejected_even_with_resealed_hashes(tmp_path: Path) -> None:
    text = tmp_path / "text.jsonl"
    atomic_write_jsonl(text, [{"id": "W1", "text": "graph", "timestamp": 1}])
    build_bge_m3_embeddings(
        text,
        tmp_path,
        config=EmbeddingConfig(corpus_id="semantic-tamper"),
        model_dir=_model(tmp_path),
        encoder=_unit_encoder,
    )
    output = tmp_path / "embeddings/semantic-tamper-bge-m3-v1"
    shard = output / "embeddings-00000.npz"
    arrays = load_npz_safe(
        shard,
        expected={
            "embedding": (np.dtype(np.float32).str, 2),
            "id_hash": (np.dtype(np.uint64).str, 1),
            "timestamp": (np.dtype(np.int64).str, 1),
        },
    )
    arrays["embedding"][:] = 0.0
    atomic_write_npz(shard, arrays)
    manifest = read_json_object(output / "manifest.json")
    manifest["shards"][0]["sha256"] = file_sha256(shard)
    manifest["shards"][0]["arrays"] = array_inventory(arrays)
    payload = {
        key: value for key, value in manifest.items() if key not in {"logicalHash", "createdAt"}
    }
    manifest["logicalHash"] = canonical_sha256(payload)
    atomic_write_json(output / "manifest.json", manifest)
    with pytest.raises(ContractViolation, match="non-normalized"):
        verify_embedding_artifact(output)


def test_existing_artifact_is_revalidated_without_reencoding(tmp_path: Path) -> None:
    text = tmp_path / "text.jsonl"
    atomic_write_jsonl(text, [{"id": "W1", "text": "graph", "timestamp": 1}])
    model = _model(tmp_path)
    config = EmbeddingConfig(corpus_id="existing")
    first = build_bge_m3_embeddings(
        text, tmp_path, config=config, model_dir=model, encoder=_unit_encoder
    )

    def forbidden_encoder(_: Sequence[str]) -> np.ndarray:
        raise AssertionError("existing artifacts must not be re-encoded")

    second = build_bge_m3_embeddings(
        text, tmp_path, config=config, model_dir=model, encoder=forbidden_encoder
    )
    assert second["logicalHash"] == first["logicalHash"]
