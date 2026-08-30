"""Pinned, offline BGE-M3 embedding artifacts for immutable corpus text."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import time
import uuid
import importlib.metadata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator, Mapping
from typing import Any, Callable, Literal, Sequence

import numpy as np

from ...canonical import canonical_json, canonical_sha256, file_sha256
from ...errors import ContractViolation, MissingRuntimeDependency
from ...runtime import RuntimeLayout
from .common import (
    PORTABLE_ID_HASH_ALGORITHM,
    NumericShardWriter,
    ShardRecord,
    array_inventory,
    atomic_write_json,
    build_manifest,
    exclusive_file_lock,
    load_npz_safe,
    portable_id_hash,
    read_json_object,
    read_jsonl,
    resolve_within,
    safe_relative_path,
    verify_manifest,
)

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MODEL_LICENSE = "MIT"
OUTPUT_DIM = 1024
MAX_TOKENS = 512
EMBEDDING_ROWS_PER_SHARD = 8192
EMBEDDING_ACCESS_SCHEMA = "gfm.embedding-role-access/1.0"
EMBEDDING_ACCESS_ROLES = ("train", "validation", "test", "shadow")
EMBEDDING_RESUME_SCHEMA = "gfm.text-embedding-resume/1.0"
EMBEDDING_PROGRESS_SCHEMA = "gfm.text-embedding-progress/1.0"
Encoder = Callable[[Sequence[str]], np.ndarray]
ProgressCallback = Callable[[Mapping[str, Any]], None]
EmbeddingRole = Literal["train", "validation", "test", "shadow"]
RoleResolver = Callable[[str, int], EmbeddingRole]


def _fail(message: str) -> ContractViolation:
    return ContractViolation(f"BGE-M3 embeddings: {message}")


@dataclass(frozen=True)
class EmbeddingConfig:
    corpus_id: str
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    output_dim: int = OUTPUT_DIM
    max_tokens: int = MAX_TOKENS
    batch_size: int = 16
    normalized: bool = True

    def validate(self) -> None:
        if not self.corpus_id or any(char in self.corpus_id for char in "/\\:"):
            raise _fail("corpus_id is unsafe")
        if (
            self.model_id != MODEL_ID
            or self.revision != MODEL_REVISION
            or self.output_dim != OUTPUT_DIM
            or self.max_tokens != MAX_TOKENS
        ):
            raise _fail("model identity/dimensions differ from the pinned formal configuration")
        if not 1 <= self.batch_size <= 256 or self.normalized is not True:
            raise _fail("batch size or normalization contract is invalid")

    @property
    def identity(self) -> str:
        return canonical_sha256(self.__dict__)


@dataclass(frozen=True)
class EmbeddingShard:
    """One verified, bounded shard exposed without joining the full matrix."""

    index: int
    embedding: np.ndarray
    id_hash: np.ndarray
    timestamp: np.ndarray


@dataclass(frozen=True)
class VerifiedEmbeddingArtifact:
    """A once-verified artifact handle for efficient bounded random access."""

    directory: Path
    logical_hash: str
    rows: int
    shard_hashes: tuple[str, ...]
    shard_rows: tuple[int, ...]
    shard_paths: tuple[str, ...] = ()

    @property
    def shard_count(self) -> int:
        return len(self.shard_hashes)

    def load_shard(self, index: int) -> EmbeddingShard:
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.shard_count
        ):
            raise _fail("embedding shard index is outside the manifest")
        relative = self.shard_paths[index] if self.shard_paths else f"embeddings-{index:05d}.npz"
        path = self.directory / relative
        expected_hash = self.shard_hashes[index]
        if not path.is_file() or path.is_symlink():
            raise _fail(f"embedding shard is absent or unsafe: {path.name}")
        try:
            if file_sha256(path) != expected_hash:
                raise _fail(f"artifact hash mismatch: {path.name}")
            arrays = _load_embedding_arrays(path)
            if file_sha256(path) != expected_hash:
                raise _fail(f"artifact changed while loading: {path.name}")
        except OSError as exc:
            raise _fail(f"embedding shard cannot be read: {path.name}") from exc
        rows = self.shard_rows[index]
        if (
            arrays["embedding"].shape != (rows, OUTPUT_DIM)
            or arrays["id_hash"].shape != (rows,)
            or arrays["timestamp"].shape != (rows,)
            or not np.allclose(
                np.linalg.norm(arrays["embedding"], axis=1),
                1.0,
                rtol=1e-4,
                atol=1e-5,
            )
            or np.unique(arrays["id_hash"]).size != rows
        ):
            raise _fail("embedding shard violates its verified row schema")
        for array in arrays.values():
            array.flags.writeable = False
        return EmbeddingShard(
            index=index,
            embedding=arrays["embedding"],
            id_hash=arrays["id_hash"],
            timestamp=arrays["timestamp"],
        )

    def iter_shards(self) -> Iterator[EmbeddingShard]:
        for index in range(self.shard_count):
            yield self.load_shard(index)


def _model_inventory(model_dir: Path) -> tuple[list[dict[str, Any]], str]:
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise _fail("pinned local model directory is absent or unsafe")
    # A normal Hugging Face snapshot may represent files as links into the
    # model repository's ``blobs`` directory.  Permit only those contained
    # links, while still rejecting a link that escapes the model cache.
    repository_root = model_dir.parent.parent.resolve()
    files = []
    # ``Path`` ordering is platform-dependent (Windows compares paths without
    # the same case-sensitive semantics as POSIX).  The manifest verifier and
    # portable identity operate on POSIX relative strings, so inventory files
    # must be ordered by that exact representation on every host.
    for path in sorted(
        model_dir.rglob("*"),
        key=lambda item: item.relative_to(model_dir).as_posix(),
    ):
        if path.is_symlink():
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(repository_root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise _fail("local model snapshot contains an escaping symbolic link") from exc
            if not resolved.is_file():
                raise _fail("local model snapshot contains a non-file symbolic link")
        if not path.is_file():
            continue
        relative = path.relative_to(model_dir).as_posix()
        if relative.startswith("../") or path.stat().st_size > 8 * 1024 * 1024 * 1024:
            raise _fail("local model file inventory is unsafe")
        files.append({"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    if not files:
        raise _fail("local model directory contains no files")
    return files, canonical_sha256(files)


def _require_formal_cuda() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise MissingRuntimeDependency(
            "torch with CUDA is required for formal BGE-M3 materialization"
        ) from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise _fail("formal BGE-M3 materialization requires CUDA; CPU fallback is forbidden")
    return torch


def _offline_encoder(config: EmbeddingConfig, model_dir: Path) -> Encoder:
    # Hugging Face offline guards are scoped to this call; no downloader is
    # invoked and local_files_only=True is mandatory.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise MissingRuntimeDependency(
            "FlagEmbedding and torch are required only for offline BGE-M3 materialization"
        ) from exc
    torch = _require_formal_cuda()
    device = torch.device("cuda:0")
    try:
        # Use the official BGE-M3 dense inference wrapper.  The path is the
        # already inventory-hashed local snapshot and both Hugging Face
        # offline guards are active, so this constructor cannot substitute a
        # mutable remote revision.
        model = BGEM3FlagModel(
            str(model_dir),
            use_fp16=True,
            devices=str(device),
            trust_remote_code=False,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise _fail("pinned BGE-M3 files cannot be loaded offline") from exc

    # FlagEmbedding's public ``encode`` wraps every caller batch in two local
    # tqdm bars and contains an unbounded OOM shrink loop.  Formal materialization
    # uses the same official tokenizer/model and dense head directly, with one
    # fixed, caller-bounded batch and an explicit CUDA-only failure boundary.
    model.model.to(device)
    model.model.eval()
    torch.cuda.reset_peak_memory_stats(device)

    def encode(texts: Sequence[str]) -> np.ndarray:
        values = list(texts)
        if not 1 <= len(values) <= config.batch_size:
            raise _fail("encoder batch is empty or exceeds the fixed formal batch size")
        inputs: Any = None
        output: Any = None
        try:
            inputs = model.tokenizer(
                values,
                padding=True,
                truncation=True,
                max_length=config.max_tokens,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                output = model.model(
                    inputs,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                    truncate_dim=model.truncate_dim,
                )
            dense_value = output.get("dense_vecs") if isinstance(output, Mapping) else None
            if dense_value is None:
                raise _fail("official BGE-M3 encoder omitted dense_vecs")
            dense = dense_value.detach().to(dtype=torch.float32, device="cpu").numpy()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise _fail(
                f"CUDA out of memory at fixed embedding batch {len(values)}; "
                "the batch was not silently reduced"
            ) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).casefold():
                torch.cuda.empty_cache()
                raise _fail(
                    f"CUDA out of memory at fixed embedding batch {len(values)}; "
                    "the batch was not silently reduced"
                ) from exc
            raise
        finally:
            del inputs, output
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        if np.any(norms <= 0) or not bool(np.isfinite(norms).all()):
            raise _fail("official BGE-M3 encoder returned invalid dense vectors")
        return np.ascontiguousarray(dense / norms, dtype=np.float32)

    return encode


def _validated_text_row(row: Mapping[str, Any]) -> tuple[str, str, int]:
    if set(row) != {"id", "text", "timestamp"}:
        raise _fail("text JSONL rows must contain exactly id/text/timestamp")
    identifier, text, timestamp = row["id"], row["text"], row["timestamp"]
    int64 = np.iinfo(np.int64)
    if (
        not isinstance(identifier, str)
        or not identifier
        or not isinstance(text, str)
        or not text.strip()
    ):
        raise _fail("text row has an invalid identifier or text")
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, int)
        or timestamp < int64.min
        or timestamp > int64.max
    ):
        raise _fail("text row timestamp must be a signed 64-bit integer")
    return identifier, text.strip(), timestamp


def _source_plan(source: Path, role_resolver: RoleResolver | None) -> tuple[int, str | None]:
    """Validate/count the immutable source and bind the complete role policy."""

    rows = 0
    role_digest = hashlib.sha256() if role_resolver is not None else None
    for row in read_jsonl(source):
        identifier, _text, timestamp = _validated_text_row(row)
        rows += 1
        if role_digest is not None:
            assert role_resolver is not None
            role = role_resolver(identifier, timestamp)
            if role not in EMBEDDING_ACCESS_ROLES:
                raise _fail("embedding role resolver returned an invalid role")
            role_digest.update(
                (
                    canonical_json({"id": identifier, "timestamp": timestamp, "role": role}) + "\n"
                ).encode("utf-8")
            )
    if rows == 0:
        raise _fail("text source is empty")
    return rows, role_digest.hexdigest() if role_digest is not None else None


def _default_progress(value: Mapping[str, Any]) -> None:
    print(canonical_json(dict(value)), file=sys.stderr, flush=True)


def _cuda_peak_mib() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0), 3)


def _progress_payload(
    *,
    config: EmbeddingConfig,
    phase: Literal["canonical", "roles", "complete"],
    rows_completed: int,
    total_rows: int,
    shards_completed: int,
    resumed_rows: int,
    started: float,
) -> dict[str, Any]:
    elapsed = max(0.0, time.monotonic() - started)
    session_rows = max(0, rows_completed - resumed_rows)
    rate = session_rows / elapsed if elapsed > 0 and session_rows > 0 else None
    remaining_rows = max(0, total_rows - rows_completed)
    eta = (
        0.0
        if remaining_rows == 0
        else remaining_rows / rate
        if rate is not None and rate > 0
        else None
    )
    return {
        "schemaVersion": EMBEDDING_PROGRESS_SCHEMA,
        "corpusId": config.corpus_id,
        "phase": phase,
        "rowsCompleted": rows_completed,
        "totalRows": total_rows,
        "shardsCompleted": shards_completed,
        "resumedRows": resumed_rows,
        "elapsedSeconds": round(elapsed, 3),
        "rowsPerSecond": round(rate, 3) if rate is not None else None,
        "etaSeconds": round(eta, 3) if eta is not None else None,
        "cudaPeakMiB": _cuda_peak_mib(),
    }


def _record_payload(record: ShardRecord) -> dict[str, Any]:
    return {
        "path": record.path,
        "sha256": record.sha256,
        "rows": record.rows,
        "arrays": [dict(value) for value in record.arrays],
    }


def _resume_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("logicalHash", None)
    return payload


def _write_resume_state(path: Path, value: Mapping[str, Any]) -> None:
    payload = _resume_payload(value)
    payload["logicalHash"] = canonical_sha256(payload)
    atomic_write_json(path, payload)


def _load_or_create_resume_state(
    staging: Path,
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    state_path = staging.with_name(f"{staging.name}.json")
    if not state_path.exists():
        state: dict[str, Any] = {
            "schemaVersion": EMBEDDING_RESUME_SCHEMA,
            "binding": dict(binding),
            "canonicalComplete": False,
            "rolesComplete": False,
            "canonicalShards": [],
            "roleShards": {role: [] for role in EMBEDDING_ACCESS_ROLES},
        }
        _write_resume_state(state_path, state)
        return read_json_object(state_path)
    state = read_json_object(state_path)
    logical_hash = state.get("logicalHash")
    payload = _resume_payload(state)
    if (
        set(payload)
        != {
            "schemaVersion",
            "binding",
            "canonicalComplete",
            "rolesComplete",
            "canonicalShards",
            "roleShards",
        }
        or state.get("schemaVersion") != EMBEDDING_RESUME_SCHEMA
        or not isinstance(logical_hash, str)
        or logical_hash != canonical_sha256(payload)
        or state.get("binding") != dict(binding)
        or not isinstance(state.get("canonicalComplete"), bool)
        or not isinstance(state.get("rolesComplete"), bool)
        or not isinstance(state.get("canonicalShards"), list)
        or not isinstance(state.get("roleShards"), dict)
        or set(state["roleShards"]) != set(EMBEDDING_ACCESS_ROLES)
        or any(not isinstance(state["roleShards"][role], list) for role in EMBEDDING_ACCESS_ROLES)
    ):
        raise _fail("resumable embedding state is invalid or belongs to different inputs")
    return state


def _resume_records(
    staging: Path,
    raw_records: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    allow_zero: bool,
) -> list[ShardRecord]:
    records: list[ShardRecord] = []
    for index, raw in enumerate(raw_records):
        rows = raw.get("rows")
        expected_path = f"{prefix}-{index:05d}.npz"
        if (
            set(raw) != {"path", "sha256", "rows", "arrays"}
            or raw.get("path") != expected_path
            or not _is_sha256(raw.get("sha256"))
            or isinstance(rows, bool)
            or not isinstance(rows, int)
            or not (0 if allow_zero else 1) <= rows <= EMBEDDING_ROWS_PER_SHARD
        ):
            raise _fail("resumable embedding shard descriptor is invalid")
        _validate_shard_array_inventory(raw.get("arrays"), rows)
        path = staging / expected_path
        if not path.is_file() or path.is_symlink() or file_sha256(path) != raw["sha256"]:
            raise _fail(f"resumable embedding shard hash mismatch: {expected_path}")
        arrays = _load_embedding_arrays(path)
        if (
            arrays["embedding"].shape != (rows, OUTPUT_DIM)
            or arrays["id_hash"].shape != (rows,)
            or arrays["timestamp"].shape != (rows,)
            or array_inventory(arrays) != raw["arrays"]
            or not np.allclose(
                np.linalg.norm(arrays["embedding"], axis=1),
                1.0,
                rtol=1e-4,
                atol=1e-5,
            )
        ):
            raise _fail(f"resumable embedding shard content mismatch: {expected_path}")
        records.append(
            ShardRecord(
                path=expected_path,
                sha256=str(raw["sha256"]),
                rows=rows,
                arrays=tuple(dict(value) for value in raw["arrays"]),
            )
        )
    if any(record.rows != EMBEDDING_ROWS_PER_SHARD for record in records[:-1]):
        raise _fail("only the final resumable shard may be partial")
    if allow_zero and any(record.rows == 0 for record in records):
        if len(records) != 1:
            raise _fail("an empty role shard must be the role's only shard")
    return records


def _verify_resume_source_alignment(
    source: Path, staging: Path, records: Sequence[ShardRecord]
) -> None:
    source_rows = iter(read_jsonl(source))
    for record in records:
        arrays = _load_embedding_arrays(staging / record.path)
        for index in range(record.rows):
            try:
                row = next(source_rows)
            except StopIteration as exc:
                raise _fail("resumable shards extend beyond the immutable text source") from exc
            identifier, _text, timestamp = _validated_text_row(row)
            if (
                int(arrays["id_hash"][index]) != int(portable_id_hash(identifier))
                or int(arrays["timestamp"][index]) != timestamp
            ):
                raise _fail("resumable shard rows are not aligned to the text source")


def build_bge_m3_embeddings(
    text_jsonl: str | Path,
    root: str | Path,
    *,
    config: EmbeddingConfig,
    model_dir: str | Path,
    encoder: Encoder | None = None,
    offline: bool = True,
    role_resolver: RoleResolver | None = None,
    source_manifest_hash: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Materialize resumable embeddings; an online path does not exist."""

    config.validate()
    if offline is not True:
        raise _fail("online or training-time embedding is forbidden")
    source = Path(text_jsonl).expanduser().resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise _fail("text source must be a regular JSONL artifact")
    if source_manifest_hash is not None and not _is_sha256(source_manifest_hash):
        raise _fail("source manifest hash must be SHA-256 when supplied")
    if encoder is None:
        _require_formal_cuda()
    local_model = Path(model_dir).expanduser().resolve(strict=True)
    if local_model.name != config.revision:
        raise _fail("local model directory must be the pinned Hugging Face snapshot revision")
    inventory, model_hash = _model_inventory(local_model)
    source_hash = file_sha256(source)
    source_rows, role_assignment_hash = _source_plan(source, role_resolver)
    if encoder is None:
        try:
            producer_version = importlib.metadata.version("FlagEmbedding")
        except importlib.metadata.PackageNotFoundError as exc:
            raise MissingRuntimeDependency(
                "FlagEmbedding==1.4.0 is required for formal BGE-M3 materialization"
            ) from exc
        if producer_version != "1.4.0":
            raise _fail("formal BGE-M3 producer must be exactly FlagEmbedding 1.4.0")
        producer = {
            "implementation": "FlagEmbedding.BGEM3FlagModel",
            "distribution": "FlagEmbedding",
            "version": producer_version,
            "formalEligible": True,
        }
    else:
        producer = {
            "implementation": "injected-test-encoder",
            "distribution": None,
            "version": None,
            "formalEligible": False,
        }
    output = RuntimeLayout.from_root(root).embeddings / f"{config.corpus_id}-bge-m3-v1"
    output.parent.mkdir(parents=True, exist_ok=True)
    binding = {
        "sourceName": source.name,
        "sourceSha256": source_hash,
        "sourceRows": source_rows,
        "sourceManifestHash": source_manifest_hash,
        "configHash": config.identity,
        "modelInventoryHash": model_hash,
        "roleAssignmentHash": role_assignment_hash,
        "formalProducer": encoder is None,
        "rowsPerShard": EMBEDDING_ROWS_PER_SHARD,
        "resumeSchemaVersion": EMBEDDING_RESUME_SCHEMA,
    }
    binding_hash = canonical_sha256(binding)
    staging = output.parent / f".{output.name}.{binding_hash[:16]}.resume"
    state_path = staging.with_name(f"{staging.name}.json")
    lock_path = output.parent / f".{output.name}.lock"
    selected_progress = progress_callback or (_default_progress if encoder is None else None)
    started = time.monotonic()

    with exclusive_file_lock(lock_path):
        if output.exists():
            existing = verify_embedding_artifact(output)
            _verify_requested_identity(
                existing,
                source_name=source.name,
                source_hash=source_hash,
                config_hash=config.identity,
                model_hash=model_hash,
                formal_eligible=encoder is None,
                physical_access_required=role_resolver is not None,
            )
            return existing
        staging.mkdir(exist_ok=True)
        state = _load_or_create_resume_state(staging, binding=binding)
        canonical_records = _resume_records(
            staging,
            state["canonicalShards"],
            prefix="embeddings",
            allow_zero=False,
        )
        _verify_resume_source_alignment(source, staging, canonical_records)
        resumed_canonical_rows = sum(record.rows for record in canonical_records)
        if resumed_canonical_rows > source_rows:
            raise _fail("resumable canonical rows exceed the immutable source")
        if state["canonicalComplete"] and resumed_canonical_rows != source_rows:
            raise _fail("resumable canonical completion marker is inconsistent")

        def save_state() -> None:
            _write_resume_state(state_path, state)

        def canonical_progress(records: Sequence[ShardRecord], rows: int) -> None:
            state["canonicalShards"] = [_record_payload(record) for record in records]
            save_state()
            if selected_progress is not None:
                selected_progress(
                    _progress_payload(
                        config=config,
                        phase="canonical",
                        rows_completed=rows,
                        total_rows=source_rows,
                        shards_completed=len(records),
                        resumed_rows=resumed_canonical_rows,
                        started=started,
                    )
                )

        if not state["canonicalComplete"]:
            encode = encoder or _offline_encoder(config, local_model)
            writer = NumericShardWriter(
                staging, prefix="embeddings", rows_per_shard=EMBEDDING_ROWS_PER_SHARD
            )
            writer._index = len(canonical_records)
            canonical_records, encoded_rows = _stream_embedding_shards(
                source,
                config=config,
                encoder=encode,
                writer=writer,
                initial_shards=canonical_records,
                on_shard=canonical_progress,
            )
            if encoded_rows != source_rows:
                raise _fail("embedding row count differs from the immutable source plan")
            state["canonicalComplete"] = True
            state["canonicalShards"] = [_record_payload(record) for record in canonical_records]
            save_state()

        role_records: dict[str, list[ShardRecord]] = {
            role: _resume_records(
                staging,
                state["roleShards"][role],
                prefix=f"role-{role}-embeddings",
                allow_zero=True,
            )
            for role in EMBEDDING_ACCESS_ROLES
        }
        resumed_role_rows = sum(
            record.rows for records in role_records.values() for record in records
        )
        role_started = time.monotonic()

        def role_progress(records: Mapping[str, Sequence[ShardRecord]], rows: int) -> None:
            state["roleShards"] = {
                role: [_record_payload(record) for record in records[role]]
                for role in EMBEDDING_ACCESS_ROLES
            }
            save_state()
            if selected_progress is not None:
                selected_progress(
                    _progress_payload(
                        config=config,
                        phase="roles",
                        rows_completed=rows,
                        total_rows=source_rows,
                        shards_completed=sum(len(value) for value in records.values()),
                        resumed_rows=resumed_role_rows,
                        started=role_started,
                    )
                )

        if role_resolver is not None:
            if not state["rolesComplete"]:
                role_records = {
                    role: list(records)
                    for role, records in _write_embedding_role_shards(
                        staging,
                        shards=canonical_records,
                        source=source,
                        resolver=role_resolver,
                        initial_records=role_records,
                        on_shard=role_progress,
                    ).items()
                }
                state["rolesComplete"] = True
                state["roleShards"] = {
                    role: [_record_payload(record) for record in role_records[role]]
                    for role in EMBEDDING_ACCESS_ROLES
                }
                save_state()
        else:
            if any(role_records.values()):
                raise _fail("resumable role shards exist without a role policy")
            state["rolesComplete"] = True
            save_state()

        # Inputs are immutable provenance. Detect a source/model mutation that
        # raced the streaming pass instead of publishing a mixed artifact.
        if file_sha256(source) != source_hash:
            raise _fail("text source changed during embedding materialization")
        _, final_model_hash = _model_inventory(local_model)
        if final_model_hash != model_hash:
            raise _fail("pinned model files changed during embedding materialization")
        manifest = build_manifest(
            schema_version="gfm.text-embeddings/1.0",
            corpus_id=config.corpus_id,
            license_id=MODEL_LICENSE,
            source={
                "textPathName": source.name,
                "textSha256": source_hash,
                "modelId": config.model_id,
                "modelRevision": config.revision,
                "modelInventoryHash": model_hash,
            },
            shards=(
                *canonical_records,
                *(
                    record
                    for role in EMBEDDING_ACCESS_ROLES
                    for record in role_records.get(role, ())
                ),
            ),
            splits={
                "alignedBy": "id_hash-and-timestamp",
                "physicalAccess": (
                    {
                        "schemaVersion": EMBEDDING_ACCESS_SCHEMA,
                        "roles": list(EMBEDDING_ACCESS_ROLES),
                        "roleShards": {
                            role: [record.path for record in role_records[role]]
                            for role in EMBEDDING_ACCESS_ROLES
                        },
                    }
                    if role_resolver is not None
                    else None
                ),
            },
            privacy={
                "rawTextCopied": False,
                "onlineEncoding": False,
                "trainingTimeEncoding": False,
            },
            extra={
                "artifactType": "frozen_text_embeddings",
                "rows": source_rows,
                "rowsPerShard": EMBEDDING_ROWS_PER_SHARD,
                "dimension": config.output_dim,
                "normalized": True,
                "idHashAlgorithm": PORTABLE_ID_HASH_ALGORITHM,
                "configHash": config.identity,
                "modelFiles": inventory,
                "producer": producer,
            },
        )
        atomic_write_json(staging / "manifest.json", manifest)
        verify_embedding_artifact(staging)

        # A concurrent publisher wins. Existing artifacts are never replaced;
        # they are only revalidated and identity-checked.
        if output.exists():
            existing = verify_embedding_artifact(output)
            _verify_requested_identity(
                existing,
                source_name=source.name,
                source_hash=source_hash,
                config_hash=config.identity,
                model_hash=model_hash,
                formal_eligible=encoder is None,
                physical_access_required=role_resolver is not None,
            )
            return existing
        os.replace(staging, output)
        state_path.unlink(missing_ok=True)
        if selected_progress is not None:
            selected_progress(
                _progress_payload(
                    config=config,
                    phase="complete",
                    rows_completed=source_rows,
                    total_rows=source_rows,
                    shards_completed=len(canonical_records)
                    + sum(len(value) for value in role_records.values()),
                    resumed_rows=0,
                    started=started,
                )
            )
        # The exact directory was fully verified immediately before the atomic
        # rename. Existing artifacts are always re-read on the idempotent path.
        return manifest


def _verify_requested_identity(
    manifest: dict[str, Any],
    *,
    source_name: str,
    source_hash: str,
    config_hash: str,
    model_hash: str,
    formal_eligible: bool,
    physical_access_required: bool,
) -> None:
    source = manifest.get("source")
    if not isinstance(source, dict) or (
        source.get("textPathName") != source_name
        or source.get("textSha256") != source_hash
        or source.get("modelInventoryHash") != model_hash
        or manifest.get("configHash") != config_hash
        or manifest.get("producer", {}).get("formalEligible") is not formal_eligible
        or (physical_access_required and manifest.get("splits", {}).get("physicalAccess") is None)
    ):
        raise _fail("existing immutable artifact does not match the requested inputs")


def _stream_embedding_shards(
    source: Path,
    *,
    config: EmbeddingConfig,
    encoder: Encoder,
    writer: NumericShardWriter,
    initial_shards: Sequence[ShardRecord] = (),
    on_shard: Callable[[Sequence[ShardRecord], int], None] | None = None,
) -> tuple[list[ShardRecord], int]:
    """Encode one bounded text batch at a time and flush bounded NPZ shards."""

    embedding_buffer = np.empty((EMBEDDING_ROWS_PER_SHARD, config.output_dim), dtype=np.float32)
    id_buffer = np.empty(EMBEDDING_ROWS_PER_SHARD, dtype=np.uint64)
    timestamp_buffer = np.empty(EMBEDDING_ROWS_PER_SHARD, dtype=np.int64)
    shard_rows = 0
    shards = list(initial_shards)
    completed_rows = sum(record.rows for record in shards)
    total_rows = completed_rows
    batch_texts: list[str] = []
    batch_hashes: list[np.uint64] = []
    batch_timestamps: list[int] = []

    def flush_shard() -> None:
        nonlocal shard_rows
        if shard_rows == 0:
            return
        shards.append(
            writer.write(
                {
                    "embedding": embedding_buffer[:shard_rows],
                    "id_hash": id_buffer[:shard_rows],
                    "timestamp": timestamp_buffer[:shard_rows],
                }
            )
        )
        shard_rows = 0
        if on_shard is not None:
            on_shard(tuple(shards), total_rows)

    def encode_batch() -> None:
        nonlocal shard_rows, total_rows
        if not batch_texts:
            return
        vectors = np.asarray(encoder(batch_texts))
        expected_shape = (len(batch_texts), config.output_dim)
        if vectors.shape != expected_shape:
            raise _fail("encoder returned the wrong shape")
        if vectors.dtype != np.float32 or not bool(np.isfinite(vectors).all()):
            raise _fail("encoder output must be finite float32")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
            raise _fail("encoder output is not L2-normalized")
        vectors = np.ascontiguousarray(vectors)
        offset = 0
        while offset < len(batch_texts):
            available = EMBEDDING_ROWS_PER_SHARD - shard_rows
            count = min(available, len(batch_texts) - offset)
            upper = offset + count
            destination = slice(shard_rows, shard_rows + count)
            embedding_buffer[destination] = vectors[offset:upper]
            id_buffer[destination] = batch_hashes[offset:upper]
            timestamp_buffer[destination] = batch_timestamps[offset:upper]
            shard_rows += count
            total_rows += count
            offset = upper
            if shard_rows == EMBEDDING_ROWS_PER_SHARD:
                flush_shard()
        batch_texts.clear()
        batch_hashes.clear()
        batch_timestamps.clear()

    for source_index, row in enumerate(read_jsonl(source)):
        identifier, text, timestamp = _validated_text_row(row)
        if source_index < completed_rows:
            continue
        if shards and shards[-1].rows != EMBEDDING_ROWS_PER_SHARD:
            raise _fail("a partial resumed shard cannot precede unencoded source rows")
        identifier_hash = portable_id_hash(identifier)
        batch_texts.append(text)
        batch_hashes.append(identifier_hash)
        batch_timestamps.append(timestamp)
        if len(batch_texts) == config.batch_size:
            encode_batch()
    encode_batch()
    flush_shard()
    return shards, total_rows


def _write_embedding_role_shards(
    output: Path,
    *,
    shards: Sequence[ShardRecord],
    source: Path,
    resolver: RoleResolver,
    initial_records: Mapping[str, Sequence[ShardRecord]] | None = None,
    on_shard: Callable[[Mapping[str, Sequence[ShardRecord]], int], None] | None = None,
) -> dict[str, tuple[ShardRecord, ...]]:
    """Partition vectors with one-source-shard memory and resumable role shards."""

    existing = initial_records or {}
    writers = {
        role: NumericShardWriter(
            output,
            prefix=f"role-{role}-embeddings",
            rows_per_shard=EMBEDDING_ROWS_PER_SHARD,
        )
        for role in EMBEDDING_ACCESS_ROLES
    }
    records: dict[str, list[ShardRecord]] = {
        role: list(existing.get(role, ())) for role in EMBEDDING_ACCESS_ROLES
    }
    for role in EMBEDDING_ACCESS_ROLES:
        writers[role]._index = len(records[role])
    buffers: dict[str, dict[str, list[np.ndarray]]] = {
        role: {"embedding": [], "id_hash": [], "timestamp": []} for role in EMBEDDING_ACCESS_ROLES
    }
    counts = {role: 0 for role in EMBEDDING_ACCESS_ROLES}
    skip_remaining = {
        role: sum(record.rows for record in records[role]) for role in EMBEDDING_ACCESS_ROLES
    }
    observed = {role: 0 for role in EMBEDDING_ACCESS_ROLES}

    def flush(role: str) -> None:
        if counts[role] == 0:
            return
        records[role].append(
            writers[role].write(
                {name: np.concatenate(values, axis=0) for name, values in buffers[role].items()}
            )
        )
        buffers[role] = {"embedding": [], "id_hash": [], "timestamp": []}
        counts[role] = 0
        if on_shard is not None:
            durable_rows = sum(record.rows for values in records.values() for record in values)
            on_shard(records, durable_rows)

    source_rows = iter(read_jsonl(source))
    consumed = 0
    for record in shards:
        arrays = _load_embedding_arrays(output / record.path)
        rows = int(record.rows)
        role_values: list[str] = []
        for _ in range(rows):
            try:
                source_row = next(source_rows)
            except StopIteration as exc:
                raise _fail("embedding role source rows are misaligned") from exc
            identifier, _text, timestamp = _validated_text_row(source_row)
            role = resolver(identifier, timestamp)
            if role not in EMBEDDING_ACCESS_ROLES:
                raise _fail("embedding role resolver returned an invalid role")
            role_values.append(role)
            consumed += 1
        roles = np.asarray(role_values, dtype="U10")
        for access_role in EMBEDDING_ACCESS_ROLES:
            indices = np.flatnonzero(roles == access_role)
            observed[access_role] += int(indices.size)
            skipped = min(skip_remaining[access_role], int(indices.size))
            skip_remaining[access_role] -= skipped
            indices = indices[skipped:]
            if (
                indices.size
                and records[access_role]
                and (records[access_role][-1].rows != EMBEDDING_ROWS_PER_SHARD)
            ):
                raise _fail("a partial resumed role shard precedes unpartitioned rows")
            cursor = 0
            while cursor < indices.size:
                available = EMBEDDING_ROWS_PER_SHARD - counts[access_role]
                selection = indices[cursor : cursor + available]
                for name in buffers[access_role]:
                    buffers[access_role][name].append(np.ascontiguousarray(arrays[name][selection]))
                counts[access_role] += selection.size
                cursor += selection.size
                if counts[access_role] == EMBEDDING_ROWS_PER_SHARD:
                    flush(access_role)
    try:
        next(source_rows)
    except StopIteration:
        pass
    else:
        raise _fail("embedding role partition did not consume the source")
    if consumed != sum(record.rows for record in shards):
        raise _fail("embedding role partition row count differs from canonical shards")
    if any(skip_remaining.values()):
        raise _fail("resumable role shards extend beyond the immutable source")
    for access_role in EMBEDDING_ACCESS_ROLES:
        flush(access_role)
        if not records[access_role]:
            # Empty role artifacts are explicit, immutable proof of absence.
            records[access_role].append(
                writers[access_role].write(
                    {
                        "embedding": np.empty((0, OUTPUT_DIM), dtype=np.float32),
                        "id_hash": np.empty(0, dtype=np.uint64),
                        "timestamp": np.empty(0, dtype=np.int64),
                    }
                )
            )
            if on_shard is not None:
                durable_rows = sum(record.rows for values in records.values() for record in values)
                on_shard(records, durable_rows)
        if sum(record.rows for record in records[access_role]) != observed[access_role]:
            raise _fail("resumable role shard rows differ from the role assignment")
    return {role: tuple(values) for role, values in records.items()}


@contextmanager
def _embedding_audit_database(parent: Path) -> Iterator[sqlite3.Connection]:
    """Use disk-backed uniqueness/partition checks instead of per-row Python objects."""

    path = parent / f".embedding-audit-{uuid.uuid4().hex}.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-32768")
        for table in ("canonical_rows", "role_rows"):
            connection.execute(
                f"""
                CREATE TABLE {table} (
                    id_hash BLOB PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    embedding_sha256 BLOB NOT NULL
                ) WITHOUT ROWID
                """
            )
        yield connection
    finally:
        connection.close()
        path.unlink(missing_ok=True)


def _insert_audit_rows(
    connection: sqlite3.Connection,
    table: Literal["canonical_rows", "role_rows"],
    arrays: Mapping[str, np.ndarray],
) -> None:
    payload = [
        (
            int(raw_hash).to_bytes(8, "big", signed=False),
            int(arrays["timestamp"][index]),
            hashlib.sha256(
                np.ascontiguousarray(arrays["embedding"][index]).tobytes(order="C")
            ).digest(),
        )
        for index, raw_hash in enumerate(arrays["id_hash"])
    ]
    try:
        connection.executemany(
            f"INSERT INTO {table}(id_hash,timestamp,embedding_sha256) VALUES (?,?,?)",
            payload,
        )
        connection.commit()
    except sqlite3.IntegrityError as exc:
        if table == "canonical_rows":
            raise _fail(
                "text source contains a duplicate identifier or uint64 hash collision"
            ) from exc
        raise _fail("embedding role shards contain duplicate identifier hashes") from exc


def verify_embedding_artifact(output_dir: str | Path) -> dict[str, Any]:
    try:
        output = Path(output_dir).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _fail("embedding artifact directory is absent or unsafe") from exc
    if not output.is_dir() or output.is_symlink():
        raise _fail("embedding artifact path must be a regular directory")
    manifest = read_json_object(output / "manifest.json")
    rows, shards = _validate_embedding_manifest_structure(manifest)
    # Only call the generic loader after exact dtypes/shapes/capacities have
    # been established. This prevents a resealed manifest from inducing an
    # unbounded allocation during verification.
    verify_manifest(output, manifest)
    try:
        with _embedding_audit_database(output.parent) as audit:
            counted_rows = 0
            for index, record in enumerate(shards):
                if not isinstance(record, dict):
                    raise _fail("embedding shard descriptor is invalid")
                expected_path = f"embeddings-{index:05d}.npz"
                shard_rows = record.get("rows")
                if (
                    record.get("path") != expected_path
                    or isinstance(shard_rows, bool)
                    or not isinstance(shard_rows, int)
                    or shard_rows < 1
                    or shard_rows > EMBEDDING_ROWS_PER_SHARD
                    or (index < len(shards) - 1 and shard_rows != EMBEDDING_ROWS_PER_SHARD)
                ):
                    raise _fail("embedding shard ordering or row capacity is invalid")
                arrays = _load_embedding_arrays(output / expected_path)
                if (
                    arrays["embedding"].shape != (shard_rows, OUTPUT_DIM)
                    or arrays["id_hash"].shape != (shard_rows,)
                    or arrays["timestamp"].shape != (shard_rows,)
                ):
                    raise _fail("embedding shard array shape is invalid")
                norms = np.linalg.norm(arrays["embedding"], axis=1)
                if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
                    raise _fail("embedding shard contains non-normalized vectors")
                _insert_audit_rows(audit, "canonical_rows", arrays)
                counted_rows += shard_rows
            canonical_count = int(
                audit.execute("SELECT COUNT(*) FROM canonical_rows").fetchone()[0]
            )
            if counted_rows != rows or canonical_count != rows:
                raise _fail("embedding manifest total row count is inconsistent")

            access = manifest["splits"].get("physicalAccess")
            if access is not None:
                records_by_path = {
                    str(record["path"]): record
                    for record in manifest["shards"]
                    if isinstance(record, dict)
                }
                role_rows = 0
                for role in EMBEDDING_ACCESS_ROLES:
                    for path in access["roleShards"][role]:
                        arrays = _load_embedding_arrays(output / path)
                        record = records_by_path[path]
                        if int(record["rows"]) != int(arrays["id_hash"].size):
                            raise _fail("embedding role shard row count differs")
                        _insert_audit_rows(audit, "role_rows", arrays)
                        role_rows += int(arrays["id_hash"].size)
                role_count = int(audit.execute("SELECT COUNT(*) FROM role_rows").fetchone()[0])
                mismatch = audit.execute(
                    """
                    SELECT 1
                    FROM canonical_rows AS canonical
                    LEFT JOIN role_rows AS role USING(id_hash)
                    WHERE role.id_hash IS NULL
                       OR role.timestamp != canonical.timestamp
                       OR role.embedding_sha256 != canonical.embedding_sha256
                    LIMIT 1
                    """
                ).fetchone()
                if role_rows != rows or role_count != rows or mismatch is not None:
                    raise _fail("embedding role shards are not an exact partition")
    except sqlite3.Error as exc:
        raise _fail("disk-backed embedding verification failed") from exc
    return manifest


def _validate_embedding_manifest_structure(
    manifest: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    expected_keys = {
        "schemaVersion",
        "corpusId",
        "licenseId",
        "source",
        "shards",
        "splits",
        "privacy",
        "artifactType",
        "rows",
        "rowsPerShard",
        "dimension",
        "normalized",
        "idHashAlgorithm",
        "configHash",
        "modelFiles",
        "producer",
        "logicalHash",
        "createdAt",
    }
    source = manifest.get("source")
    privacy = manifest.get("privacy")
    model_files = manifest.get("modelFiles")
    producer = manifest.get("producer")
    corpus_id = manifest.get("corpusId")
    if set(manifest) != expected_keys or (
        manifest.get("schemaVersion") != "gfm.text-embeddings/1.0"
        or manifest.get("licenseId") != MODEL_LICENSE
        or not isinstance(corpus_id, str)
        or not corpus_id
        or any(char in corpus_id for char in "/\\:")
        or not isinstance(source, dict)
        or set(source)
        != {
            "textPathName",
            "textSha256",
            "modelId",
            "modelRevision",
            "modelInventoryHash",
        }
        or source.get("modelId") != MODEL_ID
        or source.get("modelRevision") != MODEL_REVISION
        or manifest.get("dimension") != OUTPUT_DIM
        or manifest.get("rowsPerShard") != EMBEDDING_ROWS_PER_SHARD
        or manifest.get("normalized") is not True
        or manifest.get("artifactType") != "frozen_text_embeddings"
        or manifest.get("idHashAlgorithm") != PORTABLE_ID_HASH_ALGORITHM
        or not isinstance(manifest.get("splits"), dict)
        or manifest["splits"].get("alignedBy") != "id_hash-and-timestamp"
        or set(manifest["splits"]) != {"alignedBy", "physicalAccess"}
    ):
        raise _fail("embedding manifest violates the pinned model contract")
    if producer not in (
        {
            "implementation": "FlagEmbedding.BGEM3FlagModel",
            "distribution": "FlagEmbedding",
            "version": "1.4.0",
            "formalEligible": True,
        },
        {
            "implementation": "injected-test-encoder",
            "distribution": None,
            "version": None,
            "formalEligible": False,
        },
    ):
        raise _fail("embedding producer identity is invalid")
    if privacy != {
        "rawTextCopied": False,
        "onlineEncoding": False,
        "trainingTimeEncoding": False,
    }:
        raise _fail("embedding manifest does not prohibit online/training-time encoding")
    text_name = source["textPathName"]
    if (
        not isinstance(text_name, str)
        or not text_name
        or Path(text_name).name != text_name
        or any(char in text_name for char in "/\\:")
        or not _is_sha256(source["textSha256"])
        or not _is_sha256(source["modelInventoryHash"])
        or not _is_sha256(manifest["configHash"])
    ):
        raise _fail("embedding manifest source identity is invalid")
    if not isinstance(model_files, list) or not model_files:
        raise _fail("embedding manifest model inventory is invalid")
    inventory_paths: list[str] = []
    for item in model_files:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "bytes"}
            or not isinstance(item["path"], str)
            or not item["path"]
            or not _is_sha256(item["sha256"])
            or isinstance(item["bytes"], bool)
            or not isinstance(item["bytes"], int)
            or item["bytes"] < 0
            or item["bytes"] > 8 * 1024 * 1024 * 1024
        ):
            raise _fail("embedding manifest model inventory is invalid")
        try:
            safe_relative_path(item["path"])
        except ContractViolation as exc:
            raise _fail("embedding manifest model inventory is invalid") from exc
        inventory_paths.append(item["path"])
    if inventory_paths != sorted(set(inventory_paths)):
        raise _fail("embedding manifest model inventory is invalid")
    if source["modelInventoryHash"] != canonical_sha256(model_files):
        raise _fail("embedding manifest model inventory is invalid")
    rows = manifest.get("rows")
    raw_shards = manifest.get("shards")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows < 1
        or not isinstance(raw_shards, list)
        or not raw_shards
    ):
        raise _fail("embedding manifest row inventory is invalid")
    shards: list[dict[str, Any]] = []
    canonical_records = [
        record
        for record in raw_shards
        if isinstance(record, dict) and str(record.get("path", "")).startswith("embeddings-")
    ]
    access_records = [record for record in raw_shards if record not in canonical_records]
    counted_rows = 0
    for index, raw_record in enumerate(canonical_records):
        if not isinstance(raw_record, dict):
            raise _fail("embedding shard descriptor is invalid")
        shard_rows = raw_record.get("rows")
        if (
            set(raw_record) != {"path", "sha256", "rows", "arrays"}
            or raw_record.get("path") != f"embeddings-{index:05d}.npz"
            or not _is_sha256(raw_record.get("sha256"))
            or isinstance(shard_rows, bool)
            or not isinstance(shard_rows, int)
            or shard_rows < 1
            or shard_rows > EMBEDDING_ROWS_PER_SHARD
            or (index < len(canonical_records) - 1 and shard_rows != EMBEDDING_ROWS_PER_SHARD)
        ):
            raise _fail("embedding shard ordering or row capacity is invalid")
        _validate_shard_array_inventory(raw_record.get("arrays"), shard_rows)
        counted_rows += shard_rows
        shards.append(raw_record)
    if counted_rows != rows:
        raise _fail("embedding manifest total row count is inconsistent")
    access = manifest["splits"].get("physicalAccess")
    if access is None:
        if access_records:
            raise _fail("embedding access shards lack a physical contract")
    else:
        if (
            not isinstance(access, dict)
            or access.get("schemaVersion") != EMBEDDING_ACCESS_SCHEMA
            or access.get("roles") != list(EMBEDDING_ACCESS_ROLES)
            or not isinstance(access.get("roleShards"), dict)
            or set(access["roleShards"]) != set(EMBEDDING_ACCESS_ROLES)
        ):
            raise _fail("embedding physical role contract is invalid")
        declared = []
        for role in EMBEDDING_ACCESS_ROLES:
            paths = access["roleShards"][role]
            if not isinstance(paths, list) or not paths:
                raise _fail("embedding role shard inventory is invalid")
            for index, path in enumerate(paths):
                if path != f"role-{role}-embeddings-{index:05d}.npz":
                    raise _fail("embedding role shard ordering is invalid")
                declared.append(path)
        access_row_count = 0
        access_records_by_path: dict[str, dict[str, Any]] = {}
        if len(access_records) != len(declared):
            raise _fail("embedding role shard declaration differs from manifest order")
        for expected_path, record in zip(declared, access_records, strict=True):
            if not isinstance(record, dict):
                raise _fail("embedding role shard descriptor is invalid")
            shard_rows = record.get("rows")
            if (
                set(record) != {"path", "sha256", "rows", "arrays"}
                or record.get("path") != expected_path
                or not _is_sha256(record.get("sha256"))
                or isinstance(shard_rows, bool)
                or not isinstance(shard_rows, int)
                or not 0 <= shard_rows <= EMBEDDING_ROWS_PER_SHARD
            ):
                raise _fail("embedding role shard descriptor is invalid")
            _validate_shard_array_inventory(record.get("arrays"), shard_rows)
            access_records_by_path[expected_path] = record
            access_row_count += shard_rows
        for role in EMBEDDING_ACCESS_ROLES:
            role_paths = access["roleShards"][role]
            role_records = [access_records_by_path[path] for path in role_paths]
            if any(
                int(record["rows"]) != EMBEDDING_ROWS_PER_SHARD
                for record in role_records[:-1]
            ):
                raise _fail("only the final embedding role shard may be partial")
            if any(int(record["rows"]) == 0 for record in role_records) and len(
                role_records
            ) != 1:
                raise _fail("an empty embedding role must have exactly one shard")
        if access_row_count != rows:
            raise _fail("embedding role shard row inventory is inconsistent")
    return rows, shards


def _validate_shard_array_inventory(value: Any, rows: int) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise _fail("embedding shard array inventory is invalid")
    items = {item.get("name"): item for item in value if isinstance(item, dict)}
    if set(items) != {"embedding", "id_hash", "timestamp"} or len(items) != len(value):
        raise _fail("embedding shard array inventory is invalid")
    expected = {
        "embedding": (np.dtype(np.float32).str, [rows, OUTPUT_DIM], rows * OUTPUT_DIM * 4),
        "id_hash": (np.dtype(np.uint64).str, [rows], rows * 8),
        "timestamp": (np.dtype(np.int64).str, [rows], rows * 8),
    }
    for name, (dtype, shape, byte_length) in expected.items():
        item = items[name]
        if (
            set(item) != {"name", "dtype", "shape", "sha256", "byteLength"}
            or item.get("dtype") != dtype
            or item.get("shape") != shape
            or item.get("byteLength") != byte_length
            or not _is_sha256(item.get("sha256"))
        ):
            raise _fail("embedding shard array inventory is invalid")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def iter_embedding_shards(output_dir: str | Path) -> Iterator[EmbeddingShard]:
    """Yield one verified shard at a time; memory is bounded by one shard."""

    yield from open_embedding_artifact(output_dir).iter_shards()


def open_embedding_artifact(output_dir: str | Path) -> VerifiedEmbeddingArtifact:
    """Verify once and return a handle suitable for a small shard LRU cache."""

    try:
        output = Path(output_dir).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _fail("embedding artifact directory is absent or unsafe") from exc
    manifest = verify_embedding_artifact(output)
    # Role shards are physical copies for access isolation; the compatibility
    # handle must continue to expose only the canonical embedding matrix.
    records = [
        record
        for record in manifest["shards"]
        if str(record.get("path", "")).startswith("embeddings-")
    ]
    if sum(int(record["rows"]) for record in records) != int(manifest["rows"]):
        raise _fail("canonical embedding handle row inventory is inconsistent")
    return VerifiedEmbeddingArtifact(
        directory=output,
        logical_hash=str(manifest["logicalHash"]),
        rows=int(manifest["rows"]),
        shard_hashes=tuple(str(record["sha256"]) for record in records),
        shard_rows=tuple(int(record["rows"]) for record in records),
    )


def open_embedding_artifact_view(
    output_dir: str | Path, *, maximum_role: EmbeddingRole
) -> VerifiedEmbeddingArtifact:
    """Open a cumulative embedding view without touching later-role files.

    CorpusReady must already have performed full semantic verification.  This
    formal-worker entry point checks only the signed manifest and the selected
    role records; a corrupt test shard therefore cannot be observed by a
    train/validation process.
    """

    if maximum_role not in EMBEDDING_ACCESS_ROLES:
        raise _fail("embedding maximum role is invalid")
    try:
        output = Path(output_dir).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _fail("embedding artifact directory is absent or unsafe") from exc
    manifest = read_json_object(output / "manifest.json")
    logical_hash = manifest.get("logicalHash")
    payload = {
        key: value for key, value in manifest.items() if key not in {"logicalHash", "createdAt"}
    }
    if (
        manifest.get("schemaVersion") != "gfm.text-embeddings/1.0"
        or not isinstance(logical_hash, str)
        or logical_hash != canonical_sha256(payload)
    ):
        raise _fail("embedding role-view manifest identity is invalid")
    # Validate the complete signed JSON contract before selecting a physical
    # role.  This checks every future record's schema, fixed name, dtype,
    # shape, capacity and row accounting without opening any future shard.
    # Physical bytes remain isolated below and are read only for authorised
    # cumulative roles.
    _validate_embedding_manifest_structure(manifest)
    splits = manifest.get("splits")
    access = splits.get("physicalAccess") if isinstance(splits, dict) else None
    if (
        not isinstance(access, dict)
        or access.get("schemaVersion") != EMBEDDING_ACCESS_SCHEMA
        or access.get("roles") != list(EMBEDDING_ACCESS_ROLES)
        or not isinstance(access.get("roleShards"), dict)
    ):
        raise _fail("embedding artifact has no physical role-view contract")
    role_index = EMBEDDING_ACCESS_ROLES.index(maximum_role)
    paths = [
        path
        for role in EMBEDDING_ACCESS_ROLES[: role_index + 1]
        for path in access["roleShards"].get(role, [])
    ]
    records = {
        str(record.get("path")): record
        for record in manifest.get("shards", [])
        if isinstance(record, dict)
    }
    hashes: list[str] = []
    rows: list[int] = []
    for path in paths:
        record = records.get(path)
        if record is None or not isinstance(record.get("rows"), int):
            raise _fail("selected embedding role shard is undeclared")
        try:
            artifact = resolve_within(output, path)
        except ContractViolation as exc:
            raise _fail("selected embedding role shard is absent or unsafe") from exc
        if file_sha256(artifact) != record.get("sha256"):
            raise _fail(f"artifact hash mismatch: {path}")
        hashes.append(str(record["sha256"]))
        rows.append(int(record["rows"]))
    return VerifiedEmbeddingArtifact(
        directory=output,
        logical_hash=str(logical_hash),
        rows=sum(rows),
        shard_hashes=tuple(hashes),
        shard_rows=tuple(rows),
        shard_paths=tuple(paths),
    )


def load_embedding_shard(output_dir: str | Path, index: int) -> EmbeddingShard:
    """One-shot verified shard load; repeated callers should reuse an opened handle."""

    return open_embedding_artifact(output_dir).load_shard(index)


def lookup_embedding_rows(
    output_dir: str | Path,
    id_hashes: Sequence[int | np.uint64],
) -> dict[int, tuple[np.ndarray, int]]:
    """Scan verified shards and copy only requested rows into the result."""

    wanted: set[int] = set()
    uint64 = np.iinfo(np.uint64)
    for raw_value in id_hashes:
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, np.integer)):
            raise _fail("embedding lookup identifiers must be uint64 integers")
        value = int(raw_value)
        if value < uint64.min or value > uint64.max or value in wanted:
            raise _fail("embedding lookup identifiers are duplicate or outside uint64")
        wanted.add(value)
    if not wanted:
        return {}
    found: dict[int, tuple[np.ndarray, int]] = {}
    for shard in open_embedding_artifact(output_dir).iter_shards():
        for index, raw_value in enumerate(shard.id_hash):
            value = int(raw_value)
            if value in wanted:
                vector = np.array(shard.embedding[index], dtype=np.float32, copy=True)
                vector.flags.writeable = False
                found[value] = (vector, int(shard.timestamp[index]))
        if len(found) == len(wanted):
            break
    missing = wanted - found.keys()
    if missing:
        raise _fail(f"embedding lookup is missing {len(missing)} requested identifier(s)")
    return found


def _load_embedding_arrays(path: Path) -> dict[str, np.ndarray]:
    return load_npz_safe(
        path,
        expected={
            "embedding": (np.dtype(np.float32).str, 2),
            "id_hash": (np.dtype(np.uint64).str, 1),
            "timestamp": (np.dtype(np.int64).str, 1),
        },
    )
