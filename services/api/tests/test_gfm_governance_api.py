from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import threading
import zipfile
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.requests import Request as StarletteRequest

from app.config import Settings
from app import gfm_client as gfm_client_module
from app.gfm_client import GfmProxyError, GfmServiceClient
from app.gfm_hashing import canonical_sha256
from app.gfm_governance_artifacts import GovernanceArtifactInbox, inspect_governance_bundle
from app.gfm_governance_schemas import (
    GOVERNANCE_SCHEMA_VERSION,
    CaseCreateRequest,
    CaseItemRequest,
    CaseTransitionRequest,
    ReviewEventRequest,
)
from app.gfm_governance_store import GovernanceStore
from app.gfm_governance_uploads import (
    GOVERNANCE_MULTIPART_OVERHEAD_MAX_BYTES,
    GOVERNANCE_UPLOAD_CHUNK_BYTES,
)
from app.main import create_app


def _bundle(
    *,
    relations: str = "a,b,coRT,1\n",
    feature_ids: tuple[str | bytes, ...] = ("a", "b"),
    manifest_overrides: dict[str, object] | None = None,
    archive_comment: bytes = b"",
) -> bytes:
    nodes = b"node_id,display_name\na,Account A\nb,Account B\n"
    relation_bytes = (
        "source,target,modality,weight\n" + relations
    ).encode("utf-8")
    feature_stream = io.BytesIO()
    np.savez_compressed(
        feature_stream,
        node_ids=np.asarray(feature_ids),
        text_features=np.arange(2 * 768, dtype=np.float32).reshape(2, 768),
    )
    features = feature_stream.getvalue()
    files = {
        name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
        for name, value in {
            "nodes.csv": nodes,
            "relations.csv": relation_bytes,
            "features.npz": features,
        }.items()
    }
    manifest: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-input/2.0",
        "datasetId": "test:tiny",
        "displayName": "Tiny contract graph",
        "nodeCount": 2,
        "relationRowCount": len(relations.strip().splitlines()),
        "featureDimension": 768,
        "modalities": ["coRT"],
        "files": files,
    }
    manifest.update(manifest_overrides or {})
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.comment = archive_comment
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("nodes.csv", nodes)
        archive.writestr("relations.csv", relation_bytes)
        archive.writestr("features.npz", features)
    return output.getvalue()


class _ChunkedRequestBody(httpx.AsyncByteStream):
    def __init__(self, payload: bytes, *, chunk_bytes: int = 64 * 1024) -> None:
        self.payload = payload
        self.chunk_bytes = chunk_bytes

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        for offset in range(0, len(self.payload), self.chunk_bytes):
            yield self.payload[offset : offset + self.chunk_bytes]


def _multipart_file_body(payload: bytes) -> tuple[str, bytes]:
    boundary = "governance-upload-limit"
    body = b"".join(
        (
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="file"; filename="oversized.zip"\r\n',
            b"Content-Type: application/zip\r\n\r\n",
            payload,
            f"\r\n--{boundary}--\r\n".encode(),
        )
    )
    return boundary, body


@pytest.mark.anyio
@pytest.mark.parametrize(
    "endpoint",
    (
        "/api/v2/gfm/governance/artifacts",
        "/api/v2/gfm/governance/artifacts/compatibility",
    ),
)
async def test_declared_oversized_multipart_is_rejected_before_form_parsing(
    endpoint: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unconfigured_settings: Settings,
) -> None:
    max_bundle_bytes = 1024
    root = tmp_path / "governance"
    settings = unconfigured_settings.model_copy(
        update={
            "gfm_governance_root": str(root),
            "gfm_governance_bundle_max_bytes": max_bundle_bytes,
            "gfm_governance_expanded_max_bytes": 2 * 1024,
        }
    )
    form_calls = 0
    original_form = StarletteRequest.form

    async def tracked_form(self: StarletteRequest, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        nonlocal form_calls
        form_calls += 1
        return await original_form(self, *args, **kwargs)

    monkeypatch.setattr(StarletteRequest, "form", tracked_form)
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            endpoint,
            content=b"multipart parser must not receive this body",
            headers={
                "Content-Type": "multipart/form-data; boundary=unused",
                "Content-Length": str(
                    max_bundle_bytes
                    + GOVERNANCE_MULTIPART_OVERHEAD_MAX_BYTES
                    + 1
                ),
            },
        )

    assert response.status_code == 413, response.text
    assert response.json()["detail"]["code"] == "GOVERNANCE_BUNDLE_TOO_LARGE"
    assert form_calls == 0
    assert list((root / "incoming").iterdir()) == []


@pytest.mark.anyio
@pytest.mark.parametrize("declared_length", (None, "1", "not-a-number"))
async def test_missing_or_untrusted_length_still_hits_chunked_file_limit(
    declared_length: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unconfigured_settings: Settings,
) -> None:
    max_bundle_bytes = GOVERNANCE_UPLOAD_CHUNK_BYTES
    root = tmp_path / "governance"
    settings = unconfigured_settings.model_copy(
        update={
            "gfm_governance_root": str(root),
            "gfm_governance_bundle_max_bytes": max_bundle_bytes,
            "gfm_governance_expanded_max_bytes": 2 * max_bundle_bytes,
        }
    )
    boundary, body = _multipart_file_body(b"x" * (max_bundle_bytes + 1))
    read_sizes: list[int] = []
    original_read = StarletteUploadFile.read

    async def tracked_read(self: StarletteUploadFile, size: int = -1) -> bytes:
        read_sizes.append(size)
        return await original_read(self, size)

    monkeypatch.setattr(StarletteUploadFile, "read", tracked_read)
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if declared_length is not None:
        headers["Content-Length"] = declared_length
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        request = client.build_request(
            "POST",
            "/api/v2/gfm/governance/artifacts",
            content=_ChunkedRequestBody(body),
            headers=headers,
        )
        response = await client.send(request)

    assert response.status_code == 413, response.text
    assert response.json()["detail"]["code"] == "GOVERNANCE_BUNDLE_TOO_LARGE"
    assert read_sizes == [GOVERNANCE_UPLOAD_CHUNK_BYTES, 1]
    assert list((root / "incoming").iterdir()) == []


@pytest.mark.anyio
async def test_underreported_length_cannot_bypass_streaming_multipart_body_limit(
    tmp_path: Path,
    unconfigured_settings: Settings,
) -> None:
    max_bundle_bytes = 1024
    root = tmp_path / "governance"
    settings = unconfigured_settings.model_copy(
        update={
            "gfm_governance_root": str(root),
            "gfm_governance_bundle_max_bytes": max_bundle_bytes,
            "gfm_governance_expanded_max_bytes": 2 * 1024,
        }
    )
    boundary, body = _multipart_file_body(
        b"x" * (max_bundle_bytes + GOVERNANCE_MULTIPART_OVERHEAD_MAX_BYTES)
    )
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/v2/gfm/governance/artifacts",
            content=_ChunkedRequestBody(body),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": "1",
            },
        )

    assert response.status_code == 413, response.text
    assert response.json()["detail"]["code"] == "GOVERNANCE_BUNDLE_TOO_LARGE"
    assert list((root / "incoming").iterdir()) == []


@pytest.mark.anyio
async def test_v2_upload_boundary_does_not_change_existing_json_length_contract(
    unconfigured_settings: Settings,
) -> None:
    app = create_app(unconfigured_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        oversized = await client.post(
            "/api/v2/gfm/governance/runs",
            content=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(unconfigured_settings.gfm_request_max_bytes + 1),
            },
        )
        missing = await client.post(
            "/api/v2/gfm/governance/runs",
            content=_ChunkedRequestBody(b"{}"),
            headers={"Content-Type": "application/json"},
        )

    assert oversized.status_code == 413
    assert oversized.json()["detail"]["code"] == "GFM_REQUEST_SIZE_INVALID"
    assert missing.status_code == 400
    assert missing.json()["detail"]["code"] == "GFM_REQUEST_SIZE_INVALID"


def test_bundle_hashes_are_deterministic_and_align_with_materializer_contract() -> None:
    payload = _bundle()
    manifest, first = inspect_governance_bundle(
        payload, clean_self_loops=False, max_expanded_bytes=1024 * 1024 * 1024
    )
    _, second = inspect_governance_bundle(
        payload, clean_self_loops=False, max_expanded_bytes=1024 * 1024 * 1024
    )
    assert manifest.schema_version == "socialgraph-fm.governance-input/2.0"
    assert first == second
    assert first["artifactId"] == f"governance-artifact-{first['datasetContentHash'][:32]}"
    assert first["nodeCount"] == 2
    assert first["relationRowCount"] == 1
    assert first["selfLoopsRemoved"] == 0


def test_bundle_rejects_feature_misalignment_and_manifest_tamper() -> None:
    with pytest.raises(GfmProxyError) as misaligned:
        inspect_governance_bundle(
            _bundle(feature_ids=("b", "a")),
            clean_self_loops=False,
            max_expanded_bytes=1024 * 1024 * 1024,
        )
    assert misaligned.value.code == "GOVERNANCE_FEATURE_NODE_ALIGNMENT_MISMATCH"

    payload = _bundle(manifest_overrides={"nodeCount": 3})
    with pytest.raises(GfmProxyError) as tampered:
        inspect_governance_bundle(
            payload,
            clean_self_loops=False,
            max_expanded_bytes=1024 * 1024 * 1024,
        )
    assert tampered.value.code == "GOVERNANCE_NODE_COUNT_MISMATCH"


@pytest.mark.anyio
async def test_invalid_utf8_feature_node_ids_are_explicit_contract_errors(
    unconfigured_settings: Settings,
) -> None:
    payload = _bundle(feature_ids=(b"\xff", b"b"))
    app = create_app(unconfigured_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        for endpoint, form in (
            ("/api/v2/gfm/governance/artifacts/compatibility", None),
            ("/api/v2/gfm/governance/artifacts", {"cleanSelfLoops": "false"}),
        ):
            response = await client.post(
                endpoint,
                files={"file": ("invalid-ids.zip", payload, "application/zip")},
                data=form,
            )
            assert response.status_code == 400, response.text
            assert response.json()["detail"]["code"] == (
                "GOVERNANCE_FEATURE_NODE_IDS_UTF8_INVALID"
            )


@pytest.mark.anyio
async def test_service_client_uses_larger_bounded_response_limit_only_for_governance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = json.dumps({"payload": "x" * 1_500}).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    monkeypatch.setattr(gfm_client_module, "MAX_GFM_RESPONSE_BYTES", 1_024)
    monkeypatch.setattr(
        gfm_client_module, "MAX_GOVERNANCE_RESPONSE_BYTES", 2_048
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = tmp_path / "token"
    token.write_text("x" * 64, encoding="utf-8")
    client = GfmServiceClient(
        f"http://127.0.0.1:{server.server_address[1]}", token_file=token
    )
    try:
        response = await client.governance_capabilities()
        assert len(response["payload"]) == 1_500
        with pytest.raises(GfmProxyError) as raised:
            await client.core_capabilities()
        assert raised.value.code == "GFM_CORE_RESPONSE_TOO_LARGE"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.anyio
async def test_service_client_forwards_target_policy_fit_identity_body(
    tmp_path: Path,
) -> None:
    received: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            received.update(json.loads(self.rfile.read(length)))
            body = b'{"accepted":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = tmp_path / "token"
    token.write_text("x" * 64, encoding="utf-8")
    client = GfmServiceClient(
        f"http://127.0.0.1:{server.server_address[1]}", token_file=token
    )
    request = {
        "schemaVersion": "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
        "targetTaskRegistrationId": "target-task-" + "1" * 32,
        "runId": "governance-" + "2" * 32,
        "resultHash": "3" * 64,
    }
    try:
        assert await client.fit_governance_policy("4" * 64, request) == {
            "accepted": True
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert received == request


def test_self_loop_cleaning_requires_confirmation_and_changes_identity() -> None:
    payload = _bundle(relations="a,a,coRT,0\na,b,coRT,1\n")
    with pytest.raises(GfmProxyError) as blocked:
        inspect_governance_bundle(
            payload,
            clean_self_loops=False,
            max_expanded_bytes=1024 * 1024 * 1024,
        )
    assert blocked.value.code == "GOVERNANCE_SELF_LOOP_CONFIRMATION_REQUIRED"
    _, cleaned = inspect_governance_bundle(
        payload,
        clean_self_loops=True,
        max_expanded_bytes=1024 * 1024 * 1024,
    )
    assert cleaned["selfLoopsRemoved"] == 1
    assert cleaned["relationRowCount"] == 1


def test_inbox_is_immutable_and_idempotent(tmp_path: Path) -> None:
    inbox = GovernanceArtifactInbox(tmp_path / "governance")
    first = inbox.commit(
        _bundle(), clean_self_loops=False, max_expanded_bytes=1024 * 1024 * 1024
    )
    second = inbox.commit(
        _bundle(archive_comment=b"semantically-identical-repack"),
        clean_self_loops=False,
        max_expanded_bytes=1024 * 1024 * 1024,
    )
    assert first == second
    assert (tmp_path / "governance" / "incoming" / first.artifact_id / "bundle.zip").is_file()
    assert inbox.list(offset=0, limit=10).total == 1
    with pytest.raises(GfmProxyError):
        inbox.get("governance-artifact-../../outside-invalid-artifact")


def test_governance_store_is_append_only_and_reopenable(tmp_path: Path) -> None:
    store = GovernanceStore(tmp_path / "governance")
    run_id = "governance-" + "1" * 32
    store.put_run_binding(
        {
            "runId": run_id,
            "requestHash": "1" * 64,
            "artifactId": "governance-artifact-" + "2" * 32,
            "datasetContentHash": "3" * 64,
            "graphVersionHash": "4" * 64,
            "modelVersionId": "socialgraph-fm-global/test",
            "modelVersionHash": "5" * 64,
            "modelStateHash": "6" * 64,
            "createdAt": "2026-08-18T00:00:00.000000Z",
        }
    )
    case = store.create_case(
        CaseCreateRequest(
            schemaVersion=GOVERNANCE_SCHEMA_VERSION,
            runId=run_id,
            title="Local review",
            description="Evidence-led review.",
        )
    )
    case = store.add_item(
        case.case_id,
        CaseItemRequest(
            schemaVersion=GOVERNANCE_SCHEMA_VERSION,
            targetType="node",
            targetId="a",
            note="High-priority candidate",
        ),
    )
    case = store.transition(
        case.case_id,
        CaseTransitionRequest(
            schemaVersion=GOVERNANCE_SCHEMA_VERSION,
            state="active",
            reason="Begin analyst review",
        ),
    )
    case = store.add_review(
        case.case_id,
        ReviewEventRequest(
            schemaVersion=GOVERNANCE_SCHEMA_VERSION,
            targetType="node",
            targetId="a",
            decision="pending",
            reason="Needs secondary evidence",
        ),
    )
    first_event_hash = case.review_events[0].event_hash
    case = store.add_review(
        case.case_id,
        ReviewEventRequest(
            schemaVersion=GOVERNANCE_SCHEMA_VERSION,
            targetType="node",
            targetId="a",
            decision="confirmed",
            reason="Secondary evidence reviewed",
        ),
    )
    assert case.review_events[1].previous_event_hash == first_event_hash
    assert case.current_decisions["node:a"] == "confirmed"
    for state in ("concluded", "archived", "active"):
        case = store.transition(
            case.case_id,
            CaseTransitionRequest(
                schemaVersion=GOVERNANCE_SCHEMA_VERSION,
                state=state,
                reason=f"Move to {state}",
            ),
        )
    assert case.state == "active"


@pytest.mark.parametrize(
    ("table", "trigger", "column", "replacement"),
    (
        (
            "case_state_events",
            "no_state_event_update",
            "reason",
            "tampered-state-reason",
        ),
        ("case_items", "no_case_item_update", "note", "tampered-item-note"),
        ("review_events", "no_review_event_update", "decision", "rejected"),
        ("review_events", "no_review_event_update", "actor", "tampered-actor"),
        (
            "review_events",
            "no_review_event_update",
            "previous_event_hash",
            "f" * 64,
        ),
        (
            "governance_cases",
            "no_governance_case_update",
            "run_id",
            "governance-" + "9" * 32,
        ),
        (
            "governance_cases",
            "no_governance_case_update",
            "title",
            "tampered-case-title",
        ),
        (
            "governance_cases",
            "no_governance_case_update",
            "description",
            "tampered-case-description",
        ),
    ),
)
def test_governance_store_rejects_direct_sqlite_audit_tampering(
    tmp_path: Path,
    table: str,
    trigger: str,
    column: str,
    replacement: str,
) -> None:
    store = GovernanceStore(tmp_path / "governance")
    run_id = "governance-" + "1" * 32
    case = store.create_case(
        CaseCreateRequest(
            schemaVersion=GOVERNANCE_SCHEMA_VERSION,
            runId=run_id,
            title="Tamper regression",
            description="The audit trail must fail closed.",
        )
    )
    case = store.add_item(
        case.case_id,
        CaseItemRequest(
            schemaVersion=GOVERNANCE_SCHEMA_VERSION,
            targetType="node",
            targetId="a",
            note="Original note",
        ),
    )
    case = store.transition(
        case.case_id,
        CaseTransitionRequest(
            schemaVersion=GOVERNANCE_SCHEMA_VERSION,
            state="active",
            reason="Begin review",
        ),
    )
    for decision in ("pending", "confirmed"):
        case = store.add_review(
            case.case_id,
            ReviewEventRequest(
                schemaVersion=GOVERNANCE_SCHEMA_VERSION,
                targetType="node",
                targetId="a",
                decision=decision,
                reason=f"Record {decision}",
                actor="local-analyst",
            ),
        )

    with sqlite3.connect(store.database_path) as connection:
        connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute(
            f"UPDATE {table} SET {column} = ? WHERE case_id = ?",
            (replacement, case.case_id),
        )

    for read in (
        lambda: store.get_case(case.case_id),
        lambda: store.list_cases(offset=0, limit=10),
        lambda: store.case_state_timeline(case.case_id),
        lambda: GovernanceStore(store.root),
    ):
        with pytest.raises(GfmProxyError) as raised:
            read()
        assert raised.value.status_code == 502
        assert raised.value.code == "GOVERNANCE_AUDIT_INVALID"


def test_governance_item_hash_binds_item_to_its_case(tmp_path: Path) -> None:
    store = GovernanceStore(tmp_path / "governance")
    run_id = "governance-" + "1" * 32
    source = store.create_case(
        CaseCreateRequest(
            schemaVersion=GOVERNANCE_SCHEMA_VERSION,
            runId=run_id,
            title="Source case",
            description="Owns the item.",
        )
    )
    source = store.add_item(
        source.case_id,
        CaseItemRequest(
            schemaVersion=GOVERNANCE_SCHEMA_VERSION,
            targetType="node",
            targetId="a",
            note="Bound to the source case",
        ),
    )
    destination = store.create_case(
        CaseCreateRequest(
            schemaVersion=GOVERNANCE_SCHEMA_VERSION,
            runId=run_id,
            title="Destination case",
            description="Must not inherit the item.",
        )
    )

    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP TRIGGER no_case_item_update")
        connection.execute(
            "UPDATE case_items SET case_id = ? WHERE item_id = ?",
            (destination.case_id, source.items[0].item_id),
        )

    assert store.get_case(source.case_id).items == ()
    for read in (
        lambda: store.get_case(destination.case_id),
        lambda: store.list_cases(offset=0, limit=10),
        lambda: store.case_state_timeline(destination.case_id),
        lambda: GovernanceStore(store.root),
    ):
        with pytest.raises(GfmProxyError) as raised:
            read()
        assert raised.value.status_code == 502
        assert raised.value.code == "GOVERNANCE_AUDIT_INVALID"


def test_governance_store_migrates_legacy_hashes_once_without_data_loss(
    tmp_path: Path,
) -> None:
    root = tmp_path / "governance"
    root.mkdir()
    database_path = root / "governance.sqlite3"
    case_id = "case-" + "a" * 32
    item_id = "item-" + "b" * 32
    run_id = "governance-" + "c" * 32
    created_at = "2026-08-18T00:00:00.000000Z"
    state_payload = {
        "caseId": case_id,
        "state": "draft",
        "reason": "case-created",
        "createdAt": created_at,
    }
    legacy_item_payload = {
        "itemId": item_id,
        "targetType": "node",
        "targetId": "legacy-node",
        "note": "Preserve this item",
        "createdAt": created_at,
    }
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE governance_cases (
                case_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE case_state_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                state TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );
            CREATE TABLE case_items (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT NOT NULL UNIQUE,
                case_id TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                note TEXT NOT NULL,
                created_at TEXT NOT NULL,
                item_hash TEXT NOT NULL UNIQUE,
                UNIQUE(case_id, target_type, target_id)
            );
            """
        )
        connection.execute(
            "INSERT INTO governance_cases VALUES (?, ?, ?, ?, ?)",
            (case_id, run_id, "Legacy case", "Must survive migration", created_at),
        )
        connection.execute(
            """INSERT INTO case_state_events
               (case_id, state, reason, created_at, event_hash)
               VALUES (?, 'draft', 'case-created', ?, ?)""",
            (case_id, created_at, canonical_sha256(state_payload)),
        )
        connection.execute(
            """INSERT INTO case_items
               (item_id, case_id, target_type, target_id, note, created_at, item_hash)
               VALUES (?, ?, 'node', 'legacy-node', 'Preserve this item', ?, ?)""",
            (item_id, case_id, created_at, canonical_sha256(legacy_item_payload)),
        )

    migrated = GovernanceStore(root)
    case = migrated.get_case(case_id)
    assert case.run_id == run_id
    assert case.title == "Legacy case"
    assert case.description == "Must survive migration"
    assert tuple(item.item_id for item in case.items) == (item_id,)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        marker = connection.execute(
            "SELECT schema_version FROM governance_store_metadata WHERE singleton = 1"
        ).fetchone()
        root_row = connection.execute(
            "SELECT * FROM governance_cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        item_row = connection.execute(
            "SELECT * FROM case_items WHERE item_id = ?", (item_id,)
        ).fetchone()
        collection_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(review_collections)")
        }
        assert marker is not None and marker["schema_version"] == 4
    assert "result_hash" in collection_columns
    assert root_row is not None and len(str(root_row["root_hash"])) == 64
    assert item_row is not None
    migrated_item_hash = str(item_row["item_hash"])
    assert migrated_item_hash != canonical_sha256(legacy_item_payload)

    reopened = GovernanceStore(root)
    assert reopened.get_case(case_id) == case
    with sqlite3.connect(database_path) as connection:
        stored_again = connection.execute(
            "SELECT item_hash FROM case_items WHERE item_id = ?", (item_id,)
        ).fetchone()
    assert stored_again is not None and stored_again[0] == migrated_item_hash


def test_governance_store_migrates_v2_run_bindings_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "governance"
    root.mkdir()
    database_path = root / "governance.sqlite3"
    legacy = {
        "runId": "governance-" + "a" * 32,
        "requestHash": "1" * 64,
        "artifactId": "governance-artifact-" + "2" * 32,
        "datasetContentHash": "3" * 64,
        "graphVersionHash": "4" * 64,
        "modelVersionId": "socialgraph-fm-global/legacy",
        "modelVersionHash": "5" * 64,
        "createdAt": "2026-08-18T00:00:00Z",
    }
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE governance_cases (
                case_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL,
                root_hash TEXT
            );
            CREATE TABLE governance_store_metadata (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                schema_version INTEGER NOT NULL
            );
            CREATE TABLE online_run_bindings (
                run_id TEXT PRIMARY KEY,
                request_hash TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                dataset_content_hash TEXT NOT NULL,
                graph_version_hash TEXT NOT NULL,
                model_version_id TEXT NOT NULL,
                model_version_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                binding_hash TEXT NOT NULL UNIQUE
            );
            """
        )
        connection.execute(
            "INSERT INTO governance_store_metadata VALUES (1, 2)"
        )
        connection.execute(
            "INSERT INTO online_run_bindings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*legacy.values(), canonical_sha256(legacy)),
        )

    migrated = GovernanceStore(root)

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        marker = connection.execute(
            "SELECT schema_version FROM governance_store_metadata WHERE singleton = 1"
        ).fetchone()
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(online_run_bindings)")
        }
        collection_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(review_collections)")
        }
    assert marker is not None and marker["schema_version"] == 4
    assert "model_state_hash" in columns
    assert "result_hash" in collection_columns
    with pytest.raises(GfmProxyError) as missing_state:
        migrated.get_run_binding(legacy["runId"])
    assert missing_state.value.code == "GOVERNANCE_RUN_BINDING_MODEL_STATE_MISSING"


@pytest.mark.anyio
async def test_v2_unavailable_contract_and_input_boundary(
    unconfigured_settings: Settings,
) -> None:
    app = create_app(unconfigured_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        capabilities = await client.get("/api/v2/gfm/governance/capabilities")
        assert capabilities.status_code == 200
        assert capabilities.json()["onlineForwardReady"] is False
        contract = await client.get("/api/v2/gfm/governance/input-contract")
        assert contract.status_code == 200
        assert contract.json()["rawTextAccepted"] is False
        limits = contract.json()["limits"]
        assert limits["maxMultipartOverheadBytes"] == (
            GOVERNANCE_MULTIPART_OVERHEAD_MAX_BYTES
        )
        assert limits["maxMultipartBodyBytes"] == (
            limits["maxBundleBytes"] + limits["maxMultipartOverheadBytes"]
        )
        assert "cannot run SocialGraph-FM Governance" in contract.json()[
            "ordinaryGraphPolicy"
        ]


@pytest.mark.anyio
async def test_compatibility_preflight_detects_self_loops_without_committing(
    unconfigured_settings: Settings,
) -> None:
    app = create_app(unconfigured_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/v2/gfm/governance/artifacts/compatibility",
            files={
                "file": (
                    "self-loop.zip",
                    _bundle(relations="a,a,coRT,0\na,b,coRT,1\n"),
                    "application/zip",
                )
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["requiresSelfLoopCleaning"] is True
        assert response.json()["selfLoopsDetected"] == 1
        artifacts = await client.get("/api/v2/gfm/governance/artifacts")
        assert artifacts.json()["total"] == 0


def test_report_hash_uses_canonical_json() -> None:
    first = canonical_sha256({"b": 2, "a": 1})
    second = canonical_sha256({"a": 1, "b": 2})
    assert first == second


def _hash_field(payload: dict[str, Any], field: str) -> dict[str, Any]:
    payload[field] = canonical_sha256(payload)
    return payload


class FakeOnlineClient:
    def __init__(self) -> None:
        self.artifact: dict[str, Any] | None = None
        self.request: dict[str, Any] | None = None
        self.include_unbound_run = False
        self.run_id = "governance-" + "a" * 32
        self.model_id = "socialgraph-fm-global/test"
        self.model_hash = "b" * 64
        self.state_hash = "c" * 64

    async def governance_capabilities(self) -> dict[str, Any]:
        return _hash_field(
            {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "channel": "governance",
                "taskId": "coordination_risk",
                "servingReady": True,
                "onlineForwardReady": True,
                "unavailableReason": None,
                "modelVersionId": self.model_id,
                "modelVersionHash": self.model_hash,
                "modelStateHash": self.state_hash,
                "supportedProtocols": ["global"],
                "skills": [
                    "inspect_graph",
                    "run_governance_analysis",
                    "get_evidence_subgraph",
                    "discover_coordination_groups",
                    "rank_coordination_relations",
                    "retrieve_similar_cases",
                    "get_model_dataset_cards",
                    "draft_review_report",
                ],
                "inputSchemaVersion": "socialgraph-fm.governance-input/2.0",
                "modalities": ["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"],
                "sampleArtifactId": None,
                "limits": {
                    "maxNodes": 10_000,
                    "maxRelationRows": 500_000,
                    "maxEvidenceNodes": 300,
                    "maxEvidenceEdges": 1_000,
                    "maxPreviewNodes": 3_000,
                    "maxPreviewEdges": 12_000,
                },
            },
            "capabilityHash",
        )

    async def governance_health(self) -> dict[str, Any]:
        return _hash_field(
            {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "serviceIdentity": "d" * 64,
                "servingReady": True,
                "onlineForwardReady": True,
                "modelVersionId": self.model_id,
                "modelVersionHash": self.model_hash,
                "modelStateHash": self.state_hash,
                "device": "cpu",
                "dtype": "float16",
                "loadedAt": "2026-08-18T00:00:00Z",
                "queueDepth": 0,
                "activeRunId": None,
                "runtimeRecipeHash": "e" * 64,
            },
            "healthHash",
        )

    async def validate_governance_artifact(
        self, artifact_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        self.artifact = _hash_field(
            {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "artifactId": artifact_id,
                "datasetContentHash": payload["datasetContentHash"],
                "graphVersionHash": payload["graphVersionHash"],
                "nodeCount": 2,
                "relationRowCount": 1,
                "selfLoopsRemoved": 0,
                "modalities": ["coRT"],
                "createdAt": "2026-08-18T00:00:00Z",
                "compatibility": "compatible",
            },
            "artifactHash",
        )
        return deepcopy(self.artifact)

    async def get_governance_preview(self, artifact_id: str) -> dict[str, Any]:
        assert self.artifact is not None
        return _hash_field(
            {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "artifactId": artifact_id,
                "datasetContentHash": self.artifact["datasetContentHash"],
                "graphVersionHash": self.artifact["graphVersionHash"],
                "runId": None,
                "resultHash": None,
                "nodes": [
                    {
                        "id": "a",
                        "label": "Account A",
                        "degree": 1,
                        "structureMissing": False,
                        "score": None,
                        "riskBand": None,
                        "groupId": None,
                    },
                    {
                        "id": "b",
                        "label": "Account B",
                        "degree": 1,
                        "structureMissing": False,
                        "score": None,
                        "riskBand": None,
                        "groupId": None,
                    },
                ],
                "edges": [
                    {
                        "id": "a:b",
                        "source": "a",
                        "target": "b",
                        "modalities": ["coRT"],
                        "factual": True,
                    }
                ],
                "nodeCount": 2,
                "edgeCount": 1,
                "partialPreview": False,
            },
            "previewHash",
        )

    def _status(self, status: str, stage: str) -> dict[str, Any]:
        assert self.request is not None
        return _hash_field(
            {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "runId": self.run_id,
                "requestHash": canonical_sha256(self.request),
                "artifactId": self.request["artifactId"],
                "datasetContentHash": self.request["datasetContentHash"],
                "graphVersionHash": self.request["graphVersionHash"],
                "modelVersionId": self.model_id,
                "modelVersionHash": self.model_hash,
                "modelStateHash": self.state_hash,
                "status": status,
                "stage": stage,
                "progress": 100 if status == "succeeded" else 0,
                "createdAt": "2026-08-18T00:00:00Z",
                "updatedAt": "2026-08-18T00:00:01Z",
                "errorCode": None,
                "cancelRequested": False,
            },
            "statusHash",
        )

    async def create_governance_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.request = deepcopy(payload)
        return self._status("queued", "queued")

    async def list_governance_runs(self, offset: int, limit: int) -> dict[str, Any]:
        items = [self._status("succeeded", "completed")]
        if self.include_unbound_run:
            unbound = dict(items[0])
            unbound.pop("statusHash")
            unbound["runId"] = "governance-" + "e" * 32
            items.append(_hash_field(unbound, "statusHash"))
        return {
            "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
            "items": items,
            "total": len(items),
            "offset": offset,
            "limit": limit,
        }

    async def get_governance_run(self, run_id: str) -> dict[str, Any]:
        assert run_id == self.run_id
        return self._status("succeeded", "completed")

    async def cancel_governance_run(self, run_id: str) -> dict[str, Any]:
        assert run_id == self.run_id
        return self._status("succeeded", "completed")

    async def retry_governance_run(self, run_id: str) -> dict[str, Any]:
        assert run_id == self.run_id
        status = self._status("queued", "queued")
        status.pop("statusHash")
        status["runId"] = "governance-" + "f" * 32
        return _hash_field(status, "statusHash")

    def _finding(self, node_id: str, rank: int, score: float) -> dict[str, Any]:
        return {
            "nodeId": node_id,
            "label": f"Account {node_id.upper()}",
            "score": score,
            "logit": 1.0 if score > 0.5 else -1.0,
            "rank": rank,
            "riskBand": "high" if score > 0.8 else "low",
            "predictedPositive": score > 0.5,
            "structureMissing": False,
            "routes": [
                {"expert": "shared", "weight": 1.0},
                {"expert": "domain:russia", "weight": 0.6},
                {"expert": "null", "weight": 0.4},
            ],
            "modalityContribution": {"text": 0.6, "structure": 0.4},
            "modalityEvidence": {
                "coRT": 1,
                "coURL": 0,
                "hashSeq": 0,
                "fastRT": 0,
                "tweetSim": 0,
            },
            "communityId": "group-1",
        }

    def _result(self) -> dict[str, Any]:
        assert self.request is not None
        return _hash_field(
            {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "runId": self.run_id,
                "requestHash": canonical_sha256(self.request),
                "artifactId": self.request["artifactId"],
                "datasetContentHash": self.request["datasetContentHash"],
                "graphVersionHash": self.request["graphVersionHash"],
                "modelVersionId": self.model_id,
                "modelVersionHash": self.model_hash,
                "modelStateHash": self.state_hash,
                "threshold": 0.5,
                "calibration": {
                    "temperature": 1.0,
                    "bias": 0.0,
                    "referenceThreshold": 0.5,
                    "applicability": "out_of_domain_unverified",
                },
                "referenceMetrics": {"global": {"macroF1": 0.9}},
                "datasetMetrics": None,
                "distribution": {
                    "low": 1,
                    "review": 0,
                    "high": 1,
                    "predictedPositive": 1,
                    "total": 2,
                },
                "findings": [self._finding("a", 1, 0.9)],
                "totalFindings": 2,
                "limitations": ["Human review required."],
                "completedAt": "2026-08-18T00:00:01Z",
            },
            "resultHash",
        )

    async def get_governance_result(self, run_id: str) -> dict[str, Any]:
        assert run_id == self.run_id
        return self._result()

    async def get_governance_run_preview(self, run_id: str) -> dict[str, Any]:
        assert self.artifact is not None
        result = self._result()
        return _hash_field(
            {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "artifactId": self.artifact["artifactId"],
                "datasetContentHash": self.artifact["datasetContentHash"],
                "graphVersionHash": self.artifact["graphVersionHash"],
                "runId": run_id,
                "resultHash": result["resultHash"],
                "nodes": [
                    {
                        "id": "a",
                        "label": "Account A",
                        "degree": 1,
                        "structureMissing": False,
                        "score": 0.9,
                        "riskBand": "high",
                        "groupId": "group-1",
                    },
                    {
                        "id": "b",
                        "label": "Account B",
                        "degree": 1,
                        "structureMissing": False,
                        "score": 0.1,
                        "riskBand": "low",
                        "groupId": "group-1",
                    },
                ],
                "edges": [
                    {
                        "id": "a:b",
                        "source": "a",
                        "target": "b",
                        "modalities": ["coRT"],
                        "factual": True,
                    }
                ],
                "nodeCount": 2,
                "edgeCount": 1,
                "partialPreview": False,
            },
            "previewHash",
        )

    async def get_governance_findings(
        self, run_id: str, offset: int, limit: int
    ) -> dict[str, Any]:
        items = [self._finding("a", 1, 0.9), self._finding("b", 2, 0.1)]
        return _hash_field(
            {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "runId": run_id,
                "items": items[offset : offset + limit],
                "total": 2,
                "offset": offset,
                "limit": limit,
            },
            "pageHash",
        )

    async def get_governance_evidence(self, run_id: str, node_id: str) -> dict[str, Any]:
        result = self._result()
        finding = self._finding(node_id, 1 if node_id == "a" else 2, 0.9 if node_id == "a" else 0.1)
        return _hash_field(
            {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "runId": run_id,
                "resultHash": result["resultHash"],
                "artifactId": result["artifactId"],
                "datasetContentHash": result["datasetContentHash"],
                "graphVersionHash": result["graphVersionHash"],
                "modelVersionId": result["modelVersionId"],
                "modelVersionHash": result["modelVersionHash"],
                "modelStateHash": result["modelStateHash"],
                "threshold": result["threshold"],
                "node": finding,
                "neighbors": [],
                "structuralSignals": {
                    "fusedDegree": 1,
                    "structureMissing": False,
                    "relationNeighborCounts": {
                        "coRT": 1,
                        "coURL": 0,
                        "hashSeq": 0,
                        "fastRT": 0,
                        "tweetSim": 0,
                    },
                    "twoHopNodeCount": 1,
                    "relationEvidenceRole": "explanationOnly",
                },
                "evidenceSubgraph": {
                    "depth": 2,
                    "nodeCount": 1,
                    "edgeCount": 0,
                    "truncated": False,
                    "nodes": [
                        {
                            "nodeId": node_id,
                            "score": finding["score"],
                            "hop": 0,
                            "riskBand": finding["riskBand"],
                            "predictedPositive": finding["predictedPositive"],
                            "structureMissing": False,
                        }
                    ],
                    "edges": [],
                },
                "truncated": False,
                "limitation": "Relations are explanation-only; human review is required.",
            },
            "evidenceHash",
        )

    async def get_governance_derivations(
        self, run_id: str, kind: str, offset: int, limit: int
    ) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        if kind == "groups":
            items = [
                {
                    "id": "group-1",
                    "kind": "group",
                    "priority": 0.8,
                    "nodeIds": ["a", "b"],
                    "source": None,
                    "target": None,
                    "modalities": ["coRT"],
                    "memberCount": 2,
                    "meanScore": 0.5,
                    "p90Score": 0.82,
                    "scoreComponents": {"p90": 0.82, "mean": 0.5},
                    "factual": False,
                    "limitation": "Derived analyst priority.",
                }
            ]
        return _hash_field(
            {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "runId": run_id,
                "items": items[offset : offset + limit],
                "total": len(items),
                "offset": offset,
                "limit": limit,
            },
            "pageHash",
        )


@pytest.mark.anyio
async def test_online_run_rejects_a_stale_model_state(
    unconfigured_settings: Settings, tmp_path: Path
) -> None:
    fake = FakeOnlineClient()
    settings = unconfigured_settings.model_copy(
        update={"gfm_governance_root": str(tmp_path / "governance")}
    )
    app = create_app(settings, gfm_governance_client=fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        uploaded = await client.post(
            "/api/v2/gfm/governance/artifacts",
            files={"file": ("tiny.zip", _bundle(), "application/zip")},
            data={"cleanSelfLoops": "false"},
        )
        assert uploaded.status_code == 201, uploaded.text
        artifact = uploaded.json()
        created = await client.post(
            "/api/v2/gfm/governance/runs",
            json={
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "protocol": "global",
                "artifactId": artifact["artifactId"],
                "datasetContentHash": artifact["datasetContentHash"],
                "graphVersionHash": artifact["graphVersionHash"],
                "modelVersionId": fake.model_id,
                "modelStateHash": "0" * 64,
                "topK": 1,
            },
        )
        assert created.status_code == 409, created.text
        assert created.json()["detail"]["code"] == "GFM_GOVERNANCE_MODEL_MISMATCH"
        assert fake.request is None


@pytest.mark.anyio
async def test_online_run_status_cannot_change_model_state_after_creation(
    unconfigured_settings: Settings, tmp_path: Path
) -> None:
    fake = FakeOnlineClient()
    settings = unconfigured_settings.model_copy(
        update={"gfm_governance_root": str(tmp_path / "governance")}
    )
    app = create_app(settings, gfm_governance_client=fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        uploaded = await client.post(
            "/api/v2/gfm/governance/artifacts",
            files={"file": ("tiny.zip", _bundle(), "application/zip")},
            data={"cleanSelfLoops": "false"},
        )
        assert uploaded.status_code == 201, uploaded.text
        artifact = uploaded.json()
        created = await client.post(
            "/api/v2/gfm/governance/runs",
            json={
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "protocol": "global",
                "artifactId": artifact["artifactId"],
                "datasetContentHash": artifact["datasetContentHash"],
                "graphVersionHash": artifact["graphVersionHash"],
                "modelVersionId": fake.model_id,
                "modelStateHash": fake.state_hash,
                "topK": 1,
            },
        )
        assert created.status_code == 202, created.text

        fake.state_hash = "d" * 64
        stale = await client.get(
            f"/api/v2/gfm/governance/runs/{created.json()['runId']}"
        )

        assert stale.status_code == 502, stale.text
        assert stale.json()["detail"]["code"] == "GFM_GOVERNANCE_RUN_BINDING_MISMATCH"


@pytest.mark.anyio
async def test_true_online_http_governance_loop(
    unconfigured_settings: Settings, tmp_path: Path
) -> None:
    fake = FakeOnlineClient()
    settings = unconfigured_settings.model_copy(
        update={"gfm_governance_root": str(tmp_path / "governance")}
    )
    app = create_app(settings, gfm_governance_client=fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        uploaded = await client.post(
            "/api/v2/gfm/governance/artifacts",
            files={"file": ("tiny.zip", _bundle(), "application/zip")},
            data={"cleanSelfLoops": "false"},
        )
        assert uploaded.status_code == 201, uploaded.text
        artifact = uploaded.json()
        created = await client.post(
            "/api/v2/gfm/governance/runs",
            json={
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "protocol": "global",
                "artifactId": artifact["artifactId"],
                "datasetContentHash": artifact["datasetContentHash"],
                "graphVersionHash": artifact["graphVersionHash"],
                    "modelVersionId": fake.model_id,
                    "modelStateHash": fake.state_hash,
                    "topK": 1,
            },
        )
        assert created.status_code == 202, created.text
        run_id = created.json()["runId"]
        fake.include_unbound_run = True
        legacy_binding = {
            "runId": "governance-" + "e" * 32,
            "requestHash": created.json()["requestHash"],
            "artifactId": artifact["artifactId"],
            "datasetContentHash": artifact["datasetContentHash"],
            "graphVersionHash": artifact["graphVersionHash"],
            "modelVersionId": fake.model_id,
            "modelVersionHash": fake.model_hash,
            "createdAt": "2026-08-18T00:00:00Z",
        }
        with sqlite3.connect(tmp_path / "governance" / "governance.sqlite3") as connection:
            connection.execute(
                """INSERT INTO online_run_bindings
                   (run_id, request_hash, artifact_id, dataset_content_hash,
                    graph_version_hash, model_version_id, model_version_hash,
                    created_at, binding_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (*legacy_binding.values(), canonical_sha256(legacy_binding)),
            )
        run_list = await client.get("/api/v2/gfm/governance/runs")
        assert run_list.status_code == 200, run_list.text
        assert run_list.json()["total"] == 1
        assert run_list.json()["items"][0]["runId"] == run_id
        result = await client.get(f"/api/v2/gfm/governance/runs/{run_id}/result")
        assert result.status_code == 200, result.text
        assert result.json()["datasetMetrics"] is None
        run_preview = await client.get(
            f"/api/v2/gfm/governance/runs/{run_id}/graph-preview"
        )
        assert run_preview.status_code == 200, run_preview.text
        assert run_preview.json()["nodes"][0]["score"] == 0.9
        retried = await client.post(
            f"/api/v2/gfm/governance/runs/{run_id}/retry"
        )
        assert retried.status_code == 202, retried.text
        assert retried.json()["runId"] != run_id
        evidence = await client.get(
            f"/api/v2/gfm/governance/runs/{run_id}/nodes/a/evidence"
        )
        assert evidence.status_code == 200
        encoded_evidence = await client.get(
            f"/api/v2/gfm/governance/runs/{run_id}/nodes/a%2Fb/evidence"
        )
        assert encoded_evidence.status_code == 200
        assert encoded_evidence.json()["node"]["nodeId"] == "a/b"
        case_response = await client.post(
            "/api/v2/gfm/governance/cases",
            json={
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "runId": run_id,
                "title": "Tiny graph review",
                "description": "Local governance flow",
            },
        )
        assert case_response.status_code == 201, case_response.text
        case_id = case_response.json()["caseId"]
        item = await client.post(
            f"/api/v2/gfm/governance/cases/{case_id}/items",
            json={
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "targetType": "node",
                "targetId": "a",
                "note": "Review candidate",
            },
        )
        assert item.status_code == 201
        active = await client.post(
            f"/api/v2/gfm/governance/cases/{case_id}/transitions",
            json={
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "state": "active",
                "reason": "Start review",
            },
        )
        assert active.status_code == 200
        review = await client.post(
            f"/api/v2/gfm/governance/cases/{case_id}/review-events",
            json={
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "targetType": "node",
                "targetId": "a",
                "decision": "pending",
                "reason": "Needs human corroboration",
                "actor": "governance-analyst",
            },
        )
        assert review.status_code == 201
        report = await client.get(
            f"/api/v2/gfm/governance/cases/{case_id}/report?format=json"
        )
        assert report.status_code == 200
        assert report.json()["audit"]["reviewEventHashes"]
        assert report.json()["case"]["reviewEvents"][0]["reason"] == (
            "Needs human corroboration"
        )
        assert len(report.json()["reportHash"]) == 64
        markdown = await client.get(
            f"/api/v2/gfm/governance/cases/{case_id}/report?format=markdown"
        )
        assert markdown.status_code == 200
        assert "Case hash:" in markdown.text
        assert "Needs human corroboration" in markdown.text
        html_report = await client.get(
            f"/api/v2/gfm/governance/cases/{case_id}/report?format=html"
        )
        assert html_report.status_code == 200
        assert "Audit timeline" in html_report.text
        assert "Needs human corroboration" in html_report.text
        assert (
            "<td>pending<br><small>Needs human corroboration</small></td>"
            in html_report.text
        )

        with sqlite3.connect(tmp_path / "governance" / "governance.sqlite3") as connection:
            connection.execute("DROP TRIGGER no_review_event_update")
            connection.execute(
                "UPDATE review_events SET reason = ? WHERE case_id = ?",
                ("tampered-after-report", case_id),
            )
        for endpoint in (
            f"/api/v2/gfm/governance/cases/{case_id}",
            "/api/v2/gfm/governance/cases",
            f"/api/v2/gfm/governance/cases/{case_id}/report?format=json",
        ):
            rejected = await client.get(endpoint)
            assert rejected.status_code == 502, rejected.text
            assert rejected.json()["detail"]["code"] == (
                "GOVERNANCE_AUDIT_INVALID"
            )


@pytest.mark.anyio
@pytest.mark.parametrize("tamper_kind", ("case-root", "item-case-binding"))
async def test_http_case_reads_and_reports_reject_root_or_item_binding_tampering(
    unconfigured_settings: Settings,
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    fake = FakeOnlineClient()
    root = tmp_path / "governance"
    settings = unconfigured_settings.model_copy(
        update={"gfm_governance_root": str(root)}
    )
    app = create_app(settings, gfm_governance_client=fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        uploaded = await client.post(
            "/api/v2/gfm/governance/artifacts",
            files={"file": ("tiny.zip", _bundle(), "application/zip")},
            data={"cleanSelfLoops": "false"},
        )
        assert uploaded.status_code == 201, uploaded.text
        artifact = uploaded.json()
        created_run = await client.post(
            "/api/v2/gfm/governance/runs",
            json={
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "protocol": "global",
                "artifactId": artifact["artifactId"],
                "datasetContentHash": artifact["datasetContentHash"],
                "graphVersionHash": artifact["graphVersionHash"],
                    "modelVersionId": fake.model_id,
                    "modelStateHash": fake.state_hash,
                    "topK": 1,
            },
        )
        assert created_run.status_code == 202, created_run.text
        run_id = created_run.json()["runId"]

        source_response = await client.post(
            "/api/v2/gfm/governance/cases",
            json={
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "runId": run_id,
                "title": "Source case",
                "description": "Owns the candidate.",
            },
        )
        assert source_response.status_code == 201, source_response.text
        source_id = source_response.json()["caseId"]
        item_response = await client.post(
            f"/api/v2/gfm/governance/cases/{source_id}/items",
            json={
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "targetType": "node",
                "targetId": "a",
                "note": "Case-bound candidate",
            },
        )
        assert item_response.status_code == 201, item_response.text

        target_id = source_id
        with sqlite3.connect(root / "governance.sqlite3") as connection:
            if tamper_kind == "case-root":
                connection.execute("DROP TRIGGER no_governance_case_update")
                connection.execute(
                    "UPDATE governance_cases SET title = ? WHERE case_id = ?",
                    ("Tampered title", source_id),
                )
            else:
                destination_response = await client.post(
                    "/api/v2/gfm/governance/cases",
                    json={
                        "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                        "runId": run_id,
                        "title": "Destination case",
                        "description": "Must not receive the candidate.",
                    },
                )
                assert destination_response.status_code == 201
                target_id = destination_response.json()["caseId"]
                connection.execute("DROP TRIGGER no_case_item_update")
                connection.execute(
                    "UPDATE case_items SET case_id = ? WHERE case_id = ?",
                    (target_id, source_id),
                )

        for endpoint in (
            f"/api/v2/gfm/governance/cases/{target_id}",
            "/api/v2/gfm/governance/cases",
            f"/api/v2/gfm/governance/cases/{target_id}/report?format=json",
        ):
            rejected = await client.get(endpoint)
            assert rejected.status_code == 502, rejected.text
            assert rejected.json()["detail"]["code"] == (
                "GOVERNANCE_AUDIT_INVALID"
            )
