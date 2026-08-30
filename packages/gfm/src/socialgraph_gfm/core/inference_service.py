"""Loopback-only HTTP service and durable core inference run store."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol, Sequence

from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.artifact_catalog import ArtifactCatalog, CapturedGraphLease
from socialgraph_gfm.core.governance import GovernanceFinding
from socialgraph_gfm.core.inference_contracts import (
    GfmRunRequest,
    GfmRunResult,
    GfmRunStatus,
    InternalCreateRunRequest,
    InternalCreateRunReceipt,
    MAX_INTERNAL_REQUEST_BYTES,
    MAX_INTERNAL_RESPONSE_BYTES,
    RunExecutionSnapshot,
    RunSuccessMarker,
    _EarlyHistoricalInternalCreateRunRequest,
    _HistoricalInternalCreateRunRequest,
    _HistoricalRunExecutionSnapshot,
    _HistoricalRunExecutionSnapshotRound1,
    _HistoricalRunExecutionSnapshotRound2,
    _PersistedCreateRunRequest,
    _PersistedRunExecutionSnapshot,
    _decode_persisted_create_run_request,
    _decode_persisted_execution_snapshot,
    _lease_identity_projection,
)
from socialgraph_gfm.core.safe_paths import reject_link_components, secure_existing_root
from socialgraph_gfm.core.serving_control import (
    CapturedServingControl,
    ServingControlStore,
)
from socialgraph_gfm.core.serving_registry import (
    CapturedModelLease,
    ServingModel,
    ServingRegistry,
)
from socialgraph_gfm.core.serving_head import CoreServingHead
from socialgraph_gfm.research.service import (
    ResearchServiceError,
    ResearchServingRuntime,
)
from socialgraph_gfm.global_model.service import (
    GlobalServiceError,
    GlobalServingRuntime,
)
from socialgraph_gfm.governance.errors import GovernanceServiceError

MAX_REQUEST_BYTES = MAX_INTERNAL_REQUEST_BYTES
MAX_RESPONSE_BYTES = MAX_INTERNAL_RESPONSE_BYTES
MAX_GOVERNANCE_RESPONSE_BYTES = 64 * 1024 * 1024


class _GovernanceRuntime(Protocol):
    def dispatch_get(self, path: str) -> dict[str, Any]: ...

    def dispatch_post(
        self,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _PUBLICATION_SEAM(_stage: str, _run_id: str) -> None:
    return


def _EXECUTION_SEAM(_stage: str, _run_id: str) -> None:
    return


@dataclass(frozen=True)
class RunLease:
    model: CapturedModelLease
    graph: CapturedGraphLease
    control: CapturedServingControl

    def materialize(self):
        model, checkpoint, calibrations, adapter_schema_hash = self.model.materialize()
        return (
            model,
            checkpoint,
            calibrations,
            adapter_schema_hash,
            self.graph.materialize(),
        )


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ServingControlStaleError(ValueError):
    """Capability/create expectation no longer matches the atomic operator control."""


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_atomic_bytes(path: Path) -> bytes:
    """Read through the short Windows replace/share-violation window."""

    path = reject_link_components(path)
    for attempt in range(20):
        try:
            return path.read_bytes()
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.005)
    raise AssertionError("bounded atomic read loop did not return")


def atomic_publish_session_token(path: str | Path) -> str:
    """Generate and atomically publish a high-entropy session secret with private ACL bits."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(48)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    published = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(token)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        if os.name == "nt":
            sid_result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            sid = sid_result.stdout.strip()
            if sid_result.returncode != 0 or not sid.startswith("S-"):
                raise OSError("failed to resolve current Windows SID")
            completed = subprocess.run(
                [
                    "icacls",
                    str(temporary),
                    "/inheritance:r",
                    "/grant:r",
                    f"*{sid}:(M)",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if completed.returncode != 0:
                raise OSError("failed to apply private Windows ACL to session token")
            _verify_windows_private_acl(temporary, sid)
            os.replace(temporary, destination)
            published = True
            _verify_windows_private_acl(destination, sid)
        else:
            os.replace(temporary, destination)
            published = True
            os.chmod(destination, 0o600)
    except Exception:
        if published:
            destination.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return token


def _verify_windows_private_acl(path: Path, sid: str) -> None:
    escaped = str(path).replace("'", "''")
    script = (
        f"$a=Get-Acl -LiteralPath '{escaped}';"
        "$a.Access|%{$s=$_.IdentityReference.Translate("
        "[System.Security.Principal.SecurityIdentifier]).Value;"
        "Write-Output ($s+'|'+$_.AccessControlType+'|'+$_.IsInherited)}"
    )
    environment = dict(os.environ)
    system_root = Path(environment.get("SystemRoot", r"C:\Windows"))
    # Host runtimes may prepend shadow copies of Microsoft.PowerShell.Security
    # whose type data conflicts with Windows PowerShell.  ACL verification is a
    # security boundary, so load only the OS-owned module inventory here.
    environment["PSModulePath"] = str(
        system_root / "System32" / "WindowsPowerShell" / "v1.0" / "Modules"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    rules = [line.split("|") for line in completed.stdout.splitlines() if line]
    if (
        completed.returncode != 0
        or not rules
        or any(len(rule) != 3 or rule[2] != "False" for rule in rules)
        or {rule[0] for rule in rules if rule[1] == "Allow"} != {sid}
        or any(rule[1] != "Allow" for rule in rules)
    ):
        raise OSError("session token Windows ACL is not private")


def _status_payload(
    *,
    run_id: str,
    request_hash: str,
    status: str,
    progress: int,
    created_at: datetime,
    updated_at: datetime,
    error_code: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-run-status/2.0",
        "runId": run_id,
        "requestHash": request_hash,
        "status": status,
        "progress": progress,
        "createdAt": created_at,
        "updatedAt": updated_at,
        "errorCode": error_code,
    }
    payload["stateHash"] = canonical_sha256(payload)
    return payload


class RunStore:
    """Immutable requests/results with atomic monotonic state transitions.

    Recovery policy is deterministic: terminal runs are preserved byte-for-byte;
    queued or running runs become failed with ``GFM_CORE_RUN_INTERRUPTED``. They are
    never automatically replayed, avoiding duplicate governance findings.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        registry: ServingRegistry,
        artifact_catalog: ArtifactCatalog,
        serving_control: ServingControlStore,
    ) -> None:
        if not isinstance(serving_control, ServingControlStore):
            raise TypeError("serving_control must be a ServingControlStore")
        lexical_root = reject_link_components(root)
        lexical_root.mkdir(parents=True, exist_ok=True)
        self.root = secure_existing_root(lexical_root)
        self.runs_root = self.root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.runs_root = secure_existing_root(self.runs_root)
        self.registry = registry
        self.artifact_catalog = artifact_catalog
        self.serving_control = serving_control
        self._lock = threading.RLock()
        self._recovery_diagnostics: list[dict[str, str]] = []
        self._recover()

    def _run_dir(self, run_id: str) -> Path:
        try:
            uuid.UUID(run_id)
        except ValueError as error:
            raise LookupError("run not found") from error
        return reject_link_components(self.runs_root / run_id)

    def _recover(self) -> None:
        with self._lock:
            for run_dir in sorted(self.runs_root.iterdir(), key=lambda item: item.name):
                try:
                    reject_link_components(run_dir)
                except ValueError:
                    self._diagnose(run_dir.name, "GFM_CORE_RUN_PATH_UNSAFE")
                    continue
                if run_dir.is_symlink() or not run_dir.is_dir():
                    continue
                run_id = run_dir.name
                try:
                    uuid.UUID(run_id)
                except ValueError:
                    self._diagnose(run_id, "GFM_CORE_RUN_ID_INVALID")
                    continue
                if not (run_dir / "manifest.json").is_file():
                    self._diagnose(run_id, "GFM_CORE_RUN_MANIFEST_MISSING")
                    continue
                try:
                    request, status, snapshot = self._load_core(run_id)
                    result_path = run_dir / "result.json"
                    marker_path = run_dir / "success.json"
                    if result_path.is_file():
                        result = GfmRunResult.model_validate_json(_read_atomic_bytes(result_path))
                        self._validate_result_binding(result, request, snapshot)
                        marker = self._success_marker(snapshot, result)
                        if marker_path.is_file():
                            observed_marker = RunSuccessMarker.model_validate_json(
                                _read_atomic_bytes(marker_path)
                            )
                            if observed_marker != marker:
                                raise ValueError("success marker does not bind result")
                        else:
                            _atomic_json(
                                marker_path,
                                marker.model_dump(mode="json", by_alias=True),
                            )
                        if status.status != "succeeded":
                            succeeded = GfmRunStatus.model_validate(
                                _status_payload(
                                    run_id=run_id,
                                    request_hash=snapshot.request_hash,
                                    status="succeeded",
                                    progress=100,
                                    created_at=status.created_at,
                                    updated_at=result.completed_at,
                                )
                            )
                            _atomic_json(
                                run_dir / "state.json",
                                succeeded.model_dump(mode="json", by_alias=True),
                            )
                    elif marker_path.exists() or status.status == "succeeded":
                        raise ValueError("terminal success is missing its result")
                    elif status.status in {"queued", "running"}:
                        failed = GfmRunStatus.model_validate(
                            _status_payload(
                                run_id=status.run_id,
                                request_hash=status.request_hash,
                                status="failed",
                                progress=100,
                                created_at=status.created_at,
                                updated_at=_utcnow(),
                                error_code="GFM_CORE_RUN_INTERRUPTED",
                            )
                        )
                        _atomic_json(
                            run_dir / "state.json",
                            failed.model_dump(mode="json", by_alias=True),
                        )
                except (OSError, ValueError, ValidationError):
                    self._diagnose(run_id, "GFM_CORE_RUN_RECOVERY_INVALID")

    def _diagnose(self, run_id: str, code: str) -> None:
        self._recovery_diagnostics.append({"runId": run_id, "code": code})

    def recovery_diagnostics(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(item) for item in self._recovery_diagnostics)

    def _execution_snapshot(
        self,
        *,
        run_id: str,
        envelope: InternalCreateRunRequest,
        model: ServingModel,
        lease: RunLease,
        adapter_schema_hash: str,
        created_at: datetime,
    ) -> RunExecutionSnapshot:
        graph = envelope.graph_reference
        calibration_bytes = dict(lease.model.calibration_snapshots)
        calibration_identities = [
            {
                "entityType": binding.entity_type,
                "calibrationVersion": binding.calibration_version,
                "method": binding.calibration_method,
                "calibrationArtifactHash": binding.calibration_artifact_hash,
                "calibrationProtocolHash": binding.calibration_protocol_hash,
                "confidenceKind": binding.confidence_kind,
                "adapterDomain": binding.adapter_domain,
                "adapterSchemaHash": binding.adapter_schema_hash,
                "adapterStateHash": binding.adapter_state_hash,
                "featureContractHash": binding.graph_feature_contract_hash,
                "sha256": hashlib.sha256(calibration_bytes[binding.entity_type]).hexdigest(),
            }
            for binding in sorted(
                model.task_head(envelope.request.task_id).calibrations,
                key=lambda item: item.entity_type,
            )
        ]
        payload: dict[str, object] = {
            "schemaVersion": "socialgraph-fm.core-run-execution-snapshot/2.2",
            "runId": run_id,
            "requestHash": envelope.request_hash,
            "controlSourceSha256": hashlib.sha256(lease.control.control_snapshot).hexdigest(),
            "controlHash": lease.control.document.control_hash,
            "controlGeneration": lease.control.document.generation,
            "registryHash": lease.model.registry_hash,
            "registrySourceSha256": lease.model.registry_source_sha256,
            "registryGeneration": lease.model.registry_generation,
            "modelVersionId": model.model_version_id,
            "modelVersionHash": model.model_version_hash,
            "checkpointSha256": model.checkpoint.sha256,
            "servingManifestSha256": model.checkpoint.serving_manifest_sha256,
            "adapterSchemaHash": adapter_schema_hash,
            "calibrationIdentities": calibration_identities,
            "calibrationSetHash": canonical_sha256(calibration_identities),
            "taskId": envelope.request.task_id,
            "graphVersionId": graph.graph_version_id,
            "sourceGraphFactHash": graph.source_graph_fact_hash,
            "graphVersionHash": graph.graph_version_hash,
            "artifactId": graph.artifact_id,
            "artifactHash": graph.artifact_hash,
            "artifactCatalogSha256": lease.graph.catalog_sha256,
            "artifactCatalogHash": lease.control.catalog_hash,
            "artifactCatalogGeneration": lease.graph.catalog_generation,
            "bundleSha256": graph.bundle_sha256,
            "graphSchemaVersion": graph.graph_schema_version,
            "featureContractHash": graph.feature_contract_hash,
            "nodeCount": graph.node_count,
            "edgeCount": graph.edge_count,
            "createdAt": created_at,
        }
        payload["snapshotHash"] = canonical_sha256(payload)
        return RunExecutionSnapshot.model_validate(payload)

    @staticmethod
    def _success_marker(
        snapshot: _PersistedRunExecutionSnapshot, result: GfmRunResult
    ) -> RunSuccessMarker:
        payload: dict[str, object] = {
            "schemaVersion": "socialgraph-fm.core-run-success-marker/2.0",
            "runId": result.run_id,
            "requestHash": result.request_hash,
            "snapshotHash": snapshot.snapshot_hash,
            "resultHash": result.result_hash,
            "completedAt": result.completed_at,
        }
        payload["markerHash"] = canonical_sha256(payload)
        return RunSuccessMarker.model_validate(payload)

    def _load_core(
        self, run_id: str
    ) -> tuple[
        _PersistedCreateRunRequest,
        GfmRunStatus,
        _PersistedRunExecutionSnapshot,
    ]:
        run_dir = self._run_dir(run_id)
        request = _decode_persisted_create_run_request(_read_atomic_bytes(run_dir / "request.json"))
        status = GfmRunStatus.model_validate_json(_read_atomic_bytes(run_dir / "state.json"))
        snapshot = _decode_persisted_execution_snapshot(
            _read_atomic_bytes(run_dir / "manifest.json")
        )
        snapshot_type = type(snapshot)
        if snapshot_type in {
            _HistoricalRunExecutionSnapshotRound1,
            _HistoricalRunExecutionSnapshotRound2,
        }:
            versions_match = isinstance(
                request, _EarlyHistoricalInternalCreateRunRequest
            ) and not isinstance(request, _HistoricalInternalCreateRunRequest)
        elif snapshot_type is _HistoricalRunExecutionSnapshot:
            versions_match = (
                snapshot.schema_version.endswith("/2.0")
                and isinstance(request, _HistoricalInternalCreateRunRequest)
            ) or (
                snapshot.schema_version.endswith("/2.1")
                and isinstance(request, InternalCreateRunRequest)
            )
        else:
            versions_match = isinstance(request, InternalCreateRunRequest)
        if not versions_match:
            raise ValueError("persisted request/snapshot versions are inconsistent")
        if isinstance(request, InternalCreateRunRequest):
            if not isinstance(snapshot, (RunExecutionSnapshot, _HistoricalRunExecutionSnapshot)):
                raise ValueError("controlled request requires a controlled snapshot")
            expected = request.expected_serving_control
            if (
                expected.control_hash != snapshot.control_hash
                or expected.control_generation != snapshot.control_generation
                or expected.registry_hash != snapshot.registry_hash
                or expected.registry_generation != snapshot.registry_generation
                or expected.catalog_hash != snapshot.artifact_catalog_hash
                or expected.catalog_generation != snapshot.artifact_catalog_generation
                or expected.model_version_hash != snapshot.model_version_hash
            ):
                raise ValueError("persisted serving-control expectation does not match snapshot")
        graph = request.graph_reference
        if (
            status.run_id != run_id
            or snapshot.run_id != run_id
            or request.request_hash != snapshot.request_hash
            or status.request_hash != snapshot.request_hash
            or request.request.task_id != snapshot.task_id
            or request.request.model_version_id != snapshot.model_version_id
            or graph.graph_version_id != snapshot.graph_version_id
            or graph.source_graph_fact_hash != snapshot.source_graph_fact_hash
            or graph.graph_version_hash != snapshot.graph_version_hash
            or graph.artifact_id != snapshot.artifact_id
            or graph.artifact_hash != snapshot.artifact_hash
            or graph.bundle_sha256 != snapshot.bundle_sha256
            or graph.graph_schema_version != snapshot.graph_schema_version
            or graph.feature_contract_hash != snapshot.feature_contract_hash
            or graph.node_count != snapshot.node_count
            or graph.edge_count != snapshot.edge_count
        ):
            raise ValueError("persisted run does not match execution snapshot binding")
        return request, status, snapshot

    @staticmethod
    def _validate_result_binding(
        result: GfmRunResult,
        request: _PersistedCreateRunRequest,
        snapshot: _PersistedRunExecutionSnapshot,
    ) -> None:
        if (
            result.run_id != snapshot.run_id
            or result.request_hash != snapshot.request_hash
            or result.task_id != snapshot.task_id
            or result.graph_version_id != snapshot.graph_version_id
            or result.graph_version_hash != snapshot.graph_version_hash
            or result.model_version_id != snapshot.model_version_id
            or result.model_version_hash != snapshot.model_version_hash
            or request.request_hash != snapshot.request_hash
        ):
            raise ValueError("result does not match immutable execution snapshot binding")

    def create(self, envelope: InternalCreateRunRequest) -> GfmRunStatus | InternalCreateRunReceipt:
        graph = envelope.graph_reference
        request = envelope.request
        expectation = envelope.expected_serving_control
        control_snapshot = self.serving_control.capture()
        actual = (
            control_snapshot.document.control_hash,
            control_snapshot.document.generation,
            control_snapshot.registry_hash,
            control_snapshot.registry_document.generation,
            control_snapshot.catalog_hash,
            control_snapshot.catalog_document.generation,
        )
        expected = (
            expectation.control_hash,
            expectation.control_generation,
            expectation.registry_hash,
            expectation.registry_generation,
            expectation.catalog_hash,
            expectation.catalog_generation,
        )
        if actual != expected:
            raise ServingControlStaleError("serving control expectation is stale")
        model_lease = self.registry.acquire_model_lease(
            request.model_version_id,
            request.task_id,
            registry_snapshot=control_snapshot.registry_snapshot,
        )
        graph_lease = self.artifact_catalog.acquire_graph_lease(
            graph, catalog_snapshot=control_snapshot.catalog_snapshot
        )
        model, _checkpoint, _calibrations, _primary_adapter_schema_hash = model_lease.materialize()
        entity_type = CoreServingHead._requested_entity_type(request)
        entity_binding = model.task_head(request.task_id).calibration(entity_type)
        adapter_schema_hash = entity_binding.adapter_schema_hash
        if model.model_version_hash != expectation.model_version_hash:
            raise ServingControlStaleError("selected model version expectation is stale")
        if request.task_id not in model.tasks:
            raise ValueError("model does not support requested task")
        if graph.graph_schema_version not in model.graph_schema_versions:
            raise ValueError("model does not support graph schema")
        if graph.feature_contract_hash != entity_binding.graph_feature_contract_hash:
            raise ValueError("graph feature contract does not match model")
        if graph.node_count > model.max_nodes or graph.edge_count > model.max_edges:
            raise ValueError("graph exceeds registered model limits")
        graph_lease.materialize()
        self.serving_control.accept(control_snapshot)
        lease = RunLease(model=model_lease, graph=graph_lease, control=control_snapshot)
        run_id = str(uuid.uuid4())
        run_dir = self._run_dir(run_id)
        now = _utcnow()
        state = GfmRunStatus.model_validate(
            _status_payload(
                run_id=run_id,
                request_hash=envelope.request_hash,
                status="queued",
                progress=0,
                created_at=now,
                updated_at=now,
            )
        )
        snapshot = self._execution_snapshot(
            run_id=run_id,
            envelope=envelope,
            model=model,
            lease=lease,
            adapter_schema_hash=adapter_schema_hash,
            created_at=now,
        )
        with self._lock:
            run_dir.mkdir(parents=False, exist_ok=False)
            try:
                _atomic_json(
                    run_dir / "request.json",
                    envelope.model_dump(mode="json", by_alias=True),
                )
                _atomic_json(
                    run_dir / "state.json",
                    state.model_dump(mode="json", by_alias=True),
                )
                _atomic_json(
                    run_dir / "manifest.json",
                    snapshot.model_dump(mode="json", by_alias=True),
                )
            except Exception:
                for child in run_dir.iterdir():
                    child.unlink(missing_ok=True)
                run_dir.rmdir()
                raise
        threading.Thread(
            target=self._execute,
            args=(run_id, envelope, model, lease),
            daemon=True,
            name=f"gfm-run-{run_id[:8]}",
        ).start()
        receipt_payload: dict[str, object] = {
            "schemaVersion": "socialgraph-fm.core-internal-create-run-receipt/2.0",
            "status": state.model_dump(mode="python", by_alias=True),
            "executionSnapshot": snapshot.model_dump(mode="python", by_alias=True),
            "leaseIdentityHash": canonical_sha256(_lease_identity_projection(snapshot)),
        }
        receipt_payload["receiptHash"] = canonical_sha256(receipt_payload)
        return InternalCreateRunReceipt.model_validate(receipt_payload)

    def _write_transition(
        self,
        run_id: str,
        *,
        expected: set[str],
        status: str,
        progress: int,
        error_code: str | None = None,
    ) -> GfmRunStatus:
        with self._lock:
            current = self.get(run_id)
            if current.status not in expected:
                raise RuntimeError("run state transition is not monotonic")
            updated = GfmRunStatus.model_validate(
                _status_payload(
                    run_id=current.run_id,
                    request_hash=current.request_hash,
                    status=status,
                    progress=progress,
                    created_at=current.created_at,
                    updated_at=_utcnow(),
                    error_code=error_code,
                )
            )
            _atomic_json(
                self._run_dir(run_id) / "state.json",
                updated.model_dump(mode="json", by_alias=True),
            )
            return updated

    def _production_execute(
        self,
        request: GfmRunRequest,
        lease: RunLease,
    ) -> Sequence[GovernanceFinding]:
        model, checkpoint, calibrations, _adapter_schema_hash, bundle = lease.materialize()
        return CoreServingHead().execute(request, bundle, model, checkpoint, calibrations)

    def _execute(
        self,
        run_id: str,
        envelope: InternalCreateRunRequest,
        model: ServingModel,
        lease: RunLease,
    ) -> None:
        try:
            self._write_transition(run_id, expected={"queued"}, status="running", progress=10)
            raw_findings: Sequence[GovernanceFinding | dict[str, object]]
            _EXECUTION_SEAM("before-production-materialize", run_id)
            raw_findings = self._production_execute(envelope.request, lease)
            findings = tuple(
                item
                if isinstance(item, GovernanceFinding)
                else GovernanceFinding.model_validate_json(
                    json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                )
                for item in raw_findings
            )
            now = _utcnow()
            result_payload: dict[str, object] = {
                "schemaVersion": "socialgraph-fm.core-run-result/2.0",
                "runId": run_id,
                "requestHash": envelope.request_hash,
                "taskId": envelope.request.task_id,
                "graphVersionId": envelope.graph_reference.graph_version_id,
                "graphVersionHash": envelope.graph_reference.graph_version_hash,
                "modelVersionId": model.model_version_id,
                "modelVersionHash": model.model_version_hash,
                "findings": [item.model_dump(mode="json", by_alias=True) for item in findings],
                "completedAt": now,
            }
            result_payload["resultHash"] = canonical_sha256(result_payload)
            validation_payload = dict(result_payload)
            validation_payload["findings"] = findings
            result = GfmRunResult.model_validate(validation_payload)
            with self._lock:
                persisted_request, current, snapshot = self._load_core(run_id)
                if current.status != "running":
                    return
                self._validate_result_binding(result, persisted_request, snapshot)
                _atomic_json(
                    self._run_dir(run_id) / "result.json",
                    result.model_dump(mode="json", by_alias=True),
                )
                marker = self._success_marker(snapshot, result)
                _atomic_json(
                    self._run_dir(run_id) / "success.json",
                    marker.model_dump(mode="json", by_alias=True),
                )
                _PUBLICATION_SEAM("after-success-marker", run_id)
                self._write_transition(
                    run_id,
                    expected={"running"},
                    status="succeeded",
                    progress=100,
                )
        except Exception:
            try:
                self._write_transition(
                    run_id,
                    expected={"queued", "running"},
                    status="failed",
                    progress=100,
                    error_code="GFM_CORE_EXECUTION_FAILED",
                )
            except (LookupError, RuntimeError, ValueError):
                pass

    def get(self, run_id: str) -> GfmRunStatus:
        run_dir = self._run_dir(run_id)
        if not (run_dir / "state.json").is_file():
            raise LookupError("run not found")
        _, status, _ = self._load_core(run_id)
        return status

    def get_result(self, run_id: str) -> GfmRunResult:
        status = self.get(run_id)
        if status.status != "succeeded":
            raise LookupError("result is not ready")
        path = self._run_dir(run_id) / "result.json"
        result = GfmRunResult.model_validate_json(_read_atomic_bytes(path))
        request, rebound_status, snapshot = self._load_core(run_id)
        if rebound_status != status:
            raise ValueError("run state changed while reading result")
        self._validate_result_binding(result, request, snapshot)
        marker = RunSuccessMarker.model_validate_json(
            _read_atomic_bytes(self._run_dir(run_id) / "success.json")
        )
        if marker != self._success_marker(snapshot, result):
            raise ValueError("success marker does not match immutable result")
        return result


class InferenceRuntime:
    def __init__(
        self,
        store: RunStore,
        registry: ServingRegistry,
        serving_control: ServingControlStore,
    ) -> None:
        if not isinstance(serving_control, ServingControlStore):
            raise TypeError("serving_control must be a ServingControlStore")
        if store.registry is not registry:
            raise ValueError("runtime store and capability registry must be identical")
        self.store = store
        self.registry = registry
        self.serving_control = serving_control
        if store.serving_control is not serving_control:
            raise ValueError("runtime store and serving control must be identical")

    def capabilities(self) -> dict[str, object]:
        validated: dict[str, object] = {}

        def validate(control: CapturedServingControl) -> None:
            validated.update(
                self.registry.capabilities(registry_snapshot=control.registry_snapshot)
            )

        control = self.serving_control.acquire(validate)
        payload = validated
        payload.update(
            {
                "controlHash": control.document.control_hash,
                "controlGeneration": control.document.generation,
                "catalogHash": control.catalog_hash,
                "catalogGeneration": control.catalog_document.generation,
            }
        )
        return payload

    def health(self) -> dict[str, object]:
        return {
            "schemaVersion": "socialgraph-fm.core-internal-health/2.0",
            "ok": True,
            "recoveryIssueCount": len(self.store.recovery_diagnostics()),
        }


class _LoopbackServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        address: tuple[str, int],
        handler,
        *,
        token: str,
        runtime,
        research_runtime: ResearchServingRuntime | None,
        global_model_runtime: GlobalServingRuntime | None,
        governance_runtime: _GovernanceRuntime | None,
    ):
        self.session_token = token
        self.runtime = runtime
        self.research_runtime = research_runtime
        self.global_model_runtime = global_model_runtime
        self.governance_runtime = governance_runtime
        super().__init__(address, handler)

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            if self.research_runtime is not None:
                self.research_runtime.close()
            if self.global_model_runtime is not None:
                self.global_model_runtime.close()
            if self.governance_runtime is not None:
                self.governance_runtime.close()


class _Handler(BaseHTTPRequestHandler):
    server: _LoopbackServer
    server_version = "SocialGraphGFM/2"
    sys_version = ""

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(self, status: int, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        response_limit = (
            MAX_GOVERNANCE_RESPONSE_BYTES
            if self.path.startswith("/internal/governance/")
            else MAX_RESPONSE_BYTES
        )
        if len(encoded) > response_limit:
            status = 500
            encoded = b'{"error":{"code":"GFM_CORE_RESPONSE_TOO_LARGE"}}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _authorized(self) -> bool:
        try:
            client = ipaddress.ip_address(self.client_address[0])
        except ValueError:
            self._send(403, {"error": {"code": "GFM_CORE_LOOPBACK_ONLY"}})
            return False
        expected_host = f"127.0.0.1:{self.server.server_address[1]}"
        forwarded = any(
            name.lower() == "forwarded" or name.lower().startswith("x-forwarded-")
            for name in self.headers.keys()
        )
        if not client.is_loopback or self.headers.get("Host") != expected_host or forwarded:
            self._send(403, {"error": {"code": "GFM_CORE_LOOPBACK_ONLY"}})
            return False
        authorization = self.headers.get("Authorization", "")
        prefix = "Bearer "
        supplied = authorization[len(prefix) :] if authorization.startswith(prefix) else ""
        if not hmac.compare_digest(supplied, self.server.session_token):
            self._send(401, {"error": {"code": "GFM_CORE_UNAUTHORIZED"}})
            return False
        return True

    def _read_json(self) -> dict[str, object] | None:
        if self.headers.get_content_type() != "application/json":
            self._send(415, {"error": {"code": "GFM_CORE_JSON_REQUIRED"}})
            return None
        if self.headers.get("Transfer-Encoding"):
            self._send(400, {"error": {"code": "GFM_CORE_INVALID_LENGTH"}})
            return None
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if length < 1 or length > MAX_REQUEST_BYTES:
            self._send(
                413 if length > MAX_REQUEST_BYTES else 400,
                {"error": {"code": "GFM_CORE_INVALID_LENGTH"}},
            )
            return None
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send(400, {"error": {"code": "GFM_CORE_INVALID_JSON"}})
            return None
        if not isinstance(payload, dict):
            self._send(400, {"error": {"code": "GFM_CORE_INVALID_JSON"}})
            return None
        return payload

    def _read_optional_json(self) -> dict[str, object] | None:
        """Read a JSON object when present, allowing explicitly bodyless commands."""

        if self.headers.get("Transfer-Encoding"):
            self._send(400, {"error": {"code": "GFM_CORE_INVALID_LENGTH"}})
            raise ValueError("invalid transfer encoding")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return None
        try:
            length = int(raw_length)
        except ValueError as error:
            self._send(400, {"error": {"code": "GFM_CORE_INVALID_LENGTH"}})
            raise ValueError("invalid content length") from error
        if length == 0:
            return None
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._send(
                413 if length > MAX_REQUEST_BYTES else 400,
                {"error": {"code": "GFM_CORE_INVALID_LENGTH"}},
            )
            raise ValueError("invalid content length")
        if self.headers.get_content_type() != "application/json":
            self._send(415, {"error": {"code": "GFM_CORE_JSON_REQUIRED"}})
            raise ValueError("JSON content type required")
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self._send(400, {"error": {"code": "GFM_CORE_INVALID_JSON"}})
            raise ValueError("invalid JSON") from error
        if not isinstance(payload, dict):
            self._send(400, {"error": {"code": "GFM_CORE_INVALID_JSON"}})
            raise ValueError("JSON body is not an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path.startswith("/internal/governance/"):
            if self.server.governance_runtime is None:
                self._send(
                    503,
                    {"error": {"code": "GFM_GOVERNANCE_MODEL_NOT_INSTALLED"}},
                )
                return
            try:
                payload = self.server.governance_runtime.dispatch_get(self.path)
            except GovernanceServiceError as error:
                self._send(error.status, {"error": {"code": error.code}})
                return
            except (OSError, ValueError, ValidationError):
                self._send(
                    409,
                    {"error": {"code": "GFM_GOVERNANCE_PERSISTED_STATE_INVALID"}},
                )
                return
            self._send(200, payload)
            return
        if self.path.startswith("/internal/global-model/"):
            if self.server.global_model_runtime is None:
                self._send(
                    503,
                    {"error": {"code": "GFM_GLOBAL_MODEL_NOT_INSTALLED"}},
                )
                return
            try:
                payload = self.server.global_model_runtime.dispatch_get(self.path)
            except GlobalServiceError as error:
                self._send(error.status, {"error": {"code": error.code}})
                return
            except (OSError, ValueError, ValidationError):
                self._send(
                    409,
                    {"error": {"code": "GFM_GLOBAL_MODEL_PERSISTED_STATE_INVALID"}},
                )
                return
            self._send(200, payload)
            return
        if self.path.startswith("/internal/research/"):
            if self.server.research_runtime is None:
                self._send(
                    503,
                    {"error": {"code": "GFM_RESEARCH_MODEL_NOT_INSTALLED"}},
                )
                return
            try:
                payload = self.server.research_runtime.dispatch_get(self.path)
            except ResearchServiceError as error:
                self._send(error.status, {"error": {"code": error.code}})
                return
            except (OSError, ValueError, ValidationError):
                self._send(
                    409,
                    {"error": {"code": "GFM_RESEARCH_PERSISTED_STATE_INVALID"}},
                )
                return
            self._send(200, payload)
            return
        if self.path == "/internal/core/health":
            self._send(200, self.server.runtime.health())
            return
        if self.path == "/internal/core/capabilities":
            try:
                self._send(200, self.server.runtime.capabilities())
            except (OSError, ValueError, ValidationError):
                self._send(503, {"error": {"code": "GFM_CORE_REGISTRY_INVALID"}})
            return
        prefix = "/internal/core/runs/"
        if self.path.startswith(prefix):
            suffix = self.path[len(prefix) :]
            result = suffix.endswith("/result")
            run_id = suffix[:-7] if result else suffix
            if "/" in run_id:
                self._send(404, {"error": {"code": "GFM_CORE_NOT_FOUND"}})
                return
            try:
                record = (
                    self.server.runtime.store.get_result(run_id)
                    if result
                    else self.server.runtime.store.get(run_id)
                )
            except LookupError as error:
                code = "GFM_CORE_RESULT_NOT_READY" if "not ready" in str(error) else "GFM_CORE_RUN_NOT_FOUND"
                self._send(
                    409 if code == "GFM_CORE_RESULT_NOT_READY" else 404, {"error": {"code": code}}
                )
                return
            except (OSError, ValueError, ValidationError):
                self._send(409, {"error": {"code": "GFM_CORE_PERSISTED_STATE_INVALID"}})
                return
            self._send(200, record.model_dump(mode="json", by_alias=True))
            return
        self._send(404, {"error": {"code": "GFM_CORE_NOT_FOUND"}})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path.startswith("/internal/governance/"):
            if self.server.governance_runtime is None:
                self._send(
                    503,
                    {"error": {"code": "GFM_GOVERNANCE_MODEL_NOT_INSTALLED"}},
                )
                return
            try:
                payload = self._read_optional_json()
            except ValueError:
                return
            try:
                response = self.server.governance_runtime.dispatch_post(self.path, payload)
            except GovernanceServiceError as error:
                self._send(error.status, {"error": {"code": error.code}})
                return
            except (OSError, ValueError, ValidationError):
                self._send(
                    409,
                    {"error": {"code": "GFM_GOVERNANCE_PERSISTED_STATE_INVALID"}},
                )
                return
            accepted = self.path == "/internal/governance/runs" or self.path.endswith(
                "/retry"
            )
            self._send(202 if accepted else 200, response)
            return
        if self.path.startswith("/internal/global-model/"):
            if self.server.global_model_runtime is None:
                self._send(
                    503,
                    {"error": {"code": "GFM_GLOBAL_MODEL_NOT_INSTALLED"}},
                )
                return
            payload = self._read_json()
            if payload is None:
                return
            try:
                response = self.server.global_model_runtime.dispatch_post(self.path, payload)
            except GlobalServiceError as error:
                self._send(error.status, {"error": {"code": error.code}})
                return
            except (OSError, ValueError, ValidationError):
                self._send(
                    409,
                    {"error": {"code": "GFM_GLOBAL_MODEL_PERSISTED_STATE_INVALID"}},
                )
                return
            self._send(202 if self.path == "/internal/global-model/runs" else 200, response)
            return
        if self.path.startswith("/internal/research/"):
            if self.server.research_runtime is None:
                self._send(
                    503,
                    {"error": {"code": "GFM_RESEARCH_MODEL_NOT_INSTALLED"}},
                )
                return
            payload = self._read_json()
            if payload is None:
                return
            try:
                response = self.server.research_runtime.dispatch_post(self.path, payload)
            except ResearchServiceError as error:
                self._send(error.status, {"error": {"code": error.code}})
                return
            except (OSError, ValueError, ValidationError):
                self._send(
                    409,
                    {"error": {"code": "GFM_RESEARCH_PERSISTED_STATE_INVALID"}},
                )
                return
            self._send(202 if self.path == "/internal/research/runs" else 200, response)
            return
        if self.path != "/internal/core/runs":
            self._send(404, {"error": {"code": "GFM_CORE_NOT_FOUND"}})
            return
        payload = self._read_json()
        if payload is None:
            return
        try:
            envelope = InternalCreateRunRequest.model_validate(payload)
            state = self.server.runtime.store.create(envelope)
        except ValidationError:
            self._send(422, {"error": {"code": "GFM_CORE_REQUEST_INVALID"}})
            return
        except LookupError:
            self._send(409, {"error": {"code": "GFM_CORE_MODEL_UNAVAILABLE"}})
            return
        except ServingControlStaleError:
            self._send(409, {"error": {"code": "GFM_CORE_SERVING_CONTROL_STALE"}})
            return
        except (OSError, ValueError):
            self._send(409, {"error": {"code": "GFM_CORE_COMPATIBILITY_REJECTED"}})
            return
        self._send(202, state.model_dump(mode="json", by_alias=True))


def create_server(
    host: str,
    port: int,
    *,
    token: str,
    runtime: InferenceRuntime,
    research_runtime: ResearchServingRuntime | None = None,
    global_model_runtime: GlobalServingRuntime | None = None,
    governance_runtime: _GovernanceRuntime | None = None,
) -> ThreadingHTTPServer:
    if host != "127.0.0.1":
        raise ValueError("GFM service bind host must be literal loopback 127.0.0.1")
    if not (0 <= port <= 65535):
        raise ValueError("port is outside valid range")
    if len(token) < 64:
        raise ValueError("session token does not meet the entropy floor")
    return _LoopbackServer(
        (host, port),
        _Handler,
        token=token,
        runtime=runtime,
        research_runtime=research_runtime,
        global_model_runtime=global_model_runtime,
        governance_runtime=governance_runtime,
    )


__all__ = [
    "InferenceRuntime",
    "RunStore",
    "atomic_publish_session_token",
    "create_server",
]
