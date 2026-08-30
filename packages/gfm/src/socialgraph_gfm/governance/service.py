"""Durable loopback runtime for true Governance Global online inference."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_sha256, file_sha256

from .adaptation import (
    ADAPTATION_CODE_HASH,
    AdaptationBinding,
    AdaptationComparison,
    AdaptationComparisonV2,
    FittedReviewPolicy,
    FittedReviewPolicyV2,
    LabelEvidence,
    TargetLabelSet,
    TargetPackageReceipt,
    TargetReviewPolicy,
    TargetReviewPolicyV2,
    build_target_label_set,
    fit_target_review_policy,
    fit_target_review_policy_v2,
    validate_sidecar_against_fused_graph,
)
from .analytics import DerivedAnalytics, derive_analytics
from .contracts import (
    ARTIFACT_ID_PATTERN,
    INPUT_SCHEMA_VERSION,
    MAX_EVIDENCE_EDGES,
    MAX_EVIDENCE_NODES,
    MAX_NODES,
    MAX_PREVIEW_EDGES,
    MAX_PREVIEW_NODES,
    MAX_RELATION_ROWS,
    MODALITIES,
    RUN_ID_PATTERN,
    SCHEMA_VERSION,
)
from .errors import (
    GovernanceAdaptationPolicyNotReady,
    GovernanceInvalid,
    GovernanceNotFound,
    GovernanceNotReady,
    GovernanceServiceError,
    GovernanceUnavailable,
)
from .inference import (
    InferenceCancelled,
    LoadedGlobalModel,
    OnlineInferenceOutputs,
    load_global_model,
    run_online_inference,
)
from .materialize import (
    BundleValidationError,
    OnlineInferenceData,
    load_materialized_artifact,
    materialize_bundle,
)
from .projection import ProjectionRequest, select_projection
from .target_tasks import TargetLabelSetV2

_HASH = re.compile(r"^[0-9a-f]{64}$")
_TARGET_TASK_REGISTRATION = re.compile(r"^target-task-[0-9a-f]{32}$")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_NPZ_BYTES = 512 * 1024 * 1024
_TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}
_WINDOWS_SHARING_ERRORS = frozenset({5, 32, 33, 1224})
_WINDOWS_SHARING_DELAYS = (0.002, 0.01, 0.025, 0.05)
_PUBLISHABLE_ADAPTATION_LAMBDAS = frozenset({0.25, 0.5, 1.0})
_LIMITATIONS = (
    "Scores are analyst-facing risk candidates and never automatic enforcement decisions.",
    "Calibration is validation-derived; unverified external graphs may be affected by domain shift.",
    "Factual relations explain the graph input but do not prove coordination or intent.",
    "Potential links are bounded same-community leads, not factual or predicted future edges.",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _with_hash(payload: dict[str, Any], field: str) -> dict[str, Any]:
    payload[field] = canonical_sha256(payload)
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, maximum: int = _MAX_JSON_BYTES) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= maximum:
        raise ValueError(f"invalid SocialGraph-FM Governance JSON artifact: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"SocialGraph-FM Governance JSON artifact must be an object: {path.name}")
    return payload


def _read_json_with_windows_retry(
    path: Path, *, maximum: int = _MAX_JSON_BYTES
) -> dict[str, Any]:
    attempt = 0
    while True:
        try:
            return _read_json(path, maximum=maximum)
        except OSError as error:
            code = getattr(error, "winerror", None)
            access_denied_limit = 2 if code == 5 else len(_WINDOWS_SHARING_DELAYS)
            if (
                os.name != "nt"
                or code not in _WINDOWS_SHARING_ERRORS
                or attempt >= access_denied_limit
                or not path.exists()
                or path.is_symlink()
                or not path.is_file()
                or not os.access(path, os.R_OK)
            ):
                raise
            time.sleep(_WINDOWS_SHARING_DELAYS[attempt])
            attempt += 1


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x+b") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_npz(path: Path, expected_hash: str) -> dict[str, np.ndarray]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > _MAX_NPZ_BYTES
        or file_sha256(path) != expected_hash
    ):
        raise ValueError("persisted SocialGraph-FM Governance NPZ identity is invalid")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("persisted SocialGraph-FM Governance NPZ contains an object array")
    return arrays


def _page(query: Mapping[str, Sequence[str]], *, maximum: int = 10_000) -> tuple[int, int]:
    try:
        offset = int(query.get("offset", ("0",))[0])
        limit = int(query.get("limit", ("100",))[0])
    except (TypeError, ValueError, IndexError) as exc:
        raise GovernanceInvalid from exc
    if offset < 0 or not 1 <= limit <= maximum:
        raise GovernanceInvalid
    return offset, limit


def _projection_request(
    query: Mapping[str, Sequence[str]],
) -> ProjectionRequest | None:
    if "preset" not in query:
        return None
    allowed = {
        "preset",
        "nodeBudget",
        "edgeBudget",
        "relation",
        "anchorNodeId",
        "groupBudget",
    }
    if set(query) - allowed:
        raise GovernanceInvalid

    def one(name: str) -> str | None:
        values = query.get(name)
        if values is None:
            return None
        if len(values) != 1 or not values[0]:
            raise GovernanceInvalid
        return values[0]

    payload: dict[str, Any] = {"preset": one("preset")}
    for name in ("nodeBudget", "edgeBudget", "groupBudget"):
        raw = one(name)
        if raw is not None:
            try:
                payload[name] = int(raw)
            except ValueError as error:
                raise GovernanceInvalid from error
    relation = one("relation")
    if relation is not None:
        payload["relation"] = relation
    payload["anchorNodeIds"] = tuple(query.get("anchorNodeId", ()))
    try:
        return ProjectionRequest.model_validate(payload)
    except ValidationError as error:
        raise GovernanceInvalid from error


def _risk_band(score: float, threshold: float) -> str:
    if score >= threshold:
        return "high"
    if score >= max(0.0, threshold - 0.15):
        return "review"
    return "low"


def _rank_order(scores: np.ndarray, node_ids: Sequence[str]) -> np.ndarray:
    return np.asarray(
        sorted(range(len(node_ids)), key=lambda index: (-float(scores[index]), node_ids[index])),
        dtype=np.int32,
    )


class GovernanceServingRuntime:
    """One verified Global model, one serialized GPU queue, immutable run artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        global_model_root: str | Path,
        device: str = "auto",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.global_model_root = Path(global_model_root).expanduser().resolve()
        self.artifact_root = self.root / "artifacts"
        self.run_root = self.root / "runs"
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="governance")
        self._active_run_id: str | None = None
        self._queued: set[str] = set()
        self._model: LoadedGlobalModel | None = None
        self._unavailable_reason: str | None = None
        self._reference_artifact_ids: set[str] = set()
        self._verified_runs: set[str] = set()
        try:
            self._model = load_global_model(self.global_model_root, device=device)
        except Exception:  # noqa: BLE001 - health must remain queryable after load failure
            self._unavailable_reason = "GFM_GOVERNANCE_GLOBAL_LOAD_FAILED"
        from .skills import GovernanceSkillExecutor

        self._skills = GovernanceSkillExecutor(self)
        self._recover_interrupted()
        self._refresh_reference_artifacts()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _recover_interrupted(self) -> None:
        for directory in sorted(self.run_root.glob("governance-*")):
            if not directory.is_dir() or RUN_ID_PATTERN.fullmatch(directory.name) is None:
                continue
            path = directory / "state.json"
            if not path.is_file():
                continue
            try:
                state = self._read_state(directory.name)
            except (OSError, ValueError):
                continue
            if state.get("status") not in {"queued", "running"}:
                continue
            state.update(
                {
                    "status": "interrupted",
                    "updatedAt": _utc_now(),
                    "errorCode": "GFM_GOVERNANCE_INTERRUPTED",
                }
            )
            self._write_state(directory.name, state)

    def health(self) -> dict[str, Any]:
        model = self._model
        with self._lock:
            active = self._active_run_id
            queue_depth = len(self._queued)
        ready = model is not None
        recipe_hash = (
            model.runtime_recipe_hash
            if model is not None
            else canonical_sha256(
                {"service": "socialgraph-fm-gfm/governance", "state": "unavailable"}
            )
        )
        identity = canonical_sha256(
            {
                "service": "socialgraph-fm-gfm/governance",
                "modelVersionId": model.model_version_id if model else None,
                "modelVersionHash": model.model_version_hash if model else None,
                "modelStateHash": model.model_state_hash if model else None,
                "runtimeRecipeHash": recipe_hash,
            }
        )
        payload: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "serviceIdentity": identity,
            "servingReady": ready,
            "onlineForwardReady": ready,
            "modelVersionId": model.model_version_id if model else None,
            "modelVersionHash": model.model_version_hash if model else None,
            "modelStateHash": model.model_state_hash if model else None,
            "device": model.device_name if model else "cpu",
            "dtype": model.dtype_name if model else "float32",
            "loadedAt": model.loaded_at if model else None,
            "queueDepth": queue_depth,
            "activeRunId": active,
            "runtimeRecipeHash": recipe_hash,
        }
        return _with_hash(payload, "healthHash")

    def capabilities(self) -> dict[str, Any]:
        model = self._model
        ready = model is not None
        payload: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "channel": "governance",
            "taskId": "coordination_risk",
            "servingReady": ready,
            "onlineForwardReady": ready,
            "unavailableReason": None if ready else self._unavailable_reason,
            "modelVersionId": model.model_version_id if model else None,
            "modelVersionHash": model.model_version_hash if model else None,
            "modelStateHash": model.model_state_hash if model else None,
            "supportedProtocols": ["global"],
            "skills": list(self._skills.skill_names),
            "inputSchemaVersion": INPUT_SCHEMA_VERSION,
            "modalities": list(MODALITIES),
            "sampleArtifactId": self._sample_artifact_id(),
            "limits": {
                "maxNodes": MAX_NODES,
                "maxRelationRows": MAX_RELATION_ROWS,
                "maxEvidenceNodes": MAX_EVIDENCE_NODES,
                "maxEvidenceEdges": MAX_EVIDENCE_EDGES,
                "maxPreviewNodes": MAX_PREVIEW_NODES,
                "maxPreviewEdges": MAX_PREVIEW_EDGES,
            },
        }
        return _with_hash(payload, "capabilityHash")

    def _sample_artifact_id(self) -> str | None:
        return min(self._reference_artifact_ids, default=None)

    def _refresh_reference_artifacts(self) -> None:
        for directory in sorted(self.artifact_root.glob("governance-artifact-*")):
            try:
                data = load_materialized_artifact(directory)
            except (OSError, ValueError):
                continue
            if self._is_reference_replay(data):
                self._reference_artifact_ids.add(directory.name)

    def _is_reference_replay(self, data: OnlineInferenceData) -> bool:
        if data.artifact.document.get("datasetId") != "socialgraph-fm:russia:dynamic-replay":
            return False
        try:
            from socialgraph_gfm.global_model.corpus import load_corpus_index

            index = load_corpus_index(self.global_model_root / "corpus", verify_manifests=True)
            russia = index.load_country(
                "russia", verify_hashes=True, verify_values=True, mmap_mode="r"
            )
        except (OSError, ValueError):
            return False
        if (
            data.node_ids != tuple(f"russia:{index}" for index in range(russia.manifest.node_count))
            or not np.array_equal(data.text_features, russia.text_features)
            or not np.array_equal(data.edge_index, russia.edge_index)
            or not np.array_equal(data.degree_bucket, russia.degree_bucket)
            or not np.array_equal(data.structure_missing, russia.structure_missing)
        ):
            return False
        for modality in MODALITIES:
            token = modality.lower()
            relation = russia.relation(modality)
            if not all(
                np.array_equal(data.arrays[f"relation_{token}_{suffix}"], expected)
                for suffix, expected in (
                    ("indptr", relation.indptr),
                    ("indices", relation.indices),
                    ("weights", relation.weights),
                )
            ):
                return False
        return True

    def _incoming_receipt(self, artifact_id: str, expected_hash: str) -> dict[str, Any]:
        path = self.root / "incoming" / artifact_id / "receipt.json"
        receipt = _read_json(path)
        logical = {key: value for key, value in receipt.items() if key != "artifactHash"}
        if (
            receipt.get("schemaVersion") != SCHEMA_VERSION
            or receipt.get("artifactId") != artifact_id
            or receipt.get("artifactHash") != canonical_sha256(logical)
            or receipt.get("artifactHash") != expected_hash
            or not isinstance(receipt.get("cleanSelfLoops"), bool)
        ):
            raise GovernanceInvalid
        return receipt

    def materialize(self, artifact_id: str, request: Mapping[str, Any]) -> dict[str, Any]:
        if ARTIFACT_ID_PATTERN.fullmatch(artifact_id) is None:
            raise GovernanceNotFound
        if (
            request.get("schemaVersion") != SCHEMA_VERSION
            or request.get("artifactId") != artifact_id
            or not isinstance(request.get("datasetContentHash"), str)
            or _HASH.fullmatch(str(request.get("datasetContentHash"))) is None
            or not isinstance(request.get("graphVersionHash"), str)
            or _HASH.fullmatch(str(request.get("graphVersionHash"))) is None
            or not isinstance(request.get("artifactHash"), str)
        ):
            raise GovernanceInvalid
        receipt = self._incoming_receipt(artifact_id, str(request["artifactHash"]))
        if (
            receipt.get("datasetContentHash") != request["datasetContentHash"]
            or receipt.get("graphVersionHash") != request["graphVersionHash"]
        ):
            raise GovernanceInvalid
        artifact = materialize_bundle(
            self.root,
            artifact_id,
            expected_dataset_content_hash=str(request["datasetContentHash"]),
            expected_graph_version_hash=str(request["graphVersionHash"]),
            clean_self_loops=bool(receipt["cleanSelfLoops"]),
        )
        loaded_artifact = load_materialized_artifact(artifact.root)
        if self._is_reference_replay(loaded_artifact):
            self._reference_artifact_ids.add(artifact_id)
        document = artifact.document
        if (
            document.get("bundleSha256") != receipt.get("bundleSha256")
            or document.get("nodeCount") != receipt.get("nodeCount")
            or document.get("relationRowCount") != receipt.get("relationRowCount")
            or document.get("selfLoopsRemoved") != receipt.get("selfLoopsRemoved")
            or document.get("modalities") != receipt.get("modalities")
        ):
            raise BundleValidationError("API receipt disagrees with authoritative materialization")
        response = {
            "schemaVersion": SCHEMA_VERSION,
            "artifactId": artifact_id,
            "datasetContentHash": artifact.dataset_content_hash,
            "graphVersionHash": artifact.graph_version_hash,
            "nodeCount": int(document["nodeCount"]),
            "relationRowCount": int(document["relationRowCount"]),
            "selfLoopsRemoved": int(document["selfLoopsRemoved"]),
            "modalities": list(document["modalities"]),
            "createdAt": str(document["createdAt"]),
            "compatibility": "compatible",
        }
        return _with_hash(response, "artifactHash")

    def _artifact(self, artifact_id: str) -> OnlineInferenceData:
        if ARTIFACT_ID_PATTERN.fullmatch(artifact_id) is None:
            raise GovernanceNotFound
        path = (self.artifact_root / artifact_id).resolve()
        if not path.is_relative_to(self.artifact_root.resolve()) or not path.is_dir():
            raise GovernanceNotFound
        return load_materialized_artifact(path)

    def _run_dir(self, run_id: str) -> Path:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise GovernanceNotFound
        return self.run_root / run_id

    def _write_state(self, run_id: str, state: Mapping[str, Any]) -> dict[str, Any]:
        payload = {key: value for key, value in state.items() if key != "statusHash"}
        _with_hash(payload, "statusHash")
        _atomic_json(self._run_dir(run_id) / "state.json", payload)
        return payload

    def _read_state(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._run_dir(run_id) / "state.json"
            if not path.is_file():
                raise GovernanceNotFound
            state = _read_json_with_windows_retry(path)
            logical = {key: value for key, value in state.items() if key != "statusHash"}
            if state.get("statusHash") != canonical_sha256(logical):
                raise ValueError("persisted SocialGraph-FM Governance status hash mismatch")
            if (
                state.get("schemaVersion") != SCHEMA_VERSION
                or state.get("runId") != run_id
                or _HASH.fullmatch(str(state.get("requestHash"))) is None
                or ARTIFACT_ID_PATTERN.fullmatch(str(state.get("artifactId"))) is None
                or _HASH.fullmatch(str(state.get("datasetContentHash"))) is None
                or _HASH.fullmatch(str(state.get("graphVersionHash"))) is None
                or not isinstance(state.get("modelVersionId"), str)
                or not state["modelVersionId"]
                or _HASH.fullmatch(str(state.get("modelVersionHash"))) is None
                or _HASH.fullmatch(str(state.get("modelStateHash"))) is None
            ):
                raise ValueError("persisted SocialGraph-FM Governance status identity mismatch")
            return state

    def _update_state(self, run_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            state = self._read_state(run_id)
            state.update(changes)
            state["updatedAt"] = _utc_now()
            return self._write_state(run_id, state)

    def _validated_run_request(
        self,
        run_id: str,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        persisted_state = self._read_state(run_id) if state is None else state
        document = _read_json(self._run_dir(run_id) / "request.json")
        expected_keys = {
            "schemaVersion",
            "request",
            "requestHash",
            "retryOf",
            "requestDocumentHash",
        }
        request = document.get("request")
        retry_of = document.get("retryOf")
        logical = {
            key: value for key, value in document.items() if key != "requestDocumentHash"
        }
        if (
            set(document) != expected_keys
            or document.get("schemaVersion") != SCHEMA_VERSION
            or not isinstance(request, dict)
            or (
                retry_of is not None
                and (
                    not isinstance(retry_of, str)
                    or RUN_ID_PATTERN.fullmatch(retry_of) is None
                )
            )
            or document.get("requestDocumentHash") != canonical_sha256(logical)
        ):
            raise ValueError("persisted SocialGraph-FM Governance request identity mismatch")
        request_hash = canonical_sha256(request)
        expected_state_identity = {
            "schemaVersion": SCHEMA_VERSION,
            "runId": run_id,
            "requestHash": request_hash,
            "artifactId": request.get("artifactId"),
            "datasetContentHash": request.get("datasetContentHash"),
            "graphVersionHash": request.get("graphVersionHash"),
            "modelVersionId": request.get("modelVersionId"),
            "modelStateHash": request.get("modelStateHash"),
        }
        if document.get("requestHash") != request_hash or any(
            persisted_state.get(key) != value
            for key, value in expected_state_identity.items()
        ):
            raise ValueError("persisted SocialGraph-FM Governance request identity mismatch")
        return dict(request)

    def create_run(
        self,
        request: Mapping[str, Any],
        *,
        retry_of: str | None = None,
    ) -> dict[str, Any]:
        model = self._model
        if model is None:
            raise GovernanceUnavailable
        required = {
            "schemaVersion",
            "protocol",
            "artifactId",
            "datasetContentHash",
            "graphVersionHash",
            "modelVersionId",
            "modelStateHash",
            "topK",
        }
        if (
            set(request) != required
            or request.get("schemaVersion") != SCHEMA_VERSION
            or request.get("protocol") != "global"
            or request.get("modelVersionId") != model.model_version_id
            or request.get("modelStateHash") != model.model_state_hash
            or ARTIFACT_ID_PATTERN.fullmatch(str(request.get("artifactId"))) is None
            or _HASH.fullmatch(str(request.get("datasetContentHash"))) is None
            or _HASH.fullmatch(str(request.get("graphVersionHash"))) is None
            or isinstance(request.get("topK"), bool)
            or not isinstance(request.get("topK"), int)
            or not 1 <= int(request["topK"]) <= MAX_NODES
        ):
            raise GovernanceInvalid
        artifact = self._artifact(str(request["artifactId"]))
        if (
            artifact.artifact.dataset_content_hash != request["datasetContentHash"]
            or artifact.artifact.graph_version_hash != request["graphVersionHash"]
            or int(request["topK"]) > len(artifact.node_ids)
        ):
            raise GovernanceInvalid
        normalized = dict(request)
        request_hash = canonical_sha256(normalized)
        run_id = f"governance-{uuid.uuid4().hex}"
        now = _utc_now()
        state: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "runId": run_id,
            "requestHash": request_hash,
            "artifactId": request["artifactId"],
            "datasetContentHash": request["datasetContentHash"],
            "graphVersionHash": request["graphVersionHash"],
            "modelVersionId": model.model_version_id,
            "modelVersionHash": model.model_version_hash,
            "modelStateHash": model.model_state_hash,
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "createdAt": now,
            "updatedAt": now,
            "errorCode": None,
            "cancelRequested": False,
        }
        run_dir = self._run_dir(run_id)
        with self._lock:
            run_dir.mkdir(parents=False, exist_ok=False)
            request_document = {
                "schemaVersion": SCHEMA_VERSION,
                "request": normalized,
                "requestHash": request_hash,
                "retryOf": retry_of,
            }
            request_document["requestDocumentHash"] = canonical_sha256(request_document)
            _atomic_json(run_dir / "request.json", request_document)
            state = self._write_state(run_id, state)
            self._queued.add(run_id)
        self._executor.submit(self._execute, run_id)
        return state

    def _cancelled(self, run_id: str) -> bool:
        return (self._run_dir(run_id) / "cancel.requested").is_file()

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        state = self._read_state(run_id)
        if state["status"] in _TERMINAL:
            return state
        flag = self._run_dir(run_id) / "cancel.requested"
        if not flag.exists():
            with flag.open("xb") as stream:
                stream.write(b"cancel\n")
                stream.flush()
                os.fsync(stream.fileno())
        return self._update_state(run_id, cancelRequested=True)

    def retry_run(self, run_id: str) -> dict[str, Any]:
        state = self._read_state(run_id)
        if state.get("status") not in {"failed", "cancelled", "interrupted"}:
            raise GovernanceInvalid
        request = self._validated_run_request(run_id, state=state)
        return self.create_run(request, retry_of=run_id)

    def list_runs(self, *, offset: int, limit: int) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for directory in self.run_root.glob("governance-*"):
            if directory.is_dir() and RUN_ID_PATTERN.fullmatch(directory.name):
                try:
                    items.append(self._read_state(directory.name))
                except (OSError, ValueError, GovernanceServiceError):
                    continue
        items.sort(key=lambda item: (str(item["createdAt"]), str(item["runId"])), reverse=True)
        return {
            "schemaVersion": SCHEMA_VERSION,
            "items": items[offset : offset + limit],
            "total": len(items),
            "offset": offset,
            "limit": limit,
        }

    def _save_outputs(
        self,
        run_id: str,
        outputs: OnlineInferenceOutputs,
        analytics: DerivedAnalytics,
    ) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        state = self._read_state(run_id)
        data = self._artifact(str(state["artifactId"]))
        model = self._model
        if model is None:
            raise GovernanceUnavailable
        if (
            state.get("modelVersionId") != model.model_version_id
            or state.get("modelVersionHash") != model.model_version_hash
            or state.get("modelStateHash") != model.model_state_hash
        ):
            raise ValueError("persisted SocialGraph-FM Governance run identity mismatch")
        rank_order = _rank_order(outputs.scores, data.node_ids)
        ranks = np.empty(outputs.scores.shape[0], dtype=np.int32)
        ranks[rank_order] = np.arange(1, outputs.scores.shape[0] + 1, dtype=np.int32)
        output_path = run_dir / "outputs.npz"
        _atomic_npz(
            output_path,
            {
                "logits": outputs.logits.astype(np.float32),
                "scores": outputs.scores.astype(np.float32),
                "embeddings": outputs.embeddings.astype(np.float16),
                "router_indices": outputs.router_indices.astype(np.int16),
                "router_weights": outputs.router_weights.astype(np.float32),
                "modality_contributions": outputs.modality_contributions.astype(np.float32),
                "modality_counts": outputs.modality_counts.astype(np.int32),
                "rank_order": rank_order,
                "ranks": ranks,
                "community_ids": analytics.community_ids.astype(np.int32),
            },
        )
        analytics_path = run_dir / "analytics.npz"
        _atomic_npz(analytics_path, analytics.relation_arrays)
        documents = {
            "schemaVersion": SCHEMA_VERSION,
            "groups": list(analytics.groups),
            "links": list(analytics.links),
            "limitations": [
                "Potential links inspect at most the 512 highest-risk members per large community.",
                "Each inspected node considers at most 32 cosine-nearest same-community peers.",
                "Potential links are capped at 10 per node and 500 per run.",
            ],
        }
        _atomic_json(run_dir / "analytics.json", documents)
        manifest: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "outputsSha256": file_sha256(output_path),
            "analyticsSha256": file_sha256(analytics_path),
            "analyticsJsonSha256": file_sha256(run_dir / "analytics.json"),
            "batchSize": outputs.batch_size,
            "peakMemoryMiB": outputs.peak_memory_mib,
            "inferenceSeed": outputs.seed,
            "nodeCount": int(outputs.scores.shape[0]),
            "relationCount": int(analytics.relation_arrays["source"].shape[0]),
            "groupCount": len(analytics.groups),
            "potentialLinkCount": len(analytics.links),
            "embeddingDtype": "float16",
            "embeddingDimension": 256,
            "requestHash": state["requestHash"],
            "runtimeRecipeHash": model.runtime_recipe_hash,
            "modelStateHash": state["modelStateHash"],
            "device": model.device_name,
            "dtype": model.dtype_name,
            "datasetContentHash": data.artifact.dataset_content_hash,
            "graphVersionHash": data.artifact.graph_version_hash,
            "fanout": [20, 10],
            "amp": bool(self._model and self._model.device_name == "cuda"),
            "numWorkers": 0,
        }
        return _with_hash(manifest, "runArtifactHash")

    def _execute(self, run_id: str) -> None:
        with self._lock:
            self._queued.discard(run_id)
            self._active_run_id = run_id
        try:
            if self._cancelled(run_id):
                raise InferenceCancelled
            state = self._update_state(run_id, status="running", stage="validating", progress=3)
            artifact = self._artifact(str(state["artifactId"]))
            if (
                artifact.artifact.dataset_content_hash != state["datasetContentHash"]
                or artifact.artifact.graph_version_hash != state["graphVersionHash"]
            ):
                raise ValueError("run artifact identity changed before execution")
            self._update_state(run_id, stage="preprocessing", progress=8)
            if self._cancelled(run_id):
                raise InferenceCancelled
            model = self._model
            if model is None:
                raise GovernanceUnavailable
            self._update_state(run_id, stage="inferencing", progress=10)

            def progress(value: float) -> None:
                self._update_state(
                    run_id,
                    stage="inferencing",
                    progress=max(10, min(68, 10 + int(value * 58))),
                )

            outputs = run_online_inference(
                artifact,
                model,
                progress=progress,
                cancelled=lambda: self._cancelled(run_id),
            )
            self._update_state(run_id, stage="deriving", progress=72)
            analytics = derive_analytics(artifact, outputs, seed=outputs.seed % (2**32 - 1))
            if self._cancelled(run_id):
                raise InferenceCancelled
            self._update_state(run_id, stage="freezing", progress=90)
            manifest = self._save_outputs(run_id, outputs, analytics)
            _atomic_json(self._run_dir(run_id) / "run-artifacts.json", manifest)
            result = self._build_result(run_id, artifact, outputs, analytics)
            _atomic_json(self._run_dir(run_id) / "result.json", result)
            self._update_state(
                run_id,
                status="succeeded",
                stage="completed",
                progress=100,
                errorCode=None,
            )
        except InferenceCancelled:
            self._update_state(
                run_id,
                status="cancelled",
                progress=100,
                errorCode="GFM_GOVERNANCE_CANCELLED",
                cancelRequested=True,
            )
        except Exception:  # noqa: BLE001 - background failure is persisted, never hidden as success
            try:
                self._update_state(
                    run_id,
                    status="failed",
                    progress=100,
                    errorCode="GFM_GOVERNANCE_EXECUTION_FAILED",
                )
            except Exception:  # noqa: BLE001, S110 - preserve original background failure
                pass
        finally:
            with self._lock:
                if self._active_run_id == run_id:
                    self._active_run_id = None

    def _finding(
        self,
        data: OnlineInferenceData,
        arrays: Mapping[str, np.ndarray],
        index: int,
        *,
        rank: int | None = None,
        community_sizes: np.ndarray | None = None,
    ) -> dict[str, Any]:
        model = self._model
        if model is None:
            raise GovernanceUnavailable
        score = float(arrays["scores"][index])
        actual_rank = int(arrays["ranks"][index]) if rank is None else rank
        routes: list[dict[str, Any]] = [{"expert": "shared", "weight": 1.0}]
        for expert_index, weight in zip(
            arrays["router_indices"][index], arrays["router_weights"][index], strict=True
        ):
            position = int(expert_index)
            if not 1 <= position < len(model.expert_names):
                raise ValueError("persisted router expert index is invalid")
            routes.append({"expert": model.expert_names[position], "weight": float(weight)})
        community_index = int(arrays["community_ids"][index])
        if community_sizes is None:
            community_sizes = np.bincount(arrays["community_ids"].astype(np.int64))
        community_id = (
            f"group-{community_index + 1}"
            if community_index >= 0 and int(community_sizes[community_index]) >= 2
            else None
        )
        contribution = arrays["modality_contributions"][index]
        counts = arrays["modality_counts"][index]
        band = _risk_band(score, model.threshold)
        return {
            "nodeId": data.node_ids[index],
            "label": data.labels[index],
            "score": score,
            "logit": float(arrays["logits"][index]),
            "rank": actual_rank,
            "riskBand": band,
            "predictedPositive": band == "high",
            "structureMissing": bool(data.structure_missing[index]),
            "routes": routes,
            "modalityContribution": {
                "text": float(contribution[0]),
                "structure": float(contribution[1]),
            },
            "modalityEvidence": {
                modality: int(counts[column]) for column, modality in enumerate(MODALITIES)
            },
            "communityId": community_id,
        }

    def _build_result(
        self,
        run_id: str,
        data: OnlineInferenceData,
        outputs: OnlineInferenceOutputs,
        analytics: DerivedAnalytics,
    ) -> dict[str, Any]:
        model = self._model
        if model is None:
            raise GovernanceUnavailable
        state = self._read_state(run_id)
        request = self._validated_run_request(run_id, state=state)
        if (
            state.get("modelVersionId") != model.model_version_id
            or state.get("modelVersionHash") != model.model_version_hash
            or state.get("modelStateHash") != model.model_state_hash
        ):
            raise ValueError("persisted SocialGraph-FM Governance run identity mismatch")
        self._run_manifest(run_id)
        top_k = min(int(request["topK"]), len(data.node_ids))
        order = _rank_order(outputs.scores, data.node_ids)
        ranks = np.empty(outputs.scores.shape[0], dtype=np.int32)
        ranks[order] = np.arange(1, outputs.scores.shape[0] + 1, dtype=np.int32)
        arrays = {
            "scores": outputs.scores,
            "logits": outputs.logits,
            "router_indices": outputs.router_indices,
            "router_weights": outputs.router_weights,
            "modality_contributions": outputs.modality_contributions,
            "modality_counts": outputs.modality_counts,
            "community_ids": analytics.community_ids,
            "ranks": ranks,
        }
        community_sizes = np.bincount(analytics.community_ids.astype(np.int64))
        findings = [
            self._finding(
                data,
                arrays,
                int(node),
                rank=rank,
                community_sizes=community_sizes,
            )
            for rank, node in enumerate(order[:top_k], start=1)
        ]
        bands = [_risk_band(float(score), model.threshold) for score in outputs.scores]
        distribution = {
            "low": bands.count("low"),
            "review": bands.count("review"),
            "high": bands.count("high"),
            "predictedPositive": bands.count("high"),
            "total": len(bands),
        }
        payload: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "runId": run_id,
            "requestHash": canonical_sha256(request),
            "artifactId": data.artifact.artifact_id,
            "datasetContentHash": data.artifact.dataset_content_hash,
            "graphVersionHash": data.artifact.graph_version_hash,
            "modelVersionId": model.model_version_id,
            "modelVersionHash": model.model_version_hash,
            "modelStateHash": model.model_state_hash,
            "threshold": model.threshold,
            "calibration": {
                "temperature": model.temperature,
                "bias": model.bias,
                "referenceThreshold": model.threshold,
                "applicability": (
                    "reference_replay"
                    if data.artifact.artifact_id in self._reference_artifact_ids
                    else "out_of_domain_unverified"
                ),
            },
            "referenceMetrics": dict(model.reference_metrics),
            "datasetMetrics": None,
            "distribution": distribution,
            "findings": findings,
            "totalFindings": len(data.node_ids),
            "limitations": list(_LIMITATIONS),
            "completedAt": _utc_now(),
        }
        return _with_hash(payload, "resultHash")

    def _run_manifest(self, run_id: str) -> dict[str, Any]:
        manifest = _read_json(self._run_dir(run_id) / "run-artifacts.json")
        logical = {key: value for key, value in manifest.items() if key != "runArtifactHash"}
        if manifest.get("runArtifactHash") != canonical_sha256(logical):
            raise ValueError("run artifact manifest hash mismatch")
        state = self._read_state(run_id)
        self._validated_run_request(run_id, state=state)
        expected_identity = {
            "requestHash": state.get("requestHash"),
            "modelStateHash": state.get("modelStateHash"),
            "datasetContentHash": state.get("datasetContentHash"),
            "graphVersionHash": state.get("graphVersionHash"),
        }
        if any(manifest.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("persisted SocialGraph-FM Governance run identity mismatch")
        return manifest

    def _outputs(self, run_id: str) -> dict[str, np.ndarray]:
        manifest = self._run_manifest(run_id)
        return _load_npz(self._run_dir(run_id) / "outputs.npz", str(manifest["outputsSha256"]))

    def _analytics_arrays(self, run_id: str) -> dict[str, np.ndarray]:
        manifest = self._run_manifest(run_id)
        return _load_npz(self._run_dir(run_id) / "analytics.npz", str(manifest["analyticsSha256"]))

    def _analytics_document(self, run_id: str) -> dict[str, Any]:
        manifest = self._run_manifest(run_id)
        path = self._run_dir(run_id) / "analytics.json"
        if file_sha256(path) != manifest.get("analyticsJsonSha256"):
            raise ValueError("analytics document hash mismatch")
        return _read_json(path)

    def result(self, run_id: str) -> dict[str, Any]:
        state = self._read_state(run_id)
        if state.get("status") != "succeeded":
            raise GovernanceNotReady
        manifest = self._run_manifest(run_id)
        if run_id not in self._verified_runs:
            _load_npz(self._run_dir(run_id) / "outputs.npz", str(manifest["outputsSha256"]))
            _load_npz(self._run_dir(run_id) / "analytics.npz", str(manifest["analyticsSha256"]))
            analytics_path = self._run_dir(run_id) / "analytics.json"
            if file_sha256(analytics_path) != manifest.get("analyticsJsonSha256"):
                raise ValueError("analytics document hash mismatch")
            self._verified_runs.add(run_id)
        else:
            if (
                file_sha256(self._run_dir(run_id) / "outputs.npz")
                != manifest.get("outputsSha256")
                or file_sha256(self._run_dir(run_id) / "analytics.npz")
                != manifest.get("analyticsSha256")
            ):
                raise ValueError("persisted SocialGraph-FM Governance NPZ identity mismatch")
            if (
                file_sha256(self._run_dir(run_id) / "analytics.json")
                != manifest.get("analyticsJsonSha256")
            ):
                raise ValueError("analytics document hash mismatch")
        result = _read_json(self._run_dir(run_id) / "result.json")
        logical = {key: value for key, value in result.items() if key != "resultHash"}
        if result.get("resultHash") != canonical_sha256(logical):
            raise ValueError("persisted SocialGraph-FM Governance result hash mismatch")
        expected_identity = {
            "schemaVersion": SCHEMA_VERSION,
            "runId": run_id,
            "requestHash": state.get("requestHash"),
            "artifactId": state.get("artifactId"),
            "datasetContentHash": state.get("datasetContentHash"),
            "graphVersionHash": state.get("graphVersionHash"),
            "modelVersionId": state.get("modelVersionId"),
            "modelVersionHash": state.get("modelVersionHash"),
            "modelStateHash": state.get("modelStateHash"),
        }
        if any(result.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("persisted SocialGraph-FM Governance run identity mismatch")
        return result

    def findings(self, run_id: str, *, offset: int, limit: int) -> dict[str, Any]:
        result = self.result(run_id)
        data = self._artifact(str(result["artifactId"]))
        arrays = self._outputs(run_id)
        order = arrays["rank_order"]
        selected = order[offset : offset + limit]
        community_sizes = np.bincount(arrays["community_ids"].astype(np.int64))
        items = [
            self._finding(data, arrays, int(index), community_sizes=community_sizes)
            for index in selected
        ]
        payload: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "runId": run_id,
            "items": items,
            "total": len(data.node_ids),
            "offset": offset,
            "limit": limit,
        }
        return _with_hash(payload, "pageHash")

    @staticmethod
    def _group_derivation(document: Mapping[str, Any]) -> dict[str, Any]:
        relation_counts = document["relationCounts"]
        return {
            "id": document["groupId"],
            "kind": "group",
            "priority": document["priority"],
            "nodeIds": document["memberNodeIds"],
            "source": None,
            "target": None,
            "modalities": [
                modality for modality in MODALITIES if int(relation_counts[modality]) > 0
            ],
            "memberCount": document["memberCount"],
            "meanScore": document["averageRisk"],
            "p90Score": document["p90Risk"],
            "scoreComponents": {
                "p90": document["p90Risk"],
                "mean": document["averageRisk"],
            },
            "factual": False,
            "limitation": "Louvain community priority is derived from member risk, not proof of coordination.",
        }

    def _relation_derivation(
        self,
        data: OnlineInferenceData,
        arrays: Mapping[str, np.ndarray],
        index: int,
    ) -> dict[str, Any]:
        source_index, target_index = int(arrays["source"][index]), int(arrays["target"][index])
        source, target = data.node_ids[source_index], data.node_ids[target_index]
        mask = int(arrays["modality_mask"][index])
        modalities = [
            modality for column, modality in enumerate(MODALITIES) if mask & (1 << column)
        ]
        diversity = len(modalities) / len(MODALITIES)
        return {
            "id": f"relation-{source_index}-{target_index}",
            "kind": "factual_relation",
            "priority": float(arrays["priority"][index]),
            "nodeIds": [source, target],
            "source": source,
            "target": target,
            "modalities": modalities,
            "memberCount": None,
            "meanScore": None,
            "p90Score": None,
            "scoreComponents": {
                "endpointRisk": float(arrays["endpoint_risk"][index]),
                "diversity": diversity,
                "weightPercentile": float(arrays["weight_percentile"][index]),
            },
            "factual": True,
            "limitation": "Derived analyst priority over a factual input relation; it is not proof of coordination.",
        }

    @staticmethod
    def _link_derivation(document: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": document["linkId"],
            "kind": "potential_link",
            "priority": document["priority"],
            "nodeIds": [document["source"], document["target"]],
            "source": document["source"],
            "target": document["target"],
            "modalities": [],
            "memberCount": None,
            "meanScore": None,
            "p90Score": None,
            "scoreComponents": {
                "cosine": document["embeddingCosine"],
                "cosineSimilarity": document["embeddingSimilarity"],
                "endpointRisk": document["endpointRisk"],
                "jaccard": document["commonNeighborJaccard"],
            },
            "factual": False,
            "limitation": "Bounded same-community similarity lead; this is not a factual or future edge.",
        }

    def derivations(
        self,
        run_id: str,
        kind: str,
        *,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        result = self.result(run_id)
        data = self._artifact(str(result["artifactId"]))
        document = self._analytics_document(run_id)
        if kind == "groups":
            raw_items = document["groups"]
            total = len(raw_items)
            items = [self._group_derivation(item) for item in raw_items[offset : offset + limit]]
        elif kind == "relations":
            arrays = self._analytics_arrays(run_id)
            order = arrays["order"]
            total = int(order.shape[0])
            items = [
                self._relation_derivation(data, arrays, int(index))
                for index in order[offset : offset + limit]
            ]
        elif kind == "links":
            raw_items = document["links"]
            total = len(raw_items)
            items = [self._link_derivation(item) for item in raw_items[offset : offset + limit]]
        else:
            raise GovernanceNotFound
        payload: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "runId": run_id,
            "items": items,
            "total": total,
            "offset": offset,
            "limit": limit,
        }
        return _with_hash(payload, "pageHash")

    @staticmethod
    def _pair_relations(
        data: OnlineInferenceData, source: int, target: int
    ) -> list[dict[str, Any]]:
        relations: list[dict[str, Any]] = []
        for modality in MODALITIES:
            token = modality.lower()
            indptr = np.asarray(data.arrays[f"relation_{token}_indptr"])
            indices = np.asarray(data.arrays[f"relation_{token}_indices"])
            weights = np.asarray(data.arrays[f"relation_{token}_weights"])
            start, stop = int(indptr[source]), int(indptr[source + 1])
            row = indices[start:stop]
            position = int(np.searchsorted(row, target))
            if position < row.shape[0] and int(row[position]) == target:
                relations.append(
                    {"modality": modality, "rawWeight": float(weights[start + position])}
                )
        return relations

    def _preview(
        self,
        data: OnlineInferenceData,
        *,
        run_id: str | None = None,
        result_hash: str | None = None,
        arrays: Mapping[str, np.ndarray] | None = None,
        projection: ProjectionRequest | None = None,
        group_documents: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        node_count = len(data.node_ids)
        degrees = np.diff(np.asarray(data.arrays["fused_indptr"]))
        edge_index = np.asarray(data.edge_index)
        undirected = [
            (int(source), int(target))
            for source, target in zip(edge_index[0], edge_index[1], strict=True)
            if int(source) < int(target)
        ]
        selection = None
        if projection is not None:
            try:
                selection = select_projection(
                    data,
                    projection,
                    arrays=arrays,
                    groups=group_documents,
                    threshold=self._model.threshold if arrays is not None and self._model else None,
                )
            except ValueError as error:
                raise GovernanceInvalid from error
            selected_order = list(selection.selected_order)
            preview_edges = list(selection.edges)
        elif arrays is None:
            selected_order = sorted(
                range(node_count), key=lambda index: (-int(degrees[index]), data.node_ids[index])
            )[:MAX_PREVIEW_NODES]
        else:
            model = self._model
            if model is None:
                raise GovernanceUnavailable
            buckets: dict[str, list[int]] = {"high": [], "review": [], "low": []}
            for index in arrays["rank_order"]:
                integer = int(index)
                buckets[_risk_band(float(arrays["scores"][integer]), model.threshold)].append(
                    integer
                )
            quotas = {"high": 1500, "review": 900, "low": 600}
            selected_order = []
            leftovers: list[int] = []
            for band in ("high", "review", "low"):
                selected_order.extend(buckets[band][: quotas[band]])
                leftovers.extend(buckets[band][quotas[band] :])
            selected_order.extend(leftovers[: MAX_PREVIEW_NODES - len(selected_order)])
        if projection is None:
            selected_order = selected_order[:MAX_PREVIEW_NODES]
            selected = set(selected_order)
            preview_edges = [
                pair for pair in undirected if pair[0] in selected and pair[1] in selected
            ]
            if not preview_edges and undirected and MAX_PREVIEW_NODES >= 2:
                first = undirected[0]
                for endpoint in first:
                    if endpoint not in selected:
                        if len(selected_order) == MAX_PREVIEW_NODES:
                            selected.discard(selected_order.pop())
                        selected_order.append(endpoint)
                        selected.add(endpoint)
                preview_edges = [first]
            preview_edges = preview_edges[:MAX_PREVIEW_EDGES]
        community_sizes = (
            np.bincount(arrays["community_ids"].astype(np.int64)) if arrays is not None else None
        )
        nodes: list[dict[str, Any]] = []
        for index in selected_order:
            item: dict[str, Any] = {
                "id": data.node_ids[index],
                "label": data.labels[index],
                "degree": int(degrees[index]),
                "structureMissing": bool(data.structure_missing[index]),
                "score": None,
                "riskBand": None,
                "groupId": None,
            }
            if arrays is not None:
                score = float(arrays["scores"][index])
                community = int(arrays["community_ids"][index])
                item.update(
                    {
                        "score": score,
                        "riskBand": _risk_band(score, self._model.threshold),  # type: ignore[union-attr]
                        "groupId": (
                            f"group-{community + 1}"
                            if community_sizes is not None and int(community_sizes[community]) >= 2
                            else None
                        ),
                    }
                )
            nodes.append(item)
        edges = []
        for source, target in preview_edges:
            relations = self._pair_relations(data, source, target)
            edges.append(
                {
                    "id": f"edge-{source}-{target}",
                    "source": data.node_ids[source],
                    "target": data.node_ids[target],
                    "modalities": [item["modality"] for item in relations],
                    "factual": True,
                }
            )
        if selection is not None and selection.supernodes:
            nodes = [dict(value) for value in selection.supernodes]
            edges = [dict(value) for value in selection.aggregate_edges]
        payload: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "artifactId": data.artifact.artifact_id,
            "datasetContentHash": data.artifact.dataset_content_hash,
            "graphVersionHash": data.artifact.graph_version_hash,
            "runId": run_id,
            "resultHash": result_hash,
            "nodes": nodes,
            "edges": edges,
            "nodeCount": node_count,
            "edgeCount": len(undirected),
            "partialPreview": len(nodes) < node_count or len(edges) < len(undirected),
        }
        if projection is not None and selection is not None:
            payload.update(
                {
                    "preset": projection.preset,
                    "budgets": dict(selection.budgets),
                    "selectionRecipeId": selection.selection_recipe_id,
                    "isPartial": payload["partialPreview"],
                    "groups": list(selection.groups),
                    "sourceCounts": dict(selection.source_counts),
                    "inventoryCounts": {
                        "nodes": node_count,
                        "edges": len(undirected),
                        "groups": len(group_documents),
                    },
                }
            )
        return _with_hash(payload, "previewHash")

    def artifact_preview(
        self, artifact_id: str, *, projection: ProjectionRequest | None = None
    ) -> dict[str, Any]:
        return self._preview(self._artifact(artifact_id), projection=projection)

    def run_preview(
        self, run_id: str, *, projection: ProjectionRequest | None = None
    ) -> dict[str, Any]:
        result = self.result(run_id)
        data = self._artifact(str(result["artifactId"]))
        group_documents = (
            self._analytics_document(run_id).get("groups", ()) if projection is not None else ()
        )
        return self._preview(
            data,
            run_id=run_id,
            result_hash=str(result["resultHash"]),
            arrays=self._outputs(run_id),
            projection=projection,
            group_documents=group_documents,
        )

    def evidence(self, run_id: str, node_id: str) -> dict[str, Any]:
        if (
            not node_id
            or len(node_id) > 128
            or any(ord(character) < 32 or ord(character) == 127 for character in node_id)
        ):
            raise GovernanceNotFound
        result = self.result(run_id)
        data = self._artifact(str(result["artifactId"]))
        try:
            root = data.node_ids.index(node_id)
        except ValueError as exc:
            raise GovernanceNotFound from exc
        arrays = self._outputs(run_id)
        fused_indptr = np.asarray(data.arrays["fused_indptr"])
        fused_indices = np.asarray(data.arrays["fused_indices"])
        hops: dict[int, int] = {root: 0}
        frontier = {root}
        for hop in (1, 2):
            following: set[int] = set()
            for source in frontier:
                start, stop = int(fused_indptr[source]), int(fused_indptr[source + 1])
                following.update(int(value) for value in fused_indices[start:stop])
            following.difference_update(hops)
            for target in following:
                hops[target] = hop
            frontier = following
            if not frontier:
                break
        candidates = [index for index in hops if index != root]
        candidates.sort(
            key=lambda index: (
                hops[index],
                -float(arrays["scores"][index]),
                data.node_ids[index],
            )
        )
        selected_order = [root, *candidates[: MAX_EVIDENCE_NODES - 1]]
        selected = set(selected_order)
        all_edges: list[tuple[int, int]] = []
        for source in selected:
            start, stop = int(fused_indptr[source]), int(fused_indptr[source + 1])
            for target_value in fused_indices[start:stop]:
                target = int(target_value)
                if source < target and target in selected:
                    all_edges.append((source, target))
        all_edges.sort(
            key=lambda pair: (
                hops[pair[0]] + hops[pair[1]],
                -max(float(arrays["scores"][pair[0]]), float(arrays["scores"][pair[1]])),
                data.node_ids[pair[0]],
                data.node_ids[pair[1]],
            )
        )
        selected_edges = all_edges[:MAX_EVIDENCE_EDGES]
        truncated = len(hops) > len(selected_order) or len(all_edges) > len(selected_edges)
        model = self._model
        if model is None:
            raise GovernanceUnavailable
        community_sizes = np.bincount(arrays["community_ids"].astype(np.int64))
        root_finding = self._finding(data, arrays, root, community_sizes=community_sizes)

        def scored_node(index: int) -> dict[str, Any]:
            score = float(arrays["scores"][index])
            band = _risk_band(score, model.threshold)
            return {
                "nodeId": data.node_ids[index],
                "score": score,
                "hop": hops[index],
                "riskBand": band,
                "predictedPositive": band == "high",
                "structureMissing": bool(data.structure_missing[index]),
            }

        neighbors = []
        for target in selected_order:
            if hops[target] != 1:
                continue
            relations = self._pair_relations(data, root, target)
            if not relations:
                relations = self._pair_relations(data, target, root)
            neighbors.append(
                {
                    **scored_node(target),
                    "modalities": [item["modality"] for item in relations],
                    "relations": relations,
                }
            )
        neighbors.sort(key=lambda item: (-float(item["score"]), str(item["nodeId"])))
        subgraph_nodes = [scored_node(index) for index in selected_order]
        subgraph_nodes.sort(
            key=lambda item: (int(item["hop"]), -float(item["score"]), str(item["nodeId"]))
        )
        subgraph_edges = []
        for source, target in selected_edges:
            relations = self._pair_relations(data, source, target)
            if not relations:
                relations = self._pair_relations(data, target, source)
            subgraph_edges.append(
                {
                    "id": f"evidence-{source}-{target}",
                    "source": data.node_ids[source],
                    "target": data.node_ids[target],
                    "relations": relations,
                    "evidenceRole": "explanationOnly",
                }
            )
        relation_counts = {
            modality: int(arrays["modality_counts"][root, column])
            for column, modality in enumerate(MODALITIES)
        }
        payload: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "runId": run_id,
            "resultHash": result["resultHash"],
            "artifactId": result["artifactId"],
            "datasetContentHash": result["datasetContentHash"],
            "graphVersionHash": result["graphVersionHash"],
            "modelVersionId": result["modelVersionId"],
            "modelVersionHash": result["modelVersionHash"],
            "modelStateHash": result["modelStateHash"],
            "threshold": result["threshold"],
            "node": root_finding,
            "neighbors": neighbors,
            "structuralSignals": {
                "fusedDegree": int(fused_indptr[root + 1] - fused_indptr[root]),
                "structureMissing": bool(data.structure_missing[root]),
                "relationNeighborCounts": relation_counts,
                "twoHopNodeCount": len(hops) - 1,
                "relationEvidenceRole": "explanationOnly",
            },
            "evidenceSubgraph": {
                "depth": 2,
                "nodeCount": len(subgraph_nodes),
                "edgeCount": len(subgraph_edges),
                "truncated": truncated,
                "nodes": subgraph_nodes,
                "edges": subgraph_edges,
            },
            "truncated": truncated,
            "limitation": (
                "The two-hop view is capped at 300 nodes and 1000 factual edges; relation "
                "weights are explanation-only and do not prove coordination or intent."
            ),
        }
        return _with_hash(payload, "evidenceHash")

    @staticmethod
    def _adaptation_hash(value: str) -> str:
        if _HASH.fullmatch(value) is None:
            raise GovernanceNotFound
        return value

    def _adaptation_binding(self, run_id: str) -> AdaptationBinding:
        result = self.result(run_id)
        manifest = self._run_manifest(run_id)
        return AdaptationBinding.model_validate(
            {
                "artifactId": result["artifactId"],
                "datasetContentHash": result["datasetContentHash"],
                "graphVersionHash": result["graphVersionHash"],
                "runId": run_id,
                "requestHash": result["requestHash"],
                "resultHash": result["resultHash"],
                "runArtifactHash": manifest["runArtifactHash"],
                "modelVersionId": result["modelVersionId"],
                "modelVersionHash": result["modelVersionHash"],
                "modelStateHash": result["modelStateHash"],
                "recipeHash": manifest["runtimeRecipeHash"],
                "codeHash": ADAPTATION_CODE_HASH,
                "seed": manifest["inferenceSeed"],
            }
        )

    def _adaptation_path(self, kind: str, record_hash: str) -> Path:
        return self.root / "adaptations" / kind / f"{self._adaptation_hash(record_hash)}.json"

    def _persist_immutable_adaptation(
        self, path: Path, payload: Mapping[str, Any]
    ) -> None:
        with self._lock:
            if path.exists():
                if path.is_symlink() or _read_json(path) != dict(payload):
                    raise ValueError("immutable adaptation artifact conflicts with persisted data")
                return
            _atomic_json(path, payload)

    def _label_set(self, label_set_hash: str) -> TargetLabelSet:
        path = self._adaptation_path("label-sets", label_set_hash)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            raise GovernanceNotFound
        label_set = TargetLabelSet.model_validate(_read_json(path))
        if label_set.label_set_hash != label_set_hash:
            raise ValueError("persisted label-set identity mismatch")
        if self._adaptation_binding(label_set.binding.run_id) != label_set.binding:
            raise ValueError("persisted label-set run binding is stale")
        if label_set.sidecar_receipt is not None:
            artifact = self._artifact(label_set.binding.artifact_id)
            validate_sidecar_against_fused_graph(
                label_set,
                artifact_document=artifact.artifact.document,
                node_ids=artifact.node_ids,
                fused_indptr=np.asarray(artifact.arrays["fused_indptr"]),
            )
        return label_set

    def _v2_label_set(
        self,
        label_set_hash: str,
        fit_request: Mapping[str, Any] | None = None,
    ) -> tuple[TargetLabelSetV2, AdaptationBinding]:
        path = self._adaptation_path("label-sets", label_set_hash)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            raise GovernanceNotFound
        label_set = TargetLabelSetV2.model_validate(_read_json(path))
        if label_set.label_set_hash != label_set_hash:
            raise GovernanceServiceError("persisted v2 label-set identity mismatch")
        binding = self._v2_label_binding(label_set_hash, fit_request)
        if self._adaptation_binding(binding.run_id) != binding:
            raise GovernanceServiceError("persisted v2 label-set run binding is stale")
        artifact = self._artifact(binding.artifact_id)
        if artifact.artifact.document.get("bundleSha256") != label_set.inference_sha256:
            raise GovernanceServiceError(
                "persisted v2 label-set inference binding is stale"
            )
        return label_set, binding

    @staticmethod
    def _v2_binding_hash(
        label_set_hash: str,
        target_task_registration_id: str,
        run_id: str,
        result_hash: str,
    ) -> str:
        return canonical_sha256(
            {
                "labelSetHash": label_set_hash,
                "targetTaskRegistrationId": target_task_registration_id,
                "runId": run_id,
                "resultHash": result_hash,
            }
        )

    def _read_v2_binding_document(
        self, path: Path, label_set_hash: str
    ) -> tuple[str | None, AdaptationBinding]:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > 64 * 1024
            or _HASH.fullmatch(path.stem) is None
        ):
            raise GovernanceServiceError("persisted v2 label binding is invalid")
        document = _read_json(path)
        if document.get("schemaVersion") != "TargetLabelSetBinding/1.0":
            if path.stem != label_set_hash:
                raise GovernanceServiceError("persisted v2 label binding is invalid")
            return None, AdaptationBinding.model_validate(document)
        if set(document) != {
            "schemaVersion",
            "bindingHash",
            "labelSetHash",
            "targetTaskRegistrationId",
            "runId",
            "resultHash",
            "binding",
        }:
            raise GovernanceServiceError("persisted v2 label binding is invalid")
        registration_id = str(document.get("targetTaskRegistrationId", ""))
        run_id = str(document.get("runId", ""))
        result_hash = str(document.get("resultHash", ""))
        binding_hash = self._v2_binding_hash(
            label_set_hash, registration_id, run_id, result_hash
        )
        binding = AdaptationBinding.model_validate(document.get("binding"))
        if (
            document.get("labelSetHash") != label_set_hash
            or document.get("bindingHash") != path.stem
            or binding_hash != path.stem
            or _TARGET_TASK_REGISTRATION.fullmatch(registration_id) is None
            or binding.run_id != run_id
            or binding.result_hash != result_hash
        ):
            raise GovernanceServiceError("persisted v2 label binding identity mismatch")
        return registration_id, binding

    def _v2_label_binding(
        self,
        label_set_hash: str,
        fit_request: Mapping[str, Any] | None,
    ) -> AdaptationBinding:
        binding_root = self.root / "adaptations" / "label-set-bindings"
        if fit_request is not None:
            if set(fit_request) != {
                "schemaVersion",
                "targetTaskRegistrationId",
                "runId",
                "resultHash",
            } or fit_request.get("schemaVersion") != "socialgraph-fm.governance-target-review-policy-fit-request/1.0":
                raise GovernanceInvalid
            registration_id = str(fit_request.get("targetTaskRegistrationId", ""))
            run_id = str(fit_request.get("runId", ""))
            result_hash = str(fit_request.get("resultHash", ""))
            if (
                _TARGET_TASK_REGISTRATION.fullmatch(registration_id) is None
                or RUN_ID_PATTERN.fullmatch(run_id) is None
                or _HASH.fullmatch(result_hash) is None
            ):
                raise GovernanceInvalid
            binding_hash = self._v2_binding_hash(
                label_set_hash, registration_id, run_id, result_hash
            )
            path = self._adaptation_path("label-set-bindings", binding_hash)
            if not path.is_file():
                legacy_path = self._adaptation_path(
                    "label-set-bindings", label_set_hash
                )
                if not legacy_path.is_file():
                    raise GovernanceNotFound
                stored_registration, binding = self._read_v2_binding_document(
                    legacy_path, label_set_hash
                )
                if (
                    stored_registration is not None
                    or binding.run_id != run_id
                    or binding.result_hash != result_hash
                ):
                    raise GovernanceNotFound
                return binding
            stored_registration, binding = self._read_v2_binding_document(
                path, label_set_hash
            )
            if stored_registration != registration_id:
                raise GovernanceServiceError(
                    "persisted v2 label binding registration mismatch"
                )
            return binding

        if not binding_root.is_dir() or binding_root.is_symlink():
            raise GovernanceNotFound
        candidates: dict[str, AdaptationBinding] = {}
        for path in binding_root.iterdir():
            if path.is_symlink() or not path.is_file() or _HASH.fullmatch(path.stem) is None:
                continue
            document = _read_json(path) if path.stat().st_size <= 64 * 1024 else {}
            if (
                path.stem != label_set_hash
                and document.get("labelSetHash") != label_set_hash
            ):
                continue
            _, binding = self._read_v2_binding_document(path, label_set_hash)
            identity = canonical_sha256(
                binding.model_dump(mode="json", by_alias=True)
            )
            candidates[identity] = binding
        if not candidates:
            raise GovernanceNotFound
        if len(candidates) != 1:
            raise GovernanceInvalid
        return next(iter(candidates.values()))

    def _create_v2_adaptation_label_set(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        legacy_fields = {
            "schemaVersion",
            "taskId",
            "inferenceSha256",
            "runId",
            "resultHash",
            "labels",
        }
        modern_fields = legacy_fields | {"targetTaskRegistrationId"}
        if frozenset(payload) not in {
            frozenset(legacy_fields),
            frozenset(modern_fields),
        }:
            raise GovernanceInvalid
        run_id = str(payload.get("runId", ""))
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise GovernanceInvalid
        binding = self._adaptation_binding(run_id)
        if payload.get("resultHash") != binding.result_hash:
            raise GovernanceInvalid
        artifact = self._artifact(binding.artifact_id)
        if artifact.artifact.document.get("bundleSha256") != payload.get(
            "inferenceSha256"
        ):
            raise GovernanceInvalid
        logical: dict[str, Any] = {
            "schemaVersion": payload["schemaVersion"],
            "taskId": payload["taskId"],
            "inferenceSha256": payload["inferenceSha256"],
            "labels": payload["labels"],
        }
        labels = payload.get("labels")
        if not isinstance(labels, list) or len(labels) > 256:
            raise GovernanceInvalid
        logical["positiveCount"] = sum(
            isinstance(row, dict) and row.get("label") == "positive" for row in labels
        )
        logical["negativeCount"] = sum(
            isinstance(row, dict) and row.get("label") == "negative" for row in labels
        )
        logical["labelSetHash"] = canonical_sha256(logical)
        try:
            label_set = TargetLabelSetV2.model_validate(logical)
        except (TypeError, ValueError, ValidationError) as error:
            raise GovernanceInvalid from error
        document = label_set.model_dump(mode="json", by_alias=True)
        registration_id = payload.get("targetTaskRegistrationId")
        if registration_id is not None:
            registration_id = str(registration_id)
            if _TARGET_TASK_REGISTRATION.fullmatch(registration_id) is None:
                raise GovernanceInvalid
        self._persist_immutable_adaptation(
            self._adaptation_path("label-sets", label_set.label_set_hash), document
        )
        if registration_id is None:
            binding_path = self._adaptation_path(
                "label-set-bindings", label_set.label_set_hash
            )
            binding_document = binding.model_dump(mode="json", by_alias=True)
        else:
            binding_hash = self._v2_binding_hash(
                label_set.label_set_hash,
                registration_id,
                binding.run_id,
                binding.result_hash,
            )
            binding_path = self._adaptation_path("label-set-bindings", binding_hash)
            binding_document = {
                "schemaVersion": "TargetLabelSetBinding/1.0",
                "bindingHash": binding_hash,
                "labelSetHash": label_set.label_set_hash,
                "targetTaskRegistrationId": registration_id,
                "runId": binding.run_id,
                "resultHash": binding.result_hash,
                "binding": binding.model_dump(mode="json", by_alias=True),
            }
        self._persist_immutable_adaptation(binding_path, binding_document)
        return document

    def create_adaptation_label_set(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        schema_version = payload.get("schemaVersion")
        if schema_version == "socialgraph-fm.governance-target-label-set/2.0":
            return self._create_v2_adaptation_label_set(payload)
        expected_fields = (
            {"schemaVersion", "runId", "resultHash", "labels"}
            if schema_version == "socialgraph-fm.governance-target-label-set/1.0"
            else {"schemaVersion", "runId", "resultHash", "sidecarReceipt", "labels"}
        )
        if set(payload) != expected_fields:
            raise GovernanceInvalid
        if schema_version not in {
            "socialgraph-fm.governance-target-label-set/1.0",
            "socialgraph-fm.governance-target-label-set/1.1",
        }:
            raise GovernanceInvalid
        run_id = str(payload.get("runId", ""))
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise GovernanceInvalid
        binding = self._adaptation_binding(run_id)
        if payload.get("resultHash") != binding.result_hash:
            raise GovernanceInvalid
        raw_labels = payload.get("labels")
        if not isinstance(raw_labels, list) or len(raw_labels) > 256:
            raise GovernanceInvalid
        try:
            labels = [
                LabelEvidence.model_validate({**raw, "binding": binding})
                for raw in raw_labels
                if isinstance(raw, dict)
            ]
        except ValidationError as error:
            raise GovernanceInvalid from error
        if len(labels) != len(raw_labels):
            raise GovernanceInvalid
        if schema_version == "socialgraph-fm.governance-target-label-set/1.0" and any(
            label.source_type == "imported_sidecar" for label in labels
        ):
            raise GovernanceInvalid
        try:
            receipt = (
                TargetPackageReceipt.model_validate(payload.get("sidecarReceipt"))
                if schema_version == "socialgraph-fm.governance-target-label-set/1.1"
                else None
            )
            label_set = build_target_label_set(
                binding,
                labels,
                schema_version=str(schema_version),
                sidecar_receipt=receipt,
            )
            if receipt is not None:
                artifact = self._artifact(binding.artifact_id)
                validate_sidecar_against_fused_graph(
                    label_set,
                    artifact_document=artifact.artifact.document,
                    node_ids=artifact.node_ids,
                    fused_indptr=np.asarray(artifact.arrays["fused_indptr"]),
                )
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise GovernanceInvalid from error
        document = label_set.model_dump(mode="json", by_alias=True)
        self._persist_immutable_adaptation(
            self._adaptation_path("label-sets", label_set.label_set_hash), document
        )
        return document

    def fit_adaptation_policy(
        self,
        label_set_hash: str,
        fit_request: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        label_path = self._adaptation_path("label-sets", label_set_hash)
        if label_path.is_symlink() or not label_path.is_file():
            raise GovernanceNotFound
        is_v2 = _read_json(label_path).get("schemaVersion") == (
            "socialgraph-fm.governance-target-label-set/2.0"
        )
        if is_v2:
            v2_label_set, binding = self._v2_label_set(label_set_hash, fit_request)
            run_id, artifact_id = binding.run_id, binding.artifact_id
        else:
            if fit_request is not None:
                raise GovernanceInvalid
            label_set = self._label_set(label_set_hash)
            binding = label_set.binding
            run_id, artifact_id = binding.run_id, binding.artifact_id
        outputs = self._outputs(run_id)
        artifact = self._artifact(artifact_id)
        if {"logits", "scores", "ranks", "embeddings"} - set(outputs):
            raise ValueError("frozen run arrays required for adaptation are missing")
        fitted: FittedReviewPolicy | FittedReviewPolicyV2 = (
            fit_target_review_policy_v2(
                v2_label_set,
                binding,
                artifact.node_ids,
                np.asarray(outputs["logits"]),
                np.asarray(outputs["embeddings"]),
                base_scores=np.asarray(outputs["scores"]),
                base_ranks=np.asarray(outputs["ranks"]),
            )
            if is_v2
            else fit_target_review_policy(
                label_set,
                artifact.node_ids,
                np.asarray(outputs["logits"]),
                np.asarray(outputs["embeddings"]),
                base_scores=np.asarray(outputs["scores"]),
                base_ranks=np.asarray(outputs["ranks"]),
            )
        )
        policy = fitted.policy.model_dump(mode="json", by_alias=True)
        comparison = fitted.comparison.model_dump(mode="json", by_alias=True)
        policy_path = self._adaptation_path("policies", fitted.policy.policy_hash)
        comparison_path = self._adaptation_path("comparisons", fitted.policy.policy_hash)
        private_path = (
            self.root
            / "adaptations"
            / "policy-private"
            / f"{fitted.policy.policy_hash}.npz"
        )
        with self._lock:
            self._persist_immutable_adaptation(policy_path, policy)
            self._persist_immutable_adaptation(comparison_path, comparison)
            if private_path.exists():
                arrays = _load_npz(private_path, file_sha256(private_path))
                if (
                    set(arrays) != {"positive_centroid", "negative_centroid"}
                    or not np.array_equal(
                        arrays["positive_centroid"], fitted.positive_centroid
                    )
                    or not np.array_equal(
                        arrays["negative_centroid"], fitted.negative_centroid
                    )
                ):
                    raise ValueError("immutable private policy artifact conflicts")
            else:
                _atomic_npz(
                    private_path,
                    {
                        "positive_centroid": fitted.positive_centroid,
                        "negative_centroid": fitted.negative_centroid,
                    },
                )
        return policy

    def adaptation_policy(self, policy_hash: str) -> dict[str, Any]:
        path = self._adaptation_path("policies", policy_hash)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 512 * 1024:
            raise GovernanceNotFound
        document = _read_json(path)
        is_v2 = document.get("schemaVersion") == "socialgraph-fm.governance-target-review-policy/2.0"
        policy: TargetReviewPolicy | TargetReviewPolicyV2 = (
            TargetReviewPolicyV2.model_validate(document)
            if is_v2
            else TargetReviewPolicy.model_validate(document)
        )
        if policy.policy_hash != policy_hash:
            if isinstance(policy, TargetReviewPolicyV2):
                raise GovernanceServiceError(
                    "persisted v2 adaptation-policy identity mismatch"
                )
            raise ValueError("persisted adaptation-policy identity mismatch")
        if self._adaptation_binding(policy.binding.run_id) != policy.binding:
            if isinstance(policy, TargetReviewPolicyV2):
                raise GovernanceServiceError(
                    "persisted v2 adaptation-policy run binding is stale"
                )
            raise ValueError("persisted adaptation-policy run binding is stale")
        return policy.model_dump(mode="json", by_alias=True)

    def adaptation_comparison(
        self,
        run_id: str,
        policy_hash: str,
        *,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 500:
            raise GovernanceInvalid
        policy_document = self.adaptation_policy(policy_hash)
        is_v2 = policy_document.get("schemaVersion") == "socialgraph-fm.governance-target-review-policy/2.0"
        policy: TargetReviewPolicy | TargetReviewPolicyV2 = (
            TargetReviewPolicyV2.model_validate(policy_document)
            if is_v2
            else TargetReviewPolicy.model_validate(policy_document)
        )
        if isinstance(policy, TargetReviewPolicyV2) and (
            policy.status != "ready"
            or policy.selected_lambda not in _PUBLISHABLE_ADAPTATION_LAMBDAS
        ):
            raise GovernanceAdaptationPolicyNotReady
        if policy.binding.run_id != run_id:
            raise GovernanceInvalid
        path = self._adaptation_path("comparisons", policy_hash)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise GovernanceNotFound
        comparison_document = _read_json(path)
        comparison: AdaptationComparison | AdaptationComparisonV2 = (
            AdaptationComparisonV2.model_validate(comparison_document)
            if is_v2
            else AdaptationComparison.model_validate(comparison_document)
        )
        if comparison.binding != policy.binding or comparison.policy_hash != policy_hash:
            if isinstance(comparison, AdaptationComparisonV2):
                raise GovernanceServiceError(
                    "persisted v2 adaptation comparison binding is stale"
                )
            raise ValueError("persisted adaptation comparison binding mismatch")
        if isinstance(comparison, AdaptationComparisonV2):
            return comparison.model_dump(mode="json", by_alias=True)
        assert isinstance(comparison, AdaptationComparison)
        payload: dict[str, Any] = {
            "schemaVersion": comparison.schema_version,
            "binding": comparison.binding.model_dump(mode="json", by_alias=True),
            "policyHash": policy_hash,
            "total": comparison.total,
            "offset": offset,
            "limit": limit,
            "rows": [
                row.model_dump(mode="json", by_alias=True)
                for row in comparison.rows[offset : offset + limit]
            ],
            "comparisonHash": comparison.comparison_hash,
        }
        return _with_hash(payload, "pageHash")

    def dispatch_get(self, raw_path: str) -> dict[str, Any]:
        parsed = urlsplit(raw_path)
        path = parsed.path
        query = parse_qs(parsed.query, keep_blank_values=True)
        prefix = "/internal/governance"
        if path == f"{prefix}/health":
            return self.health()
        if path == f"{prefix}/capabilities":
            return self.capabilities()
        artifact_match = re.fullmatch(
            rf"{re.escape(prefix)}/artifacts/(governance-artifact-[0-9a-f]{{32}})/preview", path
        )
        if artifact_match:
            return self.artifact_preview(
                artifact_match.group(1), projection=_projection_request(query)
            )
        if path == f"{prefix}/runs":
            offset, limit = _page(query)
            return self.list_runs(offset=offset, limit=limit)
        policy_match = re.fullmatch(
            rf"{re.escape(prefix)}/adaptations/policies/([0-9a-f]{{64}})", path
        )
        if policy_match:
            return self.adaptation_policy(policy_match.group(1))
        comparison_match = re.fullmatch(
            rf"{re.escape(prefix)}/adaptations/runs/(governance-[0-9a-f]{{32}})"
            rf"/policies/([0-9a-f]{{64}})/comparison",
            path,
        )
        if comparison_match:
            offset, limit = _page(query, maximum=500)
            return self.adaptation_comparison(
                comparison_match.group(1),
                comparison_match.group(2),
                offset=offset,
                limit=limit,
            )
        run_prefix = f"{prefix}/runs/"
        if path.startswith(run_prefix):
            suffix = path[len(run_prefix) :]
            evidence_match = re.fullmatch(r"(governance-[0-9a-f]{32})/nodes/(.+)/evidence", suffix)
            if evidence_match:
                try:
                    node_id = unquote(evidence_match.group(2), errors="strict")
                except UnicodeDecodeError as exc:
                    raise GovernanceNotFound from exc
                return self.evidence(evidence_match.group(1), node_id)
            resource_match = re.fullmatch(
                r"(governance-[0-9a-f]{32})/(result|findings|groups|relations|links|preview)",
                suffix,
            )
            if resource_match:
                run_id, resource = resource_match.groups()
                if resource == "result":
                    return self.result(run_id)
                if resource == "preview":
                    return self.run_preview(run_id, projection=_projection_request(query))
                offset, limit = _page(query)
                if resource == "findings":
                    return self.findings(run_id, offset=offset, limit=limit)
                return self.derivations(run_id, resource, offset=offset, limit=limit)
            if RUN_ID_PATTERN.fullmatch(suffix):
                return self._read_state(suffix)
        raise GovernanceNotFound

    def dispatch_post(
        self, raw_path: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        path = urlsplit(raw_path).path
        prefix = "/internal/governance"
        materialize_match = re.fullmatch(
            rf"{re.escape(prefix)}/artifacts/(governance-artifact-[0-9a-f]{{32}})/(?:materialize|validate)",
            path,
        )
        if materialize_match:
            if payload is None:
                raise GovernanceInvalid
            return self.materialize(materialize_match.group(1), payload)
        if path in {f"{prefix}/skills/execute", f"{prefix}/reviewed-cases/index"}:
            if payload is None:
                raise GovernanceInvalid
            try:
                with self._lock:
                    response = self._skills.execute(payload)
            except (KeyError, OSError, TypeError, ValueError, ValidationError) as error:
                raise GovernanceInvalid from error
            if path.endswith("/reviewed-cases/index") and response.command != "index_case":
                raise GovernanceInvalid
            return response.model_dump(mode="json", by_alias=True)
        if path == f"{prefix}/runs":
            if payload is None:
                raise GovernanceInvalid
            return self.create_run(payload)
        if path == f"{prefix}/adaptations/label-sets":
            if payload is None:
                raise GovernanceInvalid
            return self.create_adaptation_label_set(payload)
        fit_match = re.fullmatch(
            rf"{re.escape(prefix)}/adaptations/label-sets/([0-9a-f]{{64}})/policies",
            path,
        )
        if fit_match:
            return self.fit_adaptation_policy(fit_match.group(1), payload)
        action_match = re.fullmatch(
            rf"{re.escape(prefix)}/runs/(governance-[0-9a-f]{{32}})/(cancel|retry)",
            path,
        )
        if action_match:
            run_id, action = action_match.groups()
            return self.cancel_run(run_id) if action == "cancel" else self.retry_run(run_id)
        raise GovernanceNotFound


__all__ = [
    "GovernanceInvalid",
    "GovernanceNotFound",
    "GovernanceNotReady",
    "GovernanceServiceError",
    "GovernanceServingRuntime",
    "GovernanceUnavailable",
]
