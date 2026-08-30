"""Fail-closed loopback serving for the isolated SocialGraph-FM Research channel."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
import zipfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_json, canonical_sha256, file_sha256

from ..core.adapters import BundleInputAdapter, derive_training_selection
from ..core.bundle import CoreGraphBundle, calculate_graph_version_hash
from ..core.structure_features import (
    STRUCTURE_FEATURE_NAMES,
    StructureAlgorithmConfig,
    compute_structure_rows,
)
from .contracts import (
    ACCOUNT_RISK_TASK,
    COLLABORATION_TASK,
    CONTENT_POLICY_TASK,
    RESEARCH_SEED,
    SIGNED_RELATION_TASK,
)
from .routing import SHARED_NULL_ROUTE, task_route_domain, task_route_name
from .wire import (
    WIRE_SCHEMA,
    WireGraphRegistrationEnvelope,
    WireRunEnvelope,
    WireSimilarEnvelope,
    capabilities_payload,
    model_capability,
    scenarios_payload,
)
from .workflow import (
    _atomic_json as _workflow_atomic_json,
)
from .workflow import (
    _bundle_edge_index,
    _load_exported_runtime,
    _read_hashed_document,
    _safe_root,
    _tensor_state_hash,
    load_export_manifest,
    load_registry,
)

_RUN_ID = re.compile(r"^research-[0-9a-f]{32}$")
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_MAX_ARTIFACT_JSON_BYTES = 4 * 1024 * 1024
_MAX_NPZ_BYTES = 512 * 1024 * 1024
_MAX_NPZ_MEMBERS = 2_048
_ATOMIC_READ_ATTEMPTS = 20
_ATOMIC_READ_DELAY_SECONDS = 0.005


class ResearchServiceError(Exception):
    status = 409
    code = "GFM_RESEARCH_CONFLICT"


class ResearchUnavailable(ResearchServiceError):
    status = 503
    code = "GFM_RESEARCH_MODEL_NOT_INSTALLED"


class ResearchNotFound(ResearchServiceError):
    status = 404
    code = "GFM_RESEARCH_NOT_FOUND"


class ResearchInvalid(ResearchServiceError):
    status = 422
    code = "GFM_RESEARCH_REQUEST_INVALID"


class ResearchResultNotReady(ResearchServiceError):
    status = 409
    code = "GFM_RESEARCH_RESULT_NOT_READY"


class ResearchRegistrationFailed(ResearchServiceError):
    status = 409
    code = "GFM_RESEARCH_REGISTRATION_FAILED"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Any) -> None:
    for attempt in range(20):
        try:
            _workflow_atomic_json(path, payload)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.005)
    raise AssertionError("bounded atomic write retry did not return")


def _read_runtime_hashed_document(
    path: Path, *, schema: str, hash_field: str
) -> dict[str, Any]:
    """Read an atomically replaced runtime document across Windows sharing races.

    The worker publishes a complete temporary file with ``os.replace``. A Windows
    reader can briefly lose an open/replace race even though both the old and new
    documents are valid. Only the two transient filesystem failures from that
    race are retried. Parse, schema, and hash failures remain immediately fatal so
    a damaged or tampered document is never accepted.
    """
    for attempt in range(_ATOMIC_READ_ATTEMPTS):
        try:
            return _read_hashed_document(path, schema=schema, hash_field=hash_field)
        except (FileNotFoundError, PermissionError):
            if attempt == _ATOMIC_READ_ATTEMPTS - 1:
                raise
            time.sleep(_ATOMIC_READ_DELAY_SECONDS)
    raise AssertionError("bounded atomic read retry did not return")


def _array_sha256(value) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(b"socialgraph-fm-array-v1\x00")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\x00")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _bounded_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > _MAX_ARTIFACT_JSON_BYTES:
        raise ResearchInvalid("DatasetStore artifact document is missing or oversized")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResearchInvalid("DatasetStore artifact document is invalid") from error
    if not isinstance(payload, dict):
        raise ResearchInvalid("DatasetStore artifact document must be an object")
    return payload


def _safe_artifact_paths(dataset_store_root: Path, artifact_id: str) -> tuple[Path, Path]:
    if _UUID.fullmatch(artifact_id) is None:
        raise ResearchInvalid("uploaded artifact id is not a canonical UUID")
    root = dataset_store_root.resolve(strict=True)
    expected = root / "artifacts" / artifact_id
    try:
        resolved = expected.resolve(strict=True)
    except OSError as error:
        raise ResearchNotFound("uploaded DatasetStore artifact is missing") from error
    if resolved != expected or not resolved.is_relative_to((root / "artifacts").resolve()):
        raise ResearchInvalid("uploaded artifact path uses a link or escapes DatasetStore")
    artifact_json = resolved / "artifact.json"
    graph_npz = resolved / "graph.npz"
    if artifact_json.is_symlink() or graph_npz.is_symlink():
        raise ResearchInvalid("uploaded artifact files must not be links")
    return artifact_json, graph_npz


def _load_verified_npz(path: Path, descriptors: Mapping[str, Mapping[str, Any]]):
    import numpy as np

    if not path.is_file() or path.stat().st_size > _MAX_NPZ_BYTES:
        raise ResearchInvalid("DatasetStore graph.npz is missing or oversized")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > _MAX_NPZ_MEMBERS:
                raise ResearchInvalid("DatasetStore graph.npz has too many arrays")
            if sum(item.file_size for item in members) > _MAX_NPZ_BYTES:
                raise ResearchInvalid("DatasetStore graph.npz expands beyond its bound")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ResearchInvalid("DatasetStore graph.npz is invalid") from error
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ResearchInvalid("DatasetStore graph.npz contains an object array")
    for name, value in arrays.items():
        descriptor = descriptors.get(name)
        if descriptor is None:
            raise ResearchInvalid(f"DatasetStore array lacks descriptor: {name}")
        if (
            descriptor.get("dtype") != value.dtype.str
            or descriptor.get("shape") != list(value.shape)
            or descriptor.get("sha256") != _array_sha256(value)
        ):
            raise ResearchInvalid(f"DatasetStore array identity mismatch: {name}")
    return arrays


def _normal_edge_index(value):
    import numpy as np

    array = np.asarray(value)
    if array.ndim != 2:
        raise ResearchInvalid("uploaded edge index must be rank two")
    if array.shape[0] == 2:
        normalized = array
    elif array.shape[1] == 2:
        normalized = array.T
    else:
        raise ResearchInvalid("uploaded edge index must have shape [2,E] or [E,2]")
    if normalized.dtype.kind not in {"i", "u"}:
        raise ResearchInvalid("uploaded edge index must be integral")
    return normalized.astype("<i8", copy=False)


def _dataset_content_hash(
    artifact: Mapping[str, Any], arrays: Mapping[str, Any]
) -> str:
    descriptors_by_name = {
        item["name"]: item
        for item in artifact.get("arrays", ())
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if (
        len(descriptors_by_name) != len(artifact.get("arrays", ()))
        or any("#" in name for name in descriptors_by_name)
        or set(descriptors_by_name) != set(arrays)
    ):
        raise ResearchInvalid("DatasetStore array inventory is not exact")
    descriptors = []
    for name in sorted(arrays):
        value = arrays[name]
        declared = descriptors_by_name[name]
        actual = {
            "name": name,
            "role": declared.get("role"),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
        }
        if actual != declared:
            raise ResearchInvalid(f"DatasetStore array descriptor mismatch: {name}")
        descriptors.append(actual)
    schema = artifact.get("schemaVersion")
    value = {
        "arrays": descriptors,
        "nodeIdentity": artifact.get("nodeIdentity"),
        "graphSemantics": artifact.get("graphSemantics"),
        "graphVariants": artifact.get("graphVariants", []),
        "featureSchemas": artifact.get("featureSchemas", []),
        "labelSchemas": artifact.get("labelSchemas", []),
        "featureRecipes": artifact.get("featureRecipes", []),
        "splitSets": artifact.get("splitSets", []),
        "taskSpecs": artifact.get("taskSpecs", []),
    }
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        f"socialgraph-fm-dataset-artifact-v{schema}\x00".encode("ascii") + encoded
    ).hexdigest()


def _canonical_graph_hash_from_arrays(
    *, node_count: int, directed: bool, edge_index
) -> str:
    import numpy as np

    edges = np.asarray(edge_index, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(str(node_count).encode("ascii"))
    digest.update(b"\x01" if directed else b"\x00")
    if edges.size:
        order = np.lexsort((edges[1], edges[0]))
        digest.update(np.ascontiguousarray(edges[:, order], dtype="<i8").tobytes())
    return digest.hexdigest()


def _json_attributes(value: Any, *, field: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ResearchInvalid(f"DatasetStore {field} JSON is invalid") from error
    if not isinstance(parsed, dict):
        raise ResearchInvalid(f"DatasetStore {field} JSON must be an object")
    return parsed


def _graph_fact_hash_from_arrays(
    *, artifact: Mapping[str, Any], arrays: Mapping[str, Any], edge_index
) -> str:
    import numpy as np

    required = {
        "node_id_map",
        "node_label",
        "node_type",
        "node_attributes_json",
        "edge_id_map",
        "edge_type",
        "edge_weight",
        "edge_timestamp",
        "edge_directed",
        "edge_attributes_json",
    }
    if not required.issubset(arrays):
        raise ResearchInvalid("DatasetStore handoff arrays are incomplete")
    node_ids = np.asarray(arrays["node_id_map"]).reshape(-1)
    node_count = len(node_ids)
    vectors = {
        name: np.asarray(arrays[name]).reshape(-1)
        for name in required - {"node_id_map"}
    }
    node_fields = ("node_label", "node_type", "node_attributes_json")
    edge_fields = (
        "edge_id_map",
        "edge_type",
        "edge_weight",
        "edge_timestamp",
        "edge_directed",
        "edge_attributes_json",
    )
    if any(len(vectors[name]) != node_count for name in node_fields) or any(
        len(vectors[name]) != edge_index.shape[1] for name in edge_fields
    ):
        raise ResearchInvalid("DatasetStore handoff vector shape is inconsistent")
    node_id_values = tuple(str(value) for value in node_ids.tolist())
    if len(set(node_id_values)) != node_count:
        raise ResearchInvalid("DatasetStore node identities are not unique")
    nodes: list[dict[str, Any]] = []
    for index, node_id in enumerate(node_id_values):
        node_type = str(vectors["node_type"][index])
        nodes.append(
            {
                "id": node_id,
                "label": str(vectors["node_label"][index]),
                "type": node_type or None,
                "attributes": _json_attributes(
                    vectors["node_attributes_json"][index], field="node attributes"
                ),
            }
        )
    edges: list[dict[str, Any]] = []
    edge_ids: set[str] = set()
    for index, (left, right) in enumerate(edge_index.T.tolist()):
        if not 0 <= int(left) < node_count or not 0 <= int(right) < node_count:
            raise ResearchInvalid("DatasetStore edge endpoint is outside node inventory")
        edge_id = str(vectors["edge_id_map"][index])
        if edge_id in edge_ids:
            raise ResearchInvalid("DatasetStore edge identities are not unique")
        edge_ids.add(edge_id)
        edge_type = str(vectors["edge_type"][index])
        timestamp = str(vectors["edge_timestamp"][index])
        weight = float(vectors["edge_weight"][index])
        if math.isinf(weight):
            raise ResearchInvalid("DatasetStore edge weight is not finite")
        raw_directed = int(vectors["edge_directed"][index])
        if raw_directed not in {-1, 0, 1}:
            raise ResearchInvalid("DatasetStore per-edge directedness is invalid")
        edges.append(
            {
                "id": edge_id,
                "source": node_id_values[int(left)],
                "target": node_id_values[int(right)],
                "type": edge_type or None,
                "weight": None if math.isnan(weight) else weight,
                "timestamp": timestamp or None,
                "directed": None if raw_directed < 0 else bool(raw_directed),
                "attributes": _json_attributes(
                    vectors["edge_attributes_json"][index], field="edge attributes"
                ),
            }
        )
    directedness = (artifact.get("graphSemantics") or {}).get("directedness")
    if directedness not in {"directed", "undirected", "mixed", "unspecified"}:
        raise ResearchInvalid("DatasetStore directedness is invalid")
    facts = {
        "directedness": directedness,
        "nodes": sorted(nodes, key=lambda item: item["id"]),
        "edges": sorted(
            edges,
            key=lambda item: (item["id"], item["source"], item["target"]),
        ),
    }
    return hashlib.sha256(
        b"socialgraph-fm-graph-fact-v1\x00" + canonical_json(facts).encode("utf-8")
    ).hexdigest()


def _uploaded_bundle(
    *,
    artifact: Mapping[str, Any],
    arrays: Mapping[str, Any],
    graph_reference: Mapping[str, Any],
) -> tuple[CoreGraphBundle, dict[str, Any]]:
    import numpy as np

    variants = artifact.get("graphVariants") or []
    edge_name = variants[0].get("edgeIndexArray") if variants else "edge_index"
    if not isinstance(edge_name, str) or edge_name not in arrays:
        raise ResearchInvalid("uploaded artifact has no usable graph variant")
    edge_index = _normal_edge_index(arrays[edge_name])
    node_count = int(graph_reference["nodeCount"])
    if edge_index.size and (int(edge_index.min()) < 0 or int(edge_index.max()) >= node_count):
        raise ResearchInvalid("uploaded edge endpoint is outside node inventory")
    identity = artifact.get("nodeIdentity") or {}
    identity_name = identity.get("arrayName", "node_id_map")
    if identity_name in arrays:
        raw_ids = np.asarray(arrays[identity_name]).reshape(-1)
        if raw_ids.shape[0] != node_count or raw_ids.dtype.hasobject:
            raise ResearchInvalid("uploaded node identity array is invalid")
        row_ids = tuple(str(item) for item in raw_ids.tolist())
    else:
        row_ids = tuple(str(index) for index in range(node_count))
    if (
        len(set(row_ids)) != node_count
        or any(not item or len(item) > 300 for item in row_ids)
    ):
        raise ResearchInvalid("uploaded node identities must be unique bounded strings")
    sorted_ids = tuple(sorted(row_ids))
    row_to_sorted = {row: index for index, row in enumerate(sorted_ids)}
    directed = bool((artifact.get("graphSemantics") or {}).get("directed", False))
    semantic_pairs: list[tuple[str, str]] = []
    orientation_counts: dict[tuple[str, str], int] = {}
    self_loop_count = 0
    for left, right in edge_index.T.tolist():
        source = row_ids[int(left)]
        target = row_ids[int(right)]
        if source == target:
            self_loop_count += 1
            continue
        orientation = (source, target)
        orientation_counts[orientation] = orientation_counts.get(orientation, 0) + 1
        semantic_pairs.append(
            (source, target) if directed or source < target else (target, source)
        )
    unique_pairs = sorted(set(semantic_pairs))
    duplicate_count = (
        len(semantic_pairs) - len(unique_pairs)
        if directed
        else sum(max(0, count - 1) for count in orientation_counts.values())
    )
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": directed,
        "nodes": [
            {"id": node_id, "index": index} for index, node_id in enumerate(sorted_ids)
        ],
        "edges": [
            {
                "sourceId": source,
                "targetId": target,
                "edgeType": "uploaded-relation",
                "weight": 1.0,
            }
            for source, target in unique_pairs
        ],
        "nodeFeatures": [],
        "structuralFeatures": None,
        "source": {
            "sourceName": "DatasetStore uploaded graph",
            "sourceUri": f"urn:socialgraph-fm:dataset-artifact:{artifact['id']}",
            "citation": "User-provided graph; inference only",
            "sourceSha256": str(artifact["contentHash"]),
        },
        "splitManifest": {"strategy": "all-visible-training", "assignments": []},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    preliminary = CoreGraphBundle.model_validate(payload)
    rows = compute_structure_rows(
        preliminary,
        visible_edge_indices=tuple(range(len(preliminary.edges))),
        config=StructureAlgorithmConfig.fixed(),
    )
    payload["structuralFeatures"] = {
        "names": list(STRUCTURE_FEATURE_NAMES),
        "values": rows.tolist(),
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    metadata = {
        "rawEdgeCount": int(edge_index.shape[1]),
        "semanticEdgeCount": len(unique_pairs),
        "selfLoopCount": self_loop_count,
        "duplicateCount": duplicate_count,
        "directed": directed,
        "rowToSortedHash": canonical_sha256(row_to_sorted),
    }
    return CoreGraphBundle.model_validate(payload), metadata


def _facts(bundle: CoreGraphBundle, index: int) -> dict[str, Any]:
    if bundle.structural_features is None:
        raise ResearchInvalid("structure facts are unavailable")
    row = bundle.structural_features.values[index]
    by_name = dict(zip(bundle.structural_features.names, row, strict=True))
    return {
        "degree": max(0, round(by_name["degree"])),
        "inDegree": max(0, round(by_name["in-degree"])),
        "outDegree": max(0, round(by_name["out-degree"])),
        "pagerank": min(1.0, max(0.0, float(by_name["pagerank"]))),
        "clustering": min(1.0, max(0.0, float(by_name["clustering"]))),
        "coreNumber": max(0, round(by_name["k-core"])),
    }


class ResearchServingRuntime:
    def __init__(self, research_root: str | Path, dataset_store_root: str | Path | None) -> None:
        self.root = _safe_root(research_root)
        self.dataset_store_root = (
            None if dataset_store_root is None else Path(dataset_store_root).expanduser().resolve()
        )
        self._state_lock = RLock()
        self._model_lock = RLock()
        self._scenario_cache_lock = RLock()
        self._loaded = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="socialgraph-research"
        )
        self._recover_runs()
        self._recover_registrations()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    @staticmethod
    def _status_payload(
        *,
        run_id: str,
        request_hash: str,
        status: str,
        created_at: str,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schemaVersion": WIRE_SCHEMA,
            "runId": run_id,
            "requestHash": request_hash,
            "status": status,
            "progress": {"queued": 0, "running": 10, "succeeded": 100, "failed": 100}[
                status
            ],
            "createdAt": created_at,
            "updatedAt": _utc_now(),
            "errorCode": error_code,
        }
        payload["stateHash"] = canonical_sha256(payload)
        return payload

    def _recover_runs(self) -> None:
        runs_root = self.root / "serving/runs"
        if not runs_root.is_dir():
            return
        for path in runs_root.iterdir():
            if not path.is_dir() or _RUN_ID.fullmatch(path.name) is None:
                continue
            status_path = path / "status.json"
            try:
                status = _read_hashed_document(
                    status_path, schema=WIRE_SCHEMA, hash_field="stateHash"
                )
            except (OSError, ValueError):
                continue
            if status["status"] in {"queued", "running"}:
                failed = self._status_payload(
                    run_id=path.name,
                    request_hash=status["requestHash"],
                    status="failed",
                    created_at=status["createdAt"],
                    error_code="GFM_RESEARCH_RUN_INTERRUPTED",
                )
                _atomic_json(status_path, failed)

    @staticmethod
    def _registration_status_payload(
        *,
        graph_version_id: str,
        request_hash: str,
        status: str,
        created_at: str,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.research-registration-state/1.0",
            "graphVersionId": graph_version_id,
            "requestHash": request_hash,
            "status": status,
            "createdAt": created_at,
            "updatedAt": _utc_now(),
            "errorCode": error_code,
        }
        payload["stateHash"] = canonical_sha256(payload)
        return payload

    @staticmethod
    def _registration_response(
        envelope: WireGraphRegistrationEnvelope,
        registry: Mapping[str, Any],
        *,
        adapter_status: str,
    ) -> dict[str, Any]:
        reference = envelope.graph_reference
        payload: dict[str, Any] = {
            "schemaVersion": WIRE_SCHEMA,
            "graphVersionId": reference.graph_version_id,
            "graphVersionHash": reference.graph_version_hash,
            "modelVersionId": registry["modelVersionId"],
            "modelVersionHash": registry["modelVersionHash"],
            "adapterStatus": adapter_status,
            "compatibleTaskIds": list(envelope.compatible_task_ids),
            "auxiliaryCapabilities": list(envelope.auxiliary_capabilities),
        }
        payload["registrationHash"] = canonical_sha256(payload)
        return payload

    def _registration_root(
        self, graph_version_id: str, model_version_hash: str
    ) -> Path:
        return (
            self.root
            / "serving/registration-jobs"
            / canonical_sha256(
                {
                    "graphVersionId": graph_version_id,
                    "modelVersionHash": model_version_hash,
                }
            )
        )

    def _recover_registrations(self) -> None:
        jobs_root = self.root / "serving/registration-jobs"
        if not jobs_root.is_dir():
            return
        for job_root in jobs_root.iterdir():
            if not job_root.is_dir() or re.fullmatch(r"[0-9a-f]{64}", job_root.name) is None:
                continue
            try:
                status = _read_hashed_document(
                    job_root / "status.json",
                    schema="socialgraph-fm.research-registration-state/1.0",
                    hash_field="stateHash",
                )
                envelope = WireGraphRegistrationEnvelope.model_validate_json(
                    (job_root / "request.json").read_bytes()
                )
            except (OSError, ValueError, ValidationError):
                continue
            expected_root = self._registration_root(
                envelope.graph_reference.graph_version_id,
                envelope.expected_model.model_version_hash,
            )
            if job_root != expected_root:
                continue
            if status["status"] in {"pending_registration", "running"}:
                self._executor.submit(
                    self._execute_registration,
                    envelope,
                    status["createdAt"],
                    status["requestHash"],
                )

    def capabilities(self) -> dict[str, Any]:
        payload = capabilities_payload(self.root)
        if payload["researchServingReady"]:
            self._published()
        return payload

    def scenarios(self) -> dict[str, Any]:
        payload = scenarios_payload(self.root)
        if any(item["enabled"] for item in payload["scenarios"]):
            self._published()
        return payload

    def _published(self):
        try:
            registry = load_registry(self.root)
            export = load_export_manifest(self.root)
        except FileNotFoundError as error:
            raise ResearchUnavailable("SocialGraph-FM Research is not published") from error
        corpus_identity = (export.get("corpusKind"), export.get("testOnly"))
        if (
            corpus_identity not in {("real", False), ("test-fixture", True)}
            or (registry.get("corpusKind"), registry.get("testOnly"))
            != corpus_identity
        ):
            raise ResearchUnavailable("published corpus identity is invalid")
        if corpus_identity == ("test-fixture", True) and os.environ.get(
            "SOCIALGRAPH_FM_INTERNAL_TEST_FIXTURE"
        ) != "1":
            raise ResearchUnavailable("test fixture exports cannot be served")
        return registry, export

    def _model_runtime(self):
        with self._model_lock:
            if self._loaded is None:
                registry, export = self._published()
                loaded = _load_exported_runtime(self.root, device="cpu")
                loaded_export, checkpoint, corpus, documents, model, adapters = loaded
                if (
                    loaded_export != export
                    or registry["modelVersionId"] != export["modelVersionId"]
                    or registry["modelVersionHash"] != export["modelVersionHash"]
                    or registry["artifactHash"] != export["artifactHash"]
                    or registry["checkpointSha256"] != export["checkpointSha256"]
                ):
                    raise ResearchUnavailable("published export binding is invalid")
                self._loaded = (
                    registry,
                    export,
                    checkpoint,
                    corpus,
                    documents,
                    model,
                    adapters,
                )
            return self._loaded

    @staticmethod
    def _verify_expected_model(expected: Mapping[str, Any], registry: Mapping[str, Any]) -> None:
        if expected != model_capability(dict(registry)):
            raise ResearchServiceError("expected model does not match current registry")

    def graph_preview(self, scenario_id: str) -> dict[str, Any]:
        registry, _export = self._published()
        scenario = next(
            (item for item in registry["scenarios"] if item["scenarioId"] == scenario_id), None
        )
        if scenario is None:
            raise ResearchNotFound("research scenario is not registered")
        path = (self.root / registry["exportManifestPath"]).parent / scenario["previewPath"]
        if not path.resolve().is_relative_to((self.root / "exports/research").resolve()):
            raise ResearchServiceError("scenario preview path escapes export root")
        if file_sha256(path) != scenario["previewSha256"]:
            raise ResearchServiceError("scenario preview file identity mismatch")
        payload = _read_hashed_document(path, schema=WIRE_SCHEMA, hash_field="previewHash")
        if (
            payload["scenarioId"] != scenario_id
            or payload["graphVersionHash"] != scenario["graphVersionHash"]
            or payload["modelVersionId"] != registry["modelVersionId"]
            or payload["modelVersionHash"] != registry["modelVersionHash"]
        ):
            raise ResearchServiceError("scenario preview registry binding mismatch")
        return payload

    def _artifact(self, reference: Mapping[str, Any]):
        import numpy as np

        if self.dataset_store_root is None:
            raise ResearchUnavailable("DatasetStore root is not configured")
        artifact_json, graph_npz = _safe_artifact_paths(
            self.dataset_store_root, str(reference["artifactId"])
        )
        artifact = _bounded_json(artifact_json)
        if (
            artifact.get("schemaVersion") not in {"2.1", "2.2"}
            or artifact.get("sourceFormat") != "graph_version_target_domain"
            or artifact.get("id") != reference["artifactId"]
        ):
            raise ResearchInvalid("uploaded artifact identity differs from graph reference")
        descriptors = {
            item["name"]: item
            for item in artifact.get("arrays", ())
            if isinstance(item, dict) and isinstance(item.get("name"), str) and "#" not in item["name"]
        }
        arrays = _load_verified_npz(graph_npz, descriptors)
        variants = artifact.get("graphVariants") or []
        edge_name = variants[0].get("edgeIndexArray") if variants else "edge_index"
        if not isinstance(edge_name, str) or edge_name not in arrays:
            raise ResearchInvalid("uploaded artifact has no usable graph variant")
        edge_index = _normal_edge_index(arrays[edge_name])
        node_identity = artifact.get("nodeIdentity") or {}
        node_id_name = node_identity.get("arrayName", "node_id_map")
        if not isinstance(node_id_name, str) or node_id_name not in arrays:
            raise ResearchInvalid("uploaded artifact has no node identity array")
        node_count = int(arrays[node_id_name].reshape(-1).shape[0])
        edge_count = int(edge_index.shape[1])
        semantics = artifact.get("graphSemantics") or {}
        declared_directed = semantics.get("directed")
        num_nodes = np.asarray(arrays.get("num_nodes"))
        directed_array = np.asarray(arrays.get("directed"))
        variant_directed = variants[0].get("directed") if variants else None
        if (
            not isinstance(declared_directed, bool)
            or not isinstance(variant_directed, bool)
            or declared_directed != variant_directed
            or num_nodes.size != 1
            or num_nodes.dtype.kind not in {"i", "u"}
            or int(num_nodes.reshape(-1)[0]) != node_count
            or directed_array.size != 1
            or directed_array.dtype.kind != "b"
            or bool(directed_array.reshape(-1)[0]) != declared_directed
        ):
            raise ResearchInvalid("uploaded graph semantics differ from verified arrays")
        directed = declared_directed
        actual_content_hash = _dataset_content_hash(artifact, arrays)
        actual_graph_hash = _canonical_graph_hash_from_arrays(
            node_count=node_count,
            directed=directed,
            edge_index=edge_index,
        )
        actual_fact_hash = _graph_fact_hash_from_arrays(
            artifact=artifact,
            arrays=arrays,
            edge_index=edge_index,
        )
        handoff = (artifact.get("rawManifest") or {}).get("graphVersionHandoff") or {}
        if (
            actual_content_hash != artifact.get("contentHash")
            or actual_content_hash != reference["artifactHash"]
            or actual_graph_hash != artifact.get("canonicalGraphHash")
            or actual_graph_hash != reference["graphVersionHash"]
            or actual_fact_hash != handoff.get("graphFactHash")
            or actual_fact_hash != reference["graphFactHash"]
            or node_count != int(reference["nodeCount"])
            or edge_count != int(reference["edgeCount"])
            or node_count != int((artifact.get("profile") or {}).get("nodeCount") or -1)
            or edge_count != int((artifact.get("profile") or {}).get("edgeCount") or -1)
        ):
            raise ResearchInvalid("uploaded artifact hash or count differs from graph reference")
        return artifact, arrays

    def register_graph(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            envelope = WireGraphRegistrationEnvelope.model_validate(payload)
        except ValidationError as error:
            raise ResearchInvalid("uploaded graph registration envelope is invalid") from error
        registry, _export = self._published()
        self._verify_expected_model(
            envelope.expected_model.model_dump(mode="json", by_alias=True), registry
        )
        request_payload = envelope.model_dump(mode="json", by_alias=True)
        request_hash = canonical_sha256(request_payload)
        job_root = self._registration_root(
            envelope.graph_reference.graph_version_id,
            envelope.expected_model.model_version_hash,
        )
        status_path = job_root / "status.json"
        request_path = job_root / "request.json"
        result_path = job_root / "result.json"
        with self._state_lock:
            if status_path.is_file():
                try:
                    persisted = WireGraphRegistrationEnvelope.model_validate_json(
                        request_path.read_bytes()
                    )
                    status = _read_hashed_document(
                        status_path,
                        schema="socialgraph-fm.research-registration-state/1.0",
                        hash_field="stateHash",
                    )
                except (OSError, ValueError, ValidationError) as error:
                    raise ResearchServiceError(
                        "persisted graph registration is invalid"
                    ) from error
                if persisted != envelope or status["requestHash"] != request_hash:
                    raise ResearchServiceError(
                        "graph version registration binding conflict"
                    )
                if status["status"] == "failed":
                    raise ResearchRegistrationFailed(
                        "uploaded graph registration did not complete"
                    )
                if status["status"] == "ready":
                    return _read_hashed_document(
                        result_path,
                        schema=WIRE_SCHEMA,
                        hash_field="registrationHash",
                    )
                return self._registration_response(
                    envelope, registry, adapter_status="pending_registration"
                )
            job_root.mkdir(parents=True, exist_ok=True)
            _atomic_json(request_path, request_payload)
            created_at = _utc_now()
            _atomic_json(
                status_path,
                self._registration_status_payload(
                    graph_version_id=envelope.graph_reference.graph_version_id,
                    request_hash=request_hash,
                    status="pending_registration",
                    created_at=created_at,
                ),
            )
            self._executor.submit(
                self._execute_registration,
                envelope,
                created_at,
                request_hash,
            )
            return self._registration_response(
                envelope, registry, adapter_status="pending_registration"
            )

    def _execute_registration(
        self,
        envelope: WireGraphRegistrationEnvelope,
        created_at: str,
        request_hash: str,
    ) -> None:
        graph_version_id = envelope.graph_reference.graph_version_id
        job_root = self._registration_root(
            graph_version_id, envelope.expected_model.model_version_hash
        )
        status_path = job_root / "status.json"
        try:
            with self._state_lock:
                _atomic_json(
                    status_path,
                    self._registration_status_payload(
                        graph_version_id=graph_version_id,
                        request_hash=request_hash,
                        status="running",
                        created_at=created_at,
                    ),
                )
            result = self._complete_registration(envelope)
            with self._state_lock:
                _atomic_json(job_root / "result.json", result)
                _atomic_json(
                    status_path,
                    self._registration_status_payload(
                        graph_version_id=graph_version_id,
                        request_hash=request_hash,
                        status="ready",
                        created_at=created_at,
                    ),
                )
        except Exception as error:  # noqa: BLE001 - worker boundary is fail-closed
            error_code = (
                error.code
                if isinstance(error, ResearchServiceError)
                else "GFM_RESEARCH_REGISTRATION_FAILED"
            )
            try:
                with self._state_lock:
                    _atomic_json(
                        status_path,
                        self._registration_status_payload(
                            graph_version_id=graph_version_id,
                            request_hash=request_hash,
                            status="failed",
                            created_at=created_at,
                            error_code=error_code,
                        ),
                    )
            except (OSError, ValueError):
                return

    def _complete_registration(
        self, envelope: WireGraphRegistrationEnvelope
    ) -> dict[str, Any]:
        registry, _export, checkpoint, _corpus, _documents, model, _adapters = self._model_runtime()
        self._verify_expected_model(
            envelope.expected_model.model_dump(mode="json", by_alias=True), registry
        )
        reference = envelope.graph_reference.model_dump(mode="json", by_alias=True)
        artifact, arrays = self._artifact(reference)
        bundle, graph_metadata = _uploaded_bundle(
            artifact=artifact, arrays=arrays, graph_reference=reference
        )
        if "similar-nodes" in envelope.auxiliary_capabilities and (
            len(bundle.nodes) < 5 or len(bundle.edges) < 4
        ):
            raise ResearchInvalid("uploaded graph is too small for structural similarity")
        if COLLABORATION_TASK in envelope.compatible_task_ids and (
            graph_metadata["directed"]
            or graph_metadata["selfLoopCount"]
            or graph_metadata["duplicateCount"]
            or not 20 <= len(bundle.nodes) <= 50_000
            or len(bundle.edges) > 1_500_000
            or len(bundle.nodes) * (len(bundle.nodes) - 1) // 2 - len(bundle.edges) < 10
        ):
            raise ResearchInvalid("uploaded graph violates collaboration completion contract")
        adapter = BundleInputAdapter(bundle, mode="training", multi_hot_buckets=256)
        email_state = checkpoint["adapterStates"]["email-eu-core"]
        adapter.load_state_dict(email_state, strict=True)
        adapter.eval()
        model.eval()
        import numpy as np
        import torch

        with torch.inference_mode():
            encoded = model.encode_domain(
                adapter(), _bundle_edge_index(bundle, visible_only=True), None
            )
            head_embeddings = encoded.cpu().numpy().astype("<f4")
            embeddings = (
                torch.nn.functional.normalize(encoded, dim=-1)
                .cpu()
                .numpy()
                .astype("<f4")
            )
        adapter_state_hash = _tensor_state_hash(adapter.state_dict())
        cache_key = canonical_sha256(
            {
                "modelVersionHash": registry["modelVersionHash"],
                "graphVersionId": reference["graphVersionId"],
                "graphVersionHash": reference["graphVersionHash"],
                "graphFactHash": reference["graphFactHash"],
                "artifactId": reference["artifactId"],
                "artifactHash": reference["artifactHash"],
                "adapterSchemaHash": adapter.schema.adapter_schema_hash,
                "adapterStateHash": adapter_state_hash,
                "route": SHARED_NULL_ROUTE,
            }
        )
        target = self.root / "serving/uploaded" / cache_key
        target.parent.mkdir(parents=True, exist_ok=True)
        registration_base: dict[str, Any] = {
            "schemaVersion": WIRE_SCHEMA,
            "graphVersionId": reference["graphVersionId"],
            "graphVersionHash": reference["graphVersionHash"],
            "modelVersionId": registry["modelVersionId"],
            "modelVersionHash": registry["modelVersionHash"],
            "adapterStatus": "ready",
            "compatibleTaskIds": list(envelope.compatible_task_ids),
            "auxiliaryCapabilities": list(envelope.auxiliary_capabilities),
        }
        registration_base["registrationHash"] = canonical_sha256(registration_base)
        if not target.exists():
            staging = target.parent / f".{cache_key}.{uuid.uuid4().hex}.staging"
            staging.mkdir()
            try:
                bundle_path = staging / "bundle.json"
                _atomic_json(bundle_path, bundle.model_dump(mode="json", by_alias=True))
                embedding_path = staging / "embeddings.npz"
                np.savez_compressed(
                    embedding_path,
                    embeddings=embeddings,
                    head_embeddings=head_embeddings,
                    node_ids=np.asarray([node.id for node in bundle.nodes], dtype="U300"),
                )
                manifest: dict[str, Any] = {
                    "schemaVersion": "socialgraph-fm.research-upload-cache/1.0",
                    "cacheKey": cache_key,
                    "graphReference": reference,
                    "modelVersionId": registry["modelVersionId"],
                    "modelVersionHash": registry["modelVersionHash"],
                    "bundleGraphVersionHash": bundle.graph_version_hash,
                    "bundleSha256": file_sha256(bundle_path),
                    "embeddingSha256": file_sha256(embedding_path),
                    "embeddingShape": list(embeddings.shape),
                    "adapterSchema": adapter.schema.model_dump(mode="json", by_alias=True),
                    "adapterStateHash": adapter_state_hash,
                    "route": SHARED_NULL_ROUTE,
                    "graphMetadata": graph_metadata,
                    "registration": registration_base,
                }
                manifest["manifestHash"] = canonical_sha256(manifest)
                _atomic_json(staging / "manifest.json", manifest)
                os.replace(staging, target)
            except Exception:
                import shutil

                shutil.rmtree(staging, ignore_errors=True)
                raise
        manifest = self._uploaded_manifest(
            reference["graphVersionId"], cache_key=cache_key
        )
        if manifest["registration"] != registration_base:
            raise ResearchServiceError("uploaded graph cache binding conflict")
        index: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.research-upload-index/1.0",
            "graphVersionId": reference["graphVersionId"],
            "cacheKey": cache_key,
            "manifestHash": manifest["manifestHash"],
        }
        index["indexHash"] = canonical_sha256(index)
        _atomic_json(
            self.root / "serving/uploaded-index" / f"{canonical_sha256(reference['graphVersionId'])}.json",
            index,
        )
        return registration_base

    def _uploaded_manifest(self, graph_version_id: str, *, cache_key: str | None = None):
        expected_manifest_hash = None
        if cache_key is None:
            index = _read_hashed_document(
                self.root
                / "serving/uploaded-index"
                / f"{canonical_sha256(graph_version_id)}.json",
                schema="socialgraph-fm.research-upload-index/1.0",
                hash_field="indexHash",
            )
            if index["graphVersionId"] != graph_version_id:
                raise ResearchServiceError("uploaded graph index identity mismatch")
            cache_key = index["cacheKey"]
            expected_manifest_hash = index["manifestHash"]
        if re.fullmatch(r"[0-9a-f]{64}", cache_key) is None:
            raise ResearchServiceError("uploaded cache key is invalid")
        root = self.root / "serving/uploaded" / cache_key
        manifest = _read_hashed_document(
            root / "manifest.json",
            schema="socialgraph-fm.research-upload-cache/1.0",
            hash_field="manifestHash",
        )
        if manifest["cacheKey"] != cache_key:
            raise ResearchServiceError("uploaded cache manifest identity mismatch")
        if (
            expected_manifest_hash is not None
            and manifest["manifestHash"] != expected_manifest_hash
        ):
            raise ResearchServiceError("uploaded cache index binding mismatch")
        if file_sha256(root / "bundle.json") != manifest["bundleSha256"] or file_sha256(
            root / "embeddings.npz"
        ) != manifest["embeddingSha256"]:
            raise ResearchServiceError("uploaded cache artifact hash mismatch")
        return manifest

    @staticmethod
    def _verify_uploaded_model(
        manifest: Mapping[str, Any], registry: Mapping[str, Any]
    ) -> None:
        if (
            manifest.get("modelVersionId") != registry["modelVersionId"]
            or manifest.get("modelVersionHash") != registry["modelVersionHash"]
            or manifest.get("route") != SHARED_NULL_ROUTE
        ):
            raise ResearchServiceError(
                "uploaded graph must be registered for the current model"
            )

    @staticmethod
    def _embedding_array(path: Path, bundle: CoreGraphBundle, name: str):
        import numpy as np

        try:
            with np.load(path, allow_pickle=False) as archive:
                if name not in archive.files or "node_ids" not in archive.files:
                    raise ResearchServiceError("embedding archive is incomplete")
                values = np.asarray(archive[name], dtype="<f4")
                node_ids = tuple(str(item) for item in archive["node_ids"].tolist())
        except (OSError, ValueError) as error:
            raise ResearchServiceError("embedding archive is invalid") from error
        if (
            values.shape != (len(bundle.nodes), 128)
            or not np.isfinite(values).all()
            or node_ids != tuple(node.id for node in bundle.nodes)
        ):
            raise ResearchServiceError("embedding shape or node order mismatch")
        return values

    def _scenario_head_embeddings(
        self,
        *,
        registry: Mapping[str, Any],
        export: Mapping[str, Any],
        scenario: Mapping[str, Any],
        bundle: CoreGraphBundle,
        model,
        adapter: BundleInputAdapter,
        domain: str,
    ):
        import numpy as np
        import torch

        route = task_route_name(str(scenario["taskId"]), domain)
        if scenario.get("route") != route:
            raise ResearchServiceError("registered scenario route contract mismatch")
        route_domain = task_route_domain(str(scenario["taskId"]), domain)

        cache_key = canonical_sha256(
            {
                "schemaVersion": "socialgraph-fm.research-head-embedding-cache/1.0",
                "modelVersionHash": registry["modelVersionHash"],
                "checkpointSha256": export["checkpointSha256"],
                "graphVersionHash": scenario["graphVersionHash"],
                "domain": domain,
                "route": route,
            }
        )
        # Keep the writable serving path bounded on Windows; the manifest retains the full key.
        target = self.root / "serving/e" / cache_key[:24]
        manifest_path = target / "manifest.json"
        with self._scenario_cache_lock:
            if not manifest_path.is_file():
                staging = target.parent / f".s-{uuid.uuid4().hex[:8]}"
                staging.parent.mkdir(parents=True, exist_ok=True)
                staging.mkdir()
                try:
                    model.eval()
                    adapter.eval()
                    with torch.inference_mode():
                        values = (
                            model.encode_domain(
                                adapter(),
                                _bundle_edge_index(bundle, visible_only=True),
                                route_domain,
                            )
                            .cpu()
                            .numpy()
                            .astype("<f4")
                        )
                    archive_path = staging / "head-embeddings.npz"
                    np.savez_compressed(
                        archive_path,
                        head_embeddings=values,
                        node_ids=np.asarray(
                            [node.id for node in bundle.nodes], dtype="U500"
                        ),
                    )
                    manifest: dict[str, Any] = {
                        "schemaVersion": "socialgraph-fm.research-head-embedding-cache/1.0",
                        "cacheKey": cache_key,
                        "modelVersionHash": registry["modelVersionHash"],
                        "checkpointSha256": export["checkpointSha256"],
                        "graphVersionHash": scenario["graphVersionHash"],
                        "domain": domain,
                        "route": route,
                        "nodeCount": len(bundle.nodes),
                        "width": int(values.shape[1]),
                        "archiveSha256": file_sha256(archive_path),
                    }
                    manifest["manifestHash"] = canonical_sha256(manifest)
                    _atomic_json(staging / "manifest.json", manifest)
                    os.replace(staging, target)
                except Exception:
                    import shutil

                    shutil.rmtree(staging, ignore_errors=True)
                    raise
            manifest = _read_hashed_document(
                manifest_path,
                schema="socialgraph-fm.research-head-embedding-cache/1.0",
                hash_field="manifestHash",
            )
            expected = {
                "cacheKey": cache_key,
                "modelVersionHash": registry["modelVersionHash"],
                "checkpointSha256": export["checkpointSha256"],
                "graphVersionHash": scenario["graphVersionHash"],
                "domain": domain,
                "route": route,
                "nodeCount": len(bundle.nodes),
                "width": 128,
            }
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ResearchServiceError("scenario embedding cache binding mismatch")
            archive_path = target / "head-embeddings.npz"
            if file_sha256(archive_path) != manifest["archiveSha256"]:
                raise ResearchServiceError("scenario embedding cache hash mismatch")
            return self._embedding_array(archive_path, bundle, "head_embeddings")

    def _graph_runtime(
        self,
        reference: Mapping[str, Any],
        scenario_id: str | None,
        task_id: str,
    ):
        registry, export, checkpoint, _corpus, documents, model, adapters = self._model_runtime()
        if reference["kind"] == "registered-scenario":
            scenario = next(
                (
                    item
                    for item in registry["scenarios"]
                    if item["scenarioId"] == scenario_id
                ),
                None,
            )
            if scenario is None or (
                scenario["graphVersionId"] != reference["graphVersionId"]
                or scenario["graphVersionHash"] != reference["graphVersionHash"]
                or scenario["taskId"] != task_id
            ):
                raise ResearchServiceError("registered scenario graph binding mismatch")
            domain = scenario["domain"]
            if domain not in documents:
                raise ResearchServiceError("registered scenario domain is unavailable")
            bundle = documents[domain][0]
            if bundle.graph_version_hash != scenario["graphVersionHash"]:
                raise ResearchServiceError("registered scenario bundle identity mismatch")
            encoded = self._scenario_head_embeddings(
                registry=registry,
                export=export,
                scenario=scenario,
                bundle=bundle,
                model=model,
                adapter=adapters[domain],
                domain=domain,
            )
            return registry, checkpoint, model, bundle, encoded, domain
        manifest = self._uploaded_manifest(reference["graphVersionId"])
        self._verify_uploaded_model(manifest, registry)
        if manifest["graphReference"] != dict(reference):
            raise ResearchServiceError("uploaded graph reference differs from registration")
        cache_root = self.root / "serving/uploaded" / manifest["cacheKey"]
        bundle = CoreGraphBundle.model_validate_json((cache_root / "bundle.json").read_bytes())
        encoded = self._embedding_array(
            cache_root / "embeddings.npz", bundle, "head_embeddings"
        )
        return registry, checkpoint, model, bundle, encoded, None

    @staticmethod
    def _calibrated_scores(logits, calibrator: Mapping[str, Any]):
        import torch

        binary = logits[:, 1] - logits[:, 0] if logits.ndim == 2 else logits
        calibrated = bool(calibrator["adequate"])
        values = (
            (binary + float(calibrator["bias"])) / float(calibrator["temperature"])
            if calibrated
            else binary
        )
        return torch.sigmoid(values), calibrated

    @classmethod
    def _review_scores(cls, logits, calibrator: Mapping[str, Any], task_id: str):
        scores, calibrated = cls._calibrated_scores(logits, calibrator)
        # The signed head's positive target is support. Governance review ranks
        # opposition, matching the offline opposition-class AUPRC.
        if task_id == SIGNED_RELATION_TASK:
            scores = 1.0 - scores
        return scores, calibrated

    def _result(self, run_id: str, envelope: WireRunEnvelope) -> dict[str, Any]:
        import torch

        request = envelope.request
        reference = envelope.graph_reference.model_dump(mode="json", by_alias=True)
        registry, checkpoint, model, bundle, encoded_array, _domain = self._graph_runtime(
            reference, request.scenario_id, request.task_id
        )
        if request.model_version_id != registry["modelVersionId"]:
            raise ResearchServiceError("run model version is stale")
        self._verify_expected_model(
            envelope.expected_model.model_dump(mode="json", by_alias=True), registry
        )
        task_id = request.task_id
        if reference["kind"] == "uploaded-artifact" and task_id != COLLABORATION_TASK:
            raise ResearchServiceError("uploaded graphs support collaboration completion only")
        by_id = {node.id: node.index for node in bundle.nodes}
        encoded = torch.from_numpy(encoded_array)
        model.eval()
        with torch.inference_mode():
            scope = request.target_scope
            entity_ids: list[tuple[str, ...]]
            if scope.kind == "nodes":
                missing = [item for item in scope.node_ids if item not in by_id]
                if missing:
                    raise ResearchInvalid("node target is absent from graph")
                entity_ids = [(item,) for item in scope.node_ids]
                indices = torch.tensor([by_id[item] for item in scope.node_ids])
                head = (
                    model.content_policy_head
                    if task_id == CONTENT_POLICY_TASK
                    else model.account_risk_head
                )
                logits = head(encoded[indices])
                entity_type = "node"
            elif scope.kind == "directed-node-pairs":
                if any(left not in by_id or right not in by_id for left, right in scope.pairs):
                    raise ResearchInvalid("directed edge target is absent from graph")
                entity_ids = [tuple(item) for item in scope.pairs]
                pairs = torch.tensor([(by_id[left], by_id[right]) for left, right in scope.pairs])
                logits = model.signed_edge_head(encoded, pairs)
                entity_type = "directed-edge"
            else:
                if scope.anchor_node_id not in by_id:
                    raise ResearchInvalid("collaboration anchor is absent from graph")
                anchor = by_id[scope.anchor_node_id]
                neighbors = {anchor}
                excluded_edge_indices = (
                    tuple(range(len(bundle.edges)))
                    if reference["kind"] == "uploaded-artifact"
                    else derive_training_selection(bundle).visible_edge_indices
                )
                for edge_index in excluded_edge_indices:
                    edge = bundle.edges[edge_index]
                    left = by_id[edge.source_id]
                    right = by_id[edge.target_id]
                    if left == anchor:
                        neighbors.add(right)
                    if right == anchor:
                        neighbors.add(left)
                candidates = [
                    index
                    for index in range(len(bundle.nodes))
                    if index not in neighbors
                ]
                if not candidates:
                    raise ResearchInvalid("collaboration graph has no non-edge candidates")
                entity_ids = [
                    (scope.anchor_node_id, bundle.nodes[index].id) for index in candidates
                ]
                chunks = []
                for offset in range(0, len(candidates), 8_192):
                    pairs = torch.tensor(
                        [
                            (anchor, index)
                            for index in candidates[offset : offset + 8_192]
                        ]
                    )
                    chunks.append(model.collaboration_head(encoded, pairs))
                logits = torch.cat(chunks, dim=0)
                entity_type = "node-pair"
        calibrator = checkpoint["calibrators"][task_id]
        task_calibration_status = (registry.get("taskCalibrationStatus") or {}).get(
            task_id
        )
        if (
            reference["kind"] == "uploaded-artifact"
            or task_calibration_status != "calibrated"
        ):
            calibrator = {**calibrator, "adequate": False}
        scores, calibrated = self._review_scores(logits, calibrator, task_id)
        ranked = sorted(
            zip(entity_ids, (float(item) for item in scores.tolist()), strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        if request.target_scope.kind == "collaboration-candidates":
            ranked = ranked[: request.target_scope.top_k]
        ranked = ranked[: request.parameters.candidate_limit]
        limitations = {
            CONTENT_POLICY_TASK: "Explicit-language history supports review ranking only; it is not unlawful-content detection.",
            ACCOUNT_RISK_TASK: "Historical ban labels support review priority only; this output must not trigger an automatic ban.",
            SIGNED_RELATION_TASK: "The model estimates support/opposition relation stance; it does not measure toxicity or credibility.",
            COLLABORATION_TASK: "This is static unobserved-relation completion, not a forecast of future collaboration.",
        }[task_id]
        reason_codes = (
            ["WIKI_OPPOSITION_REVIEW_SCORE", "HUMAN_REVIEW_REQUIRED"]
            if task_id == SIGNED_RELATION_TASK
            else ["STRUCTURE_GFM_SCORE", "HUMAN_REVIEW_REQUIRED"]
        )
        findings = [
            {
                "id": f"finding:{canonical_sha256([task_id, *ids])[:24]}",
                "rank": rank,
                "entityType": entity_type,
                "entityIds": list(ids),
                "score": score,
                "scoreKind": "probability" if calibrated else "ranking-score",
                "calibrated": calibrated,
                "reasonCodes": reason_codes,
                "limitations": [limitations],
                "reviewRequired": True,
            }
            for rank, (ids, score) in enumerate(ranked, start=1)
        ]
        result: dict[str, Any] = {
            "schemaVersion": WIRE_SCHEMA,
            "runId": run_id,
            "requestHash": request.request_hash,
            "taskId": task_id,
            "graphVersionId": reference["graphVersionId"],
            "graphVersionHash": reference["graphVersionHash"],
            "modelVersionId": registry["modelVersionId"],
            "modelVersionHash": registry["modelVersionHash"],
            "seed": RESEARCH_SEED,
            "preliminary": True,
            "calibrationStatus": "calibrated" if calibrated else "ranking_only",
            "findings": findings,
            "completedAt": _utc_now(),
        }
        result["resultHash"] = canonical_sha256(result)
        return result

    def create_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            envelope = WireRunEnvelope.model_validate(payload)
        except ValidationError as error:
            raise ResearchInvalid("research run envelope is invalid") from error
        registry, _export = self._published()
        self._verify_expected_model(
            envelope.expected_model.model_dump(mode="json", by_alias=True), registry
        )
        request = envelope.request
        reference = envelope.graph_reference.model_dump(mode="json", by_alias=True)
        if (
            request.graph_version_id != reference["graphVersionId"]
            or request.model_version_id != registry["modelVersionId"]
        ):
            raise ResearchServiceError("research run request binding mismatch")
        if reference["kind"] == "registered-scenario":
            scenario = next(
                (
                    item
                    for item in registry["scenarios"]
                    if item["scenarioId"] == request.scenario_id
                ),
                None,
            )
            if scenario is None or (
                scenario["graphVersionId"] != reference["graphVersionId"]
                or scenario["graphVersionHash"] != reference["graphVersionHash"]
                or scenario["taskId"] != request.task_id
            ):
                raise ResearchServiceError("registered scenario run binding mismatch")
        else:
            if request.scenario_id is not None or request.task_id != COLLABORATION_TASK:
                raise ResearchServiceError("uploaded run task binding mismatch")
            manifest = self._uploaded_manifest(reference["graphVersionId"])
            self._verify_uploaded_model(manifest, registry)
            if manifest["graphReference"] != reference:
                raise ResearchServiceError("uploaded graph differs from registration")
        request_hash = envelope.request.request_hash
        run_id = f"research-{canonical_sha256({'requestHash': request_hash, 'graph': envelope.graph_reference.graph_version_hash, 'model': envelope.expected_model.model_version_hash})[:32]}"
        run_root = self.root / "serving/runs" / run_id
        status_path = run_root / "status.json"
        envelope_path = run_root / "request.json"
        with self._state_lock:
            if status_path.is_file():
                try:
                    persisted = WireRunEnvelope.model_validate_json(envelope_path.read_bytes())
                except (OSError, ValidationError) as error:
                    raise ResearchServiceError("persisted run request is invalid") from error
                if persisted != envelope:
                    raise ResearchServiceError("deterministic run id binding conflict")
                return self.get_run(run_id)
            run_root.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                envelope_path,
                envelope.model_dump(mode="json", by_alias=True),
            )
            created_at = _utc_now()
            queued = self._status_payload(
                run_id=run_id,
                request_hash=request_hash,
                status="queued",
                created_at=created_at,
            )
            _atomic_json(status_path, queued)
            self._executor.submit(self._execute_run, run_id, envelope, created_at)
            return queued

    def _execute_run(
        self, run_id: str, envelope: WireRunEnvelope, created_at: str
    ) -> None:
        run_root = self.root / "serving/runs" / run_id
        status_path = run_root / "status.json"
        request_hash = envelope.request.request_hash
        try:
            _atomic_json(
                status_path,
                self._status_payload(
                    run_id=run_id,
                    request_hash=request_hash,
                    status="running",
                    created_at=created_at,
                ),
            )
            result = self._result(run_id, envelope)
            _atomic_json(run_root / "result.json", result)
            succeeded = self._status_payload(
                run_id=run_id,
                request_hash=request_hash,
                status="succeeded",
                created_at=created_at,
            )
            succeeded["updatedAt"] = result["completedAt"]
            succeeded["stateHash"] = canonical_sha256(
                {key: value for key, value in succeeded.items() if key != "stateHash"}
            )
            _atomic_json(status_path, succeeded)
        except Exception as error:  # noqa: BLE001 - worker boundary is fail-closed
            error_code = (
                error.code
                if isinstance(error, ResearchServiceError)
                else "GFM_RESEARCH_EXECUTION_FAILED"
            )
            try:
                _atomic_json(
                    status_path,
                    self._status_payload(
                        run_id=run_id,
                        request_hash=request_hash,
                        status="failed",
                        created_at=created_at,
                        error_code=error_code,
                    ),
                )
            except (OSError, ValueError):
                return

    def _run_path(self, run_id: str, name: str) -> Path:
        if _RUN_ID.fullmatch(run_id) is None:
            raise ResearchNotFound("research run id is invalid")
        return self.root / "serving/runs" / run_id / name

    def get_run(self, run_id: str) -> dict[str, Any]:
        path = self._run_path(run_id, "status.json")
        if not path.is_file():
            raise ResearchNotFound("research run does not exist")
        return _read_runtime_hashed_document(
            path, schema=WIRE_SCHEMA, hash_field="stateHash"
        )

    def get_result(self, run_id: str) -> dict[str, Any]:
        path = self._run_path(run_id, "result.json")
        if not path.is_file():
            status_path = self._run_path(run_id, "status.json")
            if status_path.is_file():
                raise ResearchResultNotReady("research result is not ready")
            raise ResearchNotFound("research result does not exist")
        return _read_runtime_hashed_document(
            path, schema=WIRE_SCHEMA, hash_field="resultHash"
        )

    def _embedding_sources(self):
        import numpy as np

        registry, _export, _checkpoint, _corpus, documents, _model, _adapters = self._model_runtime()
        export_root = self.root / "exports/research"
        for entry in registry["embeddings"]:
            if entry.get("route") != SHARED_NULL_ROUTE:
                raise ResearchServiceError("published similarity route mismatch")
            path = (export_root / entry["path"]).resolve()
            if not path.is_relative_to(export_root.resolve()) or file_sha256(path) != entry["sha256"]:
                raise ResearchServiceError("published embedding identity mismatch")
            with np.load(path, allow_pickle=False) as archive:
                values = np.asarray(archive["embeddings"], dtype="<f4")
                node_ids = tuple(str(item) for item in archive["node_ids"].tolist())
            bundle, _labels, graph_entry = documents[entry["domain"]]
            if values.shape != (len(bundle.nodes), 128) or node_ids != tuple(
                node.id for node in bundle.nodes
            ):
                raise ResearchServiceError("published embedding shape or node order mismatch")
            if entry["domain"].startswith("twitch-"):
                language = entry["domain"].removeprefix("twitch-")
                graph_id = f"research:twitch-language:{language}"
                dataset_id = f"twitch-language:{language}"
            else:
                graph_id = f"research:{graph_entry['datasetFamily']}"
                dataset_id = graph_entry["datasetFamily"]
            yield graph_id, dataset_id, entry["domain"], bundle, values

    def similar_nodes(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        try:
            envelope = WireSimilarEnvelope.model_validate(payload)
        except ValidationError as error:
            raise ResearchInvalid("similar-nodes envelope is invalid") from error
        registry, _export = self._published()
        self._verify_expected_model(
            envelope.expected_model.model_dump(mode="json", by_alias=True), registry
        )
        request = envelope.request
        reference = envelope.graph_reference.model_dump(mode="json", by_alias=True)
        if (
            request.graph_version_id != reference["graphVersionId"]
            or request.model_version_id != registry["modelVersionId"]
        ):
            raise ResearchServiceError("similar-nodes request binding mismatch")
        source_bundle: CoreGraphBundle | None = None
        source_values = None
        source_domain = None
        sources = list(self._embedding_sources())
        if reference["kind"] == "registered-scenario":
            scenario = next(
                (
                    item
                    for item in registry["scenarios"]
                    if item["graphVersionId"] == reference["graphVersionId"]
                    and item["graphVersionHash"] == reference["graphVersionHash"]
                ),
                None,
            )
            if scenario is None:
                raise ResearchServiceError("similarity scenario graph binding mismatch")
            source_domain = scenario["domain"]
        for graph_id, _dataset, domain, bundle, values in sources:
            if (
                (
                    domain == source_domain
                    if source_domain is not None
                    else graph_id == request.graph_version_id
                )
                and bundle.graph_version_hash == reference["graphVersionHash"]
            ):
                source_bundle, source_values = bundle, values
                break
        if source_bundle is None and reference["kind"] == "uploaded-artifact":
            manifest = self._uploaded_manifest(request.graph_version_id)
            self._verify_uploaded_model(manifest, registry)
            if manifest["graphReference"] != reference:
                raise ResearchServiceError("uploaded similarity graph binding mismatch")
            cache_root = self.root / "serving/uploaded" / manifest["cacheKey"]
            source_bundle = CoreGraphBundle.model_validate_json(
                (cache_root / "bundle.json").read_bytes()
            )
            with np.load(cache_root / "embeddings.npz", allow_pickle=False) as archive:
                source_values = np.asarray(archive["embeddings"], dtype="<f4")
        if source_bundle is None or source_values is None:
            raise ResearchNotFound("similarity source graph is unavailable")
        source_by_id = {node.id: node.index for node in source_bundle.nodes}
        if request.node_id not in source_by_id:
            raise ResearchNotFound("similarity source node is unavailable")
        query = source_values[source_by_id[request.node_id]]
        candidates: list[
            tuple[float, str, str, str | None, CoreGraphBundle, int]
        ] = []
        for graph_id, dataset_id, domain, bundle, values in sources:
            if source_domain is not None and domain == source_domain:
                continue
            similarities = values @ query
            for index, similarity in enumerate(similarities.tolist()):
                node_id = bundle.nodes[index].id
                candidates.append(
                    (float(similarity), graph_id, node_id, dataset_id, bundle, index)
                )
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        matches = [
            {
                "graphVersionId": graph_id,
                "nodeId": node_id,
                "datasetId": dataset_id,
                "similarity": min(1.0, max(-1.0, similarity)),
                "structuralFacts": _facts(bundle, index),
            }
            for similarity, graph_id, node_id, dataset_id, bundle, index in candidates[: request.top_k]
        ]
        response: dict[str, Any] = {
            "schemaVersion": WIRE_SCHEMA,
            "graphVersionId": request.graph_version_id,
            "nodeId": request.node_id,
            "modelVersionId": registry["modelVersionId"],
            "modelVersionHash": registry["modelVersionHash"],
            "matches": matches,
        }
        response["resultHash"] = canonical_sha256(response)
        return response

    def dispatch_get(self, path: str) -> dict[str, Any]:
        if path == "/internal/research/capabilities":
            return self.capabilities()
        if path == "/internal/research/scenarios":
            return self.scenarios()
        preview_prefix = "/internal/research/scenarios/"
        if path.startswith(preview_prefix) and path.endswith("/graph-preview"):
            scenario_id = path[len(preview_prefix) : -len("/graph-preview")]
            if not scenario_id or "/" in scenario_id:
                raise ResearchNotFound("research scenario path is invalid")
            return self.graph_preview(scenario_id)
        run_prefix = "/internal/research/runs/"
        if path.startswith(run_prefix):
            suffix = path[len(run_prefix) :]
            if suffix.endswith("/result"):
                return self.get_result(suffix[: -len("/result")])
            if "/" not in suffix:
                return self.get_run(suffix)
        raise ResearchNotFound("SocialGraph-FM Research route does not exist")

    def dispatch_post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if path == "/internal/research/runs":
            return self.create_run(payload)
        if path == "/internal/research/similar-nodes":
            return self.similar_nodes(payload)
        if path == "/internal/research/graphs/register":
            return self.register_graph(payload)
        raise ResearchNotFound("SocialGraph-FM Research route does not exist")


__all__ = [
    "ResearchInvalid",
    "ResearchNotFound",
    "ResearchServiceError",
    "ResearchServingRuntime",
    "ResearchUnavailable",
]
