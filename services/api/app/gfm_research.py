"""Fail-closed API gateway for the separately labelled SocialGraph-FM Research runtime."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
from pydantic import Field, model_validator

from .dataset_schemas import DatasetArtifact, GraphVersionTargetDomainEnvelope
from .dataset_storage import DatasetArtifactStore
from .gfm_client import GfmProxyError, _reject_link_components
from .gfm_hashing import canonical_sha256
from .gfm_research_schemas import (
    RESEARCH_RELEASE_LABEL,
    RESEARCH_SCHEMA_VERSION,
    RESEARCH_SEED,
    ResearchCapabilities,
    ResearchGraphCompatibility,
    ResearchModel,
    ResearchRunRequest,
    ResearchRunResult,
    ResearchRunStatus,
    ResearchScenarioGraphPreview,
    ResearchScenariosResponse,
    SimilarNodesRequest,
    SimilarNodesResponse,
    build_unavailable_capabilities,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_BINDING_LOCK = threading.RLock()
_MAX_BINDING_BYTES = 512 * 1024


class ResearchClientProtocol(Protocol):
    async def research_capabilities(self) -> dict[str, Any]: ...

    async def research_scenarios(self) -> dict[str, Any]: ...
    async def research_scenario_preview(self, scenario_id: str) -> dict[str, Any]: ...

    async def create_research_run(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def get_research_run(self, run_id: str) -> dict[str, Any]: ...

    async def get_research_result(self, run_id: str) -> dict[str, Any]: ...

    async def research_similar_nodes(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def register_research_graph(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class ResearchGraphReference(ResearchModel):
    kind: str
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(
        alias="graphVersionHash", pattern=r"^[0-9a-f]{64}$"
    )
    graph_fact_hash: str | None = Field(
        alias="graphFactHash", default=None, pattern=r"^[0-9a-f]{64}$"
    )
    artifact_id: str | None = Field(alias="artifactId", default=None, max_length=300)
    artifact_hash: str | None = Field(
        alias="artifactHash", default=None, pattern=r"^[0-9a-f]{64}$"
    )
    node_count: int = Field(alias="nodeCount", ge=0, le=50_000)
    edge_count: int = Field(alias="edgeCount", ge=0, le=1_500_000)

    @model_validator(mode="after")
    def validate_kind(self) -> ResearchGraphReference:
        if self.kind not in {"registered-scenario", "uploaded-artifact"}:
            raise ValueError("unsupported SocialGraph-FM Research graph reference kind")
        uploaded = self.kind == "uploaded-artifact"
        if uploaded != all(
            item is not None
            for item in (self.graph_fact_hash, self.artifact_id, self.artifact_hash)
        ):
            raise ValueError("uploaded graph references require an exact artifact identity")
        return self


class ResearchRunBinding(ResearchModel):
    schema_version: str = Field(alias="schemaVersion")
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    request_hash: str = Field(alias="requestHash", pattern=r"^[0-9a-f]{64}$")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(
        alias="graphVersionHash", pattern=r"^[0-9a-f]{64}$"
    )
    task_id: str = Field(alias="taskId", min_length=1, max_length=100)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(
        alias="modelVersionHash", pattern=r"^[0-9a-f]{64}$"
    )
    created_at: datetime = Field(alias="createdAt")
    binding_hash: str = Field(alias="bindingHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_binding(self) -> ResearchRunBinding:
        if self.schema_version != "socialgraph-fm.research-run-binding/1.0":
            raise ValueError("unsupported SocialGraph-FM Research run binding schema")
        if not _RUN_ID.fullmatch(self.run_id):
            raise ValueError("unsafe SocialGraph-FM Research run id")
        if self.binding_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"binding_hash"})
        ):
            raise ValueError("bindingHash mismatch")
        return self


class ResearchGraphRegistration(ResearchModel):
    schema_version: str = Field(alias="schemaVersion")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(
        alias="graphVersionHash", pattern=r"^[0-9a-f]{64}$"
    )
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(
        alias="modelVersionHash", pattern=r"^[0-9a-f]{64}$"
    )
    adapter_status: Literal["pending_registration", "ready"] = Field(alias="adapterStatus")
    compatible_task_ids: tuple[str, ...] = Field(
        alias="compatibleTaskIds", strict=False
    )
    auxiliary_capabilities: tuple[str, ...] = Field(
        alias="auxiliaryCapabilities", strict=False
    )
    registration_hash: str = Field(
        alias="registrationHash", pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def validate_registration(self) -> ResearchGraphRegistration:
        if self.schema_version != RESEARCH_SCHEMA_VERSION:
            raise ValueError("invalid SocialGraph-FM Research graph registration state")
        if self.registration_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"registration_hash"})
        ):
            raise ValueError("registrationHash mismatch")
        return self


class ResearchRunBindingStore:
    def __init__(self, root: str | Path) -> None:
        self.root = _reject_link_components(root)

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not _RUN_ID.fullmatch(run_id):
            raise GfmProxyError(404, "GFM_RESEARCH_RUN_NOT_FOUND")

    def _path(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self.root / f"{run_id}.json"

    def put(self, binding: ResearchRunBinding) -> None:
        payload = binding.model_dump_json(by_alias=True).encode("utf-8")
        if len(payload) > _MAX_BINDING_BYTES:
            raise GfmProxyError(502, "GFM_RESEARCH_RUN_BINDING_INVALID")
        destination = self._path(binding.run_id)
        temporary = self.root / f".{binding.run_id}.{uuid.uuid4().hex}.tmp"
        with _BINDING_LOCK:
            self.root.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                existing = self.get(binding.run_id)
                if existing != binding:
                    raise GfmProxyError(502, "GFM_RESEARCH_RUN_BINDING_CONFLICT")
                return
            try:
                with temporary.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            except OSError as error:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                raise GfmProxyError(
                    502, "GFM_RESEARCH_RUN_BINDING_PERSIST_FAILED"
                ) from error

    def get(self, run_id: str) -> ResearchRunBinding:
        path = self._path(run_id)
        try:
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError
            payload = path.read_bytes()
            if not payload or len(payload) > _MAX_BINDING_BYTES:
                raise ValueError("invalid binding size")
            binding = ResearchRunBinding.model_validate_json(payload)
        except FileNotFoundError as error:
            raise GfmProxyError(404, "GFM_RESEARCH_RUN_NOT_FOUND") from error
        except (OSError, ValueError) as error:
            raise GfmProxyError(502, "GFM_RESEARCH_RUN_BINDING_INVALID") from error
        if binding.run_id != run_id:
            raise GfmProxyError(502, "GFM_RESEARCH_RUN_BINDING_INVALID")
        return binding


def _compatibility(
    *,
    node_count: int,
    edge_count: int,
    directedness: str,
    simple_undirected_edge_count: int | None,
) -> ResearchGraphCompatibility:
    blockers: list[dict[str, str]] = []
    tasks: list[str] = []
    auxiliary: list[str] = []

    if 5 <= node_count <= 50_000 and 4 <= edge_count <= 1_500_000:
        auxiliary.append("similar-nodes")
    else:
        structural_code = (
            "RESEARCH_STRUCTURAL_RETRIEVAL_TOO_SMALL"
            if node_count < 5 or edge_count < 4
            else "RESEARCH_STRUCTURAL_RETRIEVAL_SIZE_UNSUPPORTED"
        )
        blockers.append(
            {
                "code": structural_code,
                "message": (
                    "Structural retrieval requires 5 to 50,000 nodes and "
                    "4 to 1,500,000 valid edges."
                ),
            }
        )

    collaboration_reasons: list[dict[str, str]] = []
    if not 20 <= node_count <= 50_000:
        collaboration_reasons.append(
            {
                "code": "RESEARCH_COLLABORATION_NODE_COUNT_UNSUPPORTED",
                "message": "Collaboration completion requires between 20 and 50,000 nodes.",
            }
        )
    if edge_count > 1_500_000:
        collaboration_reasons.append(
            {
                "code": "RESEARCH_COLLABORATION_EDGE_COUNT_UNSUPPORTED",
                "message": "Collaboration completion supports at most 1,500,000 edges.",
            }
        )
    if directedness != "undirected":
        collaboration_reasons.append(
            {
                "code": "RESEARCH_COLLABORATION_UNDIRECTED_REQUIRED",
                "message": "Uploaded collaboration completion requires a simple undirected graph.",
            }
        )
    if simple_undirected_edge_count is None:
        collaboration_reasons.append(
            {
                "code": "RESEARCH_COLLABORATION_SIMPLE_GRAPH_REQUIRED",
                "message": "Self-loops or duplicate undirected edges are not supported.",
            }
        )
    elif node_count * (node_count - 1) // 2 - simple_undirected_edge_count < 10:
        collaboration_reasons.append(
            {
                "code": "RESEARCH_COLLABORATION_CANDIDATES_INSUFFICIENT",
                "message": "At least 10 absent node pairs are required for candidate ranking.",
            }
        )
    if not collaboration_reasons:
        tasks.append("core.collaboration_completion")
    blockers.extend(collaboration_reasons)
    return ResearchGraphCompatibility.model_validate(
        {
            "intendedUse": "gfm_research",
            "status": "compatible" if tasks or auxiliary else "blocked",
            "compatibleTaskIds": tasks,
            "auxiliaryCapabilities": auxiliary,
            "blockers": blockers,
            "adapterStatus": "pending_registration",
        }
    )


def graph_envelope_research_compatibility(
    envelope: GraphVersionTargetDomainEnvelope,
) -> ResearchGraphCompatibility:
    if len(envelope.edges) > 1_500_000 or len(envelope.nodes) > 50_000:
        return _compatibility(
            node_count=len(envelope.nodes),
            edge_count=len(envelope.edges),
            directedness=envelope.directedness,
            simple_undirected_edge_count=None,
        )
    node_ids = {node.id for node in envelope.nodes}
    pairs: set[tuple[str, str]] = set()
    simple = envelope.directedness == "undirected"
    for edge in envelope.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            simple = False
            continue
        if edge.source == edge.target or edge.directed is True:
            simple = False
            continue
        pair = cast(tuple[str, str], tuple(sorted((edge.source, edge.target))))
        if pair in pairs:
            simple = False
        pairs.add(pair)
    return _compatibility(
        node_count=len(envelope.nodes),
        edge_count=len(envelope.edges),
        directedness=envelope.directedness,
        simple_undirected_edge_count=len(pairs) if simple else None,
    )


def _artifact_edge_index(
    store: DatasetArtifactStore, artifact: DatasetArtifact
) -> np.ndarray | None:
    try:
        arrays = store.load_arrays(artifact.id)
    except (FileNotFoundError, OSError, ValueError):
        return None
    names = [item.edge_index_array for item in artifact.graph_variants]
    names.extend(["edge_index", "source_edge_index"])
    for name in names:
        value = arrays.get(name)
        if value is None:
            continue
        candidate = np.asarray(value, dtype=np.int64)
        if candidate.ndim == 2 and candidate.shape[0] == 2:
            return candidate
        if candidate.ndim == 2 and candidate.shape[1] == 2:
            return candidate.T
    return None


def _simple_undirected_edge_count(edge_index: np.ndarray) -> int | None:
    source = edge_index[0]
    target = edge_index[1]
    if source.size != target.size or np.any(source == target):
        return None
    canonical = np.column_stack((np.minimum(source, target), np.maximum(source, target)))
    unique = np.unique(canonical, axis=0)
    if unique.shape[0] == canonical.shape[0]:
        return int(unique.shape[0])
    if canonical.shape[0] != 2 * unique.shape[0]:
        return None

    # A paired COO representation must contain exactly one edge in each direction.
    direction = (source < target).astype(np.int8, copy=False)
    oriented = np.column_stack((canonical, direction))
    unique_oriented, counts = np.unique(oriented, axis=0, return_counts=True)
    if unique_oriented.shape[0] != canonical.shape[0] or np.any(counts != 1):
        return None
    return int(unique.shape[0])


def artifact_research_compatibility(
    store: DatasetArtifactStore, artifact: DatasetArtifact
) -> ResearchGraphCompatibility:
    node_count = artifact.profile.node_count or 0
    edge_count = artifact.profile.edge_count or 0
    semantics = artifact.graph_semantics
    directedness = (
        semantics.directedness
        if semantics is not None and semantics.directedness is not None
        else "directed"
        if semantics is not None and semantics.directed
        else "undirected"
    )
    edge_index = (
        _artifact_edge_index(store, artifact)
        if node_count <= 50_000 and edge_count <= 1_500_000
        else None
    )
    simple_count: int | None = None
    if directedness == "undirected" and edge_index is not None:
        simple_count = _simple_undirected_edge_count(edge_index)
    return _compatibility(
        node_count=node_count,
        edge_count=edge_count,
        directedness=directedness,
        simple_undirected_edge_count=simple_count,
    )


def _default_scenarios() -> ResearchScenariosResponse:
    rows = [
        {
            "scenarioId": "twitch-content-policy",
            "datasetId": "twitch-language",
            "title": "Content policy review",
            "taskId": "research.content_policy_review",
            "graphVersionId": "research:twitch-language",
            "graphVersionHash": None,
            "modelVersionId": None,
            "enabled": False,
            "unavailableReason": "RESEARCH_MODEL_NOT_INSTALLED",
            "defaultTargetScope": {"kind": "nodes", "nodeIds": ["0"]},
            "primaryMetric": None,
            "scratchDelta": None,
        },
        {
            "scenarioId": "tolokers-account-risk",
            "datasetId": "tolokers",
            "title": "Historical account status review",
            "taskId": "research.account_risk_review",
            "graphVersionId": "research:tolokers",
            "graphVersionHash": None,
            "modelVersionId": None,
            "enabled": False,
            "unavailableReason": "RESEARCH_MODEL_NOT_INSTALLED",
            "defaultTargetScope": {"kind": "nodes", "nodeIds": ["0"]},
            "primaryMetric": None,
            "scratchDelta": None,
        },
        {
            "scenarioId": "wiki-rfa-signed-relation",
            "datasetId": "wiki-rfa",
            "title": "Governance relation stance review",
            "taskId": "research.signed_relation_review",
            "graphVersionId": "research:wiki-rfa",
            "graphVersionHash": None,
            "modelVersionId": None,
            "enabled": False,
            "unavailableReason": "RESEARCH_MODEL_NOT_INSTALLED",
            "defaultTargetScope": {
                "kind": "directed-node-pairs",
                "pairs": [["0", "1"]],
            },
            "primaryMetric": None,
            "scratchDelta": None,
        },
        {
            "scenarioId": "email-eu-collaboration",
            "datasetId": "email-eu-core",
            "title": "Collaboration relation candidates",
            "taskId": "core.collaboration_completion",
            "graphVersionId": "research:email-eu-core",
            "graphVersionHash": None,
            "modelVersionId": None,
            "enabled": False,
            "unavailableReason": "RESEARCH_MODEL_NOT_INSTALLED",
            "defaultTargetScope": {
                "kind": "collaboration-candidates",
                "anchorNodeId": "0",
                "topK": 10,
            },
            "primaryMetric": None,
            "scratchDelta": None,
        },
    ]
    payload: dict[str, Any] = {
        "schemaVersion": RESEARCH_SCHEMA_VERSION,
        "releaseLabel": RESEARCH_RELEASE_LABEL,
        "seed": RESEARCH_SEED,
        "preliminary": True,
        "scenarios": rows,
    }
    payload["scenariosHash"] = canonical_sha256(payload)
    return ResearchScenariosResponse.model_validate(payload)


class ResearchGateway:
    def __init__(
        self,
        client: ResearchClientProtocol | None,
        *,
        dataset_store: DatasetArtifactStore,
        binding_store: ResearchRunBindingStore,
    ) -> None:
        self.client = client
        self.dataset_store = dataset_store
        self.binding_store = binding_store

    async def capabilities(self) -> ResearchCapabilities:
        if self.client is None:
            return build_unavailable_capabilities()
        try:
            payload = await self.client.research_capabilities()
            return ResearchCapabilities.model_validate_json(
                json.dumps(payload, ensure_ascii=False)
            )
        except GfmProxyError:
            raise
        except (TypeError, ValueError) as error:
            raise GfmProxyError(502, "GFM_RESEARCH_CAPABILITIES_INVALID") from error

    async def scenarios(self) -> ResearchScenariosResponse:
        if self.client is None:
            return _default_scenarios()
        try:
            payload = await self.client.research_scenarios()
            scenarios = ResearchScenariosResponse.model_validate_json(
                json.dumps(payload, ensure_ascii=False)
            )
            capabilities = await self.capabilities()
        except GfmProxyError:
            raise
        except (TypeError, ValueError) as error:
            raise GfmProxyError(502, "GFM_RESEARCH_SCENARIOS_INVALID") from error
        expected_model = capabilities.model.model_version_id if capabilities.model else None
        if any(
            item.enabled != capabilities.research_serving_ready
            or item.model_version_id != expected_model
            for item in scenarios.scenarios
        ):
            raise GfmProxyError(503, "GFM_RESEARCH_SCENARIOS_STALE")
        return scenarios

    async def scenario_preview(self, scenario_id: str) -> ResearchScenarioGraphPreview:
        scenarios = await self.scenarios()
        scenario = next(
            (item for item in scenarios.scenarios if item.scenario_id == scenario_id),
            None,
        )
        if scenario is None:
            raise GfmProxyError(404, "GFM_RESEARCH_SCENARIO_NOT_FOUND")
        if not scenario.enabled or scenario.graph_version_hash is None:
            raise GfmProxyError(503, "GFM_RESEARCH_SCENARIO_UNAVAILABLE")
        capabilities = await self.capabilities()
        if capabilities.model is None or self.client is None:
            raise GfmProxyError(503, "GFM_RESEARCH_MODEL_NOT_INSTALLED")
        raw = await self.client.research_scenario_preview(scenario_id)
        try:
            preview = ResearchScenarioGraphPreview.model_validate_json(
                json.dumps(raw, ensure_ascii=False)
            )
        except (TypeError, ValueError) as error:
            raise GfmProxyError(502, "GFM_RESEARCH_PREVIEW_INVALID") from error
        if (
            preview.scenario_id != scenario.scenario_id
            or preview.graph_version_id != scenario.graph_version_id
            or preview.graph_version_hash != scenario.graph_version_hash
            or preview.model_version_id != capabilities.model.model_version_id
            or preview.model_version_hash != capabilities.model.model_version_hash
        ):
            raise GfmProxyError(502, "GFM_RESEARCH_PREVIEW_BINDING_MISMATCH")
        return preview

    def _uploaded_graph(self, graph_version_id: str) -> tuple[ResearchGraphReference, ResearchGraphCompatibility]:
        try:
            binding = self.dataset_store.resolve_graph_version_binding(graph_version_id)
        except ValueError as error:
            raise GfmProxyError(409, "GFM_RESEARCH_GRAPH_IDENTITY_CONFLICT") from error
        if binding is None:
            raise LookupError(graph_version_id)
        artifact = self.dataset_store.get_artifact(binding.artifact_id)
        if artifact is None:
            raise GfmProxyError(409, "GFM_RESEARCH_GRAPH_ARTIFACT_MISSING")
        compatibility = artifact_research_compatibility(self.dataset_store, artifact)
        try:
            reference = ResearchGraphReference(
                kind="uploaded-artifact",
                graphVersionId=binding.graph_version_id,
                graphVersionHash=artifact.canonical_graph_hash,
                graphFactHash=binding.graph_fact_hash,
                artifactId=artifact.id,
                artifactHash=artifact.content_hash,
                nodeCount=artifact.profile.node_count or 0,
                edgeCount=artifact.profile.edge_count or 0,
            )
        except ValueError as error:
            raise GfmProxyError(
                409, "GFM_RESEARCH_GRAPH_ARTIFACT_IDENTITY_INVALID"
            ) from error
        return reference, compatibility

    async def _resolve_graph(
        self, request: ResearchRunRequest
    ) -> ResearchGraphReference:
        scenarios = await self.scenarios()
        if request.scenario_id is not None:
            scenario = next(
                (
                    item
                    for item in scenarios.scenarios
                    if item.scenario_id == request.scenario_id
                ),
                None,
            )
            if scenario is None:
                raise GfmProxyError(404, "GFM_RESEARCH_SCENARIO_NOT_FOUND")
            if not scenario.enabled:
                raise GfmProxyError(503, "GFM_RESEARCH_SCENARIO_UNAVAILABLE")
            if (
                scenario.graph_version_id != request.graph_version_id
                or scenario.task_id != request.task_id
                or scenario.model_version_id != request.model_version_id
                or scenario.graph_version_hash is None
            ):
                raise GfmProxyError(409, "GFM_RESEARCH_SCENARIO_MISMATCH")
            return ResearchGraphReference(
                kind="registered-scenario",
                graphVersionId=scenario.graph_version_id,
                graphVersionHash=scenario.graph_version_hash,
                nodeCount=0,
                edgeCount=0,
            )
        if request.task_id != "core.collaboration_completion":
            raise GfmProxyError(409, "GFM_RESEARCH_REGISTERED_SCENARIO_REQUIRED")
        try:
            reference, compatibility = self._uploaded_graph(request.graph_version_id)
        except LookupError as error:
            raise GfmProxyError(404, "GFM_RESEARCH_GRAPH_VERSION_NOT_FOUND") from error
        if request.task_id not in compatibility.compatible_task_ids:
            raise GfmProxyError(409, "GFM_RESEARCH_GRAPH_INCOMPATIBLE")
        registered = await self.register_uploaded_graph(request.graph_version_id)
        if registered.adapter_status != "ready":
            raise GfmProxyError(503, "GFM_RESEARCH_GRAPH_REGISTRATION_PENDING")
        return reference

    @staticmethod
    def _binding(
        status: ResearchRunStatus,
        request: ResearchRunRequest,
        graph: ResearchGraphReference,
        capabilities: ResearchCapabilities,
    ) -> ResearchRunBinding:
        assert capabilities.model is not None
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.research-run-binding/1.0",
            "runId": status.run_id,
            "requestHash": request.request_hash,
            "graphVersionId": graph.graph_version_id,
            "graphVersionHash": graph.graph_version_hash,
            "taskId": request.task_id,
            "modelVersionId": capabilities.model.model_version_id,
            "modelVersionHash": capabilities.model.model_version_hash,
            "createdAt": status.created_at,
        }
        payload["bindingHash"] = canonical_sha256(payload)
        return ResearchRunBinding.model_validate(payload)

    async def create_run(self, request: ResearchRunRequest) -> ResearchRunStatus:
        capabilities = await self.capabilities()
        if not capabilities.research_serving_ready or capabilities.model is None:
            raise GfmProxyError(503, "GFM_RESEARCH_MODEL_NOT_INSTALLED")
        if request.model_version_id != capabilities.model.model_version_id:
            raise GfmProxyError(409, "GFM_RESEARCH_MODEL_MISMATCH")
        graph = await self._resolve_graph(request)
        payload = {
            "schemaVersion": RESEARCH_SCHEMA_VERSION,
            "request": request.model_dump(mode="json", by_alias=True),
            "graphReference": graph.model_dump(mode="json", by_alias=True),
            "expectedModel": capabilities.model.model_dump(mode="json", by_alias=True),
        }
        assert self.client is not None
        raw = await self.client.create_research_run(payload)
        try:
            status = ResearchRunStatus.model_validate_json(
                json.dumps(raw, ensure_ascii=False)
            )
        except (TypeError, ValueError) as error:
            raise GfmProxyError(502, "GFM_RESEARCH_RESPONSE_INVALID") from error
        if status.request_hash != request.request_hash:
            raise GfmProxyError(502, "GFM_RESEARCH_REQUEST_HASH_MISMATCH")
        self.binding_store.put(self._binding(status, request, graph, capabilities))
        return status

    async def get_run(self, run_id: str) -> ResearchRunStatus:
        binding = self.binding_store.get(run_id)
        if self.client is None:
            raise GfmProxyError(503, "GFM_RESEARCH_SERVICE_UNAVAILABLE")
        raw = await self.client.get_research_run(run_id)
        try:
            status = ResearchRunStatus.model_validate_json(
                json.dumps(raw, ensure_ascii=False)
            )
        except (TypeError, ValueError) as error:
            raise GfmProxyError(502, "GFM_RESEARCH_RESPONSE_INVALID") from error
        if status.run_id != binding.run_id or status.request_hash != binding.request_hash:
            raise GfmProxyError(502, "GFM_RESEARCH_RUN_BINDING_MISMATCH")
        return status

    async def get_result(self, run_id: str) -> ResearchRunResult:
        binding = self.binding_store.get(run_id)
        if self.client is None:
            raise GfmProxyError(503, "GFM_RESEARCH_SERVICE_UNAVAILABLE")
        raw = await self.client.get_research_result(run_id)
        try:
            result = ResearchRunResult.model_validate_json(
                json.dumps(raw, ensure_ascii=False)
            )
        except (TypeError, ValueError) as error:
            raise GfmProxyError(502, "GFM_RESEARCH_RESPONSE_INVALID") from error
        identity = (
            result.run_id,
            result.request_hash,
            result.task_id,
            result.graph_version_id,
            result.graph_version_hash,
            result.model_version_id,
            result.model_version_hash,
        )
        expected = (
            binding.run_id,
            binding.request_hash,
            binding.task_id,
            binding.graph_version_id,
            binding.graph_version_hash,
            binding.model_version_id,
            binding.model_version_hash,
        )
        if identity != expected:
            raise GfmProxyError(502, "GFM_RESEARCH_RESULT_BINDING_MISMATCH")
        return result

    async def similar_nodes(self, request: SimilarNodesRequest) -> SimilarNodesResponse:
        capabilities = await self.capabilities()
        if not capabilities.research_serving_ready or capabilities.model is None:
            raise GfmProxyError(503, "GFM_RESEARCH_MODEL_NOT_INSTALLED")
        if request.model_version_id != capabilities.model.model_version_id:
            raise GfmProxyError(409, "GFM_RESEARCH_MODEL_MISMATCH")
        scenario = next(
            (
                item
                for item in (await self.scenarios()).scenarios
                if item.graph_version_id == request.graph_version_id and item.enabled
            ),
            None,
        )
        if scenario is not None and scenario.graph_version_hash is not None:
            graph = ResearchGraphReference(
                kind="registered-scenario",
                graphVersionId=scenario.graph_version_id,
                graphVersionHash=scenario.graph_version_hash,
                nodeCount=0,
                edgeCount=0,
            )
        else:
            try:
                graph, compatibility = self._uploaded_graph(request.graph_version_id)
            except LookupError as error:
                raise GfmProxyError(
                    404, "GFM_RESEARCH_GRAPH_VERSION_NOT_FOUND"
                ) from error
            if "similar-nodes" not in compatibility.auxiliary_capabilities:
                raise GfmProxyError(409, "GFM_RESEARCH_GRAPH_INCOMPATIBLE")
            registered = await self.register_uploaded_graph(request.graph_version_id)
            if registered.adapter_status != "ready":
                raise GfmProxyError(503, "GFM_RESEARCH_GRAPH_REGISTRATION_PENDING")
        assert self.client is not None
        raw = await self.client.research_similar_nodes(
            {
                "schemaVersion": RESEARCH_SCHEMA_VERSION,
                "request": request.model_dump(mode="json", by_alias=True),
                "graphReference": graph.model_dump(mode="json", by_alias=True),
                "expectedModel": capabilities.model.model_dump(
                    mode="json", by_alias=True
                ),
            }
        )
        try:
            response = SimilarNodesResponse.model_validate_json(
                json.dumps(raw, ensure_ascii=False)
            )
        except (TypeError, ValueError) as error:
            raise GfmProxyError(502, "GFM_RESEARCH_RESPONSE_INVALID") from error
        if (
            response.graph_version_id != request.graph_version_id
            or response.node_id != request.node_id
            or response.model_version_id != request.model_version_id
            or response.model_version_hash != capabilities.model.model_version_hash
            or len(response.matches) > request.top_k
        ):
            raise GfmProxyError(502, "GFM_RESEARCH_RESULT_BINDING_MISMATCH")
        return response

    async def register_uploaded_graph(
        self, graph_version_id: str
    ) -> ResearchGraphCompatibility:
        capabilities = await self.capabilities()
        if not capabilities.research_serving_ready or capabilities.model is None:
            raise GfmProxyError(503, "GFM_RESEARCH_MODEL_NOT_INSTALLED")
        try:
            graph, compatibility = self._uploaded_graph(graph_version_id)
        except LookupError as error:
            raise GfmProxyError(404, "GFM_RESEARCH_GRAPH_VERSION_NOT_FOUND") from error
        if compatibility.status == "blocked":
            return compatibility
        assert self.client is not None
        raw = await self.client.register_research_graph(
            {
                "schemaVersion": RESEARCH_SCHEMA_VERSION,
                "graphReference": graph.model_dump(mode="json", by_alias=True),
                "compatibleTaskIds": list(compatibility.compatible_task_ids),
                "auxiliaryCapabilities": list(
                    compatibility.auxiliary_capabilities
                ),
                "expectedModel": capabilities.model.model_dump(
                    mode="json", by_alias=True
                ),
            }
        )
        try:
            registration = ResearchGraphRegistration.model_validate_json(
                json.dumps(raw, ensure_ascii=False)
            )
        except (TypeError, ValueError) as error:
            raise GfmProxyError(502, "GFM_RESEARCH_REGISTRATION_INVALID") from error
        if (
            registration.graph_version_id != graph.graph_version_id
            or registration.graph_version_hash != graph.graph_version_hash
            or registration.model_version_id != capabilities.model.model_version_id
            or registration.model_version_hash != capabilities.model.model_version_hash
            or registration.compatible_task_ids != compatibility.compatible_task_ids
            or registration.auxiliary_capabilities
            != compatibility.auxiliary_capabilities
        ):
            raise GfmProxyError(502, "GFM_RESEARCH_REGISTRATION_MISMATCH")
        return compatibility.model_copy(update={"adapter_status": registration.adapter_status})


__all__ = [
    "ResearchClientProtocol",
    "ResearchGateway",
    "ResearchRunBindingStore",
    "artifact_research_compatibility",
    "graph_envelope_research_compatibility",
]
