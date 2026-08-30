"""Fail-closed API gateway for the SocialGraph-FM Global GFM runtime."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .gfm_client import GfmProxyError, _reject_link_components
from .gfm_hashing import canonical_sha256
from .gfm_global_model_schemas import (
    GLOBAL_MODEL_DATASET_VERSION_ID,
    GLOBAL_MODEL_PROTOCOLS,
    GLOBAL_MODEL_SCHEMA_VERSION,
    GlobalModelCapabilities,
    GlobalModelHealth,
    GlobalModelCard,
    GlobalModelNodeEvidence,
    GlobalModelReviewRecord,
    GlobalModelReviewRequest,
    GlobalModelRunRequest,
    GlobalModelRunResult,
    GlobalModelRunStatus,
    GlobalModelScenario,
    GlobalModelScenarioPreview,
    build_unavailable_capabilities,
)

_RUN_ID = re.compile(r"^global-model-[0-9a-f]{32}$")
_LOCK = threading.RLock()
_MAX_RECORD_BYTES = 512 * 1024


class GlobalModelClientProtocol(Protocol):
    async def global_model_health(self) -> dict[str, Any]: ...

    async def global_model_capabilities(self) -> dict[str, Any]: ...

    async def global_model_card(self) -> dict[str, Any]: ...

    async def global_model_scenario(self) -> dict[str, Any]: ...

    async def global_model_scenario_preview(self) -> dict[str, Any]: ...

    async def create_global_model_run(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def get_global_model_run(self, run_id: str) -> dict[str, Any]: ...

    async def get_global_model_result(self, run_id: str) -> dict[str, Any]: ...

    async def get_global_model_evidence(self, run_id: str, node_id: str) -> dict[str, Any]: ...


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        protected_namespaces=("model_dump",),
    )


class GlobalModelRunBinding(_FrozenModel):
    schema_version: str = Field(alias="schemaVersion")
    run_id: str = Field(alias="runId", pattern=r"^global-model-[0-9a-f]{32}$")
    request_hash: str = Field(alias="requestHash", pattern=r"^[0-9a-f]{64}$")
    task_id: str = Field(alias="taskId")
    protocol: str
    dataset_version_id: str = Field(alias="datasetVersionId")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=r"^[0-9a-f]{64}$")
    model_version_id: str = Field(alias="modelVersionId")
    model_version_hash: str = Field(alias="modelVersionHash", pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(alias="createdAt")
    binding_hash: str = Field(alias="bindingHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hash(self) -> GlobalModelRunBinding:
        if self.schema_version != "socialgraph-fm.global-model-run-binding/1.0":
            raise ValueError("unsupported GlobalModel run binding schema")
        if self.binding_hash != canonical_sha256(
            self.model_dump(mode="json", by_alias=True, exclude={"binding_hash"})
        ):
            raise ValueError("bindingHash mismatch")
        return self


def _atomic_record(path: Path, payload: bytes) -> None:
    if len(payload) > _MAX_RECORD_BYTES:
        raise ValueError("GlobalModel record exceeds size limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the atomic sibling name bounded so deeply nested Windows checkouts do not
    # exceed the legacy MAX_PATH limit merely by duplicating the destination name.
    temporary = path.parent / f".tmp-{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class GlobalModelRunBindingStore:
    def __init__(self, root: str | Path) -> None:
        self.root = _reject_link_components(root)

    def _path(self, run_id: str) -> Path:
        if _RUN_ID.fullmatch(run_id) is None:
            raise GfmProxyError(404, "GFM_GLOBAL_MODEL_RUN_NOT_FOUND")
        return self.root / f"{run_id}.json"

    def put(self, binding: GlobalModelRunBinding) -> None:
        path = self._path(binding.run_id)
        payload = binding.model_dump_json(by_alias=True).encode("utf-8")
        with _LOCK:
            if path.exists():
                if self.get(binding.run_id) != binding:
                    raise GfmProxyError(502, "GFM_GLOBAL_MODEL_RUN_BINDING_CONFLICT")
                return
            try:
                _atomic_record(path, payload)
            except OSError as error:
                raise GfmProxyError(502, "GFM_GLOBAL_MODEL_RUN_BINDING_PERSIST_FAILED") from error

    def get(self, run_id: str) -> GlobalModelRunBinding:
        path = self._path(run_id)
        try:
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError
            payload = path.read_bytes()
            if not payload or len(payload) > _MAX_RECORD_BYTES:
                raise ValueError("invalid binding size")
            binding = GlobalModelRunBinding.model_validate_json(payload)
        except FileNotFoundError as error:
            raise GfmProxyError(404, "GFM_GLOBAL_MODEL_RUN_NOT_FOUND") from error
        except (OSError, ValueError) as error:
            raise GfmProxyError(502, "GFM_GLOBAL_MODEL_RUN_BINDING_INVALID") from error
        if binding.run_id != run_id:
            raise GfmProxyError(502, "GFM_GLOBAL_MODEL_RUN_BINDING_INVALID")
        return binding


class GlobalModelReviewStore:
    def __init__(self, root: str | Path) -> None:
        self.root = _reject_link_components(root)

    def put(
        self, run_id: str, request: GlobalModelReviewRequest
    ) -> GlobalModelReviewRecord:
        if _RUN_ID.fullmatch(run_id) is None:
            raise GfmProxyError(404, "GFM_GLOBAL_MODEL_RUN_NOT_FOUND")
        review_id = f"review-{uuid.uuid4().hex}"
        now = datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
        payload: dict[str, Any] = {
            "schemaVersion": GLOBAL_MODEL_SCHEMA_VERSION,
            "reviewId": review_id,
            "runId": run_id,
            "nodeId": request.node_id,
            "decision": request.decision,
            "reason": request.reason,
            "createdAt": now,
        }
        payload["reviewHash"] = canonical_sha256(payload)
        record = GlobalModelReviewRecord.model_validate(payload)
        path = self.root / run_id / f"{review_id}.json"
        try:
            with _LOCK:
                _atomic_record(path, record.model_dump_json(by_alias=True).encode("utf-8"))
        except OSError as error:
            raise GfmProxyError(502, "GFM_GLOBAL_MODEL_REVIEW_PERSIST_FAILED") from error
        return record


def _unavailable_scenario() -> GlobalModelScenario:
    payload: dict[str, Any] = {
        "schemaVersion": GLOBAL_MODEL_SCHEMA_VERSION,
        "scenarioId": "russia-coordination-risk",
        "datasetVersionId": GLOBAL_MODEL_DATASET_VERSION_ID,
        "graphVersionHash": None,
        "modelVersionId": None,
        "enabled": False,
        "unavailableReason": "GFM_GLOBAL_MODEL_NOT_INSTALLED",
        "nodeCount": 716,
        "edgeCount": 0,
        "protocols": list(GLOBAL_MODEL_PROTOCOLS),
        "metrics": {protocol: None for protocol in GLOBAL_MODEL_PROTOCOLS},
        "limitations": [
            "Anonymous research identifiers only; no real-world identity claim.",
            "Predictions are review candidates and never automatic enforcement decisions.",
        ],
    }
    payload["scenarioHash"] = canonical_sha256(payload)
    return GlobalModelScenario.model_validate(payload)


class GlobalModelGateway:
    def __init__(
        self,
        client: GlobalModelClientProtocol | None,
        *,
        binding_store: GlobalModelRunBindingStore,
        review_store: GlobalModelReviewStore,
    ) -> None:
        self.client = client
        self.binding_store = binding_store
        self.review_store = review_store

    @staticmethod
    def _parse(model_type, payload: dict[str, Any], code: str):
        try:
            return model_type.model_validate_json(json.dumps(payload, ensure_ascii=False))
        except (TypeError, ValueError) as error:
            raise GfmProxyError(502, code) from error

    async def capabilities(self) -> GlobalModelCapabilities:
        if self.client is None:
            return build_unavailable_capabilities()
        raw = await self.client.global_model_capabilities()
        return self._parse(
            GlobalModelCapabilities, raw, "GFM_GLOBAL_MODEL_CAPABILITIES_INVALID"
        )

    async def health(self) -> GlobalModelHealth:
        if self.client is None:
            payload: dict[str, Any] = {
                "schemaVersion": "socialgraph-fm.global-model-health/1.0",
                "serviceIdentity": canonical_sha256(
                    {
                        "service": "socialgraph-fm-gfm/global-model",
                        "datasetVersionId": GLOBAL_MODEL_DATASET_VERSION_ID,
                        "modelVersionId": None,
                        "modelVersionHash": None,
                        "corpusHash": None,
                    }
                ),
                "servingReady": False,
                "modelVersionId": None,
                "modelVersionHash": None,
                "corpusHash": None,
                "datasetVersionId": GLOBAL_MODEL_DATASET_VERSION_ID,
            }
            payload["healthHash"] = canonical_sha256(payload)
            return GlobalModelHealth.model_validate(payload)
        health = self._parse(
            GlobalModelHealth,
            await self.client.global_model_health(),
            "GFM_GLOBAL_MODEL_HEALTH_INVALID",
        )
        capabilities = await self.capabilities()
        model = capabilities.model
        expected = (
            capabilities.serving_ready,
            None if model is None else model.model_version_id,
            None if model is None else model.model_version_hash,
            None if model is None else model.corpus_hash,
            capabilities.dataset_version_id,
        )
        observed = (
            health.serving_ready,
            health.model_version_id,
            health.model_version_hash,
            health.corpus_hash,
            health.dataset_version_id,
        )
        if observed != expected:
            raise GfmProxyError(503, "GFM_GLOBAL_MODEL_HEALTH_STALE")
        return health

    async def model_card(self) -> GlobalModelCard:
        if self.client is None:
            raise GfmProxyError(503, "GFM_GLOBAL_MODEL_NOT_INSTALLED")
        card = self._parse(
            GlobalModelCard,
            await self.client.global_model_card(),
            "GFM_GLOBAL_MODEL_CARD_INVALID",
        )
        capabilities = await self.capabilities()
        model = capabilities.model
        if model is None or (
            card.model_version_id,
            card.model_version_hash,
            card.artifact_hash,
        ) != (model.model_version_id, model.model_version_hash, model.artifact_hash):
            raise GfmProxyError(503, "GFM_GLOBAL_MODEL_CARD_STALE")
        return card

    async def scenario(self) -> GlobalModelScenario:
        if self.client is None:
            return _unavailable_scenario()
        raw = await self.client.global_model_scenario()
        scenario = self._parse(GlobalModelScenario, raw, "GFM_GLOBAL_MODEL_SCENARIO_INVALID")
        capabilities = await self.capabilities()
        expected_model = capabilities.model.model_version_id if capabilities.model else None
        if (
            scenario.enabled != capabilities.serving_ready
            or scenario.model_version_id != expected_model
        ):
            raise GfmProxyError(503, "GFM_GLOBAL_MODEL_SCENARIO_STALE")
        return scenario

    async def scenario_preview(self) -> GlobalModelScenarioPreview:
        scenario = await self.scenario()
        if not scenario.enabled or self.client is None:
            raise GfmProxyError(503, "GFM_GLOBAL_MODEL_SCENARIO_UNAVAILABLE")
        raw = await self.client.global_model_scenario_preview()
        preview = self._parse(
            GlobalModelScenarioPreview, raw, "GFM_GLOBAL_MODEL_PREVIEW_INVALID"
        )
        if preview.graph_version_hash != scenario.graph_version_hash:
            raise GfmProxyError(502, "GFM_GLOBAL_MODEL_PREVIEW_BINDING_MISMATCH")
        return preview

    async def create_run(self, request: GlobalModelRunRequest) -> GlobalModelRunStatus:
        capabilities = await self.capabilities()
        scenario = await self.scenario()
        if not capabilities.serving_ready or capabilities.model is None or not scenario.enabled:
            raise GfmProxyError(503, "GFM_GLOBAL_MODEL_NOT_INSTALLED")
        protocol_model = capabilities.model.protocol_models[request.protocol]
        if request.model_version_id != protocol_model.model_version_id:
            raise GfmProxyError(409, "GFM_GLOBAL_MODEL_MISMATCH")
        if self.client is None or scenario.graph_version_hash is None:
            raise GfmProxyError(503, "GFM_GLOBAL_MODEL_SERVICE_UNAVAILABLE")
        envelope = {
            "schemaVersion": GLOBAL_MODEL_SCHEMA_VERSION,
            "request": request.model_dump(mode="json", by_alias=True),
            "expectedModel": capabilities.model.model_dump(mode="json", by_alias=True),
            "datasetBinding": {
                "datasetVersionId": scenario.dataset_version_id,
                "graphVersionHash": scenario.graph_version_hash,
            },
        }
        status = self._parse(
            GlobalModelRunStatus,
            await self.client.create_global_model_run(envelope),
            "GFM_GLOBAL_MODEL_RESPONSE_INVALID",
        )
        if status.request_hash != request.request_hash:
            raise GfmProxyError(502, "GFM_GLOBAL_MODEL_REQUEST_HASH_MISMATCH")
        binding_payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.global-model-run-binding/1.0",
            "runId": status.run_id,
            "requestHash": request.request_hash,
            "taskId": request.task_id,
            "protocol": request.protocol,
            "datasetVersionId": request.dataset_version_id,
            "graphVersionHash": scenario.graph_version_hash,
            "modelVersionId": protocol_model.model_version_id,
            "modelVersionHash": protocol_model.model_version_hash,
            "createdAt": status.model_dump(mode="json", by_alias=True)["createdAt"],
        }
        binding_payload["bindingHash"] = canonical_sha256(binding_payload)
        self.binding_store.put(GlobalModelRunBinding.model_validate(binding_payload))
        return status

    async def get_run(self, run_id: str) -> GlobalModelRunStatus:
        binding = self.binding_store.get(run_id)
        if self.client is None:
            raise GfmProxyError(503, "GFM_GLOBAL_MODEL_SERVICE_UNAVAILABLE")
        status = self._parse(
            GlobalModelRunStatus,
            await self.client.get_global_model_run(run_id),
            "GFM_GLOBAL_MODEL_RESPONSE_INVALID",
        )
        if status.run_id != binding.run_id or status.request_hash != binding.request_hash:
            raise GfmProxyError(502, "GFM_GLOBAL_MODEL_RUN_BINDING_MISMATCH")
        return status

    async def get_result(self, run_id: str) -> GlobalModelRunResult:
        binding = self.binding_store.get(run_id)
        if self.client is None:
            raise GfmProxyError(503, "GFM_GLOBAL_MODEL_SERVICE_UNAVAILABLE")
        result = self._parse(
            GlobalModelRunResult,
            await self.client.get_global_model_result(run_id),
            "GFM_GLOBAL_MODEL_RESPONSE_INVALID",
        )
        actual = (
            result.run_id,
            result.request_hash,
            result.task_id,
            result.protocol,
            result.dataset_version_id,
            result.graph_version_hash,
            result.model_version_id,
            result.model_version_hash,
        )
        expected = (
            binding.run_id,
            binding.request_hash,
            binding.task_id,
            binding.protocol,
            binding.dataset_version_id,
            binding.graph_version_hash,
            binding.model_version_id,
            binding.model_version_hash,
        )
        if actual != expected:
            raise GfmProxyError(502, "GFM_GLOBAL_MODEL_RESULT_BINDING_MISMATCH")
        return result

    async def evidence(self, run_id: str, node_id: str) -> GlobalModelNodeEvidence:
        binding = self.binding_store.get(run_id)
        if not node_id or len(node_id) > 100 or "/" in node_id or "\\" in node_id:
            raise GfmProxyError(404, "GFM_GLOBAL_MODEL_NODE_NOT_FOUND")
        if self.client is None:
            raise GfmProxyError(503, "GFM_GLOBAL_MODEL_SERVICE_UNAVAILABLE")
        result = await self.get_result(run_id)
        evidence = self._parse(
            GlobalModelNodeEvidence,
            await self.client.get_global_model_evidence(run_id, node_id),
            "GFM_GLOBAL_MODEL_EVIDENCE_INVALID",
        )
        acceptable_node_ids = {node_id}
        if node_id.startswith("russia:"):
            acceptable_node_ids.add(node_id.removeprefix("russia:"))
        else:
            acceptable_node_ids.add(f"russia:{node_id}")
        if (
            evidence.run_id != binding.run_id
            or evidence.node.node_id not in acceptable_node_ids
            or (
                evidence.result_hash,
                evidence.graph_version_hash,
                evidence.model_version_id,
                evidence.model_version_hash,
                evidence.threshold,
            )
            != (
                result.result_hash,
                result.graph_version_hash,
                result.model_version_id,
                result.model_version_hash,
                result.threshold,
            )
        ):
            raise GfmProxyError(502, "GFM_GLOBAL_MODEL_EVIDENCE_BINDING_MISMATCH")
        return evidence

    async def review(
        self, run_id: str, request: GlobalModelReviewRequest
    ) -> GlobalModelReviewRecord:
        result = await self.get_result(run_id)
        if request.node_id not in {item.node_id for item in result.findings}:
            raise GfmProxyError(404, "GFM_GLOBAL_MODEL_NODE_NOT_FOUND")
        return self.review_store.put(run_id, request)


__all__ = [
    "GlobalModelClientProtocol",
    "GlobalModelGateway",
    "GlobalModelReviewStore",
    "GlobalModelRunBindingStore",
]
