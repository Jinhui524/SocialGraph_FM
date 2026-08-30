"""SSRF-safe client and validation gateway for the isolated GFM service."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import stat
import threading
import uuid
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol, TypeVar
from urllib.parse import quote, urlencode, urlsplit

import httpx

from . import gfm_core_serving_control as _serving_control
from .dataset_storage import DatasetArtifactStore
from .gfm_hashing import canonical_json, canonical_sha256
from .gfm_core_serving_control import (
    CoreServingControlStore,
    CoreServingSnapshot,
    _bounded_read,
    _flush_parent_directory,
    _read_confined_snapshot,
)
from .gfm_core_schemas import (
    MAX_INTERNAL_RESPONSE_BYTES,
    CoreRunBinding,
    CoreRunBindingAnchor,
    CoreRunExpectation,
    CoreAuthorizedGraphReference,
    CoreCalibratedConfidence,
    CoreCapabilities,
    CoreRunRequest,
    CoreRunResult,
    CoreRunStatus,
    CoreInternalCreateRunReceipt,
    CoreInternalCreateRunRequest,
    CoreRegressionConfidenceInterval,
    CoreRunExecutionSnapshot,
)

MAX_GFM_RESPONSE_BYTES = MAX_INTERNAL_RESPONSE_BYTES
MAX_GOVERNANCE_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_CORE_RUN_BINDING_RECORD_BYTES = 2 * MAX_INTERNAL_RESPONSE_BYTES
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_CORE_RUN_BINDING_LOCK = threading.RLock()
_CoreRunBindingRecord = TypeVar(
    "_CoreRunBindingRecord", CoreRunBinding, CoreRunBindingAnchor
)


def _service_error_code(path: str, suffix: str) -> str:
    """Return the product-specific safe code for one internal service route."""

    namespaces = (
        ("/internal/global-model/", "GLOBAL_MODEL"),
        ("/internal/governance/", "GOVERNANCE"),
        ("/internal/research/", "RESEARCH"),
        ("/internal/core/", "CORE"),
    )
    for prefix, namespace in namespaces:
        if path.startswith(prefix):
            return f"GFM_{namespace}_{suffix}"
    return f"GFM_{suffix}"


def _reject_link_components(path: str | Path) -> Path:
    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        if current.exists() or current.is_symlink():
            details = current.lstat()
            if stat.S_ISLNK(details.st_mode) or bool(
                getattr(details, "st_file_attributes", 0) & _REPARSE_POINT
            ):
                raise ValueError("configured GFM path contains a link or reparse point")
    return candidate


class GfmProxyError(RuntimeError):
    def __init__(self, status_code: int, code: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code


class GfmClientProtocol(Protocol):
    async def core_capabilities(self) -> dict[str, Any]: ...
    async def create_core_run(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def get_core_run(self, run_id: str) -> dict[str, Any]: ...
    async def get_core_result(self, run_id: str) -> dict[str, Any]: ...

    async def research_capabilities(self) -> dict[str, Any]: ...
    async def research_scenarios(self) -> dict[str, Any]: ...
    async def research_scenario_preview(self, scenario_id: str) -> dict[str, Any]: ...
    async def create_research_run(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def get_research_run(self, run_id: str) -> dict[str, Any]: ...
    async def get_research_result(self, run_id: str) -> dict[str, Any]: ...
    async def research_similar_nodes(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def register_research_graph(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def global_model_capabilities(self) -> dict[str, Any]: ...
    async def global_model_health(self) -> dict[str, Any]: ...
    async def global_model_card(self) -> dict[str, Any]: ...
    async def global_model_scenario(self) -> dict[str, Any]: ...
    async def global_model_scenario_preview(self) -> dict[str, Any]: ...
    async def create_global_model_run(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def get_global_model_run(self, run_id: str) -> dict[str, Any]: ...
    async def get_global_model_result(self, run_id: str) -> dict[str, Any]: ...
    async def get_global_model_evidence(self, run_id: str, node_id: str) -> dict[str, Any]: ...

    async def governance_capabilities(self) -> dict[str, Any]: ...
    async def governance_health(self) -> dict[str, Any]: ...
    async def validate_governance_artifact(
        self, artifact_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...
    async def get_governance_preview(
        self, artifact_id: str, query: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...
    async def create_governance_run(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def list_governance_runs(self, offset: int, limit: int) -> dict[str, Any]: ...
    async def get_governance_run(self, run_id: str) -> dict[str, Any]: ...
    async def cancel_governance_run(self, run_id: str) -> dict[str, Any]: ...
    async def retry_governance_run(self, run_id: str) -> dict[str, Any]: ...
    async def get_governance_result(self, run_id: str) -> dict[str, Any]: ...
    async def get_governance_run_preview(
        self, run_id: str, query: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...
    async def get_governance_findings(
        self, run_id: str, offset: int, limit: int
    ) -> dict[str, Any]: ...
    async def get_governance_evidence(self, run_id: str, node_id: str) -> dict[str, Any]: ...
    async def get_governance_derivations(
        self, run_id: str, kind: str, offset: int, limit: int
    ) -> dict[str, Any]: ...
    async def execute_governance_skill(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def index_governance_case(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def create_governance_label_set(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]: ...
    async def fit_governance_policy(
        self, label_set_hash: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...
    async def get_governance_policy(self, policy_hash: str) -> dict[str, Any]: ...
    async def get_governance_adaptation_comparison(
        self, run_id: str, policy_hash: str, offset: int, limit: int
    ) -> dict[str, Any]: ...


class GfmServiceClient:
    """Fixed-origin client; redirects, environment proxies, and caller URLs are disabled."""

    def __init__(
        self,
        base_url: str,
        *,
        token_file: str | Path,
        timeout_seconds: float = 5.0,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("GFM service URL must use literal loopback 127.0.0.1")
        self._base_url = f"http://127.0.0.1:{parsed.port}"
        token_path = _reject_link_components(token_file)
        if not token_path.is_file():
            raise ValueError("GFM session token must be an existing regular file")
        self._token_file = token_path.resolve(strict=True)
        self._timeout = httpx.Timeout(timeout_seconds)
        self._governance_materialize_timeout = httpx.Timeout(
            max(timeout_seconds, 180.0)
        )

    def __repr__(self) -> str:
        return f"GfmServiceClient(base_url={self._base_url!r})"

    def _token(self) -> str:
        token = _reject_link_components(self._token_file).read_text(encoding="utf-8")
        if len(token) < 64 or token != token.strip():
            raise GfmProxyError(503, "GFM_SESSION_TOKEN_INVALID")
        return token

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> dict[str, Any]:
        def error_code(suffix: str) -> str:
            return _service_error_code(path, suffix)

        response_limit = (
            MAX_GOVERNANCE_RESPONSE_BYTES
            if path.startswith("/internal/governance/")
            else MAX_GFM_RESPONSE_BYTES
        )
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=timeout or self._timeout,
                follow_redirects=False,
                trust_env=False,
                headers={"Authorization": f"Bearer {self._token()}"},
            ) as client, client.stream(method, path, json=payload) as response:
                if 300 <= response.status_code < 400:
                    raise GfmProxyError(502, error_code("REDIRECT_REJECTED"))
                declared = response.headers.get("content-length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError as error:
                        raise GfmProxyError(502, error_code("RESPONSE_INVALID")) from error
                    if declared_size < 0:
                        raise GfmProxyError(502, error_code("RESPONSE_INVALID"))
                    if declared_size > response_limit:
                        raise GfmProxyError(502, error_code("RESPONSE_TOO_LARGE"))
                media_type = response.headers.get("content-type", "").partition(";")[0].lower()
                if media_type != "application/json":
                    raise GfmProxyError(502, error_code("RESPONSE_INVALID"))
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > response_limit:
                        raise GfmProxyError(502, error_code("RESPONSE_TOO_LARGE"))
                    chunks.append(chunk)
                content = b"".join(chunks)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError, OSError) as error:
            raise GfmProxyError(503, error_code("SERVICE_UNAVAILABLE")) from error
        try:
            body = json.loads(content)
        except (UnicodeDecodeError, ValueError) as error:
            raise GfmProxyError(502, error_code("RESPONSE_INVALID")) from error
        if not isinstance(body, dict):
            raise GfmProxyError(502, error_code("RESPONSE_INVALID"))
        if response.status_code >= 400:
            error_value = body.get("error")
            code = error_value.get("code") if isinstance(error_value, dict) else None
            safe_code = (
                code
                if isinstance(code, str) and re.fullmatch(r"[A-Z0-9_]{1,100}", code)
                else error_code("SERVICE_ERROR")
            )
            raise GfmProxyError(response.status_code, safe_code)
        return body

    async def core_capabilities(self) -> dict[str, Any]:
        return await self._request("GET", "/internal/core/capabilities")

    async def create_core_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/internal/core/runs", payload=payload)

    async def get_core_run(self, run_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/internal/core/runs/{run_id}")

    async def get_core_result(self, run_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/internal/core/runs/{run_id}/result")

    async def research_capabilities(self) -> dict[str, Any]:
        return await self._request("GET", "/internal/research/capabilities")

    async def research_scenarios(self) -> dict[str, Any]:
        return await self._request("GET", "/internal/research/scenarios")

    async def research_scenario_preview(self, scenario_id: str) -> dict[str, Any]:
        if re.fullmatch(r"[a-z0-9-]{1,100}", scenario_id) is None:
            raise GfmProxyError(400, "GFM_RESEARCH_SCENARIO_NOT_FOUND")
        return await self._request(
            "GET", f"/internal/research/scenarios/{scenario_id}/graph-preview"
        )

    async def create_research_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/internal/research/runs", payload=payload)

    async def get_research_run(self, run_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/internal/research/runs/{run_id}")

    async def get_research_result(self, run_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", f"/internal/research/runs/{run_id}/result"
        )

    async def research_similar_nodes(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request(
            "POST", "/internal/research/similar-nodes", payload=payload
        )

    async def register_research_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request(
            "POST", "/internal/research/graphs/register", payload=payload
        )

    async def global_model_capabilities(self) -> dict[str, Any]:
        return await self._request("GET", "/internal/global-model/capabilities")

    async def global_model_health(self) -> dict[str, Any]:
        return await self._request("GET", "/internal/global-model/health")

    async def global_model_card(self) -> dict[str, Any]:
        return await self._request("GET", "/internal/global-model/model-card")

    async def global_model_scenario(self) -> dict[str, Any]:
        return await self._request("GET", "/internal/global-model/scenario")

    async def global_model_scenario_preview(self) -> dict[str, Any]:
        return await self._request("GET", "/internal/global-model/scenario/graph-preview")

    async def create_global_model_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/internal/global-model/runs", payload=payload)

    async def get_global_model_run(self, run_id: str) -> dict[str, Any]:
        if re.fullmatch(r"global-model-[0-9a-f]{32}", run_id) is None:
            raise GfmProxyError(404, "GFM_GLOBAL_MODEL_RUN_NOT_FOUND")
        return await self._request("GET", f"/internal/global-model/runs/{run_id}")

    async def get_global_model_result(self, run_id: str) -> dict[str, Any]:
        if re.fullmatch(r"global-model-[0-9a-f]{32}", run_id) is None:
            raise GfmProxyError(404, "GFM_GLOBAL_MODEL_RUN_NOT_FOUND")
        return await self._request("GET", f"/internal/global-model/runs/{run_id}/result")

    async def get_global_model_evidence(self, run_id: str, node_id: str) -> dict[str, Any]:
        if re.fullmatch(r"global-model-[0-9a-f]{32}", run_id) is None:
            raise GfmProxyError(404, "GFM_GLOBAL_MODEL_RUN_NOT_FOUND")
        if re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", node_id) is None:
            raise GfmProxyError(404, "GFM_GLOBAL_MODEL_NODE_NOT_FOUND")
        return await self._request(
            "GET", f"/internal/global-model/runs/{run_id}/nodes/{node_id}/evidence"
        )

    @staticmethod
    def _governance_run_id(run_id: str) -> str:
        if re.fullmatch(r"governance-[0-9a-f]{32}", run_id) is None:
            raise GfmProxyError(404, "GOVERNANCE_RUN_NOT_FOUND")
        return run_id

    @staticmethod
    def _governance_artifact_id(artifact_id: str) -> str:
        if re.fullmatch(r"governance-artifact-[0-9a-f]{32}", artifact_id) is None:
            raise GfmProxyError(404, "GOVERNANCE_ARTIFACT_NOT_FOUND")
        return artifact_id

    async def governance_capabilities(self) -> dict[str, Any]:
        return await self._request("GET", "/internal/governance/capabilities")

    async def governance_health(self) -> dict[str, Any]:
        return await self._request("GET", "/internal/governance/health")

    async def validate_governance_artifact(
        self, artifact_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        artifact_id = self._governance_artifact_id(artifact_id)
        return await self._request(
            "POST",
            f"/internal/governance/artifacts/{artifact_id}/materialize",
            payload=payload,
            timeout=self._governance_materialize_timeout,
        )

    async def get_governance_preview(
        self, artifact_id: str, query: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        artifact_id = self._governance_artifact_id(artifact_id)
        suffix = f"?{urlencode(query, doseq=True)}" if query else ""
        return await self._request(
            "GET", f"/internal/governance/artifacts/{artifact_id}/preview{suffix}"
        )

    async def create_governance_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/internal/governance/runs", payload=payload)

    async def list_governance_runs(self, offset: int, limit: int) -> dict[str, Any]:
        query = urlencode({"offset": offset, "limit": limit})
        return await self._request("GET", f"/internal/governance/runs?{query}")

    async def get_governance_run(self, run_id: str) -> dict[str, Any]:
        run_id = self._governance_run_id(run_id)
        return await self._request("GET", f"/internal/governance/runs/{run_id}")

    async def cancel_governance_run(self, run_id: str) -> dict[str, Any]:
        run_id = self._governance_run_id(run_id)
        return await self._request("POST", f"/internal/governance/runs/{run_id}/cancel")

    async def retry_governance_run(self, run_id: str) -> dict[str, Any]:
        run_id = self._governance_run_id(run_id)
        return await self._request("POST", f"/internal/governance/runs/{run_id}/retry")

    async def get_governance_result(self, run_id: str) -> dict[str, Any]:
        run_id = self._governance_run_id(run_id)
        return await self._request("GET", f"/internal/governance/runs/{run_id}/result")

    async def get_governance_run_preview(
        self, run_id: str, query: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        run_id = self._governance_run_id(run_id)
        suffix = f"?{urlencode(query, doseq=True)}" if query else ""
        return await self._request(
            "GET", f"/internal/governance/runs/{run_id}/preview{suffix}"
        )

    async def get_governance_findings(
        self, run_id: str, offset: int, limit: int
    ) -> dict[str, Any]:
        run_id = self._governance_run_id(run_id)
        query = urlencode({"offset": offset, "limit": limit})
        return await self._request(
            "GET", f"/internal/governance/runs/{run_id}/findings?{query}"
        )

    async def get_governance_evidence(self, run_id: str, node_id: str) -> dict[str, Any]:
        run_id = self._governance_run_id(run_id)
        if not node_id or len(node_id) > 128 or any(ord(value) < 0x20 for value in node_id):
            raise GfmProxyError(404, "GOVERNANCE_NODE_NOT_FOUND")
        encoded = quote(node_id, safe="")
        return await self._request(
            "GET", f"/internal/governance/runs/{run_id}/nodes/{encoded}/evidence"
        )

    async def get_governance_derivations(
        self, run_id: str, kind: str, offset: int, limit: int
    ) -> dict[str, Any]:
        run_id = self._governance_run_id(run_id)
        endpoint = {
            "groups": "groups",
            "relations": "relations",
            "potential-links": "links",
        }.get(kind)
        if endpoint is None:
            raise GfmProxyError(404, "GOVERNANCE_DERIVATION_NOT_FOUND")
        query = urlencode({"offset": offset, "limit": limit})
        return await self._request(
            "GET", f"/internal/governance/runs/{run_id}/{endpoint}?{query}"
        )

    async def execute_governance_skill(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._request(
            "POST", "/internal/governance/skills/execute", payload=payload
        )

    async def index_governance_case(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._request(
            "POST", "/internal/governance/reviewed-cases/index", payload=payload
        )

    @staticmethod
    def _adaptation_hash(value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise GfmProxyError(404, "GOVERNANCE_ADAPTATION_NOT_FOUND")
        return value

    async def create_governance_label_set(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._request(
            "POST", "/internal/governance/adaptations/label-sets", payload=payload
        )

    async def fit_governance_policy(
        self, label_set_hash: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        label_set_hash = self._adaptation_hash(label_set_hash)
        return await self._request(
            "POST",
            f"/internal/governance/adaptations/label-sets/{label_set_hash}/policies",
            payload=payload,
        )

    async def get_governance_policy(self, policy_hash: str) -> dict[str, Any]:
        policy_hash = self._adaptation_hash(policy_hash)
        return await self._request(
            "GET", f"/internal/governance/adaptations/policies/{policy_hash}"
        )

    async def get_governance_adaptation_comparison(
        self, run_id: str, policy_hash: str, offset: int, limit: int
    ) -> dict[str, Any]:
        run_id = self._governance_run_id(run_id)
        policy_hash = self._adaptation_hash(policy_hash)
        query = urlencode({"offset": offset, "limit": limit})
        return await self._request(
            "GET",
            f"/internal/governance/adaptations/runs/{run_id}/policies/"
            f"{policy_hash}/comparison?{query}",
        )


def _requested_entity_type(request: CoreRunRequest) -> str:
    scope = request.target_scope
    if scope.kind == "community":
        return "community"
    if scope.kind == "node-pairs":
        return "node-pair"
    return "node" if scope.node_ids else "edge"


def _validate_capability_snapshot(
    capabilities: CoreCapabilities, snapshot: CoreServingSnapshot
) -> None:
    if (
        capabilities.control_hash != snapshot.control.control_hash
        or capabilities.control_generation != snapshot.control.generation
        or capabilities.registry_hash != snapshot.registry_hash
        or capabilities.registry_generation != snapshot.registry.generation
        or capabilities.catalog_hash != snapshot.catalog_hash
        or capabilities.catalog_generation != snapshot.catalog.generation
    ):
        raise ValueError("GFM capabilities do not match API serving-control identity")
    expected_models = tuple(
        {
            "modelVersionId": model.model_version_id,
            "modelVersionHash": model.model_version_hash,
            "state": model.state,
            "tasks": list(model.tasks),
            "graphSchemaVersions": list(model.graph_schema_versions),
            "graphFeatureContractHash": model.graph_feature_contract_hash,
            "taskBindings": [
                {
                    "taskId": head.task_id,
                    "entityType": binding.entity_type,
                    "confidenceKind": binding.confidence_kind,
                    "calibrationVersion": binding.calibration_version,
                    "method": binding.calibration_method,
                    "calibrationArtifactHash": binding.calibration_artifact_hash,
                    "calibrationProtocolHash": binding.calibration_protocol_hash,
                    "adapterDomain": binding.adapter_domain,
                    "adapterSchemaHash": binding.adapter_schema_hash,
                    "adapterStateHash": binding.adapter_state_hash,
                    "featureContractHash": binding.graph_feature_contract_hash,
                }
                for head in model.task_heads
                for binding in head.calibrations
            ],
            "maxNodes": model.max_nodes,
            "maxEdges": model.max_edges,
        }
        for model in snapshot.registry.models
    )
    observed_models = tuple(
        model.model_dump(mode="json", by_alias=True) for model in capabilities.models
    )
    expected_tasks = tuple(
        sorted({task for model in snapshot.registry.models for task in model.tasks})
    )
    if observed_models != expected_models or capabilities.tasks != expected_tasks:
        raise ValueError("GFM capability model projection does not match API registry")


def _expectation_from_snapshot(
    snapshot: CoreServingSnapshot,
    create_request: CoreInternalCreateRunRequest,
) -> CoreRunExpectation:
    request = create_request.request
    graph = create_request.graph_reference
    model = snapshot.model(request.model_version_id)
    manifest = snapshot.manifest(request.model_version_id)
    entry = next(
        (
            item
            for item in snapshot.catalog.artifacts
            if item.artifact_id == graph.artifact_id
            and item.graph_version_id == graph.graph_version_id
        ),
        None,
    )
    if entry is None:
        raise ValueError("authorized graph is absent from accepted API catalog")
    expected_graph = CoreAuthorizedGraphReference(
        schemaVersion="socialgraph-fm.core-authorized-graph-reference/2.1",
        graphVersionId=entry.graph_version_id,
        sourceGraphFactHash=entry.source_graph_fact_hash,
        graphVersionHash=entry.graph_version_hash,
        artifactId=entry.artifact_id,
        artifactHash=entry.artifact_hash,
        bundleSha256=entry.bundle_sha256,
        graphSchemaVersion=entry.graph_schema_version,
        featureContractHash=entry.feature_contract_hash,
        nodeCount=entry.node_count,
        edgeCount=entry.edge_count,
    )
    if graph != expected_graph:
        raise ValueError("authorized graph does not match accepted API catalog")
    entity_type = _requested_entity_type(request)
    head = next(
        (item for item in model.task_heads if item.task_id == request.task_id), None
    )
    binding = (
        next(
            (
                item
                for item in head.calibrations
                if item.entity_type == entity_type
            ),
            None,
        )
        if head is not None
        else None
    )
    if (
        model.state != "servingReady"
        or request.task_id not in model.tasks
        or binding is None
        or graph.graph_schema_version not in model.graph_schema_versions
        or graph.feature_contract_hash != binding.graph_feature_contract_hash
        or graph.node_count > model.max_nodes
        or graph.edge_count > model.max_edges
    ):
        raise ValueError("request is incompatible with accepted API model metadata")
    assert head is not None and binding is not None
    calibration_identities = [
        {
            "entityType": binding.entity_type,
            "confidenceKind": binding.confidence_kind,
            "calibrationVersion": binding.calibration_version,
            "method": binding.calibration_method,
            "calibrationArtifactHash": binding.calibration_artifact_hash,
            "calibrationProtocolHash": binding.calibration_protocol_hash,
            "adapterDomain": binding.adapter_domain,
            "adapterSchemaHash": binding.adapter_schema_hash,
            "adapterStateHash": binding.adapter_state_hash,
            "featureContractHash": binding.graph_feature_contract_hash,
            "sha256": binding.calibration_sha256,
        }
        for binding in sorted(head.calibrations, key=lambda item: item.entity_type)
    ]
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-api-run-expectation/2.2",
        "createRequest": create_request.model_dump(mode="json", by_alias=True),
        "controlSourceSha256": hashlib.sha256(
            snapshot.control_source_bytes
        ).hexdigest(),
        "registrySourceSha256": snapshot.registry_source_sha256,
        "artifactCatalogSha256": snapshot.catalog_source_sha256,
        "checkpointSha256": model.checkpoint.sha256,
        "servingManifestSha256": manifest.source_sha256,
        "adapterSchemaHash": binding.adapter_schema_hash,
        "calibrationIdentities": calibration_identities,
        "calibrationSetHash": canonical_sha256(calibration_identities),
    }
    return CoreRunExpectation.model_validate(payload)


def _result_findings_match_snapshot(
    result: CoreRunResult, snapshot: CoreRunExecutionSnapshot
) -> bool:
    identities = {
        identity.entity_type: identity
        for identity in snapshot.calibration_identities
    }
    for finding in result.findings:
        score = finding.score
        confidence = finding.calibrated_confidence
        identity = identities.get(score.entity_type)
        if isinstance(confidence, CoreCalibratedConfidence):
            confidence_kind = "binary-calibration"
            confidence_version = confidence.calibration_version
            artifact_hash = confidence.calibration_artifact_hash
            protocol_hash = confidence.calibration_protocol_hash
        elif isinstance(confidence, CoreRegressionConfidenceInterval):
            confidence_kind = "regression-interval"
            confidence_version = confidence.confidence_version
            artifact_hash = confidence.confidence_artifact_hash
            protocol_hash = confidence.confidence_protocol_hash
        else:  # pragma: no cover - the closed schema union makes this unreachable
            return False
        if (
            identity is None
            or finding.task_id != snapshot.task_id
            or score.task_id != snapshot.task_id
            or score.graph_version_hash != snapshot.graph_version_hash
            or score.model_version != snapshot.model_version_id
            or score.model_version_hash != snapshot.model_version_hash
            or confidence.score_hash != score.score_hash
            or confidence.task_id != score.task_id
            or confidence.entity_type != score.entity_type
            or confidence.entity_ids != score.entity_ids
            or confidence.graph_version_hash != score.graph_version_hash
            or confidence.model_version != score.model_version
            or confidence.model_version_hash != score.model_version_hash
            or identity.confidence_kind != confidence_kind
            or identity.calibration_version != confidence_version
            or identity.method != confidence.method
            or identity.calibration_artifact_hash != artifact_hash
            or identity.calibration_protocol_hash != protocol_hash
        ):
            return False
    return True


class CoreGraphResolver:
    def __init__(
        self,
        store: DatasetArtifactStore,
        *,
        serving_control_store: CoreServingControlStore | None = None,
        control_file: str | Path | None = None,
    ) -> None:
        self.store = store
        if serving_control_store is not None and control_file is not None:
            raise ValueError("configure one API serving-control authority")
        if serving_control_store is None and control_file is not None:
            control_path = _reject_link_components(control_file).resolve(strict=True)
            serving_control_store = CoreServingControlStore(
                control_path,
                high_water_root=control_path.parent / ".api-serving-control-high-water",
            )
        self.serving_control_store = serving_control_store

    def resolve(
        self,
        graph_version_id: str,
        capabilities: CoreCapabilities,
        *,
        required_model_id: str | None = None,
    ) -> CoreAuthorizedGraphReference:
        binding = self.store.resolve_graph_version_binding(graph_version_id)
        if binding is None:
            raise LookupError("graph version not found")
        artifact = self.store.get_artifact(binding.artifact_id)
        if artifact is None:
            raise LookupError("graph artifact not active")
        if self.serving_control_store is None:
            raise LookupError("graph is not present in the serving graph catalog")
        try:
            snapshot = self.serving_control_store.acquire(required_model_id)
        except (LookupError, OSError, ValueError) as error:
            raise GfmProxyError(503, "GFM_CORE_SERVING_CONTROL_INVALID") from error
        try:
            _validate_capability_snapshot(capabilities, snapshot)
        except ValueError as error:
            raise GfmProxyError(503, "GFM_CORE_SERVING_CONTROL_STALE") from error
        catalog = snapshot.catalog
        entry = next(
            (
                item
                for item in catalog.artifacts
                if item.artifact_id == binding.artifact_id
                and item.graph_version_id == binding.graph_version_id
            ),
            None,
        )
        if entry is None:
            raise LookupError("graph is not present in the serving graph catalog")
        artifact_hash = artifact.content_hash
        if re.fullmatch(r"[0-9a-f]{64}", artifact_hash or "") is None:
            raise ValueError("graph artifact has no immutable hash")
        if artifact_hash != entry.artifact_hash:
            raise ValueError("artifact identity does not match serving graph catalog")
        if binding.graph_fact_hash != entry.source_graph_fact_hash:
            raise ValueError("graph identity does not match serving graph catalog")
        if (
            artifact.profile.node_count != entry.node_count
            or artifact.profile.edge_count != entry.edge_count
        ):
            raise ValueError("artifact counts do not match serving graph catalog")
        return CoreAuthorizedGraphReference(
            schemaVersion="socialgraph-fm.core-authorized-graph-reference/2.1",
            graphVersionId=binding.graph_version_id,
            sourceGraphFactHash=binding.graph_fact_hash,
            graphVersionHash=entry.graph_version_hash,
            artifactId=binding.artifact_id,
            artifactHash=artifact_hash,
            bundleSha256=entry.bundle_sha256,
            graphSchemaVersion=entry.graph_schema_version,
            featureContractHash=entry.feature_contract_hash,
            nodeCount=entry.node_count,
            edgeCount=entry.edge_count,
        )


class CoreRunBindingStore:
    """API-owned immutable create-time bindings used for every later read.

    The unkeyed continuity guarantee assumes this private publication namespace is
    append-only after publication. A coherent rewrite of both the binding and its
    independently published anchor is outside that guarantee.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = _reject_link_components(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = _reject_link_components(self.root)
        self.root = self.root.resolve(strict=True)
        self._lock = threading.RLock()

    def _path(self, run_id: str) -> Path:
        CoreGateway._validate_run_id(run_id)
        return _reject_link_components(self.root / f"{run_id}.json")

    def _anchor_path(self, run_id: str) -> Path:
        CoreGateway._validate_run_id(run_id)
        return _reject_link_components(self.root / f"{run_id}.anchor.json")

    def save(self, binding: CoreRunBinding) -> None:
        with self._lock:
            self._save(binding)

    def _save(self, binding: CoreRunBinding) -> None:
        try:
            anchor_payload: dict[str, Any] = {
                "schemaVersion": "socialgraph-fm.core-api-run-binding-anchor/1.0",
                "runId": binding.run_id,
                "bindingHash": binding.binding_hash,
            }
            anchor_payload["anchorHash"] = canonical_sha256(anchor_payload)
            anchor = CoreRunBindingAnchor.model_validate(anchor_payload)
            with _CORE_RUN_BINDING_LOCK:
                path = self._path(binding.run_id)
                anchor_path = self._anchor_path(binding.run_id)
                anchor_exists = anchor_path.exists()
                binding_exists = path.exists()
                if anchor_exists:
                    if not binding_exists:
                        raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID")
                    _flush_parent_directory(self.root)
                    self._require_same_record(anchor_path, anchor)
                    self._require_same_record(path, binding)
                    return
                if binding_exists:
                    _flush_parent_directory(self.root)
                    self._require_same_record(path, binding)
            if not binding_exists:
                self._publish_record(path, binding)
            owned_anchor = self._publish_record(anchor_path, anchor)
            # The anchor is append-only by contract; capture it before the
            # mutable binding, which must be the last input to acceptance.
            self._require_same_record(anchor_path, anchor)
            try:
                self._require_same_record(path, binding)
            except GfmProxyError:
                if owned_anchor is not None:
                    self._remove_owned_record(anchor_path, anchor, owned_anchor)
                raise
        except GfmProxyError:
            raise
        except (OSError, ValueError) as error:
            raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID") from error

    @staticmethod
    def _canonical_record_bytes(
        record: CoreRunBinding | CoreRunBindingAnchor,
    ) -> bytes:
        payload = record.model_dump(mode="json", by_alias=True)
        return (canonical_json(payload) + "\n").encode("utf-8")

    def _publish_record(
        self, path: Path, record: _CoreRunBindingRecord
    ) -> tuple[int, int] | None:
        expected_bytes = self._canonical_record_bytes(record)
        # The sibling is unique without repeating the long destination name; this
        # keeps atomic publication usable in deep Windows checkout paths.
        temporary = path.with_name(f".tmp-{uuid.uuid4().hex}.tmp")
        owned_identity: tuple[int, int] | None = None
        try:
            if not path.exists():
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_BINARY", 0),
                    0o600,
                )
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(expected_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary_identity = self._require_same_record(
                    temporary, record, expected_bytes=expected_bytes
                )
                try:
                    # A hard-link publication is atomic and never replaces an
                    # existing winner. Exact concurrent writers verify that winner.
                    os.link(temporary, path)
                except FileExistsError:
                    pass
                else:
                    owned_identity = temporary_identity
            # Removing the temporary hard link changes the published inode's
            # ctime on POSIX. Keep cleanup and every in-process verifier in the
            # same short critical section without serializing the no-clobber link.
            with _CORE_RUN_BINDING_LOCK:
                temporary.unlink(missing_ok=True)
                _flush_parent_directory(self.root)
                final_identity = self._require_same_record(
                    path, record, expected_bytes=expected_bytes
                )
                if owned_identity is not None and final_identity != owned_identity:
                    raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID")
                return owned_identity
        except GfmProxyError:
            raise
        except (OSError, ValueError) as error:
            raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID") from error
        finally:
            with _CORE_RUN_BINDING_LOCK:
                temporary.unlink(missing_ok=True)

    def _remove_owned_record(
        self,
        path: Path,
        record: _CoreRunBindingRecord,
        owned_identity: tuple[int, int],
    ) -> None:
        with _CORE_RUN_BINDING_LOCK:
            try:
                current_identity = self._require_same_record(path, record)
            except GfmProxyError:
                return
            if current_identity != owned_identity:
                return
            expected_bytes = self._canonical_record_bytes(record)
            if not self._remove_owned_path(path, expected_bytes, owned_identity):
                return
            _flush_parent_directory(self.root)
            try:
                os.lstat(path)
            except FileNotFoundError:
                return
            raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID")

    def _remove_owned_path(
        self,
        path: Path,
        expected_bytes: bytes,
        owned_identity: tuple[int, int],
    ) -> bool:
        if os.name == "nt":
            return self._remove_owned_windows(path, expected_bytes, owned_identity)
        return self._remove_owned_posix(path, expected_bytes, owned_identity)

    def _remove_owned_windows(
        self,
        path: Path,
        expected_bytes: bytes,
        owned_identity: tuple[int, int],
    ) -> bool:
        create_file = _serving_control._CreateFileW
        close_handle = _serving_control._CloseHandle
        invalid_handle = _serving_control._INVALID
        read_file = _serving_control._ReadFile
        win_id = _serving_control._win_id
        win_final_path = _serving_control._win_final_path
        win_info = _serving_control._win_info
        flags = 0x00200000 | 0x02000000
        parent_handle = create_file(str(self.root), 0x80, 1, None, 3, flags, None)
        if parent_handle == invalid_handle:
            raise OSError(ctypes.get_last_error(), "failed to hold binding root")
        final_handle: int | None = None
        try:
            parent_info = win_info(parent_handle)
            if parent_info.attributes & _REPARSE_POINT or not (
                parent_info.attributes & 0x10
            ):
                raise ValueError("binding root must be a held non-reparse directory")
            final_handle = create_file(
                str(path),
                0x80000000 | 0x00010000 | 0x80,
                1,
                None,
                3,
                flags,
                None,
            )
            if final_handle == invalid_handle:
                error = ctypes.get_last_error()
                if error in {2, 3}:
                    return False
                raise OSError(error, "failed to hold owned binding record")
            assert final_handle is not None
            before = win_info(final_handle)
            observed_path = os.path.normcase(os.path.abspath(win_final_path(final_handle)))
            expected_path = os.path.normcase(os.path.abspath(path))
            if (
                before.attributes & (_REPARSE_POINT | 0x10)
                or win_id(final_handle) != owned_identity
                or observed_path != expected_path
            ):
                return False
            size = (before.size_high << 32) | before.size_low
            if size != len(expected_bytes) or size > MAX_CORE_RUN_BINDING_RECORD_BYTES:
                return False
            chunks: list[bytes] = []
            remaining = size
            while remaining:
                length = min(1024 * 1024, remaining)
                buffer = ctypes.create_string_buffer(length)
                read = ctypes.c_ulong()
                if (
                    not read_file(
                        final_handle,
                        buffer,
                        length,
                        ctypes.byref(read),
                        None,
                    )
                    or not read.value
                ):
                    raise ValueError("owned binding record changed while reading")
                chunks.append(buffer.raw[: read.value])
                remaining -= read.value
            after = win_info(final_handle)
            if (
                b"".join(chunks) != expected_bytes
                or win_id(final_handle) != owned_identity
                or (
                    after.size_high,
                    after.size_low,
                    after.written.dwHighDateTime,
                    after.written.dwLowDateTime,
                )
                != (
                    before.size_high,
                    before.size_low,
                    before.written.dwHighDateTime,
                    before.written.dwLowDateTime,
                )
            ):
                return False

            class _Disposition(ctypes.Structure):
                _fields_ = [("delete_file", ctypes.c_ubyte)]

            kernel32 = _serving_control._kernel32
            set_information = kernel32.SetFileInformationByHandle
            set_information.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_ulong,
            ]
            set_information.restype = ctypes.c_int
            disposition = _Disposition(1)
            if not set_information(
                final_handle,
                4,
                ctypes.byref(disposition),
                ctypes.sizeof(disposition),
            ):
                raise OSError(
                    ctypes.get_last_error(), "failed to remove owned binding record"
                )
            return True
        finally:
            if final_handle is not None and final_handle != invalid_handle:
                close_handle(final_handle)
            close_handle(parent_handle)

    def _remove_owned_posix(
        self,
        path: Path,
        expected_bytes: bytes,
        owned_identity: tuple[int, int],
    ) -> bool:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        root_descriptor = os.open(self.root, directory_flags)
        quarantine_name = f".{path.name}.{uuid.uuid4().hex}.cleanup"
        quarantine_descriptor: int | None = None
        quarantine_created = False
        record_quarantined = False
        try:
            root_before = self.root.lstat()
            root_opened = os.fstat(root_descriptor)
            if (
                not stat.S_ISDIR(root_opened.st_mode)
                or (root_opened.st_dev, root_opened.st_ino)
                != (root_before.st_dev, root_before.st_ino)
            ):
                raise ValueError("binding root changed before cleanup")
            os.mkdir(quarantine_name, 0o700, dir_fd=root_descriptor)
            quarantine_created = True
            quarantine_descriptor = os.open(
                quarantine_name, directory_flags, dir_fd=root_descriptor
            )
            try:
                os.rename(
                    path.name,
                    "record",
                    src_dir_fd=root_descriptor,
                    dst_dir_fd=quarantine_descriptor,
                )
            except FileNotFoundError:
                return False
            record_quarantined = True
            final_descriptor: int | None = None
            matches_owned = False
            try:
                final_descriptor = os.open(
                    "record",
                    os.O_RDONLY
                    | getattr(os, "O_BINARY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=quarantine_descriptor,
                )
                before = os.fstat(final_descriptor)
                if (
                    stat.S_ISREG(before.st_mode)
                    and (before.st_dev, before.st_ino) == owned_identity
                    and before.st_size == len(expected_bytes)
                ):
                    payload = _bounded_read(
                        final_descriptor,
                        size=before.st_size,
                        max_bytes=MAX_CORE_RUN_BINDING_RECORD_BYTES,
                    )
                    after = os.fstat(final_descriptor)
                    latest = os.stat(
                        "record",
                        dir_fd=quarantine_descriptor,
                        follow_symlinks=False,
                    )
                    matches_owned = (
                        payload == expected_bytes
                        and (after.st_dev, after.st_ino) == owned_identity
                        and (latest.st_dev, latest.st_ino) == owned_identity
                        and (
                            after.st_size,
                            after.st_mtime_ns,
                            after.st_ctime_ns,
                        )
                        == (
                            before.st_size,
                            before.st_mtime_ns,
                            before.st_ctime_ns,
                        )
                        and (
                            latest.st_size,
                            latest.st_mtime_ns,
                            latest.st_ctime_ns,
                        )
                        == (
                            after.st_size,
                            after.st_mtime_ns,
                            after.st_ctime_ns,
                        )
                    )
            except (OSError, ValueError):
                matches_owned = False
            finally:
                if final_descriptor is not None:
                    os.close(final_descriptor)
            if matches_owned:
                os.unlink("record", dir_fd=quarantine_descriptor)
                record_quarantined = False
                os.fsync(quarantine_descriptor)
                return True
            try:
                os.link(
                    "record",
                    path.name,
                    src_dir_fd=quarantine_descriptor,
                    dst_dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                os.fsync(quarantine_descriptor)
                os.fsync(root_descriptor)
                return False
            os.unlink("record", dir_fd=quarantine_descriptor)
            record_quarantined = False
            os.fsync(quarantine_descriptor)
            os.fsync(root_descriptor)
            return False
        finally:
            if quarantine_descriptor is not None:
                os.close(quarantine_descriptor)
            if quarantine_created and not record_quarantined:
                os.rmdir(quarantine_name, dir_fd=root_descriptor)
                os.fsync(root_descriptor)
            os.close(root_descriptor)

    def _read_record(
        self,
        path: Path,
        record_type: type[_CoreRunBindingRecord],
    ) -> tuple[_CoreRunBindingRecord, bytes, tuple[int, int]]:
        with _CORE_RUN_BINDING_LOCK:
            try:
                captured = _read_confined_snapshot(
                    self.root,
                    path.name,
                    max_bytes=MAX_CORE_RUN_BINDING_RECORD_BYTES,
                )
                observed = record_type.model_validate_json(captured.payload)
            except (OSError, ValueError) as error:
                raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID") from error
            return observed, captured.payload, (captured.token[0], captured.token[1])

    def _require_same_record(
        self,
        path: Path,
        record: _CoreRunBindingRecord,
        *,
        expected_bytes: bytes | None = None,
    ) -> tuple[int, int]:
        observed, observed_bytes, identity = self._read_record(path, type(record))
        canonical_bytes = expected_bytes or self._canonical_record_bytes(record)
        if observed != record or observed_bytes != canonical_bytes:
            raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID")
        return identity

    def get(self, run_id: str) -> CoreRunBinding:
        with self._lock:
            return self._get(run_id)

    def _get(self, run_id: str) -> CoreRunBinding:
        try:
            path = self._path(run_id)
            anchor_path = self._anchor_path(run_id)
        except GfmProxyError:
            raise
        except (OSError, ValueError) as error:
            raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID") from error
        if not path.is_file():
            if anchor_path.exists():
                raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID")
            raise GfmProxyError(404, "GFM_CORE_RUN_NOT_FOUND")
        if not anchor_path.is_file():
            raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID")
        anchor, anchor_bytes, _anchor_identity = self._read_record(
            anchor_path, CoreRunBindingAnchor
        )
        binding, binding_bytes, _binding_identity = self._read_record(
            path, CoreRunBinding
        )
        if (
            binding.run_id != run_id
            or anchor.run_id != run_id
            or anchor.binding_hash != binding.binding_hash
            or anchor_bytes != self._canonical_record_bytes(anchor)
            or binding_bytes != self._canonical_record_bytes(binding)
        ):
            raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID")
        return binding


class CoreGateway:
    def __init__(
        self,
        client: GfmClientProtocol | None,
        *,
        binding_store: CoreRunBindingStore | None = None,
        serving_control_store: CoreServingControlStore | None = None,
    ) -> None:
        self.client = client
        self.binding_store = binding_store
        self.serving_control_store = serving_control_store
        self._orphaned_create_count = 0

    @staticmethod
    def canonical_hash(value: Any) -> str:
        return canonical_sha256(value)

    @staticmethod
    def request_hash(envelope: dict[str, Any]) -> str:
        return canonical_sha256(envelope)

    def _default_capabilities(self) -> dict[str, Any]:
        control_path = files("app").joinpath("contracts/core-serving-control.json")
        registry_path = files("app").joinpath("contracts/core-serving-registry.json")
        catalog_path = files("app").joinpath("contracts/core-serving-graph-catalog.json")
        control = json.loads(control_path.read_text(encoding="utf-8"))
        document = json.loads(registry_path.read_text(encoding="utf-8"))
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if (
            control["registry"]["semanticHash"] != canonical_sha256(document)
            or control["catalog"]["semanticHash"] != canonical_sha256(catalog)
        ):
            raise GfmProxyError(503, "GFM_CORE_SERVING_CONTROL_INVALID")
        return {
            "schemaVersion": "socialgraph-fm.core-capabilities/2.0",
            "controlHash": control["controlHash"],
            "controlGeneration": control["generation"],
            "registryHash": canonical_sha256(document),
            "registryGeneration": document["generation"],
            "catalogHash": canonical_sha256(catalog),
            "catalogGeneration": catalog["generation"],
            "servingReady": False,
            "models": [],
            "tasks": [],
            "readiness": {"modelValidated": False, "coreServingReady": False},
        }

    async def capabilities(self) -> CoreCapabilities:
        payload = (
            self._default_capabilities()
            if self.client is None
            else await self.client.core_capabilities()
        )
        try:
            capabilities = CoreCapabilities.model_validate_json(
                json.dumps(payload, ensure_ascii=False)
            )
        except ValueError as error:
            raise GfmProxyError(502, "GFM_CORE_CAPABILITIES_INVALID") from error
        if self.serving_control_store is None:
            return capabilities
        try:
            snapshot = self.serving_control_store.acquire()
        except (LookupError, OSError, ValueError) as error:
            raise GfmProxyError(503, "GFM_CORE_SERVING_CONTROL_INVALID") from error
        try:
            _validate_capability_snapshot(capabilities, snapshot)
        except ValueError as error:
            raise GfmProxyError(503, "GFM_CORE_SERVING_CONTROL_STALE") from error
        return capabilities

    @staticmethod
    def validate_compatibility(
        request: CoreRunRequest,
        graph: CoreAuthorizedGraphReference,
        capabilities: CoreCapabilities,
    ) -> None:
        model = next(
            (
                item
                for item in capabilities.models
                if item.model_version_id == request.model_version_id
            ),
            None,
        )
        if model is None or model.state != "servingReady" or request.task_id not in model.tasks:
            raise ValueError("model/task unavailable")
        entity_type = _requested_entity_type(request)
        binding = next(
            (
                item
                for item in model.task_bindings
                if item.task_id == request.task_id and item.entity_type == entity_type
            ),
            None,
        )
        if (
            binding is None
            or graph.graph_schema_version not in model.graph_schema_versions
            or graph.feature_contract_hash != binding.feature_contract_hash
            or graph.node_count > model.max_nodes
            or graph.edge_count > model.max_edges
        ):
            raise ValueError("model/graph incompatible")

    async def create_run(
        self,
        request: CoreRunRequest,
        graph: CoreAuthorizedGraphReference,
        capabilities: CoreCapabilities,
    ) -> CoreRunStatus:
        if self.client is None:
            raise GfmProxyError(503, "GFM_CORE_MODEL_NOT_INSTALLED")
        if self.serving_control_store is None or self.binding_store is None:
            raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID")
        try:
            serving_snapshot = self.serving_control_store.acquire(
                request.model_version_id
            )
        except (LookupError, OSError, ValueError) as error:
            raise GfmProxyError(503, "GFM_CORE_SERVING_CONTROL_INVALID") from error
        try:
            _validate_capability_snapshot(capabilities, serving_snapshot)
            local_model = serving_snapshot.model(request.model_version_id)
            create_request = CoreInternalCreateRunRequest.model_validate(
                {
                    "schemaVersion": "socialgraph-fm.core-internal-create-run/2.1",
                    "request": request.model_dump(mode="json", by_alias=True),
                    "graphReference": graph.model_dump(mode="json", by_alias=True),
                    "expectedServingControl": {
                        "controlHash": serving_snapshot.control.control_hash,
                        "controlGeneration": serving_snapshot.control.generation,
                        "registryHash": serving_snapshot.registry_hash,
                        "registryGeneration": serving_snapshot.registry.generation,
                        "catalogHash": serving_snapshot.catalog_hash,
                        "catalogGeneration": serving_snapshot.catalog.generation,
                        "modelVersionHash": local_model.model_version_hash,
                    },
                }
            )
            expectation = _expectation_from_snapshot(serving_snapshot, create_request)
        except (LookupError, ValueError) as error:
            raise GfmProxyError(503, "GFM_CORE_SERVING_CONTROL_STALE") from error
        envelope = create_request.model_dump(mode="json", by_alias=True)
        payload = await self.client.create_core_run(envelope)
        try:
            receipt = CoreInternalCreateRunReceipt.model_validate_json(
                json.dumps(payload, ensure_ascii=False)
            )
        except ValueError as error:
            raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID") from error
        binding_payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-api-run-binding/2.2",
            "runId": receipt.status.run_id,
            "receipt": receipt.model_dump(mode="json", by_alias=True),
            "expectation": expectation.model_dump(mode="json", by_alias=True),
        }
        binding_payload["bindingHash"] = canonical_sha256(binding_payload)
        try:
            binding = CoreRunBinding.model_validate_json(
                json.dumps(binding_payload, ensure_ascii=False)
            )
        except ValueError as error:
            raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID") from error
        try:
            self.binding_store.save(binding)
        except (OSError, GfmProxyError) as error:
            self._orphaned_create_count += 1
            raise GfmProxyError(502, "GFM_CORE_CREATE_RECEIPT_PERSIST_FAILED") from error
        return receipt.status

    def diagnostics(self) -> dict[str, int | str]:
        return {
            "code": "GFM_CORE_INTERNAL_ORPHANED_CREATE_COUNT",
            "count": self._orphaned_create_count,
        }

    async def get_run(self, run_id: str) -> CoreRunStatus:
        self._validate_run_id(run_id)
        if self.client is None:
            raise GfmProxyError(404, "GFM_CORE_RUN_NOT_FOUND")
        if self.binding_store is None:
            raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID")
        binding = self.binding_store.get(run_id)
        payload = await self.client.get_core_run(run_id)
        try:
            status = CoreRunStatus.model_validate_json(json.dumps(payload, ensure_ascii=False))
        except ValueError as error:
            raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID") from error
        receipt_status = binding.receipt.status
        if (
            status.run_id != run_id
            or status.request_hash != receipt_status.request_hash
            or status.created_at != receipt_status.created_at
        ):
            raise GfmProxyError(502, "GFM_CORE_RUN_BINDING_INVALID")
        return status

    async def get_result(self, run_id: str) -> CoreRunResult:
        self._validate_run_id(run_id)
        if self.client is None:
            raise GfmProxyError(404, "GFM_CORE_RUN_NOT_FOUND")
        if self.binding_store is None:
            raise GfmProxyError(502, "GFM_CORE_RESULT_BINDING_INVALID")
        binding = self.binding_store.get(run_id)
        payload = await self.client.get_core_result(run_id)
        try:
            result = CoreRunResult.model_validate_json(json.dumps(payload, ensure_ascii=False))
        except ValueError as error:
            raise GfmProxyError(502, "GFM_CORE_RESULT_BINDING_INVALID") from error
        snapshot = binding.receipt.execution_snapshot
        if (
            result.run_id != run_id
            or result.request_hash != snapshot.request_hash
            or result.task_id != snapshot.task_id
            or result.graph_version_id != snapshot.graph_version_id
            or result.graph_version_hash != snapshot.graph_version_hash
            or result.model_version_id != snapshot.model_version_id
            or result.model_version_hash != snapshot.model_version_hash
            or not _result_findings_match_snapshot(result, snapshot)
        ):
            raise GfmProxyError(502, "GFM_CORE_RESULT_BINDING_INVALID")
        return result

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        try:
            parsed = uuid.UUID(run_id)
        except ValueError as error:
            raise GfmProxyError(404, "GFM_CORE_RUN_NOT_FOUND") from error
        if str(parsed) != run_id:
            raise GfmProxyError(404, "GFM_CORE_RUN_NOT_FOUND")


__all__ = [
    "GfmClientProtocol",
    "CoreGateway",
    "CoreGraphResolver",
    "GfmProxyError",
    "CoreRunBindingStore",
    "GfmServiceClient",
]
