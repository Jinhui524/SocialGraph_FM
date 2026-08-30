"""Hash-bound API gateway for SocialGraph-FM Governance online inference."""

from __future__ import annotations

import html
import hashlib
from datetime import UTC, datetime
from typing import Any, NoReturn, Protocol, TypeVar, cast

from pydantic import BaseModel, TypeAdapter, ValidationError

from .gfm_client import GfmProxyError
from .gfm_hashing import canonical_sha256
from .gfm_governance_artifacts import GovernanceArtifactInbox
from .gfm_governance_schemas import (
    GOVERNANCE_INPUT_SCHEMA_VERSION,
    GOVERNANCE_MODALITIES,
    GOVERNANCE_PUBLIC_SKILLS,
    GOVERNANCE_CHANNEL,
    GOVERNANCE_SCHEMA_VERSION,
    CaseCreateRequest,
    CaseItemRequest,
    CaseList,
    CaseTransitionRequest,
    ConcludedReviewLabelSource,
    DerivationPage,
    FindingsPage,
    GovernanceCase,
    GovernanceArtifact,
    GovernanceArtifactList,
    GovernanceArtifactReceipt,
    GovernanceGraphPreview,
    GovernancePreviewQuery,
    GovernanceCapabilities,
    GovernanceHealth,
    ImportedSidecarLabelSource,
    NodeEvidenceV2,
    OnlineRunRequest,
    OnlineRunResult,
    OnlineRunStatus,
    ReviewEventRequest,
    AdaptationComparisonPage,
    TargetLabelSet,
    TargetLabelSetCreateRequest,
    TargetReviewPolicy,
    TargetTaskRegistration,
    AdaptationLabelSetCreateRequestV2,
    ImportedSidecarLabelSetCreateRequestV2,
    ConcludedReviewLabelSetCreateRequestV2,
    TargetLabelSetV2,
    TargetReviewPolicyFitRequest,
    TargetReviewPolicyV2,
    AdaptationComparisonV2,
    ReviewCollection,
    ReviewCollectionCreateRequest,
    AdaptationHandoffCreateRequest,
    AdaptationGovernanceHandoff,
    AdaptationOverlayActivation,
    AdaptationOverlayActivationRequest,
    RunComparison,
    RunList,
)
from .gfm_governance_store import GovernanceStore
from .gfm_governance_target_tasks import InspectedTargetTask, inspect_target_task_bundle

_Model = TypeVar("_Model", bound=BaseModel)
_V2_LABEL_REQUEST_ADAPTER: TypeAdapter[AdaptationLabelSetCreateRequestV2] = TypeAdapter(
    AdaptationLabelSetCreateRequestV2
)
_STALE_PROVENANCE_CODE = "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH"
_PUBLISHABLE_ADAPTATION_LAMBDAS = frozenset({0.25, 0.5, 1.0})
_STALE_CONTEXT_CODES = frozenset(
    {
        "GOVERNANCE_ARTIFACT_NOT_FOUND",
        "GOVERNANCE_ARTIFACT_RECEIPT_INVALID",
        "GFM_GOVERNANCE_RUN_BINDING_MISMATCH",
        "GFM_GOVERNANCE_RESULT_BINDING_MISMATCH",
        "GFM_GOVERNANCE_RUN_NOT_FOUND",
        "GFM_GOVERNANCE_NOT_FOUND",
        "GFM_GOVERNANCE_CONFLICT",
    }
)


def _raise_stale_provenance(error: GfmProxyError) -> NoReturn:
    if error.status_code in {404, 409} or error.code in _STALE_CONTEXT_CODES:
        raise GfmProxyError(409, _STALE_PROVENANCE_CODE) from error
    raise error


def _require_publishable_v2_policy(policy: TargetReviewPolicyV2) -> None:
    if (
        policy.status != "ready"
        or policy.selected_lambda not in _PUBLISHABLE_ADAPTATION_LAMBDAS
    ):
        raise GfmProxyError(409, "GOVERNANCE_ADAPTATION_POLICY_NOT_READY")


class GovernanceClientProtocol(Protocol):
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


def _parse(model_type: type[_Model], payload: dict[str, Any], code: str) -> _Model:
    try:
        return model_type.model_validate(payload)
    except (TypeError, ValueError, ValidationError) as error:
        raise GfmProxyError(502, code) from error


def _unavailable_capabilities() -> GovernanceCapabilities:
    payload: dict[str, Any] = {
        "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
        "channel": GOVERNANCE_CHANNEL,
        "taskId": "coordination_risk",
        "servingReady": False,
        "onlineForwardReady": False,
        "unavailableReason": "GFM_GOVERNANCE_MODEL_NOT_INSTALLED",
        "modelVersionId": None,
        "modelVersionHash": None,
        "modelStateHash": None,
        "supportedProtocols": ["global"],
        "skills": list(GOVERNANCE_PUBLIC_SKILLS),
        "inputSchemaVersion": GOVERNANCE_INPUT_SCHEMA_VERSION,
        "modalities": list(GOVERNANCE_MODALITIES),
        "sampleArtifactId": None,
        "limits": {
            "maxNodes": 10_000,
            "maxRelationRows": 500_000,
            "maxEvidenceNodes": 300,
            "maxEvidenceEdges": 1_000,
            "maxPreviewNodes": 3_000,
            "maxPreviewEdges": 12_000,
        },
    }
    payload["capabilityHash"] = canonical_sha256(payload)
    return GovernanceCapabilities.model_validate(payload)


def _unavailable_health() -> GovernanceHealth:
    runtime_recipe_hash = canonical_sha256(
        {"service": "socialgraph-fm-gfm/governance", "state": "unavailable"}
    )
    payload: dict[str, Any] = {
        "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
        "serviceIdentity": canonical_sha256(
            {"service": "socialgraph-fm-gfm/governance", "modelVersionId": None}
        ),
        "servingReady": False,
        "onlineForwardReady": False,
        "modelVersionId": None,
        "modelVersionHash": None,
        "modelStateHash": None,
        "device": "cpu",
        "dtype": "float32",
        "loadedAt": None,
        "queueDepth": 0,
        "activeRunId": None,
        "runtimeRecipeHash": runtime_recipe_hash,
    }
    payload["healthHash"] = canonical_sha256(payload)
    return GovernanceHealth.model_validate(payload)


class GovernanceGateway:
    def __init__(
        self,
        client: GovernanceClientProtocol | None,
        *,
        inbox: GovernanceArtifactInbox,
        governance: GovernanceStore,
        max_expanded_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.client = client
        self.inbox = inbox
        self.governance = governance
        self.max_expanded_bytes = max_expanded_bytes

    async def capabilities(self) -> GovernanceCapabilities:
        if self.client is None:
            return _unavailable_capabilities()
        return _parse(
            GovernanceCapabilities,
            await self.client.governance_capabilities(),
            "GFM_GOVERNANCE_CAPABILITIES_INVALID",
        )

    async def health(self) -> GovernanceHealth:
        if self.client is None:
            return _unavailable_health()
        health = _parse(
            GovernanceHealth,
            await self.client.governance_health(),
            "GFM_GOVERNANCE_HEALTH_INVALID",
        )
        capabilities = await self.capabilities()
        if (
            health.serving_ready,
            health.online_forward_ready,
            health.model_version_id,
            health.model_version_hash,
            health.model_state_hash,
        ) != (
            capabilities.serving_ready,
            capabilities.online_forward_ready,
            capabilities.model_version_id,
            capabilities.model_version_hash,
            capabilities.model_state_hash,
        ):
            raise GfmProxyError(503, "GFM_GOVERNANCE_HEALTH_STALE")
        return health

    def list_artifacts(self, *, offset: int, limit: int) -> GovernanceArtifactList:
        return self.inbox.list(offset=offset, limit=limit)

    def artifact(self, artifact_id: str) -> GovernanceArtifactReceipt:
        return self.inbox.get(artifact_id)

    async def materialize(self, artifact_id: str) -> GovernanceArtifact:
        receipt = self.inbox.get(artifact_id)
        if self.client is None:
            raise GfmProxyError(503, "GFM_GOVERNANCE_SERVICE_UNAVAILABLE")
        request = {
            "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
            "artifactId": receipt.artifact_id,
            "datasetContentHash": receipt.dataset_content_hash,
            "graphVersionHash": receipt.graph_version_hash,
            "artifactHash": receipt.artifact_hash,
        }
        artifact = _parse(
            GovernanceArtifact,
            await self.client.validate_governance_artifact(artifact_id, request),
            "GFM_GOVERNANCE_ARTIFACT_INVALID",
        )
        if (
            artifact.artifact_id,
            artifact.dataset_content_hash,
            artifact.graph_version_hash,
            artifact.node_count,
            artifact.relation_row_count,
            artifact.self_loops_removed,
            artifact.modalities,
        ) != (
            receipt.artifact_id,
            receipt.dataset_content_hash,
            receipt.graph_version_hash,
            receipt.node_count,
            receipt.relation_row_count,
            receipt.self_loops_removed,
            receipt.modalities,
        ):
            raise GfmProxyError(502, "GFM_GOVERNANCE_ARTIFACT_BINDING_MISMATCH")
        return artifact

    async def register_target_task(
        self, payload: bytes, *, max_expanded_bytes: int
    ) -> TargetTaskRegistration:
        inspected = inspect_target_task_bundle(
            payload, max_expanded_bytes=max_expanded_bytes
        )
        outer_sha256 = hashlib.sha256(payload).hexdigest()
        registration_id = f"target-task-{outer_sha256[:32]}"
        try:
            return self.target_task_registration(
                registration_id, max_expanded_bytes=max_expanded_bytes
            )
        except GfmProxyError as error:
            if error.status_code != 404:
                raise
        receipt = self.inbox.commit(
            inspected.inference,
            clean_self_loops=False,
            max_expanded_bytes=max_expanded_bytes,
        )
        await self.materialize(receipt.artifact_id)
        logical: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.governance-target-task-registration/1.0",
            "registrationId": registration_id,
            "outerBundleSha256": outer_sha256,
            "task": inspected.task.model_dump(mode="json", by_alias=True),
            "targetReceipt": inspected.receipt.model_dump(mode="json", by_alias=True),
            "labels": (
                inspected.labels.model_dump(mode="json", by_alias=True)
                if inspected.labels is not None
                else None
            ),
            "labelReceipt": (
                inspected.label_receipt.model_dump(mode="json", by_alias=True)
                if inspected.label_receipt is not None
                else None
            ),
            "artifact": receipt.model_dump(mode="json", by_alias=True),
            "createdAt": datetime.now(UTC).isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            ),
        }
        logical["registrationHash"] = canonical_sha256(logical)
        registration = TargetTaskRegistration.model_validate(logical)
        self.governance.put_target_task_registration(registration, payload)
        return registration

    def target_task_registration(
        self, registration_id: str, *, max_expanded_bytes: int
    ) -> TargetTaskRegistration:
        registration, _ = self._live_target_task(
            registration_id, max_expanded_bytes=max_expanded_bytes
        )
        return registration

    def _live_target_task(
        self, registration_id: str, *, max_expanded_bytes: int
    ) -> tuple[TargetTaskRegistration, InspectedTargetTask]:
        registration = self.governance.get_target_task_registration(registration_id)
        bundle_path = self.governance.target_task_bundle_path(registration_id)
        try:
            if bundle_path.is_symlink() or not bundle_path.is_file():
                raise OSError("target task bundle missing")
            payload = bundle_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != registration.outer_bundle_sha256:
                raise ValueError("outer target task digest mismatch")
            inspected = inspect_target_task_bundle(
                payload, max_expanded_bytes=max_expanded_bytes
            )
            if (
                inspected.task != registration.task
                or inspected.receipt != registration.target_receipt
                or inspected.labels != registration.labels
                or inspected.label_receipt != registration.label_receipt
            ):
                raise ValueError("target task registration is stale")
            receipt = self.inbox.get(registration.artifact.artifact_id)
            inner_path = (
                self.inbox.incoming_root
                / receipt.artifact_id
                / "bundle.zip"
            )
            if (
                receipt != registration.artifact
                or inner_path.is_symlink()
                or not inner_path.is_file()
                or hashlib.sha256(inner_path.read_bytes()).hexdigest()
                != inspected.task.inference.sha256
            ):
                raise ValueError("registered inference artifact is stale")
        except GfmProxyError as error:
            _raise_stale_provenance(error)
        except (OSError, ValueError) as error:
            raise GfmProxyError(
                409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH"
            ) from error
        return registration, inspected

    @staticmethod
    def _preview_query(projection: GovernancePreviewQuery) -> dict[str, Any]:
        query = projection.model_dump(mode="json", by_alias=True, exclude_none=True)
        anchors = query.pop("anchorNodeIds", [])
        if anchors:
            query["anchorNodeId"] = anchors
        return query

    async def preview(
        self, artifact_id: str, projection: GovernancePreviewQuery | None = None
    ) -> GovernanceGraphPreview:
        receipt = self.inbox.get(artifact_id)
        if self.client is None:
            raise GfmProxyError(503, "GFM_GOVERNANCE_SERVICE_UNAVAILABLE")
        raw = (
            await self.client.get_governance_preview(artifact_id)
            if projection is None
            else await self.client.get_governance_preview(
                artifact_id, self._preview_query(projection)
            )
        )
        preview = _parse(
            GovernanceGraphPreview,
            raw,
            "GFM_GOVERNANCE_PREVIEW_INVALID",
        )
        if (
            preview.artifact_id,
            preview.dataset_content_hash,
            preview.graph_version_hash,
        ) != (
            receipt.artifact_id,
            receipt.dataset_content_hash,
            receipt.graph_version_hash,
        ):
            raise GfmProxyError(502, "GFM_GOVERNANCE_PREVIEW_BINDING_MISMATCH")
        return preview

    async def create_run(self, request: OnlineRunRequest) -> OnlineRunStatus:
        receipt = self.inbox.get(request.artifact_id)
        if (request.dataset_content_hash, request.graph_version_hash) != (
            receipt.dataset_content_hash,
            receipt.graph_version_hash,
        ):
            raise GfmProxyError(409, "GFM_GOVERNANCE_ARTIFACT_HASH_MISMATCH")
        capabilities = await self.capabilities()
        if not capabilities.online_forward_ready or self.client is None:
            raise GfmProxyError(503, "GFM_GOVERNANCE_MODEL_NOT_INSTALLED")
        if (
            request.model_version_id != capabilities.model_version_id
            or request.model_state_hash != capabilities.model_state_hash
        ):
            raise GfmProxyError(409, "GFM_GOVERNANCE_MODEL_MISMATCH")
        status = _parse(
            OnlineRunStatus,
            await self.client.create_governance_run(
                request.model_dump(mode="json", by_alias=True)
            ),
            "GFM_GOVERNANCE_RUN_INVALID",
        )
        if (
            status.request_hash,
            status.artifact_id,
            status.dataset_content_hash,
            status.graph_version_hash,
            status.model_version_id,
            status.model_version_hash,
            status.model_state_hash,
        ) != (
            request.request_hash,
            request.artifact_id,
            request.dataset_content_hash,
            request.graph_version_hash,
            request.model_version_id,
            capabilities.model_version_hash,
            request.model_state_hash,
        ):
            raise GfmProxyError(502, "GFM_GOVERNANCE_RUN_BINDING_MISMATCH")
        self.governance.put_run_binding(
            {
                "runId": status.run_id,
                "requestHash": status.request_hash,
                "artifactId": status.artifact_id,
                "datasetContentHash": status.dataset_content_hash,
                "graphVersionHash": status.graph_version_hash,
                "modelVersionId": status.model_version_id,
                "modelVersionHash": status.model_version_hash,
                "modelStateHash": status.model_state_hash,
                "createdAt": status.model_dump(mode="json", by_alias=True)["createdAt"],
            }
        )
        return status

    def _validate_status_binding(self, status: OnlineRunStatus) -> None:
        binding = self.governance.get_run_binding(status.run_id)
        observed = (
            status.run_id,
            status.request_hash,
            status.artifact_id,
            status.dataset_content_hash,
            status.graph_version_hash,
            status.model_version_id,
            status.model_version_hash,
            status.model_state_hash,
        )
        expected = tuple(
            binding[key]
            for key in (
                "runId",
                "requestHash",
                "artifactId",
                "datasetContentHash",
                "graphVersionHash",
                "modelVersionId",
                "modelVersionHash",
                "modelStateHash",
            )
        )
        if observed != expected:
            raise GfmProxyError(502, "GFM_GOVERNANCE_RUN_BINDING_MISMATCH")

    async def get_run(self, run_id: str) -> OnlineRunStatus:
        self.governance.get_run_binding(run_id)
        if self.client is None:
            raise GfmProxyError(503, "GFM_GOVERNANCE_SERVICE_UNAVAILABLE")
        status = _parse(
            OnlineRunStatus,
            await self.client.get_governance_run(run_id),
            "GFM_GOVERNANCE_RUN_INVALID",
        )
        self._validate_status_binding(status)
        return status

    async def list_runs(self, *, offset: int, limit: int) -> RunList:
        if self.client is None:
            return RunList.model_validate(
                {
                    "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                    "items": [],
                    "total": 0,
                    "offset": offset,
                    "limit": limit,
                }
            )
        internal = _parse(
            RunList,
            await self.client.list_governance_runs(0, 10_000),
            "GFM_GOVERNANCE_RUN_LIST_INVALID",
        )
        if internal.total != len(internal.items):
            raise GfmProxyError(502, "GFM_GOVERNANCE_RUN_LIST_TRUNCATED")
        visible: list[OnlineRunStatus] = []
        for status in internal.items:
            try:
                self._validate_status_binding(status)
            except GfmProxyError as error:
                if error.code in {
                    "GOVERNANCE_RUN_NOT_FOUND",
                    "GOVERNANCE_RUN_BINDING_MODEL_STATE_MISSING",
                }:
                    continue
                raise
            visible.append(status)
        return RunList.model_validate(
            {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "items": visible[offset : offset + limit],
                "total": len(visible),
                "offset": offset,
                "limit": limit,
            }
        )

    async def cancel(self, run_id: str) -> OnlineRunStatus:
        await self.get_run(run_id)
        assert self.client is not None
        status = _parse(
            OnlineRunStatus,
            await self.client.cancel_governance_run(run_id),
            "GFM_GOVERNANCE_RUN_INVALID",
        )
        self._validate_status_binding(status)
        return status

    async def retry(self, run_id: str) -> OnlineRunStatus:
        previous = await self.get_run(run_id)
        assert self.client is not None
        status = _parse(
            OnlineRunStatus,
            await self.client.retry_governance_run(run_id),
            "GFM_GOVERNANCE_RUN_INVALID",
        )
        if status.run_id == previous.run_id or (
            status.request_hash,
            status.artifact_id,
            status.dataset_content_hash,
            status.graph_version_hash,
            status.model_version_id,
            status.model_version_hash,
            status.model_state_hash,
        ) != (
            previous.request_hash,
            previous.artifact_id,
            previous.dataset_content_hash,
            previous.graph_version_hash,
            previous.model_version_id,
            previous.model_version_hash,
            previous.model_state_hash,
        ):
            raise GfmProxyError(502, "GFM_GOVERNANCE_RETRY_BINDING_MISMATCH")
        self.governance.put_run_binding(
            {
                "runId": status.run_id,
                "requestHash": status.request_hash,
                "artifactId": status.artifact_id,
                "datasetContentHash": status.dataset_content_hash,
                "graphVersionHash": status.graph_version_hash,
                "modelVersionId": status.model_version_id,
                "modelVersionHash": status.model_version_hash,
                "modelStateHash": status.model_state_hash,
                "createdAt": status.model_dump(mode="json", by_alias=True)["createdAt"],
            }
        )
        return status

    async def result(self, run_id: str) -> OnlineRunResult:
        status = await self.get_run(run_id)
        if status.status != "succeeded":
            raise GfmProxyError(409, "GFM_GOVERNANCE_RUN_NOT_SUCCEEDED")
        assert self.client is not None
        result = _parse(
            OnlineRunResult,
            await self.client.get_governance_result(run_id),
            "GFM_GOVERNANCE_RESULT_INVALID",
        )
        if (
            result.run_id,
            result.request_hash,
            result.artifact_id,
            result.dataset_content_hash,
            result.graph_version_hash,
            result.model_version_id,
            result.model_version_hash,
            result.model_state_hash,
        ) != (
            status.run_id,
            status.request_hash,
            status.artifact_id,
            status.dataset_content_hash,
            status.graph_version_hash,
            status.model_version_id,
            status.model_version_hash,
            status.model_state_hash,
        ):
            raise GfmProxyError(502, "GFM_GOVERNANCE_RESULT_BINDING_MISMATCH")
        return result

    async def _current_adaptation_context(
        self, run_id: str, result_hash: str
    ) -> tuple[
        OnlineRunStatus,
        OnlineRunResult,
        GovernanceArtifactReceipt,
        GovernanceGraphPreview | None,
    ]:
        try:
            status = await self.get_run(run_id)
            if status.status != "succeeded":
                raise GfmProxyError(409, "GFM_GOVERNANCE_RUN_NOT_SUCCEEDED")
            result = await self.result(run_id)
            receipt = self.inbox.get(status.artifact_id)
        except GfmProxyError as error:
            _raise_stale_provenance(error)
        if (
            result.result_hash != result_hash
            or (receipt.artifact_id, receipt.dataset_content_hash, receipt.graph_version_hash)
            != (status.artifact_id, status.dataset_content_hash, status.graph_version_hash)
        ):
            raise GfmProxyError(409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH")
        preview: GovernanceGraphPreview | None = None
        if receipt.node_count <= 3_000:
            try:
                preview = await self.run_preview(
                    run_id,
                    GovernancePreviewQuery(
                        preset="overview",
                        nodeBudget=receipt.node_count,
                        edgeBudget=12_000,
                    ),
                )
            except GfmProxyError as error:
                _raise_stale_provenance(error)
            if (
                preview.run_id != run_id
                or preview.result_hash != result_hash
                or preview.artifact_id != receipt.artifact_id
                or preview.dataset_content_hash != receipt.dataset_content_hash
                or preview.graph_version_hash != receipt.graph_version_hash
                or preview.node_count != receipt.node_count
                or len(preview.nodes) != receipt.node_count
                or len({node.id for node in preview.nodes}) != receipt.node_count
            ):
                raise GfmProxyError(
                    409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH"
                )
        return status, result, receipt, preview

    @staticmethod
    def _adaptation_binding_matches_live(
        binding: Any, status: OnlineRunStatus, result: OnlineRunResult
    ) -> bool:
        return (
            binding.run_id,
            binding.request_hash,
            binding.result_hash,
            binding.artifact_id,
            binding.dataset_content_hash,
            binding.graph_version_hash,
            binding.model_version_id,
            binding.model_version_hash,
            binding.model_state_hash,
        ) == (
            status.run_id,
            status.request_hash,
            result.result_hash,
            status.artifact_id,
            status.dataset_content_hash,
            status.graph_version_hash,
            status.model_version_id,
            status.model_version_hash,
            status.model_state_hash,
        )

    @staticmethod
    def _preview_degree_strata(
        preview: GovernanceGraphPreview | None,
    ) -> tuple[dict[str, int], dict[str, int]]:
        if preview is None or preview.node_count != 128:
            raise GfmProxyError(409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH")
        degrees = {node.id: node.degree for node in preview.nodes}
        ordered = sorted(preview.nodes, key=lambda node: (node.degree, node.id))
        strata = {
            node.id: min(3, position * 4 // len(ordered))
            for position, node in enumerate(ordered)
        }
        return degrees, strata

    def _validate_sidecar_sources(
        self,
        request: TargetLabelSetCreateRequest,
        receipt: GovernanceArtifactReceipt,
        preview: GovernanceGraphPreview | None,
    ) -> None:
        imported = tuple(
            source
            for source in request.sources
            if isinstance(source, ImportedSidecarLabelSource)
        )
        if not imported:
            return
        sidecar = request.sidecar_receipt
        if (
            sidecar is None
            or sidecar.bundle_sha256 != receipt.bundle_sha256
            or sidecar.dataset_id != receipt.dataset_id
            or sidecar.coverage.get("nodeCount") != receipt.node_count
        ):
            raise GfmProxyError(409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH")
        degrees, strata = self._preview_degree_strata(preview)
        if any(
            degrees.get(source.node_id) != source.fused_degree
            or strata.get(source.node_id) != source.structural_stratum
            for source in imported
        ):
            raise GfmProxyError(409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH")

    def _validate_label_set_sidecar(
        self,
        label_set: TargetLabelSet,
        receipt: GovernanceArtifactReceipt,
        preview: GovernanceGraphPreview | None,
    ) -> None:
        imported = tuple(
            label
            for label in label_set.labels
            if label.source_type == "imported_sidecar"
        )
        if not imported:
            return
        sidecar = label_set.sidecar_receipt
        if sidecar is None or sidecar.bundle_sha256 != receipt.bundle_sha256:
            raise GfmProxyError(409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH")
        degrees, strata = self._preview_degree_strata(preview)
        if any(
            degrees.get(label.node_id) != label.fused_degree
            or strata.get(label.node_id) != label.structural_stratum
            for label in imported
        ):
            raise GfmProxyError(409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH")

    async def create_adaptation_label_set(
        self, request: TargetLabelSetCreateRequest | AdaptationLabelSetCreateRequestV2
    ) -> TargetLabelSet | TargetLabelSetV2:
        if isinstance(
            request,
            (ImportedSidecarLabelSetCreateRequestV2, ConcludedReviewLabelSetCreateRequestV2),
        ):
            return await self._create_v2_adaptation_label_set(request)
        if self.client is None:
            raise GfmProxyError(503, "GFM_GOVERNANCE_MODEL_NOT_INSTALLED")
        status, result, receipt, preview = await self._current_adaptation_context(
            request.run_id, request.result_hash
        )
        self._validate_sidecar_sources(request, receipt, preview)
        labels: list[dict[str, Any]] = []
        for source in request.sources:
            if isinstance(source, ImportedSidecarLabelSource):
                labels.append(
                    {
                        "nodeId": source.node_id,
                        "label": "positive" if source.cohort == "io" else "negative",
                        "sourceType": source.source_type,
                        "sourceRecordId": source.source_record_id,
                        "sourceRecordHash": source.source_record_hash,
                        "reviewEventHash": None,
                        "structuralStratum": source.structural_stratum,
                        "fusedDegree": source.fused_degree,
                        "labelsSha256": source.labels_sha256,
                        "receiptHash": source.receipt_hash,
                    }
                )
                continue
            if not isinstance(source, ConcludedReviewLabelSource):
                raise GfmProxyError(422, "GOVERNANCE_ADAPTATION_SOURCE_INVALID")
            case = self.governance.get_case(source.case_id)
            if case.run_id != request.run_id:
                raise GfmProxyError(409, "GOVERNANCE_ADAPTATION_REVIEW_RUN_MISMATCH")
            if case.state not in {"concluded", "archived"}:
                raise GfmProxyError(409, "GOVERNANCE_ADAPTATION_REVIEW_NOT_CONCLUDED")
            event = next(
                (
                    candidate
                    for candidate in case.review_events
                    if candidate.event_hash == source.event_hash
                ),
                None,
            )
            if event is None or event.target_type != "node":
                raise GfmProxyError(422, "GOVERNANCE_ADAPTATION_REVIEW_INVALID")
            current = case.current_decisions.get(f"node:{event.target_id}")
            if current != event.decision or current == "pending":
                raise GfmProxyError(422, "GOVERNANCE_ADAPTATION_REVIEW_INELIGIBLE")
            labels.append(
                {
                    "nodeId": event.target_id,
                    "label": "positive" if event.decision == "confirmed" else "negative",
                    "sourceType": "concluded_review",
                    "sourceRecordId": event.event_id,
                    "sourceRecordHash": event.event_hash,
                    "reviewEventHash": event.event_hash,
                }
            )

        by_node: dict[str, str] = {}
        for label in labels:
            node_id, value = str(label["nodeId"]), str(label["label"])
            previous = by_node.get(node_id)
            if previous is not None:
                code = (
                    "GOVERNANCE_ADAPTATION_SOURCE_CONFLICT"
                    if previous != value
                    else "GOVERNANCE_ADAPTATION_SOURCE_DUPLICATE"
                )
                raise GfmProxyError(422, code)
            by_node[node_id] = value
        positive = sum(value == "positive" for value in by_node.values())
        negative = sum(value == "negative" for value in by_node.values())
        if len(by_node) < 8 or positive < 4 or negative < 4:
            raise GfmProxyError(422, "GOVERNANCE_ADAPTATION_LABELS_INSUFFICIENT")
        payload: dict[str, Any] = {
            "schemaVersion": request.schema_version,
            "runId": request.run_id,
            "resultHash": request.result_hash,
            "labels": labels,
        }
        if request.sidecar_receipt is not None:
            payload["sidecarReceipt"] = request.sidecar_receipt.model_dump(
                mode="json", by_alias=True
            )
        label_set = _parse(
            TargetLabelSet,
            await self.client.create_governance_label_set(payload),
            "GFM_GOVERNANCE_ADAPTATION_LABEL_SET_INVALID",
        )
        if (
            label_set.binding.run_id != request.run_id
            or label_set.binding.result_hash != request.result_hash
            or not self._adaptation_binding_matches_live(
                label_set.binding, status, result
            )
            or label_set.sidecar_receipt != request.sidecar_receipt
        ):
            raise GfmProxyError(502, "GFM_GOVERNANCE_ADAPTATION_BINDING_MISMATCH")
        expected_sources = {
            (
                str(label["nodeId"]),
                str(label["label"]),
                str(label["sourceType"]),
                str(label["sourceRecordId"]),
                str(label["sourceRecordHash"]),
                label["reviewEventHash"],
            )
            for label in labels
        }
        actual_sources = {
            (
                label.node_id,
                label.label,
                label.source_type,
                label.source_record_id,
                label.source_record_hash,
                label.review_event_hash,
            )
            for label in label_set.labels
        }
        if actual_sources != expected_sources:
            raise GfmProxyError(502, "GFM_GOVERNANCE_ADAPTATION_SOURCE_MISMATCH")
        self.governance.put_adaptation_label_set(label_set)
        return label_set

    async def _v2_source_context(
        self,
        request: ImportedSidecarLabelSetCreateRequestV2
        | ConcludedReviewLabelSetCreateRequestV2,
    ) -> tuple[
        TargetTaskRegistration,
        InspectedTargetTask,
        OnlineRunStatus,
        OnlineRunResult,
        list[dict[str, Any]],
    ]:
        try:
            registration, inspected = self._live_target_task(
                request.target_task_registration_id,
                max_expanded_bytes=self.max_expanded_bytes,
            )
        except GfmProxyError as error:
            _raise_stale_provenance(error)
        status, result, _, _ = await self._current_adaptation_context(
            request.run_id, request.result_hash
        )
        if (
            status.artifact_id != registration.artifact.artifact_id
            or status.dataset_content_hash != registration.artifact.dataset_content_hash
            or status.graph_version_hash != registration.artifact.graph_version_hash
        ):
            raise GfmProxyError(
                409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH"
            )
        if isinstance(request, ImportedSidecarLabelSetCreateRequestV2):
            if registration.labels is None or registration.label_receipt is None:
                raise GfmProxyError(409, "GOVERNANCE_TARGET_LABELS_NOT_AVAILABLE")
            labels = [
                row.model_dump(mode="json", by_alias=True)
                for row in registration.labels.labels
            ]
        else:
            labels = []
            seen: set[str] = set()
            for reference in request.reviews:
                case = self.governance.get_case(reference.case_id)
                if case.run_id != request.run_id or case.state not in {"concluded", "archived"}:
                    raise GfmProxyError(409, "GOVERNANCE_ADAPTATION_REVIEW_NOT_CONCLUDED")
                event = next(
                    (
                        row
                        for row in case.review_events
                        if row.event_hash == reference.event_hash
                    ),
                    None,
                )
                if (
                    event is None
                    or event.target_type != "node"
                    or event.target_id in seen
                    or event.decision == "pending"
                    or case.current_decisions.get(f"node:{event.target_id}")
                    != event.decision
                    or event.target_id not in inspected.node_degrees
                ):
                    raise GfmProxyError(409, "GOVERNANCE_ADAPTATION_REVIEW_STALE")
                seen.add(event.target_id)
                labels.append(
                    {
                        "nodeId": event.target_id,
                        "label": (
                            "positive" if event.decision == "confirmed" else "negative"
                        ),
                        "structuralStratum": inspected.node_strata[event.target_id],
                        "fusedDegree": inspected.node_degrees[event.target_id],
                    }
                )
        return registration, inspected, status, result, labels

    async def _create_v2_adaptation_label_set(
        self,
        request: ImportedSidecarLabelSetCreateRequestV2
        | ConcludedReviewLabelSetCreateRequestV2,
    ) -> TargetLabelSetV2:
        if self.client is None:
            raise GfmProxyError(503, "GFM_GOVERNANCE_MODEL_NOT_INSTALLED")
        registration, _, _, _, labels = await self._v2_source_context(request)
        payload = {
            "schemaVersion": request.schema_version,
            "taskId": registration.task.task_id,
            "inferenceSha256": registration.task.inference.sha256,
            "targetTaskRegistrationId": registration.registration_id,
            "runId": request.run_id,
            "resultHash": request.result_hash,
            "labels": labels,
        }
        try:
            remote_label_set = await self.client.create_governance_label_set(payload)
        except GfmProxyError as error:
            _raise_stale_provenance(error)
        label_set = _parse(
            TargetLabelSetV2,
            remote_label_set,
            "GFM_GOVERNANCE_ADAPTATION_LABEL_SET_INVALID",
        )
        if (
            label_set.task_id != registration.task.task_id
            or label_set.inference_sha256 != registration.task.inference.sha256
            or [row.model_dump(mode="json", by_alias=True) for row in label_set.labels]
            != labels
        ):
            raise GfmProxyError(502, "GFM_GOVERNANCE_ADAPTATION_SOURCE_MISMATCH")
        self.governance.put_target_label_set(
            label_set,
            run_id=request.run_id,
            source=request.model_dump(mode="json", by_alias=True),
        )
        return label_set

    async def fit_adaptation_policy(
        self,
        label_set_hash: str,
        request: TargetReviewPolicyFitRequest | None = None,
    ) -> TargetReviewPolicy | TargetReviewPolicyV2:
        try:
            v2_label_set, source = self.governance.get_target_label_set(
                label_set_hash,
                target_task_registration_id=(
                    request.target_task_registration_id if request is not None else None
                ),
                run_id=request.run_id if request is not None else None,
                result_hash=request.result_hash if request is not None else None,
            )
        except GfmProxyError as error:
            if error.status_code != 404:
                raise
        else:
            v2_request = _V2_LABEL_REQUEST_ADAPTER.validate_python(source)
            if request is not None and (
                request.target_task_registration_id
                != v2_request.target_task_registration_id
                or request.run_id != v2_request.run_id
                or request.result_hash != v2_request.result_hash
            ):
                raise GfmProxyError(
                    409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH"
                )
            registration, _, status, result, labels = await self._v2_source_context(
                v2_request
            )
            if (
                v2_label_set.task_id != registration.task.task_id
                or v2_label_set.inference_sha256 != registration.task.inference.sha256
                or [row.model_dump(mode="json", by_alias=True) for row in v2_label_set.labels]
                != labels
            ):
                raise GfmProxyError(409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH")
            if self.client is None:
                raise GfmProxyError(503, "GFM_GOVERNANCE_MODEL_NOT_INSTALLED")
            fit_request = TargetReviewPolicyFitRequest(
                schemaVersion="socialgraph-fm.governance-target-review-policy-fit-request/1.0",
                targetTaskRegistrationId=v2_request.target_task_registration_id,
                runId=v2_request.run_id,
                resultHash=v2_request.result_hash,
            )
            try:
                remote_policy = await self.client.fit_governance_policy(
                    label_set_hash,
                    fit_request.model_dump(mode="json", by_alias=True),
                )
            except GfmProxyError as error:
                _raise_stale_provenance(error)
            policy = _parse(
                TargetReviewPolicyV2,
                remote_policy,
                "GFM_GOVERNANCE_ADAPTATION_POLICY_INVALID",
            )
            if (
                policy.label_set_hash != label_set_hash
                or not self._adaptation_binding_matches_live(policy.binding, status, result)
            ):
                raise GfmProxyError(502, "GFM_GOVERNANCE_ADAPTATION_BINDING_MISMATCH")
            self.governance.put_target_policy(policy)
            return policy  # type: ignore[return-value]
        label_set = self.governance.get_adaptation_label_set(label_set_hash)
        if request is not None:
            raise GfmProxyError(409, "GOVERNANCE_ADAPTATION_BINDING_INVALID")
        if self.client is None:
            raise GfmProxyError(503, "GFM_GOVERNANCE_MODEL_NOT_INSTALLED")
        status, result, receipt, preview = await self._current_adaptation_context(
            label_set.binding.run_id, label_set.binding.result_hash
        )
        if not self._adaptation_binding_matches_live(label_set.binding, status, result):
            raise GfmProxyError(409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH")
        self._validate_label_set_sidecar(label_set, receipt, preview)
        legacy_policy = _parse(
            TargetReviewPolicy,
            await self.client.fit_governance_policy(label_set_hash),
            "GFM_GOVERNANCE_ADAPTATION_POLICY_INVALID",
        )
        if (
            legacy_policy.label_set_hash != label_set_hash
            or legacy_policy.binding != label_set.binding
        ):
            raise GfmProxyError(502, "GFM_GOVERNANCE_ADAPTATION_BINDING_MISMATCH")
        self.governance.put_adaptation_policy(legacy_policy)
        return legacy_policy

    async def adaptation_policy(
        self, policy_hash: str
    ) -> TargetReviewPolicy | TargetReviewPolicyV2:
        try:
            stored_v2 = self.governance.get_target_policy(policy_hash)
        except GfmProxyError as error:
            if error.status_code != 404:
                raise
        else:
            label_set, source = self.governance.get_target_label_set(
                stored_v2.label_set_hash,
                run_id=stored_v2.binding.run_id,
                result_hash=stored_v2.binding.result_hash,
            )
            v2_request = _V2_LABEL_REQUEST_ADAPTER.validate_python(source)
            registration, _, status, result, labels = await self._v2_source_context(
                v2_request
            )
            if (
                label_set.task_id != registration.task.task_id
                or [row.model_dump(mode="json", by_alias=True) for row in label_set.labels]
                != labels
                or not self._adaptation_binding_matches_live(stored_v2.binding, status, result)
            ):
                raise GfmProxyError(409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH")
            if self.client is None:
                raise GfmProxyError(503, "GFM_GOVERNANCE_MODEL_NOT_INSTALLED")
            try:
                remote_payload = await self.client.get_governance_policy(policy_hash)
            except GfmProxyError as error:
                _raise_stale_provenance(error)
            remote_v2 = _parse(
                TargetReviewPolicyV2,
                remote_payload,
                "GFM_GOVERNANCE_ADAPTATION_POLICY_INVALID",
            )
            if remote_v2 != stored_v2:
                raise GfmProxyError(409, "GFM_GOVERNANCE_ADAPTATION_POLICY_STALE")
            return remote_v2
        stored = self.governance.get_adaptation_policy(policy_hash)
        if self.client is None:
            return stored
        remote = _parse(
            TargetReviewPolicy,
            await self.client.get_governance_policy(policy_hash),
            "GFM_GOVERNANCE_ADAPTATION_POLICY_INVALID",
        )
        if remote != stored:
            raise GfmProxyError(502, "GFM_GOVERNANCE_ADAPTATION_POLICY_STALE")
        return remote

    async def adaptation_comparison(
        self,
        run_id: str,
        policy_hash: str,
        *,
        offset: int,
        limit: int,
    ) -> AdaptationComparisonPage | AdaptationComparisonV2:
        policy = await self.adaptation_policy(policy_hash)
        if isinstance(policy, TargetReviewPolicyV2):
            _require_publishable_v2_policy(policy)
        if policy.binding.run_id != run_id:
            raise GfmProxyError(409, "GFM_GOVERNANCE_ADAPTATION_RUN_MISMATCH")
        if self.client is None:
            raise GfmProxyError(503, "GFM_GOVERNANCE_MODEL_NOT_INSTALLED")
        try:
            raw_page = await self.client.get_governance_adaptation_comparison(
                run_id, policy_hash, offset, limit
            )
        except GfmProxyError as error:
            if isinstance(policy, TargetReviewPolicyV2):
                _raise_stale_provenance(error)
            raise
        page: AdaptationComparisonPage | AdaptationComparisonV2
        if isinstance(policy, TargetReviewPolicyV2):
            page = _parse(
                AdaptationComparisonV2,
                raw_page,
                "GFM_GOVERNANCE_ADAPTATION_COMPARISON_INVALID",
            )
        else:
            page = _parse(
                AdaptationComparisonPage,
                raw_page,
                "GFM_GOVERNANCE_ADAPTATION_COMPARISON_INVALID",
            )
        if page.binding != policy.binding or page.policy_hash != policy_hash:
            raise GfmProxyError(502, "GFM_GOVERNANCE_ADAPTATION_BINDING_MISMATCH")
        if isinstance(page, AdaptationComparisonV2):
            self.governance.put_target_adaptation_metadata(
                kind="comparison",
                record_hash=page.comparison_hash,
                run_id=run_id,
                payload=page.model_dump(mode="json", by_alias=True),
            )
        return page

    async def create_review_collection(
        self, request: ReviewCollectionCreateRequest
    ) -> ReviewCollection:
        registration, inspected = self._live_target_task(
            request.target_task_registration_id,
            max_expanded_bytes=self.max_expanded_bytes,
        )
        status, _, _, _ = await self._current_adaptation_context(
            request.run_id, request.result_hash
        )
        if status.artifact_id != registration.artifact.artifact_id or any(
            item.target_type == "node" and item.target_id not in inspected.node_degrees
            for item in request.items
        ):
            raise GfmProxyError(409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH")
        return self.governance.create_review_collection(request)

    async def create_adaptation_handoff(
        self, request: AdaptationHandoffCreateRequest
    ) -> AdaptationGovernanceHandoff:
        policy = await self.adaptation_policy(request.policy_hash)
        if not isinstance(policy, TargetReviewPolicyV2):
            raise GfmProxyError(409, "GOVERNANCE_HANDOFF_REQUIRES_V2_POLICY")
        _require_publishable_v2_policy(policy)
        comparison = await self.adaptation_comparison(
            policy.binding.run_id, policy.policy_hash, offset=0, limit=500
        )
        if not isinstance(comparison, AdaptationComparisonV2):
            raise GfmProxyError(409, "GOVERNANCE_HANDOFF_REQUIRES_V2_POLICY")
        label_set, source = self.governance.get_target_label_set(
            policy.label_set_hash,
            target_task_registration_id=request.target_task_registration_id,
            run_id=policy.binding.run_id,
            result_hash=policy.binding.result_hash,
        )
        source_request = _V2_LABEL_REQUEST_ADAPTER.validate_python(source)
        if source_request.target_task_registration_id != request.target_task_registration_id:
            raise GfmProxyError(409, "GOVERNANCE_HANDOFF_TARGET_MISMATCH")
        registration, _ = self._live_target_task(
            request.target_task_registration_id,
            max_expanded_bytes=self.max_expanded_bytes,
        )
        payload: dict[str, Any] = {
            "schemaVersion": request.schema_version,
            "targetTaskRegistrationId": registration.registration_id,
            "targetReceiptHash": registration.target_receipt.receipt_hash,
            "labelSetHash": label_set.label_set_hash,
            "binding": policy.binding.model_dump(mode="json", by_alias=True),
            "policyHash": policy.policy_hash,
            "comparisonHash": comparison.comparison_hash,
            "decision": request.decision,
            "baseModelMutation": False,
        }
        payload["handoffHash"] = canonical_sha256(payload)
        handoff = AdaptationGovernanceHandoff.model_validate(payload)
        self.governance.put_target_adaptation_metadata(
            kind="handoff",
            record_hash=handoff.handoff_hash,
            run_id=policy.binding.run_id,
            payload=payload,
        )
        return handoff

    async def adaptation_handoff(
        self, handoff_hash: str
    ) -> AdaptationGovernanceHandoff:
        stored = AdaptationGovernanceHandoff.model_validate(
            self.governance.get_target_adaptation_metadata(
                kind="handoff", record_hash=handoff_hash
            )
        )
        fresh = await self.create_adaptation_handoff(
            AdaptationHandoffCreateRequest(
                schemaVersion=stored.schema_version,
                targetTaskRegistrationId=stored.target_task_registration_id,
                policyHash=stored.policy_hash,
                decision=stored.decision,
            )
        )
        if fresh != stored:
            raise GfmProxyError(409, "GFM_GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH")
        return stored

    async def activate_adaptation_overlay(
        self,
        policy_hash: str,
        request: AdaptationOverlayActivationRequest,
    ) -> AdaptationOverlayActivation:
        policy = await self.adaptation_policy(policy_hash)
        if not isinstance(policy, TargetReviewPolicyV2):
            raise GfmProxyError(409, "GOVERNANCE_ADAPTATION_POLICY_NOT_READY")
        _require_publishable_v2_policy(policy)
        comparison = await self.adaptation_comparison(
            policy.binding.run_id, policy_hash, offset=0, limit=500
        )
        if not isinstance(comparison, AdaptationComparisonV2):
            raise GfmProxyError(409, "GOVERNANCE_ADAPTATION_POLICY_NOT_READY")
        label_set, source = self.governance.get_target_label_set(
            policy.label_set_hash,
            target_task_registration_id=request.target_task_registration_id,
            run_id=policy.binding.run_id,
            result_hash=policy.binding.result_hash,
        )
        source_request = _V2_LABEL_REQUEST_ADAPTER.validate_python(source)
        if (
            source_request.target_task_registration_id
            != request.target_task_registration_id
        ):
            raise GfmProxyError(409, "GOVERNANCE_ADAPTATION_TARGET_MISMATCH")
        registration, _ = self._live_target_task(
            request.target_task_registration_id,
            max_expanded_bytes=self.max_expanded_bytes,
        )
        payload: dict[str, Any] = {
            "schemaVersion": request.schema_version,
            "targetTaskRegistrationId": registration.registration_id,
            "targetReceiptHash": registration.target_receipt.receipt_hash,
            "labelSetHash": label_set.label_set_hash,
            "binding": policy.binding.model_dump(mode="json", by_alias=True),
            "policyHash": policy.policy_hash,
            "comparisonHash": comparison.comparison_hash,
            "active": True,
            "baseModelMutation": False,
        }
        payload["activationHash"] = canonical_sha256(payload)
        activation = AdaptationOverlayActivation.model_validate(payload)
        self.governance.put_target_adaptation_metadata(
            kind="overlay",
            record_hash=activation.activation_hash,
            run_id=policy.binding.run_id,
            payload=payload,
        )
        return activation

    async def compare_runs(
        self, left_run_id: str, right_run_id: str, *, limit: int
    ) -> RunComparison:
        if left_run_id == right_run_id:
            raise GfmProxyError(400, "GFM_GOVERNANCE_COMPARE_RUNS_IDENTICAL")
        left_result, right_result = await self.result(left_run_id), await self.result(
            right_run_id
        )
        left_identity = (
            left_result.artifact_id,
            left_result.dataset_content_hash,
            left_result.graph_version_hash,
        )
        right_identity = (
            right_result.artifact_id,
            right_result.dataset_content_hash,
            right_result.graph_version_hash,
        )
        if left_identity != right_identity:
            raise GfmProxyError(409, "GFM_GOVERNANCE_COMPARE_ARTIFACT_MISMATCH")
        left_page = await self.findings(left_run_id, offset=0, limit=10_000)
        right_page = await self.findings(right_run_id, offset=0, limit=10_000)
        left_nodes = {item.node_id: item for item in left_page.items}
        right_nodes = {item.node_id: item for item in right_page.items}
        if (
            set(left_nodes) != set(right_nodes)
            or left_page.total != right_page.total
            or len(left_nodes) != left_page.total
            or len(right_nodes) != right_page.total
        ):
            raise GfmProxyError(502, "GFM_GOVERNANCE_COMPARE_NODE_SET_MISMATCH")
        changes = [
            {
                "nodeId": node_id,
                "leftScore": left_nodes[node_id].score,
                "rightScore": right_nodes[node_id].score,
                "scoreDelta": right_nodes[node_id].score - left_nodes[node_id].score,
                "leftRank": left_nodes[node_id].rank,
                "rightRank": right_nodes[node_id].rank,
                "rankDelta": right_nodes[node_id].rank - left_nodes[node_id].rank,
                "riskBandChanged": (
                    right_nodes[node_id].risk_band != left_nodes[node_id].risk_band
                ),
            }
            for node_id in left_nodes
        ]
        changes.sort(
            key=lambda item: (
                -abs(cast(float, item["scoreDelta"])),
                str(item["nodeId"]),
            )
        )
        left_groups = await self.derivations(
            left_run_id, "groups", offset=0, limit=10_000
        )
        right_groups = await self.derivations(
            right_run_id, "groups", offset=0, limit=10_000
        )
        left_group_ids = {item.id for item in left_groups.items}
        right_group_ids = {item.id for item in right_groups.items}
        left_reviews = self.governance.run_review_summary(left_run_id)
        right_reviews = self.governance.run_review_summary(right_run_id)
        payload: dict[str, Any] = {
            "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
            "leftRunId": left_run_id,
            "rightRunId": right_run_id,
            "artifactId": left_result.artifact_id,
            "datasetContentHash": left_result.dataset_content_hash,
            "graphVersionHash": left_result.graph_version_hash,
            "comparedNodes": len(left_nodes),
            "changes": changes[:limit],
            "groupSummary": {
                "leftCount": len(left_group_ids),
                "rightCount": len(right_group_ids),
                "sharedCount": len(left_group_ids & right_group_ids),
                "addedCount": len(right_group_ids - left_group_ids),
                "removedCount": len(left_group_ids - right_group_ids),
            },
            "reviewSummary": {
                "leftCaseCount": left_reviews["caseCount"],
                "rightCaseCount": right_reviews["caseCount"],
                "leftReviewEventCount": left_reviews["reviewEventCount"],
                "rightReviewEventCount": right_reviews["reviewEventCount"],
            },
        }
        payload["comparisonHash"] = canonical_sha256(payload)
        return RunComparison.model_validate(payload)

    async def findings(self, run_id: str, *, offset: int, limit: int) -> FindingsPage:
        await self.result(run_id)
        assert self.client is not None
        page = _parse(
            FindingsPage,
            await self.client.get_governance_findings(run_id, offset, limit),
            "GFM_GOVERNANCE_FINDINGS_INVALID",
        )
        if page.run_id != run_id:
            raise GfmProxyError(502, "GFM_GOVERNANCE_RESULT_BINDING_MISMATCH")
        return page

    async def run_preview(
        self, run_id: str, projection: GovernancePreviewQuery | None = None
    ) -> GovernanceGraphPreview:
        result = await self.result(run_id)
        assert self.client is not None
        raw = (
            await self.client.get_governance_run_preview(run_id)
            if projection is None
            else await self.client.get_governance_run_preview(
                run_id, self._preview_query(projection)
            )
        )
        preview = _parse(
            GovernanceGraphPreview,
            raw,
            "GFM_GOVERNANCE_PREVIEW_INVALID",
        )
        if (
            preview.run_id,
            preview.result_hash,
            preview.artifact_id,
            preview.dataset_content_hash,
            preview.graph_version_hash,
        ) != (
            result.run_id,
            result.result_hash,
            result.artifact_id,
            result.dataset_content_hash,
            result.graph_version_hash,
        ):
            raise GfmProxyError(502, "GFM_GOVERNANCE_PREVIEW_BINDING_MISMATCH")
        return preview

    async def evidence(self, run_id: str, node_id: str) -> NodeEvidenceV2:
        result = await self.result(run_id)
        assert self.client is not None
        evidence = _parse(
            NodeEvidenceV2,
            await self.client.get_governance_evidence(run_id, node_id),
            "GFM_GOVERNANCE_EVIDENCE_INVALID",
        )
        if (
            evidence.run_id,
            evidence.result_hash,
            evidence.artifact_id,
            evidence.dataset_content_hash,
            evidence.graph_version_hash,
            evidence.model_version_id,
            evidence.model_version_hash,
            evidence.model_state_hash,
            evidence.threshold,
            evidence.node.node_id,
        ) != (
            run_id,
            result.result_hash,
            result.artifact_id,
            result.dataset_content_hash,
            result.graph_version_hash,
            result.model_version_id,
            result.model_version_hash,
            result.model_state_hash,
            result.threshold,
            node_id,
        ):
            raise GfmProxyError(502, "GFM_GOVERNANCE_EVIDENCE_BINDING_MISMATCH")
        return evidence

    async def derivations(
        self, run_id: str, kind: str, *, offset: int, limit: int
    ) -> DerivationPage:
        await self.result(run_id)
        assert self.client is not None
        page = _parse(
            DerivationPage,
            await self.client.get_governance_derivations(run_id, kind, offset, limit),
            "GFM_GOVERNANCE_DERIVATION_INVALID",
        )
        if page.run_id != run_id:
            raise GfmProxyError(502, "GFM_GOVERNANCE_RESULT_BINDING_MISMATCH")
        expected_kind = {
            "groups": "group",
            "relations": "factual_relation",
            "potential-links": "potential_link",
        }.get(kind)
        if expected_kind is None or any(item.kind != expected_kind for item in page.items):
            raise GfmProxyError(502, "GFM_GOVERNANCE_DERIVATION_KIND_MISMATCH")
        return page

    async def create_case(self, request: CaseCreateRequest) -> GovernanceCase:
        await self.result(request.run_id)
        return self.governance.create_case(request)

    def cases(self, *, offset: int, limit: int) -> CaseList:
        return self.governance.list_cases(offset=offset, limit=limit)

    def case(self, case_id: str) -> GovernanceCase:
        return self.governance.get_case(case_id)

    def transition_case(
        self, case_id: str, request: CaseTransitionRequest
    ) -> GovernanceCase:
        return self.governance.transition(case_id, request)

    async def add_case_item(
        self, case_id: str, request: CaseItemRequest
    ) -> GovernanceCase:
        case = self.governance.get_case(case_id)
        if request.target_type == "node":
            await self.evidence(case.run_id, request.target_id)
        else:
            kinds = (
                ("groups",)
                if request.target_type == "group"
                else ("relations", "potential-links")
            )
            found = False
            for kind in kinds:
                offset = 0
                while True:
                    page = await self.derivations(
                        case.run_id, kind, offset=offset, limit=10_000
                    )
                    if request.target_id in {item.id for item in page.items}:
                        found = True
                        break
                    offset += len(page.items)
                    if not page.items or offset >= page.total:
                        break
                if found:
                    break
            if not found:
                raise GfmProxyError(404, "GOVERNANCE_TARGET_NOT_FOUND")
        return self.governance.add_item(case_id, request)

    def add_review(
        self, case_id: str, request: ReviewEventRequest
    ) -> GovernanceCase:
        return self.governance.add_review(case_id, request)

    async def report(self, case_id: str) -> dict[str, Any]:
        case = self.governance.get_case(case_id)
        result = await self.result(case.run_id)
        receipt = self.inbox.get(result.artifact_id)
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.governance-report/2.0",
            "case": case.model_dump(mode="json", by_alias=True),
            "provenance": {
                "runId": result.run_id,
                "requestHash": result.request_hash,
                "resultHash": result.result_hash,
                "artifactId": result.artifact_id,
                "artifactHash": receipt.artifact_hash,
                "datasetContentHash": result.dataset_content_hash,
                "graphVersionHash": result.graph_version_hash,
                "modelVersionId": result.model_version_id,
                "modelVersionHash": result.model_version_hash,
                "modelStateHash": result.model_state_hash,
            },
            "summary": {
                "state": case.state,
                "candidateCount": len(case.items),
                "reviewEventCount": len(case.review_events),
                "currentDecisions": case.current_decisions,
            },
            "audit": {
                "caseHash": case.case_hash,
                "stateTimeline": self.governance.case_state_timeline(case_id),
                "reviewEventHashes": [
                    event.event_hash for event in case.review_events
                ],
            },
            "limitations": list(result.limitations),
        }
        payload["reportHash"] = canonical_sha256(payload)
        return payload

    async def markdown_report(self, case_id: str) -> str:
        report = await self.report(case_id)
        case = report["case"]
        provenance = report["provenance"]
        lines = [
            f"# {case['title']}",
            "",
            f"Case: `{case['caseId']}`  ",
            f"State: `{case['state']}`  ",
            f"Run: `{provenance['runId']}`  ",
            f"Report hash: `{report['reportHash']}`",
            "",
            "## Review summary",
            "",
        ]
        decisions = report["summary"]["currentDecisions"]
        if decisions:
            lines.extend(f"- `{target}`: **{decision}**" for target, decision in decisions.items())
        else:
            lines.append("No analyst decision has been recorded.")
        lines.extend(["", "## Provenance", ""])
        lines.extend(f"- {key}: `{value}`" for key, value in provenance.items())
        lines.extend(["", "## Audit timeline", "", f"Case hash: `{case['caseHash']}`", ""])
        lines.extend(
            f"- State {event['sequence']}: **{event['state']}** / `{event['eventHash']}`"
            for event in report["audit"]["stateTimeline"]
        )
        lines.extend(
            f"- Review {event['sequence']}: **{event['decision']}** - "
            f"{event['reason'].replace(chr(10), ' ')} / `{event['eventHash']}`"
            for event in case["reviewEvents"]
        )
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {value}" for value in report["limitations"])
        return "\n".join(lines) + "\n"

    async def html_report(self, case_id: str) -> str:
        report = await self.report(case_id)
        case = report["case"]
        provenance = report["provenance"]
        decisions = report["summary"]["currentDecisions"]
        decision_rows = "".join(
            f"<tr><td>{html.escape(target)}</td><td>{html.escape(decision)}</td></tr>"
            for target, decision in decisions.items()
        ) or '<tr><td colspan="2">No analyst decision recorded.</td></tr>'
        provenance_rows = "".join(
            f"<tr><th>{html.escape(key)}</th><td><code>{html.escape(str(value))}</code></td></tr>"
            for key, value in provenance.items()
        )
        limitations = "".join(
            f"<li>{html.escape(value)}</li>" for value in report["limitations"]
        )
        timeline_rows = "".join(
            f"<tr><td>State {event['sequence']}</td><td>{html.escape(event['state'])}</td><td><code>{event['eventHash']}</code></td></tr>"
            for event in report["audit"]["stateTimeline"]
        ) + "".join(
            f"<tr><td>Review {event['sequence']}</td>"
            f"<td>{html.escape(event['decision'])}<br>"
            f"<small>{html.escape(event['reason'])}</small></td>"
            f"<td><code>{event['eventHash']}</code></td></tr>"
            for event in case["reviewEvents"]
        )
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(case['title'])}</title>
<style>@page{{size:A4;margin:18mm}}body{{font:14px/1.5 system-ui;color:#172126;max-width:900px;margin:auto}}
h1{{font-size:24px}}h2{{font-size:16px;margin-top:28px}}table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #d8e0e3;padding:8px;text-align:left;vertical-align:top}}code{{word-break:break-all}}
.stamp{{padding:10px;border-left:4px solid #d65f4a;background:#f7f9f9}}</style></head><body>
<h1>{html.escape(case['title'])}</h1><p class="stamp">Human-review governance report. It is not an automatic enforcement decision.</p>
<p>Case <code>{html.escape(case['caseId'])}</code> / state <strong>{html.escape(case['state'])}</strong></p>
<h2>Current decisions</h2><table><thead><tr><th>Target</th><th>Decision</th></tr></thead><tbody>{decision_rows}</tbody></table>
<h2>Audit timeline</h2><p>Case hash: <code>{case['caseHash']}</code></p><table>{timeline_rows}</table>
<h2>Provenance</h2><table>{provenance_rows}</table><h2>Limitations</h2><ul>{limitations}</ul>
<p>Report hash: <code>{report['reportHash']}</code></p></body></html>"""


__all__ = ["GovernanceClientProtocol", "GovernanceGateway"]
