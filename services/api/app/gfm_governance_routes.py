"""FastAPI routes for the isolated SocialGraph-FM Governance channel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import ValidationError

from .gfm_client import GfmProxyError
from .gfm_governance import GovernanceGateway
from .gfm_governance_artifacts import inspect_governance_bundle
from .gfm_hashing import canonical_sha256
from .gfm_governance_schemas import (
    GOVERNANCE_MODALITIES,
    GOVERNANCE_SCHEMA_VERSION,
    CaseCreateRequest,
    CaseItemRequest,
    CaseList,
    CaseTransitionRequest,
    DerivationPage,
    FindingsPage,
    GovernanceCase,
    GovernanceArtifact,
    GovernanceArtifactCompatibility,
    GovernanceArtifactList,
    GovernanceArtifactReceipt,
    GovernanceGraphPreview,
    GovernancePreviewQuery,
    GovernanceCapabilities,
    GovernanceHealth,
    NodeEvidenceV2,
    OnlineRunRequest,
    OnlineRunResult,
    OnlineRunStatus,
    ReviewEventRequest,
    RunComparison,
    RunList,
    AdaptationComparisonPage,
    TargetLabelSet,
    TargetLabelSetCreateRequest,
    TargetReviewPolicy,
    TargetTaskRegistration,
    AdaptationLabelSetCreateRequestV2,
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
)
from .gfm_governance_uploads import (
    GOVERNANCE_MULTIPART_OVERHEAD_MAX_BYTES,
    GOVERNANCE_UPLOAD_CHUNK_BYTES,
)

PREFIX = "/api/v2/gfm/governance"

if TYPE_CHECKING:
    from .governance_skills import GovernanceSkillsGateway


def _proxy(error: GfmProxyError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail={"code": error.code})


def _preview_projection(
    *,
    preset: str | None,
    node_budget: int | None,
    edge_budget: int | None,
    relation: str | None,
    anchor_node_ids: list[str] | None,
    group_budget: int | None,
) -> GovernancePreviewQuery | None:
    if preset is None:
        if any(
            value is not None
            for value in (node_budget, edge_budget, relation, group_budget)
        ) or anchor_node_ids:
            raise HTTPException(
                status_code=422, detail={"code": "GOVERNANCE_PREVIEW_QUERY_INVALID"}
            )
        return None
    try:
        return GovernancePreviewQuery.model_validate(
            {
                "preset": preset,
                "nodeBudget": node_budget,
                "edgeBudget": edge_budget,
                "relation": relation,
                "anchorNodeIds": anchor_node_ids or [],
                "groupBudget": group_budget,
            }
        )
    except ValidationError as error:
        raise HTTPException(
            status_code=422, detail={"code": "GOVERNANCE_PREVIEW_QUERY_INVALID"}
        ) from error


def _preview_response(preview: GovernanceGraphPreview) -> JSONResponse:
    payload = preview.model_dump(mode="json", by_alias=True)
    if preview.preset is None:
        for field in (
            "preset",
            "budgets",
            "selectionRecipeId",
            "isPartial",
            "groups",
            "sourceCounts",
            "inventoryCounts",
        ):
            payload.pop(field, None)
    return JSONResponse(payload)


async def _read_bounded_upload(file: UploadFile, *, max_bytes: int) -> bytes:
    payload = bytearray()
    try:
        while True:
            remaining_with_sentinel = max_bytes - len(payload) + 1
            chunk = await file.read(
                min(GOVERNANCE_UPLOAD_CHUNK_BYTES, remaining_with_sentinel)
            )
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail={"code": "GOVERNANCE_BUNDLE_TOO_LARGE"},
                )
    finally:
        await file.close()
    if not payload:
        raise HTTPException(
            status_code=400, detail={"code": "GOVERNANCE_BUNDLE_EMPTY"}
        )
    return bytes(payload)


def build_governance_router(
    gateway: GovernanceGateway,
    *,
    max_bundle_bytes: int,
    max_expanded_bytes: int,
    skills: GovernanceSkillsGateway | None = None,
) -> APIRouter:
    router = APIRouter(prefix=PREFIX, tags=["SocialGraph-FM Governance"])

    @router.get("/capabilities", response_model=GovernanceCapabilities)
    async def capabilities() -> GovernanceCapabilities:
        try:
            return await gateway.capabilities()
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/health", response_model=GovernanceHealth)
    async def health() -> GovernanceHealth:
        try:
            return await gateway.health()
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/input-contract")
    async def input_contract() -> dict[str, object]:
        return {
            "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
            "inputSchemaVersion": "socialgraph-fm.governance-input/2.0",
            "archiveMembers": [
                "manifest.json",
                "nodes.csv",
                "relations.csv",
                "features.npz",
            ],
            "nodeColumns": ["node_id", "display_name?"],
            "relationColumns": ["source", "target", "modality", "weight"],
            "featureArrays": {"node_ids": "[N] unicode", "text_features": "[N,768] float32"},
            "modalities": list(GOVERNANCE_MODALITIES),
            "limits": {
                "maxNodes": 10_000,
                "maxRelationRows": 500_000,
                "maxBundleBytes": max_bundle_bytes,
                "maxMultipartOverheadBytes": GOVERNANCE_MULTIPART_OVERHEAD_MAX_BYTES,
                "maxMultipartBodyBytes": (
                    max_bundle_bytes + GOVERNANCE_MULTIPART_OVERHEAD_MAX_BYTES
                ),
                "maxExpandedBytes": max_expanded_bytes,
            },
            "ordinaryGraphPolicy": (
                "Topology-only uploads remain available for structural exploration but cannot "
                "run SocialGraph-FM Governance analysis."
            ),
            "rawTextAccepted": False,
        }

    @router.post("/artifacts", response_model=GovernanceArtifact, status_code=201)
    async def upload_artifact(
        file: Annotated[UploadFile, File()],
        clean_self_loops: Annotated[bool, Form(alias="cleanSelfLoops")] = False,
    ) -> GovernanceArtifact:
        if not (file.filename or "").lower().endswith(".zip"):
            await file.close()
            raise HTTPException(
                status_code=400, detail={"code": "GOVERNANCE_ZIP_REQUIRED"}
            )
        payload = await _read_bounded_upload(file, max_bytes=max_bundle_bytes)
        try:
            receipt = gateway.inbox.commit(
                payload,
                clean_self_loops=clean_self_loops,
                max_expanded_bytes=max_expanded_bytes,
            )
            return await gateway.materialize(receipt.artifact_id)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post(
        "/target-tasks", response_model=TargetTaskRegistration, status_code=201
    )
    async def register_target_task(
        file: Annotated[UploadFile, File()],
    ) -> TargetTaskRegistration:
        if not (file.filename or "").lower().endswith((".zip", ".sgtask")):
            await file.close()
            raise HTTPException(
                status_code=400, detail={"code": "GOVERNANCE_ZIP_REQUIRED"}
            )
        payload = await _read_bounded_upload(file, max_bytes=max_bundle_bytes)
        try:
            return await gateway.register_target_task(
                payload, max_expanded_bytes=max_expanded_bytes
            )
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get(
        "/target-tasks/{registration_id}", response_model=TargetTaskRegistration
    )
    async def get_target_task(registration_id: str) -> TargetTaskRegistration:
        try:
            return gateway.target_task_registration(
                registration_id, max_expanded_bytes=max_expanded_bytes
            )
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post("/artifacts/compatibility", response_model=GovernanceArtifactCompatibility)
    async def inspect_artifact_compatibility(
        file: Annotated[UploadFile, File()],
    ) -> GovernanceArtifactCompatibility:
        if not (file.filename or "").lower().endswith(".zip"):
            await file.close()
            raise HTTPException(
                status_code=400, detail={"code": "GOVERNANCE_ZIP_REQUIRED"}
            )
        payload = await _read_bounded_upload(file, max_bytes=max_bundle_bytes)
        requires_cleaning = False
        try:
            try:
                manifest, inspected = inspect_governance_bundle(
                    payload,
                    clean_self_loops=False,
                    max_expanded_bytes=max_expanded_bytes,
                )
            except GfmProxyError as error:
                if error.code != "GOVERNANCE_SELF_LOOP_CONFIRMATION_REQUIRED":
                    raise
                requires_cleaning = True
                manifest, inspected = inspect_governance_bundle(
                    payload,
                    clean_self_loops=True,
                    max_expanded_bytes=max_expanded_bytes,
                )
            response: dict[str, object] = {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "inputSchemaVersion": manifest.schema_version,
                "compatible": True,
                "requiresSelfLoopCleaning": requires_cleaning,
                "prospectiveArtifactId": inspected["artifactId"],
                "datasetContentHash": inspected["datasetContentHash"],
                "graphVersionHash": inspected["graphVersionHash"],
                "nodeCount": inspected["nodeCount"],
                "relationRowCount": inspected["relationRowCount"],
                "selfLoopsDetected": inspected["selfLoopsRemoved"],
                "modalities": inspected["modalities"],
                "issues": (
                    ["SELF_LOOPS_REQUIRE_EXPLICIT_CLEANING"] if requires_cleaning else []
                ),
            }
            response["compatibilityHash"] = canonical_sha256(response)
            return GovernanceArtifactCompatibility.model_validate(response)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/artifacts", response_model=GovernanceArtifactList)
    async def list_artifacts(
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> GovernanceArtifactList:
        try:
            return gateway.list_artifacts(offset=offset, limit=limit)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/artifacts/{artifact_id}", response_model=GovernanceArtifactReceipt)
    async def get_artifact(artifact_id: str) -> GovernanceArtifactReceipt:
        try:
            return gateway.artifact(artifact_id)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post("/artifacts/{artifact_id}/materialize", response_model=GovernanceArtifact)
    async def materialize_artifact(artifact_id: str) -> GovernanceArtifact:
        try:
            return await gateway.materialize(artifact_id)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get(
        "/artifacts/{artifact_id}/preview",
        response_model=GovernanceGraphPreview,
    )
    async def preview_artifact(
        artifact_id: str,
        preset: Annotated[str | None, Query()] = None,
        node_budget: Annotated[int | None, Query(alias="nodeBudget")] = None,
        edge_budget: Annotated[int | None, Query(alias="edgeBudget")] = None,
        relation: Annotated[str | None, Query()] = None,
        anchor_node_id: Annotated[
            list[str] | None, Query(alias="anchorNodeId")
        ] = None,
        group_budget: Annotated[int | None, Query(alias="groupBudget")] = None,
    ) -> Response:
        try:
            projection = _preview_projection(
                preset=preset,
                node_budget=node_budget,
                edge_budget=edge_budget,
                relation=relation,
                anchor_node_ids=anchor_node_id,
                group_budget=group_budget,
            )
            return _preview_response(await gateway.preview(artifact_id, projection))
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post("/runs", response_model=OnlineRunStatus, status_code=202)
    async def create_run(body: OnlineRunRequest) -> OnlineRunStatus:
        try:
            return await gateway.create_run(body)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/runs", response_model=RunList)
    async def list_runs(
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> RunList:
        try:
            return await gateway.list_runs(offset=offset, limit=limit)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/runs/compare", response_model=RunComparison)
    async def compare_runs(
        left_run_id: Annotated[str, Query(alias="leftRunId")],
        right_run_id: Annotated[str, Query(alias="rightRunId")],
        limit: Annotated[int, Query(ge=1, le=1_000)] = 200,
    ) -> RunComparison:
        try:
            return await gateway.compare_runs(left_run_id, right_run_id, limit=limit)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/runs/{run_id}", response_model=OnlineRunStatus)
    async def get_run(run_id: str) -> OnlineRunStatus:
        try:
            return await gateway.get_run(run_id)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post("/runs/{run_id}/cancel", response_model=OnlineRunStatus)
    async def cancel_run(run_id: str) -> OnlineRunStatus:
        try:
            return await gateway.cancel(run_id)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post("/runs/{run_id}/retry", response_model=OnlineRunStatus, status_code=202)
    async def retry_run(run_id: str) -> OnlineRunStatus:
        try:
            return await gateway.retry(run_id)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/runs/{run_id}/result", response_model=OnlineRunResult)
    async def get_result(run_id: str) -> OnlineRunResult:
        try:
            return await gateway.result(run_id)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get(
        "/runs/{run_id}/graph-preview",
        response_model=GovernanceGraphPreview,
    )
    async def get_run_graph_preview(
        run_id: str,
        preset: Annotated[str | None, Query()] = None,
        node_budget: Annotated[int | None, Query(alias="nodeBudget")] = None,
        edge_budget: Annotated[int | None, Query(alias="edgeBudget")] = None,
        relation: Annotated[str | None, Query()] = None,
        anchor_node_id: Annotated[
            list[str] | None, Query(alias="anchorNodeId")
        ] = None,
        group_budget: Annotated[int | None, Query(alias="groupBudget")] = None,
    ) -> Response:
        try:
            projection = _preview_projection(
                preset=preset,
                node_budget=node_budget,
                edge_budget=edge_budget,
                relation=relation,
                anchor_node_ids=anchor_node_id,
                group_budget=group_budget,
            )
            return _preview_response(await gateway.run_preview(run_id, projection))
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/runs/{run_id}/nodes", response_model=FindingsPage)
    async def get_nodes(
        run_id: str,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=10_000)] = 100,
    ) -> FindingsPage:
        try:
            return await gateway.findings(run_id, offset=offset, limit=limit)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post(
        "/adaptations/label-sets",
        response_model=TargetLabelSet | TargetLabelSetV2,
        status_code=201,
    )
    async def create_adaptation_label_set(
        body: TargetLabelSetCreateRequest | AdaptationLabelSetCreateRequestV2,
    ) -> TargetLabelSet | TargetLabelSetV2:
        try:
            return await gateway.create_adaptation_label_set(body)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post(
        "/adaptations/label-sets/{label_set_hash}/policies",
        response_model=TargetReviewPolicy | TargetReviewPolicyV2,
        status_code=201,
    )
    async def fit_adaptation_policy(
        label_set_hash: str,
        body: TargetReviewPolicyFitRequest | None = None,
    ) -> TargetReviewPolicy | TargetReviewPolicyV2:
        try:
            return await gateway.fit_adaptation_policy(label_set_hash, body)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get(
        "/adaptations/policies/{policy_hash}",
        response_model=TargetReviewPolicy | TargetReviewPolicyV2,
    )
    async def get_adaptation_policy(
        policy_hash: str,
    ) -> TargetReviewPolicy | TargetReviewPolicyV2:
        try:
            return await gateway.adaptation_policy(policy_hash)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get(
        "/adaptations/runs/{run_id}/policies/{policy_hash}/comparison",
        response_model=AdaptationComparisonPage | AdaptationComparisonV2,
    )
    async def get_adaptation_comparison(
        run_id: str,
        policy_hash: str,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> AdaptationComparisonPage | AdaptationComparisonV2:
        try:
            return await gateway.adaptation_comparison(
                run_id, policy_hash, offset=offset, limit=limit
            )
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post(
        "/adaptations/review-collections",
        response_model=ReviewCollection,
        status_code=201,
    )
    async def create_review_collection(
        body: ReviewCollectionCreateRequest,
    ) -> ReviewCollection:
        try:
            return await gateway.create_review_collection(body)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post(
        "/adaptations/handoffs",
        response_model=AdaptationGovernanceHandoff,
        status_code=201,
    )
    async def create_adaptation_handoff(
        body: AdaptationHandoffCreateRequest,
    ) -> AdaptationGovernanceHandoff:
        try:
            return await gateway.create_adaptation_handoff(body)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get(
        "/adaptations/handoffs/{handoff_hash}",
        response_model=AdaptationGovernanceHandoff,
    )
    async def get_adaptation_handoff(
        handoff_hash: str,
    ) -> AdaptationGovernanceHandoff:
        try:
            return await gateway.adaptation_handoff(handoff_hash)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post(
        "/adaptations/policies/{policy_hash}/activate",
        response_model=AdaptationOverlayActivation,
        status_code=201,
    )
    async def activate_adaptation_overlay(
        policy_hash: str,
        body: AdaptationOverlayActivationRequest,
    ) -> AdaptationOverlayActivation:
        try:
            return await gateway.activate_adaptation_overlay(policy_hash, body)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get(
        "/runs/{run_id}/nodes/{node_id:path}/evidence", response_model=NodeEvidenceV2
    )
    async def get_evidence(run_id: str, node_id: str) -> NodeEvidenceV2:
        try:
            return await gateway.evidence(run_id, node_id)
        except GfmProxyError as error:
            raise _proxy(error) from error

    async def _derivations(
        run_id: str, kind: str, offset: int, limit: int
    ) -> DerivationPage:
        try:
            return await gateway.derivations(run_id, kind, offset=offset, limit=limit)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/runs/{run_id}/groups", response_model=DerivationPage)
    async def get_groups(
        run_id: str,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=10_000)] = 100,
    ) -> DerivationPage:
        return await _derivations(run_id, "groups", offset, limit)

    @router.get("/runs/{run_id}/relations", response_model=DerivationPage)
    async def get_relations(
        run_id: str,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=10_000)] = 100,
    ) -> DerivationPage:
        return await _derivations(run_id, "relations", offset, limit)

    @router.get("/runs/{run_id}/potential-links", response_model=DerivationPage)
    async def get_potential_links(
        run_id: str,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=10_000)] = 100,
    ) -> DerivationPage:
        return await _derivations(run_id, "potential-links", offset, limit)

    @router.post("/cases", response_model=GovernanceCase, status_code=201)
    async def create_case(body: CaseCreateRequest) -> GovernanceCase:
        try:
            return await gateway.create_case(body)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/cases", response_model=CaseList)
    async def list_cases(
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> CaseList:
        try:
            return gateway.cases(offset=offset, limit=limit)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/cases/{case_id}", response_model=GovernanceCase)
    async def get_case(case_id: str) -> GovernanceCase:
        try:
            return gateway.case(case_id)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post("/cases/{case_id}/transitions", response_model=GovernanceCase)
    async def transition_case(
        case_id: str, body: CaseTransitionRequest
    ) -> GovernanceCase:
        try:
            case = gateway.transition_case(case_id, body)
            if skills is not None and case.state == "concluded":
                await skills.ensure_case_indexed(case.case_id)
            return case
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post("/cases/{case_id}/items", response_model=GovernanceCase, status_code=201)
    async def add_case_item(case_id: str, body: CaseItemRequest) -> GovernanceCase:
        try:
            return await gateway.add_case_item(case_id, body)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post(
        "/cases/{case_id}/review-events", response_model=GovernanceCase, status_code=201
    )
    async def add_review_event(
        case_id: str, body: ReviewEventRequest
    ) -> GovernanceCase:
        try:
            return gateway.add_review(case_id, body)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/cases/{case_id}/report")
    async def export_report(
        case_id: str,
        format: Annotated[Literal["json", "markdown", "html"], Query()] = "json",
    ) -> Response:
        try:
            if format == "json":
                return JSONResponse(await gateway.report(case_id))
            if format == "markdown":
                return PlainTextResponse(
                    await gateway.markdown_report(case_id),
                    media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{case_id}.md"'},
                )
            return HTMLResponse(await gateway.html_report(case_id))
        except GfmProxyError as error:
            raise _proxy(error) from error

    return router


__all__ = ["PREFIX", "build_governance_router"]
