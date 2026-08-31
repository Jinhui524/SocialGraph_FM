from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, cast
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import Settings, get_settings
from .dataset_imports import DatasetImportService
from .dataset_schemas import (
    DatasetArtifact,
    DatasetArtifactDeletionImpact,
    DatasetArtifactLifecycleResponse,
    DatasetArtifactPurgeRequest,
    DatasetArtifactPurgeResponse,
    DatasetArtifactRef,
    DatasetInspection,
    DatasetInspectionCancellation,
    DatasetReadiness,
    DatasetStoreDiagnostics,
    GraphDatasetHandoffRequest,
    GraphDatasetHandoffResponse,
    GraphHandoffCancellation,
    GraphHandoffCancelRequest,
    GraphHandoffReservation,
    GraphHandoffReserveRequest,
    MaterializedDatasetBundle,
    OrphanArtifactDirectory,
    OrphanArtifactRecoveryResponse,
    TrainingRefResolveRequest,
    TrainingRefResolveResponse,
    TrustedConversionAuthorizeRequest,
    TrustedConversionJob,
    TrustedLocalInspection,
    TrustedLocalInspectRequest,
)
from .gfm_client import (
    CoreGateway,
    CoreGraphResolver,
    CoreRunBindingStore,
    GfmClientProtocol,
    GfmProxyError,
    GfmServiceClient,
)
from .gfm_global_model import (
    GlobalModelClientProtocol,
    GlobalModelGateway,
    GlobalModelReviewStore,
    GlobalModelRunBindingStore,
)
from .gfm_global_model_schemas import (
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
)
from .gfm_governance import GovernanceClientProtocol, GovernanceGateway
from .gfm_governance_artifacts import GovernanceArtifactInbox
from .gfm_governance_routes import build_governance_router
from .gfm_governance_store import GovernanceStore
from .gfm_governance_uploads import GovernanceUploadLimitMiddleware
from .governance_skills import GovernanceSkillsClientProtocol, GovernanceSkillsGateway
from .governance_skills_routes import build_governance_skills_router
from .governance_skills_store import GovernanceSkillsStore
from .gfm_research import (
    ResearchClientProtocol,
    ResearchGateway,
    ResearchRunBindingStore,
)
from .gfm_research_schemas import (
    ResearchCapabilities,
    ResearchGraphCompatibility,
    ResearchRunRequest,
    ResearchRunResult,
    ResearchRunStatus,
    ResearchScenarioGraphPreview,
    ResearchScenariosResponse,
    SimilarNodesRequest,
    SimilarNodesResponse,
)
from .gfm_core_serving_control import CoreServingControlStore
from .gfm_core_schemas import (
    CoreCapabilitiesResponse,
    CoreRunRequest,
    CoreRunResult,
    CoreRunStatus,
)
from .graph_build_intents import GraphBuildIntentService
from .normalizer import IntentNormalizerService
from .provider import IntentProvider, OpenAICompatibleProvider, ProviderFailure
from .runtime_fingerprint import (
    converter_environment_details,
    converter_environment_fingerprint,
    public_converter_environment_summary,
)
from .schemas import (
    AnalysisCapabilities,
    CapabilitiesResponse,
    DataBoundaryCapability,
    GraphBuildIntentResponse,
    HealthResponse,
    IntentNormalizationCapability,
    IntentNormalizationResponse,
    NormalizeGraphBuildIntentRequest,
    NormalizeIntentRequest,
    ResearchDatasetCapability,
    RuntimeContractCapability,
)
from .trusted_conversion import TrustedConversionService, is_loopback_host

logger = logging.getLogger(__name__)

_RETRYABLE_RESEARCH_REGISTRATION_CODES = frozenset(
    {
        "GFM_CORE_SERVICE_UNAVAILABLE",
        "GFM_RESEARCH_SERVICE_UNAVAILABLE",
        "GFM_RESEARCH_GRAPH_REGISTRATION_PENDING",
        "GFM_CORE_MODEL_NOT_INSTALLED",
        "GFM_RESEARCH_MODEL_NOT_INSTALLED",
    }
)


def create_app(
    settings: Settings | None = None,
    *,
    provider: IntentProvider | None = None,
    gfm_client: GfmClientProtocol | None = None,
    gfm_research_client: ResearchClientProtocol | None = None,
    gfm_global_model_client: GlobalModelClientProtocol | None = None,
    gfm_governance_client: GovernanceClientProtocol | None = None,
) -> FastAPI:
    runtime_settings = settings or get_settings()
    logging.basicConfig(level=getattr(logging, runtime_settings.log_level))

    actual_provider = provider
    if actual_provider is None and runtime_settings.llm_configured:
        actual_provider = OpenAICompatibleProvider(runtime_settings)
    normalizer = IntentNormalizerService(actual_provider)
    graph_build_intents = GraphBuildIntentService(actual_provider)
    dataset_imports = DatasetImportService(runtime_settings)
    trusted_conversions = TrustedConversionService(runtime_settings, dataset_imports)
    actual_gfm_client = gfm_client
    if actual_gfm_client is None and runtime_settings.gfm_service_url:
        actual_gfm_client = GfmServiceClient(
            runtime_settings.gfm_service_url,
            token_file=runtime_settings.gfm_session_token_file or "",
            timeout_seconds=runtime_settings.gfm_timeout_seconds,
        )
    core_binding_root = runtime_settings.gfm_core_run_binding_root or str(
        Path(runtime_settings.dataset_storage_root) / "core-run-bindings"
    )
    control_file = runtime_settings.gfm_core_serving_control_file or str(
        Path(__file__).parent / "contracts" / "core-serving-control.json"
    )
    high_water_root = runtime_settings.gfm_core_serving_high_water_root or str(
        Path(core_binding_root).parent / "core-serving-control-high-water"
    )
    core_serving_control_store = CoreServingControlStore(
        control_file,
        high_water_root=high_water_root,
    )
    core_gateway = CoreGateway(
        actual_gfm_client,
        binding_store=CoreRunBindingStore(core_binding_root),
        serving_control_store=core_serving_control_store,
    )
    core_graph_resolver = CoreGraphResolver(
        dataset_imports.store,
        serving_control_store=core_serving_control_store,
    )
    actual_research_client = gfm_research_client
    if actual_research_client is None and actual_gfm_client is not None and hasattr(
        actual_gfm_client, "research_capabilities"
    ):
        actual_research_client = actual_gfm_client
    research_binding_root = runtime_settings.gfm_research_run_binding_root or str(
        Path(core_binding_root).parent / "research-run-bindings"
    )
    gfm_research_gateway = ResearchGateway(
        actual_research_client,
        dataset_store=dataset_imports.store,
        binding_store=ResearchRunBindingStore(research_binding_root),
    )
    actual_global_model_client = gfm_global_model_client
    if actual_global_model_client is None and actual_gfm_client is not None and hasattr(
        actual_gfm_client, "global_model_capabilities"
    ):
        actual_global_model_client = actual_gfm_client
    global_model_binding_root = runtime_settings.gfm_global_model_run_binding_root or str(
        Path(core_binding_root).parent / "global-model-run-bindings"
    )
    global_model_review_root = runtime_settings.gfm_global_model_review_root or str(
        Path(core_binding_root).parent / "global-model-reviews"
    )
    gfm_global_model_gateway = GlobalModelGateway(
        actual_global_model_client,
        binding_store=GlobalModelRunBindingStore(global_model_binding_root),
        review_store=GlobalModelReviewStore(global_model_review_root),
    )
    actual_governance_client = gfm_governance_client
    if actual_governance_client is None and actual_gfm_client is not None and hasattr(
        actual_gfm_client, "governance_capabilities"
    ):
        actual_governance_client = actual_gfm_client
    governance_root = runtime_settings.gfm_governance_root or str(
        Path(runtime_settings.dataset_storage_root).resolve().parent
        / "gfm"
        / "governance"
    )
    governance_store = GovernanceStore(governance_root)
    gfm_governance_gateway = GovernanceGateway(
        actual_governance_client,
        inbox=GovernanceArtifactInbox(governance_root),
        governance=governance_store,
        max_expanded_bytes=runtime_settings.gfm_governance_expanded_max_bytes,
    )
    gfm_governance_skills_gateway = GovernanceSkillsGateway(
        cast(GovernanceSkillsClientProtocol | None, actual_governance_client),
        governance=gfm_governance_gateway,
        store=GovernanceSkillsStore(governance_root),
        confirmation_ttl_seconds=runtime_settings.gfm_governance_confirmation_ttl_seconds,
        provider=actual_provider,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await trusted_conversions.close()
        close = getattr(actual_provider, "aclose", None)
        if close is not None:
            await close()

    app = FastAPI(
        title="SocialGraph-FM API",
        version="0.1.0",
        description="Narrow intent-normalization gateway. It never receives raw graph data.",
        lifespan=lifespan,
    )
    app.state.settings = runtime_settings
    app.state.intent_provider = actual_provider
    app.state.intent_normalizer = normalizer
    app.state.graph_build_intents = graph_build_intents
    app.state.dataset_imports = dataset_imports
    app.state.trusted_conversions = trusted_conversions
    app.state.core_gateway = core_gateway
    app.state.core_graph_resolver = core_graph_resolver
    app.state.core_serving_control_store = core_serving_control_store
    app.state.gfm_research_gateway = gfm_research_gateway
    app.state.gfm_global_model_gateway = gfm_global_model_gateway
    app.state.gfm_governance_gateway = gfm_governance_gateway
    app.state.gfm_governance_skills_gateway = gfm_governance_skills_gateway

    @app.exception_handler(ProviderFailure)
    async def provider_failure_handler(
        _: Request, error: ProviderFailure
    ) -> JSONResponse:
        unavailable = {
            "LLM_NOT_CONFIGURED",
            "LLM_TIMEOUT",
            "LLM_NETWORK_ERROR",
            "LLM_RATE_LIMITED",
            "LLM_UPSTREAM_ERROR",
        }
        return JSONResponse(
            status_code=(
                503 if error.retryable or error.code in unavailable else 502
            ),
            content={"detail": {"code": error.code}},
        )

    if runtime_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=runtime_settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID", "X-Isolated-Artifact-Rows"],
        )
    app.add_middleware(
        GovernanceUploadLimitMiddleware,
        max_bundle_bytes=runtime_settings.gfm_governance_bundle_max_bytes,
    )

    @app.middleware("http")
    async def metadata_logging(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID", "").strip()[:128] or str(uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1_000, 2)
            logger.error(
                "request_failed request_id=%s method=%s path=%s duration_ms=%s",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        duration_ms = round((time.perf_counter() - started) * 1_000, 2)
        content_type = response.headers.get("Content-Type", "")
        media_type = content_type.partition(";")[0].strip().lower()
        is_json = media_type == "application/json" or media_type.endswith("+json")
        if is_json and "charset=" not in content_type.lower():
            response.headers["Content-Type"] = f"{content_type}; charset=utf-8"
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_complete request_id=%s method=%s path=%s status=%d duration_ms=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.middleware("http")
    async def enforce_local_demo_boundary(request: Request, call_next):  # type: ignore[no-untyped-def]
        client_host = request.client.host if request.client else None
        test_client = client_host == "testclient" and "pytest" in sys.modules
        if request.method == "POST" and (
            request.url.path in {
            "/api/v1/gfm/runs",
            "/api/v1/gfm/research/runs",
            "/api/v1/gfm/research/similar-nodes",
            }
            or request.url.path == "/api/v2/gfm/governance/runs"
            or request.url.path == "/api/v2/gfm/governance/cases"
            or request.url.path in {
                "/api/v2/gfm/governance/skills/execute",
                "/api/v2/gfm/governance/skills/confirm",
                "/api/v2/gfm/governance/knowledge/search",
                "/api/v2/gfm/governance/similar-cases/search",
                "/api/v2/gfm/governance/case-index/backfill",
                "/api/v2/gfm/governance/assistant/execute",
            }
            or (
                request.url.path.startswith("/api/v2/gfm/governance/skills/")
                and request.url.path.endswith("/execute")
            )
            or request.url.path.endswith("/transitions")
            or request.url.path.endswith("/items")
            or request.url.path.endswith("/review-events")
        ):
            media_type = request.headers.get("content-type", "").partition(";")[0].lower()
            if media_type != "application/json":
                return JSONResponse(
                    status_code=415,
                    content={"detail": {"code": "GFM_JSON_REQUIRED"}},
                )
            try:
                content_length = int(request.headers.get("content-length", ""))
            except ValueError:
                content_length = -1
            if content_length < 1 or content_length > runtime_settings.gfm_request_max_bytes:
                return JSONResponse(
                    status_code=413 if content_length > runtime_settings.gfm_request_max_bytes else 400,
                    content={"detail": {"code": "GFM_REQUEST_SIZE_INVALID"}},
                )
        if (
            runtime_settings.local_demo_loopback_only
            and not test_client
            and not is_loopback_host(client_host)
        ):
            trusted_conversion_path = request.url.path.startswith(
                "/api/v1/dataset-imports/inspect-local"
            ) or request.url.path.startswith("/api/v1/dataset-imports/local-jobs/")
            return JSONResponse(
                status_code=403,
                content={
                    "detail": {
                        "code": (
                            "TRUSTED_CONVERSION_LOOPBACK_ONLY"
                            if trusted_conversion_path
                            else "LOOPBACK_ONLY"
                        ),
                        "message": "本地 API 只接受 loopback 客户端。",
                    }
                },
            )
        return await call_next(request)

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/api/v1/capabilities", response_model=CapabilitiesResponse)
    async def capabilities() -> CapabilitiesResponse:
        configured = actual_provider is not None
        converter_environment = converter_environment_details(runtime_settings)
        converter_fingerprint = converter_environment_fingerprint(runtime_settings)
        return CapabilitiesResponse(
            intentNormalization=IntentNormalizationCapability(
                configured=configured,
                mode="llm_required",
                provider="openai_compatible" if configured else None,
                model=actual_provider.model if actual_provider else None,
                apiMode="chat_completions",
                connectionStatus=(
                    getattr(actual_provider, "connection_status", "configured_unverified")
                    if actual_provider
                    else "not_configured"
                ),
            ),
            analysis=AnalysisCapabilities(
                localTasks=["overview", "centrality", "bridge_detection", "community"],
                gfmTasks=["link_prediction", "node_role", "similar_structure"],
                gfmConnected=False,
            ),
            dataBoundary=DataBoundaryCapability(
                sendsRawGraph=False,
                allowedGraphFields=[
                    "nodeCount",
                    "edgeCount",
                    "density",
                    "connectedComponents",
                    "nodeTypes",
                    "edgeTypes",
                    "hasWeight",
                    "hasTimestamp",
                    "timeRange",
                ],
            ),
            researchDatasets=ResearchDatasetCapability(
                persistentArtifacts=True,
                trustedLocalEnabled=runtime_settings.enable_trusted_local_conversion
                and bool(runtime_settings.trusted_roots),
                loopbackOnly=True,
                safeUploadFormats=[
                    "geom_gcn_text",
                    "graph_npz",
                    "strict_split_npz",
                    "fewshot_json_npz",
                    "socialgraph_dataset_package",
                    "graph_version_target_domain",
                ],
            ),
            runtime=RuntimeContractCapability(
                buildId=runtime_settings.runtime_build_id,
                datasetArtifactSchemas=["1.0", "2.0", "2.1", "2.2"],
                trainingRefSchemas=["1.0", "1.1"],
                graphHandoffSchemas=[
                    "socialgraph-fm-graph/1.0",
                    "socialgraph-fm-graph/1.1",
                ],
                converterEnvironmentFingerprint=converter_fingerprint,
                converterEnvironment=public_converter_environment_summary(
                    converter_environment
                ),
            ),
        )

    @app.get(
        "/api/v1/gfm/capabilities",
        response_model=CoreCapabilitiesResponse,
    )
    async def core_capabilities() -> CoreCapabilitiesResponse:
        try:
            capabilities = await core_gateway.capabilities()
            payload = capabilities.model_dump(mode="python", by_alias=True)
            payload["schemaVersion"] = "socialgraph-fm.core-capabilities/2.0"
            return CoreCapabilitiesResponse.model_validate(payload)
        except (GfmProxyError, ValueError) as error:
            code = error.code if isinstance(error, GfmProxyError) else "GFM_REGISTRY_INVALID"
            status = error.status_code if isinstance(error, GfmProxyError) else 503
            raise HTTPException(status_code=status, detail={"code": code}) from error

    @app.get(
        "/api/v1/gfm/research/capabilities",
        response_model=ResearchCapabilities,
    )
    async def gfm_research_capabilities() -> ResearchCapabilities:
        try:
            return await gfm_research_gateway.capabilities()
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.get(
        "/api/v1/gfm/research/scenarios",
        response_model=ResearchScenariosResponse,
    )
    async def gfm_research_scenarios() -> ResearchScenariosResponse:
        try:
            return await gfm_research_gateway.scenarios()
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.get(
        "/api/v1/gfm/research/scenarios/{scenario_id}/graph-preview",
        response_model=ResearchScenarioGraphPreview,
    )
    async def gfm_research_scenario_preview(
        scenario_id: str,
    ) -> ResearchScenarioGraphPreview:
        try:
            return await gfm_research_gateway.scenario_preview(scenario_id)
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.post(
        "/api/v1/gfm/research/runs",
        response_model=ResearchRunStatus,
        status_code=202,
    )
    async def create_gfm_research_run(body: ResearchRunRequest) -> ResearchRunStatus:
        try:
            return await gfm_research_gateway.create_run(body)
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.get(
        "/api/v1/gfm/research/runs/{run_id}",
        response_model=ResearchRunStatus,
    )
    async def get_gfm_research_run(run_id: str) -> ResearchRunStatus:
        try:
            return await gfm_research_gateway.get_run(run_id)
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.get(
        "/api/v1/gfm/research/runs/{run_id}/result",
        response_model=ResearchRunResult,
    )
    async def get_gfm_research_result(run_id: str) -> ResearchRunResult:
        try:
            return await gfm_research_gateway.get_result(run_id)
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.post(
        "/api/v1/gfm/research/similar-nodes",
        response_model=SimilarNodesResponse,
    )
    async def get_gfm_research_similar_nodes(
        body: SimilarNodesRequest,
    ) -> SimilarNodesResponse:
        try:
            return await gfm_research_gateway.similar_nodes(body)
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.get(
        "/api/v1/gfm/global-model/capabilities",
        response_model=GlobalModelCapabilities,
    )
    async def gfm_global_model_capabilities() -> GlobalModelCapabilities:
        try:
            return await gfm_global_model_gateway.capabilities()
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.get(
        "/api/v1/gfm/global-model/health",
        response_model=GlobalModelHealth,
    )
    async def gfm_global_model_health() -> GlobalModelHealth:
        try:
            return await gfm_global_model_gateway.health()
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.get(
        "/api/v1/gfm/global-model/model-card",
        response_model=GlobalModelCard,
    )
    async def gfm_global_model_card() -> GlobalModelCard:
        try:
            return await gfm_global_model_gateway.model_card()
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.get(
        "/api/v1/gfm/global-model/scenario",
        response_model=GlobalModelScenario,
    )
    async def gfm_global_model_scenario() -> GlobalModelScenario:
        try:
            return await gfm_global_model_gateway.scenario()
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.get(
        "/api/v1/gfm/global-model/scenario/graph-preview",
        response_model=GlobalModelScenarioPreview,
    )
    async def gfm_global_model_scenario_preview() -> GlobalModelScenarioPreview:
        try:
            return await gfm_global_model_gateway.scenario_preview()
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.post(
        "/api/v1/gfm/global-model/runs",
        response_model=GlobalModelRunStatus,
        status_code=202,
    )
    async def create_gfm_global_model_run(body: GlobalModelRunRequest) -> GlobalModelRunStatus:
        try:
            return await gfm_global_model_gateway.create_run(body)
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.get(
        "/api/v1/gfm/global-model/runs/{run_id}",
        response_model=GlobalModelRunStatus,
    )
    async def get_gfm_global_model_run(run_id: str) -> GlobalModelRunStatus:
        try:
            return await gfm_global_model_gateway.get_run(run_id)
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.get(
        "/api/v1/gfm/global-model/runs/{run_id}/result",
        response_model=GlobalModelRunResult,
    )
    async def get_gfm_global_model_result(run_id: str) -> GlobalModelRunResult:
        try:
            return await gfm_global_model_gateway.get_result(run_id)
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.get(
        "/api/v1/gfm/global-model/runs/{run_id}/nodes/{node_id}/evidence",
        response_model=GlobalModelNodeEvidence,
    )
    async def get_gfm_global_model_evidence(
        run_id: str, node_id: str
    ) -> GlobalModelNodeEvidence:
        try:
            return await gfm_global_model_gateway.evidence(run_id, node_id)
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.post(
        "/api/v1/gfm/global-model/runs/{run_id}/reviews",
        response_model=GlobalModelReviewRecord,
        status_code=201,
    )
    async def create_gfm_global_model_review(
        run_id: str, body: GlobalModelReviewRequest
    ) -> GlobalModelReviewRecord:
        try:
            return await gfm_global_model_gateway.review(run_id, body)
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code, detail={"code": error.code}
            ) from error

    @app.post("/api/v1/gfm/runs", response_model=CoreRunStatus, status_code=202)
    async def create_core_run(body: CoreRunRequest) -> CoreRunStatus:
        try:
            capabilities = await core_gateway.capabilities()
            if not capabilities.serving_ready:
                raise HTTPException(
                    status_code=503,
                    detail={"code": "GFM_CORE_MODEL_NOT_INSTALLED"},
                )
            try:
                graph = core_graph_resolver.resolve(
                    body.graph_version_id,
                    capabilities,
                    required_model_id=body.model_version_id,
                )
            except LookupError as error:
                raise HTTPException(
                    status_code=404,
                    detail={"code": "GFM_GRAPH_VERSION_NOT_FOUND"},
                ) from error
            try:
                core_gateway.validate_compatibility(body, graph, capabilities)
            except ValueError as error:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "GFM_MODEL_GRAPH_INCOMPATIBLE"},
                ) from error
            return await core_gateway.create_run(body, graph, capabilities)
        except GfmProxyError as error:
            raise HTTPException(
                status_code=error.status_code,
                detail={"code": error.code},
            ) from error

    @app.get("/api/v1/gfm/runs/{run_id}", response_model=CoreRunStatus)
    async def get_core_run(run_id: str) -> CoreRunStatus:
        try:
            return await core_gateway.get_run(run_id)
        except (GfmProxyError, ValueError) as error:
            code = error.code if isinstance(error, GfmProxyError) else "GFM_CORE_RESPONSE_INVALID"
            status = error.status_code if isinstance(error, GfmProxyError) else 502
            raise HTTPException(status_code=status, detail={"code": code}) from error

    @app.get("/api/v1/gfm/runs/{run_id}/result", response_model=CoreRunResult)
    async def get_core_run_result(run_id: str) -> CoreRunResult:
        try:
            return await core_gateway.get_result(run_id)
        except (GfmProxyError, ValueError) as error:
            code = error.code if isinstance(error, GfmProxyError) else "GFM_CORE_RESPONSE_INVALID"
            status = error.status_code if isinstance(error, GfmProxyError) else 502
            raise HTTPException(status_code=status, detail={"code": code}) from error

    @app.post("/api/v1/intents/normalize", response_model=IntentNormalizationResponse)
    async def normalize_intent(
        body: NormalizeIntentRequest,
        request: Request,
    ) -> IntentNormalizationResponse:
        return await app.state.intent_normalizer.normalize(
            body,
            request_id=request.state.request_id,
        )

    @app.post(
        "/api/v1/graph-build-intents/normalize",
        response_model=GraphBuildIntentResponse,
    )
    async def normalize_graph_build_intent(
        body: NormalizeGraphBuildIntentRequest,
        request: Request,
    ) -> GraphBuildIntentResponse:
        return await app.state.graph_build_intents.normalize(
            body,
            request_id=request.state.request_id,
        )

    @app.post("/api/v1/dataset-imports/inspect", response_model=DatasetInspection)
    async def inspect_dataset(
        files: Annotated[list[UploadFile] | None, File()] = None,
        file: Annotated[UploadFile | None, File()] = None,
        dataset: Annotated[str | None, Form(max_length=200)] = None,
        project_id: Annotated[
            str,
            Header(
                alias="X-SocialGraph-Project-ID",
                min_length=1,
                max_length=64,
                pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
            ),
        ] = "local-default",
    ) -> DatasetInspection:
        uploads = list(files or [])
        if file is not None:
            uploads.append(file)
        return await app.state.dataset_imports.inspect(
            uploads,
            selected_dataset=dataset,
            project_id=project_id,
        )

    @app.post(
        "/api/v1/dataset-imports/{inspection_id}/commit",
        response_model=DatasetArtifact,
    )
    async def commit_dataset(inspection_id: str) -> DatasetArtifact:
        return app.state.dataset_imports.commit(inspection_id)

    @app.post(
        "/api/v1/dataset-imports/{inspection_id}/cancel",
        response_model=DatasetInspectionCancellation,
    )
    async def cancel_dataset_inspection(
        inspection_id: str,
    ) -> DatasetInspectionCancellation:
        return app.state.dataset_imports.cancel_inspection(inspection_id)

    @app.get(
        "/api/v1/dataset-artifacts/{artifact_id}",
        response_model=DatasetArtifact,
    )
    async def get_dataset_artifact(artifact_id: str) -> DatasetArtifact:
        return app.state.dataset_imports.get_artifact(artifact_id)

    @app.get("/api/v1/dataset-artifacts", response_model=list[DatasetArtifactRef])
    async def list_dataset_artifacts(
        response: Response,
        include_trashed: Annotated[bool, Query(alias="includeTrashed")] = False,
    ) -> list[DatasetArtifactRef]:
        artifacts = app.state.dataset_imports.list_artifacts(
            include_trashed=include_trashed
        )
        response.headers["X-Isolated-Artifact-Rows"] = str(
            len(app.state.dataset_imports.store.last_list_issues)
        )
        return artifacts

    @app.get(
        "/api/v1/dataset-store/diagnostics",
        response_model=DatasetStoreDiagnostics,
    )
    async def get_dataset_store_diagnostics() -> DatasetStoreDiagnostics:
        # Refresh the isolated-row audit without allowing one malformed legacy
        # row to fail the endpoint.
        app.state.dataset_imports.list_artifacts(include_trashed=True)
        return DatasetStoreDiagnostics(
            isolatedArtifactRows=app.state.dataset_imports.store.last_list_issues,
            trashedArtifactIds=app.state.dataset_imports.store.list_trashed_artifact_ids(),
            orphanArtifacts=app.state.dataset_imports.list_orphan_artifacts(),
        )

    @app.get(
        "/api/v1/dataset-artifacts/{artifact_id}/deletion-impact",
        response_model=DatasetArtifactDeletionImpact,
    )
    async def preview_dataset_artifact_deletion(
        artifact_id: str,
    ) -> DatasetArtifactDeletionImpact:
        return app.state.dataset_imports.artifact_deletion_impact(artifact_id)

    @app.post(
        "/api/v1/dataset-artifacts/{artifact_id}/trash",
        response_model=DatasetArtifactLifecycleResponse,
    )
    async def trash_dataset_artifact(
        artifact_id: str,
    ) -> DatasetArtifactLifecycleResponse:
        return app.state.dataset_imports.trash_artifact(artifact_id)

    @app.post(
        "/api/v1/dataset-artifacts/{artifact_id}/restore",
        response_model=DatasetArtifactLifecycleResponse,
    )
    async def restore_dataset_artifact(
        artifact_id: str,
    ) -> DatasetArtifactLifecycleResponse:
        return app.state.dataset_imports.restore_artifact(artifact_id)

    @app.post(
        "/api/v1/dataset-artifacts/{artifact_id}/purge",
        response_model=DatasetArtifactPurgeResponse,
    )
    async def purge_dataset_artifact(
        artifact_id: str,
        body: DatasetArtifactPurgeRequest,
    ) -> DatasetArtifactPurgeResponse:
        return app.state.dataset_imports.purge_artifact(
            artifact_id,
            impact_hash=body.impact_hash,
            confirmation=body.confirmation,
        )

    @app.get(
        "/api/v1/dataset-store/orphans",
        response_model=list[OrphanArtifactDirectory],
    )
    async def list_dataset_store_orphans() -> list[OrphanArtifactDirectory]:
        return app.state.dataset_imports.list_orphan_artifacts()

    @app.post(
        "/api/v1/dataset-store/orphans/{artifact_id}/recover",
        response_model=OrphanArtifactRecoveryResponse,
    )
    async def recover_dataset_store_orphan(
        artifact_id: str,
    ) -> OrphanArtifactRecoveryResponse:
        return app.state.dataset_imports.recover_orphan_artifact(artifact_id)

    @app.get(
        "/api/v1/dataset-artifacts/{artifact_id}/readiness",
        response_model=DatasetReadiness,
    )
    async def get_dataset_artifact_readiness(
        artifact_id: str,
        training_ref_hash: Annotated[
            str | None,
            Query(alias="trainingRefHash", pattern=r"^[0-9a-f]{64}$"),
        ] = None,
    ) -> DatasetReadiness:
        return app.state.dataset_imports.readiness(
            artifact_id,
            training_ref_hash=training_ref_hash,
        )

    @app.post(
        "/api/v1/training-dataset-refs/resolve",
        response_model=TrainingRefResolveResponse,
    )
    async def resolve_training_dataset_ref(
        body: TrainingRefResolveRequest,
    ) -> TrainingRefResolveResponse:
        return app.state.dataset_imports.resolve_training_ref(body)

    @app.get(
        "/api/v1/dataset-artifacts/{artifact_id}/materialized-contract",
        response_model=MaterializedDatasetBundle,
    )
    async def materialize_dataset_contract(
        artifact_id: str,
        training_ref_hash: Annotated[
            str,
            Query(alias="trainingRefHash", pattern=r"^[0-9a-f]{64}$"),
        ],
    ) -> MaterializedDatasetBundle:
        return app.state.dataset_imports.materialize_contract(
            artifact_id,
            training_ref_hash=training_ref_hash,
        )

    @app.post(
        "/api/v1/graph-dataset-handoffs/reserve",
        response_model=GraphHandoffReservation,
    )
    async def reserve_graph_dataset_handoff(
        body: GraphHandoffReserveRequest,
    ) -> GraphHandoffReservation:
        return app.state.dataset_imports.reserve_graph_handoff(body)

    @app.post(
        "/api/v1/graph-dataset-handoffs/cancel",
        response_model=GraphHandoffCancellation,
    )
    async def cancel_graph_dataset_handoff(
        body: GraphHandoffCancelRequest,
    ) -> GraphHandoffCancellation:
        app.state.dataset_imports.cancel_graph_handoff(body.token)
        return GraphHandoffCancellation()

    @app.post(
        "/api/v1/graph-dataset-handoffs/commit",
        response_model=GraphDatasetHandoffResponse,
    )
    async def commit_graph_dataset_handoff(
        body: GraphDatasetHandoffRequest,
    ) -> GraphDatasetHandoffResponse:
        response = app.state.dataset_imports.commit_graph_handoff(body)
        if (
            body.intended_use == "gfm_research"
            and response.research_compatibility is not None
            and response.research_compatibility.status == "compatible"
        ):
            try:
                compatibility = await gfm_research_gateway.register_uploaded_graph(
                    response.binding.graph_version_id
                )
            except GfmProxyError as error:
                if error.code in _RETRYABLE_RESEARCH_REGISTRATION_CODES:
                    # The immutable artifact handoff remains successful while a
                    # temporarily unavailable isolated runtime catches up.
                    logger.info(
                        "research_graph_registration_pending graph_version_id=%s code=%s",
                        response.binding.graph_version_id,
                        error.code,
                    )
                else:
                    prior = response.research_compatibility
                    blocked = ResearchGraphCompatibility.model_validate(
                        {
                            "intendedUse": "gfm_research",
                            "status": "blocked",
                            "compatibleTaskIds": [],
                            "auxiliaryCapabilities": [],
                            "blockers": [
                                *[
                                    item.model_dump(mode="json", by_alias=True)
                                    for item in prior.blockers
                                ],
                                {
                                    "code": error.code,
                                    "message": (
                                        "SocialGraph-FM Research adapter registration failed "
                                        "immutable contract validation."
                                    ),
                                },
                            ],
                            "adapterStatus": "pending_registration",
                        }
                    )
                    response = response.model_copy(
                        update={"research_compatibility": blocked}
                    )
            else:
                response = response.model_copy(
                    update={"research_compatibility": compatibility}
                )
        return response

    @app.post(
        "/api/v1/dataset-imports/inspect-local",
        response_model=TrustedLocalInspection,
    )
    async def inspect_local_dataset(
        body: TrustedLocalInspectRequest,
        request: Request,
    ) -> TrustedLocalInspection:
        return app.state.trusted_conversions.inspect_local(
            body.source_path,
            client_host=request.client.host if request.client else None,
        )

    @app.post(
        "/api/v1/dataset-imports/local-jobs/{job_id}/authorize",
        response_model=TrustedConversionJob,
    )
    async def authorize_local_dataset(
        job_id: str,
        body: TrustedConversionAuthorizeRequest,
        request: Request,
    ) -> TrustedConversionJob:
        return await app.state.trusted_conversions.authorize(
            job_id,
            body.authorization_token,
            client_host=request.client.host if request.client else None,
        )

    @app.get(
        "/api/v1/dataset-imports/local-jobs/{job_id}",
        response_model=TrustedConversionJob,
    )
    async def get_local_dataset_job(job_id: str, request: Request) -> TrustedConversionJob:
        return app.state.trusted_conversions.get_job(
            job_id,
            client_host=request.client.host if request.client else None,
        )

    @app.post(
        "/api/v1/dataset-imports/local-jobs/{job_id}/cancel",
        response_model=TrustedConversionJob,
    )
    async def cancel_local_dataset_job(job_id: str, request: Request) -> TrustedConversionJob:
        return await app.state.trusted_conversions.cancel(
            job_id,
            client_host=request.client.host if request.client else None,
        )

    app.include_router(
        build_governance_router(
            gfm_governance_gateway,
            max_bundle_bytes=runtime_settings.gfm_governance_bundle_max_bytes,
            max_expanded_bytes=runtime_settings.gfm_governance_expanded_max_bytes,
            skills=gfm_governance_skills_gateway,
        )
    )
    app.include_router(build_governance_skills_router(gfm_governance_skills_gateway))

    web_root_raw = os.environ.get("SOCIALGRAPH_WEB_CLIENT_ROOT", "").strip()
    if web_root_raw:
        try:
            web_root = Path(web_root_raw).expanduser().resolve(strict=True)
        except OSError as error:
            raise RuntimeError("SOCIALGRAPH_WEB_CLIENT_ROOT does not exist") from error
        index_file = web_root / "index.html"
        if not web_root.is_dir() or not index_file.is_file() or index_file.is_symlink():
            raise RuntimeError(
                "SOCIALGRAPH_WEB_CLIENT_ROOT must contain a regular index.html"
            )

        @app.get("/", include_in_schema=False)
        async def web_index() -> FileResponse:
            return FileResponse(index_file)

        @app.get("/{full_path:path}", include_in_schema=False)
        async def web_asset_or_spa(full_path: str) -> FileResponse:
            if full_path == "api" or full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail={"code": "API_ROUTE_NOT_FOUND"})
            try:
                candidate = (web_root / full_path).resolve(strict=True)
                candidate.relative_to(web_root)
            except (OSError, ValueError):
                return FileResponse(index_file)
            if candidate.is_file() and not candidate.is_symlink():
                return FileResponse(candidate)
            return FileResponse(index_file)
    return app


def _module_default_settings() -> Settings:
    settings = get_settings()
    # Importing ``app.main`` during pytest collection must never open the
    # developer's formal store before fixtures have a chance to run.
    if "pytest" in sys.modules:
        isolated = os.path.join(
            tempfile.gettempdir(),
            "socialgraph-fm-api-tests",
            str(os.getpid()),
        )
        return settings.model_copy(update={"dataset_storage_root": isolated})
    return settings


app = create_app(_module_default_settings())
