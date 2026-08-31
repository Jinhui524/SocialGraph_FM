"""Public versioned routes for SocialGraph-FM Governance skills and retrieval."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .gfm_client import GfmProxyError
from .gfm_governance_routes import PREFIX
from .governance_skills import GovernanceSkillsGateway
from .governance_skills_schemas import (
    AssistantSkillCatalog,
    AssistantSkillExecuteRequest,
    AssistantSkillExecutionResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    SimilarCasesSearchRequest,
    SimilarCasesSearchResponse,
    SkillCatalog,
    SkillConfirmationRequest,
    SkillConfirmationResponse,
    SkillExecuteRequest,
    SkillExecutionResponse,
    SkillInvocationRequest,
    SkillName,
)
from .provider import ProviderFailure


def _proxy(error: GfmProxyError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail={"code": error.code})


def _provider_proxy(error: ProviderFailure) -> HTTPException:
    unavailable = {
        "LLM_NOT_CONFIGURED",
        "LLM_TIMEOUT",
        "LLM_NETWORK_ERROR",
        "LLM_RATE_LIMITED",
        "LLM_UPSTREAM_ERROR",
    }
    return HTTPException(
        status_code=503 if error.retryable or error.code in unavailable else 502,
        detail={"code": error.code},
    )


def build_governance_skills_router(gateway: GovernanceSkillsGateway) -> APIRouter:
    router = APIRouter(prefix=PREFIX, tags=["SocialGraph-FM Governance skills"])

    @router.get("/skills", response_model=SkillCatalog)
    async def catalog() -> SkillCatalog:
        return gateway.catalog()

    @router.post("/skills/execute", response_model=SkillExecutionResponse)
    async def execute(body: SkillExecuteRequest) -> SkillExecutionResponse:
        try:
            return await gateway.execute(body)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post("/skills/{skill}/execute", response_model=SkillExecutionResponse)
    async def execute_named(
        skill: SkillName, body: SkillInvocationRequest
    ) -> SkillExecutionResponse:
        try:
            return await gateway.execute(body.bind(skill))
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post("/skills/confirm", response_model=SkillConfirmationResponse)
    async def confirm(body: SkillConfirmationRequest) -> SkillConfirmationResponse:
        try:
            return await gateway.confirm(body)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post("/knowledge/search", response_model=KnowledgeSearchResponse)
    async def search_knowledge(body: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
        try:
            return await gateway.search_knowledge(body)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.get("/assistant/skills", response_model=AssistantSkillCatalog)
    async def assistant_catalog() -> AssistantSkillCatalog:
        return gateway.assistant_catalog()

    @router.post("/assistant/execute", response_model=AssistantSkillExecutionResponse)
    async def assistant_execute(
        body: AssistantSkillExecuteRequest,
    ) -> AssistantSkillExecutionResponse:
        try:
            return await gateway.assistant_execute(body)
        except GfmProxyError as error:
            raise _proxy(error) from error
        except ProviderFailure as error:
            raise _provider_proxy(error) from error

    @router.post("/similar-cases/search", response_model=SimilarCasesSearchResponse)
    async def search_similar_cases(
        body: SimilarCasesSearchRequest,
    ) -> SimilarCasesSearchResponse:
        try:
            return await gateway.similar_cases(body)
        except GfmProxyError as error:
            raise _proxy(error) from error

    @router.post("/case-index/backfill")
    async def backfill_case_index() -> dict[str, object]:
        return {
            "schemaVersion": "socialgraph-fm.governance-skills/1.0",
            "result": await gateway.backfill_concluded_cases(),
        }

    @router.get("/skill-audit/validation")
    async def validate_audit() -> dict[str, object]:
        try:
            return {
                "schemaVersion": "socialgraph-fm.governance-skills/1.0",
                **gateway.store.validate(),
            }
        except GfmProxyError as error:
            raise _proxy(error) from error

    return router


__all__ = ["build_governance_skills_router"]
