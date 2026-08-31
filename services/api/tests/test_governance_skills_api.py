from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.gfm_client import GfmProxyError
from app.gfm_hashing import canonical_sha256
from app.gfm_governance_schemas import GovernancePreviewQuery
from app.governance_skills import GovernanceSkillsGateway
from app.governance_skills_schemas import (
    ASSISTANT_CATALOG_SCHEMA_VERSION,
    ASSISTANT_REQUEST_SCHEMA_VERSION,
    ASSISTANT_RESULT_SCHEMA_VERSION,
    AssistantSkillExecuteRequest,
    GOVERNANCE_RESULT_SCHEMA_VERSION,
    SKILL_SCHEMA_VERSION,
    SimilarCasesSearchRequest,
    SkillConfirmationRequest,
    SkillExecuteRequest,
)
from app.governance_skills_store import GovernanceSkillsStore
from app.gfm_governance_routes import _preview_projection
from app.main import create_app
from app.provider import ProviderFailure

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64

PUBLIC_SKILLS = [
    "inspect_graph",
    "run_governance_analysis",
    "get_evidence_subgraph",
    "discover_coordination_groups",
    "rank_coordination_relations",
    "retrieve_similar_cases",
    "get_model_dataset_cards",
    "draft_review_report",
]


class FakeRunStatus:
    def model_dump(self, **_: Any) -> dict[str, Any]:
        return {"schemaVersion": "socialgraph-fm.gfm-governance/2.0", "runId": "created"}


class FakeGovernance:
    def __init__(self) -> None:
        self.inbox = SimpleNamespace(
            get=lambda _artifact_id: SimpleNamespace(
                dataset_content_hash=HASH_A,
                graph_version_hash=HASH_B,
            )
        )
        self.created = 0

    async def capabilities(self) -> Any:
        return SimpleNamespace(model_version_id="model-v1", model_state_hash=HASH_C)

    async def create_run(self, _request: Any) -> FakeRunStatus:
        self.created += 1
        return FakeRunStatus()

    async def result(self, run_id: str) -> Any:
        return SimpleNamespace(
            run_id=run_id,
            result_hash=HASH_A,
            artifact_id="governance-artifact-" + "1" * 32,
            dataset_content_hash=HASH_A,
            graph_version_hash=HASH_B,
            model_version_id="model-v1",
            model_state_hash=HASH_C,
        )


class FakeSkillsClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute_governance_skill(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(payload)
        execution_request = {
            "schemaVersion": "socialgraph-fm.gfm-governance/2.0",
            "protocol": "global",
            "artifactId": "governance-artifact-" + "1" * 32,
            "datasetContentHash": HASH_A,
            "graphVersionHash": HASH_B,
            "modelVersionId": "model-v1",
            "modelStateHash": HASH_C,
            "topK": 25,
        }
        return {
            "schemaVersion": GOVERNANCE_RESULT_SCHEMA_VERSION,
            "commandId": payload["commandId"],
            "command": payload["command"],
            "status": "confirmation_required",
            "graph": payload["graph"],
            "model": payload["model"],
            "result": {
                "confirmationPlan": {
                    "action": "run_governance_analysis",
                    "requiresConfirmation": True,
                    "requestDigest": canonical_sha256(execution_request),
                    "steps": ["validate", "materialize", "enqueue", "review"],
                    "estimatedScope": {
                        "nodeCount": 10,
                        "relationRowCount": 20,
                        "topK": 25,
                    },
                    "executionRequest": execution_request,
                }
            },
            "provenance": {
                "generatedAt": "2026-08-18T00:00:00Z",
                "implementationVersion": "test",
                "inputHash": canonical_sha256(payload),
            },
            "warnings": [],
        }

    async def index_governance_case(
        self, _payload: dict[str, Any]
    ) -> dict[str, Any]:
        raise AssertionError("not used")


def _request() -> SkillExecuteRequest:
    return SkillExecuteRequest(
        schemaVersion=SKILL_SCHEMA_VERSION,
        skill="run_governance_analysis",
        graph={
            "artifactId": "governance-artifact-" + "1" * 32,
            "datasetContentHash": HASH_A,
            "graphVersionHash": HASH_B,
        },
        model={"modelVersionId": "model-v1", "modelStateHash": HASH_C},
        params={"protocol": "global", "topK": 25},
    )


def test_catalog_route_exposes_only_frozen_public_names(tmp_path: Path) -> None:
    settings = Settings(dataset_storage_root=str(tmp_path / "datasets"))
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v2/gfm/governance/skills")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schemaVersion"] == SKILL_SCHEMA_VERSION
    assert [item["name"] for item in payload["items"]] == PUBLIC_SKILLS
    assert len({item["name"] for item in payload["items"]}) == 8
    assert "trace_evidence" not in response.text


def test_run_skill_requires_one_time_confirmation_before_create(tmp_path: Path) -> None:
    async def exercise() -> None:
        governance = FakeGovernance()
        client = FakeSkillsClient()
        gateway = GovernanceSkillsGateway(
            client,
            governance=governance,  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
            confirmation_ttl_seconds=60,
        )

        prepared = await gateway.execute(_request())

        assert prepared.status == "confirmation_required"
        assert prepared.confirmation is not None
        assert governance.created == 0
        assert len(client.calls) == 1

        confirmed = await gateway.confirm(
            SkillConfirmationRequest(
                schemaVersion=SKILL_SCHEMA_VERSION,
                token=prepared.confirmation.token,
            )
        )
        assert confirmed.action == "run_governance_analysis"
        assert governance.created == 1

        with pytest.raises(GfmProxyError) as reused:
            await gateway.confirm(
                SkillConfirmationRequest(
                    schemaVersion=SKILL_SCHEMA_VERSION,
                    token=prepared.confirmation.token,
                )
            )
        assert reused.value.code == "GOVERNANCE_CONFIRMATION_ALREADY_USED"
        assert governance.created == 1

    asyncio.run(exercise())
class _DispatchClient:
    def __init__(
        self,
        *,
        knowledge_text: str | None = "模型适用范围需要结合输入合同与人工复核。",
        inspection_override: dict[str, Any] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.knowledge_text = knowledge_text
        self.inspection_override = inspection_override

    async def execute_governance_skill(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        command = payload["command"]
        run_id = payload["params"].get("runId")
        if command == "inspect_graph":
            result = {
                "nodeCount": 12,
                "fusedEdgeCount": 18,
                "componentCount": 2,
                "isolateCount": 1,
                "modalities": ["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"],
                "relationCounts": {
                    "coRT": 4,
                    "coURL": 3,
                    "hashSeq": 2,
                    "fastRT": 1,
                    "tweetSim": 8,
                },
                "scopeNodeIds": [],
                "runId": run_id,
                "distribution": {
                    "low": 8,
                    "review": 2,
                    "high": 2,
                    "predictedPositive": 2,
                    "total": 12,
                },
                "topCandidates": [
                    {
                        "nodeId": f"node-{index}",
                        "label": f"Anonymous account {index}",
                        "score": round(0.9 - index * 0.05, 2),
                        "rank": index + 1,
                        "riskBand": "high" if index < 2 else "review",
                        "structureMissing": False,
                        "communityId": "group-1",
                    }
                    for index in range(5)
                ],
                "candidateLimit": 5,
            }
            if self.inspection_override is not None:
                result.update(self.inspection_override)
            result["inspectionHash"] = canonical_sha256(result)
        elif command == "discover_coordination_groups":
            result = {
                "schemaVersion": "socialgraph-fm.gfm-governance/2.0",
                "runId": run_id,
                "items": [
                    {
                        "groupId": f"group-{index}",
                        "memberCount": index + 2,
                        "priority": round(0.8 - index * 0.1, 2),
                    }
                    for index in range(1, 4)
                ],
                "total": 3,
                "offset": 0,
                "limit": 3,
            }
            result["pageHash"] = canonical_sha256(result)
        elif command == "rank_coordination_relations":
            relation_kind = payload["params"].get("relationKind", "factual")
            factual = relation_kind == "factual"
            result = {
                "runId": run_id,
                "items": [
                    {
                        "id": f"{'relation' if factual else 'link'}-{index}",
                        "kind": "factual_relation" if factual else "potential_link",
                        "nodeIds": [f"node-{index}", f"node-{index + 1}"],
                        "priority": round(0.7 - index * 0.1, 2),
                        "modalities": ["coRT"] if factual else [],
                        "factual": factual,
                        "scoreComponents": (
                            {"endpointRisk": 0.6}
                            if factual
                            else {"cosineSimilarity": 0.8, "jaccard": 0.2}
                        ),
                    }
                    for index in range(1, 4)
                ],
                "total": 3,
                "offset": 0,
                "limit": 3,
                "relationKind": relation_kind,
                "modalities": payload["params"].get("modalities", []),
            }
            result["pageHash"] = canonical_sha256(result)
        elif command == "get_evidence_subgraph":
            result = {
                "runId": run_id,
                "node": {"nodeId": payload["params"]["nodeId"], "score": 0.9},
                "neighbors": [{"nodeId": "node-1", "modalities": ["coRT"]}],
                "structuralSignals": {
                    "fusedDegree": 3,
                    "twoHopNodeCount": 7,
                    "relationNeighborCounts": {"coRT": 2, "coURL": 1},
                },
                "truncated": False,
            }
            result["evidenceHash"] = canonical_sha256(result)
        elif command == "get_model_dataset_cards":
            result = {
                "modelCard": {"method": "Global", "scope": "analyst ranking"},
                "datasetCard": {"modalities": ["coRT", "coURL"]},
                "inputContractCard": {"facts": "uploaded package only"},
            }
            result["cardHash"] = canonical_sha256(result)
        elif command == "search_knowledge":
            items = []
            if self.knowledge_text is not None:
                items.append(
                    {
                        "sourceLabel": "治理方法说明",
                        "sourceUri": "knowledge://governance/method",
                        "contentHash": HASH_A,
                        "chunkHash": HASH_B,
                        "text": self.knowledge_text,
                        "rank": 1,
                    }
                )
            result = {
                "items": items,
                "indexHash": HASH_C,
            }
            result["searchHash"] = canonical_sha256(result)
        else:
            raise AssertionError(command)
        return {
            "schemaVersion": GOVERNANCE_RESULT_SCHEMA_VERSION,
            "commandId": payload["commandId"],
            "command": command,
            "status": "completed",
            "graph": payload["graph"],
            "model": payload["model"],
            "result": result,
            "provenance": {
                "generatedAt": "2026-08-19T00:00:00Z",
                "implementationVersion": "test",
                "inputHash": canonical_sha256(payload),
            },
            "warnings": [],
        }

    async def index_governance_case(self, _payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("not used")


def test_skill_audit_validation_rejects_tampering(tmp_path: Path) -> None:
    store = GovernanceSkillsStore(tmp_path)
    store.append_audit(
        kind="skill",
        subject_id="governance-exec-" + "1" * 32,
        request_hash=HASH_A,
        response_hash=HASH_B,
        status="completed",
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP TRIGGER no_skill_audit_events_update")
        connection.execute(
            "UPDATE skill_audit_events SET status = 'failed' WHERE sequence = 1"
        )

    with pytest.raises(GfmProxyError) as invalid:
        store.validate()
    assert invalid.value.code == "GOVERNANCE_SKILL_AUDIT_INVALID"


class _AssistantProvider:
    model = "assistant-test"

    def __init__(self, *, fail_answer: bool = False, invent_hash: bool = False) -> None:
        self.fail_answer = fail_answer
        self.invent_hash = invent_hash
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    async def generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.system_prompts.append(system_prompt)
        self.prompts.append(user_prompt)
        if "toolCalls" in system_prompt:
            return {
                "toolCalls": [
                    {
                        "skill": "get_evidence_subgraph",
                        "params": {
                            "runId": "governance-" + "2" * 32,
                            "nodeId": "node-1",
                        },
                    }
                ]
            }
        if self.fail_answer:
            raise ProviderFailure("LLM_UPSTREAM_ERROR", "provider unavailable", retryable=True)
        if self.invent_hash:
            return {"answer": f"错误引用：{'d' * 64}"}
        if "全局态势报告" in system_prompt:
            return {"answer": "## 全局态势报告\n\n已根据受控证据形成态势摘要。"}
        return {"answer": "The bounded evidence should be reviewed by an analyst."}


def _assistant_request(
    skill: str = "answer_governance_question",
) -> AssistantSkillExecuteRequest:
    context: dict[str, Any] = {
        "runId": "governance-" + "2" * 32,
        "selectedTarget": {"targetType": "node", "targetId": "node-1"},
    }
    if skill == "generate_case_review_draft":
        context.update({"caseId": "case-" + "3" * 32, "caseHash": HASH_A})
    return AssistantSkillExecuteRequest.model_validate(
        {
            "schemaVersion": ASSISTANT_REQUEST_SCHEMA_VERSION,
            "skill": skill,
            "message": "What supports this finding?",
            "graph": {
                "artifactId": "governance-artifact-" + "1" * 32,
                "datasetContentHash": HASH_A,
                "graphVersionHash": HASH_B,
            },
            "model": {"modelVersionId": "model-v1", "modelStateHash": HASH_C},
            "context": context,
        }
    )


def test_assistant_catalog_exposes_six_read_only_skills_and_old_routes_are_gone(
    tmp_path: Path,
) -> None:
    settings = Settings(dataset_storage_root=str(tmp_path / "datasets"))
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v2/gfm/governance/assistant/skills")
        paths = client.app.openapi()["paths"]

    assert response.status_code == 200
    payload = response.json()
    assert payload["schemaVersion"] == ASSISTANT_CATALOG_SCHEMA_VERSION
    assert [item["name"] for item in payload["items"]] == [
        "answer_governance_question",
        "summarize_node_evidence",
        "generate_global_situation_report",
        "generate_account_evidence_report",
        "generate_coordination_report",
        "generate_case_review_draft",
    ]
    assert all(
        item["readOnly"] is True and item["confirmationRequired"] is False
        for item in payload["items"]
    )
    assert "/api/v2/gfm/governance/assistant/execute" in paths
    assert "/api/v2/gfm/governance/assistant/turn" not in paths
    assert "/api/v2/gfm/governance/assistant/dispatch" not in paths


def test_fixed_assistant_plans_match_the_catalog_call_chains() -> None:
    catalog = GovernanceSkillsGateway.assistant_catalog()
    by_name = {item.name: item for item in catalog.items}
    for skill in (
        "summarize_node_evidence",
        "generate_global_situation_report",
        "generate_account_evidence_report",
        "generate_coordination_report",
        "generate_case_review_draft",
    ):
        request = _assistant_request(skill)
        plan = GovernanceSkillsGateway._fixed_assistant_plan(
            request,
            run_id=request.context.run_id or "governance-" + "2" * 32,
            node_id="node-1",
        )
        actual = tuple(dict.fromkeys(item[0] for item in plan))
        assert actual == by_name[request.skill].governance_skills


def test_assistant_execute_route_fails_when_llm_is_not_configured(
    tmp_path: Path,
) -> None:
    settings = Settings(dataset_storage_root=str(tmp_path / "datasets"))
    payload = _assistant_request().model_dump(mode="json", by_alias=True)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v2/gfm/governance/assistant/execute", json=payload
        )

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "LLM_NOT_CONFIGURED"}}


def test_assistant_uses_bounded_read_only_context(tmp_path: Path) -> None:
    async def exercise() -> None:
        client = _DispatchClient()
        provider = _AssistantProvider()
        gateway = GovernanceSkillsGateway(
            client,
            governance=FakeGovernance(),  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
            provider=provider,
        )
        response = await gateway.assistant_execute(_assistant_request())

        assert response.schema_version == ASSISTANT_RESULT_SCHEMA_VERSION
        assert len(response.skill_calls) == 1
        assert len(client.calls) == 2
        assert len(provider.prompts) == 2
        assert '"nodes"' not in provider.prompts[1]
        assert '"edges"' not in provider.prompts[1]
        assert all(call["command"] != "run_governance_analysis" for call in client.calls)
        assert gateway.governance.created == 0
        serialized = response.model_dump(mode="json", by_alias=True)
        assert "deterministicFallback" not in serialized
        assert "fallbackPhase" not in serialized
        assert "generationMode" not in serialized

    asyncio.run(exercise())


def test_assistant_provider_failure_is_explicit_and_has_no_fallback_response(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        gateway = GovernanceSkillsGateway(
            _DispatchClient(),
            governance=FakeGovernance(),  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
            provider=_AssistantProvider(fail_answer=True),
        )
        with pytest.raises(ProviderFailure) as failed:
            await gateway.assistant_execute(_assistant_request())
        assert failed.value.code == "LLM_UPSTREAM_ERROR"
        assert gateway.governance.created == 0

    asyncio.run(exercise())


def test_assistant_repairs_once_then_rejects_an_invented_hash(tmp_path: Path) -> None:
    async def exercise() -> None:
        provider = _AssistantProvider(invent_hash=True)
        gateway = GovernanceSkillsGateway(
            _DispatchClient(),
            governance=FakeGovernance(),  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
            provider=provider,
        )
        with pytest.raises(ProviderFailure) as failed:
            await gateway.assistant_execute(_assistant_request())
        assert failed.value.code == "LLM_INVALID_RESPONSE"
        assert len(provider.prompts) == 3
        assert "上一响应不符合既定 JSON 结构" in provider.prompts[-1]

    asyncio.run(exercise())


def test_global_report_runs_fixed_read_only_plan_and_requires_llm(tmp_path: Path) -> None:
    async def exercise() -> None:
        client = _DispatchClient()
        gateway = GovernanceSkillsGateway(
            client,
            governance=FakeGovernance(),  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
            provider=_AssistantProvider(),
        )
        response = await gateway.assistant_execute(
            _assistant_request("generate_global_situation_report")
        )
        assert response.answer.startswith("## 全局态势报告")
        assert [call["command"] for call in client.calls] == [
            "inspect_graph",
            "discover_coordination_groups",
            "rank_coordination_relations",
            "rank_coordination_relations",
        ]
        assert len(response.skill_calls) == 4

        unavailable = GovernanceSkillsGateway(
            client,
            governance=FakeGovernance(),  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path / "missing"),
            provider=None,
        )
        with pytest.raises(ProviderFailure) as failed:
            await unavailable.assistant_execute(
                _assistant_request("generate_global_situation_report")
            )
        assert failed.value.code == "LLM_NOT_CONFIGURED"

    asyncio.run(exercise())


def test_preview_query_preserves_repeated_anchors_and_rejects_unscoped_options() -> None:
    query = _preview_projection(
        preset="evidence",
        node_budget=20,
        edge_budget=30,
        relation=None,
        anchor_node_ids=["node-1", "node-2"],
        group_budget=None,
    )
    assert query == GovernancePreviewQuery(
        preset="evidence",
        nodeBudget=20,
        edgeBudget=30,
        anchorNodeIds=["node-1", "node-2"],
    )
    with pytest.raises(Exception) as invalid:
        _preview_projection(
            preset=None,
            node_budget=20,
            edge_budget=None,
            relation=None,
            anchor_node_ids=None,
            group_budget=None,
        )
    assert getattr(invalid.value, "status_code", None) == 422


class FakeIndexGovernance(FakeGovernance):
    def __init__(self) -> None:
        super().__init__()
        self.governance = SimpleNamespace(
            case_state_timeline=lambda _case_id: [
                {"state": "concluded", "createdAt": "2026-08-18T00:00:00Z"}
            ]
        )
        self._case = SimpleNamespace(
            case_id="case-" + "3" * 32,
            case_hash=HASH_A,
            run_id="governance-" + "2" * 32,
            state="concluded",
            items=(
                SimpleNamespace(target_type="group", target_id="group-2"),
                SimpleNamespace(target_type="node", target_id="node-2"),
                SimpleNamespace(target_type="relation", target_id="relation-1"),
                SimpleNamespace(target_type="node", target_id="node-1"),
            ),
            review_events=(),
            current_decisions={},
        )

    def case(self, _case_id: str) -> Any:
        return self._case


class FakeIndexClient:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    async def index_governance_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payload = payload
        result = {
            "caseId": payload["params"]["caseId"],
            "recordHash": HASH_A,
            "indexHash": HASH_B,
            "indexedAt": "2026-08-18T00:00:00Z",
            "idempotent": False,
        }
        return {
            "schemaVersion": GOVERNANCE_RESULT_SCHEMA_VERSION,
            "commandId": payload["commandId"],
            "command": "index_case",
            "status": "completed",
            "graph": payload["graph"],
            "model": payload["model"],
            "result": result,
            "provenance": {
                "generatedAt": "2026-08-18T00:00:00Z",
                "implementationVersion": "test",
                "inputHash": canonical_sha256(payload),
            },
            "warnings": [],
        }

    async def execute_governance_skill(
        self, _payload: dict[str, Any]
    ) -> dict[str, Any]:
        raise AssertionError("not used")


class BlockingIndexClient(FakeIndexClient):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def index_governance_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        self.entered.set()
        await self.release.wait()
        return await super().index_governance_case(payload)


class RecoveringIndexClient(FakeIndexClient):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    async def index_governance_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        if self.call_count == 1:
            raise GfmProxyError(503, "GFM_GOVERNANCE_SERVICE_UNAVAILABLE")
        return await super().index_governance_case(payload)


class FakeSimilarCasesClient:
    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.items = items or []

    async def execute_governance_skill(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append(payload)
        params = payload["params"]
        query = (
            {"caseId": params["caseId"]}
            if params.get("caseId") is not None
            else {
                "runId": params["runId"],
                "kindKey": "+".join(item["kind"] for item in params["kindEntries"]),
                "kindEntries": params["kindEntries"],
            }
        )
        result = {
            "query": query,
            "items": self.items,
            "weights": {"embedding": 0.7, "structure": 0.2, "modality": 0.1},
            "indexHash": HASH_B,
        }
        result["retrievalHash"] = canonical_sha256(result)
        return {
            "schemaVersion": GOVERNANCE_RESULT_SCHEMA_VERSION,
            "commandId": payload["commandId"],
            "command": payload["command"],
            "status": "completed",
            "graph": payload["graph"],
            "model": payload["model"],
            "result": result,
            "provenance": {
                "generatedAt": "2026-08-18T00:00:00Z",
                "implementationVersion": "test",
                "inputHash": canonical_sha256(payload),
            },
            "warnings": [],
        }

    async def index_governance_case(
        self, _payload: dict[str, Any]
    ) -> dict[str, Any]:
        raise AssertionError("search must not index or backfill cases")


def _similar_cases_request(**params: Any) -> SimilarCasesSearchRequest:
    return SimilarCasesSearchRequest(
        schemaVersion=SKILL_SCHEMA_VERSION,
        graph={
            "artifactId": "governance-artifact-" + "1" * 32,
            "datasetContentHash": HASH_A,
            "graphVersionHash": HASH_B,
        },
        model={"modelVersionId": "model-v1", "modelStateHash": HASH_C},
        limit=10,
        **params,
    )


def test_similar_cases_converts_active_case_to_bound_targets_without_backfill(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        governance = FakeIndexGovernance()
        governance._case.state = "active"
        client = FakeSimilarCasesClient()
        gateway = GovernanceSkillsGateway(
            client,
            governance=governance,  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
        )

        response = await gateway.similar_cases(
            _similar_cases_request(caseId=governance._case.case_id)
        )

        assert len(client.calls) == 1
        params = client.calls[0]["params"]
        assert params["caseId"] is None
        assert params["runId"] == governance._case.run_id
        assert params["kindEntries"] == [
            {"kind": "node", "targetIds": ["node-1", "node-2"]},
            {"kind": "relation", "targetIds": ["relation-1"]},
            {"kind": "group", "targetIds": ["group-2"]},
        ]
        assert response.query["runId"] == governance._case.run_id
        assert response.backfill == {"attempted": 0, "succeeded": 0, "failed": 0}

    asyncio.run(exercise())


def test_similar_cases_uses_indexed_concluded_case_for_self_exclusion(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        governance = FakeIndexGovernance()
        client = FakeSimilarCasesClient()
        store = GovernanceSkillsStore(tmp_path)
        pending_hash = store.append_index_event(
            case_id=governance._case.case_id,
            case_hash=governance._case.case_hash,
            status="pending",
        )
        store.append_index_event(
            case_id=governance._case.case_id,
            case_hash=governance._case.case_hash,
            status="succeeded",
            index_hash=HASH_B,
            expected_pending_hash=pending_hash,
        )
        gateway = GovernanceSkillsGateway(
            client,
            governance=governance,  # type: ignore[arg-type]
            store=store,
        )

        response = await gateway.similar_cases(
            _similar_cases_request(caseId=governance._case.case_id)
        )

        assert client.calls[0]["params"]["caseId"] == governance._case.case_id
        assert client.calls[0]["params"]["runId"] is None
        assert response.query == {"caseId": governance._case.case_id}

    asyncio.run(exercise())


def test_similar_cases_rejects_empty_case_with_specific_precondition(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        governance = FakeIndexGovernance()
        governance._case.state = "active"
        governance._case.items = ()
        client = FakeSimilarCasesClient()
        gateway = GovernanceSkillsGateway(
            client,
            governance=governance,  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
        )

        with pytest.raises(GfmProxyError) as invalid:
            await gateway.similar_cases(
                _similar_cases_request(caseId=governance._case.case_id)
            )

        assert invalid.value.status_code == 409
        assert invalid.value.code == "GOVERNANCE_SIMILAR_CASE_TARGETS_REQUIRED"
        assert client.calls == []

    asyncio.run(exercise())


def test_similar_cases_rejects_unindexed_concluded_case_as_not_ready(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        governance = FakeIndexGovernance()
        client = FakeSimilarCasesClient()
        gateway = GovernanceSkillsGateway(
            client,
            governance=governance,  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
        )

        with pytest.raises(GfmProxyError) as invalid:
            await gateway.similar_cases(
                _similar_cases_request(caseId=governance._case.case_id)
            )

        assert invalid.value.status_code == 409
        assert invalid.value.code == "GOVERNANCE_SIMILAR_CASE_INDEX_NOT_READY"
        assert client.calls == []

    asyncio.run(exercise())


def test_similar_cases_current_target_query_succeeds_without_case_lookup_or_backfill(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        governance = FakeIndexGovernance()
        governance.case = lambda _case_id: (_ for _ in ()).throw(AssertionError("not used"))
        client = FakeSimilarCasesClient()
        gateway = GovernanceSkillsGateway(
            client,
            governance=governance,  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
        )

        response = await gateway.similar_cases(
            _similar_cases_request(
                runId=governance._case.run_id,
                kindEntries=[{"kind": "node", "targetIds": ["node-1"]}],
            )
        )

        assert response.query == {
            "runId": governance._case.run_id,
            "kindKey": "node",
            "kindEntries": [{"kind": "node", "targetIds": ["node-1"]}],
        }
        assert response.backfill == {"attempted": 0, "succeeded": 0, "failed": 0}
        assert len(client.calls) == 1

    asyncio.run(exercise())


def test_similar_cases_returns_manifest_verified_seed_case_without_mutable_api_record(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        governance = FakeIndexGovernance()
        bundled = {
            "caseId": "case-" + "b" * 32,
            "score": 0.9,
            "components": {"embedding": 0.9, "structure": 0.8, "modality": 0.7},
            "graphVersionHash": HASH_B,
            "modelStateHash": HASH_C,
            "kindKey": "node",
            "kindEntries": [{"kind": "node", "targetIds": ["node-9"]}],
            "concludedAt": "2026-08-18T00:00:00Z",
            "recordHash": HASH_A,
        }
        local_case = governance.case
        governance.case = lambda case_id: (
            local_case(case_id)
            if case_id == governance._case.case_id
            else (_ for _ in ()).throw(
                GfmProxyError(404, "GOVERNANCE_CASE_NOT_FOUND")
            )
        )
        client = FakeSimilarCasesClient([bundled])
        gateway = GovernanceSkillsGateway(
            client,
            governance=governance,  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
        )

        response = await gateway.similar_cases(
            _similar_cases_request(
                runId=governance._case.run_id,
                kindEntries=[{"kind": "node", "targetIds": ["node-1"]}],
            )
        )

        assert list(response.items) == [bundled]

    asyncio.run(exercise())


def test_case_index_backfill_skips_concluded_cases_without_targets(tmp_path: Path) -> None:
    async def exercise() -> None:
        governance = FakeIndexGovernance()
        governance._case.items = ()
        governance.cases = lambda **_kwargs: SimpleNamespace(
            items=(governance._case,), total=1
        )
        client = FakeSimilarCasesClient()
        gateway = GovernanceSkillsGateway(
            client,
            governance=governance,  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
        )

        assert await gateway.backfill_concluded_cases() == {
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
        }
        assert client.calls == []

    asyncio.run(exercise())


def test_case_index_concurrent_requests_share_owned_attempt_and_survive_reopen(
    tmp_path: Path,
) -> None:
    store_root = tmp_path / "gfm" / "governance"

    async def exercise() -> None:
        client = BlockingIndexClient()
        store = GovernanceSkillsStore(store_root)
        gateway = GovernanceSkillsGateway(
            client,
            governance=FakeIndexGovernance(),  # type: ignore[arg-type]
            store=store,
        )
        case_id = "case-" + "3" * 32

        first = asyncio.create_task(gateway.ensure_case_indexed(case_id))
        await client.entered.wait()
        second = asyncio.create_task(gateway.ensure_case_indexed(case_id))
        await asyncio.sleep(0)
        client.release.set()

        assert await asyncio.gather(first, second) == [True, True]
        with sqlite3.connect(store.database_path) as connection:
            statuses = [
                str(row[0])
                for row in connection.execute(
                    "SELECT status FROM case_index_events ORDER BY sequence"
                )
            ]
        assert statuses == ["pending", "succeeded"]
        assert client.call_count == 1

    asyncio.run(exercise())

    reopened = GovernanceSkillsStore(store_root)
    assert reopened.validate()["valid"] is True
    settings = Settings(dataset_storage_root=str(tmp_path / "datasets"))
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v2/gfm/governance/skill-audit/validation")
    assert response.status_code == 200
    assert response.json()["valid"] is True


def test_case_index_concurrent_gateways_share_sqlite_attempt_and_terminal(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        client = BlockingIndexClient()
        first_gateway = GovernanceSkillsGateway(
            client,
            governance=FakeIndexGovernance(),  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
        )
        second_gateway = GovernanceSkillsGateway(
            client,
            governance=FakeIndexGovernance(),  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
        )
        case_id = "case-" + "3" * 32

        first = asyncio.create_task(first_gateway.ensure_case_indexed(case_id))
        await client.entered.wait()
        second = asyncio.create_task(second_gateway.ensure_case_indexed(case_id))
        await asyncio.sleep(0)
        calls_while_first_is_pending = client.call_count
        client.release.set()

        assert await asyncio.gather(first, second) == [True, True]
        assert calls_while_first_is_pending == 1
        assert client.call_count == 1
        with sqlite3.connect(first_gateway.store.database_path) as connection:
            statuses = connection.execute(
                "SELECT status FROM case_index_events ORDER BY sequence"
            ).fetchall()
        assert statuses == [("pending",), ("succeeded",)]

    asyncio.run(exercise())


def test_case_index_terminal_event_must_own_open_attempt(tmp_path: Path) -> None:
    store = GovernanceSkillsStore(tmp_path)
    case_id = "case-" + "3" * 32
    pending_hash = store.append_index_event(
        case_id=case_id,
        case_hash=HASH_A,
        status="pending",
    )

    with pytest.raises(GfmProxyError) as conflict:
        store.append_index_event(
            case_id=case_id,
            case_hash=HASH_A,
            status="succeeded",
            index_hash=HASH_B,
            expected_pending_hash=HASH_C,
        )
    assert conflict.value.code == "GOVERNANCE_CASE_INDEX_STATE_CONFLICT"
    current = store.index_status(case_id)
    assert current is not None
    assert current["status"] == "pending"

    store.append_index_event(
        case_id=case_id,
        case_hash=HASH_A,
        status="succeeded",
        index_hash=HASH_B,
        expected_pending_hash=pending_hash,
    )
    with pytest.raises(GfmProxyError) as duplicate:
        store.append_index_event(
            case_id=case_id,
            case_hash=HASH_A,
            status="succeeded",
            index_hash=HASH_B,
            expected_pending_hash=pending_hash,
        )
    assert duplicate.value.code == "GOVERNANCE_CASE_INDEX_STATE_CONFLICT"
    assert GovernanceSkillsStore(tmp_path).validate()["valid"] is True


def test_case_index_validation_rejects_deletion_of_entire_case_chain(tmp_path: Path) -> None:
    store = GovernanceSkillsStore(tmp_path)
    case_id = "case-" + "3" * 32
    pending_hash = store.append_index_event(
        case_id=case_id,
        case_hash=HASH_A,
        status="pending",
    )
    store.append_index_event(
        case_id=case_id,
        case_hash=HASH_A,
        status="succeeded",
        index_hash=HASH_B,
        expected_pending_hash=pending_hash,
    )

    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP TRIGGER no_case_index_events_delete")
        connection.execute("DELETE FROM case_index_events WHERE case_id = ?", (case_id,))

    with pytest.raises(GfmProxyError) as invalid:
        store.validate()
    assert invalid.value.code == "GOVERNANCE_SKILL_AUDIT_INVALID"


def test_case_index_integrity_migrates_existing_local_database(tmp_path: Path) -> None:
    store = GovernanceSkillsStore(tmp_path)
    case_id = "case-" + "3" * 32
    pending_hash = store.append_index_event(
        case_id=case_id,
        case_hash=HASH_A,
        status="pending",
    )
    store.append_index_event(
        case_id=case_id,
        case_hash=HASH_A,
        status="succeeded",
        index_hash=HASH_B,
        expected_pending_hash=pending_hash,
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP TABLE case_index_integrity_events")
        connection.execute("DROP TABLE case_index_claims")

    validation = GovernanceSkillsStore(tmp_path).validate()

    assert validation["valid"] is True
    assert validation["caseIndexEventCount"] == 2
    assert len(validation["caseIndexHeadHash"]) == 64


def test_case_index_failed_attempt_remains_retryable(tmp_path: Path) -> None:
    async def exercise() -> None:
        client = RecoveringIndexClient()
        store = GovernanceSkillsStore(tmp_path)
        gateway = GovernanceSkillsGateway(
            client,
            governance=FakeIndexGovernance(),  # type: ignore[arg-type]
            store=store,
        )
        case_id = "case-" + "3" * 32

        assert await gateway.ensure_case_indexed(case_id) is False
        assert await gateway.ensure_case_indexed(case_id) is True
        with sqlite3.connect(store.database_path) as connection:
            events = connection.execute(
                "SELECT status, error_code FROM case_index_events ORDER BY sequence"
            ).fetchall()
        assert events == [
            ("pending", None),
            ("failed", "GFM_GOVERNANCE_SERVICE_UNAVAILABLE"),
            ("pending", None),
            ("succeeded", None),
        ]

    asyncio.run(exercise())
    assert GovernanceSkillsStore(tmp_path).validate()["valid"] is True


def test_case_index_reclaims_interrupted_pending_attempt(tmp_path: Path) -> None:
    case_id = "case-" + "3" * 32
    store = GovernanceSkillsStore(tmp_path)
    store.append_index_event(case_id=case_id, case_hash=HASH_A, status="pending")

    async def exercise() -> None:
        reopened = GovernanceSkillsStore(tmp_path)
        gateway = GovernanceSkillsGateway(
            FakeIndexClient(),
            governance=FakeIndexGovernance(),  # type: ignore[arg-type]
            store=reopened,
        )
        assert await gateway.ensure_case_indexed(case_id) is True
        with sqlite3.connect(reopened.database_path) as connection:
            events = connection.execute(
                "SELECT status, error_code FROM case_index_events ORDER BY sequence"
            ).fetchall()
        assert events == [
            ("pending", None),
            ("failed", "GOVERNANCE_CASE_INDEX_INTERRUPTED"),
            ("pending", None),
            ("succeeded", None),
        ]

    asyncio.run(exercise())
    assert GovernanceSkillsStore(tmp_path).validate()["valid"] is True


def test_case_index_uses_canonical_typed_entries(tmp_path: Path) -> None:
    async def exercise() -> None:
        client = FakeIndexClient()
        gateway = GovernanceSkillsGateway(
            client,
            governance=FakeIndexGovernance(),  # type: ignore[arg-type]
            store=GovernanceSkillsStore(tmp_path),
        )
        assert await gateway.ensure_case_indexed("case-" + "3" * 32) is True
        assert client.payload is not None
        assert client.payload["params"]["kindEntries"] == [
            {"kind": "node", "targetIds": ["node-1", "node-2"]},
            {"kind": "relation", "targetIds": ["relation-1"]},
            {"kind": "group", "targetIds": ["group-2"]},
        ]

    asyncio.run(exercise())


class FakeDraftClient:
    async def execute_governance_skill(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        assert payload["command"] == "draft_review_report"
        factual = {
            "format": payload["params"]["format"],
            "content": "# Factual draft\n\nScore facts remain immutable.\n",
            "caseId": payload["params"]["caseId"],
            "citedHashes": [payload["params"]["caseHash"], payload["params"]["resultHash"]],
            "generatedWithoutLlm": True,
        }
        factual["draftHash"] = canonical_sha256(factual)
        return {
            "schemaVersion": GOVERNANCE_RESULT_SCHEMA_VERSION,
            "commandId": payload["commandId"],
            "command": payload["command"],
            "status": "completed",
            "graph": payload["graph"],
            "model": payload["model"],
            "result": factual,
            "provenance": {
                "generatedAt": "2026-08-18T00:00:00Z",
                "implementationVersion": "test",
                "inputHash": canonical_sha256(payload),
            },
            "warnings": [],
        }

    async def index_governance_case(
        self, _payload: dict[str, Any]
    ) -> dict[str, Any]:
        raise AssertionError("not used")


class DraftProvider:
    model = "draft-model"

    def __init__(self, *, fail: bool) -> None:
        self.fail = fail

    async def generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        assert "untrusted evidence data" in system_prompt
        assert "factualDraftSummary" in user_prompt
        if self.fail:
            raise ProviderFailure("LLM_DOWN", "provider unavailable", retryable=True)
        return {"narrative": "Analyst narrative that does not alter factual values."}


@pytest.mark.parametrize("provider_fails", [False, True])
def test_draft_preserves_factual_payload_and_requires_save_confirmation(
    tmp_path: Path, provider_fails: bool
) -> None:
    async def exercise() -> None:
        store = GovernanceSkillsStore(tmp_path)
        gateway = GovernanceSkillsGateway(
            FakeDraftClient(),
            governance=FakeIndexGovernance(),  # type: ignore[arg-type]
            store=store,
            provider=DraftProvider(fail=provider_fails),
        )
        request = SkillExecuteRequest(
            schemaVersion=SKILL_SCHEMA_VERSION,
            skill="draft_review_report",
            graph={
                "artifactId": "governance-artifact-" + "1" * 32,
                "datasetContentHash": HASH_A,
                "graphVersionHash": HASH_B,
            },
            model={"modelVersionId": "model-v1", "modelStateHash": HASH_C},
            params={"caseId": "case-" + "3" * 32, "format": "markdown"},
        )

        drafted = await gateway.execute(request)

        assert drafted.status == "confirmation_required"
        assert drafted.confirmation is not None
        assert drafted.result["factualDraft"]["content"].startswith("# Factual draft")
        assert drafted.result["generatedWithoutLlm"] is provider_fails
        assert (drafted.result["narrative"] is None) is provider_fails
        with sqlite3.connect(store.database_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM confirmed_reports").fetchone()[0] == 0

        saved = await gateway.confirm(
            SkillConfirmationRequest(
                schemaVersion=SKILL_SCHEMA_VERSION,
                token=drafted.confirmation.token,
            )
        )
        assert saved.action == "save_draft_report"
        assert saved.result["draft"]["factualDraft"] == drafted.result["factualDraft"]
        assert saved.result["draft"]["narrative"] == drafted.result["narrative"]

    asyncio.run(exercise())
