from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.main import create_app

from .conftest import SequenceProvider


@pytest.mark.anyio
async def test_health_and_capabilities_without_llm(api_client: httpx.AsyncClient) -> None:
    health = await api_client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "service": "socialgraph-fm-api",
        "version": "0.1.0",
    }

    response = await api_client.get("/api/v1/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert body["intentNormalization"] == {
        "configured": False,
        "mode": "llm_required",
        "provider": None,
        "model": None,
        "apiMode": "chat_completions",
        "connectionStatus": "not_configured",
    }
    assert body["analysis"]["gfmConnected"] is False
    assert body["dataBoundary"]["sendsRawGraph"] is False


@pytest.mark.anyio
async def test_cors_allows_only_configured_frontend(api_client: httpx.AsyncClient) -> None:
    allowed = await api_client.options(
        "/api/v1/intents/normalize",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-request-id",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"

    rejected = await api_client.options(
        "/api/v1/intents/normalize",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert rejected.status_code == 400


@pytest.mark.anyio
async def test_normalization_requires_configured_llm(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.post(
        "/api/v1/intents/normalize",
        headers={"X-Request-ID": "frontend-request-1"},
        json={"text": "分析 2020 至 2024 年的核心节点和影响力"},
    )
    assert response.status_code == 503
    assert response.headers["X-Request-ID"] == "frontend-request-1"
    assert response.json() == {"detail": {"code": "LLM_NOT_CONFIGURED"}}


@pytest.mark.anyio
async def test_normal_chat_has_no_unconfigured_fallback(api_client: httpx.AsyncClient) -> None:
    response = await api_client.post("/api/v1/intents/normalize", json={"text": "你好，你能做什么？"})
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "LLM_NOT_CONFIGURED"}}


@pytest.mark.anyio
async def test_json_response_declares_utf8_and_preserves_chinese_bytes(
) -> None:
    chinese_text = "识别桥接节点"
    provider = SequenceProvider(
        [
            {
                "kind": "analysis_request",
                "normalizedText": chinese_text,
                "task": "bridge_detection",
                "targets": [],
                "confidence": 0.9,
                "filters": {},
            }
        ]
    )
    app = create_app(Settings(), provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/v1/intents/normalize", json={"text": chinese_text}
        )

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json; charset=utf-8"
    decoded = json.loads(response.content.decode("utf-8"))
    normalized_text = decoded["normalizedText"]
    assert "桥接节点" in normalized_text
    assert normalized_text.encode("utf-8") in response.content


@pytest.mark.anyio
async def test_json_validation_error_also_declares_utf8(api_client: httpx.AsyncClient) -> None:
    response = await api_client.post(
        "/api/v1/intents/normalize",
        json={"text": ""},
    )

    assert response.status_code == 422
    assert response.headers["Content-Type"] == "application/json; charset=utf-8"


@pytest.mark.anyio
@pytest.mark.parametrize("forbidden", ["nodes", "edges", "sourceFile"])
async def test_request_rejects_raw_graph_fields(
    api_client: httpx.AsyncClient,
    forbidden: str,
) -> None:
    response = await api_client.post(
        "/api/v1/intents/normalize",
        json={"text": "分析网络", forbidden: [] if forbidden != "sourceFile" else "secret.csv"},
    )
    assert response.status_code == 422


@pytest.mark.anyio
async def test_graph_context_accepts_only_allowlisted_summary(api_client: httpx.AsyncClient) -> None:
    valid_context = {
        "nodeCount": 10,
        "edgeCount": 16,
        "density": 0.35,
        "connectedComponents": 1,
        "nodeTypes": ["person", "organization"],
        "edgeTypes": ["cooperate"],
        "hasWeight": True,
        "hasTimestamp": False,
    }
    accepted = await api_client.post(
        "/api/v1/intents/normalize",
        json={"text": "给出图谱概览", "graphContext": valid_context},
    )
    assert accepted.status_code == 503
    assert accepted.json() == {"detail": {"code": "LLM_NOT_CONFIGURED"}}

    rejected = await api_client.post(
        "/api/v1/intents/normalize",
        json={
            "text": "给出图谱概览",
            "graphContext": {**valid_context, "nodes": [{"id": "sensitive"}]},
        },
    )
    assert rejected.status_code == 422


@pytest.mark.anyio
async def test_llm_result_is_sanitized_and_prompt_contains_no_raw_graph() -> None:
    provider = SequenceProvider(
        [
            {
                "kind": "analysis_request",
                "normalizedText": "计算张三与未知人物的中心性",
                "task": "centrality",
                "targets": ["张三", "不存在的人"],
                "confidence": 0.91,
                "timeRange": None,
                "filters": {"nodeType": "研究者", "query": "MATCH (n) RETURN n"},
            }
        ]
    )
    app = create_app(Settings(), provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/intents/normalize",
            json={
                "text": "分析张三的中心性",
                "graphContext": {
                    "nodeCount": 3,
                    "edgeCount": 2,
                    "density": 0.67,
                    "connectedComponents": 1,
                    "nodeTypes": ["研究者"],
                    "edgeTypes": ["合作"],
                    "hasWeight": False,
                    "hasTimestamp": False,
                },
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["targets"] == ["张三"]
    # Aggregate graph vocabularies may inform classification, but the model may
    # not copy them into an actionable filter unless the user wrote the term.
    assert body["filters"] == {}
    assert body["meta"]["source"] == "llm"
    assert body["meta"]["model"] == "test-model"
    assert "UNSUPPORTED_TARGET_DISCARDED" in body["meta"]["warnings"]
    assert "UNSUPPORTED_FILTER_DISCARDED" in body["meta"]["warnings"]
    prompt = provider.calls[0][1]
    assert "graphContextSummary" in prompt
    assert "sourceFile" not in prompt
    assert '"nodes"' not in prompt
    assert '"edges"' not in prompt


@pytest.mark.anyio
async def test_llm_view_command_is_parsed_and_all_terms_are_grounded_in_user_text() -> None:
    provider = SequenceProvider(
        [
            {
                "kind": "analysis_request",
                "normalizedText": "查看张三的两跳合作邻域",
                "task": "overview",
                "targets": ["张三", "秘密节点"],
                "confidence": 0.96,
                "timeRange": {"start": "2024", "end": "2025"},
                "filters": {"edgeType": "合作", "nodeType": "研究者"},
                "view": {
                    "mode": "local",
                    "focusTerms": ["张三", "秘密节点"],
                    "depth": 2,
                    "nodeTypeTerms": ["研究者"],
                    "edgeTypeTerms": ["合作"],
                    "layoutPreset": "balanced",
                    "overlay": "degree",
                },
            }
        ]
    )
    app = create_app(Settings(), provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/intents/normalize",
            json={
                "text": "查看张三的两跳邻居，只看合作关系",
                "graphContext": {
                    "nodeCount": 20,
                    "edgeCount": 30,
                    "density": 0.16,
                    "connectedComponents": 1,
                    "nodeTypes": ["研究者"],
                    "edgeTypes": ["合作"],
                    "hasWeight": False,
                    "hasTimestamp": True,
                    "timeRange": {"start": "2024", "end": "2025"},
                },
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["schemaVersion"] == "1.1"
    assert body["targets"] == ["张三"]
    assert body["filters"] == {"edgeType": "合作"}
    assert body["timeRange"] is None
    assert body["view"] == {
        "mode": "local",
        "focusTerms": ["张三"],
        "depth": 2,
        "nodeTypeTerms": [],
        "edgeTypeTerms": ["合作"],
        "layoutPreset": "balanced",
        "overlay": "degree",
    }
    assert "UNSUPPORTED_TARGET_DISCARDED" in body["meta"]["warnings"]
    assert "UNSUPPORTED_FILTER_DISCARDED" in body["meta"]["warnings"]
    assert "UNSUPPORTED_TIME_RANGE_DISCARDED" in body["meta"]["warnings"]
    assert "UNSUPPORTED_VIEW_TERM_DISCARDED" in body["meta"]["warnings"]


@pytest.mark.anyio
async def test_schema_10_model_shape_without_view_remains_accepted() -> None:
    provider = SequenceProvider(
        [
            {
                "kind": "analysis_request",
                "normalizedText": "生成图谱概览",
                "task": "overview",
                "targets": [],
                "confidence": 0.9,
                "timeRange": None,
                "filters": {},
            }
        ]
    )
    app = create_app(Settings(), provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/api/v1/intents/normalize", json={"text": "给出图谱概览"})

    assert response.status_code == 200
    assert response.json()["view"] is None
    assert response.json()["meta"]["schemaVersion"] == "1.1"


@pytest.mark.anyio
async def test_explicit_path_command_overrides_incorrect_llm_chat_response() -> None:
    provider = SequenceProvider(
        [{"kind": "chat", "reply": "当前不支持节点间路径查询。"}]
    )
    app = create_app(Settings(), provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/intents/normalize",
            json={"text": "显示张三到李四的路径"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "analysis_request"
    assert body["task"] == "overview"
    assert body["targets"] == ["张三", "李四"]
    assert body["view"]["mode"] == "path"
    assert body["view"]["focusTerms"] == ["张三", "李四"]
    assert body["meta"]["source"] == "llm"
    assert body["meta"]["model"] == "test-model"
    assert body["meta"]["warnings"] == ["LLM_CHAT_OVERRIDDEN_BY_EXPLICIT_VIEW"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("text", "expected_view"),
    [
        (
            "查看张三的两跳邻居",
            {"mode": "local", "focusTerms": ["张三"], "depth": 2},
        ),
        (
            "只看合作关系",
            {"edgeTypeTerms": ["合作"]},
        ),
    ],
)
async def test_explicit_view_is_merged_when_llm_analysis_omits_view(
    text: str,
    expected_view: dict[str, object],
) -> None:
    provider = SequenceProvider(
        [
            {
                "kind": "analysis_request",
                "normalizedText": "生成图谱概览",
                "task": "overview",
                "targets": [],
                "confidence": 0.9,
                "timeRange": None,
                "filters": {},
            }
        ]
    )
    app = create_app(Settings(), provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/api/v1/intents/normalize", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "analysis_request"
    for key, value in expected_view.items():
        assert body["view"][key] == value
    assert "DETERMINISTIC_VIEW_MERGED" in body["meta"]["warnings"]


@pytest.mark.anyio
async def test_regular_chat_remains_chat_without_explicit_view_command() -> None:
    provider = SequenceProvider([{"kind": "chat", "reply": "你好，我可以帮助规范图分析需求。"}])
    app = create_app(Settings(), provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/api/v1/intents/normalize", json={"text": "你好"})

    assert response.status_code == 200
    assert response.json()["kind"] == "chat"
    assert response.json()["meta"]["warnings"] == []


@pytest.mark.anyio
async def test_invalid_model_output_gets_one_repair_attempt() -> None:
    provider = SequenceProvider(
        [
            {"kind": "analysis_request", "task": "not_allowed"},
            {
                "kind": "analysis_request",
                "normalizedText": "识别桥接节点",
                "task": "bridge_detection",
                "targets": [],
                "confidence": 0.9,
                "filters": {},
            },
        ]
    )
    app = create_app(Settings(), provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/api/v1/intents/normalize", json={"text": "识别桥接节点"})
    assert response.status_code == 200
    assert len(provider.calls) == 2
    assert response.json()["meta"]["warnings"] == ["LLM_OUTPUT_REPAIRED"]


@pytest.mark.anyio
async def test_low_confidence_is_returned_for_explicit_review() -> None:
    provider = SequenceProvider(
        [
            {
                "kind": "analysis_request",
                "normalizedText": "预测潜在关系",
                "task": "link_prediction",
                "targets": [],
                "confidence": 0.2,
                "filters": {},
            }
        ]
    )
    app = create_app(Settings(), provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post("/api/v1/intents/normalize", json={"text": "也许分析一下"})
    body = response.json()
    assert body["task"] == "link_prediction"
    assert "LOW_CONFIDENCE_REQUIRES_REVIEW" in body["meta"]["warnings"]


def test_evaluation_fixture_has_at_least_50_cases() -> None:
    fixture = Path(__file__).parent / "fixtures" / "intent_eval_cases.json"
    cases = json.loads(fixture.read_text(encoding="utf-8"))
    assert len(cases) >= 50
@pytest.mark.anyio
async def test_local_demo_rejects_non_loopback_clients(
    unconfigured_settings: Settings,
) -> None:
    app = create_app(unconfigured_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("203.0.113.10", 5000)),
        base_url="http://api.example",
    ) as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "LOOPBACK_ONLY"
