from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.normalizer import IntentNormalizerService
from app.provider import ProviderFailure
from app.schemas import NormalizeIntentRequest

from .conftest import SequenceProvider


@pytest.mark.anyio
async def test_provider_failure_falls_back_without_exposing_details() -> None:
    provider = SequenceProvider([ProviderFailure("LLM_RATE_LIMITED", "secret upstream detail")])
    result = await IntentNormalizerService(provider).normalize(
        NormalizeIntentRequest(text="找出关键成员"),
        request_id="test-id",
    )
    body = result.model_dump(by_alias=True)
    assert body["kind"] == "analysis_request"
    assert body["task"] == "centrality"
    assert body["meta"]["warnings"] == ["LLM_RATE_LIMITED"]
    assert "secret" not in json.dumps(body, ensure_ascii=False)


@pytest.mark.anyio
async def test_two_invalid_model_outputs_fall_back_after_one_repair() -> None:
    provider = SequenceProvider(
        [
            {"kind": "analysis_request", "task": "invalid"},
            {"kind": "analysis_request", "task": "still-invalid"},
        ]
    )
    result = await IntentNormalizerService(provider).normalize(
        NormalizeIntentRequest(text="分析社区结构"),
        request_id="repair-failed",
    )
    body = result.model_dump(by_alias=True)
    assert len(provider.calls) == 2
    assert body["task"] == "community"
    assert body["meta"]["source"] == "deterministic_fallback"
    assert body["meta"]["warnings"] == ["LLM_INVALID_RESPONSE"]


@pytest.mark.anyio
async def test_invalid_json_provider_failure_gets_one_repair_attempt() -> None:
    provider = SequenceProvider(
        [
            ProviderFailure("LLM_INVALID_RESPONSE", "malformed JSON"),
            {"kind": "chat", "reply": "你好，我可以帮助规范研究需求。"},
        ]
    )
    result = await IntentNormalizerService(provider).normalize(
        NormalizeIntentRequest(text="你好"),
        request_id="invalid-json-repaired",
    )
    body = result.model_dump(by_alias=True)
    assert len(provider.calls) == 2
    assert body["kind"] == "chat"
    assert body["meta"]["source"] == "llm"
    assert body["meta"]["warnings"] == ["LLM_OUTPUT_REPAIRED"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("text", "expected_task", "expected_view", "expected_targets"),
    [
        (
            "查看张三的两跳邻居",
            "overview",
            {"mode": "local", "depth": 2, "focusTerms": ["张三"]},
            ["张三"],
        ),
        (
            "显示张三到李四的路径",
            "overview",
            {"mode": "path", "focusTerms": ["张三", "李四"]},
            ["张三", "李四"],
        ),
        (
            "只看2022年后的合作关系",
            "overview",
            {"edgeTypeTerms": ["合作"]},
            [],
        ),
        (
            "找出桥接节点并高亮",
            "bridge_detection",
            {"overlay": "articulation"},
            [],
        ),
    ],
)
async def test_deterministic_fallback_builds_bounded_view_commands(
    text: str,
    expected_task: str,
    expected_view: dict[str, object],
    expected_targets: list[str],
) -> None:
    result = await IntentNormalizerService().normalize(
        NormalizeIntentRequest(text=text),
        request_id="fallback-view",
    )
    body = result.model_dump(by_alias=True)
    assert body["task"] == expected_task
    assert body["targets"] == expected_targets
    assert body["meta"]["schemaVersion"] == "1.1"
    for key, value in expected_view.items():
        assert body["view"][key] == value
    if "2022" in text:
        assert body["timeRange"] == {"start": "2022", "end": None}
        assert body["filters"] == {"startYear": "2022"}


@pytest.mark.anyio
async def test_malformed_view_gets_one_repair_then_schema_11_response() -> None:
    provider = SequenceProvider(
        [
            {
                "kind": "analysis_request",
                "normalizedText": "查看张三的邻域",
                "task": "overview",
                "targets": ["张三"],
                "confidence": 0.9,
                "filters": {},
                "view": {
                    "mode": "local",
                    "focusTerms": ["张三"],
                    "depth": 4,
                    "nodeTypeTerms": [],
                    "edgeTypeTerms": [],
                },
            },
            {
                "kind": "analysis_request",
                "normalizedText": "查看张三的两跳邻域",
                "task": "overview",
                "targets": ["张三"],
                "confidence": 0.9,
                "filters": {},
                "view": {
                    "mode": "local",
                    "focusTerms": ["张三"],
                    "depth": 2,
                    "nodeTypeTerms": [],
                    "edgeTypeTerms": [],
                },
            },
        ]
    )
    result = await IntentNormalizerService(provider).normalize(
        NormalizeIntentRequest(text="查看张三的两跳邻居"),
        request_id="repair-view",
    )
    body = result.model_dump(by_alias=True)
    assert len(provider.calls) == 2
    assert body["view"]["depth"] == 2
    assert body["view"]["focusTerms"] == ["张三"]
    assert body["meta"]["schemaVersion"] == "1.1"
    assert body["meta"]["warnings"] == ["LLM_OUTPUT_REPAIRED"]


@pytest.mark.anyio
async def test_fallback_eval_accuracy_is_at_least_90_percent() -> None:
    fixture = Path(__file__).parent / "fixtures" / "intent_eval_cases.json"
    cases = json.loads(fixture.read_text(encoding="utf-8"))
    service = IntentNormalizerService()
    correct = 0
    for index, case in enumerate(cases):
        result = await service.normalize(
            NormalizeIntentRequest(text=case["text"]),
            request_id=f"eval-{index}",
        )
        actual = result.kind if result.kind == "chat" else result.task
        if actual == case["expected"]:
            correct += 1
    assert correct / len(cases) >= 0.90
