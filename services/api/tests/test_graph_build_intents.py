from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.main import create_app

from .conftest import SequenceProvider


def profiles() -> list[dict[str, object]]:
    return [
        {
            "name": "from_user",
            "inferredType": "string",
            "nonNullCount": 20,
            "nullCount": 0,
            "uniqueCount": 10,
        },
        {
            "name": "to_user",
            "inferredType": "string",
            "nonNullCount": 20,
            "nullCount": 0,
            "uniqueCount": 12,
        },
        {
            "name": "weight",
            "inferredType": "float",
            "nonNullCount": 20,
            "nullCount": 0,
            "uniqueCount": 8,
        },
        {
            "name": "event_time",
            "inferredType": "datetime",
            "nonNullCount": 20,
            "nullCount": 0,
            "uniqueCount": 20,
        },
    ]


async def _normalize_with_llm(
    payload: dict[str, object],
    **output_updates: object,
) -> httpx.Response:
    output: dict[str, object] = {
        "sourceColumn": None,
        "targetColumn": None,
        "edgeTypeColumn": None,
        "weightColumn": None,
        "timestampColumn": None,
        "nodeIdColumn": None,
        "nodeLabelColumn": None,
        "nodeTypeColumn": None,
        "directedness": "unspecified",
        "confidence": 0.9,
        **output_updates,
    }
    app = create_app(Settings(), provider=SequenceProvider([output]))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        return await client.post(
            "/api/v1/graph-build-intents/normalize", json=payload
        )


@pytest.mark.anyio
async def test_graph_build_normalization_requires_llm(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.post(
        "/api/v1/graph-build-intents/normalize",
        json={"description": "构建图", "columnProfiles": profiles()},
    )
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "LLM_NOT_CONFIGURED"}}


@pytest.mark.anyio
async def test_grounded_graph_build_mapping_is_bounded() -> None:
    response = await _normalize_with_llm(
        {
            "description": "这是用户之间的有向消息关系，from_user 指向 to_user。",
            "columnProfiles": profiles(),
        },
        sourceColumn="from_user",
        targetColumn="to_user",
        weightColumn="weight",
        timestampColumn="event_time",
        directedness="directed",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "graph_build_intent"
    assert body["mapping"] == {
        "sourceColumn": "from_user",
        "targetColumn": "to_user",
        "edgeTypeColumn": None,
        "weightColumn": "weight",
        "timestampColumn": "event_time",
    }
    assert body["directedness"] == "directed"
    assert body["requiresMapping"] is False
    assert body["meta"] == {
        "schemaVersion": "1.0",
        "source": "llm",
        "requestId": body["meta"]["requestId"],
        "model": "test-model",
        "warnings": [],
    }


@pytest.mark.anyio
@pytest.mark.parametrize("forbidden", ["rows", "sampleValues", "values", "examples"])
async def test_graph_build_request_rejects_raw_or_sample_values(
    api_client: httpx.AsyncClient,
    forbidden: str,
) -> None:
    body: dict[str, object] = {
        "description": "构建关系图",
        "columnProfiles": profiles(),
    }
    if forbidden == "rows":
        body[forbidden] = [{"from_user": "alice", "to_user": "bob"}]
    else:
        first = dict(profiles()[0])
        first[forbidden] = ["alice"]
        body["columnProfiles"] = [first, *profiles()[1:]]
    response = await api_client.post(
        "/api/v1/graph-build-intents/normalize",
        json=body,
    )
    assert response.status_code == 422


@pytest.mark.anyio
async def test_llm_columns_are_grounded_and_prompt_has_no_rows() -> None:
    provider = SequenceProvider(
        [
            {
                "sourceColumn": "invented_source",
                "targetColumn": "to_user",
                "edgeTypeColumn": None,
                "weightColumn": "weight",
                "timestampColumn": "secret_time",
                "directedness": "directed",
                "confidence": 0.93,
            }
        ]
    )
    app = create_app(Settings(), provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/graph-build-intents/normalize",
            json={
                "description": "from_user 指向 to_user",
                "columnProfiles": profiles(),
            },
        )

    assert response.status_code == 200
    body = response.json()
    # The deterministic allowlisted endpoint replaces the invented source.
    assert body["mapping"]["sourceColumn"] == "from_user"
    assert body["mapping"]["targetColumn"] == "to_user"
    assert body["mapping"]["timestampColumn"] is None
    assert "UNLISTED_COLUMN_DISCARDED" in body["meta"]["warnings"]
    prompt = provider.calls[0][1]
    assert "columnProfiles" in prompt
    assert "alice" not in prompt
    assert '"rows"' not in prompt
    assert "sampleValues" not in prompt


@pytest.mark.anyio
async def test_ambiguous_columns_require_manual_mapping() -> None:
    response = await _normalize_with_llm(
        {
            "description": "这是成员关系数据",
            "columnProfiles": [
                {
                    "name": "member_a",
                    "inferredType": "string",
                    "nonNullCount": 4,
                    "nullCount": 0,
                    "uniqueCount": 4,
                },
                {
                    "name": "member_b",
                    "inferredType": "string",
                    "nonNullCount": 4,
                    "nullCount": 0,
                    "uniqueCount": 4,
                },
            ],
        },
        confidence=0.2,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["requiresMapping"] is True
    assert body["mapping"]["sourceColumn"] is None
    assert body["mapping"]["targetColumn"] is None
    assert body["confidence"] <= 0.49
    assert "SOURCE_TARGET_MAPPING_REQUIRED" in body["meta"]["warnings"]


@pytest.mark.anyio
async def test_llm_cannot_reverse_deterministic_endpoints() -> None:
    provider = SequenceProvider(
        [
            {
                "sourceColumn": "to_user",
                "targetColumn": "from_user",
                "edgeTypeColumn": None,
                "weightColumn": "weight",
                "timestampColumn": "event_time",
                "directedness": "directed",
                "confidence": 0.99,
            }
        ]
    )
    app = create_app(Settings(), provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/graph-build-intents/normalize",
            json={
                "description": "from_user 指向 to_user",
                "columnProfiles": profiles(),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["mapping"]["sourceColumn"] == "from_user"
    assert body["mapping"]["targetColumn"] == "to_user"
    assert body["requiresMapping"] is True
    assert body["confidence"] <= 0.49
    assert "ENDPOINT_MAPPING_CONFLICT" in body["meta"]["warnings"]


@pytest.mark.anyio
async def test_low_confidence_llm_mapping_requires_confirmation() -> None:
    provider = SequenceProvider(
        [
            {
                "sourceColumn": "from_user",
                "targetColumn": "to_user",
                "edgeTypeColumn": None,
                "weightColumn": "weight",
                "timestampColumn": "event_time",
                "directedness": "directed",
                "confidence": 0.79,
            }
        ]
    )
    app = create_app(Settings(), provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/graph-build-intents/normalize",
            json={
                "description": "from_user 指向 to_user",
                "columnProfiles": profiles(),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["requiresMapping"] is True
    assert body["confidence"] <= 0.49
    assert "LOW_CONFIDENCE_MAPPING" in body["meta"]["warnings"]


@pytest.mark.anyio
async def test_model_cannot_invent_or_flip_directedness() -> None:
    provider = SequenceProvider(
        [
            {
                "sourceColumn": "from_user",
                "targetColumn": "to_user",
                "edgeTypeColumn": None,
                "weightColumn": None,
                "timestampColumn": None,
                "directedness": "directed",
                "confidence": 0.95,
            },
            {
                "sourceColumn": "from_user",
                "targetColumn": "to_user",
                "edgeTypeColumn": None,
                "weightColumn": None,
                "timestampColumn": None,
                "directedness": "undirected",
                "confidence": 0.95,
            },
        ]
    )
    app = create_app(Settings(), provider=provider)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        invented = await client.post(
            "/api/v1/graph-build-intents/normalize",
            json={"description": "这是用户之间的关系", "columnProfiles": profiles()},
        )
        flipped = await client.post(
            "/api/v1/graph-build-intents/normalize",
            json={"description": "from_user 指向 to_user", "columnProfiles": profiles()},
        )

    assert invented.json()["directedness"] == "unspecified"
    assert "MODEL_DIRECTEDNESS_IGNORED" in invented.json()["meta"]["warnings"]
    assert flipped.json()["directedness"] == "directed"
    assert "MODEL_DIRECTEDNESS_CONFLICT" in flipped.json()["meta"]["warnings"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "description",
    [
        "不是无向，是有向",
        "原来无向，改成有向",
        "这是有向而非无向网络",
        "from_user 不指向 to_user",
        "not undirected, use directed",
        "not a directed graph",
    ],
)
async def test_conflicting_or_negated_direction_requires_safe_unspecified(
    description: str,
) -> None:
    response = await _normalize_with_llm(
        {"description": description, "columnProfiles": profiles()},
        sourceColumn="from_user",
        targetColumn="to_user",
        directedness="directed",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["directedness"] == "unspecified"
    assert "DIRECTEDNESS_CONFLICT" in body["meta"]["warnings"]


@pytest.mark.anyio
async def test_vector_word_is_not_direction_evidence() -> None:
    response = await _normalize_with_llm(
        {
            "description": "所有向量特征都已归一化",
            "columnProfiles": profiles(),
        },
    )

    assert response.status_code == 200
    assert response.json()["directedness"] == "unspecified"


@pytest.mark.anyio
async def test_v11_dual_table_profiles_return_grounded_node_mapping() -> None:
    response = await _normalize_with_llm(
        {
            "description": "节点表包含实体类型，关系是无向合作。",
            "files": [
                {
                    "role": "nodes",
                    "columnProfiles": [
                        {"name": "node_id", "inferredType": "string", "nonNullCount": 3, "nullCount": 0, "uniqueCount": 3},
                        {"name": "display_name", "inferredType": "string", "nonNullCount": 3, "nullCount": 0, "uniqueCount": 3},
                        {"name": "node_type", "inferredType": "string", "nonNullCount": 3, "nullCount": 0, "uniqueCount": 2},
                    ],
                },
                {"role": "edges", "columnProfiles": profiles()},
            ],
        },
        sourceColumn="from_user",
        targetColumn="to_user",
        nodeIdColumn="node_id",
        nodeLabelColumn="display_name",
        nodeTypeColumn="node_type",
        directedness="undirected",
    )

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["schemaVersion"] == "1.1"
    assert body["nodeMapping"] == {
        "idColumn": "node_id",
        "labelColumn": "display_name",
        "typeColumn": "node_type",
    }
    assert body["mapping"]["sourceColumn"] == "from_user"
    assert body["requiresMapping"] is False


@pytest.mark.anyio
async def test_v11_rejects_cross_table_payloads_and_non_unique_node_id(
    api_client: httpx.AsyncClient,
) -> None:
    duplicate_roles = await api_client.post(
        "/api/v1/graph-build-intents/normalize",
        json={
            "description": "构建图",
            "files": [
                {"role": "edges", "columnProfiles": profiles()},
                {"role": "edges", "columnProfiles": profiles()},
            ],
        },
    )
    assert duplicate_roles.status_code == 422

    non_unique_id = await _normalize_with_llm(
        {
            "description": "构建图",
            "files": [
                {
                    "role": "nodes",
                    "columnProfiles": [
                        {"name": "node_id", "inferredType": "string", "nonNullCount": 3, "nullCount": 0, "uniqueCount": 2},
                    ],
                },
                {"role": "edges", "columnProfiles": profiles()},
            ],
        },
        sourceColumn="from_user",
        targetColumn="to_user",
        nodeIdColumn="node_id",
    )
    assert non_unique_id.status_code == 200
    body = non_unique_id.json()
    assert body["requiresMapping"] is True
    assert "NODE_ID_MAPPING_REQUIRED" in body["meta"]["warnings"]
