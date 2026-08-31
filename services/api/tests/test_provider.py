from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings, derive_llm_endpoint
from app.provider import OpenAICompatibleProvider, ProviderFailure


def _settings(
    *,
    api_base: str = "https://provider.example/v1",
    api_key: str = "secret-key",
    model: str = "model-a",
) -> Settings:
    return Settings(llm_api_base=api_base, llm_api_key=api_key, llm_model=model)


@pytest.mark.parametrize(
    ("base", "endpoint"),
    [
        ("https://provider.example", "https://provider.example/v1/chat/completions"),
        ("https://provider.example/v1", "https://provider.example/v1/chat/completions"),
        (
            "https://provider.example/v1/chat/completions",
            "https://provider.example/v1/chat/completions",
        ),
        (
            "https://provider.example/openai/v1",
            "https://provider.example/openai/v1/chat/completions",
        ),
    ],
)
def test_chat_completions_endpoint_normalization(base: str, endpoint: str) -> None:
    assert derive_llm_endpoint(base) == endpoint


def test_settings_exposes_only_three_llm_inputs_and_allows_http_loopback() -> None:
    settings = _settings(api_base="http://127.0.0.1:11434/v1")
    assert settings.llm_api_base == "http://127.0.0.1:11434/v1"
    assert settings.llm_model == "model-a"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "secret-key"
    assert settings.llm_configured is True
    assert not hasattr(settings, "llm_api_mode")
    assert not hasattr(settings, "llm_auth_scheme")
    assert not hasattr(settings, "llm_timeout_seconds")


@pytest.mark.parametrize(
    "api_base",
    [
        "http://provider.example/v1",
        "https://user:pass@provider.example/v1",
        "https://provider.example/v1?key=value",
        "https://provider.example/v1#fragment",
        "https://https://provider.example/v1",
        "https://provider.example/%0aevil",
        "not-a-url",
    ],
)
def test_settings_rejects_unsafe_llm_api_base(api_base: str) -> None:
    with pytest.raises(ValidationError):
        _settings(api_base=api_base)


def test_settings_rejects_partial_or_unsafe_secret_configuration() -> None:
    with pytest.raises(ValidationError):
        Settings(llm_api_base="https://provider.example/v1", llm_model="model-a")
    with pytest.raises(ValidationError):
        _settings(api_key=" secret ")
    with pytest.raises(ValidationError):
        _settings(api_key="line1\nline2")


@pytest.mark.anyio
async def test_provider_uses_fixed_chat_completions_contract_and_bearer_auth() -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps({"answer": "ok"})}}
                ]
            },
        )

    provider = OpenAICompatibleProvider(
        _settings(api_base="https://provider.example"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    result = await provider.generate("system", "user")
    await provider.aclose()

    assert result == {"answer": "ok"}
    assert provider.connection_status == "call_succeeded"
    assert len(observed) == 1
    request = observed[0]
    assert str(request.url) == "https://provider.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret-key"
    assert "x-api-key" not in request.headers
    body = json.loads(request.content)
    assert body == {
        "model": "model-a",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "temperature": 0,
        "max_tokens": 700,
    }


@pytest.mark.anyio
async def test_owned_client_fixes_timeout_proxy_and_redirect_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def aclose(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    provider = OpenAICompatibleProvider(_settings())
    assert captured["timeout"] == 15.0
    assert captured["trust_env"] is False
    assert captured["follow_redirects"] is False
    await provider.aclose()
    assert captured["closed"] is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (400, "LLM_REQUEST_REJECTED", False),
        (401, "LLM_AUTH_ERROR", False),
        (403, "LLM_AUTH_ERROR", False),
        (404, "LLM_ENDPOINT_ERROR", False),
        (408, "LLM_TIMEOUT", True),
        (422, "LLM_REQUEST_REJECTED", False),
        (429, "LLM_RATE_LIMITED", True),
        (500, "LLM_UPSTREAM_ERROR", True),
    ],
)
async def test_provider_classifies_http_failures_without_retry_or_body_leak(
    status: int,
    code: str,
    retryable: bool,
) -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, text="sensitive upstream body")

    provider = OpenAICompatibleProvider(
        _settings(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderFailure) as failure:
        await provider.generate("system", "user")
    assert failure.value.code == code
    assert failure.value.retryable is retryable
    assert "sensitive" not in str(failure.value)
    assert provider.connection_status == "error"
    assert calls == 1
    await provider.aclose()


@pytest.mark.anyio
async def test_provider_timeout_is_explicit_and_not_retried() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("secret timeout", request=request)

    provider = OpenAICompatibleProvider(
        _settings(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderFailure) as failure:
        await provider.generate("system", "user")
    assert failure.value.code == "LLM_TIMEOUT"
    assert failure.value.retryable is True
    assert calls == 1
    await provider.aclose()


@pytest.mark.anyio
async def test_provider_rejects_response_larger_than_two_mib() -> None:
    oversized = b"x" * (2 * 1024 * 1024 + 1)

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-length": str(len(oversized))},
            content=oversized,
        )

    provider = OpenAICompatibleProvider(
        _settings(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderFailure) as failure:
        await provider.generate("system", "user")
    assert failure.value.code == "LLM_RESPONSE_TOO_LARGE"
    await provider.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": "not json"}}]},
        {"choices": [{"message": {"content": "[]"}}]},
    ],
)
async def test_provider_rejects_malformed_success_responses(payload: dict[str, Any]) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    provider = OpenAICompatibleProvider(
        _settings(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderFailure) as failure:
        await provider.generate("system", "user")
    assert failure.value.code == "LLM_INVALID_RESPONSE"
    await provider.aclose()


@pytest.mark.anyio
async def test_provider_accepts_text_parts_and_fenced_json_object() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "```json\n"},
                                {"type": "text", "text": '{"ok":true}\n```'},
                            ]
                        }
                    }
                ]
            },
        )

    provider = OpenAICompatibleProvider(
        _settings(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    assert await provider.generate("system", "user") == {"ok": True}
    await provider.aclose()
