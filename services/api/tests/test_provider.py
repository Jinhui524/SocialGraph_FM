from __future__ import annotations

import json
import socket
import ssl
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings, derive_llm_endpoint
from app.provider import (
    OpenAICompatibleProvider,
    ProviderFailure,
    _strip_leading_think_blocks,
)


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
        ("https://api.deepseek.com", "https://api.deepseek.com/chat/completions"),
        ("https://api.deepseek.com/v1", "https://api.deepseek.com/chat/completions"),
        (
            "https://api.deepseek.com/v1/chat/completions",
            "https://api.deepseek.com/chat/completions",
        ),
        (
            "https://api.deepseek.com.evil.example/v1",
            "https://api.deepseek.com.evil.example/v1/chat/completions",
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
    assert request.headers["accept"] == "application/json"
    assert request.headers["accept-encoding"] == "identity"
    assert request.headers["content-type"] == "application/json"
    assert "x-api-key" not in request.headers
    body = json.loads(request.content)
    assert body == {
        "model": "model-a",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "stream": False,
        "max_tokens": 700,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("api_base", "model", "expected_url", "expected_fields", "gemini_header"),
    [
        (
            "https://api.openai.com/v1",
            "gpt-4.1",
            "https://api.openai.com/v1/chat/completions",
            {"max_completion_tokens": 700},
            None,
        ),
        (
            "https://api.deepseek.com/v1",
            "deepseek-chat",
            "https://api.deepseek.com/chat/completions",
            {"max_tokens": 700, "thinking": {"type": "disabled"}},
            None,
        ),
        (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-example",
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            {"max_tokens": 700},
            "socialgraph-fm/1.0.0",
        ),
        (
            "https://provider.example/v1",
            "openai/gpt-5-mini",
            "https://provider.example/v1/chat/completions",
            {"max_completion_tokens": 700},
            None,
        ),
        (
            "https://provider.example/v1",
            "openai/o3-mini",
            "https://provider.example/v1/chat/completions",
            {"max_completion_tokens": 700},
            None,
        ),
        (
            "https://openrouter.ai/api/v1",
            "openai/o3:free",
            "https://openrouter.ai/api/v1/chat/completions",
            {"max_tokens": 700},
            None,
        ),
        (
            "https://openrouter.ai/api/v1",
            "openai/gpt-5-mini",
            "https://openrouter.ai/api/v1/chat/completions",
            {"max_tokens": 700},
            None,
        ),
        (
            "https://api.minimaxi.com/v1",
            "MiniMax-M3",
            "https://api.minimaxi.com/v1/chat/completions",
            {
                "max_completion_tokens": 700,
                "reasoning_split": True,
                "thinking": {"type": "disabled"},
            },
            None,
        ),
        (
            "https://api.minimax.io/v1",
            "MiniMax-M2.1",
            "https://api.minimax.io/v1/chat/completions",
            {"max_completion_tokens": 700, "reasoning_split": True},
            None,
        ),
        (
            "https://api.openai.com.evil.example/v1",
            "model-a",
            "https://api.openai.com.evil.example/v1/chat/completions",
            {"max_tokens": 700},
            None,
        ),
        (
            "https://api.deepseek.com.evil.example/v1",
            "model-a",
            "https://api.deepseek.com.evil.example/v1/chat/completions",
            {"max_tokens": 700},
            None,
        ),
        (
            "https://generativelanguage.googleapis.com.evil.example/v1",
            "model-a",
            "https://generativelanguage.googleapis.com.evil.example/v1/chat/completions",
            {"max_tokens": 700},
            None,
        ),
    ],
)
async def test_provider_applies_exact_host_and_model_request_profiles(
    api_base: str,
    model: str,
    expected_url: str,
    expected_fields: dict[str, Any],
    gemini_header: str | None,
) -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    provider = OpenAICompatibleProvider(
        _settings(api_base=api_base, model=model),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    assert await provider.generate("system", "user") == {"ok": True}
    await provider.aclose()

    request = observed[0]
    assert str(request.url) == expected_url
    body = json.loads(request.content)
    assert body["stream"] is False
    assert "temperature" not in body
    assert ("max_tokens" in body) ^ ("max_completion_tokens" in body)
    common_fields = {"model", "messages", "stream"}
    assert {key: value for key, value in body.items() if key not in common_fields} == expected_fields
    assert (request.headers.get("x-goog-api-client") or None) == gemini_header


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('<think>private reasoning</think>{"ok":true}', '{"ok":true}'),
        (
            '<think>first</think>\n<think>second</think>\n{"ok":true}',
            '{"ok":true}',
        ),
        (
            'prefix <think>private reasoning</think>{"ok":true}',
            'prefix <think>private reasoning</think>{"ok":true}',
        ),
        ('<think>not closed\n{"ok":true}', '<think>not closed\n{"ok":true}'),
    ],
)
def test_think_blocks_are_stripped_only_when_complete_and_leading(
    content: str, expected: str
) -> None:
    assert _strip_leading_think_blocks(content) == expected


@pytest.mark.anyio
async def test_provider_accepts_complete_leading_think_blocks() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "<think>first</think>\n"
                                "<think>second</think>\n"
                                '{"ok":true}'
                            )
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


@pytest.mark.anyio
async def test_provider_rejects_unclosed_leading_think_block() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '<think>not closed\n{"ok":true}'}}
                ]
            },
        )

    provider = OpenAICompatibleProvider(
        _settings(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderFailure, match="reasoning block") as failure:
        await provider.generate("system", "user")
    assert failure.value.code == "LLM_INVALID_RESPONSE"
    await provider.aclose()


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
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, text="sensitive upstream body")

    provider = OpenAICompatibleProvider(
        _settings(model="unsafe-model-secret-key-at-private.example"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderFailure) as failure:
        await provider.generate("system", "user")
    assert failure.value.code == code
    assert failure.value.retryable is retryable
    assert "sensitive" not in str(failure.value)
    assert provider.connection_status == "error"
    assert calls == 1
    assert "sensitive" not in caplog.text
    assert "unsafe" not in caplog.text
    await provider.aclose()


@pytest.mark.anyio
async def test_provider_timeout_is_explicit_and_not_retried(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("secret timeout", request=request)

    provider = OpenAICompatibleProvider(
        _settings(model="unsafe-model-secret-key-at-private.example"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderFailure) as failure:
        await provider.generate("system", "user")
    assert failure.value.code == "LLM_TIMEOUT"
    assert failure.value.retryable is True
    assert calls == 1
    assert "secret" not in caplog.text
    assert "unsafe" not in caplog.text
    await provider.aclose()


def _raise_transport_failure(kind: str, request: httpx.Request) -> None:
    wrapper: type[httpx.RequestError] = httpx.ConnectError
    cause: BaseException | None = None
    if kind == "local_refused":
        cause = ConnectionRefusedError("unsafe-local-detail")
    elif kind == "local_reset":
        cause = ConnectionResetError("unsafe-local-reset-detail")
    elif kind == "local_aborted":
        cause = ConnectionAbortedError("unsafe-local-aborted-detail")
    elif kind == "local_generic":
        cause = None
    elif kind == "dns":
        cause = socket.gaierror("unsafe-dns-detail")
    elif kind == "connect":
        cause = ConnectionResetError("unsafe-connect-detail")
    elif kind == "tls_hostname":
        cause = ssl.SSLCertVerificationError(1, "unsafe-hostname-detail")
        cause.verify_code = 62
    elif kind == "tls_certificate":
        cause = ssl.SSLCertVerificationError(1, "unsafe-certificate-detail")
        cause.verify_code = 10
    elif kind == "tls_handshake":
        cause = ssl.SSLError("unsafe-handshake-detail")
    elif kind == "protocol":
        wrapper = httpx.RemoteProtocolError
    elif kind == "proxy":
        wrapper = httpx.ProxyError
    elif kind == "network":
        wrapper = httpx.NetworkError
    else:  # pragma: no cover - the parametrization is the closed input set
        raise AssertionError("unknown transport fixture")

    if cause is None:
        raise wrapper("unsafe-url-and-secret-key", request=request)
    try:
        raise cause
    except BaseException as root:
        raise wrapper("unsafe-url-and-secret-key", request=request) from root


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("kind", "api_base", "expected_diagnostic", "retryable"),
    [
        ("local_refused", "http://127.0.0.1:11434/v1", "LOCAL_ENDPOINT", True),
        ("local_reset", "http://127.0.0.1:11434/v1", "CONNECT", True),
        ("local_aborted", "http://localhost:11434/v1", "CONNECT", True),
        ("local_generic", "http://[::1]:11434/v1", "CONNECT", True),
        ("dns", "https://provider.example/v1", "DNS", True),
        ("connect", "https://provider.example/v1", "CONNECT", True),
        ("tls_hostname", "https://provider.example/v1", "TLS_HOSTNAME", False),
        ("tls_certificate", "https://provider.example/v1", "TLS_CERTIFICATE", False),
        ("tls_handshake", "https://provider.example/v1", "TLS_HANDSHAKE", True),
        ("protocol", "https://provider.example/v1", "PROTOCOL", True),
        ("proxy", "https://provider.example/v1", "PROXY", False),
        ("network", "https://provider.example/v1", "NETWORK", True),
    ],
)
async def test_provider_classifies_network_failures_without_leaking_or_retrying(
    kind: str,
    api_base: str,
    expected_diagnostic: str,
    retryable: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        _raise_transport_failure(kind, request)
        raise AssertionError("transport failure fixture returned")  # pragma: no cover

    provider = OpenAICompatibleProvider(
        _settings(
            api_base=api_base,
            model="unsafe-model-secret-key-at-private.example",
        ),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProviderFailure) as failure:
        await provider.generate("system", "user")

    assert failure.value.code == "LLM_NETWORK_ERROR"
    assert failure.value.diagnostic_code == expected_diagnostic
    assert failure.value.retryable is retryable
    assert str(failure.value) == "LLM network request failed"
    assert calls == 1
    assert "unsafe" not in caplog.text
    with pytest.raises(AttributeError):
        failure.value.diagnostic_code = "NETWORK"  # type: ignore[misc]
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
