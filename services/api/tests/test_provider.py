from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.provider import OpenAICompatibleProvider, ProviderFailure


def _configured_settings(
    *,
    api_mode: str = "chat_completions",
    api_base: str = "https://provider.example/v1",
    verification_status: str = "configured_unverified",
    auth_scheme: str | None = None,
    anthropic_version: str | None = None,
) -> Settings:
    return Settings(
        llm_api_base=api_base,
        llm_api_key="test-key",
        llm_model="intent-model",
        llm_api_mode=api_mode,
        llm_auth_scheme=auth_scheme,
        llm_anthropic_version=anthropic_version,
        llm_timeout_seconds=1,
        llm_verification_status=verification_status,
    )


def test_cleared_launcher_values_restore_safe_llm_defaults() -> None:
    settings = Settings(
        llm_api_base="",
        llm_api_key="",
        llm_model="",
        llm_api_mode="",
        llm_timeout_seconds="",  # type: ignore[arg-type]
        llm_allow_insecure_loopback="",  # type: ignore[arg-type]
    )
    assert settings.llm_configured is False
    assert settings.llm_api_mode == "chat_completions"
    assert settings.llm_timeout_seconds == 15.0
    assert settings.llm_allow_insecure_loopback is False
    assert settings.llm_verification_status == "configured_unverified"
    assert settings.llm_auth_scheme == "bearer"
    assert settings.llm_anthropic_version is None


def _completion(content: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]},
    )


@pytest.mark.parametrize(
    "api_base",
    [
        "http://provider.example/v1",
        "https://user:password@provider.example/v1",
        "https://provider.example/v1?tenant=secret",
        "https://provider.example/v1#fragment",
        "https://https://provider.example/v1",
        "https://provider.example/v1%0aheader",
        "https://:443",
        "https://provider.example :443/v1",
        "https://provider.example/v1/%09segment",
        "https://provider.example/v1/%7fsegment",
    ],
)
def test_settings_rejects_unsafe_llm_api_base(api_base: str) -> None:
    with pytest.raises(ValidationError):
        _configured_settings(api_base=api_base)


def test_settings_allows_explicit_http_loopback_only() -> None:
    settings = Settings(
        llm_api_base="http://127.0.0.1:11434/v1/",
        llm_api_key="test-key",
        llm_model="local-model",
        llm_allow_insecure_loopback=True,
    )
    assert settings.llm_api_base == "http://127.0.0.1:11434/v1"


def test_anthropic_settings_default_from_protocol_not_key_prefix() -> None:
    settings = Settings(
        llm_api_base="https://api.anthropic.com/v1",
        llm_api_key="test-relay-key",
        llm_model="claude-model",
        llm_api_mode="anthropic_messages",
    )
    assert settings.llm_auth_scheme == "x-api-key"
    assert settings.llm_anthropic_version == "2023-06-01"


def test_anthropic_version_is_rejected_for_openai_protocol() -> None:
    with pytest.raises(ValidationError, match="anthropic_messages"):
        _configured_settings(anthropic_version="2023-06-01")


def test_settings_rejects_partial_llm_configuration() -> None:
    with pytest.raises(ValidationError, match="must be configured together"):
        Settings(llm_api_base="https://provider.example/v1")


def test_settings_rejects_unknown_llm_verification_status() -> None:
    with pytest.raises(ValidationError):
        Settings(llm_verification_status="verified")  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_owned_client_disables_environment_proxies_and_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        async def aclose(self) -> None:
            return None

    def client_factory(**options: object) -> FakeClient:
        captured.update(options)
        return FakeClient()

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    provider = OpenAICompatibleProvider(_configured_settings())
    assert captured["trust_env"] is False
    assert captured["follow_redirects"] is False
    await provider.aclose()


@pytest.mark.parametrize(
    "verification_status",
    ["configured_unverified", "call_succeeded", "fallback"],
)
@pytest.mark.anyio
async def test_provider_starts_with_persisted_connection_status(
    verification_status: str,
) -> None:
    async with httpx.AsyncClient() as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(verification_status=verification_status),
            client=client,
        )
        assert provider.connection_status == verification_status


@pytest.mark.anyio
async def test_provider_connection_status_tracks_success_and_fallback() -> None:
    responses = [
        _completion({"kind": "chat", "reply": "ok"}),
        httpx.Response(200, json={}),
    ]

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: responses.pop(0))
    ) as client:
        provider = OpenAICompatibleProvider(_configured_settings(), client=client)
        assert provider.connection_status == "configured_unverified"
        await provider.generate("system", "user")
        assert provider.connection_status == "call_succeeded"
        with pytest.raises(ProviderFailure):
            await provider.generate("system", "user")
        assert provider.connection_status == "fallback"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("api_base", "api_mode", "expected_url"),
    [
        (
            "https://api.openai.com/v1",
            "responses",
            "https://api.openai.com/v1/responses",
        ),
        (
            "https://api.deepseek.com",
            "chat_completions",
            "https://api.deepseek.com/chat/completions",
        ),
        (
            "https://api.deepseek.com",
            "responses",
            "https://api.deepseek.com/responses",
        ),
        (
            "https://open.bigmodel.cn/api/paas/v4",
            "chat_completions",
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        ),
    ],
)
async def test_provider_presets_use_bearer_openai_compatible_paths(
    api_base: str,
    api_mode: str,
    expected_url: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == expected_url
        assert request.headers["Authorization"] == "Bearer test-key"
        if api_mode == "responses":
            return httpx.Response(
                200, json={"output_text": '{"kind":"chat","reply":"ok"}'}
            )
        return _completion({"kind": "chat", "reply": "ok"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(api_base=api_base, api_mode=api_mode),
            client=client,
        )
        assert await provider.generate("system", "user") == {
            "kind": "chat",
        "reply": "ok",
    }


@pytest.mark.anyio
async def test_anthropic_messages_uses_explicit_headers_body_and_response_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.anthropic.com/v1/messages"
        assert "Authorization" not in request.headers
        assert request.headers["x-api-key"] == "test-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content)
        assert body == {
            "model": "intent-model",
            "system": "system",
            "messages": [{"role": "user", "content": "user"}],
            "max_tokens": 700,
        }
        return httpx.Response(
            200,
            json={
                "type": "message",
                "content": [
                    {
                        "type": "text",
                        "text": '{"kind":"chat","reply":"anthropic-ok"}',
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(
                api_mode="anthropic_messages",
                api_base="https://api.anthropic.com/v1",
            ),
            client=client,
        )
        result = await provider.generate("system", "user")

    assert result == {"kind": "chat", "reply": "anthropic-ok"}


@pytest.mark.anyio
async def test_anthropic_relay_can_use_explicit_bearer_without_key_prefix_guessing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        assert "x-api-key" not in request.headers
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": '{"ok":true}'}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(
                api_mode="anthropic_messages",
                auth_scheme="bearer",
            ),
            client=client,
        )
        assert await provider.generate("system", "user") == {"ok": True}


@pytest.mark.anyio
@pytest.mark.parametrize("failure_status", [429, 503])
async def test_provider_retries_retryable_status_once_then_succeeds(failure_status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url == "https://provider.example/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        if calls == 1:
            return httpx.Response(failure_status)
        return _completion({"kind": "chat", "reply": "你好"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(),
            client=client,
            retry_delay_seconds=0,
        )
        result = await provider.generate("system", "user")
    assert calls == 2
    assert result == {"kind": "chat", "reply": "你好"}


@pytest.mark.anyio
async def test_provider_does_not_retry_non_retryable_401() -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(),
            client=client,
            retry_delay_seconds=0,
        )
        with pytest.raises(ProviderFailure) as error:
            await provider.generate("system", "user")
    assert calls == 1
    assert error.value.code == "LLM_AUTH_ERROR"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (400, "LLM_REQUEST_REJECTED"),
        (403, "LLM_AUTH_ERROR"),
        (404, "LLM_ENDPOINT_ERROR"),
        (422, "LLM_REQUEST_REJECTED"),
    ],
)
async def test_provider_classifies_non_retryable_http_failures_without_body_leak(
    status: int, expected_code: str
) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            status,
            json={"error": {"message": "secret-upstream-diagnostic"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(), client=client, retry_delay_seconds=0
        )
        with pytest.raises(ProviderFailure) as captured:
            await provider.generate("system", "user")

    assert calls == 1
    assert captured.value.code == expected_code
    assert "secret-upstream-diagnostic" not in str(captured.value)


@pytest.mark.anyio
async def test_provider_retries_timeout_only_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timeout", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(),
            client=client,
            retry_delay_seconds=0,
        )
        with pytest.raises(ProviderFailure) as error:
            await provider.generate("system", "user")
    assert calls == 2
    assert error.value.code == "LLM_TIMEOUT"


@pytest.mark.anyio
async def test_responses_mode_uses_instructions_input_and_parses_output_items() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://provider.example/v1/responses"
        body = json.loads(request.content)
        assert body["model"] == "intent-model"
        assert body["instructions"] == "system"
        assert body["input"] == "user"
        assert body["max_output_tokens"] == 700
        assert body["text"] == {"format": {"type": "json_object"}}
        assert "messages" not in body
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"kind":"chat","reply":"你好"}',
                            }
                        ],
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(api_mode="responses"),
            client=client,
        )
        result = await provider.generate("system", "user")
    assert result == {"kind": "chat", "reply": "你好"}


@pytest.mark.anyio
async def test_responses_mode_accepts_direct_output_text_object() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"output_text": {"kind": "chat", "reply": "收到"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(api_mode="responses"),
            client=client,
        )
        result = await provider.generate("system", "user")
    assert result == {"kind": "chat", "reply": "收到"}


@pytest.mark.anyio
@pytest.mark.parametrize("unsupported_status", [400, 422])
async def test_responses_mode_caches_unsupported_json_format_for_compatible_relay(
    unsupported_status: int,
) -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return httpx.Response(
                unsupported_status,
                json={"error": {"message": "unsupported text.format"}},
            )
        return httpx.Response(
            200,
            json={"output_text": '{"kind":"chat","reply":"兼容成功"}'},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(api_mode="responses"),
            client=client,
            retry_delay_seconds=0,
        )
        result = await provider.generate("system", "user")
        repeated = await provider.generate("system", "another user")
    assert len(bodies) == 3
    assert "text" in bodies[0]
    assert "text" not in bodies[1]
    assert "text" not in bodies[2]
    assert result == {"kind": "chat", "reply": "兼容成功"}
    assert repeated == result


@pytest.mark.anyio
@pytest.mark.parametrize("unsupported_status", [400, 422])
async def test_chat_mode_caches_unsupported_json_format_for_compatible_relay(
    unsupported_status: int,
) -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return httpx.Response(
                unsupported_status,
                json={"error": {"message": "unsupported response_format"}},
            )
        return _completion({"kind": "chat", "reply": "兼容成功"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(),
            client=client,
            retry_delay_seconds=0,
        )
        result = await provider.generate("system", "user")
        repeated = await provider.generate("system", "another user")
    assert len(bodies) == 3
    assert "response_format" in bodies[0]
    assert "response_format" not in bodies[1]
    assert "response_format" not in bodies[2]
    assert result == {"kind": "chat", "reply": "兼容成功"}
    assert repeated == result


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("api_mode", "field", "message"),
    [
        ("responses", "text", "upstream: unsupported text format"),
        ("chat_completions", "response_format", "unknown field response_format"),
    ],
)
async def test_provider_does_not_treat_5xx_as_structured_format_rejection(
    api_mode: str,
    field: str,
    message: str,
) -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return httpx.Response(502, json={"error": {"message": message}})
        if api_mode == "responses":
            return httpx.Response(
                200, json={"output_text": '{"kind":"chat","reply":"ok"}'}
            )
        return _completion({"kind": "chat", "reply": "ok"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(api_mode=api_mode),
            client=client,
            retry_delay_seconds=0,
        )
        first = await provider.generate("system", "user")
        second = await provider.generate("system", "another user")

    assert len(bodies) == 3
    assert field in bodies[0]
    assert field in bodies[1]
    assert field in bodies[2]
    assert first == {"kind": "chat", "reply": "ok"}
    assert second == first


@pytest.mark.anyio
async def test_model_rejection_does_not_trigger_structured_format_fallback() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "model does not exist; response_format was included"
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(),
            client=client,
            retry_delay_seconds=0,
        )
        with pytest.raises(ProviderFailure) as error:
            await provider.generate("system", "user")

    assert len(bodies) == 1
    assert "response_format" in bodies[0]
    assert error.value.code == "LLM_REQUEST_REJECTED"


@pytest.mark.anyio
async def test_responses_mode_retries_unclassified_502_without_changing_request() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return httpx.Response(502)
        return httpx.Response(
            200,
            json={"output_text": '{"kind":"chat","reply":"兼容成功"}'},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(api_mode="responses"),
            client=client,
            retry_delay_seconds=0,
        )
        result = await provider.generate("system", "user")
    assert len(bodies) == 2
    assert "text" in bodies[0]
    assert "text" in bodies[1]
    assert result == {"kind": "chat", "reply": "兼容成功"}


@pytest.mark.anyio
async def test_responses_mode_does_not_retry_second_5xx_after_format_fallback() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(502)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(api_mode="responses"),
            client=client,
            retry_delay_seconds=0,
        )
        with pytest.raises(ProviderFailure) as error:
            await provider.generate("system", "user")
    assert len(bodies) == 2
    assert "text" in bodies[0]
    assert "text" in bodies[1]
    assert error.value.code == "LLM_UPSTREAM_ERROR"


@pytest.mark.anyio
async def test_endpoint_is_normalized_when_base_contains_other_protocol_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://provider.example/v1/responses"
        return httpx.Response(200, json={"output_text": '{"kind":"chat","reply":"ok"}'})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(
                api_mode="responses",
                api_base="https://provider.example/v1/chat/completions",
            ),
            client=client,
        )
        await provider.generate("system", "user")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "api_base",
    [
        "https://api.anthropic.com/v1",
        "https://api.anthropic.com/v1/messages",
    ],
)
async def test_anthropic_root_v1_and_full_endpoint_derive_identically(
    api_base: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.anthropic.com/v1/messages"
        return httpx.Response(
            200, json={"content": [{"type": "text", "text": '{"ok":true}'}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            _configured_settings(
                api_mode="anthropic_messages",
                api_base=api_base,
            ),
            client=client,
        )
        assert await provider.generate("system", "user") == {"ok": True}


@pytest.mark.anyio
async def test_provider_rejects_response_larger_than_two_mib() -> None:
    oversized = b"x" * (2 * 1024 * 1024 + 1)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=oversized)
        )
    ) as client:
        provider = OpenAICompatibleProvider(_configured_settings(), client=client)
        with pytest.raises(ProviderFailure) as captured:
            await provider.generate("system", "user")

    assert captured.value.code == "LLM_RESPONSE_TOO_LARGE"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (httpx.Response(200, text="not-json"), "LLM_INVALID_RESPONSE"),
        (httpx.Response(200, json={}), "LLM_INVALID_RESPONSE"),
        (
            httpx.Response(200, json={"error": {"message": "invalid key"}}),
            "LLM_INVALID_RESPONSE",
        ),
        (httpx.Response(200, json={"choices": []}), "LLM_INVALID_RESPONSE"),
        (
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": "not a JSON object"}}]},
            ),
            "LLM_INVALID_RESPONSE",
        ),
    ],
)
async def test_provider_rejects_malformed_success_responses(
    response: httpx.Response,
    expected_code: str,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: response)
    ) as client:
        provider = OpenAICompatibleProvider(_configured_settings(), client=client)
        with pytest.raises(ProviderFailure) as error:
            await provider.generate("system", "user")
    assert error.value.code == expected_code
