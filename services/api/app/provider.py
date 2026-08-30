from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any, Literal, Protocol

import httpx

from .config import Settings, derive_llm_endpoint

logger = logging.getLogger(__name__)
MAX_LLM_RESPONSE_BYTES = 2 * 1024 * 1024


class ProviderFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class IntentProvider(Protocol):
    @property
    def model(self) -> str | None: ...

    @property
    def connection_status(
        self,
    ) -> Literal["configured_unverified", "call_succeeded", "fallback"]: ...

    async def generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


def _endpoint_url(api_base: str, api_mode: str) -> str:
    return derive_llm_endpoint(api_base, api_mode)


def _extract_message_content(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderFailure("LLM_INVALID_RESPONSE", "LLM response did not contain choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ProviderFailure("LLM_INVALID_RESPONSE", "LLM choice was not an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise ProviderFailure("LLM_INVALID_RESPONSE", "LLM choice did not contain a message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "".join(parts)
    raise ProviderFailure("LLM_INVALID_RESPONSE", "LLM message content was empty")


def _text_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, Mapping):
        # A few compatible gateways return the structured result directly.
        return json.dumps(dict(value), ensure_ascii=False)
    return None


def _extract_responses_content(payload: Mapping[str, Any]) -> str:
    direct = _text_value(payload.get("output_text"))
    if direct is not None:
        return direct

    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                item_text = _text_value(item.get("text"))
                if item_text is not None:
                    parts.append(item_text)
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                part_text = _text_value(part.get("text"))
                if part_text is not None:
                    parts.append(part_text)
        if parts:
            return "".join(parts)

    # Some relays route /responses internally through Chat Completions and keep
    # that envelope. Supporting it costs nothing and makes the adapter tolerant.
    if "choices" in payload:
        return _extract_message_content(payload)
    raise ProviderFailure("LLM_INVALID_RESPONSE", "LLM Responses output text was empty")


def _extract_anthropic_content(payload: Mapping[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") not in {None, "text"}:
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "".join(parts)
    raise ProviderFailure(
        "LLM_INVALID_RESPONSE", "LLM Anthropic message content was empty"
    )


def _decode_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ProviderFailure("LLM_INVALID_RESPONSE", "LLM did not return a JSON object") from None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProviderFailure("LLM_INVALID_RESPONSE", "LLM returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ProviderFailure("LLM_INVALID_RESPONSE", "LLM JSON output was not an object")
    return parsed


def _rejects_structured_output(response: httpx.Response, field: str) -> bool:
    if response.status_code not in {400, 422}:
        return False
    message = response.text[:2_000].casefold()
    field_hints = {field.casefold(), "json_object", "structured output"}
    rejection_hints = {
        "unsupported",
        "not supported",
        "unknown field",
        "unrecognized field",
        "not allowed",
    }
    return any(hint in message for hint in field_hints) and any(
        hint in message for hint in rejection_hints
    )


def _safe_http_failure(status_code: int) -> ProviderFailure:
    if status_code in {401, 403}:
        return ProviderFailure("LLM_AUTH_ERROR", "LLM authentication was rejected")
    if status_code == 404:
        return ProviderFailure(
            "LLM_ENDPOINT_ERROR", "LLM endpoint or protocol was not found"
        )
    if status_code in {400, 422}:
        return ProviderFailure(
            "LLM_REQUEST_REJECTED", "LLM request or model was rejected"
        )
    if status_code in {408, 429}:
        code = "LLM_TIMEOUT" if status_code == 408 else "LLM_RATE_LIMITED"
        return ProviderFailure(
            code,
            "LLM upstream service is temporarily unavailable",
            retryable=True,
        )
    if status_code >= 500:
        return ProviderFailure(
            "LLM_UPSTREAM_ERROR",
            "LLM upstream service is temporarily unavailable",
            retryable=True,
        )
    return ProviderFailure(
        "LLM_HTTP_ERROR", f"LLM request failed with status {status_code}"
    )


class OpenAICompatibleProvider:
    """Minimal OpenAI-compatible client with bounded retries."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        retry_delay_seconds: float = 0.15,
    ) -> None:
        if not settings.llm_configured:
            raise ValueError("OpenAICompatibleProvider requires complete LLM configuration")
        assert settings.llm_api_base is not None
        assert settings.llm_api_key is not None
        assert settings.llm_model is not None
        self._model = settings.llm_model
        self._api_mode = settings.llm_api_mode
        self._url = _endpoint_url(settings.llm_api_base, self._api_mode)
        self._api_key = settings.llm_api_key.get_secret_value()
        assert settings.llm_auth_scheme is not None
        self._auth_scheme = settings.llm_auth_scheme
        self._anthropic_version = settings.llm_anthropic_version
        self._retry_delay_seconds = max(0.0, retry_delay_seconds)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=settings.llm_timeout_seconds,
            trust_env=False,
            follow_redirects=False,
        )
        self._structured_output_supported: bool | None = None
        self._connection_status = settings.llm_verification_status

    @property
    def model(self) -> str:
        return self._model

    @property
    def connection_status(
        self,
    ) -> Literal["configured_unverified", "call_succeeded", "fallback"]:
        return self._connection_status

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post_bounded(
        self, headers: dict[str, str], body: dict[str, Any]
    ) -> httpx.Response:
        request = self._client.build_request(
            "POST", self._url, headers=headers, json=body
        )
        response = await self._client.send(request, stream=True)
        try:
            content_length = response.headers.get("content-length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError:
                    declared = 0
                if declared > MAX_LLM_RESPONSE_BYTES:
                    raise ProviderFailure(
                        "LLM_RESPONSE_TOO_LARGE",
                        "LLM response exceeded the safe size limit",
                    )
            chunks: list[bytes] = []
            observed = 0
            async for chunk in response.aiter_bytes():
                observed += len(chunk)
                if observed > MAX_LLM_RESPONSE_BYTES:
                    raise ProviderFailure(
                        "LLM_RESPONSE_TOO_LARGE",
                        "LLM response exceeded the safe size limit",
                    )
                chunks.append(chunk)
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                content=b"".join(chunks),
                request=request,
                extensions=response.extensions,
            )
        finally:
            await response.aclose()

    async def generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            result = await self._generate(system_prompt, user_prompt)
        except ProviderFailure:
            self._connection_status = "fallback"
            raise
        self._connection_status = "call_succeeded"
        return result

    async def _generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        structured_output_field: str | None
        structured_output_value: dict[str, Any] | None
        if self._api_mode == "responses":
            body: dict[str, Any] = {
                "model": self._model,
                "instructions": system_prompt,
                "input": user_prompt,
                "max_output_tokens": 700,
            }
            structured_output_field = "text"
            structured_output_value = {
                "format": {"type": "json_object"}
            }
        elif self._api_mode == "chat_completions":
            body = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": 700,
            }
            structured_output_field = "response_format"
            structured_output_value = {"type": "json_object"}
        else:
            body = {
                "model": self._model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": 700,
            }
            structured_output_field = None
            structured_output_value = None
        if (
            structured_output_field is not None
            and structured_output_value is not None
            and self._structured_output_supported is not False
        ):
            body[structured_output_field] = structured_output_value
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._auth_scheme == "bearer":
            headers["Authorization"] = f"Bearer {self._api_key}"
        else:
            headers["x-api-key"] = self._api_key
        if self._api_mode == "anthropic_messages":
            assert self._anthropic_version is not None
            headers["anthropic-version"] = self._anthropic_version

        retryable_retry_used = False
        format_fallback_used = False
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._post_bounded(headers, body)
            except httpx.TimeoutException as exc:
                logger.warning(
                    "llm_request_failed code=timeout attempt=%d model=%s",
                    attempt,
                    self._model,
                )
                if not retryable_retry_used:
                    retryable_retry_used = True
                    await asyncio.sleep(self._retry_delay_seconds)
                    continue
                raise ProviderFailure("LLM_TIMEOUT", "LLM request timed out", retryable=True) from exc
            except httpx.RequestError as exc:
                logger.warning("llm_request_failed code=network model=%s", self._model)
                raise ProviderFailure("LLM_NETWORK_ERROR", "LLM network request failed") from exc

            structured_output_rejected = (
                structured_output_field is not None
                and _rejects_structured_output(response, structured_output_field)
            )
            if (
                structured_output_rejected
                and not format_fallback_used
                and structured_output_field is not None
                and structured_output_field in body
            ):
                # Some compatible relays explicitly reject structured-output
                # fields with a 400/422 response. Retry once without the field;
                # the prompt and local JSON validation still enforce the same
                # output contract. Never interpret a 5xx as feature rejection.
                format_fallback_used = True
                self._structured_output_supported = False
                body = {key: value for key, value in body.items() if key != structured_output_field}
                logger.info(
                    "llm_structured_output_unsupported status=%d model=%s",
                    response.status_code,
                    self._model,
                )
                continue

            if response.status_code in {408, 429} or response.status_code >= 500:
                failure = _safe_http_failure(response.status_code)
                logger.warning(
                    "llm_request_failed code=%s status=%d attempt=%d model=%s",
                    failure.code,
                    response.status_code,
                    attempt,
                    self._model,
                )
                if not retryable_retry_used:
                    retryable_retry_used = True
                    await asyncio.sleep(self._retry_delay_seconds)
                    continue
                raise failure

            if response.status_code >= 400:
                logger.warning(
                    "llm_request_failed code=http status=%d model=%s",
                    response.status_code,
                    self._model,
                )
                raise _safe_http_failure(response.status_code)

            try:
                envelope = response.json()
            except ValueError as exc:
                raise ProviderFailure("LLM_INVALID_RESPONSE", "LLM HTTP response was not JSON") from exc
            if not isinstance(envelope, Mapping):
                raise ProviderFailure("LLM_INVALID_RESPONSE", "LLM HTTP response was not an object")
            if self._api_mode == "responses":
                content = _extract_responses_content(envelope)
            elif self._api_mode == "anthropic_messages":
                content = _extract_anthropic_content(envelope)
            else:
                content = _extract_message_content(envelope)
            return _decode_json_object(content)
