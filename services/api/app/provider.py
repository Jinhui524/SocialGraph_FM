from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

import httpx

from .config import Settings, derive_llm_endpoint

logger = logging.getLogger(__name__)
MAX_LLM_RESPONSE_BYTES = 2 * 1024 * 1024
GEMINI_CLIENT_HEADER = "socialgraph-fm/1.0.0"
MINIMAX_HOSTS = frozenset({"api.minimaxi.com", "api.minimax.io"})
OPENROUTER_HOST = "openrouter.ai"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


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
    ) -> Literal["configured_unverified", "call_succeeded", "error"]: ...

    async def generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


def _endpoint_url(api_base: str) -> str:
    return derive_llm_endpoint(api_base)


def _exact_hostname(url: str) -> str:
    return (urlsplit(url).hostname or "").rstrip(".").lower()


def _uses_completion_token_limit(model: str) -> bool:
    last_segment = model.rsplit("/", 1)[-1].lower()
    return last_segment.startswith("gpt-5") or bool(
        re.fullmatch(r"o[1-9](?:[-._:].*)?", last_segment)
    )


def _is_minimax_m3(model: str) -> bool:
    return model.rsplit("/", 1)[-1].lower().startswith("minimax-m3")


def _strip_leading_think_blocks(content: str) -> str:
    """Strip consecutive, completely closed reasoning blocks from the start only."""

    cleaned = content.strip()
    while cleaned.startswith(_THINK_OPEN):
        close_index = cleaned.find(_THINK_CLOSE, len(_THINK_OPEN))
        if close_index < 0:
            break
        cleaned = cleaned[close_index + len(_THINK_CLOSE) :].lstrip()
    return cleaned


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


def _decode_json_object(content: str) -> dict[str, Any]:
    cleaned = _strip_leading_think_blocks(content)
    if cleaned.startswith(_THINK_OPEN):
        raise ProviderFailure(
            "LLM_INVALID_RESPONSE", "LLM reasoning block was not completely closed"
        )
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
    """Minimal client for the one supported OpenAI Chat Completions contract."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not settings.llm_configured:
            raise ValueError("OpenAICompatibleProvider requires complete LLM configuration")
        assert settings.llm_api_base is not None
        assert settings.llm_api_key is not None
        assert settings.llm_model is not None
        self._model = settings.llm_model
        self._url = _endpoint_url(settings.llm_api_base)
        self._hostname = _exact_hostname(self._url)
        self._api_key = settings.llm_api_key.get_secret_value()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=15.0,
            trust_env=False,
            follow_redirects=False,
        )
        self._connection_status: Literal[
            "configured_unverified", "call_succeeded", "error"
        ] = "configured_unverified"

    @property
    def model(self) -> str:
        return self._model

    @property
    def connection_status(
        self,
    ) -> Literal["configured_unverified", "call_succeeded", "error"]:
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
            self._connection_status = "error"
            raise
        self._connection_status = "call_succeeded"
        return result

    async def _generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if self._hostname == "api.openai.com":
            body.update({"max_completion_tokens": 700})
        elif self._hostname == "api.deepseek.com":
            body.update(
                {
                    "max_tokens": 700,
                    "thinking": {"type": "disabled"},
                }
            )
        elif self._hostname in MINIMAX_HOSTS:
            body.update(
                {
                    "max_completion_tokens": 700,
                    "reasoning_split": True,
                }
            )
            if _is_minimax_m3(self._model):
                body["thinking"] = {"type": "disabled"}
        elif self._hostname == OPENROUTER_HOST:
            body.update({"max_tokens": 700})
        elif _uses_completion_token_limit(self._model):
            body.update({"max_completion_tokens": 700})
        else:
            body.update({"max_tokens": 700})
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        if self._hostname == "generativelanguage.googleapis.com":
            headers["x-goog-api-client"] = GEMINI_CLIENT_HEADER
        try:
            response = await self._post_bounded(headers, body)
        except httpx.TimeoutException as exc:
            logger.warning("llm_request_failed code=timeout model=%s", self._model)
            raise ProviderFailure("LLM_TIMEOUT", "LLM request timed out", retryable=True) from exc
        except httpx.RequestError as exc:
            logger.warning("llm_request_failed code=network model=%s", self._model)
            raise ProviderFailure(
                "LLM_NETWORK_ERROR", "LLM network request failed", retryable=True
            ) from exc
        if response.status_code >= 400:
            failure = _safe_http_failure(response.status_code)
            logger.warning(
                "llm_request_failed code=%s status=%d model=%s",
                failure.code,
                response.status_code,
                self._model,
            )
            raise failure
        try:
            envelope = response.json()
        except ValueError as exc:
            raise ProviderFailure("LLM_INVALID_RESPONSE", "LLM HTTP response was not JSON") from exc
        if not isinstance(envelope, Mapping):
            raise ProviderFailure("LLM_INVALID_RESPONSE", "LLM HTTP response was not an object")
        return _decode_json_object(_extract_message_content(envelope))
