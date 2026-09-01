from __future__ import annotations

import json
from typing import Any

import pytest

from app import provider_check
from app.provider import ProviderFailure


class _FakeProvider:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.closed = False

    async def generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        assert "connection verifier" in system_prompt
        assert "socialgraph_fm_connection_check" in user_prompt
        return self.result

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_verifier_accepts_only_expected_provider_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeProvider({"socialgraph_fm_connection_check": "ok"})
    monkeypatch.setattr(provider_check, "Settings", lambda: object())
    monkeypatch.setattr(provider_check, "OpenAICompatibleProvider", lambda _settings: fake)

    await provider_check.verify_provider()

    assert fake.closed is True


@pytest.mark.anyio
@pytest.mark.parametrize("result", [{}, {"status": "ok"}, {"error": "invalid"}])
async def test_verifier_rejects_arbitrary_json_success_envelopes(
    monkeypatch: pytest.MonkeyPatch,
    result: dict[str, Any],
) -> None:
    fake = _FakeProvider(result)
    monkeypatch.setattr(provider_check, "Settings", lambda: object())
    monkeypatch.setattr(provider_check, "OpenAICompatibleProvider", lambda _settings: fake)

    with pytest.raises(ProviderFailure, match="expected marker"):
        await provider_check.verify_provider()

    assert fake.closed is True


def test_verifier_cli_never_prints_exception_or_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail() -> None:
        raise ProviderFailure(
            "LLM_HTTP_ERROR",
            "upstream rejected secret-key-that-must-not-be-printed",
        )

    monkeypatch.setattr(provider_check, "verify_provider", fail)

    assert provider_check.main() == 1
    output = capsys.readouterr()
    assert "LLM_HTTP_ERROR" in output.err
    assert "secret-key-that-must-not-be-printed" not in output.err
    document = json.loads(output.err)
    assert document == {
        "schemaVersion": "socialgraph-fm.llm-provider-check/1.0",
        "ok": False,
        "code": "LLM_HTTP_ERROR",
    }


def test_verifier_success_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def succeed() -> None:
        return None

    monkeypatch.setattr(provider_check, "verify_provider", succeed)

    assert provider_check.main() == 0
    document = json.loads(capsys.readouterr().out)
    assert document == {
        "schemaVersion": "socialgraph-fm.llm-provider-check/1.0",
        "ok": True,
        "code": "OK",
    }


def test_verifier_emits_only_allowlisted_network_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail() -> None:
        failure = ProviderFailure(
            "LLM_NETWORK_ERROR",
            "unsafe-url-and-secret-key",
            retryable=True,
            diagnostic_code="TLS_CERTIFICATE",
        )
        assert failure.retryable is False
        raise failure

    monkeypatch.setattr(provider_check, "verify_provider", fail)

    assert provider_check.main() == 1
    output = capsys.readouterr()
    assert "unsafe-url-and-secret-key" not in output.err
    assert json.loads(output.err) == {
        "schemaVersion": "socialgraph-fm.llm-provider-check/1.0",
        "ok": False,
        "code": "LLM_NETWORK_ERROR",
        "diagnosticCode": "TLS_CERTIFICATE",
    }


def test_verifier_rejects_non_allowlisted_diagnostic_code() -> None:
    with pytest.raises(ValueError, match="Unsupported provider diagnostic code"):
        provider_check.check_result(
            ok=False,
            code="LLM_NETWORK_ERROR",
            diagnostic_code="unsafe-url-and-secret-key",
        )

    with pytest.raises(ValueError, match="Unsupported provider diagnostic code"):
        ProviderFailure(
            "LLM_NETWORK_ERROR",
            "network failed",
            diagnostic_code="unsafe-url-and-secret-key",
        )


def test_verifier_rejects_network_diagnostic_on_other_error_code() -> None:
    with pytest.raises(ValueError, match="requires LLM_NETWORK_ERROR"):
        provider_check.check_result(
            ok=False,
            code="LLM_AUTH_ERROR",
            diagnostic_code="CONNECT",
        )

    with pytest.raises(ValueError, match="requires LLM_NETWORK_ERROR"):
        ProviderFailure(
            "LLM_AUTH_ERROR",
            "authentication failed",
            diagnostic_code="CONNECT",
        )

    with pytest.raises(ValueError, match="requires a failed check"):
        provider_check.check_result(
            ok=True,
            code="LLM_NETWORK_ERROR",
            diagnostic_code="CONNECT",
        )
