from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

import socialgraph_fm_runtime.llm as llm


def _environment(port: int, mode: str = "chat_completions") -> dict[str, str]:
    return {
        "LLM_API_BASE": f"http://127.0.0.1:{port}/v1",
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "test-model",
        "LLM_API_MODE": mode,
        "LLM_AUTH_SCHEME": (
            "x-api-key" if mode == "anthropic_messages" else "bearer"
        ),
        "LLM_ANTHROPIC_VERSION": (
            llm.DEFAULT_ANTHROPIC_VERSION if mode == "anthropic_messages" else ""
        ),
        "LLM_TIMEOUT_SECONDS": "5",
        "LLM_ALLOW_INSECURE_LOOPBACK": "true",
        "LLM_VERIFICATION_STATUS": "configured_unverified",
    }


def test_api_base_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        llm.normalize_api_base("http://provider.example/v1")
    with pytest.raises(ValueError, match="credentials"):
        llm.normalize_api_base("https://user:secret@provider.example/v1")
    with pytest.raises(ValueError, match="query"):
        llm.normalize_api_base("https://provider.example/v1?tenant=x")
    with pytest.raises(ValueError, match="backslashes"):
        llm.normalize_api_base(r"https://provider.example\v1")
    with pytest.raises(ValueError, match="encoded control"):
        llm.normalize_api_base("https://provider.example/v1%0aheader")
    for unsafe in (
        "https://:443",
        "https://provider.example :443/v1",
        "https://provider.example/v1/%09segment",
        "https://provider.example/v1/%7fsegment",
    ):
        with pytest.raises(ValueError):
            llm.normalize_api_base(unsafe)
    assert (
        llm.normalize_api_base("http://127.0.0.1:11434/v1/", allow_insecure_loopback=True)
        == "http://127.0.0.1:11434/v1"
    )


@pytest.mark.parametrize(
    ("base", "mode", "endpoint"),
    [
        (
            "https://provider.example/v1",
            "chat_completions",
            "https://provider.example/v1/chat/completions",
        ),
        (
            "https://provider.example/v1/chat/completions",
            "responses",
            "https://provider.example/v1/responses",
        ),
        (
            "https://api.anthropic.com/v1",
            "anthropic_messages",
            "https://api.anthropic.com/v1/messages",
        ),
        (
            "https://api.anthropic.com/v1/messages",
            "anthropic_messages",
            "https://api.anthropic.com/v1/messages",
        ),
    ],
)
def test_endpoint_derivation_accepts_roots_and_known_full_endpoints(
    base: str, mode: str, endpoint: str
) -> None:
    assert llm.derive_api_endpoint(base, mode) == endpoint


@pytest.mark.parametrize(
    ("supplied", "stored"),
    [
        ("https://relay.example", "https://relay.example/v1"),
        ("https://relay.example/v1", "https://relay.example/v1"),
        (
            "https://relay.example/v1/messages",
            "https://relay.example/v1/messages",
        ),
    ],
)
def test_custom_relay_root_v1_and_full_endpoint_normalization(
    supplied: str, stored: str
) -> None:
    normalized = llm.normalize_relay_api_base(supplied)
    assert normalized == stored
    assert (
        llm.derive_api_endpoint(normalized, "anthropic_messages")
        == "https://relay.example/v1/messages"
    )


def test_api_key_stdin_reads_one_complete_line() -> None:
    presets = Path(__file__).resolve().parents[3] / "scripts" / "config" / "llm-presets.json"
    environment = llm.configure_environment(
        preset_catalog=presets,
        preset="custom",
        api_base="https://provider.example/v1",
        model="model",
        api_mode="chat_completions",
        timeout_seconds=15,
        api_key_stdin=True,
        allow_insecure_loopback=False,
        stdin=io.StringIO("complete-key-value\nignored"),
    )
    assert environment["LLM_API_KEY"] == "complete-key-value"


@pytest.mark.parametrize(
    ("preset", "api_base", "api_mode"),
    [
        ("openai_responses", "https://api.openai.com/v1", "responses"),
        ("deepseek", "https://api.deepseek.com", "chat_completions"),
        ("glm", "https://open.bigmodel.cn/api/paas/v4", "chat_completions"),
        ("anthropic", "https://api.anthropic.com/v1", "anthropic_messages"),
    ],
)
def test_tracked_provider_presets_resolve_to_the_documented_protocol(
    preset: str, api_base: str, api_mode: str
) -> None:
    catalog = Path(__file__).resolve().parents[3] / "scripts" / "config" / "llm-presets.json"
    environment = llm.configure_environment(
        preset_catalog=catalog,
        preset=preset,
        api_base=None,
        model="user-selected-model",
        api_mode=None,
        timeout_seconds=15,
        api_key_stdin=True,
        allow_insecure_loopback=False,
        stdin=io.StringIO("test-provider-key\n"),
    )

    assert environment["LLM_API_BASE"] == api_base
    assert environment["LLM_API_MODE"] == api_mode
    assert environment["LLM_AUTH_SCHEME"] == (
        "x-api-key" if api_mode == "anthropic_messages" else "bearer"
    )


def test_overriding_a_direct_preset_with_a_relay_root_adds_v1() -> None:
    catalog = Path(__file__).resolve().parents[3] / "scripts" / "config" / "llm-presets.json"
    environment = llm.configure_environment(
        preset_catalog=catalog,
        preset="openai_responses",
        api_base="https://relay.example",
        model="relay-model",
        api_mode="responses",
        timeout_seconds=15,
        api_key_stdin=True,
        allow_insecure_loopback=False,
        stdin=io.StringIO("test-relay-key\n"),
    )

    assert environment["LLM_API_BASE"] == "https://relay.example/v1"
    assert llm.derive_api_endpoint(environment["LLM_API_BASE"], "responses") == (
        "https://relay.example/v1/responses"
    )


def test_noninteractive_configuration_never_adopts_an_ambient_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    presets = Path(__file__).resolve().parents[3] / "scripts" / "config" / "llm-presets.json"
    monkeypatch.setenv("LLM_API_KEY", "ambient-parent-key")

    with pytest.raises(RuntimeError, match="api-key-stdin"):
        llm.configure_environment(
            preset_catalog=presets,
            preset="custom",
            api_base="https://provider.example/v1",
            model="model",
            api_mode="chat_completions",
            timeout_seconds=15,
            api_key_stdin=False,
            allow_insecure_loopback=False,
            stdin=io.StringIO(),
        )


def test_acl_helpers_do_not_inherit_vendor_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ambient-deepseek-key")
    monkeypatch.setenv("ZHIPUAI_API_KEY", "ambient-glm-key")

    selected = llm._acl_process_environment(tmp_path / "private.env")

    assert "ANTHROPIC_API_KEY" not in selected
    assert "DEEPSEEK_API_KEY" not in selected
    assert "ZHIPUAI_API_KEY" not in selected


def test_private_configuration_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if os.name == "nt":
        monkeypatch.setattr(llm, "_windows_protect", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(llm, "_windows_assert_protected", lambda *_args, **_kwargs: None)
    path = tmp_path / "private" / "llm.env"
    environment = _environment(443)
    llm.write_private_environment(path, environment)
    assert llm.parse_private_environment(path) == {**environment, "LOG_LEVEL": "INFO"}
    assert not list(path.parent.glob("*.tmp"))
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700


def test_legacy_private_configuration_gains_protocol_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if os.name == "nt":
        monkeypatch.setattr(llm, "_windows_assert_protected", lambda *_args: None)
    path = tmp_path / "legacy.env"
    path.write_text(
        "\n".join(
            (
                "LLM_API_BASE=https://provider.example/v1",
                "LLM_API_KEY=legacy-key",
                "LLM_MODEL=legacy-model",
                "LLM_API_MODE=chat_completions",
                "LLM_TIMEOUT_SECONDS=15",
                "LLM_ALLOW_INSECURE_LOOPBACK=false",
                "LLM_VERIFICATION_STATUS=configured_unverified",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    if os.name != "nt":
        os.chmod(tmp_path, 0o700)
        os.chmod(path, 0o600)

    parsed = llm.parse_private_environment(path)

    assert parsed["LLM_AUTH_SCHEME"] == "bearer"
    assert parsed["LLM_ANTHROPIC_VERSION"] == ""


def test_configuration_summary_never_contains_key() -> None:
    environment = _environment(443, mode="anthropic_messages")
    environment["LLM_API_BASE"] = "https://api.anthropic.com/v1"
    summary = llm.configuration_summary(environment, preset_id="anthropic")

    assert summary["endpoint"] == "https://api.anthropic.com/v1/messages"
    assert summary["authScheme"] == "x-api-key"
    assert summary["keyConfigured"] is True
    assert "test-key" not in repr(summary)


class _InteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_interactive_custom_anthropic_flow_confirms_every_public_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    presets = Path(__file__).resolve().parents[3] / "scripts" / "config" / "llm-presets.json"
    stdin = _InteractiveInput(
        "6\n"
        "https://relay.example/v1\n"
        "\n"
        "relay-model\n"
        "20\n"
        "2\n"
        "y\n"
    )
    stdout = io.StringIO()
    monkeypatch.setattr(llm.getpass, "getpass", lambda _prompt: "relay-secret")

    environment = llm.configure_environment(
        preset_catalog=presets,
        preset=None,
        api_base=None,
        model=None,
        api_mode=None,
        auth_scheme=None,
        anthropic_version=None,
        timeout_seconds=15,
        api_key_stdin=False,
        allow_insecure_loopback=False,
        stdin=stdin,
        stdout=stdout,
    )

    assert environment["LLM_API_MODE"] == "anthropic_messages"
    assert environment["LLM_AUTH_SCHEME"] == "bearer"
    assert environment["LLM_ANTHROPIC_VERSION"] == "2023-06-01"
    assert environment["LLM_TIMEOUT_SECONDS"] == "20"
    assert environment[llm.CONFIGURATION_TEST_INTENT_KEY] == "true"
    output = stdout.getvalue()
    assert "https://relay.example/v1/messages" in output
    assert "may incur a charge" in output
    assert "relay-secret" not in output


def test_windows_acl_rejects_an_explicit_everyone_grant(tmp_path: Path) -> None:
    document = {
        "protected": True,
        "rules": [
            {"sid": "S-1-5-21-1-2-3-1001", "type": "Allow"},
            {"sid": "S-1-1-0", "type": "Allow"},
        ],
    }
    with pytest.raises(RuntimeError, match="unexpected principal"):
        llm._validate_windows_acl(document, "S-1-5-21-1-2-3-1001", tmp_path)


def test_existing_private_configuration_permissions_are_migrated_without_rewrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if os.name == "nt":
        monkeypatch.setattr(llm, "_windows_protect", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(llm, "_windows_assert_protected", lambda *_args, **_kwargs: None)
    path = tmp_path / "private" / "llm.env"
    path.parent.mkdir()
    original = "\n".join(f"{name}={value}" for name, value in _environment(443).items()) + "\n"
    path.write_text(original, encoding="utf-8")
    if os.name != "nt":
        os.chmod(path.parent, 0o755)
        os.chmod(path, 0o644)
    llm.migrate_private_environment_permissions(path)
    assert path.read_text(encoding="utf-8") == original
    if os.name != "nt":
        assert path.parent.stat().st_mode & 0o777 == 0o700
        assert path.stat().st_mode & 0o777 == 0o600
