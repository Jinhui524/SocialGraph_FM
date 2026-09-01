from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from socialgraph_fm_runtime import llm


class _TTYStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize(
    "value",
    [
        "http://remote.example/v1",
        "https://user:pass@provider.example/v1",
        "https://https://provider.example/v1",
        "https://provider.example/v1?secret=x",
        "https://provider.example/%0aheader",
    ],
)
def test_api_base_validation_is_fail_closed(value: str) -> None:
    with pytest.raises(ValueError):
        llm.normalize_api_base(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://provider.example", "https://provider.example/v1"),
        ("https://provider.example/v1/", "https://provider.example/v1"),
        (
            "https://provider.example/v1/chat/completions",
            "https://provider.example/v1",
        ),
        ("http://127.0.0.1:18991", "http://127.0.0.1:18991/v1"),
        ("http://localhost:18991/v1", "http://localhost:18991/v1"),
    ],
)
def test_base_root_v1_and_full_endpoint_are_normalized(
    value: str, expected: str
) -> None:
    assert llm.normalize_api_base(value) == expected
    assert llm.derive_api_endpoint(value) == f"{expected}/chat/completions"


@pytest.mark.parametrize(
    "value",
    [
        "https://api.deepseek.com",
        "https://api.deepseek.com/v1",
        "https://api.deepseek.com/chat/completions",
        "https://api.deepseek.com/v1/chat/completions",
    ],
)
def test_deepseek_compatibility_forms_use_the_official_root(value: str) -> None:
    assert llm.normalize_api_base(value) == "https://api.deepseek.com"
    assert llm.derive_api_endpoint(value) == (
        "https://api.deepseek.com/chat/completions"
    )


def test_non_chat_protocols_are_not_supported() -> None:
    with pytest.raises(ValueError, match="Only"):
        llm.derive_api_endpoint("https://provider.example/v1", "responses")


def test_noninteractive_configuration_reads_three_fields_and_strips_ps51_bom() -> None:
    environment = llm.configure_environment(
        api_base="https://provider.example",
        model="model-id",
        api_key_stdin=True,
        stdin=io.StringIO("\ufefftest-key\nignored\n"),
        stdout=io.StringIO(),
    )
    assert environment == {
        "LLM_API_BASE": "https://provider.example/v1",
        "LLM_MODEL": "model-id",
        "LLM_API_KEY": "test-key",
    }


def test_noninteractive_configuration_never_adopts_ambient_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    with pytest.raises(RuntimeError, match="--api-key-stdin"):
        llm.configure_environment(
            api_base="https://provider.example/v1",
            model="model-id",
            api_key_stdin=False,
            stdin=io.StringIO(""),
            stdout=io.StringIO(),
        )


def test_provider_presets_have_stable_order_and_addresses() -> None:
    assert llm.LLM_PROVIDER_PRESETS == (
        ("OpenAI 官方", "https://api.openai.com/v1"),
        ("DeepSeek 官方", "https://api.deepseek.com"),
        ("通义千问", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        (
            "Gemini OpenAI-compatible",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
        ("MiniMax 中国", "https://api.minimaxi.com/v1"),
        ("MiniMax 国际", "https://api.minimax.io/v1"),
        ("OpenRouter", "https://openrouter.ai/api/v1"),
        ("自定义 OpenAI-compatible", None),
    )


@pytest.mark.parametrize(
    ("selection", "expected_base", "expected_provider"),
    [
        ("1", "https://api.openai.com/v1", "OpenAI 官方"),
        ("2", "https://api.deepseek.com", "DeepSeek 官方"),
        (
            "3",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "通义千问",
        ),
        (
            "4",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "Gemini OpenAI-compatible",
        ),
        ("5", "https://api.minimaxi.com/v1", "MiniMax 中国"),
        ("6", "https://api.minimax.io/v1", "MiniMax 国际"),
        ("7", "https://openrouter.ai/api/v1", "OpenRouter"),
    ],
)
def test_interactive_provider_preset_only_prefills_editable_api_base(
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
    expected_base: str,
    expected_provider: str,
) -> None:
    monkeypatch.setattr(llm.getpass, "getpass", lambda _prompt: "hidden-key")
    stdout = io.StringIO()
    environment = llm.configure_environment(
        api_base=None,
        model=None,
        api_key_stdin=False,
        stdin=_TTYStringIO(f"{selection}\n\nmodel-from-user\n"),
        stdout=stdout,
    )

    assert environment == {
        "LLM_API_BASE": expected_base,
        "LLM_MODEL": "model-from-user",
        "LLM_API_KEY": "hidden-key",
    }
    output = stdout.getvalue()
    assert f"服务商：{expected_provider}" in output
    assert "hidden-key" not in output


def test_interactive_custom_provider_requires_and_accepts_an_editable_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm.getpass, "getpass", lambda _prompt: "hidden-key")
    stdout = io.StringIO()
    environment = llm.configure_environment(
        api_base=None,
        model=None,
        api_key_stdin=False,
        stdin=_TTYStringIO("8\nhttps://relay.example/openai\nrelay-model\n"),
        stdout=stdout,
    )

    assert environment["LLM_API_BASE"] == "https://relay.example/openai"
    assert environment["LLM_MODEL"] == "relay-model"
    assert "服务商：自定义 OpenAI-compatible" in stdout.getvalue()


def test_interactive_provider_selection_has_no_default_and_stops_after_three_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm.getpass,
        "getpass",
        lambda _prompt: pytest.fail("API Key must not be requested"),
    )
    stdout = io.StringIO()
    with pytest.raises(RuntimeError, match="连续三次无效"):
        llm.configure_environment(
            api_base="https://provider.example/v1",
            model="model-id",
            api_key_stdin=False,
            stdin=_TTYStringIO("\n0\ninvalid\n"),
            stdout=stdout,
        )

    assert "请输入序号 [1-8] [" not in stdout.getvalue()


def test_interactive_provider_selection_eof_is_a_clean_cancellation() -> None:
    with pytest.raises(RuntimeError, match="cancelled"):
        llm.configure_environment(
            api_base=None,
            model=None,
            api_key_stdin=False,
            stdin=_TTYStringIO(""),
            stdout=io.StringIO(),
        )


def test_interactive_model_is_required_before_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm.getpass,
        "getpass",
        lambda _prompt: pytest.fail("API Key must not be requested"),
    )
    with pytest.raises(ValueError, match="Model ID cannot be empty"):
        llm.configure_environment(
            api_base=None,
            model=None,
            api_key_stdin=False,
            stdin=_TTYStringIO("1\n\n\n"),
            stdout=io.StringIO(),
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission assertion")
def test_private_configuration_contains_only_three_fields(tmp_path: Path) -> None:
    path = tmp_path / "private" / "socialgraph-api.env"
    environment = {
        "LLM_API_BASE": "https://provider.example/v1",
        "LLM_MODEL": "model-id",
        "LLM_API_KEY": "test-key",
    }
    llm.write_private_environment(path, environment)

    assert llm.parse_private_environment(path) == environment
    assert path.read_text(encoding="utf-8").splitlines() == [
        "LLM_API_BASE=https://provider.example/v1",
        "LLM_MODEL=model-id",
        "LLM_API_KEY=test-key",
    ]
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX legacy fixture")
def test_legacy_chat_bearer_configuration_is_reduced_to_three_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private" / "socialgraph-api.env"
    path.parent.mkdir(mode=0o700)
    path.write_text(
        "LLM_API_BASE=https://provider.example/v1\n"
        "LLM_MODEL=model-id\n"
        "LLM_API_KEY=test-key\n"
        "LLM_API_MODE=chat_completions\n"
        "LLM_AUTH_SCHEME=bearer\n"
        "LLM_TIMEOUT_SECONDS=30\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    assert llm.parse_private_environment(path) == {
        "LLM_API_BASE": "https://provider.example/v1",
        "LLM_MODEL": "model-id",
        "LLM_API_KEY": "test-key",
    }


@pytest.mark.skipif(os.name == "nt", reason="POSIX legacy fixture")
def test_legacy_responses_or_anthropic_configuration_requires_reentry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private" / "socialgraph-api.env"
    path.parent.mkdir(mode=0o700)
    path.write_text(
        "LLM_API_BASE=https://provider.example/v1\n"
        "LLM_MODEL=model-id\n"
        "LLM_API_KEY=test-key\n"
        "LLM_API_MODE=responses\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    with pytest.raises(RuntimeError, match="retired"):
        llm.parse_private_environment(path)


def test_configuration_summary_never_contains_key() -> None:
    summary = llm.configuration_summary(
        {
            "LLM_API_BASE": "https://provider.example/v1",
            "LLM_MODEL": "model-id",
            "LLM_API_KEY": "secret-value",
        }
    )
    assert "secret-value" not in repr(summary)
    assert summary["endpoint"] == "https://provider.example/v1/chat/completions"
    assert summary["provider"] == "自定义 OpenAI-compatible"


def test_provider_inference_requires_an_exact_official_hostname() -> None:
    assert llm.provider_label("https://api.openai.com/v1") == "OpenAI 官方"
    assert (
        llm.provider_label("https://api.openai.com.evil.example/v1")
        == "自定义 OpenAI-compatible"
    )


def test_windows_acl_validator_rejects_unexpected_principal(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unexpected principal"):
        llm._validate_windows_acl(
            {
                "protected": True,
                "rules": [
                    {"sid": "S-1-5-21-1", "type": "Allow"},
                    {"sid": "S-1-1-0", "type": "Allow"},
                ],
            },
            "S-1-5-21-1",
            tmp_path,
        )
