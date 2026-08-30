from __future__ import annotations

import io
import argparse
from pathlib import Path

import pytest

from socialgraph_fm_runtime import cli
from socialgraph_fm_runtime.cli import build_parser
from socialgraph_fm_runtime.layout import RuntimeLayout


def test_public_cli_surface_is_stable() -> None:
    parser = build_parser()
    setup = parser.parse_args(
        [
            "setup",
            "--profile",
            "cpu",
            "--env-mode",
            "reuse",
            "--api-python",
            "api-python",
            "--gfm-python",
            "gfm-python",
            "--bootstrap-python",
            "bootstrap-python",
            "--skip-api",
            "--skip-web",
            "--gfm-text",
        ]
    )
    assert setup.command == "setup"
    assert setup.profile == "cpu"
    assert setup.env_mode == "reuse"
    assert setup.gfm_text_profile is True

    wheel_setup = parser.parse_args(
        [
            "setup",
            "--wheel-profile",
            "windows-x86_64-cu130-pt212",
            "--device-policy",
            "auto",
        ]
    )
    assert wheel_setup.wheel_profile == "windows-x86_64-cu130-pt212"
    assert wheel_setup.device_policy == "auto"

    onboard = parser.parse_args(
        [
            "onboard",
            "--wheel-profile",
            "cpu",
            "--device-policy",
            "cpu",
            "--preset",
            "custom_anthropic",
            "--api-base",
            "https://relay.example/v1",
            "--model",
            "claude-model",
            "--protocol",
            "anthropic_messages",
            "--auth-scheme",
            "bearer",
            "--api-key-stdin",
        ]
    )
    assert onboard.command == "onboard"
    assert onboard.wheel_profile == "cpu"
    assert onboard.api_mode == "anthropic_messages"
    assert onboard.auth_scheme == "bearer"
    assert onboard.api_key_stdin is True

    anthropic = parser.parse_args(
        [
            "configure-llm",
            "--preset",
            "custom_anthropic",
            "--api-mode",
            "anthropic_messages",
            "--auth-scheme",
            "x-api-key",
            "--anthropic-version",
            "2023-06-01",
        ]
    )
    assert anthropic.api_mode == "anthropic_messages"
    assert anthropic.auth_scheme == "x-api-key"

    exported = parser.parse_args(
        ["export-github", "--repository", "repo", "--zip", "repo.zip"]
    )
    assert exported.command == "export-github"

    development = parser.parse_args(
        [
            "dev",
            "--llm-mode",
            "disabled",
            "--no-llm-prompt",
            "--reconfigure-llm",
            "--test-llm",
        ]
    )
    assert development.command == "dev"
    assert development.no_llm_prompt is True

    diagnostic = parser.parse_args(["doctor", "--full", "--test-llm", "--json"])
    assert diagnostic.full is True
    assert diagnostic.json is True


def test_configure_test_flags_are_mutually_exclusive() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "configure-llm",
            "--preset",
            "custom",
            "--api-base",
            "https://provider.example/v1",
            "--model",
            "model",
            "--api-key-stdin",
            "--skip-llm-test",
        ]
    )
    assert arguments.test_llm is False


class _InteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_setup_profile_is_never_silently_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = build_parser()
    arguments = parser.parse_args(["setup"])
    assert arguments.profile is None

    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(""))
    with pytest.raises(RuntimeError, match="explicit --wheel-profile"):
        cli._select_setup_profile(arguments.profile)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("\n", "cpu"), ("2\n", "cuda"), ("offline\n", "offline")],
)
def test_interactive_setup_profile_defaults_to_cpu(
    monkeypatch: pytest.MonkeyPatch, answer: str, expected: str
) -> None:
    monkeypatch.setattr(cli.sys, "stdin", _InteractiveInput(answer))
    assert cli._select_setup_profile(None) == expected


def test_start_without_setup_prints_the_onboarding_and_compatible_paths(tmp_path) -> None:
    arguments = argparse.Namespace(
        reconfigure_llm=False,
        llm_mode="required",
        no_llm_prompt=True,
        test_llm=False,
    )
    with pytest.raises(RuntimeError, match="configure-llm") as captured:
        cli._start(RuntimeLayout(tmp_path), arguments, development=False)
    assert "socialgraph.py onboard" in str(captured.value)
    assert "setup --profile offline --skip-web" in str(captured.value)
    assert "setup --wheel-profile cpu|cuda" in str(captured.value)
    assert "start --llm-mode required" in str(captured.value)


def test_onboard_configures_llm_before_selecting_wheels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    order: list[str] = []
    runtime = object()
    monkeypatch.setattr(cli.sys, "stdin", _InteractiveInput(""))
    monkeypatch.setattr(
        cli,
        "_configure",
        lambda _layout, _arguments=None, *, api_python=None: order.append(
            f"llm:{api_python}"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_select_setup_profile",
        lambda value: order.append("wheel") or (value or "cpu"),
    )

    def fake_setup(_layout, options):
        order.append("api")
        assert options.profile == "auto"
        assert options.after_api is not None
        assert options.after_api(tmp_path / "api-python") == "cpu"
        order.append("gfm")
        return runtime

    monkeypatch.setattr(cli, "setup", fake_setup)
    arguments = argparse.Namespace(
        wheel_profile=None,
        device_policy="auto",
        env_mode="auto",
        api_python=None,
        gfm_python=None,
        bootstrap_python=None,
        skip_web=False,
        gfm_text_profile=False,
        preset=None,
        api_base=None,
        model=None,
        api_mode=None,
        auth_scheme=None,
        anthropic_version=None,
        timeout_seconds=15,
        api_key_stdin=False,
        allow_insecure_loopback=False,
    )

    assert cli._onboard(RuntimeLayout(tmp_path), arguments) is runtime
    assert order == ["api", f"llm:{tmp_path / 'api-python'}", "wheel", "gfm"]


def test_noninteractive_onboard_requires_all_automation_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("test-key\n"))
    arguments = build_parser().parse_args(
        ["onboard", "--wheel-profile", "cpu", "--preset", "custom"]
    )

    with pytest.raises(RuntimeError, match=r"--model.*--api-key-stdin"):
        cli._onboard(RuntimeLayout(tmp_path), arguments)


def test_noninteractive_onboard_passes_explicit_llm_configuration_after_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    order: list[str] = []
    runtime = object()
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("test-key\n"))

    def fake_configure(_layout, selected, *, api_python=None):
        order.append(f"llm:{api_python}")
        assert selected.preset == "custom"
        assert selected.api_base == "http://127.0.0.1:18991/v1"
        assert selected.model == "test-model"
        assert selected.api_mode == "responses"
        assert selected.api_key_stdin is True
        assert selected.test_llm is True

    monkeypatch.setattr(cli, "_configure", fake_configure)

    def fake_setup(_layout, options):
        order.append("api")
        assert options.after_api is not None
        assert options.after_api(tmp_path / "api-python") == "cpu"
        order.append("gfm")
        return runtime

    monkeypatch.setattr(cli, "setup", fake_setup)
    arguments = build_parser().parse_args(
        [
            "onboard",
            "--wheel-profile",
            "cpu",
            "--preset",
            "custom",
            "--api-base",
            "http://127.0.0.1:18991/v1",
            "--model",
            "test-model",
            "--api-mode",
            "responses",
            "--auth-scheme",
            "bearer",
            "--api-key-stdin",
            "--allow-insecure-loopback",
        ]
    )

    assert cli._onboard(RuntimeLayout(tmp_path), arguments) is runtime
    assert order == ["api", f"llm:{tmp_path / 'api-python'}", "gfm"]


def test_setup_rejects_conflicting_legacy_and_wheel_selectors() -> None:
    with pytest.raises(RuntimeError, match="either --profile or --wheel-profile"):
        cli._setup_selection(
            argparse.Namespace(profile="cpu", wheel_profile="windows-x86_64-cpu-pt28")
        )


def test_failed_llm_check_can_edit_only_the_model_and_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = {
        "LLM_API_BASE": "https://provider.example/v1",
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "wrong-model",
        "LLM_API_MODE": "responses",
        "LLM_AUTH_SCHEME": "bearer",
        "LLM_ANTHROPIC_VERSION": "",
        "LLM_TIMEOUT_SECONDS": "15",
        "LLM_ALLOW_INSECURE_LOOPBACK": "false",
        "LLM_VERIFICATION_STATUS": "configured_unverified",
        cli.CONFIGURATION_TEST_INTENT_KEY: "true",
    }
    saved: dict[str, str] = {}
    attempts = 0

    monkeypatch.setattr(cli, "configure_environment", lambda **_kwargs: dict(environment))

    def fake_test(_layout, selected, **_kwargs):
        nonlocal attempts
        attempts += 1
        if selected["LLM_MODEL"] == "wrong-model":
            raise RuntimeError("LLM connection check failed: REQUEST_OR_MODEL")

    monkeypatch.setattr(cli, "test_llm_configuration", fake_test)
    monkeypatch.setattr(
        cli,
        "write_private_environment",
        lambda _path, selected: saved.update(selected),
    )
    monkeypatch.setattr(cli.sys, "stdin", _InteractiveInput("m\nright-model\n"))

    cli._configure(RuntimeLayout(tmp_path), _default_configuration_arguments_for_test())

    assert attempts == 2
    assert saved["LLM_MODEL"] == "right-model"
    assert saved["LLM_VERIFICATION_STATUS"] == "call_succeeded"


def _default_configuration_arguments_for_test() -> argparse.Namespace:
    arguments = cli._default_configuration_arguments()
    arguments.test_llm = True
    return arguments
