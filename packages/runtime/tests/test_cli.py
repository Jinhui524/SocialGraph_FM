from __future__ import annotations

import io
from pathlib import Path

import pytest

from socialgraph_fm_runtime import cli
from socialgraph_fm_runtime.cli import build_parser
from socialgraph_fm_runtime.layout import RuntimeLayout


def test_public_cli_has_only_five_commands() -> None:
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {"onboard", "configure-llm", "start", "stop", "doctor"}

    onboard = parser.parse_args(
        [
            "onboard",
            "--api-base",
            "https://provider.example/v1",
            "--model",
            "model-id",
            "--api-key-stdin",
        ]
    )
    assert onboard.api_base == "https://provider.example/v1"
    assert onboard.model == "model-id"
    assert onboard.api_key_stdin is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["setup"],
        ["dev"],
        ["onboard", "--wheel-profile", "cpu"],
        ["start", "--llm-mode", "disabled"],
        ["configure-llm", "--preset", "custom"],
    ],
)
def test_retired_profiles_protocols_and_fallback_switches_are_rejected(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_noninteractive_configuration_requires_exactly_three_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("test-key\n"))
    arguments = build_parser().parse_args(["onboard", "--api-base", "https://x.example/v1"])
    with pytest.raises(RuntimeError, match="--api-base, --model, and --api-key-stdin"):
        cli._collect_configuration(arguments)


def test_onboard_validates_llm_inside_new_runtime_before_saving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    order: list[str] = []
    environment = {
        "LLM_API_BASE": "https://provider.example/v1",
        "LLM_MODEL": "model-id",
        "LLM_API_KEY": "test-key",
    }
    runtime = object()
    monkeypatch.setattr(cli, "_collect_configuration", lambda _arguments: environment)

    def fake_test(_layout, selected, *, runtime_python=None, **_kwargs):
        assert selected == environment
        order.append(f"test:{runtime_python}")

    monkeypatch.setattr(cli, "test_llm_configuration", fake_test)
    monkeypatch.setattr(
        cli,
        "write_private_environment",
        lambda _path, selected: order.append(f"save:{selected['LLM_MODEL']}"),
    )

    def fake_setup(_layout, options):
        order.append("install")
        assert options.after_runtime is not None
        options.after_runtime(tmp_path / "var" / "runtime" / "python")
        return runtime

    monkeypatch.setattr(cli, "setup", fake_setup)
    arguments = build_parser().parse_args(
        [
            "onboard",
            "--api-base",
            "https://provider.example/v1",
            "--model",
            "model-id",
            "--api-key-stdin",
        ]
    )

    assert cli._onboard(RuntimeLayout(tmp_path), arguments) is runtime
    assert order == [
        "install",
        f"test:{tmp_path / 'var' / 'runtime' / 'python'}",
        "save:model-id",
    ]


def test_failed_llm_validation_never_writes_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writes: list[object] = []
    monkeypatch.setattr(
        cli,
        "test_llm_configuration",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("LLM connection check failed: AUTH")
        ),
    )
    monkeypatch.setattr(
        cli, "write_private_environment", lambda *_args: writes.append(object())
    )

    with pytest.raises(RuntimeError, match="AUTH"):
        cli._save_verified_configuration(
            RuntimeLayout(tmp_path),
            {
                "LLM_API_BASE": "https://provider.example/v1",
                "LLM_MODEL": "model-id",
                "LLM_API_KEY": "test-key",
            },
            runtime_python=tmp_path / "python",
        )
    assert writes == []
