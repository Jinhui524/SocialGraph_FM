from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

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


def test_onboard_commits_runtime_before_validating_llm_and_saving(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    order: list[str] = []
    environment = {
        "LLM_API_BASE": "https://provider.example/v1",
        "LLM_MODEL": "model-id",
        "LLM_API_KEY": "test-key",
    }
    runtime_python = tmp_path / "var" / "runtime" / "python"
    runtime = SimpleNamespace(interpreter={"path": str(runtime_python)})
    def fake_collect(_arguments):
        order.append("collect")
        return environment

    monkeypatch.setattr(cli, "_collect_configuration", fake_collect)

    def fake_test(_layout, selected, *, runtime_python=None, **_kwargs):
        assert selected == environment
        order.append(f"test:{runtime_python}")

    monkeypatch.setattr(cli, "test_llm_configuration", fake_test)
    monkeypatch.setattr(
        cli,
        "write_private_environment",
        lambda _path, selected: order.append(f"save:{selected['LLM_MODEL']}"),
    )

    def fake_setup(_layout):
        order.append("setup")
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
        "collect",
        "setup",
        f"test:{runtime_python}",
        "save:model-id",
    ]
    stages = capsys.readouterr().err
    assert stages.index("LLM: validating") < stages.index("complete:")


def test_failed_onboard_llm_validation_preserves_committed_runtime_and_old_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    layout = RuntimeLayout(tmp_path)
    runtime_python = layout.runtime_environment / "python"
    environment = {
        "LLM_API_BASE": "https://candidate.example/v1",
        "LLM_MODEL": "candidate-model",
        "LLM_API_KEY": "candidate-key",
    }
    old_configuration = b"LLM_API_BASE=https://old.example/v1\nLLM_MODEL=old\nLLM_API_KEY=old-key\n"
    layout.llm_config_file.parent.mkdir(parents=True)
    layout.llm_config_file.write_bytes(old_configuration)
    monkeypatch.setattr(cli, "_collect_configuration", lambda _arguments: environment)

    def fake_setup(_layout):
        runtime_python.parent.mkdir(parents=True)
        runtime_python.write_text("managed", encoding="utf-8")
        layout.profile_file.parent.mkdir(parents=True, exist_ok=True)
        layout.profile_file.write_text("committed", encoding="utf-8")
        model = layout.model_root / "preserved-model.bin"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_text("preserved", encoding="utf-8")
        return SimpleNamespace(interpreter={"path": str(runtime_python)})

    monkeypatch.setattr(cli, "setup", fake_setup)
    monkeypatch.setattr(
        cli,
        "test_llm_configuration",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("大模型连接检查失败（CONNECT）：固定诊断。")
        ),
    )
    monkeypatch.setattr(
        cli,
        "write_private_environment",
        lambda *_args: pytest.fail("failed candidate must not be saved"),
    )
    arguments = build_parser().parse_args(
        [
            "onboard",
            "--api-base",
            "https://candidate.example/v1",
            "--model",
            "candidate-model",
            "--api-key-stdin",
        ]
    )

    with pytest.raises(RuntimeError, match="CONNECT") as captured:
        cli._onboard(layout, arguments)

    assert runtime_python.read_text(encoding="utf-8") == "managed"
    assert layout.profile_file.read_text(encoding="utf-8") == "committed"
    assert (layout.model_root / "preserved-model.bin").is_file()
    assert layout.llm_config_file.read_bytes() == old_configuration
    message = str(captured.value)
    assert message.splitlines()[0].startswith("大模型连接检查失败（CONNECT）")
    assert "本地 CPU 环境和模型已完成并保留。" in message
    assert "候选 LLM 配置未保存。" in message
    assert "python scripts/socialgraph.py configure-llm" in message
    assert "candidate-key" not in message
    assert "candidate.example" not in message


def test_configure_llm_never_runs_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment = {
        "LLM_API_BASE": "https://provider.example/v1",
        "LLM_MODEL": "model-id",
        "LLM_API_KEY": "test-key",
    }
    saved: list[dict[str, str]] = []
    monkeypatch.setattr(
        RuntimeLayout,
        "discover",
        lambda _explicit=None: RuntimeLayout(tmp_path),
    )
    monkeypatch.setattr(cli, "_collect_configuration", lambda _arguments: environment)
    monkeypatch.setattr(
        cli,
        "_save_verified_configuration",
        lambda _layout, selected: saved.append(selected),
    )
    monkeypatch.setattr(
        cli,
        "setup",
        lambda *_args, **_kwargs: pytest.fail("configure-llm must not run setup"),
    )
    arguments = build_parser().parse_args(
        [
            "--project-root",
            str(tmp_path),
            "configure-llm",
            "--api-base",
            "https://provider.example/v1",
            "--model",
            "model-id",
            "--api-key-stdin",
        ]
    )

    assert cli.run(arguments) == 0
    assert saved == [environment]


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
