from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from socialgraph_fm_runtime.processes import process_identity
from socialgraph_fm_runtime.subprocess_control import (
    run_captured_process,
    run_streaming_process,
)


def _wait_for_process_exit(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process_identity(pid) is None:
            return True
        time.sleep(0.05)
    return process_identity(pid) is None


def test_streaming_timeout_kills_tree_and_never_rethrows_raw_secret(
    tmp_path: Path,
) -> None:
    pids = tmp_path / "pids.txt"
    source = (
        "import os,pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(f'{os.getpid()} {child.pid}'); "
        "print('Author'+'ization: Bearer '+'SENTINEL_STREAM_SECRET',flush=True); "
        "time.sleep(30)"
    )
    logs: list[str] = []

    with pytest.raises(RuntimeError, match="timed out") as captured:
        run_streaming_process(
            [sys.executable, "-c", source, str(pids)],
            cwd=tmp_path,
            environment=dict(os.environ),
            timeout=0.75,
            logger=logs.append,
            description="fixture command",
        )

    assert pids.is_file()
    parent_pid, child_pid = (int(value) for value in pids.read_text().split())
    assert _wait_for_process_exit(parent_pid)
    assert _wait_for_process_exit(child_pid)
    assert any("[REDACTED]" in line for line in logs)
    assert all("SENTINEL_STREAM_SECRET" not in line for line in logs)
    assert "SENTINEL_STREAM_SECRET" not in str(captured.value)


def test_failed_streaming_command_only_raises_redacted_tail(tmp_path: Path) -> None:
    logs: list[str] = []
    with pytest.raises(RuntimeError) as captured:
        run_streaming_process(
            [
                sys.executable,
                "-c",
                "import sys; print('api'+'_key='+'SENTINEL_FAILURE_SECRET'); sys.exit(7)",
            ],
            cwd=tmp_path,
            environment=dict(os.environ),
            timeout=10,
            logger=logs.append,
            description="fixture command",
        )

    assert logs == ["[REDACTED]"]
    assert "[REDACTED]" in str(captured.value)
    assert "SENTINEL_FAILURE_SECRET" not in str(captured.value)


def test_captured_timeout_is_a_safe_runtime_error_and_kills_child(tmp_path: Path) -> None:
    pid_path = tmp_path / "captured.pid"
    source = (
        "import os,pathlib,sys,time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )

    with pytest.raises(RuntimeError, match="LLM connection check timed out"):
        run_captured_process(
            [sys.executable, "-c", source, str(pid_path)],
            cwd=tmp_path,
            environment=dict(os.environ),
            timeout=0.5,
            description="LLM connection check",
        )

    assert pid_path.is_file()
    assert _wait_for_process_exit(int(pid_path.read_text()))
