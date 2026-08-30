from __future__ import annotations

import os
import signal
import socket
import sys
import time
from pathlib import Path

import pytest

from socialgraph_fm_runtime.processes import (
    ProcessManager,
    ServiceSpec,
    launch_executable,
    port_open,
)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_process_manager_starts_identifies_and_stops_service(tmp_path: Path) -> None:
    port = _free_port()
    manager = ProcessManager(tmp_path / "pids", tmp_path / "logs")
    spec = ServiceSpec(
        name="fixture",
        port=port,
        executable=Path(sys.executable),
        arguments=("-m", "http.server", str(port), "--bind", "127.0.0.1"),
        cwd=tmp_path,
        environment=dict(__import__("os").environ),
        identity_tokens=("http.server", str(port)),
    )
    try:
        assert manager.start(spec, timeout=15) is True
        snapshot = manager.snapshot("fixture")
        assert snapshot is not None
        assert snapshot["alive"] is True
        assert snapshot["identityMatches"] is True
        assert snapshot["portOpen"] is True
        assert manager.start(spec, timeout=5) is False
    finally:
        manager.stop("fixture", timeout=5)
    assert not port_open(port)
    assert manager.read_record("fixture") is None


def test_running_service_with_changed_spec_requires_an_explicit_restart(
    tmp_path: Path,
) -> None:
    port = _free_port()
    manager = ProcessManager(tmp_path / "pids", tmp_path / "logs")
    base = ServiceSpec(
        name="fixture",
        port=port,
        executable=Path(sys.executable),
        arguments=("-m", "http.server", str(port), "--bind", "127.0.0.1"),
        cwd=tmp_path,
        environment=dict(__import__("os").environ),
        identity_tokens=("http.server", str(port)),
        generation="first",
    )
    changed = ServiceSpec(**{**base.__dict__, "generation": "second"})
    try:
        assert manager.start(base, timeout=15) is True
        with pytest.raises(RuntimeError, match="different configuration"):
            manager.start(changed, timeout=5)
    finally:
        manager.stop("fixture", timeout=5)


def test_launch_executable_does_not_dereference_posix_venv_symlink(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        return
    base = tmp_path / "base-python"
    base.write_text("fixture", encoding="utf-8")
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(base)

    assert launch_executable(launcher) == launcher.absolute()


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def _stop_fixture_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


def _child_server_spec(
    tmp_path: Path, port: int, *, parent_lifetime: float
) -> tuple[ServiceSpec, Path]:
    child_pid = tmp_path / "child.pid"
    source = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-m','http.server',sys.argv[1],"
        "'--bind','127.0.0.1']); "
        "pathlib.Path(sys.argv[2]).write_text(str(child.pid)); "
        "time.sleep(float(sys.argv[3]))"
    )
    return (
        ServiceSpec(
            name="tree-fixture",
            port=port,
            executable=Path(sys.executable),
            arguments=("-c", source, str(port), str(child_pid), str(parent_lifetime)),
            cwd=tmp_path,
            environment=dict(os.environ),
            identity_tokens=(str(port), str(child_pid)),
        ),
        child_pid,
    )


def test_process_manager_stops_descendant_tree(tmp_path: Path) -> None:
    port = _free_port()
    manager = ProcessManager(tmp_path / "pids", tmp_path / "logs")
    spec, child_pid_path = _child_server_spec(tmp_path, port, parent_lifetime=30)
    child_pid: int | None = None
    try:
        assert manager.start(spec, timeout=15) is True
        assert child_pid_path.is_file()
        child_pid = int(child_pid_path.read_text())
        assert manager.stop(spec.name, timeout=5) is True
        assert _wait_until(lambda: not port_open(port))
    finally:
        if child_pid is not None:
            _stop_fixture_pid(child_pid)
        try:
            manager.stop(spec.name, timeout=1)
        except RuntimeError:
            pass


def test_dead_root_with_live_port_keeps_pid_record_fail_closed(tmp_path: Path) -> None:
    port = _free_port()
    manager = ProcessManager(tmp_path / "pids", tmp_path / "logs")
    # Keep the root alive long enough for the manager to observe readiness even
    # on a saturated CI host; the assertion below still verifies the later
    # root-exited/child-listener fail-closed state.
    spec, child_pid_path = _child_server_spec(tmp_path, port, parent_lifetime=2.0)
    child_pid: int | None = None
    try:
        assert manager.start(spec, timeout=15) is True
        child_pid = int(child_pid_path.read_text())
        assert _wait_until(
            lambda: (
                (snapshot := manager.snapshot(spec.name)) is not None
                and snapshot["alive"] is False
                and snapshot["portOpen"] is True
            )
        )
        with pytest.raises(RuntimeError, match="root exited.*port remains active"):
            manager.stop(spec.name, timeout=0.5)
        assert manager.read_record(spec.name) is not None
        assert port_open(port)
    finally:
        if child_pid is not None:
            _stop_fixture_pid(child_pid)
        _wait_until(lambda: not port_open(port))
