"""Fail-closed subprocess execution shared by setup and lifecycle commands."""

from __future__ import annotations

import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?:authorization\s*[:=]\s*bearer\s+|(?:api[_-]?key|token|secret)\s*[:=]\s*)\S+"
)
_OPENAI_STYLE_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_URL_CREDENTIALS = re.compile(r"(?i)(https?://)[^/@\s]+@")


def redact_subprocess_text(message: str) -> str:
    """Remove supported credential forms before output is logged or raised."""

    selected = _SECRET_ASSIGNMENT.sub("[REDACTED]", message)
    selected = _OPENAI_STYLE_KEY.sub("[REDACTED]", selected)
    return _URL_CREDENTIALS.sub(r"\1[REDACTED]@", selected)


def _group_popen_options() -> tuple[int, dict[str, Any]]:
    if os.name == "nt":
        return (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
            {},
        )
    return 0, {"start_new_session": True}


def _windows_taskkill(pid: int) -> bool:
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    bundled = Path(system_root) / "System32" / "taskkill.exe"
    executable = str(bundled) if bundled.is_file() else shutil.which("taskkill")
    if not executable:
        return False
    try:
        completed = subprocess.run(
            [executable, "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def terminate_process_tree(
    process: subprocess.Popen[Any], *, grace_seconds: float = 1.0
) -> None:
    """Best-effort termination of a child and every child process it still owns."""

    if os.name == "nt":
        _windows_taskkill(process.pid)
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=max(grace_seconds, 0.1))
        except (OSError, subprocess.TimeoutExpired):
            pass
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)  # type: ignore[attr-defined]
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + max(grace_seconds, 0.0)
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)  # type: ignore[attr-defined]
        except ProcessLookupError:
            break
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)  # type: ignore[attr-defined]
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=max(grace_seconds, 0.1))
    except (OSError, subprocess.TimeoutExpired):
        pass


def terminate_windows_process_tree(pid: int) -> bool:
    """Terminate a verified Windows root and its descendants."""

    if os.name != "nt":
        raise RuntimeError("Windows process-tree termination is unavailable on this platform")
    return _windows_taskkill(pid)


def run_captured_process(
    command: Sequence[str],
    *,
    cwd: Path | None,
    environment: Mapping[str, str],
    timeout: float,
    input_text: str | None = None,
    description: str = "Command",
) -> subprocess.CompletedProcess[str]:
    """Run a captured command with a process-tree deadline and safe timeout error."""

    creationflags, platform_options = _group_popen_options()
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        creationflags=creationflags,
        **platform_options,
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process_tree(process)
        try:
            process.communicate(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass
        raise RuntimeError(f"{description} timed out after {timeout:g}s") from None
    except BaseException:
        terminate_process_tree(process)
        raise
    return subprocess.CompletedProcess(
        list(command), process.returncode, stdout=stdout, stderr=stderr
    )


def run_streaming_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout: float,
    logger: Callable[[str], None] | None = None,
    description: str = "Command",
) -> None:
    """Stream redacted output while enforcing a cross-platform process-tree timeout."""

    if timeout <= 0:
        raise ValueError("Subprocess timeout must be positive")
    creationflags, platform_options = _group_popen_options()
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
        **platform_options,
    )
    stream = process.stdout
    assert stream is not None
    output: queue.Queue[str] = queue.Queue()
    reader_finished = threading.Event()

    def read_output() -> None:
        try:
            try:
                for raw_line in stream:
                    output.put(raw_line)
            except (OSError, ValueError):
                pass
        finally:
            reader_finished.set()

    reader = threading.Thread(target=read_output, name="socialgraph-subprocess-output", daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    tail: list[str] = []
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminate_process_tree(process)
                raise RuntimeError(f"{description} timed out after {timeout:g}s")
            try:
                raw_line = output.get(timeout=min(0.1, remaining))
            except queue.Empty:
                raw_line = ""
            if raw_line:
                selected = redact_subprocess_text(raw_line.rstrip())
                if selected:
                    if logger is not None:
                        logger(selected)
                    tail.append(selected)
                    tail = tail[-50:]
            if process.poll() is not None and reader_finished.is_set() and output.empty():
                break
    except BaseException:
        terminate_process_tree(process)
        raise
    finally:
        reader.join(timeout=1)
        try:
            stream.close()
        except OSError:
            pass
    if process.returncode != 0:
        detail = tail[-1] if tail else "no diagnostic output"
        raise RuntimeError(
            f"{description} failed with exit code {process.returncode}: {detail}"
        )
