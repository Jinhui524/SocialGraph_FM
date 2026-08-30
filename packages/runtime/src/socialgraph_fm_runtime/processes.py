"""Cross-platform detached service lifecycle with fail-closed PID identity checks."""

from __future__ import annotations

import ctypes
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .profile import atomic_write_json
from .subprocess_control import terminate_process_tree, terminate_windows_process_tree


PID_SCHEMA_VERSION = "socialgraph-fm.managed-process/2.0"


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        return None


def http_health_ready(
    port: int,
    path: str,
    *,
    expected_json: dict[str, Any] | None = None,
    expected_text: str | None = None,
    token_file: str | Path | None = None,
    timeout: float = 0.5,
) -> bool:
    headers: dict[str, str] = {}
    if token_file is not None:
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError:
            return False
        if not token:
            return False
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", headers=headers, method="GET"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                return False
            raw = response.read(1024 * 1024 + 1)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    if len(raw) > 1024 * 1024:
        return False
    if expected_text is not None:
        try:
            return expected_text in raw.decode("utf-8")
        except UnicodeDecodeError:
            return False
    if expected_json is not None:
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        return isinstance(document, dict) and all(
            document.get(name) == value for name, value in expected_json.items()
        )
    return True


def _windows_identity(pid: int) -> dict[str, Any] | None:
    if os.name != "nt":
        return None
    from ctypes import wintypes

    class FileTime(ctypes.Structure):
        _fields_ = (("low", wintypes.DWORD), ("high", wintypes.DWORD))

        @property
        def value(self) -> int:
            return (int(self.high) << 32) | int(self.low)

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        creation = FileTime()
        exit_time = FileTime()
        kernel = FileTime()
        user = FileTime()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None
        return {
            "kind": "windows-filetime",
            "creation": str(creation.value),
            "executable": str(Path(buffer.value).resolve()),
        }
    finally:
        kernel32.CloseHandle(handle)


def _linux_identity(pid: int) -> dict[str, Any] | None:
    process = Path("/proc") / str(pid)
    try:
        stat = (process / "stat").read_text(encoding="utf-8")
        closing = stat.rfind(")")
        fields = stat[closing + 2 :].split()
        start_ticks = fields[19]
        executable = str((process / "exe").resolve(strict=True))
        command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except (OSError, IndexError):
        return None
    return {
        "kind": "linux-procfs",
        "creation": start_ticks,
        "executable": executable,
        "command": command,
    }


def _posix_ps_identity(pid: int) -> dict[str, Any] | None:
    completed = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        return None
    return {"kind": "posix-ps", "creation": value[:24], "command": value[24:].strip()}


def process_identity(pid: int) -> dict[str, Any] | None:
    if pid <= 0:
        return None
    if os.name == "nt":
        return _windows_identity(pid)
    if Path("/proc").is_dir():
        return _linux_identity(pid)
    return _posix_ps_identity(pid)


def _same_path(left: str, right: str) -> bool:
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(Path(right).resolve()))


def launch_executable(path: str | Path) -> Path:
    """Return an absolute launcher path without dereferencing a venv symlink."""

    return Path(os.path.abspath(Path(path).expanduser()))


def identity_matches(record: dict[str, Any], observed: dict[str, Any] | None) -> bool:
    if observed is None:
        return False
    expected = record.get("processIdentity")
    if not isinstance(expected, dict):
        return False
    if expected.get("kind") != observed.get("kind") or expected.get("creation") != observed.get(
        "creation"
    ):
        return False
    expected_executable = expected.get("executable")
    actual_executable = observed.get("executable")
    if expected_executable and actual_executable and not _same_path(expected_executable, actual_executable):
        return False
    command = str(observed.get("command", ""))
    if command:
        return all(str(token) in command for token in record.get("identityTokens", []))
    return True


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    port: int
    executable: Path
    arguments: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]
    identity_tokens: tuple[str, ...]
    health_path: str | None = None
    health_json: dict[str, Any] | None = None
    health_text: str | None = None
    health_token_file: Path | None = None
    generation: str | None = None


def _service_ready(service: ServiceSpec) -> bool:
    if service.health_path is None:
        return port_open(service.port)
    return http_health_ready(
        service.port,
        service.health_path,
        expected_json=service.health_json,
        expected_text=service.health_text,
        token_file=service.health_token_file,
    )


def _record_ready(record: dict[str, Any]) -> bool:
    path = record.get("healthPath")
    if not isinstance(path, str) or not path:
        return port_open(int(record["port"]))
    expected_json = record.get("healthJson")
    return http_health_ready(
        int(record["port"]),
        path,
        expected_json=expected_json if isinstance(expected_json, dict) else None,
        expected_text=(
            str(record["healthText"]) if record.get("healthText") is not None else None
        ),
        token_file=(
            str(record["healthTokenFile"])
            if record.get("healthTokenFile") is not None
            else None
        ),
    )


def _record_matches_service(record: dict[str, Any], service: ServiceSpec) -> bool:
    if int(record.get("port", -1)) != service.port:
        return False
    if not _same_path(
        str(record.get("executablePath", "")), str(launch_executable(service.executable))
    ):
        return False
    if record.get("arguments") != list(service.arguments):
        return False
    if record.get("workingDirectory") != str(Path(os.path.abspath(service.cwd))):
        return False
    if record.get("identityTokens") != list(service.identity_tokens):
        return False
    if record.get("healthPath") != service.health_path:
        return False
    if record.get("healthJson") != service.health_json:
        return False
    if record.get("healthText") != service.health_text:
        return False
    expected_token = str(service.health_token_file) if service.health_token_file else None
    if record.get("healthTokenFile") != expected_token:
        return False
    return record.get("generation") == service.generation


class ProcessManager:
    def __init__(self, pid_root: Path, log_root: Path) -> None:
        self.pid_root = pid_root
        self.log_root = log_root
        self.pid_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)

    def record_path(self, name: str) -> Path:
        return self.pid_root / f"{name}.json"

    def read_record(self, name: str) -> dict[str, Any] | None:
        path = self.record_path(name)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid PID record; inspect it before retrying: {path}") from error
        if record.get("schemaVersion") != PID_SCHEMA_VERSION or record.get("service") != name:
            raise RuntimeError(f"Unsupported PID record; inspect it before retrying: {path}")
        return record

    def snapshot(self, name: str) -> dict[str, Any] | None:
        record = self.read_record(name)
        if record is None:
            return None
        observed = process_identity(int(record["pid"]))
        return {
            "record": record,
            "alive": observed is not None,
            "identityMatches": identity_matches(record, observed),
            "portOpen": port_open(int(record["port"])),
            "healthReady": _record_ready(record),
        }

    def _archive_stale(self, path: Path) -> None:
        stale = self.pid_root / "stale"
        stale.mkdir(parents=True, exist_ok=True)
        destination = stale / f"{path.stem}.{time.time_ns()}.json"
        os.replace(path, destination)

    def start(self, service: ServiceSpec, *, timeout: float = 60.0) -> bool:
        path = self.record_path(service.name)
        existing = self.read_record(service.name)
        if existing is not None:
            observed = process_identity(int(existing["pid"]))
            if identity_matches(existing, observed) and _record_ready(existing):
                if _record_matches_service(existing, service):
                    return False
                raise RuntimeError(
                    f"{service.name} is running with a different configuration; stop it first"
                )
            if observed is not None or port_open(service.port):
                raise RuntimeError(
                    f"Refusing to replace mismatched PID record or port owner for {service.name}: {path}"
                )
            self._archive_stale(path)
        if port_open(service.port):
            raise RuntimeError(f"Port {service.port} for {service.name} is already in use")
        # Preserve a venv's launcher path on POSIX. Resolving its symlink would
        # execute the base/system interpreter and bypass the selected environment.
        executable = launch_executable(service.executable)
        if not executable.is_file():
            raise RuntimeError(f"Executable for {service.name} is missing: {executable}")
        stdout_path = self.log_root / f"{service.name}.out.log"
        stderr_path = self.log_root / f"{service.name}.err.log"
        creationflags = 0
        popen_options: dict[str, Any] = {}
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            popen_options["start_new_session"] = True
        with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
            "ab", buffering=0
        ) as stderr:
            process = subprocess.Popen(
                [str(executable), *service.arguments],
                cwd=service.cwd,
                env=service.environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
                **popen_options,
            )
        deadline = time.monotonic() + min(timeout, 5.0)
        observed = None
        while time.monotonic() < deadline:
            observed = process_identity(process.pid)
            if observed is not None:
                break
            if process.poll() is not None:
                break
            time.sleep(0.05)
        if observed is None:
            terminate_process_tree(process)
            raise RuntimeError(f"Could not bind process identity for {service.name}")
        record = {
            "schemaVersion": PID_SCHEMA_VERSION,
            "service": service.name,
            "pid": process.pid,
            "port": service.port,
            "executablePath": str(executable),
            "arguments": list(service.arguments),
            "workingDirectory": str(Path(os.path.abspath(service.cwd))),
            "identityTokens": list(service.identity_tokens),
            "healthPath": service.health_path,
            "healthJson": service.health_json,
            "healthText": service.health_text,
            "healthTokenFile": (
                str(service.health_token_file) if service.health_token_file else None
            ),
            "generation": service.generation,
            "processIdentity": observed,
            "startedAtUnixNs": time.time_ns(),
        }
        try:
            atomic_write_json(path, record)
        except Exception:
            terminate_process_tree(process)
            raise
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                if not port_open(service.port):
                    path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{service.name} exited during startup; inspect {stderr_path}"
                )
            if _service_ready(service):
                current = process_identity(process.pid)
                if process.poll() is None and identity_matches(record, current):
                    return True
                if not port_open(service.port):
                    path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{service.name} exited during startup; inspect {stderr_path}"
                )
            time.sleep(0.25)
        self.stop(service.name)
        raise RuntimeError(f"{service.name} did not listen on port {service.port} within {timeout}s")

    def stop(self, name: str, *, timeout: float = 10.0) -> bool:
        path = self.record_path(name)
        record = self.read_record(name)
        if record is None:
            return False
        pid = int(record["pid"])
        observed = process_identity(pid)
        if observed is None:
            if port_open(int(record["port"])):
                raise RuntimeError(
                    f"Refusing to discard {name} PID record: its root exited but "
                    "the managed port remains active"
                )
            path.unlink(missing_ok=True)
            return False
        if not identity_matches(record, observed):
            if port_open(int(record["port"])):
                raise RuntimeError(f"Refusing to stop mismatched process for {name}")
            self._archive_stale(path)
            return False
        try:
            if os.name == "nt":
                terminate_windows_process_tree(pid)
            else:
                os.killpg(pid, signal.SIGTERM)  # type: ignore[attr-defined]
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process_identity(pid) is None and not port_open(int(record["port"])):
                path.unlink(missing_ok=True)
                return True
            time.sleep(0.1)
        try:
            if os.name == "nt":
                current = process_identity(pid)
                if identity_matches(record, current):
                    terminate_windows_process_tree(pid)
            else:
                os.killpg(pid, signal.SIGKILL)  # type: ignore[attr-defined]
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if process_identity(pid) is None and not port_open(int(record["port"])):
                path.unlink(missing_ok=True)
                return True
            time.sleep(0.1)
        raise RuntimeError(f"{name} or its port remained active; PID record retained")
