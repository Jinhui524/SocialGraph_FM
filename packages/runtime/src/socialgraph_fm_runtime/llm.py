"""Three-field OpenAI-compatible LLM configuration with private storage."""

from __future__ import annotations

import getpass
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, TextIO

from .environment import is_ambient_llm_environment_name


MAX_API_KEY_CHARACTERS = 8_192
LLM_NAMES = ("LLM_API_BASE", "LLM_MODEL", "LLM_API_KEY")
_LEGACY_NAMES = {
    "LLM_API_MODE",
    "LLM_AUTH_SCHEME",
    "LLM_ANTHROPIC_VERSION",
    "LLM_TIMEOUT_SECONDS",
    "LLM_ALLOW_INSECURE_LOOPBACK",
    "LLM_VERIFICATION_STATUS",
    "LOG_LEVEL",
}
_PERCENT_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")


def _single_line(name: str, value: str, *, allow_empty: bool = False) -> str:
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} cannot be empty")
    if re.search(r"[\x00-\x1f\x7f]", value):
        raise ValueError(f"{name} must be single-line text without control characters")
    if value != value.strip():
        raise ValueError(f"{name} cannot have leading or trailing whitespace")
    return value


def _contains_encoded_control(value: str) -> bool:
    return any(
        int(match.group(1), 16) <= 0x1F or int(match.group(1), 16) == 0x7F
        for match in _PERCENT_ESCAPE.finditer(value)
    )


def _normalized_host(value: str) -> tuple[str, bool]:
    if not value or any(character.isspace() for character in value) or "%" in value:
        raise ValueError("API Base hostname is invalid")
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        try:
            host = value.encode("idna").decode("ascii").lower()
        except UnicodeError as error:
            raise ValueError("API Base hostname is invalid") from error
        labels = host[:-1].split(".") if host.endswith(".") else host.split(".")
        if (
            not labels
            or len(host) > 253
            or any(
                not label
                or len(label) > 63
                or not re.fullmatch(
                    r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label
                )
                for label in labels
            )
        ):
            raise ValueError("API Base hostname is invalid") from None
        return host, host.rstrip(".") == "localhost"
    return address.compressed.lower(), address.is_loopback


def normalize_api_base(value: str) -> str:
    """Normalize a root, /v1 root, or full Chat Completions endpoint."""

    selected = _single_line("API Base", value)
    if any(character.isspace() for character in selected):
        raise ValueError("API Base cannot contain whitespace")
    if "\\" in selected or _contains_encoded_control(selected):
        raise ValueError("API Base cannot contain backslashes or encoded control characters")
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://[A-Za-z][A-Za-z0-9+.-]*://", selected):
        raise ValueError("API Base contains a repeated protocol prefix")
    try:
        parsed = urllib.parse.urlsplit(selected)
        parsed_port = parsed.port
    except ValueError as error:
        raise ValueError("API Base contains an invalid host or port") from error
    if parsed.scheme.lower() not in {"https", "http"} or not parsed.netloc:
        raise ValueError("API Base must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("API Base cannot contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("API Base cannot contain a query string or fragment")
    host, loopback = _normalized_host(parsed.hostname or "")
    if parsed.scheme.lower() != "https" and not loopback:
        raise ValueError("Remote API Base URLs must use HTTPS")
    port = f":{parsed_port}" if parsed_port is not None else ""
    hostname = f"[{host}]" if ":" in host else host
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    if not path:
        path = "/v1"
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), hostname + port, path, "", "")
    )


def normalize_relay_api_base(
    value: str, *, allow_insecure_loopback: bool = False
) -> str:
    """Compatibility alias; loopback HTTP is detected automatically."""

    del allow_insecure_loopback
    return normalize_api_base(value)


def derive_api_endpoint(
    api_base: str,
    api_mode: str = "chat_completions",
    *,
    allow_insecure_loopback: bool = False,
) -> str:
    del allow_insecure_loopback
    if api_mode != "chat_completions":
        raise ValueError("Only OpenAI-compatible Chat Completions is supported")
    return f"{normalize_api_base(api_base)}/chat/completions"


def configuration_state(environment: dict[str, str]) -> str:
    present = [name for name in LLM_NAMES if environment.get(name)]
    if not present:
        return "missing"
    return "complete" if len(present) == len(LLM_NAMES) else "partial"


def _acl_process_environment(path: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if is_ambient_llm_environment_name(name):
            environment.pop(name, None)
    environment["SOCIALGRAPH_ACL_TARGET"] = str(path)
    return environment


def _windows_protect(path: Path, *, directory: bool) -> None:
    icacls = shutil.which("icacls")
    whoami = shutil.which("whoami")
    if not icacls or not whoami:
        raise RuntimeError("Windows ACL tools are unavailable")
    acl_environment = _acl_process_environment(path)
    completed = subprocess.run(
        [whoami, "/user", "/fo", "csv", "/nh"],
        check=False,
        capture_output=True,
        text=True,
        env=acl_environment,
    )
    match = re.search(r'"(S-1-[0-9-]+)"', completed.stdout)
    if completed.returncode != 0 or match is None:
        raise RuntimeError("Could not resolve the current Windows user SID")
    sid = match.group(1)
    inheritance = "(OI)(CI)F" if directory else "F"
    reset = subprocess.run(
        [icacls, str(path), "/reset"],
        check=False,
        capture_output=True,
        text=True,
        env=acl_environment,
    )
    if reset.returncode != 0:
        raise RuntimeError(f"Could not reset private configuration ACL: {path}")
    result = subprocess.run(
        [
            icacls,
            str(path),
            "/inheritance:r",
            "/grant:r",
            f"*{sid}:{inheritance}",
            f"*S-1-5-18:{inheritance}",
            f"*S-1-5-32-544:{inheritance}",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=acl_environment,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not protect private configuration ACL: {path}")


def _current_windows_sid(path: Path) -> str:
    whoami = shutil.which("whoami")
    if not whoami:
        raise RuntimeError("Windows identity tool is unavailable")
    completed = subprocess.run(
        [whoami, "/user", "/fo", "csv", "/nh"],
        check=False,
        capture_output=True,
        text=True,
        env=_acl_process_environment(path),
    )
    match = re.search(r'"(S-1-[0-9-]+)"', completed.stdout)
    if completed.returncode != 0 or match is None:
        raise RuntimeError("Could not resolve the current Windows user SID")
    return match.group(1)


def _windows_acl_document(path: Path) -> dict[str, Any]:
    system_root = os.environ.get("SystemRoot", "C:\\Windows")
    powershell = (
        Path(system_root)
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    if not powershell.is_file():
        located = shutil.which("powershell") or shutil.which("pwsh")
        if not located:
            raise RuntimeError("Windows ACL inspection shell is unavailable")
        powershell = Path(located)
    script = r"""
$ErrorActionPreference = 'Stop'
$securityManifest = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1'
Import-Module $securityManifest -Force -ErrorAction Stop
$target = [Environment]::GetEnvironmentVariable('SOCIALGRAPH_ACL_TARGET', 'Process')
$acl = Get-Acl -LiteralPath $target
$rules = @($acl.Access | ForEach-Object {
  $sid = $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
  [ordered]@{ sid = $sid; type = $_.AccessControlType.ToString() }
})
[ordered]@{ protected = [bool]$acl.AreAccessRulesProtected; rules = $rules } |
  ConvertTo-Json -Compress -Depth 4
"""
    completed = subprocess.run(
        [
            str(powershell),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_acl_process_environment(path),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise RuntimeError(f"Private configuration ACL could not be read: {path} ({detail})")
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Private configuration ACL returned invalid data: {path}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"Private configuration ACL returned invalid data: {path}")
    return document


def _validate_windows_acl(
    document: dict[str, Any], current_sid: str, path: Path
) -> None:
    if document.get("protected") is not True:
        raise RuntimeError(f"Private configuration ACL inheritance is enabled: {path}")
    rules = document.get("rules")
    if not isinstance(rules, list):
        raise RuntimeError(f"Private configuration ACL rules are invalid: {path}")
    allowed = {current_sid, "S-1-5-18", "S-1-5-32-544"}
    current_user_allowed = False
    for rule in rules:
        if not isinstance(rule, dict):
            raise RuntimeError(f"Private configuration ACL rule is invalid: {path}")
        if str(rule.get("type", "")).lower() != "allow":
            continue
        sid = str(rule.get("sid", ""))
        if sid not in allowed:
            raise RuntimeError(
                f"Private configuration grants access to an unexpected principal: {sid}"
            )
        current_user_allowed = current_user_allowed or sid == current_sid
    if not current_user_allowed:
        raise RuntimeError("Private configuration does not grant access to the current user")


def _windows_assert_protected(path: Path) -> None:
    icacls = shutil.which("icacls")
    if not icacls:
        raise RuntimeError("Windows ACL verification tool is unavailable")
    verified = subprocess.run(
        [icacls, str(path), "/verify"],
        check=False,
        capture_output=True,
        text=True,
    )
    if verified.returncode != 0:
        raise RuntimeError(f"Private configuration ACL could not be verified: {path}")
    _validate_windows_acl(_windows_acl_document(path), _current_windows_sid(path), path)


def assert_private_permissions(path: Path) -> None:
    targets = (path.parent, path)
    if os.name == "nt":
        for target in targets:
            _windows_assert_protected(target)
        return
    for target, mode in zip(targets, (0o700, 0o600), strict=True):
        actual = target.stat().st_mode & 0o777
        if actual != mode:
            raise RuntimeError(
                f"Private configuration permissions must be {mode:o}, found {actual:o}: {target}"
            )


def protect_private_path(path: Path, *, directory: bool) -> None:
    if path.is_symlink():
        raise RuntimeError(f"Private configuration path cannot be a link: {path}")
    if os.name == "nt":
        _windows_protect(path, directory=directory)
    else:
        os.chmod(path, 0o700 if directory else 0o600)


def _validated_values(environment: dict[str, str]) -> dict[str, str]:
    selected = {
        "LLM_API_BASE": normalize_api_base(environment.get("LLM_API_BASE", "")),
        "LLM_MODEL": _single_line("Model ID", environment.get("LLM_MODEL", "")),
        "LLM_API_KEY": _single_line("API Key", environment.get("LLM_API_KEY", "")),
    }
    if len(selected["LLM_API_KEY"]) > MAX_API_KEY_CHARACTERS:
        raise ValueError(f"API Key cannot exceed {MAX_API_KEY_CHARACTERS} characters")
    return selected


def parse_private_environment(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Private configuration must be a regular file: {path}")
    assert_private_permissions(path)
    result: dict[str, str] = {}
    legacy: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"Invalid private configuration line: {path}")
        name, value = line.split("=", 1)
        name = name.strip().upper()
        if name not in {*LLM_NAMES, *_LEGACY_NAMES} or name in result or name in legacy:
            raise RuntimeError(f"Unsupported or duplicate private configuration name: {name}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        elif value.startswith(('"', "'")) or value.endswith(('"', "'")):
            raise RuntimeError(f"Unbalanced quotes for private configuration name: {name}")
        _single_line(name, value, allow_empty=True)
        (result if name in LLM_NAMES else legacy)[name] = value
    if legacy.get("LLM_API_MODE", "chat_completions") != "chat_completions":
        raise RuntimeError("The saved LLM protocol is retired; configure the three fields again")
    if legacy.get("LLM_AUTH_SCHEME", "bearer") != "bearer":
        raise RuntimeError("The saved LLM authentication is retired; configure again")
    if configuration_state(result) == "missing":
        return {}
    try:
        return _validated_values(result)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def write_private_environment(path: Path, environment: dict[str, str]) -> None:
    selected = _validated_values(environment)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise RuntimeError(f"Private configuration must be a regular file: {path}")
    if path.parent.is_symlink():
        raise RuntimeError(f"Private configuration directory cannot be a link: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=True)
    protect_private_path(path.parent, directory=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        if os.name != "nt":
            fchmod = getattr(os, "fchmod", None)
            if fchmod is None:  # pragma: no cover
                raise RuntimeError("POSIX fchmod is unavailable")
            fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(f"{name}={selected[name]}" for name in LLM_NAMES) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        protect_private_path(temporary, directory=False)
        if parse_private_environment(temporary) != selected:
            raise RuntimeError("Staged LLM configuration failed read-back validation")
        os.replace(temporary, path)
        protect_private_path(path, directory=False)
        if parse_private_environment(path) != selected:
            raise RuntimeError("Saved LLM configuration failed read-back validation")
    finally:
        temporary.unlink(missing_ok=True)


def migrate_private_environment_permissions(
    path: Path, *, validate_values: bool = True
) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file() or path.parent.is_symlink():
        raise RuntimeError(f"Private configuration must be a regular, non-linked file: {path}")
    protect_private_path(path.parent, directory=True)
    protect_private_path(path, directory=False)
    if validate_values:
        parse_private_environment(path)


def configuration_summary(environment: dict[str, str]) -> dict[str, Any]:
    selected = _validated_values(environment)
    return {
        "apiBase": selected["LLM_API_BASE"],
        "endpoint": derive_api_endpoint(selected["LLM_API_BASE"]),
        "model": selected["LLM_MODEL"],
        "apiKeyConfigured": True,
    }


def _prompt_line(
    stdin: TextIO,
    stdout: TextIO,
    prompt: str,
    *,
    default: str | None = None,
) -> str:
    suffix = f" [{default}]" if default else ""
    stdout.write(f"{prompt}{suffix}: ")
    stdout.flush()
    value = stdin.readline()
    if value == "":
        raise RuntimeError("LLM configuration input was cancelled")
    selected = value.rstrip("\r\n")
    return default if not selected and default is not None else selected


def configure_environment(
    *,
    api_base: str | None,
    model: str | None,
    api_key_stdin: bool,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
) -> dict[str, str]:
    interactive = stdin.isatty()
    if interactive:
        base = _prompt_line(stdin, stdout, "大模型 API 地址", default=api_base)
        selected_model = _prompt_line(stdin, stdout, "模型 ID", default=model)
        key = getpass.getpass("API Key: ")
    else:
        if not api_base or not model or not api_key_stdin:
            raise RuntimeError(
                "Non-interactive configuration requires --api-base, --model, and --api-key-stdin"
            )
        base = api_base
        selected_model = model
        # Windows PowerShell 5.1 may prefix native-pipeline UTF-8 with a BOM.
        key = stdin.readline().lstrip("\ufeff").rstrip("\r\n")
    selected = _validated_values(
        {
            "LLM_API_BASE": base,
            "LLM_MODEL": selected_model,
            "LLM_API_KEY": key,
        }
    )
    if interactive:
        summary = configuration_summary(selected)
        stdout.write(
            "配置摘要（密钥已隐藏）：\n"
            f"  API 地址：{summary['apiBase']}\n"
            f"  模型：{summary['model']}\n"
            "  API Key：已输入（隐藏）\n"
        )
        stdout.flush()
    return selected
