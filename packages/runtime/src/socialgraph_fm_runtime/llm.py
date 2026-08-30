"""Provider-neutral LLM configuration with platform-appropriate file protection."""

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


LLM_PRESET_SCHEMA = "socialgraph-fm.llm-presets/2.0"
API_MODES = ("chat_completions", "responses", "anthropic_messages")
AUTH_SCHEMES = ("bearer", "x-api-key")
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
MAX_API_KEY_CHARACTERS = 8_192
_ANTHROPIC_VERSION = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PERCENT_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
LLM_NAMES = (
    "LLM_API_BASE",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_API_MODE",
    "LLM_AUTH_SCHEME",
    "LLM_ANTHROPIC_VERSION",
    "LLM_TIMEOUT_SECONDS",
    "LLM_ALLOW_INSECURE_LOOPBACK",
    "LLM_VERIFICATION_STATUS",
)


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
                or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
                for label in labels
            )
        ):
            raise ValueError("API Base hostname is invalid") from None
        return host, host.rstrip(".") == "localhost"
    return address.compressed.lower(), address.is_loopback


def normalize_api_base(value: str, *, allow_insecure_loopback: bool = False) -> str:
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
    if parsed.scheme.lower() != "https" and not (allow_insecure_loopback and loopback):
        raise ValueError("Remote API Base URLs must use HTTPS")
    port = f":{parsed_port}" if parsed_port is not None else ""
    hostname = f"[{host}]" if ":" in host else host
    path = parsed.path.rstrip("/")
    return urllib.parse.urlunsplit((parsed.scheme.lower(), hostname + port, path, "", ""))


def derive_api_endpoint(
    api_base: str,
    api_mode: str,
    *,
    allow_insecure_loopback: bool = False,
) -> str:
    """Derive one supported model endpoint from a validated root or full endpoint."""

    if api_mode not in API_MODES:
        raise ValueError(f"Unsupported LLM API mode: {api_mode}")
    base = normalize_api_base(
        api_base, allow_insecure_loopback=allow_insecure_loopback
    )
    for suffix in ("/chat/completions", "/responses", "/messages"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    endpoint = {
        "chat_completions": "/chat/completions",
        "responses": "/responses",
        "anthropic_messages": "/messages",
    }[api_mode]
    return f"{base}{endpoint}"


def normalize_relay_api_base(
    value: str, *, allow_insecure_loopback: bool = False
) -> str:
    """Normalize a custom relay root, adding /v1 only when it has no path."""

    base = normalize_api_base(
        value, allow_insecure_loopback=allow_insecure_loopback
    )
    if not urllib.parse.urlsplit(base).path.rstrip("/"):
        return f"{base}/v1"
    return base


def read_presets(path: Path) -> dict[str, dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"LLM preset catalog is invalid: {path}") from error
    raw_presets = document.get("presets")
    if document.get("schemaVersion") != LLM_PRESET_SCHEMA or not isinstance(
        raw_presets, dict
    ):
        raise RuntimeError(f"LLM preset catalog schema is unsupported: {path}")
    expected_ids = {
        "openai_responses",
        "deepseek",
        "glm",
        "anthropic",
        "custom",
        "custom_anthropic",
    }
    if set(raw_presets) != expected_ids:
        raise RuntimeError(f"LLM preset catalog inventory is unsupported: {path}")
    presets: dict[str, dict[str, Any]] = {}
    expected_fields = {
        "displayName",
        "connectionKind",
        "apiBase",
        "defaultApiMode",
        "allowedApiModes",
        "defaultAuthScheme",
        "allowedAuthSchemes",
        "anthropicVersion",
    }
    for identifier, raw in raw_presets.items():
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise RuntimeError(f"LLM preset {identifier} has an unsupported shape")
        display_name = raw.get("displayName")
        modes = raw.get("allowedApiModes")
        default_mode = raw.get("defaultApiMode")
        connection_kind = raw.get("connectionKind")
        auth_schemes = raw.get("allowedAuthSchemes")
        default_auth_scheme = raw.get("defaultAuthScheme")
        anthropic_version = raw.get("anthropicVersion")
        if not isinstance(display_name, str):
            raise RuntimeError(f"LLM preset {identifier} display name is invalid")
        _single_line("LLM preset display name", display_name)
        if (
            not isinstance(modes, list)
            or not modes
            or any(mode not in API_MODES for mode in modes)
            or default_mode not in modes
        ):
            raise RuntimeError(f"LLM preset {identifier} API modes are invalid")
        if connection_kind not in {"direct", "custom_relay"}:
            raise RuntimeError(f"LLM preset {identifier} connection kind is invalid")
        if (
            not isinstance(auth_schemes, list)
            or not auth_schemes
            or any(scheme not in AUTH_SCHEMES for scheme in auth_schemes)
            or default_auth_scheme not in auth_schemes
        ):
            raise RuntimeError(f"LLM preset {identifier} auth schemes are invalid")
        uses_anthropic = "anthropic_messages" in modes
        if uses_anthropic:
            if (
                not isinstance(anthropic_version, str)
                or not _ANTHROPIC_VERSION.fullmatch(anthropic_version)
            ):
                raise RuntimeError(
                    f"LLM preset {identifier} Anthropic version is invalid"
                )
        elif anthropic_version is not None:
            raise RuntimeError(
                f"LLM preset {identifier} cannot define an Anthropic version"
            )
        api_base = raw.get("apiBase")
        if connection_kind == "custom_relay":
            if api_base is not None:
                raise RuntimeError("A custom LLM preset cannot fix an API Base")
        elif not isinstance(api_base, str) or normalize_api_base(api_base) != api_base:
            raise RuntimeError(f"LLM preset {identifier} API Base is invalid")
        presets[str(identifier)] = dict(raw)
    return presets


def _configuration_defaults(environment: dict[str, str]) -> dict[str, str]:
    selected = dict(environment)
    mode = selected.get("LLM_API_MODE", "").strip() or "chat_completions"
    selected["LLM_API_MODE"] = mode
    if not selected.get("LLM_AUTH_SCHEME", "").strip():
        selected["LLM_AUTH_SCHEME"] = (
            "x-api-key" if mode == "anthropic_messages" else "bearer"
        )
    if mode == "anthropic_messages" and not selected.get(
        "LLM_ANTHROPIC_VERSION", ""
    ).strip():
        selected["LLM_ANTHROPIC_VERSION"] = DEFAULT_ANTHROPIC_VERSION
    else:
        selected.setdefault("LLM_ANTHROPIC_VERSION", "")
    selected.setdefault("LLM_TIMEOUT_SECONDS", "15")
    selected.setdefault("LLM_ALLOW_INSECURE_LOOPBACK", "false")
    selected.setdefault("LLM_VERIFICATION_STATUS", "configured_unverified")
    return selected


def _validate_environment_protocol(environment: dict[str, str]) -> None:
    mode = environment.get("LLM_API_MODE", "")
    if mode not in API_MODES:
        raise RuntimeError(
            "LLM_API_MODE must be chat_completions, responses, or anthropic_messages"
        )
    auth_scheme = environment.get("LLM_AUTH_SCHEME", "")
    if auth_scheme not in AUTH_SCHEMES:
        raise RuntimeError("LLM_AUTH_SCHEME must be bearer or x-api-key")
    version = environment.get("LLM_ANTHROPIC_VERSION", "")
    if mode == "anthropic_messages":
        if not _ANTHROPIC_VERSION.fullmatch(version):
            raise RuntimeError("LLM_ANTHROPIC_VERSION must use YYYY-MM-DD")
    elif version:
        raise RuntimeError(
            "LLM_ANTHROPIC_VERSION is valid only with anthropic_messages"
        )


def parse_private_environment(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Private configuration must be a regular file: {path}")
    assert_private_permissions(path)
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"Invalid private configuration line: {path}")
        name, value = line.split("=", 1)
        name = name.strip().upper()
        if name not in {*LLM_NAMES, "LOG_LEVEL"} or name in result:
            raise RuntimeError(f"Unsupported or duplicate private configuration name: {name}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        elif value.startswith(('"', "'")) or value.endswith(('"', "'")):
            raise RuntimeError(f"Unbalanced quotes for private configuration name: {name}")
        _single_line(name, value, allow_empty=True)
        result[name] = value
    result = _configuration_defaults(result)
    _validate_environment_protocol(result)
    timeout = result.get("LLM_TIMEOUT_SECONDS")
    if timeout is not None:
        try:
            timeout_value = int(timeout)
        except ValueError as error:
            raise RuntimeError("LLM_TIMEOUT_SECONDS must be between 1 and 60") from error
        if not 1 <= timeout_value <= 60:
            raise RuntimeError("LLM_TIMEOUT_SECONDS must be between 1 and 60")
    allow_loopback = result.get("LLM_ALLOW_INSECURE_LOOPBACK")
    if allow_loopback is not None and allow_loopback not in {"true", "false"}:
        raise RuntimeError("LLM_ALLOW_INSECURE_LOOPBACK must be true or false")
    verification = result.get("LLM_VERIFICATION_STATUS")
    if verification is not None and verification not in {
        "configured_unverified",
        "call_succeeded",
        "fallback",
    }:
        raise RuntimeError("LLM_VERIFICATION_STATUS is invalid")
    key = result.get("LLM_API_KEY", "")
    if len(key) > MAX_API_KEY_CHARACTERS:
        raise RuntimeError(
            f"LLM_API_KEY cannot exceed {MAX_API_KEY_CHARACTERS} characters"
        )
    if result.get("LLM_API_BASE"):
        normalized = normalize_api_base(
            result["LLM_API_BASE"], allow_insecure_loopback=allow_loopback == "true"
        )
        if normalized != result["LLM_API_BASE"]:
            raise RuntimeError("LLM_API_BASE is not normalized")
        derive_api_endpoint(
            normalized,
            result["LLM_API_MODE"],
            allow_insecure_loopback=allow_loopback == "true",
        )
    return result


def configuration_state(environment: dict[str, str]) -> str:
    present = [name for name in ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL") if environment.get(name)]
    any_value = any(value for name, value in environment.items() if name.startswith("LLM_"))
    if not present and not any_value:
        return "missing"
    if len(present) != 3:
        return "partial"
    return "complete"


def _acl_process_environment(path: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        upper = name.upper()
        if is_ambient_llm_environment_name(upper):
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
    command = [
        icacls,
        str(path),
        "/inheritance:r",
        "/grant:r",
        f"*{sid}:{inheritance}",
        f"*S-1-5-18:{inheritance}",
        f"*S-1-5-32-544:{inheritance}",
    ]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, env=acl_environment
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
    acl_environment = _acl_process_environment(path)
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
        env=acl_environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise RuntimeError(f"Private configuration ACL could not be read: {path} ({detail})")
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Private configuration ACL returned invalid data: {path}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"Private configuration ACL returned invalid data: {path}")
    return document


def _validate_windows_acl(document: dict[str, Any], current_sid: str, path: Path) -> None:
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
        [icacls, str(path), "/verify"], check=False, capture_output=True, text=True
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
    expected = (0o700, 0o600)
    for target, mode in zip(targets, expected, strict=True):
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


def write_private_environment(path: Path, environment: dict[str, str]) -> None:
    selected = _configuration_defaults(environment)
    _validate_environment_protocol(selected)
    if configuration_state(selected) != "complete":
        raise ValueError("A complete LLM configuration is required")
    if len(selected["LLM_API_KEY"]) > MAX_API_KEY_CHARACTERS:
        raise ValueError(
            f"API Key cannot exceed {MAX_API_KEY_CHARACTERS} characters"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    protect_private_path(path.parent, directory=True)
    lines = [f"{name}={selected[name]}" for name in LLM_NAMES]
    lines.append("LOG_LEVEL=INFO")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        if os.name != "nt":
            fchmod = getattr(os, "fchmod", None)
            if fchmod is None:  # pragma: no cover - guarded POSIX branch
                raise RuntimeError("POSIX fchmod is unavailable")
            fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        protect_private_path(temporary, directory=False)
        parsed = parse_private_environment(temporary)
        if any(parsed.get(name) != selected[name] for name in LLM_NAMES):
            raise RuntimeError("Staged LLM configuration failed read-back validation")
        os.replace(temporary, path)
        protect_private_path(path, directory=False)
        expected = {name: selected[name] for name in LLM_NAMES}
        expected["LOG_LEVEL"] = "INFO"
        if parse_private_environment(path) != expected:
            raise RuntimeError("Saved LLM configuration failed read-back validation")
    finally:
        temporary.unlink(missing_ok=True)


def migrate_private_environment_permissions(path: Path) -> None:
    """Upgrade an existing supported config without reading it before protection."""

    if not path.exists():
        return
    if path.is_symlink() or not path.is_file() or path.parent.is_symlink():
        raise RuntimeError(f"Private configuration must be a regular, non-linked file: {path}")
    protect_private_path(path.parent, directory=True)
    protect_private_path(path, directory=False)
    # Validate only after the old file is protected. Values, especially the key,
    # are never rewritten or emitted during this migration.
    parse_private_environment(path)


CONFIGURATION_TEST_INTENT_KEY = "_SOCIALGRAPH_TEST_LLM_REQUESTED"


def configuration_summary(
    environment: dict[str, str],
    *,
    preset_id: str | None = None,
    connection_kind: str | None = None,
) -> dict[str, Any]:
    """Return a display-safe configuration summary that never contains the key."""

    selected = _configuration_defaults(environment)
    _validate_environment_protocol(selected)
    allow_loopback = selected["LLM_ALLOW_INSECURE_LOOPBACK"] == "true"
    api_base = normalize_api_base(
        selected.get("LLM_API_BASE", ""),
        allow_insecure_loopback=allow_loopback,
    )
    inferred_kind = (
        "custom_relay"
        if preset_id in {"custom", "custom_anthropic"}
        else "direct"
    )
    return {
        "schemaVersion": "socialgraph-fm.llm-configuration-summary/1.0",
        "presetId": preset_id,
        "connectionKind": connection_kind or inferred_kind,
        "apiBase": api_base,
        "endpoint": derive_api_endpoint(
            api_base,
            selected["LLM_API_MODE"],
            allow_insecure_loopback=allow_loopback,
        ),
        "apiMode": selected["LLM_API_MODE"],
        "authScheme": selected["LLM_AUTH_SCHEME"],
        "anthropicVersion": selected["LLM_ANTHROPIC_VERSION"] or None,
        "model": selected.get("LLM_MODEL") or None,
        "timeoutSeconds": int(selected["LLM_TIMEOUT_SECONDS"]),
        "keyConfigured": bool(selected.get("LLM_API_KEY")),
    }


def _prompt_line(
    stdin: TextIO,
    stdout: TextIO,
    prompt: str,
    *,
    default: str | None = None,
) -> str:
    suffix = f" [{default}]" if default is not None else ""
    stdout.write(f"{prompt}{suffix}: ")
    stdout.flush()
    raw = stdin.readline()
    if raw == "":
        raise RuntimeError("LLM configuration input was cancelled")
    selected = raw.rstrip("\r\n")
    return default if not selected and default is not None else selected


def _prompt_choice(
    stdin: TextIO,
    stdout: TextIO,
    prompt: str,
    choices: list[str],
    *,
    default: str,
) -> str:
    if default not in choices:
        raise RuntimeError(f"Default {prompt} is not allowed")
    stdout.write(f"{prompt}:\n")
    for index, choice in enumerate(choices, start=1):
        marker = " (default)" if choice == default else ""
        stdout.write(f"  {index}) {choice}{marker}\n")
    stdout.flush()
    selected = _prompt_line(stdin, stdout, "Selection", default=default).strip()
    if selected in choices:
        return selected
    try:
        return choices[int(selected) - 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"Invalid {prompt} selection") from error


def _print_configuration_summary(stdout: TextIO, summary: dict[str, Any]) -> None:
    stdout.write("LLM configuration summary (API key hidden):\n")
    for label, key in (
        ("Provider", "presetId"),
        ("Connection", "connectionKind"),
        ("API Base", "apiBase"),
        ("Endpoint", "endpoint"),
        ("Protocol", "apiMode"),
        ("Authentication", "authScheme"),
        ("Anthropic version", "anthropicVersion"),
        ("Model", "model"),
        ("Timeout seconds", "timeoutSeconds"),
    ):
        value = summary.get(key)
        if value is not None:
            stdout.write(f"  {label}: {value}\n")
    stdout.write("  API key: configured (hidden)\n")
    stdout.flush()


def configure_environment(
    *,
    preset_catalog: Path,
    preset: str | None,
    api_base: str | None,
    model: str | None,
    api_mode: str | None,
    auth_scheme: str | None = None,
    anthropic_version: str | None = None,
    timeout_seconds: int,
    api_key_stdin: bool,
    allow_insecure_loopback: bool,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    test_llm: bool | None = None,
) -> dict[str, str]:
    presets = read_presets(preset_catalog)
    interactive = stdin.isatty()
    selected_id = preset
    if selected_id is None:
        if not interactive:
            raise RuntimeError("--preset is required in a non-interactive terminal")
        ordered = list(presets)
        stdout.write("LLM provider:\n")
        for index, identifier in enumerate(ordered, start=1):
            stdout.write(f"  {index}) {presets[identifier]['displayName']}\n")
        stdout.flush()
        choice = _prompt_line(
            stdin, stdout, "Provider", default="openai_responses"
        ).strip()
        if choice in presets:
            selected_id = choice
        else:
            try:
                selected_id = ordered[int(choice) - 1]
            except (ValueError, IndexError) as error:
                raise RuntimeError("Invalid LLM preset selection") from error
    if selected_id not in presets:
        raise RuntimeError(f"Unknown LLM preset: {selected_id}")
    selected = presets[selected_id]
    base = api_base or selected.get("apiBase")
    if not base:
        if not interactive:
            raise RuntimeError("--api-base is required for the custom preset")
        base = _prompt_line(stdin, stdout, "API Base").strip()
    elif interactive:
        stdout.write(f"API Base: {base}\n")
        stdout.flush()

    mode_default = api_mode or str(selected["defaultApiMode"])
    allowed_modes = [str(value) for value in selected["allowedApiModes"]]
    if mode_default not in allowed_modes:
        raise RuntimeError(f"Preset {selected_id} does not allow API mode {mode_default}")
    mode = (
        _prompt_choice(
            stdin,
            stdout,
            "Protocol",
            allowed_modes,
            default=mode_default,
        )
        if interactive
        else mode_default
    )

    selected_model = model
    if not selected_model and not interactive:
        raise RuntimeError("--model is required in a non-interactive terminal")
    if interactive:
        selected_model = _prompt_line(
            stdin,
            stdout,
            "Model ID",
            default=selected_model,
        ).strip()
    if not selected_model:
        if not interactive:
            raise RuntimeError("--model is required in a non-interactive terminal")
        raise RuntimeError("Model ID cannot be empty")

    if interactive:
        timeout_text = _prompt_line(
            stdin,
            stdout,
            "Timeout seconds",
            default=str(timeout_seconds),
        ).strip()
        try:
            timeout_seconds = int(timeout_text)
        except ValueError as error:
            raise RuntimeError("Timeout seconds must be an integer") from error
    if not 1 <= timeout_seconds <= 60:
        raise ValueError("timeout-seconds must be between 1 and 60")

    auth_default = auth_scheme or str(selected["defaultAuthScheme"])
    allowed_auth = [str(value) for value in selected["allowedAuthSchemes"]]
    if auth_default not in allowed_auth:
        raise RuntimeError(
            f"Preset {selected_id} does not allow auth scheme {auth_default}"
        )
    resolved_auth = (
        _prompt_choice(
            stdin,
            stdout,
            "Relay authentication",
            allowed_auth,
            default=auth_default,
        )
        if interactive and selected["connectionKind"] == "custom_relay"
        else auth_default
    )

    resolved_anthropic_version = ""
    if mode == "anthropic_messages":
        resolved_anthropic_version = (
            anthropic_version
            or str(selected.get("anthropicVersion") or DEFAULT_ANTHROPIC_VERSION)
        )
        if not _ANTHROPIC_VERSION.fullmatch(resolved_anthropic_version):
            raise ValueError("anthropic-version must use YYYY-MM-DD")
    if api_key_stdin:
        key = stdin.readline().rstrip("\r\n")
    elif interactive:
        key = getpass.getpass("API Key: ")
    else:
        raise RuntimeError("Use --api-key-stdin for non-interactive API key input")
    if len(key) > MAX_API_KEY_CHARACTERS:
        raise ValueError(f"API Key cannot exceed {MAX_API_KEY_CHARACTERS} characters")

    connection_kind = str(selected["connectionKind"])
    if connection_kind == "direct" and api_base is not None:
        explicit_base = normalize_api_base(
            str(base), allow_insecure_loopback=allow_insecure_loopback
        )
        if explicit_base != selected.get("apiBase"):
            connection_kind = "custom_relay"
    normalize_base = (
        normalize_relay_api_base
        if connection_kind == "custom_relay"
        else normalize_api_base
    )
    environment = {
        "LLM_API_BASE": normalize_base(
            str(base), allow_insecure_loopback=allow_insecure_loopback
        ),
        "LLM_API_KEY": _single_line("API Key", key),
        "LLM_MODEL": _single_line("Model", selected_model),
        "LLM_API_MODE": mode,
        "LLM_AUTH_SCHEME": resolved_auth,
        "LLM_ANTHROPIC_VERSION": resolved_anthropic_version,
        "LLM_TIMEOUT_SECONDS": str(timeout_seconds),
        "LLM_ALLOW_INSECURE_LOOPBACK": str(bool(allow_insecure_loopback)).lower(),
        "LLM_VERIFICATION_STATUS": "configured_unverified",
    }
    if interactive:
        summary = configuration_summary(
            environment,
            preset_id=selected_id,
            connection_kind=connection_kind,
        )
        _print_configuration_summary(stdout, summary)
        if connection_kind == "custom_relay":
            stdout.write(
                "Warning: the relay will receive all content sent to the configured model; "
                "use only a service you trust.\n"
            )
        stdout.write(
            "The compatibility check sends one fixed, non-sensitive model request and may incur a charge.\n"
        )
        stdout.flush()
        requested = test_llm
        if requested is None:
            answer = _prompt_line(
                stdin,
                stdout,
                "Run the compatibility check now? (Y/n)",
                default="y",
            ).strip().lower()
            if answer not in {"y", "yes", "n", "no"}:
                raise RuntimeError("Choose yes or no for the compatibility check")
            requested = answer in {"y", "yes"}
        environment[CONFIGURATION_TEST_INTENT_KEY] = str(bool(requested)).lower()
    return environment
