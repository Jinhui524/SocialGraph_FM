from __future__ import annotations

from functools import lru_cache
import ipaddress
from pathlib import Path
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .gfm_core_schemas import MAX_INTERNAL_REQUEST_BYTES


_ENCODED_CONTROL = re.compile(r"%0[0ad]", re.IGNORECASE)
_PERCENT_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_ANTHROPIC_VERSION = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LLM_API_MODES = ("chat_completions", "responses", "anthropic_messages")
LLM_AUTH_SCHEMES = ("bearer", "x-api-key")
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
MAX_LLM_API_KEY_CHARACTERS = 8_192


def _contains_encoded_control(value: str) -> bool:
    return any(
        int(match.group(1), 16) <= 0x1F or int(match.group(1), 16) == 0x7F
        for match in _PERCENT_ESCAPE.finditer(value)
    )


def _normalized_llm_host(value: str) -> tuple[str, bool]:
    if not value or any(character.isspace() for character in value) or "%" in value:
        raise ValueError("llm_api_base hostname is invalid")
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        try:
            host = value.encode("idna").decode("ascii").lower()
        except UnicodeError as error:
            raise ValueError("llm_api_base hostname is invalid") from error
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
            raise ValueError("llm_api_base hostname is invalid") from None
        return host, host.rstrip(".") == "localhost"
    return address.compressed.lower(), address.is_loopback


def validate_llm_api_base(value: str, *, allow_insecure_loopback: bool) -> str:
    """Validate an OpenAI-compatible base URL without performing DNS resolution."""

    normalized = value.rstrip("/")
    if (
        not normalized
        or value != value.strip()
        or any(character.isspace() for character in normalized)
        or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
    ):
        raise ValueError("llm_api_base cannot contain control characters")
    if re.match(r"^https?://https?://", normalized, re.IGNORECASE) or "\\" in normalized:
        raise ValueError("llm_api_base has an invalid or repeated protocol prefix")
    if "?" in normalized or "#" in normalized:
        raise ValueError("llm_api_base cannot contain a query string or fragment")
    if _ENCODED_CONTROL.search(normalized) or _contains_encoded_control(normalized):
        raise ValueError("llm_api_base cannot contain encoded control characters")
    try:
        parts = urlsplit(normalized)
        parsed_port = parts.port
    except ValueError as error:
        raise ValueError("llm_api_base contains an invalid host or port") from error
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("llm_api_base must be an absolute HTTP(S) URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("llm_api_base cannot contain embedded credentials")
    host, loopback = _normalized_llm_host(parts.hostname)
    if parts.scheme == "http":
        if not allow_insecure_loopback or not loopback:
            raise ValueError(
                "remote llm_api_base URLs must use HTTPS; HTTP requires an explicitly "
                "enabled loopback endpoint"
            )
    hostname = f"[{host}]" if ":" in host else host
    port = f":{parsed_port}" if parsed_port is not None else ""
    return f"{parts.scheme}://{hostname}{port}{parts.path.rstrip('/')}"


def derive_llm_endpoint(api_base: str, api_mode: str) -> str:
    if api_mode not in LLM_API_MODES:
        raise ValueError(f"Unsupported LLM API mode: {api_mode}")
    base = api_base.rstrip("/")
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


class Settings(BaseSettings):
    """Runtime configuration loaded only by the server process."""

    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    llm_api_base: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    llm_api_mode: Literal[
        "responses", "chat_completions", "anthropic_messages"
    ] = "chat_completions"
    llm_auth_scheme: Literal["bearer", "x-api-key"] | None = None
    llm_anthropic_version: str | None = None
    llm_timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    llm_allow_insecure_loopback: bool = False
    llm_verification_status: Literal[
        "configured_unverified", "call_succeeded", "fallback"
    ] = "configured_unverified"
    dataset_upload_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024)
    dataset_archive_max_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1024,
        le=250 * 1024 * 1024,
    )
    dataset_archive_max_files: int = Field(default=100, ge=1, le=1_000)
    dataset_storage_root: str = "artifacts/dataset-store"
    inspection_cache_ttl_seconds: int = Field(default=900, ge=30, le=86_400)
    inspection_cache_max_bytes: int = Field(
        default=512 * 1024 * 1024,
        ge=1024,
        le=8 * 1024 * 1024 * 1024,
    )
    inspection_cache_max_project_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=1024,
        le=8 * 1024 * 1024 * 1024,
    )
    inspection_cache_max_entry_bytes: int = Field(
        default=128 * 1024 * 1024,
        ge=1024,
        le=4 * 1024 * 1024 * 1024,
    )
    runtime_build_id: str = "dev"
    local_demo_loopback_only: bool = True
    gfm_infrastructure_ready: bool = False
    gfm_service_url: str | None = None
    gfm_session_token_file: str | None = None
    gfm_core_serving_control_file: str | None = None
    gfm_core_run_binding_root: str | None = None
    gfm_core_serving_high_water_root: str | None = None
    gfm_research_run_binding_root: str | None = None
    gfm_global_model_run_binding_root: str | None = None
    gfm_global_model_review_root: str | None = None
    gfm_governance_root: str | None = None
    gfm_governance_bundle_max_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=1024,
        le=512 * 1024 * 1024,
    )
    gfm_governance_expanded_max_bytes: int = Field(
        default=1024 * 1024 * 1024,
        ge=1024,
        le=2 * 1024 * 1024 * 1024,
    )
    gfm_governance_confirmation_ttl_seconds: int = Field(default=300, ge=30, le=900)
    gfm_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    gfm_request_max_bytes: int = Field(
        default=MAX_INTERNAL_REQUEST_BYTES,
        ge=1024,
        le=MAX_INTERNAL_REQUEST_BYTES,
    )
    graph_handoff_token_ttl_seconds: int = Field(default=300, ge=30, le=3_600)
    trusted_array_max_bytes: int = Field(
        default=2 * 1024 * 1024 * 1024,
        ge=64 * 1024 * 1024,
        le=16 * 1024 * 1024 * 1024,
    )
    enable_trusted_local_conversion: bool = False
    trusted_data_roots: str = ""
    trusted_converter_python: str | None = None
    trusted_conversion_timeout_seconds: int = Field(default=900, ge=10, le=7_200)
    trusted_conversion_max_files: int = Field(default=10_000, ge=1, le=100_000)
    trusted_conversion_max_source_bytes: int = Field(
        default=10 * 1024 * 1024 * 1024,
        ge=1024,
        le=100 * 1024 * 1024 * 1024,
    )
    trusted_conversion_max_output_bytes: int = Field(
        default=4 * 1024 * 1024 * 1024,
        ge=1024,
        le=20 * 1024 * 1024 * 1024,
    )
    trusted_conversion_memory_mb: int = Field(default=4096, ge=256, le=65_536)
    allowed_origins: str = "http://127.0.0.1:5173,http://localhost:5173"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator(
        "llm_model",
        "trusted_converter_python",
        "gfm_service_url",
        "gfm_session_token_file",
        "gfm_core_serving_control_file",
        "gfm_core_run_binding_root",
        "gfm_core_serving_high_water_root",
        "gfm_research_run_binding_root",
        "gfm_global_model_run_binding_root",
        "gfm_global_model_review_root",
        "gfm_governance_root",
        mode="before",
    )
    @classmethod
    def empty_string_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("llm_api_base", mode="before")
    @classmethod
    def empty_api_base_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def empty_secret_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            if not value.strip():
                return None
            if value != value.strip() or any(
                ord(character) < 32 or ord(character) == 127 for character in value
            ):
                raise ValueError("llm_api_key must be non-empty single-line text")
            if len(value) > MAX_LLM_API_KEY_CHARACTERS:
                raise ValueError(
                    f"llm_api_key cannot exceed {MAX_LLM_API_KEY_CHARACTERS} characters"
                )
        return value

    @field_validator("llm_api_mode", mode="before")
    @classmethod
    def empty_mode_to_default(cls, value: object) -> object:
        return "chat_completions" if isinstance(value, str) and not value.strip() else value

    @field_validator("llm_auth_scheme", mode="before")
    @classmethod
    def empty_auth_scheme_to_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("llm_anthropic_version", mode="before")
    @classmethod
    def empty_anthropic_version_to_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("llm_timeout_seconds", mode="before")
    @classmethod
    def empty_timeout_to_default(cls, value: object) -> object:
        return 15.0 if isinstance(value, str) and not value.strip() else value

    @field_validator("llm_allow_insecure_loopback", mode="before")
    @classmethod
    def empty_insecure_flag_to_default(cls, value: object) -> object:
        return False if isinstance(value, str) and not value.strip() else value

    @field_validator("llm_verification_status", mode="before")
    @classmethod
    def empty_verification_status_to_default(cls, value: object) -> object:
        return "configured_unverified" if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def validate_inspection_cache_limits(self) -> Settings:
        key = self.llm_api_key.get_secret_value().strip() if self.llm_api_key else ""
        llm_values = (bool(self.llm_api_base), bool(key), bool(self.llm_model))
        if any(llm_values) and not all(llm_values):
            raise ValueError(
                "llm_api_base, llm_api_key, and llm_model must be configured together"
            )
        if self.llm_api_base:
            self.llm_api_base = validate_llm_api_base(
                self.llm_api_base,
                allow_insecure_loopback=self.llm_allow_insecure_loopback,
            )
        if self.llm_auth_scheme is None:
            self.llm_auth_scheme = (
                "x-api-key"
                if self.llm_api_mode == "anthropic_messages"
                else "bearer"
            )
        if self.llm_api_mode == "anthropic_messages":
            self.llm_anthropic_version = (
                self.llm_anthropic_version or DEFAULT_ANTHROPIC_VERSION
            )
            if not _ANTHROPIC_VERSION.fullmatch(self.llm_anthropic_version):
                raise ValueError("llm_anthropic_version must use YYYY-MM-DD")
        elif self.llm_anthropic_version is not None:
            raise ValueError(
                "llm_anthropic_version is valid only with anthropic_messages"
            )
        if self.inspection_cache_max_project_bytes > self.inspection_cache_max_bytes:
            raise ValueError(
                "inspection_cache_max_project_bytes 不能超过 inspection_cache_max_bytes"
            )
        if self.inspection_cache_max_entry_bytes > self.inspection_cache_max_bytes:
            raise ValueError(
                "inspection_cache_max_entry_bytes 不能超过 inspection_cache_max_bytes"
            )
        if bool(self.gfm_service_url) != bool(self.gfm_session_token_file):
            raise ValueError(
                "gfm_service_url and gfm_session_token_file must be configured together"
            )
        if self.gfm_governance_bundle_max_bytes > self.gfm_governance_expanded_max_bytes:
            raise ValueError(
                "gfm_governance_bundle_max_bytes cannot exceed expanded_max_bytes"
            )
        return self

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def llm_configured(self) -> bool:
        key = self.llm_api_key.get_secret_value().strip() if self.llm_api_key else ""
        return bool(self.llm_api_base and self.llm_model and key)

    @property
    def trusted_roots(self) -> list[Path]:
        """Configured trusted roots; semicolon/newline keeps Windows drive colons intact."""

        values = self.trusted_data_roots.replace("\r", "\n").replace(";", "\n").splitlines()
        return [Path(value.strip()).expanduser() for value in values if value.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


__all__ = [
    "DEFAULT_ANTHROPIC_VERSION",
    "LLM_API_MODES",
    "LLM_AUTH_SCHEMES",
    "Settings",
    "derive_llm_endpoint",
    "get_settings",
    "validate_llm_api_base",
]
