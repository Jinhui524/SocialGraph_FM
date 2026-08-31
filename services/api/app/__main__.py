"""Supported single-machine launcher with a loopback-only managed bind."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Sequence

import uvicorn
from pydantic import ValidationError

from .config import Settings
from .provider_check import verify_provider


def selector_event_loop_factory() -> asyncio.AbstractEventLoop:
    """Use the stable Windows socket accept loop for the local API."""
    return asyncio.SelectorEventLoop()


def managed_api_port() -> int:
    raw = os.environ.get("SOCIALGRAPH_CORE_API_PORT", "5173")
    try:
        port = int(raw)
    except ValueError as error:
        raise ValueError("SOCIALGRAPH_CORE_API_PORT must be an integer") from error
    if not 1 <= port <= 65535:
        raise ValueError("SOCIALGRAPH_CORE_API_PORT must be between 1 and 65535")
    return port


def managed_runtime_identity(arguments: Sequence[str]) -> str:
    if len(arguments) != 2 or arguments[0] != "--runtime-identity-root":
        raise ValueError("--runtime-identity-root is required for the managed API process")
    declared = os.path.abspath(arguments[1])
    expected_raw = os.environ.get("GFM_GOVERNANCE_ROOT")
    if not expected_raw:
        raise ValueError("GFM_GOVERNANCE_ROOT is required for runtime identity")
    expected = os.path.abspath(expected_raw)
    if os.path.normcase(declared) != os.path.normcase(expected):
        raise ValueError("managed API runtime identity does not match its environment")
    return declared


def main(arguments: Sequence[str] | None = None) -> None:
    if arguments is not None:
        managed_runtime_identity(arguments)
    try:
        llm_configured = Settings().llm_configured
    except ValidationError:
        llm_configured = False
    if not llm_configured:
        raise RuntimeError("LLM_API_BASE, LLM_MODEL, and LLM_API_KEY are required")
    asyncio.run(verify_provider())
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=managed_api_port(),
        # Uvicorn accepts a loop factory at runtime; its public stub exposes only string aliases.
        loop=selector_event_loop_factory if sys.platform == "win32" else "auto",  # type: ignore[arg-type]
        reload=False,
        access_log=False,
    )


def console_main() -> None:
    """Run the packaged command with the same identity checks as ``python -m app``."""

    main(sys.argv[1:])


if __name__ == "__main__":
    console_main()
