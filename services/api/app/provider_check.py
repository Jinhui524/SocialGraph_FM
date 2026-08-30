from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from .config import Settings
from .provider import OpenAICompatibleProvider, ProviderFailure


_CHECK_FIELD = "socialgraph_fm_connection_check"
CHECK_SCHEMA_VERSION = "socialgraph-fm.llm-provider-check/1.0"


def check_result(*, ok: bool, code: str) -> dict[str, object]:
    """Build the only machine-readable result exposed to the runtime launcher."""

    return {"schemaVersion": CHECK_SCHEMA_VERSION, "ok": ok, "code": code}


def _is_valid_check_result(result: dict[str, Any]) -> bool:
    return result.get(_CHECK_FIELD) == "ok"


async def verify_provider() -> None:
    provider = OpenAICompatibleProvider(Settings())
    try:
        result = await provider.generate(
            "You are a connection verifier. Return only one JSON object and no prose.",
            f'Return exactly {{"{_CHECK_FIELD}":"ok"}}.',
        )
        if not _is_valid_check_result(result):
            raise ProviderFailure(
                "LLM_INVALID_RESPONSE",
                "LLM connection check did not return the expected marker",
            )
    finally:
        await provider.aclose()


def main() -> int:
    try:
        asyncio.run(verify_provider())
    except ProviderFailure as error:
        print(
            json.dumps(check_result(ok=False, code=error.code), sort_keys=True),
            file=sys.stderr,
        )
        return 1
    except (RuntimeError, ValueError):
        print(
            json.dumps(
                check_result(ok=False, code="LLM_CONFIGURATION_ERROR"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(check_result(ok=True, code="OK"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CHECK_SCHEMA_VERSION", "check_result", "main", "verify_provider"]
