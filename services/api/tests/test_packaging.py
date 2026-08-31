from __future__ import annotations

import re
import tomllib
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEV_DISTRIBUTIONS = {"mypy", "pip-audit", "pytest", "ruff"}


def _requirements(path: Path) -> set[str]:
    values: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==", line)
        if match:
            values.add(match.group(1).casefold().replace("_", "-"))
    return values


def _assert_all_records_are_hashed(path: Path) -> None:
    pending = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1].strip() + " "
            continue
        pending += stripped
        assert "--hash=sha256:" in pending
        pending = ""
    assert pending == ""


def test_api_keeps_only_a_hashed_development_lock() -> None:
    development_lock = PROJECT / "requirements-dev.lock"
    development = _requirements(development_lock)

    assert not (PROJECT / "requirements.lock").exists()
    assert not (PROJECT / "requirements.txt").exists()
    assert {"fastapi", "httpx", "numpy", "pydantic", "uvicorn"} <= development
    assert DEV_DISTRIBUTIONS <= development
    _assert_all_records_are_hashed(development_lock)


def test_api_wheel_exposes_the_safe_console_entrypoint() -> None:
    pyproject = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"]["socialgraph-api"] == (
        "app.__main__:console_main"
    )
