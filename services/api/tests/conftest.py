from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.main import create_app


@pytest.fixture
def isolated_dataset_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = tmp_path / "dataset-store"
    monkeypatch.setenv("DATASET_STORAGE_ROOT", str(root))
    return root


@pytest.fixture(autouse=True)
def _force_isolated_dataset_store(isolated_dataset_store: Path) -> None:
    """Every test, including Settings() created inside it, gets a temporary store."""


@pytest.fixture
def unconfigured_settings(isolated_dataset_store: Path) -> Settings:
    return Settings(
        llm_api_base=None,
        llm_api_key=None,
        llm_model=None,
        allowed_origins="http://localhost:5173",
        dataset_storage_root=str(isolated_dataset_store),
        gfm_infrastructure_ready=False,
    )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def api_client(unconfigured_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(unconfigured_settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


class SequenceProvider:
    def __init__(self, outputs: list[dict[str, Any] | Exception], model: str = "test-model") -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[str, str]] = []
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls.append((system_prompt, user_prompt))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output
