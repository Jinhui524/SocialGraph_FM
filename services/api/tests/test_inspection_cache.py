from __future__ import annotations

import json
from datetime import timedelta

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.dataset_imports import DatasetImportService
from app.main import create_app

from .test_dataset_imports import graph_version_handoff


async def _client_and_service(
    settings: Settings,
) -> tuple[httpx.AsyncClient, DatasetImportService]:
    app = create_app(settings)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )
    return client, app.state.dataset_imports


async def _inspect(
    client: httpx.AsyncClient,
    graph_id: str,
    *,
    project_id: str | None = None,
) -> httpx.Response:
    payload = json.loads(graph_version_handoff(graphVersionId=graph_id))
    headers = {"X-SocialGraph-Project-ID": project_id} if project_id is not None else None
    return await client.post(
        "/api/v1/dataset-imports/inspect",
        headers=headers,
        files={
            "file": (
                f"{graph_id}.sgfm-graph.json",
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json",
            )
        },
    )


def test_settings_reject_project_cache_budget_above_process_budget() -> None:
    with pytest.raises(ValidationError, match="inspection_cache_max_project_bytes"):
        Settings(
            inspection_cache_max_bytes=256 * 1024 * 1024,
            inspection_cache_max_project_bytes=512 * 1024 * 1024,
            inspection_cache_max_entry_bytes=128 * 1024 * 1024,
        )


@pytest.mark.anyio
async def test_inspection_project_header_defaults_and_rejects_unsafe_values(
    unconfigured_settings: Settings,
) -> None:
    client, service = await _client_and_service(unconfigured_settings)
    async with client:
        default_scope = await _inspect(client, "default-scope")
        assert default_scope.status_code == 200
        assert (
            service._inspections[default_scope.json()["id"]].project_id
            == "local-default"
        )

        explicit_scope = await _inspect(
            client,
            "explicit-scope",
            project_id="project.alpha-1:local",
        )
        assert explicit_scope.status_code == 200
        assert (
            service._inspections[explicit_scope.json()["id"]].project_id
            == "project.alpha-1:local"
        )

        unsafe = await _inspect(client, "unsafe-scope", project_id="bad project")
        assert unsafe.status_code == 422
        too_long = await _inspect(client, "long-scope", project_id="a" * 65)
        assert too_long.status_code == 422
        assert service.inspection_cache_count == 2


@pytest.mark.anyio
async def test_inspection_cancel_and_successful_commit_release_memory(
    unconfigured_settings: Settings,
) -> None:
    client, service = await _client_and_service(unconfigured_settings)
    async with client:
        first = await _inspect(client, "cancelled")
        assert first.status_code == 200
        assert service.inspection_cache_count == 1
        assert service.inspection_cache_bytes > 0
        assert service.inspection_cache_project_bytes("local-default") > 0

        cancelled = await client.post(
            f"/api/v1/dataset-imports/{first.json()['id']}/cancel"
        )
        assert cancelled.json() == {"status": "released"}
        assert service.inspection_cache_count == 0
        assert service.inspection_cache_bytes == 0
        assert service.inspection_cache_project_bytes("local-default") == 0

        second = await _inspect(client, "committed")
        committed = await client.post(
            f"/api/v1/dataset-imports/{second.json()['id']}/commit"
        )
        assert committed.status_code == 200
        assert service.inspection_cache_count == 0
        assert service.inspection_cache_bytes == 0
        assert service.inspection_cache_project_bytes("local-default") == 0


@pytest.mark.anyio
async def test_inspection_commit_exception_releases_memory(
    unconfigured_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service = await _client_and_service(unconfigured_settings)
    async with client:
        inspected = await _inspect(client, "failure")
        inspection_id = inspected.json()["id"]

        def fail_commit(*args: object, **kwargs: object) -> None:
            raise RuntimeError("injected commit failure")

        monkeypatch.setattr(service, "_commit_payload", fail_commit)
        with pytest.raises(RuntimeError, match="injected commit failure"):
            service.commit(inspection_id)
        assert service.inspection_cache_count == 0
        assert service.inspection_cache_bytes == 0
        assert service.inspection_cache_project_bytes("local-default") == 0


@pytest.mark.anyio
async def test_project_budget_evicts_only_same_project_before_global_lru(
    unconfigured_settings: Settings,
) -> None:
    client, service = await _client_and_service(unconfigured_settings)
    async with client:
        beta = await _inspect(client, "beta-0001", project_id="project-beta")
        beta_id = beta.json()["id"]
        beta_bytes = service.inspection_cache_project_bytes("project-beta")

        alpha_first = await _inspect(
            client,
            "alpha-001",
            project_id="project-alpha",
        )
        alpha_first_id = alpha_first.json()["id"]
        alpha_bytes = service.inspection_cache_project_bytes("project-alpha")
        service.settings.inspection_cache_max_project_bytes = (
            alpha_bytes + alpha_bytes // 2
        )

        alpha_second = await _inspect(
            client,
            "alpha-002",
            project_id="project-alpha",
        )
        assert alpha_second.status_code == 200
        alpha_second_id = alpha_second.json()["id"]

        assert beta_id in service._inspections
        assert alpha_first_id not in service._inspections
        assert alpha_second_id in service._inspections
        assert service.inspection_cache_project_bytes("project-beta") == beta_bytes
        assert (
            service.inspection_cache_project_bytes("project-alpha")
            == service._inspections[alpha_second_id].retained_bytes
        )

        released = await client.post(
            f"/api/v1/dataset-imports/{alpha_second_id}/cancel"
        )
        assert released.status_code == 200
        assert service.inspection_cache_project_bytes("project-alpha") == 0
        assert service.inspection_cache_project_bytes("project-beta") == beta_bytes


@pytest.mark.anyio
async def test_inspection_cache_enforces_entry_budget_and_lru_total_budget(
    unconfigured_settings: Settings,
) -> None:
    client, service = await _client_and_service(unconfigured_settings)
    async with client:
        first = await _inspect(client, "lru-first")
        assert first.status_code == 200
        first_id = first.json()["id"]
        retained = service.inspection_cache_bytes
        service.settings.inspection_cache_max_bytes = retained * 2 - 1
        service.settings.inspection_cache_max_entry_bytes = retained * 2 - 1

        second = await _inspect(client, "lru-second")
        assert second.status_code == 200
        assert service.inspection_cache_count == 1
        assert first_id not in service._inspections

        service.settings.inspection_cache_max_entry_bytes = 1024
        oversized = await _inspect(client, "oversized")
        assert oversized.status_code == 413
        assert oversized.json()["detail"]["code"] == "INSPECTION_CACHE_ENTRY_TOO_LARGE"


@pytest.mark.anyio
async def test_inspection_cache_prunes_expired_records(
    unconfigured_settings: Settings,
) -> None:
    client, service = await _client_and_service(unconfigured_settings)
    async with client:
        first = await _inspect(client, "expired", project_id="expiring-project")
        first_id = first.json()["id"]
        assert service.inspection_cache_project_bytes("expiring-project") > 0
        service._inspections[first_id].last_accessed_at -= timedelta(
            seconds=service.settings.inspection_cache_ttl_seconds + 1
        )

        second = await _inspect(client, "fresh", project_id="fresh-project")
        assert second.status_code == 200
        assert first_id not in service._inspections
        assert service.inspection_cache_count == 1
        assert service.inspection_cache_project_bytes("expiring-project") == 0
        assert service.inspection_cache_project_bytes("fresh-project") > 0
