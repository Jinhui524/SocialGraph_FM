from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

from app.config import Settings
from app.dataset_imports import DatasetImportService
from app.main import create_app
from app.trusted_conversion import TrustedConversionService


def _settings(
    tmp_path: Path,
    trusted_root: Path,
    *,
    enabled: bool = True,
    converter_python: str = sys.executable,
) -> Settings:
    return Settings(
        llm_api_base=None,
        llm_api_key=None,
        llm_model=None,
        enable_trusted_local_conversion=enabled,
        trusted_data_roots=str(trusted_root),
        trusted_converter_python=converter_python,
        dataset_storage_root=str(tmp_path / "store"),
        trusted_conversion_timeout_seconds=30,
        trusted_conversion_max_source_bytes=10 * 1024 * 1024,
        trusted_conversion_max_output_bytes=10 * 1024 * 1024,
    )


def _write_geom_dataset(source: Path) -> None:
    raw = source / "toy" / "raw"
    raw.mkdir(parents=True)
    (raw / "out1_node_feature_label.txt").write_text(
        "node_id\tfeature\tlabel\n0\t1,0\t0\n1\t0,1\t1\n",
        encoding="utf-8",
    )
    (raw / "out1_graph_edges.txt").write_text(
        "node_id\tnode_id\n0\t1\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        raw / "toy_split_0.npz",
        train_mask=np.asarray([1, 0], dtype=np.uint8),
        val_mask=np.asarray([0, 1], dtype=np.uint8),
        test_mask=np.asarray([0, 0], dtype=np.uint8),
    )


@pytest.mark.anyio
async def test_trusted_conversion_is_disabled_by_default(tmp_path: Path) -> None:
    source = tmp_path / "data"
    source.mkdir()
    app = create_app(_settings(tmp_path, source, enabled=False))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
    ) as client:
        response = await client.post(
            "/api/v1/dataset-imports/inspect-local",
            json={"sourcePath": str(source)},
        )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "TRUSTED_CONVERSION_DISABLED"


@pytest.mark.anyio
async def test_trusted_conversion_rejects_path_outside_allow_list(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    outside = tmp_path / "outside"
    trusted.mkdir()
    outside.mkdir()
    app = create_app(_settings(tmp_path, trusted))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
    ) as client:
        response = await client.post(
            "/api/v1/dataset-imports/inspect-local",
            json={"sourcePath": str(outside)},
        )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "SOURCE_OUTSIDE_TRUSTED_ROOTS"


@pytest.mark.anyio
async def test_trusted_conversion_rejects_non_loopback_client(tmp_path: Path) -> None:
    source = tmp_path / "trusted"
    source.mkdir()
    app = create_app(_settings(tmp_path, source))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("203.0.113.10", 5000)),
        base_url="http://api.example",
    ) as client:
        response = await client.post(
            "/api/v1/dataset-imports/inspect-local",
            json={"sourcePath": str(source)},
        )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "TRUSTED_CONVERSION_LOOPBACK_ONLY"


@pytest.mark.anyio
async def test_cancel_consumes_pending_authorization(tmp_path: Path) -> None:
    source = tmp_path / "trusted"
    _write_geom_dataset(source)
    app = create_app(_settings(tmp_path, source))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
    ) as client:
        inspected = (
            await client.post(
                "/api/v1/dataset-imports/inspect-local",
                json={"sourcePath": str(source)},
            )
        ).json()
        cancelled = await client.post(
            f"/api/v1/dataset-imports/local-jobs/{inspected['id']}/cancel"
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        authorized = await client.post(
            f"/api/v1/dataset-imports/local-jobs/{inspected['id']}/authorize",
            json={
                "authorizationToken": inspected["authorizationToken"],
                "confirmTrusted": True,
            },
        )
    assert authorized.status_code == 409


@pytest.mark.anyio
async def test_missing_converter_python_becomes_clear_job_error(tmp_path: Path) -> None:
    source = tmp_path / "trusted"
    _write_geom_dataset(source)
    missing = str(tmp_path / "missing-research-python.exe")
    app = create_app(_settings(tmp_path, source, converter_python=missing))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
    ) as client:
        inspected = (
            await client.post(
                "/api/v1/dataset-imports/inspect-local",
                json={"sourcePath": str(source)},
            )
        ).json()
        await client.post(
            f"/api/v1/dataset-imports/local-jobs/{inspected['id']}/authorize",
            json={
                "authorizationToken": inspected["authorizationToken"],
                "confirmTrusted": True,
            },
        )
        for _ in range(20):
            job = (
                await client.get(
                    f"/api/v1/dataset-imports/local-jobs/{inspected['id']}"
                )
            ).json()
            if job["status"] == "failed":
                break
            await asyncio.sleep(0.01)
    assert job["status"] == "failed"
    assert job["issues"][-1]["code"] == "CONVERTER_PYTHON_NOT_FOUND"


@pytest.mark.anyio
async def test_authorized_geom_conversion_persists_artifact_across_restart(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trusted-data"
    _write_geom_dataset(source)
    settings = _settings(tmp_path, source)
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
    ) as client:
        inspected = await client.post(
            "/api/v1/dataset-imports/inspect-local",
            json={"sourcePath": str(source)},
        )
        assert inspected.status_code == 200
        inspection = inspected.json()
        assert inspection["status"] == "awaiting_authorization"
        assert inspection["authorizationToken"]
        assert inspection["datasets"][0]["name"] == "toy"

        authorized = await client.post(
            f"/api/v1/dataset-imports/local-jobs/{inspection['id']}/authorize",
            json={
                "authorizationToken": inspection["authorizationToken"],
                "confirmTrusted": True,
            },
        )
        assert authorized.status_code == 200
        assert authorized.json()["status"] in {"queued", "running"}

        second_authorization = await client.post(
            f"/api/v1/dataset-imports/local-jobs/{inspection['id']}/authorize",
            json={
                "authorizationToken": inspection["authorizationToken"],
                "confirmTrusted": True,
            },
        )
        assert second_authorization.status_code == 409

        job: dict[str, object] = {}
        for _ in range(100):
            response = await client.get(
                f"/api/v1/dataset-imports/local-jobs/{inspection['id']}"
            )
            job = response.json()
            if job["status"] in {"succeeded", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.05)
        assert job["status"] == "succeeded", job
        artifact_id = str(job["artifactIds"][0])  # type: ignore[index]

        listed = await client.get("/api/v1/dataset-artifacts")
        assert listed.status_code == 200
        assert any(item["id"] == artifact_id for item in listed.json())

    restarted = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=restarted),
        base_url="http://127.0.0.1",
    ) as client:
        fetched = await client.get(f"/api/v1/dataset-artifacts/{artifact_id}")
    assert fetched.status_code == 200
    artifact = fetched.json()
    assert artifact["datasetName"] == "toy"
    assert artifact["canonicalGraphHash"]
    assert artifact["rawManifest"]["sourcePath"] == str(source.resolve())
    assert artifact["derivedManifest"]["splitNames"] == [
        "train_mask",
        "val_mask",
        "test_mask",
    ]
    artifact_directory = Path(settings.dataset_storage_root) / "artifacts" / artifact_id
    assert (artifact_directory / "graph.npz").is_file()
    assert (artifact_directory / "raw-manifest.json").is_file()


def test_converter_environment_excludes_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "trusted"
    source.mkdir()
    settings = _settings(tmp_path, source)
    service = TrustedConversionService(settings, DatasetImportService(settings))
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    environment = service._minimal_environment(tmp_path)
    assert "LLM_API_KEY" not in environment
    assert "OPENAI_API_KEY" not in environment
    assert all("TOKEN" not in name and not name.endswith("API_KEY") for name in environment)
