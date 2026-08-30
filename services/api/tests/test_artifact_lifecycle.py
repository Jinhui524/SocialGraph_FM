from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import httpx
import numpy as np
import pytest

from app import dataset_storage


def _npz_bytes() -> bytes:
    output = io.BytesIO()
    np.savez_compressed(
        output,
        x=np.eye(3, dtype=np.float32),
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        y=np.asarray([0, 1, 0], dtype=np.int64),
        train_mask=np.asarray([1, 1, 0], dtype=np.uint8),
    )
    return output.getvalue()


async def _commit_artifact(client: httpx.AsyncClient) -> dict[str, object]:
    inspection = await client.post(
        "/api/v1/dataset-imports/inspect",
        files={"file": ("lifecycle.npz", _npz_bytes(), "application/octet-stream")},
    )
    assert inspection.status_code == 200
    committed = await client.post(
        f"/api/v1/dataset-imports/{inspection.json()['id']}/commit"
    )
    assert committed.status_code == 200
    return committed.json()


@pytest.mark.anyio
async def test_trash_hides_artifact_and_restore_reactivates_it(
    api_client: httpx.AsyncClient,
) -> None:
    artifact = await _commit_artifact(api_client)
    artifact_id = str(artifact["id"])

    trashed = await api_client.post(f"/api/v1/dataset-artifacts/{artifact_id}/trash")
    assert trashed.status_code == 200
    assert trashed.json()["lifecycle"]["status"] == "trashed"
    assert trashed.json()["impact"]["lifecycle"] == "trashed"

    default_list = await api_client.get("/api/v1/dataset-artifacts")
    assert all(item["id"] != artifact_id for item in default_list.json())
    diagnostic_list = await api_client.get(
        "/api/v1/dataset-artifacts", params={"includeTrashed": "true"}
    )
    listed = next(item for item in diagnostic_list.json() if item["id"] == artifact_id)
    assert listed["lifecycle"] == "trashed"
    assert (await api_client.get(f"/api/v1/dataset-artifacts/{artifact_id}")).status_code == 404

    restored = await api_client.post(f"/api/v1/dataset-artifacts/{artifact_id}/restore")
    assert restored.status_code == 200
    assert restored.json()["lifecycle"]["status"] == "active"
    assert (await api_client.get(f"/api/v1/dataset-artifacts/{artifact_id}")).status_code == 200


@pytest.mark.anyio
async def test_purge_rechecks_impact_and_blocks_graph_binding_reference(
    api_client: httpx.AsyncClient,
    isolated_dataset_store: Path,
) -> None:
    artifact = await _commit_artifact(api_client)
    artifact_id = str(artifact["id"])
    await api_client.post(f"/api/v1/dataset-artifacts/{artifact_id}/trash")
    preview = (
        await api_client.get(f"/api/v1/dataset-artifacts/{artifact_id}/deletion-impact")
    ).json()
    assert preview["blockers"] == []
    assert preview["dependents"][0]["kind"] == "embedded_training_ref"
    assert preview["dependents"][0]["blocking"] is False

    with sqlite3.connect(isolated_dataset_store / "datasets.sqlite3") as connection:
        connection.execute(
            """
            INSERT INTO graph_dataset_bindings
            (id, graph_version_id, graph_fact_hash, preparation_hash, artifact_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("binding-race", "graph-race", "a" * 64, "b" * 64, artifact_id, "2026-08-11T00:00:00+00:00"),
        )

    stale = await api_client.post(
        f"/api/v1/dataset-artifacts/{artifact_id}/purge",
        json={"impactHash": preview["impactHash"], "confirmation": artifact_id[-8:]},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "REFERENCE_SET_CHANGED"

    fresh = (
        await api_client.get(f"/api/v1/dataset-artifacts/{artifact_id}/deletion-impact")
    ).json()
    assert fresh["blockers"][0]["kind"] == "graph_dataset_binding"
    blocked = await api_client.post(
        f"/api/v1/dataset-artifacts/{artifact_id}/purge",
        json={"impactHash": fresh["impactHash"], "confirmation": artifact_id[-8:]},
    )
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "ARTIFACT_REFERENCED"
    assert (isolated_dataset_store / "artifacts" / artifact_id / "graph.npz").is_file()


@pytest.mark.anyio
async def test_unreferenced_artifact_requires_confirmation_then_purges(
    api_client: httpx.AsyncClient,
    isolated_dataset_store: Path,
) -> None:
    artifact = await _commit_artifact(api_client)
    artifact_id = str(artifact["id"])
    await api_client.post(f"/api/v1/dataset-artifacts/{artifact_id}/trash")
    impact = (
        await api_client.get(f"/api/v1/dataset-artifacts/{artifact_id}/deletion-impact")
    ).json()

    rejected = await api_client.post(
        f"/api/v1/dataset-artifacts/{artifact_id}/purge",
        json={"impactHash": impact["impactHash"], "confirmation": "wrong"},
    )
    assert rejected.status_code == 422
    purged = await api_client.post(
        f"/api/v1/dataset-artifacts/{artifact_id}/purge",
        json={"impactHash": impact["impactHash"], "confirmation": artifact_id[-8:]},
    )
    assert purged.status_code == 200
    assert purged.json() == {
        "artifactId": artifact_id,
        "purged": True,
        "cleanupPending": False,
    }
    assert not (isolated_dataset_store / "artifacts" / artifact_id).exists()


@pytest.mark.anyio
async def test_orphan_directory_is_only_recovered_by_explicit_action(
    api_client: httpx.AsyncClient,
    isolated_dataset_store: Path,
) -> None:
    artifact = await _commit_artifact(api_client)
    artifact_id = str(artifact["id"])
    with sqlite3.connect(isolated_dataset_store / "datasets.sqlite3") as connection:
        connection.execute("DELETE FROM dataset_artifacts WHERE id = ?", (artifact_id,))

    orphans = await api_client.get("/api/v1/dataset-store/orphans")
    assert orphans.status_code == 200
    orphan = next(item for item in orphans.json() if item["artifactId"] == artifact_id)
    assert orphan["source"] == "artifacts"
    assert orphan["recoverable"] is True
    assert (await api_client.get("/api/v1/dataset-artifacts")).json() == []

    recovered = await api_client.post(
        f"/api/v1/dataset-store/orphans/{artifact_id}/recover"
    )
    assert recovered.status_code == 200
    assert recovered.json()["artifact"]["id"] == artifact_id
    assert recovered.json()["lifecycle"]["status"] == "active"
    assert (await api_client.get("/api/v1/dataset-store/orphans")).json() == []


@pytest.mark.anyio
async def test_failed_file_cleanup_stays_recoverable(
    api_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = await _commit_artifact(api_client)
    artifact_id = str(artifact["id"])
    await api_client.post(f"/api/v1/dataset-artifacts/{artifact_id}/trash")
    impact = (
        await api_client.get(f"/api/v1/dataset-artifacts/{artifact_id}/deletion-impact")
    ).json()

    def fail_cleanup(_: Path) -> None:
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(dataset_storage.shutil, "rmtree", fail_cleanup)
    purged = await api_client.post(
        f"/api/v1/dataset-artifacts/{artifact_id}/purge",
        json={"impactHash": impact["impactHash"], "confirmation": artifact_id[-8:]},
    )
    assert purged.status_code == 200
    assert purged.json()["cleanupPending"] is True
    orphan = next(
        item
        for item in (await api_client.get("/api/v1/dataset-store/orphans")).json()
        if item["artifactId"] == artifact_id
    )
    assert orphan["source"] == "purge_recovery"
    assert orphan["recoverable"] is True

    recovered = await api_client.post(
        f"/api/v1/dataset-store/orphans/{artifact_id}/recover"
    )
    assert recovered.status_code == 200
