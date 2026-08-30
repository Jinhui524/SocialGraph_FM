from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

from tools.dataset_store_preflight import audit_store, backup_store


def _npz_payload() -> bytes:
    import io

    output = io.BytesIO()
    np.savez_compressed(
        output,
        x=np.eye(4, dtype=np.float32),
        edge_index=np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        y=np.asarray([0, 0, 1, 1], dtype=np.int64),
        train_mask=np.asarray([1, 1, 0, 0], dtype=np.uint8),
    )
    return output.getvalue()


@pytest.mark.anyio
async def test_audit_is_read_only_and_classifies_current_and_legacy(
    api_client: httpx.AsyncClient,
    isolated_dataset_store: Path,
) -> None:
    inspected = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={"file": ("graph.npz", _npz_payload(), "application/octet-stream")},
    )
    assert inspected.status_code == 200
    committed = await api_client.post(
        f"/api/v1/dataset-imports/{inspected.json()['id']}/commit"
    )
    assert committed.status_code == 200
    current_id = committed.json()["id"]

    legacy_id = "legacy-artifact"
    legacy_dir = isolated_dataset_store / "artifacts" / legacy_id
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "graph.npz").write_bytes(_npz_payload())
    legacy = {
        "id": legacy_id,
        "inspectionId": "legacy-inspection",
        "sourceFormat": "graph_npz",
        "sourceFiles": ["graph.npz"],
        "checksum": "legacy-source",
        "canonicalGraphHash": "legacy-graph",
        "scope": "complete",
        "profile": {},
        "graphView": {
            "id": "legacy-view",
            "nodes": [],
            "edges": [],
            "summary": {
                "nodeCount": 0,
                "edgeCount": 0,
                "density": 0,
                "connectedComponents": 0,
                "visibleNodeCount": 0,
                "visibleEdgeCount": 0,
                "partialPreview": False,
            },
        },
        "rawManifest": {"schemaVersion": "1.0", "sourceFormat": "graph_npz"},
        "derivedManifest": {},
        "createdAt": "2026-08-11T00:00:00Z",
    }
    (legacy_dir / "artifact.json").write_text(json.dumps(legacy), encoding="utf-8")
    database = isolated_dataset_store / "datasets.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO dataset_artifacts
            (id, dataset_name, checksum, canonical_graph_hash, scope,
             created_at, artifact_json, tensor_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                legacy_id,
                "legacy",
                "legacy-source",
                "legacy-graph",
                "complete",
                "2026-08-11T00:00:00Z",
                json.dumps(legacy),
                f"artifacts/{legacy_id}/graph.npz",
            ),
        )

    before = database.stat().st_mtime_ns
    report = audit_store(isolated_dataset_store)
    after = database.stat().st_mtime_ns

    assert before == after
    assert report["mode"] == "read-only"
    assert report["mutationsPerformed"] is False
    assert current_id in {
        item["artifactId"] for item in report["categories"]["compatible"]
    }
    assert legacy_id in {
        item["artifactId"] for item in report["categories"]["needs-reimport"]
    }
    assert report["counts"]["quarantined"] == 0


@pytest.mark.anyio
async def test_backup_is_explicit_non_overwriting_and_self_auditing(
    api_client: httpx.AsyncClient,
    isolated_dataset_store: Path,
    tmp_path: Path,
) -> None:
    inspected = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={"file": ("graph.npz", _npz_payload(), "application/octet-stream")},
    )
    committed = await api_client.post(
        f"/api/v1/dataset-imports/{inspected.json()['id']}/commit"
    )
    assert committed.status_code == 200

    result = backup_store(isolated_dataset_store, tmp_path / "backups")
    destination = Path(result["backup"]["destination"])
    assert destination != isolated_dataset_store
    assert (destination / "datasets.sqlite3").is_file()
    assert (destination / "backup-manifest.json").is_file()
    assert result["backup"]["deletionsPerformed"] is False
    assert result["backup"]["sourceMutated"] is False
    assert result["audit"]["counts"]["compatible"] == 1

    second = backup_store(isolated_dataset_store, tmp_path / "backups")
    assert second["backup"]["destination"] != result["backup"]["destination"]


@pytest.mark.anyio
async def test_direct_cli_resolves_project_schema_outside_repository_cwd(
    api_client: httpx.AsyncClient,
    isolated_dataset_store: Path,
    tmp_path: Path,
) -> None:
    inspected = await api_client.post(
        "/api/v1/dataset-imports/inspect",
        files={"file": ("graph.npz", _npz_payload(), "application/octet-stream")},
    )
    committed = await api_client.post(
        f"/api/v1/dataset-imports/{inspected.json()['id']}/commit"
    )
    assert committed.status_code == 200
    script = Path(__file__).resolve().parents[1] / "tools" / "dataset_store_preflight.py"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(script), "--store", str(isolated_dataset_store)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["counts"]["compatible"] == 1
    assert report["counts"]["quarantined"] == 0
