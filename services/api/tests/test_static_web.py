from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_prebuilt_web_is_served_after_api_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web_root = tmp_path / "client"
    assets = web_root / "assets"
    assets.mkdir(parents=True)
    (web_root / "index.html").write_text("<html>SocialGraph-FM</html>", encoding="utf-8")
    (assets / "app.js").write_text("console.log('ready')", encoding="utf-8")
    monkeypatch.setenv("SOCIALGRAPH_WEB_CLIENT_ROOT", str(web_root))

    with TestClient(create_app(Settings(dataset_storage_root=str(tmp_path / "data")))) as client:
        health = client.get("/api/v1/health")
        index = client.get("/")
        asset = client.get("/assets/app.js")
        spa = client.get("/governance/review")
        missing_api = client.get("/api/not-a-route")

    assert health.status_code == 200
    assert health.json()["service"] == "socialgraph-fm-api"
    assert index.text == "<html>SocialGraph-FM</html>"
    assert asset.text == "console.log('ready')"
    assert spa.text == index.text
    assert missing_api.status_code == 404
    assert missing_api.json() == {"detail": {"code": "API_ROUTE_NOT_FOUND"}}


def test_invalid_prebuilt_web_root_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv("SOCIALGRAPH_WEB_CLIENT_ROOT", str(missing))
    with pytest.raises(RuntimeError, match="SOCIALGRAPH_WEB_CLIENT_ROOT"):
        create_app(Settings(dataset_storage_root=str(tmp_path / "data")))


def test_prebuilt_web_root_requires_regular_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    web_root = tmp_path / "client"
    web_root.mkdir()
    monkeypatch.setenv("SOCIALGRAPH_WEB_CLIENT_ROOT", str(web_root))
    with pytest.raises(RuntimeError, match="regular index.html"):
        create_app(Settings(dataset_storage_root=str(tmp_path / "data")))
