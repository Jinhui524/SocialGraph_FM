from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build-web-bundle.py"
DIST = ROOT / "apps" / "web" / "dist" / "client"


def _module():
    spec = importlib.util.spec_from_file_location("build_web_bundle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not (DIST / "index.html").is_file(), reason="Web dist is not built")
def test_prebuilt_web_bundle_is_deterministic_and_manifest_bound() -> None:
    module = _module()
    first_archive, first_manifest = module.build_bundle(ROOT)
    second_archive, second_manifest = module.build_bundle(ROOT)
    assert first_archive == second_archive
    assert first_manifest == second_manifest

    manifest = json.loads(first_manifest)
    assert manifest["schemaVersion"] == "socialgraph-fm.web-bundle/1.0"
    assert manifest["archive"] == {
        "path": "bundles/web/client.zip",
        "bytes": len(first_archive),
        "sha256": hashlib.sha256(first_archive).hexdigest(),
    }
    assert manifest["fileCount"] == len(manifest["files"])
    assert manifest["sourceFileCount"] > 0

    with zipfile.ZipFile(io.BytesIO(first_archive)) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert "index.html" in archive.namelist()
        assert all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist())
        for item in manifest["files"]:
            payload = archive.read(item["path"])
            assert len(payload) == item["bytes"]
            assert hashlib.sha256(payload).hexdigest() == item["sha256"]


@pytest.mark.skipif(not (DIST / "index.html").is_file(), reason="Web dist is not built")
def test_tracked_web_bundle_matches_current_build() -> None:
    module = _module()
    archive, manifest = module.build_bundle(ROOT)
    assert (ROOT / "bundles" / "web" / "client.zip").read_bytes() == archive
    assert (ROOT / "bundles" / "web" / "manifest.json").read_bytes() == manifest


def test_tracked_web_bundle_is_self_consistent_and_source_bound() -> None:
    module = _module()
    archive = (ROOT / "bundles" / "web" / "client.zip").read_bytes()
    manifest = json.loads((ROOT / "bundles" / "web" / "manifest.json").read_bytes())
    assert manifest["schemaVersion"] == module.SCHEMA_VERSION
    assert manifest["archive"]["bytes"] == len(archive)
    assert manifest["archive"]["sha256"] == hashlib.sha256(archive).hexdigest()
    source_inventory = module._source_inventory(ROOT / "apps" / "web")
    assert manifest["sourceFileCount"] == len(source_inventory)
    assert manifest["sourceHash"] == hashlib.sha256(
        module._canonical_json(source_inventory)
    ).hexdigest()

    with zipfile.ZipFile(io.BytesIO(archive)) as opened:
        assert opened.namelist() == [item["path"] for item in manifest["files"]]
        assert len(opened.namelist()) == manifest["fileCount"]
        for item in manifest["files"]:
            payload = opened.read(item["path"])
            assert len(payload) == item["bytes"]
            assert hashlib.sha256(payload).hexdigest() == item["sha256"]


def test_source_inventory_is_independent_of_text_line_endings(tmp_path: Path) -> None:
    module = _module()
    for name in module.SOURCE_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"first\nsecond\n")
    for name in module.SOURCE_DIRECTORIES:
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    text_source = tmp_path / "src" / "example.css"
    binary_source = tmp_path / "public" / "example.bin"
    text_source.write_bytes(b"alpha\nbeta\n")
    binary_source.write_bytes(b"\x00\r\n\xff")

    lf_inventory = module._source_inventory(tmp_path)
    for candidate in (*module.SOURCE_FILES, "src/example.css"):
        path = tmp_path / candidate
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    assert module._source_inventory(tmp_path) == lf_inventory
