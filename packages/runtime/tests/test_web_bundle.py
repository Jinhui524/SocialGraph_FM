from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from socialgraph_fm_runtime.layout import RuntimeLayout
from socialgraph_fm_runtime.web_bundle import install_web_bundle


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _bundle(project: Path, members: list[tuple[str, bytes]]) -> RuntimeLayout:
    output = io.BytesIO()
    files = []
    with zipfile.ZipFile(output, "w") as archive:
        for name, payload in members:
            archive.writestr(name, payload)
            files.append({"path": name, "bytes": len(payload), "sha256": _sha256(payload)})
    payload = output.getvalue()
    manifest = {
        "schemaVersion": "socialgraph-fm.web-bundle/1.0",
        "archive": {
            "path": "bundles/web/client.zip",
            "bytes": len(payload),
            "sha256": _sha256(payload),
        },
        "fileCount": len(files),
        "totalBytes": sum(item["bytes"] for item in files),
        "inventoryHash": _sha256(_canonical(files)),
        "files": files,
    }
    root = project / "bundles" / "web"
    root.mkdir(parents=True)
    (root / "client.zip").write_bytes(payload)
    (root / "manifest.json").write_bytes(_canonical(manifest))
    return RuntimeLayout(project)


def test_verified_web_bundle_is_installed_atomically(tmp_path: Path) -> None:
    layout = _bundle(
        tmp_path,
        [("index.html", b"<main>ready</main>"), ("assets/app.js", b"ready()")],
    )
    layout.initialize_directories()

    report = install_web_bundle(layout)

    assert report["fileCount"] == 2
    assert (layout.web_client_root / "index.html").read_bytes() == b"<main>ready</main>"
    assert (layout.web_client_root / "assets" / "app.js").read_bytes() == b"ready()"
    assert not list(layout.temp_root.glob("web-*.stage"))


def test_web_archive_tampering_is_rejected_before_replacement(tmp_path: Path) -> None:
    layout = _bundle(tmp_path, [("index.html", b"ready")])
    layout.initialize_directories()
    layout.web_client_root.mkdir(parents=True)
    marker = layout.web_client_root / "marker"
    marker.write_text("old", encoding="utf-8")
    layout.web_bundle_archive.write_bytes(layout.web_bundle_archive.read_bytes() + b"tamper")

    with pytest.raises(RuntimeError, match="integrity"):
        install_web_bundle(layout)
    assert marker.read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize("member", ["../escape.html", "/absolute.html", "C:/drive.html"])
def test_web_bundle_rejects_path_escape(tmp_path: Path, member: str) -> None:
    layout = _bundle(
        tmp_path,
        [("index.html", b"ready"), (member, b"escape")],
    )
    layout.initialize_directories()

    with pytest.raises(RuntimeError, match="unsafe"):
        install_web_bundle(layout)
    assert not (tmp_path / "escape.html").exists()


def test_web_bundle_rejects_case_insensitive_duplicates(tmp_path: Path) -> None:
    layout = _bundle(
        tmp_path,
        [("index.html", b"ready"), ("assets/App.js", b"a"), ("assets/app.js", b"b")],
    )
    layout.initialize_directories()

    with pytest.raises(RuntimeError, match="duplicate"):
        install_web_bundle(layout)
