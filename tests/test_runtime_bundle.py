from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "bundles" / "runtime-manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_runtime_bundle_inventory_is_exact_and_hash_bound() -> None:
    manifest = _document(MANIFEST)
    assert manifest["schemaVersion"] == "socialgraph-fm.runtime-bundle/1.0"
    assets = manifest["assets"]
    assert isinstance(assets, list)
    assert len(assets) == manifest["fileCount"]
    assert manifest["contentRoots"] == [
        "bundles/models/socialgraph-global",
        "bundles/governance",
        "bundles/web",
        "examples/governance",
    ]
    paths = [entry["path"] for entry in assets]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))

    total = 0
    inventory_lines: list[str] = []
    for entry in assets:
        relative = entry["path"]
        assert isinstance(relative, str)
        pure = PurePosixPath(relative)
        assert not pure.is_absolute() and ".." not in pure.parts and "." not in pure.parts
        path = ROOT / pure
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == entry["bytes"]
        assert _sha256(path) == entry["sha256"]
        total += path.stat().st_size
        inventory_lines.append(
            f'{relative}\t{entry["bytes"]}\t{entry["sha256"]}\t{entry["role"]}\n'
        )
    assert total == manifest["totalBytes"]
    assert hashlib.sha256("".join(inventory_lines).encode()).hexdigest() == manifest["inventoryHash"]

    actual = sorted(
        path.relative_to(ROOT).as_posix()
        for content_root in manifest["contentRoots"]
        for path in (ROOT / content_root).rglob("*")
        if path.is_file()
    )
    assert actual == paths
    assert {entry["path"] for entry in assets if entry["role"] == "web"} == {
        "bundles/web/client.zip",
        "bundles/web/manifest.json",
    }


def test_model_bundle_is_the_deployable_export_plus_russia_serving_corpus_only() -> None:
    model_root = ROOT / "bundles" / "models" / "socialgraph-global"
    export_root = model_root / "exports" / "socialgraph-global"
    export = _document(export_root / "export-manifest.json")
    manifest = _document(MANIFEST)
    for field in (
        "releaseId",
        "modelVersionId",
        "modelVersionHash",
        "artifactHash",
        "corpusHash",
    ):
        assert manifest["model"][field] == export[field]
    artifact_paths = set(export["artifacts"])
    actual_export = {
        path.relative_to(export_root).as_posix()
        for path in export_root.rglob("*")
        if path.is_file() and path.name != "export-manifest.json"
    }
    assert actual_export == artifact_paths
    assert sum(path.startswith("checkpoints/") and path.endswith(".pt") for path in artifact_paths) == 4
    assert not {"smoke-report.json", "registry.json", "registry-candidate.json"} & actual_export
    assert {path.name for path in (model_root / "corpus" / "countries").iterdir()} == {"russia"}
    assert not any("research" in path.parts or "runs" in path.parts for path in model_root.rglob("*"))


def test_russia_and_target_domain_examples_keep_their_frozen_catalog_bindings() -> None:
    russia = ROOT / "examples" / "governance" / "russia"
    answer_catalog = _document(russia / "catalog.json")
    assert _sha256(russia / "russia-full.zip") == answer_catalog["sourceSha256"]
    assert {entry["fileName"] for entry in answer_catalog["packs"]} == {
        "russia-01.zip",
        "russia-02.zip",
        "russia-03.zip",
        "russia-04.zip",
    }

    target = ROOT / "examples" / "governance" / "target-domain"
    target_catalog = _document(target / "governance-target-tasks.catalog.json")
    for entry in target_catalog["targets"]:
        path = target / entry["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == entry["bytes"]
        assert _sha256(path) == entry["sha256"]

    for archive in (*russia.glob("*.zip"), *target.rglob("*.zip")):
        with zipfile.ZipFile(archive) as opened:
            for member in opened.namelist():
                pure = PurePosixPath(member)
                assert not pure.is_absolute() and ".." not in pure.parts and "\\" not in member
