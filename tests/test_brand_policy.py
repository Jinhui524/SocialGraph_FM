from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "scripts" / "brand-scan.py"


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "brand fixture"
    (repository / "scripts").mkdir(parents=True)
    shutil.copy2(SCANNER, repository / "scripts" / SCANNER.name)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    return repository


def _scan(repository: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(repository / "scripts" / SCANNER.name),
            "--repository-root",
            str(repository),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )


def test_brand_scan_rejects_predecessor_names_in_paths_and_text(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    predecessor = "io" + "hunter"
    path = repository / f"legacy-{predecessor}.txt"
    path.write_text(f"old product={predecessor}\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)

    result = _scan(repository)

    assert result.returncode == 1
    assert "Brand scan rejected" in result.stdout


def test_brand_scan_checks_nested_archive_metadata(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    archive_path = repository / "fixture.zip"
    predecessor = "static" + "-v2"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps({"product": predecessor}, sort_keys=True),
        )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)

    result = _scan(repository)

    assert result.returncode == 1
    assert "fixture.zip -> manifest.json" in result.stdout


def test_brand_scan_allows_required_legal_and_model_provenance(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    predecessor = "io" + "hunter"
    notices = repository / "THIRD_PARTY_NOTICES.md"
    notices.write_text(f"Original source: {predecessor}\n", encoding="utf-8")
    card = (
        repository
        / "bundles"
        / "models"
        / "socialgraph-global"
        / "exports"
        / "socialgraph-global"
        / "model-card.json"
    )
    card.parent.mkdir(parents=True)
    card.write_text(
        json.dumps(
            {
                "licenses": [{"name": predecessor}],
                "sourceAttribution": {"source": predecessor},
                "limitations": ["generic limitation"],
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)

    result = _scan(repository)

    assert result.returncode == 0, result.stdout + result.stderr


def test_brand_scan_rejects_predecessor_name_outside_model_provenance(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    predecessor = "io" + "hunter"
    card = (
        repository
        / "bundles"
        / "models"
        / "socialgraph-global"
        / "exports"
        / "socialgraph-global"
        / "model-card.json"
    )
    card.parent.mkdir(parents=True)
    card.write_text(
        json.dumps(
            {
                "licenses": [{"name": predecessor}],
                "sourceAttribution": {"source": predecessor},
                "limitations": [f"copied from {predecessor}"],
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)

    result = _scan(repository)

    assert result.returncode == 1
    assert "model-card.json" in result.stdout
