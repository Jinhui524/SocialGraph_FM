from __future__ import annotations

import json
import shutil
import sqlite3
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


def test_brand_scan_checks_untracked_publication_candidates(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    predecessor = "io" + "hunter"
    (repository / "candidate.md").write_text(
        f"Untracked predecessor: {predecessor}\n", encoding="utf-8"
    )

    result = _scan(repository)

    assert result.returncode == 1
    assert "candidate.md" in result.stdout


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


def test_brand_scan_requires_knowledge_chunks_to_match_tracked_sources(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    predecessor = "io" + "hunter"
    paths = (
        "README.md",
        "docs/REFERENCE.md",
        "bundles/models/socialgraph-global/exports/socialgraph-global/model-card.json",
        "bundles/governance/knowledge/knowledge.sqlite3",
    )
    for relative in paths:
        source = ROOT / relative
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    database = repository / "bundles" / "governance" / "knowledge" / "knowledge.sqlite3"

    allowed = _scan(repository)
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr

    connection = sqlite3.connect(database)
    try:
        connection.execute(
            f"ALTER TABLE documents ADD COLUMN legacy_brand TEXT DEFAULT '{predecessor}'"
        )
        connection.commit()
    finally:
        connection.close()

    schema_rejected = _scan(repository)
    assert schema_rejected.returncode == 1
    assert "schema inventory is not canonical" in schema_rejected.stdout

    shutil.copy2(ROOT / "bundles/governance/knowledge/knowledge.sqlite3", database)
    connection = sqlite3.connect(database)
    try:
        record = json.loads(
            connection.execute(
                "SELECT record_json FROM chunks WHERE source_label = 'project-readme'"
            ).fetchone()[0]
        )
        record["text"] = f"Untracked predecessor: {predecessor}"
        connection.execute(
            "UPDATE chunks SET record_json = ? WHERE source_label = 'project-readme'",
            (json.dumps(record, sort_keys=True, separators=(",", ":")),),
        )
        connection.commit()
    finally:
        connection.close()

    rejected = _scan(repository)
    assert rejected.returncode == 1
    assert "chunk records are not canonical" in rejected.stdout


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


def test_brand_scan_allows_only_the_exact_reviewed_skill_migration_note(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    skill_name = "run_" + "io" + "hunter"
    readme = repository / "skills" / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text(
        "Migration note: the private predecessor capability formerly named "
        f"`{skill_name}` maps to the sole public canonical name "
        "`run_governance_analysis`; no compatibility alias is exposed.\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)

    allowed = _scan(repository)
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr

    readme.write_text(
        readme.read_text(encoding="utf-8") + f"Unreviewed alias: {skill_name}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)

    rejected = _scan(repository)
    assert rejected.returncode == 1
    assert "skills/README.md" in rejected.stdout
