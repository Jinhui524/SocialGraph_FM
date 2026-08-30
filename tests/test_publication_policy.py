from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _powershell() -> str:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is unavailable")
    return executable


def _run_scan(repository: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _powershell(),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(repository / "scripts" / "secret-scan.ps1"),
            "-RepositoryRoot",
            str(repository),
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )


def _fixture_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "public policy repo"
    library = repository / "scripts" / "lib"
    library.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "scripts" / "secret-scan.ps1", repository / "scripts")
    shutil.copy2(PROJECT_ROOT / "scripts" / "lib" / "NativeCommand.ps1", library)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    return repository


def test_secret_scan_does_not_treat_risk_identifier_as_openai_key(tmp_path: Path) -> None:
    repository = _fixture_repository(tmp_path)
    (repository / "fixture.txt").write_text(
        "recipe=russia-risk-skill-answer-pack-v1\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    result = _run_scan(repository)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("env_name", [".env", ".env.local", ".env.production"])
def test_secret_scan_rejects_token_and_tracked_dotenv(tmp_path: Path, env_name: str) -> None:
    repository = _fixture_repository(tmp_path)
    token = "sk-" + "A" * 32
    (repository / "credential.txt").write_text(token, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    token_result = _run_scan(repository)
    assert token_result.returncode != 0
    assert "credential.txt" in token_result.stdout + token_result.stderr

    (repository / "credential.txt").unlink()
    subprocess.run(["git", "add", "-A"], cwd=repository, check=True)
    (repository / env_name).write_text("PLACEHOLDER=true\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", env_name], cwd=repository, check=True)
    env_result = _run_scan(repository)
    assert env_result.returncode != 0
    assert env_name in env_result.stdout + env_result.stderr


def test_public_readme_is_english_and_keeps_the_quickstart() -> None:
    english = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    english_sections = re.findall(r"^## (.+)$", english, flags=re.MULTILINE)
    assert english_sections == [
        "Architecture",
        "Requirements",
        "Quick start",
        "Model API configuration",
        "Included workflows",
        "Governance Skills",
        "Repository layout",
        "GitHub export",
        "License and responsibility",
    ]
    assert not (PROJECT_ROOT / "README.zh-CN.md").exists()
    commands = [
        "python scripts/socialgraph.py onboard",
        "python scripts/socialgraph.py start --llm-mode required",
    ]
    positions = [english.index(command) for command in commands]
    assert positions == sorted(positions)


def test_skill_tables_follow_the_canonical_catalog_order() -> None:
    catalog = json.loads(
        (PROJECT_ROOT / "skills" / "governance" / "catalog.json").read_text(
            encoding="utf-8"
        )
    )
    expected = [skill["name"] for skill in catalog["items"]]
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    observed = re.findall(r"^\| `([a-z_]+)` \|", content, flags=re.MULTILINE)
    assert observed == expected


def test_public_docs_and_license_inventory_is_minimal_and_complete() -> None:
    markdown_docs = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "docs").rglob("*.md")
    )
    assert markdown_docs == ["docs/REFERENCE.md"]
    component_readmes = (
        PROJECT_ROOT / "apps" / "web" / "README.md",
        PROJECT_ROOT / "services" / "api" / "README.md",
        PROJECT_ROOT / "packages" / "gfm" / "README.md",
        PROJECT_ROOT / "packages" / "runtime" / "README.md",
    )
    assert not any(path.exists() for path in component_readmes)
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    for required in (
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CITATION.cff",
    ):
        assert (PROJECT_ROOT / required).is_file()
    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert not re.search(r"(?m)^version:\s*", citation)
