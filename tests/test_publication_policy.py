from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUICKSTART_COMMANDS = [
    "python scripts/socialgraph.py onboard",
    "python scripts/socialgraph.py start --llm-mode required",
    "python scripts/socialgraph.py stop",
]


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


def _section(content: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(.*?)(?=^## |\Z)",
        content,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing Markdown section: {heading}"
    return match.group(1)


def _quickstart_commands(content: str, heading: str) -> list[str]:
    return re.findall(
        r"^python scripts/socialgraph\.py (?:onboard|start --llm-mode required|stop)$",
        _section(content, heading),
        flags=re.MULTILINE,
    )


def _skill_rows(content: str, heading: str) -> list[str]:
    return re.findall(
        r"^\| `([a-z_]+)` \|",
        _section(content, heading),
        flags=re.MULTILINE,
    )


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


def test_public_readmes_are_bilingual_linked_and_share_the_quickstart() -> None:
    english = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (PROJECT_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    reference = (PROJECT_ROOT / "docs" / "REFERENCE.md").read_text(encoding="utf-8")

    assert "[简体中文](README.zh-CN.md)" in english
    assert "[English](README.md)" in chinese
    assert re.search(r"[\u4e00-\u9fff]", chinese)

    assert _quickstart_commands(english, "Quick start") == QUICKSTART_COMMANDS
    assert _quickstart_commands(chinese, "三步启动") == QUICKSTART_COMMANDS
    assert _quickstart_commands(reference, "Onboarding and lifecycle") == QUICKSTART_COMMANDS

    for heading in (
        "Architecture",
        "Requirements and support",
        "Model API configuration",
        "Complete user workflows",
        "Skills",
        "Repository layout",
        "Governance and responsibility boundary",
    ):
        _section(english, heading)
    for heading in (
        "系统架构",
        "环境要求与支持矩阵",
        "模型 API 配置",
        "完整用户功能",
        "Skills",
        "仓库目录",
        "治理与责任边界",
    ):
        _section(chinese, heading)


def test_skill_tables_follow_each_canonical_catalog_order() -> None:
    governance = json.loads(
        (PROJECT_ROOT / "skills" / "governance" / "catalog.json").read_text(
            encoding="utf-8"
        )
    )
    core = json.loads(
        (PROJECT_ROOT / "skills" / "core" / "catalog.json").read_text(encoding="utf-8")
    )
    governance_names = [skill["name"] for skill in governance["items"]]
    core_names = [skill["name"] for skill in core["items"]]

    english_root = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    chinese_root = (PROJECT_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    english_skills = (PROJECT_ROOT / "skills" / "README.md").read_text(encoding="utf-8")
    chinese_skills = (PROJECT_ROOT / "skills" / "README.zh-CN.md").read_text(
        encoding="utf-8"
    )

    assert _skill_rows(english_root, "Skills") == governance_names
    assert _skill_rows(chinese_root, "Skills") == governance_names
    assert _skill_rows(english_skills, "Governance catalog") == governance_names
    assert _skill_rows(chinese_skills, "Governance 正式目录") == governance_names
    assert _skill_rows(english_skills, "Experimental Core catalog") == core_names
    assert _skill_rows(chinese_skills, "实验 Core 目录") == core_names

    predecessor_name = "run_" + "io" + "hunter"
    assert sum(document.count(predecessor_name) for document in (english_skills, chinese_skills)) == 1
    assert "no compatibility alias is exposed" in english_skills


def test_public_docs_skills_and_license_inventory_is_minimal_and_complete() -> None:
    markdown_docs = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "docs").rglob("*.md")
    )
    assert markdown_docs == ["docs/REFERENCE.md"]
    assert (PROJECT_ROOT / "docs" / "status" / "readiness.json").is_file()

    skill_readmes = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "skills").glob("README*.md")
    )
    assert skill_readmes == ["skills/README.md", "skills/README.zh-CN.md"]
    assert sorted(path.name for path in PROJECT_ROOT.glob("README*.md")) == [
        "README.md",
        "README.zh-CN.md",
    ]

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
    for required in ("NOTICE", "THIRD_PARTY_NOTICES.md", "CITATION.cff"):
        assert (PROJECT_ROOT / required).is_file()
    for removed in ("CONTRIBUTING.md", "SECURITY.md"):
        assert not (PROJECT_ROOT / removed).exists()

    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert not re.search(r"(?m)^version:\s*", citation)


def test_public_docs_explain_core_readiness_and_do_not_link_removed_policies() -> None:
    paths = (
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "README.zh-CN.md",
        PROJECT_ROOT / "docs" / "REFERENCE.md",
        PROJECT_ROOT / "skills" / "README.md",
        PROJECT_ROOT / "skills" / "README.zh-CN.md",
    )
    contents = [path.read_text(encoding="utf-8") for path in paths]
    assert all("docs/status/readiness.json" in content for content in contents[:2])
    assert "experimental Core" in contents[0]
    assert "does not mean" in contents[2]

    removed_names = ("CONTRIBUTING" + ".md", "SECURITY" + ".md")
    assert all(name not in content for name in removed_names for content in contents)
