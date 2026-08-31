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
    "python scripts/socialgraph.py start",
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
        r"^python scripts/socialgraph\.py (?:onboard|start|stop)$",
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


def test_public_readme_is_chinese_and_keeps_the_three_command_quickstart() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    reference = (PROJECT_ROOT / "docs" / "REFERENCE.md").read_text(encoding="utf-8")

    assert re.search(r"[\u4e00-\u9fff]", readme)
    assert "README.zh-CN.md" not in readme

    assert _quickstart_commands(readme, "三步启动") == QUICKSTART_COMMANDS
    assert _quickstart_commands(reference, "Onboarding 与生命周期") == QUICKSTART_COMMANDS

    for heading in (
        "主要能力",
        "环境要求",
        "配置大模型",
        "研判 Assistant Skills",
        "Governance Skills",
        "项目结构",
        "使用边界",
    ):
        _section(readme, heading)


def test_skill_tables_follow_each_canonical_catalog_order() -> None:
    assistant = json.loads(
        (PROJECT_ROOT / "skills" / "assistant" / "catalog.json").read_text(
            encoding="utf-8"
        )
    )
    governance = json.loads(
        (PROJECT_ROOT / "skills" / "governance" / "catalog.json").read_text(
            encoding="utf-8"
        )
    )
    core = json.loads(
        (PROJECT_ROOT / "skills" / "core" / "catalog.json").read_text(encoding="utf-8")
    )
    assistant_names = [skill["name"] for skill in assistant["items"]]
    governance_names = [skill["name"] for skill in governance["items"]]
    core_names = [skill["name"] for skill in core["items"]]

    root = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    skills = (PROJECT_ROOT / "skills" / "README.md").read_text(encoding="utf-8")

    assert _skill_rows(root, "研判 Assistant Skills") == assistant_names
    assert _skill_rows(root, "Governance Skills") == governance_names
    assert _skill_rows(skills, "Assistant Skills 与界面对应") == assistant_names
    assert _skill_rows(skills, "Governance Skills") == governance_names
    assert _skill_rows(skills, "实验 Core Skills") == core_names

    for name in assistant_names:
        assert (PROJECT_ROOT / "skills" / "assistant" / name / "SKILL.md").is_file()
    for name in governance_names:
        assert (PROJECT_ROOT / "skills" / "governance" / name / "SKILL.md").is_file()

    predecessor_name = "run_" + "io" + "hunter"
    assert skills.count(predecessor_name) == 1
    assert "no compatibility alias is exposed" in skills


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
    assert skill_readmes == ["skills/README.md"]
    assert sorted(path.name for path in PROJECT_ROOT.glob("README*.md")) == ["README.md"]

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
        PROJECT_ROOT / "docs" / "REFERENCE.md",
        PROJECT_ROOT / "skills" / "README.md",
    )
    contents = [path.read_text(encoding="utf-8") for path in paths]
    assert "docs/status/readiness.json" in contents[0]
    assert "实验 Core" in contents[0]
    assert "不影响" in contents[1]

    removed_names = ("CONTRIBUTING" + ".md", "SECURITY" + ".md")
    assert all(name not in content for name in removed_names for content in contents)
