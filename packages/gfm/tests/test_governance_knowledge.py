from __future__ import annotations

import json
from pathlib import Path

import pytest

from socialgraph_gfm.governance.cli import main
from socialgraph_gfm.governance.knowledge import (
    KnowledgeIndex,
    KnowledgeSource,
    build_knowledge_index,
)


def _sources(tmp_path: Path) -> tuple[KnowledgeSource, ...]:
    overview = tmp_path / "overview.md"
    overview.write_text(
        "# Governance\n\nCoordination evidence requires analyst review.\n",
        encoding="utf-8",
    )
    model = tmp_path / "model-card.json"
    model.write_text(
        json.dumps({"threshold": 0.64, "model": "global"}, indent=2),
        encoding="utf-8",
    )
    return (
        KnowledgeSource("model-card", model, "repo://governance/model-card"),
        KnowledgeSource("overview", overview, "repo://project/overview"),
    )


def test_trusted_knowledge_build_is_deterministic_searchable_and_path_private(
    tmp_path: Path,
) -> None:
    first_manifest = build_knowledge_index(tmp_path / "first", _sources(tmp_path))
    second_manifest = build_knowledge_index(tmp_path / "second", _sources(tmp_path))
    first = KnowledgeIndex(first_manifest.parent)
    second = KnowledgeIndex(second_manifest.parent)
    assert first.verify() == second.verify()
    results = first.search("coordination analyst", limit=5)
    assert results
    assert results[0].source_label == "overview"
    assert results[0].source_uri == "repo://project/overview"
    assert str(tmp_path) not in json.dumps([item.__dict__ for item in results])
    assert len(results[0].content_hash) == 64
    assert len(results[0].chunk_hash) == 64


def test_knowledge_tampering_fails_closed(tmp_path: Path) -> None:
    manifest = build_knowledge_index(tmp_path / "index", _sources(tmp_path))
    database = manifest.parent / "knowledge.sqlite3"
    raw = bytearray(database.read_bytes())
    raw[-1] ^= 1
    database.write_bytes(raw)
    with pytest.raises(ValueError, match="SQLite identity"):
        KnowledgeIndex(manifest.parent).search("coordination")


@pytest.mark.parametrize("name,content", [("paper.pdf", b"%PDF-1.7"), ("raw.txt", b"a\x00b")])
def test_knowledge_rejects_pdf_and_binary_sources(
    tmp_path: Path, name: str, content: bytes
) -> None:
    source = tmp_path / name
    source.write_bytes(content)
    with pytest.raises(ValueError):
        build_knowledge_index(
            tmp_path / "index",
            (KnowledgeSource("unsafe", source, "repo://unsafe"),),
        )


def test_knowledge_rejects_directories_and_links(tmp_path: Path) -> None:
    directory = tmp_path / "docs"
    directory.mkdir()
    with pytest.raises(ValueError):
        build_knowledge_index(
            tmp_path / "directory-index",
            (KnowledgeSource("docs", directory, "repo://docs"),),
        )
    target = tmp_path / "target.md"
    target.write_text("trusted", encoding="utf-8")
    link = tmp_path / "linked.md"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable in this test environment")
    with pytest.raises(ValueError):
        build_knowledge_index(
            tmp_path / "link-index",
            (KnowledgeSource("linked", link, "repo://linked"),),
        )


def test_cli_imports_only_explicit_labeled_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "guide.txt"
    source.write_text("Governance reviewed evidence guide", encoding="utf-8")
    root = tmp_path / "runtime"
    assert (
        main(
            [
                "knowledge-import",
                "--root",
                str(root),
                "--source",
                f"guide={source}",
                "--source-uri",
                "guide=repo://guide",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out.strip()
    assert output == KnowledgeIndex(root / "knowledge").verify()
    assert str(source) not in output
