from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from socialgraph_gfm.governance.skills import PUBLIC_SKILLS
from socialgraph_gfm.core.skills import _SKILL_NAMES


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_product_skill_namespaces_and_human_skill_docs_are_separate() -> None:
    repository = Path(__file__).resolve().parents[3]
    assistant = _object(repository / "skills" / "assistant" / "catalog.json")
    governance = _object(repository / "skills" / "governance" / "catalog.json")
    core = _object(repository / "skills" / "core" / "catalog.json")

    assert assistant["namespace"] == "socialgraph-fm.product-skills.assistant"
    assert governance["namespace"] == "socialgraph-fm.product-skills.governance"
    assert core["namespace"] == "socialgraph-fm.product-skills.core"
    assert len({assistant["namespace"], governance["namespace"], core["namespace"]}) == 3
    assistant_names = tuple(item["name"] for item in assistant["items"])
    assert assistant_names == (
        "answer_governance_question",
        "summarize_node_evidence",
        "generate_global_situation_report",
        "generate_account_evidence_report",
        "generate_coordination_report",
        "generate_case_review_draft",
    )
    assert tuple(item["name"] for item in governance["items"]) == PUBLIC_SKILLS
    assert tuple(item["name"] for item in core["items"]) == _SKILL_NAMES
    skill_docs = {
        path.relative_to(repository / "skills").as_posix()
        for path in (repository / "skills").rglob("SKILL.md")
    }
    assert skill_docs == {
        *(f"assistant/{name}/SKILL.md" for name in assistant_names),
        *(f"governance/{name}/SKILL.md" for name in PUBLIC_SKILLS),
    }
