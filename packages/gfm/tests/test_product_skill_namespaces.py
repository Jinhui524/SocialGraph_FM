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


def test_governance_and_core_product_skills_have_separate_namespaces() -> None:
    repository = Path(__file__).resolve().parents[3]
    governance = _object(repository / "skills" / "governance" / "catalog.json")
    core = _object(repository / "skills" / "core" / "catalog.json")

    assert governance["namespace"] == "socialgraph-fm.product-skills.governance"
    assert core["namespace"] == "socialgraph-fm.product-skills.core"
    assert governance["namespace"] != core["namespace"]
    assert tuple(item["name"] for item in governance["items"]) == PUBLIC_SKILLS
    assert tuple(item["name"] for item in core["items"]) == _SKILL_NAMES
    assert not any((repository / "skills").rglob("SKILL.md"))
