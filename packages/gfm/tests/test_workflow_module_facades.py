from __future__ import annotations

from pathlib import Path

import socialgraph_gfm._gfm_workflow_legacy as gfm_legacy
import socialgraph_gfm.gfm_workflow as gfm_compat
from socialgraph_gfm import workflows as gfm
from socialgraph_gfm.research import _workflow_legacy as research_legacy
from socialgraph_gfm.research import workflow as research_compat
from socialgraph_gfm.research import workflows as research


def test_gfm_workflow_facades_preserve_compatibility_module_identity() -> None:
    assert gfm_compat is gfm_legacy
    assert gfm.prepare_gfm_corpus is gfm_legacy.prepare_gfm_corpus
    assert gfm.embed_gfm_text is gfm_legacy.embed_gfm_text
    assert gfm.pretrain_gfm is gfm_legacy.pretrain_gfm
    assert gfm.adapt_gfm is gfm_legacy.adapt_gfm
    assert gfm.evaluate_gfm is gfm_legacy.evaluate_gfm
    assert gfm.export_gfm is gfm_legacy.export_gfm


def test_research_workflow_facades_preserve_compatibility_module_identity() -> None:
    assert research_compat is research_legacy
    assert research.materialize_research_corpus is research_legacy.materialize_research_corpus
    assert research.train_research_model is research_legacy.train_research_model
    assert research.evaluate_research_model is research_legacy.evaluate_research_model
    assert research.export_research_model is research_legacy.export_research_model
    assert research.publish_research_model is research_legacy.publish_research_model


def test_public_compatibility_files_are_thin_and_boundaries_are_documented() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "socialgraph_gfm"
    assert len((source / "gfm_workflow.py").read_text(encoding="utf-8").splitlines()) < 60
    assert (source / "workflows" / "__init__.py").is_file()
    assert not (source / "workflows" / "README.md").exists()
    research_root = source / "research"
    assert len((research_root / "workflow.py").read_text(encoding="utf-8").splitlines()) < 60
    assert (research_root / "workflows" / "__init__.py").is_file()
    assert not (research_root / "workflows" / "README.md").exists()


def test_workflow_implementations_are_physically_split() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "socialgraph_gfm"
    roots = (
        source / "workflows",
        source / "research" / "workflows",
    )
    for workflow_root in roots:
        modules = sorted(workflow_root.glob("*.py"))
        assert len(modules) >= 6
        implementation_modules = 0
        for module in modules:
            text = module.read_text(encoding="utf-8")
            assert len(text.splitlines()) < 2_500, module
            assert "from .._workflow_legacy" not in text
            assert "from ..._workflow_legacy" not in text
            implementation_modules += int("def " in text or "class " in text)
        assert implementation_modules >= 5

    assert len((source / "_gfm_workflow_legacy.py").read_text(encoding="utf-8").splitlines()) < 60
    research = source / "research"
    assert len((research / "_workflow_legacy.py").read_text(encoding="utf-8").splitlines()) < 60
