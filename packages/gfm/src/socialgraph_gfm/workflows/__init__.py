"""Behavior-preserving implementation boundary for staged GFM workflows.

The concrete implementations live in the sibling stage modules.  This package
assembles their globals once so cross-stage calls retain the historical shared
module namespace, including callers that monkeypatch private helpers through
``socialgraph_gfm.gfm_workflow``.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from . import _shared
from . import adaptation as _adaptation
from . import corpus as _corpus
from . import embedding as _embedding
from . import evaluation as _evaluation
from . import export_promotion as _export_promotion
from . import lodo as _lodo
from . import pretrain as _pretrain
from . import product as _product
from . import sampling as _sampling

_IMPLEMENTATION_MODULES = (
    _corpus,
    _embedding,
    _sampling,
    _pretrain,
    _product,
    _lodo,
    _adaptation,
    _evaluation,
    _export_promotion,
)


def _install_shared_namespace() -> None:
    namespace = {name: getattr(_shared, name) for name in _shared.__all__}
    for module in _IMPLEMENTATION_MODULES:
        namespace.update(
            (name, value) for name, value in vars(module).items() if not name.startswith("__")
        )
    globals().update(namespace)
    for module in _IMPLEMENTATION_MODULES:
        module.__dict__.update(namespace)


_install_shared_namespace()

# Keep the supported import surface visible to static analyzers.  Runtime
# patching still flows through the compatibility module class below.
adapt_gfm = _adaptation.adapt_gfm
check_gfm_task_assets = _corpus.check_gfm_task_assets
embed_gfm_text = _embedding.embed_gfm_text
evaluate_gfm = _evaluation.evaluate_gfm
evaluate_gfm_checkpoint_test_once = _pretrain.evaluate_gfm_checkpoint_test_once
export_gfm = _export_promotion.export_gfm
fetch_gfm_openalex = _corpus.fetch_gfm_openalex
fetch_gfm_thgl_software = _corpus.fetch_gfm_thgl_software
fetch_gfm_wikimedia_talk = _corpus.fetch_gfm_wikimedia_talk
prepare_gfm_corpus = _corpus.prepare_gfm_corpus
pretrain_gfm = _pretrain.pretrain_gfm
resume_gfm = _adaptation.resume_gfm
validate_gfm = _evaluation.validate_gfm
verify_gfm_checkpoint_fresh = _pretrain.verify_gfm_checkpoint_fresh
verify_gfm_product_checkpoint_fresh = _adaptation.verify_gfm_product_checkpoint_fresh
verify_gfm_suite_checkpoint_fresh = _adaptation.verify_gfm_suite_checkpoint_fresh
_adapt_worker = _adaptation._adapt_worker
_lodo_worker = _lodo._lodo_worker
_pretrain_worker = _pretrain._pretrain_worker


class _WorkflowCompatibilityModule(ModuleType):
    """Forward compatibility-module patches into every implementation stage."""

    def __setattr__(self, name: str, value: Any) -> None:
        ModuleType.__setattr__(self, name, value)
        if name.startswith("__"):
            return
        for module in self.__dict__.get("_IMPLEMENTATION_MODULES", ()):
            if name in module.__dict__:
                ModuleType.__setattr__(module, name, value)

    def __delattr__(self, name: str) -> None:
        ModuleType.__delattr__(self, name)
        if name.startswith("__"):
            return
        for module in self.__dict__.get("_IMPLEMENTATION_MODULES", ()):
            if name in module.__dict__:
                ModuleType.__delattr__(module, name)


sys.modules[__name__].__class__ = _WorkflowCompatibilityModule

__all__ = [
    "adapt_gfm",
    "check_gfm_task_assets",
    "embed_gfm_text",
    "evaluate_gfm",
    "evaluate_gfm_checkpoint_test_once",
    "export_gfm",
    "fetch_gfm_openalex",
    "fetch_gfm_thgl_software",
    "fetch_gfm_wikimedia_talk",
    "prepare_gfm_corpus",
    "pretrain_gfm",
    "resume_gfm",
    "validate_gfm",
    "verify_gfm_checkpoint_fresh",
    "verify_gfm_product_checkpoint_fresh",
    "verify_gfm_suite_checkpoint_fresh",
]
