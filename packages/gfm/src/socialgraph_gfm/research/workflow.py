"""Compatibility exports and monkeypatch routing for the split SocialGraph-FM Research workflow."""

# ruff: noqa: F401, I001 - TYPE_CHECKING imports document the dynamic compatibility surface.
from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

from .workflows import common, evaluate, materialize, publish, runtime, serve, train
if TYPE_CHECKING:
    from .workflows import (
        candidate_grouped_signed_split, evaluate_research_model, export_research_model,
        load_comparison_manifest, load_corpus_manifest, load_export_manifest, load_registry,
        materialize_fixture_corpus, materialize_research_corpus, publish_research_model,
        readiness, research_root_from_home, smoke_research_export, stage_paths,
        train_research_comparison_matrix, train_research_model,
    )
    from .workflows.common import (
        CORPUS_SCHEMA, EVALUATION_SCHEMA, EXPECTED_SOURCE_HASHES, EXPORT_SCHEMA,
        REGISTRY_SCHEMA, SMOKE_SCHEMA, TRAINING_SCHEMA, _atomic_json,
        _read_hashed_document, _safe_root,
    )
    from .workflows.serve import _load_exported_runtime
    from .workflows.train import COMPARISON_CHECKPOINT_SCHEMA, COMPARISON_SCHEMA
    from .workflows.train import _bundle_edge_index, _tensor_state_hash
_IMPLEMENTATION_MODULES = (common, materialize, runtime, train, evaluate, serve, publish)
_OWNERS: dict[str, Any] = {}
for _module in _IMPLEMENTATION_MODULES:
    for _name in _module.COMPAT_EXPORTS:
        globals()[_name] = getattr(_module, _name)
        _OWNERS[_name] = _module
for _name in tuple(_OWNERS):
    _value = globals()[_name]
    _OWNERS[_name] = tuple(m for m in _IMPLEMENTATION_MODULES if getattr(m, _name, None) is _value)


class _CompatibilityModule(ModuleType):
    """Mirror compatibility-module patches to the extracted implementation."""
    def __setattr__(self, name: str, value: Any) -> None:
        for owner in _OWNERS.get(name, ()):
            if getattr(owner, name) is not value:
                setattr(owner, name, value)
        super().__setattr__(name, value)


_PUBLIC_EXPORTS = (
    "COMPARISON_CHECKPOINT_SCHEMA", "COMPARISON_SCHEMA", "CORPUS_SCHEMA",
    "EVALUATION_SCHEMA", "EXPECTED_SOURCE_HASHES", "EXPORT_SCHEMA", "REGISTRY_SCHEMA",
    "SMOKE_SCHEMA", "TRAINING_SCHEMA", "candidate_grouped_signed_split",
    "evaluate_research_model", "export_research_model", "load_comparison_manifest",
    "load_corpus_manifest", "load_export_manifest", "load_registry",
    "materialize_fixture_corpus", "materialize_research_corpus", "publish_research_model",
    "readiness", "research_root_from_home", "smoke_research_export", "stage_paths",
    "train_research_comparison_matrix", "train_research_model",
)
__all__ = list(_PUBLIC_EXPORTS)

sys.modules[__name__].__class__ = _CompatibilityModule
