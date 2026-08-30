"""Stage-oriented public entry points for SocialGraph-FM Research."""

from .evaluate import evaluate_research_model
from .materialize import (
    candidate_grouped_signed_split,
    load_corpus_manifest,
    materialize_fixture_corpus,
    materialize_research_corpus,
    research_root_from_home,
)
from .publish import load_registry, publish_research_model, readiness, stage_paths
from .serve import export_research_model, load_export_manifest, smoke_research_export
from .train import (
    load_comparison_manifest,
    train_research_comparison_matrix,
    train_research_model,
)

__all__ = [
    "candidate_grouped_signed_split",
    "evaluate_research_model",
    "export_research_model",
    "load_comparison_manifest",
    "load_corpus_manifest",
    "load_export_manifest",
    "load_registry",
    "materialize_fixture_corpus",
    "materialize_research_corpus",
    "publish_research_model",
    "readiness",
    "research_root_from_home",
    "smoke_research_export",
    "stage_paths",
    "train_research_comparison_matrix",
    "train_research_model",
]
