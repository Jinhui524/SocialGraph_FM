"""Independent SocialGraph-FM Research GFM application package."""

from .compatibility import (
    ResearchStructuralIndex,
    ResearchStructuralRecord,
    UploadCompatibility,
    UploadedGraphDescriptor,
    inspect_uploaded_graph,
)
from .contracts import RESEARCH_TASK_IDS
from .workflow import (
    evaluate_research_model,
    export_research_model,
    materialize_research_corpus,
    publish_research_model,
    smoke_research_export,
    train_research_comparison_matrix,
    train_research_model,
)

__all__ = [
    "RESEARCH_TASK_IDS",
    "ResearchStructuralIndex",
    "ResearchStructuralRecord",
    "UploadCompatibility",
    "UploadedGraphDescriptor",
    "evaluate_research_model",
    "export_research_model",
    "inspect_uploaded_graph",
    "materialize_research_corpus",
    "publish_research_model",
    "smoke_research_export",
    "train_research_comparison_matrix",
    "train_research_model",
]
