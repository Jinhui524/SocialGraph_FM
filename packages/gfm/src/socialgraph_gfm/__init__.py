"""SocialGraph-FM contract-first GFM infrastructure."""

from .contracts import GraphSnapshot
from .public_contracts import (
    CheckpointManifest,
    CoreFinding,
    CoreTaskManifest,
    CorpusManifest,
    FeatureManifest,
    GraphSnapshotRef,
    ModelCapability,
    TrainingRunManifest,
)

__all__ = [
    "CheckpointManifest",
    "CoreFinding",
    "CoreTaskManifest",
    "CorpusManifest",
    "FeatureManifest",
    "GraphSnapshot",
    "GraphSnapshotRef",
    "ModelCapability",
    "TrainingRunManifest",
]

__version__ = "0.1.0"
