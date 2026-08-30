"""Offline ogbl-collab baseline engine.

Importing this package keeps Torch/PyG lazy; model and trainer symbols live in
their explicit submodules so readiness/contract checks remain lightweight.
"""

from .heuristics import adjacency_sets, score_heuristic
from .protocols import FEATURE_TIME_WARNING, build_protocol, build_protocols
from .sampling import ExactUndirectedNegativeSampler, canonical_edge_set, canonical_pair
from .types import CoreRunResult, CorpusArrays, ProtocolBundle, RunSpec, TemporalStage

__all__ = [
    "CoreRunResult",
    "CorpusArrays",
    "ExactUndirectedNegativeSampler",
    "FEATURE_TIME_WARNING",
    "ProtocolBundle",
    "RunSpec",
    "TemporalStage",
    "adjacency_sets",
    "build_protocol",
    "build_protocols",
    "canonical_edge_set",
    "canonical_pair",
    "score_heuristic",
]
