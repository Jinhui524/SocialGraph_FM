"""Training-native, in-memory types for the ogbl-collab baseline.

These types deliberately do not depend on the website inference profile.  They
are a narrow adapter boundary between the safe corpus package and the baseline
algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

BaselineTrack = Literal["ogb_official", "strict_edge_time"]
BaselineModel = Literal["cn", "aa", "ra", "mlp", "graphsage"]
BaselinePhase = Literal["dev", "formal"]
StageName = Literal["train", "validation", "test"]


def _numpy():
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - exercised by runtime doctor
        raise RuntimeError("the baseline requires NumPy") from error
    return np


def edge_pairs(value: Any, *, name: str) -> Any:
    """Return a checked ``int64 [N, 2]`` edge-pair array.

    OGB and PyG use both ``[N, 2]`` and ``[2, N]`` conventions.  The ambiguity
    of a 2x2 array is resolved in favour of OGB's pair-row convention.
    """

    np = _numpy()
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be rank 2")
    if array.shape[1] == 2:
        pairs = array
    elif array.shape[0] == 2:
        pairs = array.T
    else:
        raise ValueError(f"{name} must have shape [N,2] or [2,N]")
    if pairs.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer node indices")
    return np.ascontiguousarray(pairs, dtype=np.int64)


def _required(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"required corpus array is missing; expected one of {names}")


def _optional_edges(mapping: Mapping[str, Any], name: str) -> Any | None:
    return edge_pairs(mapping[name], name=name) if name in mapping else None


@dataclass(frozen=True)
class CorpusArrays:
    """Validated arrays needed by both baseline tracks."""

    node_features: Any
    train_message_edges: Any
    train_edge_year: Any
    train_positive_edges: Any
    validation_positive_edges: Any
    validation_negative_edges: Any
    test_positive_edges: Any
    test_negative_edges: Any
    num_nodes: int
    corpus_hash: str = ""
    strict_train_message_edges: Any | None = None
    strict_train_positive_edges: Any | None = None
    strict_validation_message_edges: Any | None = None
    strict_test_message_edges: Any | None = None

    @classmethod
    def from_mapping(
        cls,
        arrays: Mapping[str, Any],
        *,
        corpus_hash: str = "",
        expected_num_nodes: int | None = None,
        expected_feature_dim: int | None = None,
    ) -> "CorpusArrays":
        """Adapt flat safe-package arrays or an OGB-style ``split_edge`` map."""

        np = _numpy()
        features = np.asarray(_required(arrays, "node_features", "node_feat", "x"))
        if features.ndim != 2 or features.dtype.kind not in "fiu":
            raise ValueError("node features must be a finite numeric rank-2 array")
        features = np.ascontiguousarray(features, dtype=np.float32)
        if not bool(np.isfinite(features).all()):
            raise ValueError("node features contain NaN or Infinity")
        num_nodes = int(features.shape[0])
        if expected_num_nodes is not None and num_nodes != expected_num_nodes:
            raise ValueError(f"expected {expected_num_nodes} nodes, got {num_nodes}")
        if expected_feature_dim is not None and features.shape[1] != expected_feature_dim:
            raise ValueError(
                f"expected feature dimension {expected_feature_dim}, got {features.shape[1]}"
            )

        split = arrays.get("split_edge")
        if isinstance(split, Mapping):
            train = split.get("train", {})
            valid = split.get("valid", split.get("validation", {}))
            test = split.get("test", {})
            if not all(isinstance(item, Mapping) for item in (train, valid, test)):
                raise ValueError("split_edge train/valid/test entries must be mappings")
            train_positive = _required(train, "edge")
            validation_positive = _required(valid, "edge")
            validation_negative = _required(valid, "edge_neg")
            test_positive = _required(test, "edge")
            test_negative = _required(test, "edge_neg")
            year_value = arrays.get("train_edge_year", arrays.get("edge_year", train.get("year")))
        else:
            train_positive = _required(
                arrays, "train_positive_edges", "train_edge", "variant_train_positive"
            )
            validation_positive = _required(
                arrays,
                "validation_positive_edges",
                "valid_edge",
                "validation_edge",
                "variant_validation_positive",
            )
            validation_negative = _required(
                arrays,
                "validation_negative_edges",
                "valid_edge_neg",
                "validation_edge_neg",
                "variant_validation_negative",
            )
            test_positive = _required(
                arrays, "test_positive_edges", "test_edge", "variant_test_positive"
            )
            test_negative = _required(
                arrays, "test_negative_edges", "test_edge_neg", "variant_test_negative"
            )
            year_value = _required(arrays, "train_edge_year", "edge_year", "edge_timestamp")

        message = edge_pairs(
            _required(arrays, "train_message_edges", "edge_index"),
            name="train_message_edges",
        )
        years = np.asarray(year_value).reshape(-1)
        if years.dtype.kind not in "iu":
            raise ValueError("train edge years must be integers")
        years = np.ascontiguousarray(years, dtype=np.int64)
        if years.shape[0] != message.shape[0]:
            # PyG undirected message edges are commonly duplicated while the
            # corresponding years are stored once.  Accept only the exact 2x case.
            if message.shape[0] == 2 * years.shape[0]:
                years = np.concatenate((years, years))
            else:
                raise ValueError("train edge years must align with message edges")

        result = cls(
            node_features=features,
            train_message_edges=message,
            train_edge_year=years,
            train_positive_edges=edge_pairs(train_positive, name="train_positive_edges"),
            validation_positive_edges=edge_pairs(
                validation_positive, name="validation_positive_edges"
            ),
            validation_negative_edges=edge_pairs(
                validation_negative, name="validation_negative_edges"
            ),
            test_positive_edges=edge_pairs(test_positive, name="test_positive_edges"),
            test_negative_edges=edge_pairs(test_negative, name="test_negative_edges"),
            num_nodes=num_nodes,
            corpus_hash=corpus_hash,
            strict_train_message_edges=_optional_edges(
                arrays, "strict_train_message_edge_index"
            ),
            strict_train_positive_edges=_optional_edges(arrays, "strict_train_positive"),
            strict_validation_message_edges=_optional_edges(
                arrays, "strict_validation_message_edge_index"
            ),
            strict_test_message_edges=_optional_edges(
                arrays, "strict_test_message_edge_index"
            ),
        )
        result.validate_indices()
        return result

    def validate_indices(self) -> None:
        np = _numpy()
        for name, pairs in (
            ("train_message_edges", self.train_message_edges),
            ("train_positive_edges", self.train_positive_edges),
            ("validation_positive_edges", self.validation_positive_edges),
            ("validation_negative_edges", self.validation_negative_edges),
            ("test_positive_edges", self.test_positive_edges),
            ("test_negative_edges", self.test_negative_edges),
        ):
            if pairs.size and (int(np.min(pairs)) < 0 or int(np.max(pairs)) >= self.num_nodes):
                raise ValueError(f"{name} contains an out-of-range node index")
        for name in (
            "strict_train_message_edges",
            "strict_train_positive_edges",
            "strict_validation_message_edges",
            "strict_test_message_edges",
        ):
            pairs = getattr(self, name)
            if pairs is not None and pairs.size and (
                int(np.min(pairs)) < 0 or int(np.max(pairs)) >= self.num_nodes
            ):
                raise ValueError(f"{name} contains an out-of-range node index")


@dataclass(frozen=True)
class TemporalStage:
    """A point-in-time message graph and its next-period supervision."""

    name: StageName
    message_cutoff_year: int
    target_year: int
    message_edges: Any
    positive_edges: Any
    negative_edges: Any | None
    repeated_mask: Any
    negative_source: Literal["exact_sampler", "official_fixed"]


@dataclass(frozen=True)
class ProtocolBundle:
    track: BaselineTrack
    train: TemporalStage
    validation: TemporalStage
    test: TemporalStage
    warnings: tuple[str, ...] = ()
    audit: Mapping[str, Any] = field(default_factory=dict)

    def stage(self, name: StageName) -> TemporalStage:
        return {"train": self.train, "validation": self.validation, "test": self.test}[name]


@dataclass(frozen=True)
class RunSpec:
    experiment_id: str
    run_id: str
    phase: BaselinePhase
    track: BaselineTrack
    model: BaselineModel
    seed: int


@dataclass(frozen=True)
class CoreRunResult:
    spec: RunSpec
    validation_metrics: Mapping[str, float]
    test_metrics: Mapping[str, float] | None
    strata: Mapping[str, Mapping[str, float | int]]
    best_epoch: int | None
    peak_cuda_memory_mib: float
    selected_batch_size: int | None
    test_read_after_selection: bool
    history: tuple[Mapping[str, float], ...] = ()
