"""Leakage-auditable protocol construction for ogbl-collab."""

from __future__ import annotations

from typing import Any

from .sampling import ExactUndirectedNegativeSampler, canonical_edge_set, forbidden_union
from .types import CorpusArrays, ProtocolBundle, TemporalStage, edge_pairs


FEATURE_TIME_WARNING = (
    "OGB author-feature generation time is not independently verifiable; "
    "strict_edge_time proves edge-time isolation only."
)


def _bidirectional_unique(edges: Any) -> Any:
    import numpy as np

    pairs = sorted(canonical_edge_set(edges))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    forward = np.asarray(pairs, dtype=np.int64)
    return np.concatenate((forward, forward[:, ::-1]), axis=0)


def _repeat_mask(message_edges: Any, positives: Any) -> Any:
    import numpy as np

    known = canonical_edge_set(message_edges)
    return np.asarray(
        [tuple(sorted((int(source), int(target)))) in known for source, target in positives],
        dtype=np.bool_,
    )


def _stage(
    name: str,
    cutoff: int,
    target: int,
    message: Any,
    positives: Any,
    negatives: Any | None,
    source: str,
) -> TemporalStage:
    message_pairs = _bidirectional_unique(message)
    positive_pairs = edge_pairs(positives, name=f"{name}_positive_edges")
    negative_pairs = None if negatives is None else edge_pairs(
        negatives, name=f"{name}_negative_edges"
    )
    return TemporalStage(
        name=name,  # type: ignore[arg-type]
        message_cutoff_year=cutoff,
        target_year=target,
        message_edges=message_pairs,
        positive_edges=positive_pairs,
        negative_edges=negative_pairs,
        repeated_mask=_repeat_mask(message_pairs, positive_pairs),
        negative_source=source,  # type: ignore[arg-type]
    )


def build_protocol(corpus: CorpusArrays, track: str) -> ProtocolBundle:
    """Construct one immutable baseline protocol without consulting future edges.

    In the strict track the only edge collection added between stages is the
    immediately preceding target-period positive collection.  Test negatives do
    not participate in train or validation construction.
    """

    import numpy as np

    if track not in ("ogb_official", "strict_edge_time"):
        raise ValueError(f"unsupported baseline track: {track}")
    train_message = corpus.train_message_edges
    if track == "ogb_official":
        train = _stage(
            "train", 2017, 2017, train_message, corpus.train_positive_edges, None, "exact_sampler"
        )
        validation = _stage(
            "validation",
            2017,
            2018,
            train_message,
            corpus.validation_positive_edges,
            corpus.validation_negative_edges,
            "official_fixed",
        )
        test = _stage(
            "test",
            2017,
            2019,
            train_message,
            corpus.test_positive_edges,
            corpus.test_negative_edges,
            "official_fixed",
        )
        return ProtocolBundle(
            track="ogb_official",
            train=train,
            validation=validation,
            test=test,
            warnings=(
                "Training supervision edges may also occur in the message graph under the "
                "official transductive protocol.",
            ),
            audit={
                "futureEdgesUsedByTrain": False,
                "validationEdgesUsedByOfficialTestMessageGraph": False,
                "officialFixedNegatives": True,
            },
        )

    years = corpus.train_edge_year
    if years.shape[0] != train_message.shape[0]:
        raise ValueError("strict track requires one train edge year per message edge")
    before_2017 = (
        corpus.strict_train_message_edges
        if corpus.strict_train_message_edges is not None
        else train_message[years <= 2016]
    )
    during_2017 = (
        corpus.strict_train_positive_edges
        if corpus.strict_train_positive_edges is not None
        else forbidden_union(train_message[years == 2017])
    )
    if during_2017.size == 0:
        raise ValueError("strict track requires 2017 train edges as supervision")
    through_2017 = (
        corpus.strict_validation_message_edges
        if corpus.strict_validation_message_edges is not None
        else train_message[years <= 2017]
    )
    through_2018 = (
        corpus.strict_test_message_edges
        if corpus.strict_test_message_edges is not None
        else forbidden_union(through_2017, corpus.validation_positive_edges)
    )
    validation_negatives = ExactUndirectedNegativeSampler(
        corpus.num_nodes,
        forbidden_union(through_2017, corpus.validation_positive_edges),
        seed=20260818,
    ).sample(len(corpus.validation_positive_edges))
    test_negatives = ExactUndirectedNegativeSampler(
        corpus.num_nodes,
        forbidden_union(through_2018, corpus.test_positive_edges),
        seed=20260819,
    ).sample(len(corpus.test_positive_edges))
    train = _stage(
        "train", 2016, 2017, before_2017, during_2017, None, "exact_sampler"
    )
    validation = _stage(
        "validation",
        2017,
        2018,
        through_2017,
        corpus.validation_positive_edges,
        validation_negatives,
        "exact_sampler",
    )
    test = _stage(
        "test",
        2018,
        2019,
        through_2018,
        corpus.test_positive_edges,
        test_negatives,
        "exact_sampler",
    )
    return ProtocolBundle(
        track="strict_edge_time",
        train=train,
        validation=validation,
        test=test,
        warnings=(FEATURE_TIME_WARNING,),
        audit={
            "futureEdgesUsedByTrain": False,
            "trainMaxMessageYear": int(np.max(years[years <= 2016])),
            "validationMaxMessageYear": 2017,
            "testMaxMessageYear": 2018,
            "testMessageIncludesValidationPositivesOnly": True,
            "featurePointInTimeVerified": False,
            "validationNegativeSeed": 20260818,
            "testNegativeSeed": 20260819,
            "futureNegativeArraysRead": False,
        },
    )


def build_protocols(corpus: CorpusArrays) -> dict[str, ProtocolBundle]:
    return {
        "ogb_official": build_protocol(corpus, "ogb_official"),
        "strict_edge_time": build_protocol(corpus, "strict_edge_time"),
    }
