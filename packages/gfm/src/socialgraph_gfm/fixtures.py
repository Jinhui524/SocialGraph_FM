"""Deterministic synthetic collaboration fixtures; these are not training data."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, cast

from .canonical import canonical_sha256
from .contracts import (
    EdgeRecord,
    FeatureModality,
    FeatureTarget,
    FitScope,
    GraphSnapshot,
    InternalFeatureManifest,
    InternalGraphSnapshotRef,
    InternalTimeRange,
    MissingPolicy,
    NodeRecord,
    PrivacyLevel,
)

FixtureName = Literal["actor", "hetero"]
ProfileName = Literal[
    "collaboration.actor-interaction/1.0", "collaboration.activity-hetero/1.0"
]


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def actor_interaction_fixture() -> GraphSnapshot:
    nodes = (
        NodeRecord(nodeId="actor:ada", nodeType="actor", features={"activity": 8.0, "role": "maintainer"}),
        NodeRecord(nodeId="actor:bo", nodeType="actor", features={"activity": 3.0, "role": "newcomer"}),
        NodeRecord(nodeId="actor:cy", nodeType="actor", features={"activity": None, "role": "visitor"}),
        NodeRecord(nodeId="actor:di", nodeType="actor", features={"activity": 5000.0, "role": None}),
    )
    edges = (
        EdgeRecord(edgeId="e:1", source="actor:ada", target="actor:bo", relation="reviews", timestamp=_utc("2026-01-01T08:00:00Z"), weight=1.0, originalRelationType="pull_request_review"),
        EdgeRecord(edgeId="e:2", source="actor:bo", target="actor:cy", relation="mentions", timestamp=_utc("2026-01-02T08:00:00Z"), weight=0.5, originalRelationType="issue_mention"),
        EdgeRecord(edgeId="e:3", source="actor:cy", target="actor:ada", relation="replies", timestamp=_utc("2026-01-03T08:00:00Z"), weight=1.5, originalRelationType="discussion_reply"),
        EdgeRecord(edgeId="e:4", source="actor:di", target="actor:bo", relation="endorses", timestamp=_utc("2026-01-04T08:00:00Z"), weight=1.0, originalRelationType="reaction"),
        EdgeRecord(edgeId="e:5", source="actor:ada", target="actor:di", relation="collaborates", timestamp=_utc("2026-01-05T08:00:00Z"), weight=2.0, originalRelationType="co_commit"),
    )
    features = (
        InternalFeatureManifest(
            name="activity", target=FeatureTarget.NODE, ownerType="actor",
            modality=FeatureModality.NUMERIC, dtype="float32",
            missingPolicy=MissingPolicy.ZERO_WITH_MASK, privacyLevel=PrivacyLevel.PUBLIC,
            inferenceAllowed=True, fitScope=FitScope.TRAIN_SPLIT_ONLY,
        ),
        InternalFeatureManifest(
            name="role", target=FeatureTarget.NODE, ownerType="actor",
            modality=FeatureModality.CATEGORICAL, dtype="string",
            missingPolicy=MissingPolicy.CATEGORY_WITH_MASK, privacyLevel=PrivacyLevel.INTERNAL,
            inferenceAllowed=True, fitScope=FitScope.TRAIN_SPLIT_ONLY,
        ),
    )
    return _snapshot(
        "collaboration.actor-interaction/1.0", "synthetic-actor-v1", nodes, edges, features
    )


def activity_hetero_fixture() -> GraphSnapshot:
    nodes = (
        NodeRecord(nodeId="actor:ada", nodeType="actor", features={"activity": 8.0}),
        NodeRecord(nodeId="actor:bo", nodeType="actor", features={"activity": 8000.0}),
        NodeRecord(nodeId="artifact:pr1", nodeType="artifact", features={"quality": 0.9}),
        NodeRecord(nodeId="artifact:pr2", nodeType="artifact", features={"quality": 99.0}),
        NodeRecord(nodeId="community:core", nodeType="community", features={"size": 12.0}),
        NodeRecord(nodeId="community:docs", nodeType="community", features={"size": 12000.0}),
        NodeRecord(nodeId="topic:graphs", nodeType="topic", features={"popularity": 0.7}),
        NodeRecord(nodeId="topic:safety", nodeType="topic", features={"popularity": 700.0}),
    )
    edges = (
        EdgeRecord(edgeId="h:1", source="actor:ada", target="actor:bo", relation="actor_interacts_actor", timestamp=_utc("2026-02-01T08:00:00Z"), weight=1.0, originalRelationType="review_reply"),
        EdgeRecord(edgeId="h:2", source="actor:bo", target="artifact:pr1", relation="actor_contributes_artifact", timestamp=_utc("2026-02-02T08:00:00Z"), weight=1.0, originalRelationType="opened_pr"),
        EdgeRecord(edgeId="h:3", source="actor:ada", target="community:core", relation="actor_joins_community", validFrom=_utc("2025-01-01T00:00:00Z"), weight=1.0),
        EdgeRecord(edgeId="h:4", source="artifact:pr1", target="community:core", relation="artifact_belongs_community", validFrom=_utc("2026-02-02T08:00:00Z"), weight=1.0),
        EdgeRecord(edgeId="h:5", source="artifact:pr1", target="topic:graphs", relation="artifact_has_topic", validFrom=_utc("2026-02-02T08:00:00Z"), weight=1.0),
    )
    features = tuple(
        InternalFeatureManifest(
            name=name, target=FeatureTarget.NODE, ownerType=owner,
            modality=FeatureModality.NUMERIC, dtype="float32",
            missingPolicy=MissingPolicy.ZERO_WITH_MASK, privacyLevel=PrivacyLevel.PUBLIC,
            inferenceAllowed=True, fitScope=FitScope.TRAIN_SPLIT_ONLY,
        )
        for name, owner in (
            ("activity", "actor"), ("quality", "artifact"),
            ("size", "community"), ("popularity", "topic"),
        )
    )
    return _snapshot(
        "collaboration.activity-hetero/1.0", "synthetic-hetero-v1", nodes, edges, features
    )


def _snapshot(
    profile: ProfileName,
    version: str,
    nodes: tuple[NodeRecord, ...],
    edges: tuple[EdgeRecord, ...],
    features: tuple[InternalFeatureManifest, ...],
) -> GraphSnapshot:
    content_payload = {"profile": profile, "featureManifest": features, "nodes": nodes, "edges": edges}
    fact_payload = {"nodes": nodes, "edges": edges}
    timestamps = [edge.timestamp for edge in edges if edge.timestamp is not None]
    ref = InternalGraphSnapshotRef(
        graphVersion=version,
        factHash=canonical_sha256(fact_payload),
        contentHash=canonical_sha256(content_payload),
        profile=profile,
        timeRange=(
            InternalTimeRange(start=min(timestamps), end=max(timestamps)) if timestamps else None
        ),
        featureManifest=features,
        inferencePropertyAllowlist=tuple(feature.name for feature in features),
        deidentification="pseudonymized",
        userDataTrainingOptIn=False,
    )
    return GraphSnapshot(ref=ref, nodes=nodes, edges=edges)


def get_fixture(name_or_profile: str) -> GraphSnapshot:
    aliases = {
        "actor": actor_interaction_fixture,
        "collaboration.actor-interaction/1.0": actor_interaction_fixture,
        "hetero": activity_hetero_fixture,
        "collaboration.activity-hetero/1.0": activity_hetero_fixture,
    }
    try:
        return aliases[name_or_profile]()
    except KeyError as error:
        raise ValueError(f"Unknown fixture/profile: {name_or_profile}") from error


def fixture_names(name_or_profile: str) -> tuple[FixtureName, ...]:
    if name_or_profile == "both":
        return ("actor", "hetero")
    snapshot = get_fixture(name_or_profile)
    result = "actor" if snapshot.ref.profile.endswith("actor-interaction/1.0") else "hetero"
    return (cast(FixtureName, result),)


def smoke_fit_node_ids(name_or_profile: str) -> dict[str, tuple[str, ...]]:
    """Deterministic synthetic fit selection with at least one held-out node per type."""

    snapshot = get_fixture(name_or_profile)
    if snapshot.ref.profile.endswith("actor-interaction/1.0"):
        return {"actor": ("actor:ada", "actor:bo")}
    return {
        "actor": ("actor:ada",),
        "artifact": ("artifact:pr1",),
        "community": ("community:core",),
        "topic": ("topic:graphs",),
    }
