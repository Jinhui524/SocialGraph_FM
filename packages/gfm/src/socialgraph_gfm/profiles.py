"""Canonical collaboration graph profiles and compatibility validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .contracts import GraphSnapshot, PrivacyLevel


class TemporalRule(str, Enum):
    OPTIONAL = "optional"
    EVENT = "event"
    STABLE = "stable"


@dataclass(frozen=True)
class RelationSpec:
    source_type: str
    relation: str
    target_type: str
    temporal_rule: TemporalRule


@dataclass(frozen=True)
class ProfileSpec:
    profile_id: str
    node_types: frozenset[str]
    relations: tuple[RelationSpec, ...]
    max_nodes: int = 2_000_000
    max_edges: int = 20_000_000

    def relation(self, name: str) -> RelationSpec | None:
        return next((item for item in self.relations if item.relation == name), None)


ACTOR_INTERACTION = ProfileSpec(
    profile_id="collaboration.actor-interaction/1.0",
    node_types=frozenset({"actor"}),
    relations=tuple(
        RelationSpec("actor", relation, "actor", TemporalRule.EVENT)
        for relation in ("collaborates", "replies", "mentions", "reviews", "endorses")
    ),
)

ACTIVITY_HETERO = ProfileSpec(
    profile_id="collaboration.activity-hetero/1.0",
    node_types=frozenset({"actor", "artifact", "community", "topic"}),
    relations=(
        RelationSpec("actor", "actor_interacts_actor", "actor", TemporalRule.EVENT),
        RelationSpec("actor", "actor_contributes_artifact", "artifact", TemporalRule.EVENT),
        RelationSpec("actor", "actor_joins_community", "community", TemporalRule.STABLE),
        RelationSpec("artifact", "artifact_belongs_community", "community", TemporalRule.STABLE),
        RelationSpec("artifact", "artifact_has_topic", "topic", TemporalRule.STABLE),
    ),
)

PROFILES: dict[str, ProfileSpec] = {
    ACTOR_INTERACTION.profile_id: ACTOR_INTERACTION,
    ACTIVITY_HETERO.profile_id: ACTIVITY_HETERO,
}


class CompatibilityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    message: str
    path: str


class CompatibilityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    compatible: bool
    profile: str
    blockers: tuple[CompatibilityIssue, ...] = ()
    warnings: tuple[CompatibilityIssue, ...] = ()
    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)
    user_data_training_opt_in: bool = Field(False, alias="userDataTrainingOptIn")


def check_compatibility(snapshot: GraphSnapshot) -> CompatibilityReport:
    profile = PROFILES[snapshot.ref.profile]
    blockers: list[CompatibilityIssue] = []
    warnings: list[CompatibilityIssue] = []
    nodes_by_id = {node.node_id: node for node in snapshot.nodes}

    if not snapshot.nodes:
        blockers.append(
            CompatibilityIssue(
                code="EMPTY_GRAPH",
                message="A collaboration graph must contain at least one node",
                path="nodes",
            )
        )
    if not any(node.node_type == "actor" for node in snapshot.nodes):
        blockers.append(
            CompatibilityIssue(
                code="ACTOR_TYPE_REQUIRED",
                message="A collaboration governance graph requires at least one actor",
                path="nodes",
            )
        )
    if not snapshot.edges:
        blockers.append(
            CompatibilityIssue(
                code="NO_RELATION_FACTS",
                message="A collaboration graph must contain at least one relation fact",
                path="edges",
            )
        )

    for index, node in enumerate(snapshot.nodes):
        if node.node_type not in profile.node_types:
            blockers.append(
                CompatibilityIssue(
                    code="UNKNOWN_NODE_TYPE",
                    message=f"Node type {node.node_type!r} is not part of {profile.profile_id}",
                    path=f"nodes[{index}].nodeType",
                )
            )

    for index, edge in enumerate(snapshot.edges):
        relation = profile.relation(edge.relation)
        if relation is None:
            blockers.append(
                CompatibilityIssue(
                    code="AMBIGUOUS_RELATION",
                    message=f"Relation {edge.relation!r} has no canonical mapping",
                    path=f"edges[{index}].relation",
                )
            )
            continue
        source_type = nodes_by_id[edge.source].node_type
        target_type = nodes_by_id[edge.target].node_type
        if (source_type, target_type) != (relation.source_type, relation.target_type):
            blockers.append(
                CompatibilityIssue(
                    code="INVALID_RELATION_ENDPOINTS",
                    message=(
                        f"{edge.relation} requires {relation.source_type}->{relation.target_type}, "
                        f"received {source_type}->{target_type}"
                    ),
                    path=f"edges[{index}]",
                )
            )
        if relation.temporal_rule == TemporalRule.EVENT and edge.timestamp is None:
            blockers.append(
                CompatibilityIssue(
                    code="EVENT_TIMESTAMP_REQUIRED",
                    message=f"Event relation {edge.relation!r} requires timestamp",
                    path=f"edges[{index}].timestamp",
                )
            )
        if relation.temporal_rule == TemporalRule.STABLE and edge.timestamp is not None:
            warnings.append(
                CompatibilityIssue(
                    code="STABLE_RELATION_EVENT_TIME_IGNORED",
                    message="Stable relation uses validFrom/validTo; timestamp is not an event label",
                    path=f"edges[{index}].timestamp",
                )
            )

    if len(snapshot.nodes) > profile.max_nodes:
        blockers.append(
            CompatibilityIssue(
                code="NODE_LIMIT_EXCEEDED",
                message=f"Snapshot has more than {profile.max_nodes} nodes",
                path="nodes",
            )
        )
    if len(snapshot.edges) > profile.max_edges:
        blockers.append(
            CompatibilityIssue(
                code="EDGE_LIMIT_EXCEEDED",
                message=f"Snapshot has more than {profile.max_edges} edges",
                path="edges",
            )
        )

    allowed = set(snapshot.ref.resolved_inference_feature_keys())
    for feature in snapshot.ref.feature_manifest:
        from .contracts import feature_key

        if feature.privacy_level == PrivacyLevel.SENSITIVE and feature_key(feature) in allowed:
            warnings.append(
                CompatibilityIssue(
                    code="SENSITIVE_INFERENCE_FEATURE",
                    message=f"Sensitive feature {feature.name!r} requires deployment review",
                    path="ref.inferencePropertyAllowlist",
                )
            )

    return CompatibilityReport(
        compatible=not blockers,
        profile=profile.profile_id,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        nodeCount=len(snapshot.nodes),
        edgeCount=len(snapshot.edges),
        userDataTrainingOptIn=False,
    )
