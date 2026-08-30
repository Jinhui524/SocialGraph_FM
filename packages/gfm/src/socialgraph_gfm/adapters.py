"""Explicit adapters between the public API wire protocol and offline internals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from .contracts import (
    FeatureModality as InternalFeatureModality,
)
from .contracts import (
    FeatureTarget as InternalFeatureTarget,
)
from .contracts import (
    FitScope as InternalFitScope,
)
from .contracts import (
    GraphSnapshot,
    InternalFeatureManifest,
    InternalGraphSnapshotRef,
    InternalTimeRange,
    SmokeCorpusManifest,
    feature_key,
)
from .contracts import (
    MissingPolicy as InternalMissingPolicy,
)
from .contracts import (
    PrivacyLevel as InternalPrivacyLevel,
)
from .errors import ContractViolation
from .public_contracts import (
    CorpusManifest,
    FeatureManifest,
    GraphSnapshotRef,
    TimeRange,
)

PublicMissing = Literal["mask", "zero", "unknown_token", "reject"]
PublicPrivacy = Literal["public", "project", "sensitive", "prohibited"]
Deidentification = Literal["none", "pseudonymized", "aggregated"]

_TO_PUBLIC_MISSING: dict[InternalMissingPolicy, PublicMissing] = {
    InternalMissingPolicy.ERROR: "reject",
    InternalMissingPolicy.ZERO_WITH_MASK: "mask",
    InternalMissingPolicy.CATEGORY_WITH_MASK: "unknown_token",
}
_TO_INTERNAL_MISSING = {
    "reject": InternalMissingPolicy.ERROR,
    "zero": InternalMissingPolicy.ZERO_WITH_MASK,
    "mask": InternalMissingPolicy.ZERO_WITH_MASK,
    "unknown_token": InternalMissingPolicy.CATEGORY_WITH_MASK,
}
_TO_PUBLIC_PRIVACY: dict[InternalPrivacyLevel, PublicPrivacy] = {
    InternalPrivacyLevel.PUBLIC: "public",
    InternalPrivacyLevel.INTERNAL: "project",
    InternalPrivacyLevel.SENSITIVE: "sensitive",
    InternalPrivacyLevel.PROHIBITED: "prohibited",
}
_TO_INTERNAL_PRIVACY = {
    "public": InternalPrivacyLevel.PUBLIC,
    "project": InternalPrivacyLevel.INTERNAL,
    "sensitive": InternalPrivacyLevel.SENSITIVE,
    "prohibited": InternalPrivacyLevel.PROHIBITED,
}


def internal_snapshot_to_public(snapshot: GraphSnapshot) -> GraphSnapshotRef:
    """Produce the sole external GraphSnapshotRef wire shape from an internal snapshot."""

    name_counts: dict[str, int] = {}
    for feature in snapshot.ref.feature_manifest:
        name_counts[feature.name] = name_counts.get(feature.name, 0) + 1
    features = [
        FeatureManifest(
            id=(feature_key(feature) if name_counts[feature.name] > 1 else feature.name),
            attribute=feature.name,
            target=feature.target.value,
            modality=feature.modality.value,
            dtype=feature.dtype,
            missingPolicy=_TO_PUBLIC_MISSING[feature.missing_policy],
            privacyLevel=_TO_PUBLIC_PRIVACY[feature.privacy_level],
            fitScope=("train_only" if feature.fit_scope == InternalFitScope.TRAIN_SPLIT_ONLY else "none"),
            availableAtField=None,
            embeddingRef=feature.embedding_model_ref,
            inferenceAllowed=feature.inference_allowed,
        )
        for feature in snapshot.ref.feature_manifest
    ]
    time_range = snapshot.ref.time_range
    return GraphSnapshotRef(
        graphVersionId=snapshot.ref.graph_version,
        graphFactHash=snapshot.ref.fact_hash,
        contentHash=snapshot.ref.content_hash,
        profile=snapshot.ref.profile,
        nodeCount=len(snapshot.nodes),
        edgeCount=len(snapshot.edges),
        timeRange=(TimeRange(start=time_range.start, end=time_range.end) if time_range else None),
        featureManifest=features,
    )


def public_snapshot_to_internal_ref(
    public: GraphSnapshotRef,
    *,
    feature_owner_types: Mapping[str, str | None],
    inference_property_allowlist: tuple[str, ...] = (),
    deidentification: Deidentification = "pseudonymized",
) -> InternalGraphSnapshotRef:
    """Validate and adapt a wire reference for internal materialization metadata.

    Node/edge records are intentionally not reconstructed: a public reference proves
    identity and compatibility but is not the graph payload. Feature owner types must be
    supplied from the workbench compatibility mapping instead of being guessed.
    """

    features: list[InternalFeatureManifest] = []
    resolved_allowlist: list[str] = []
    public_to_internal_key: dict[str, str] = {}
    attributes_to_keys: dict[str, list[str]] = {}
    for feature in public.feature_manifest:
        if feature.modality in ("boolean", "timestamp"):
            raise ContractViolation(
                f"Public feature modality {feature.modality!r} requires an explicit materializer adapter"
            )
        if feature.fit_scope == "all_nodes_transductive":
            raise ContractViolation("Transductive all-node fitting is not allowed by offline internals")
        if feature.id not in feature_owner_types:
            raise ContractViolation(f"Missing owner type mapping for feature {feature.id!r}")
        modality = InternalFeatureModality(feature.modality)
        dimensions = 1
        if modality == InternalFeatureModality.EMBEDDING:
            raise ContractViolation(
                f"Embedding feature {feature.id!r} needs dimensions from an immutable embedding artifact"
            )
        internal_name = feature.attribute
        owner_type = feature_owner_types[feature.id]
        internal = InternalFeatureManifest(
                name=internal_name,
                target=InternalFeatureTarget(feature.target),
                ownerType=owner_type,
                modality=modality,
                dtype=feature.dtype,
                dimensions=dimensions,
                missingPolicy=_TO_INTERNAL_MISSING[feature.missing_policy],
                privacyLevel=_TO_INTERNAL_PRIVACY[feature.privacy_level],
                inferenceAllowed=feature.inference_allowed,
                fitScope=(
                    InternalFitScope.TRAIN_SPLIT_ONLY
                    if feature.fit_scope == "train_only"
                    or modality in (InternalFeatureModality.NUMERIC, InternalFeatureModality.CATEGORICAL)
                    else InternalFitScope.NONE
                ),
                embeddingModelRef=feature.embedding_ref,
            )
        features.append(internal)
        internal_key = feature_key(internal)
        public_to_internal_key[feature.id] = internal_key
        attributes_to_keys.setdefault(feature.attribute, []).append(internal_key)
    for item in inference_property_allowlist:
        if item in public_to_internal_key:
            resolved_allowlist.append(public_to_internal_key[item])
            continue
        matches = attributes_to_keys.get(item, [])
        if len(matches) > 1:
            raise ContractViolation(
                f"Inference allowlist feature {item!r} is ambiguous; use its owner-qualified public id"
            )
        if not matches:
            raise ContractViolation(f"Inference allowlist feature {item!r} is undeclared")
        resolved_allowlist.extend(matches)
    time_range = public.time_range
    return InternalGraphSnapshotRef(
        graphVersion=public.graph_version_id,
        factHash=public.graph_fact_hash,
        contentHash=public.content_hash,
        profile=public.profile,
        timeRange=(
            InternalTimeRange(start=time_range.start, end=time_range.end)
            if time_range and time_range.start and time_range.end
            else None
        ),
        featureManifest=tuple(features),
        inferencePropertyAllowlist=tuple(resolved_allowlist),
        deidentification=deidentification,
        userDataTrainingOptIn=False,
    )


def internal_smoke_corpus_to_public(corpus: SmokeCorpusManifest) -> CorpusManifest:
    return CorpusManifest(
        id=corpus.corpus_id,
        version=corpus.version,
        sourceHash=corpus.source_hash,
        licenseId=corpus.license_id,
        intendedUse="synthetic_test_only",
        split="synthetic",
        adapter=corpus.adapter,
    )
