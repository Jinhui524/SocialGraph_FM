from datetime import datetime

import pytest
from pydantic import ValidationError

from socialgraph_gfm.contracts import (
    FeatureModality,
    FeatureTarget,
    FitScope,
    InternalFeatureManifest,
    InternalGraphSnapshotRef,
    InternalModelCapability,
    MissingPolicy,
    PrivacyLevel,
)
from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.fixtures import activity_hetero_fixture, actor_interaction_fixture
from socialgraph_gfm.migrations import migrate_contract


def numeric_feature(**overrides):
    values = {
        "name": "activity",
        "target": FeatureTarget.NODE,
        "ownerType": "actor",
        "modality": FeatureModality.NUMERIC,
        "dtype": "float32",
        "missingPolicy": MissingPolicy.ZERO_WITH_MASK,
        "privacyLevel": PrivacyLevel.PUBLIC,
        "inferenceAllowed": True,
        "fitScope": FitScope.TRAIN_SPLIT_ONLY,
    }
    values.update(overrides)
    return InternalFeatureManifest(**values)


def test_two_fixtures_have_self_consistent_contract_hashes():
    for fixture in (actor_interaction_fixture(), activity_hetero_fixture()):
        assert fixture.payload_hash() == fixture.ref.content_hash
        assert fixture.ref.user_data_training_opt_in is False


def test_training_opt_in_cannot_be_enabled():
    feature = numeric_feature()
    with pytest.raises(ValidationError):
        InternalGraphSnapshotRef(
            graphVersion="v1",
            factHash="0" * 64,
            contentHash="1" * 64,
            profile="collaboration.actor-interaction/1.0",
            featureManifest=(feature,),
            inferencePropertyAllowlist=("activity",),
            userDataTrainingOptIn=True,
        )


def test_prohibited_or_disabled_feature_cannot_enter_allowlist():
    prohibited = numeric_feature(privacyLevel=PrivacyLevel.PROHIBITED, inferenceAllowed=False)
    with pytest.raises(ValidationError, match="prohibited or inference-disabled"):
        InternalGraphSnapshotRef(
            graphVersion="v1",
            factHash="0" * 64,
            contentHash="1" * 64,
            profile="collaboration.actor-interaction/1.0",
            featureManifest=(prohibited,),
            inferencePropertyAllowlist=("activity",),
        )


def test_embedding_requires_model_reference_and_dimensions():
    with pytest.raises(ValidationError, match="embeddingModelRef"):
        InternalFeatureManifest(
            name="bio_embedding",
            target="node",
            ownerType="actor",
            modality="embedding",
            dtype="float32",
            dimensions=32,
            missingPolicy="zero_with_mask",
            privacyLevel="internal",
            inferenceAllowed=True,
            fitScope="none",
        )


def test_serving_capability_requires_validation():
    with pytest.raises(ValidationError, match="validated"):
        InternalModelCapability(modelId="unvalidated", validated=False, servingReady=True)


def test_unknown_schema_migration_fails_closed():
    assert migrate_contract({"schemaVersion": "gfm.feature/1.0", "name": "x"})["name"] == "x"
    with pytest.raises(ContractViolation, match="No safe migration"):
        migrate_contract({"schemaVersion": "gfm.feature/0.1"})


def test_timestamp_must_be_timezone_aware():
    from socialgraph_gfm.contracts import EdgeRecord

    with pytest.raises(ValidationError, match="timezone-aware"):
        EdgeRecord(
            edgeId="e", source="a", target="b", relation="replies",
            timestamp=datetime(2026, 1, 1),  # noqa: DTZ001 - deliberately naive
        )


def test_feature_available_at_must_be_timezone_aware_and_ids_cannot_contain_dot():
    with pytest.raises(ValidationError, match="availableAt"):
        numeric_feature(availableAt=datetime(2026, 1, 1))  # noqa: DTZ001
    with pytest.raises(ValidationError):
        numeric_feature(name="ambiguous.activity")
    with pytest.raises(ValidationError):
        numeric_feature(ownerType="team.actor")


def test_owner_qualified_feature_identity_allows_same_name_and_rejects_bare_ambiguity():
    actor = numeric_feature(ownerType="actor")
    artifact = numeric_feature(ownerType="artifact")
    ref = InternalGraphSnapshotRef(
        graphVersion="v1",
        factHash="0" * 64,
        contentHash="1" * 64,
        profile="collaboration.activity-hetero/1.0",
        featureManifest=(actor, artifact),
        inferencePropertyAllowlist=("actor.activity",),
    )
    assert ref.resolved_inference_feature_keys() == ("actor.activity",)
    with pytest.raises(ValidationError, match="ambiguous"):
        InternalGraphSnapshotRef(
            graphVersion="v1",
            factHash="0" * 64,
            contentHash="1" * 64,
            profile="collaboration.activity-hetero/1.0",
            featureManifest=(actor, artifact),
            inferencePropertyAllowlist=("activity",),
        )
