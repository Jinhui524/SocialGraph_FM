import pytest

from socialgraph_gfm.adapters import (
    internal_smoke_corpus_to_public,
    internal_snapshot_to_public,
    public_snapshot_to_internal_ref,
)
from socialgraph_gfm.contracts import SmokeCorpusManifest
from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.fixtures import actor_interaction_fixture
from socialgraph_gfm.public_contracts import FeatureManifest


def test_internal_snapshot_round_trips_through_the_public_wire_ref():
    fixture = actor_interaction_fixture()
    public = internal_snapshot_to_public(fixture)
    assert public.schema_version == "gfm-graph-snapshot-ref/1.0"
    assert public.graph_version_id == fixture.ref.graph_version
    assert public.node_count == len(fixture.nodes)
    assert public.edge_count == len(fixture.edges)
    assert [feature.id for feature in public.feature_manifest] == ["activity", "role"]

    internal = public_snapshot_to_internal_ref(
        public,
        feature_owner_types={"activity": "actor", "role": "actor"},
        inference_property_allowlist=("activity", "role"),
    )
    assert internal.content_hash == fixture.ref.content_hash
    assert internal.user_data_training_opt_in is False
    assert [feature.name for feature in internal.feature_manifest] == ["activity", "role"]


def test_public_to_internal_adapter_rejects_unsupported_semantics():
    fixture = actor_interaction_fixture()
    public = internal_snapshot_to_public(fixture)
    timestamp = FeatureManifest(
        id="joined_at",
        attribute="joined_at",
        target="node",
        modality="timestamp",
        dtype="datetime",
        missingPolicy="reject",
        privacyLevel="project",
        fitScope="none",
        inferenceAllowed=True,
    )
    incompatible = public.model_copy(update={"feature_manifest": [timestamp]})
    with pytest.raises(ContractViolation, match="explicit materializer adapter"):
        public_snapshot_to_internal_ref(
            incompatible, feature_owner_types={"joined_at": "actor"}
        )


def test_smoke_corpus_adapter_preserves_fail_closed_intended_use():
    fixture = actor_interaction_fixture()
    internal = SmokeCorpusManifest(
        corpusId="synthetic-actor",
        version="1.0",
        adapter="socialgraph_gfm.fixtures",
        sourceHash=fixture.ref.content_hash,
        snapshotRefs=(fixture.ref,),
    )
    public = internal_smoke_corpus_to_public(internal)
    assert public.intended_use == "synthetic_test_only"
    assert public.split == "synthetic"


def test_public_adapter_requires_qualified_allowlist_for_duplicate_attributes():
    public = internal_snapshot_to_public(actor_interaction_fixture())
    activity = public.feature_manifest[0]
    duplicate = public.model_copy(
        update={
            "feature_manifest": [
                activity.model_copy(update={"id": "actor.activity"}),
                activity.model_copy(update={"id": "artifact.activity"}),
            ]
        }
    )
    owners = {"actor.activity": "actor", "artifact.activity": "artifact"}
    with pytest.raises(ContractViolation, match="ambiguous"):
        public_snapshot_to_internal_ref(
            duplicate,
            feature_owner_types=owners,
            inference_property_allowlist=("activity",),
        )
    internal = public_snapshot_to_internal_ref(
        duplicate,
        feature_owner_types=owners,
        inference_property_allowlist=("actor.activity",),
    )
    assert internal.resolved_inference_feature_keys() == ("actor.activity",)
