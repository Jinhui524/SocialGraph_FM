from datetime import UTC, datetime

import pytest

from pydantic import ValidationError

from socialgraph_gfm.contracts import FeatureTransformArtifact, GraphSnapshot, NodeRecord
from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.fixtures import (
    activity_hetero_fixture,
    actor_interaction_fixture,
    smoke_fit_node_ids,
)
from socialgraph_gfm.materialize import materialize as _materialize
from socialgraph_gfm.runtime import runtime_report

CPU_AVAILABLE = runtime_report("cpu")["runtimeReady"]
CUDA_AVAILABLE = runtime_report("cuda")["runtimeReady"]
ML_AVAILABLE = CPU_AVAILABLE or CUDA_AVAILABLE
TEST_DEVICE = "cpu" if CPU_AVAILABLE else "cuda"
pytestmark = pytest.mark.skipif(not ML_AVAILABLE, reason="exact Torch/PyG/OGB runtime is not installed")


def materialize(*args, **kwargs):
    kwargs.setdefault("device", TEST_DEVICE)
    return _materialize(*args, **kwargs)


def test_actor_fixture_materializes_deterministically_with_missing_mask():
    first = materialize(
        actor_interaction_fixture(),
        purpose="training_smoke",
        fit_node_ids=smoke_fit_node_ids("actor"),
    )
    second = materialize(
        actor_interaction_fixture(),
        purpose="training_smoke",
        fit_node_ids=smoke_fit_node_ids("actor"),
    )
    assert first.manifest["materializationHash"] == second.manifest["materializationHash"]
    assert first.graph.x.shape == (4, 2)
    assert bool(first.graph.missing_mask.any())
    assert first.events.edge_ids == ("e:1", "e:2", "e:3", "e:4", "e:5")


def test_hetero_fixture_materializes_all_node_and_relation_types():
    value = materialize(
        activity_hetero_fixture(),
        purpose="training_smoke",
        fit_node_ids=smoke_fit_node_ids("hetero"),
    )
    assert set(value.graph.node_types) == {"actor", "artifact", "community", "topic"}
    assert len(value.graph.edge_types) == 5
    assert value.events.edge_ids == ("h:1", "h:2")
    assert value.manifest["fitSelection"]["actor"]["count"] == 1


def test_held_out_extreme_and_category_do_not_affect_fitted_transforms():
    snapshot = actor_interaction_fixture()
    fit = smoke_fit_node_ids("actor")
    value = materialize(snapshot, purpose="training_smoke", fit_node_ids=fit)
    states = {state.feature_key: state for state in value.transform_artifact.states}
    assert states["actor.activity"].mean == 5.5
    assert states["actor.activity"].scale == 2.5
    assert states["actor.role"].categories == ("maintainer", "newcomer")
    # actor:cy is held out with role=visitor, so it maps to UNKNOWN_INDEX=1.
    assert float(value.graph.x[2, 1]) == 1.0

    changed_nodes = tuple(
        node.model_copy(update={"features": {**node.features, "activity": 999999.0}})
        if node.node_id == "actor:di"
        else node
        for node in snapshot.nodes
    )
    provisional = snapshot.model_copy(update={"nodes": changed_nodes})
    changed_ref = snapshot.ref.model_copy(
        update={
            "content_hash": provisional.payload_hash(),
            "fact_hash": provisional.fact_payload_hash(),
        }
    )
    changed = GraphSnapshot(ref=changed_ref, nodes=changed_nodes, edges=snapshot.edges)
    changed_value = materialize(changed, purpose="training_smoke", fit_node_ids=fit)
    assert [state.logical_payload() for state in changed_value.transform_artifact.states] == [
        state.logical_payload() for state in value.transform_artifact.states
    ]
    assert changed_value.manifest["materializationHash"] != value.manifest["materializationHash"]


def test_training_materialization_requires_explicit_nonempty_heldout_fit():
    snapshot = actor_interaction_fixture()
    with pytest.raises(ContractViolation, match="requires explicit fit_node_ids"):
        materialize(snapshot, purpose="training_smoke")
    with pytest.raises(ContractViolation, match="must not be empty"):
        materialize(snapshot, purpose="training_smoke", fit_node_ids={"actor": ()})
    with pytest.raises(ContractViolation, match="held-out"):
        materialize(
            snapshot,
            purpose="training_smoke",
            fit_node_ids={"actor": tuple(node.node_id for node in snapshot.nodes)},
        )


def test_mutated_payload_with_stale_ref_hash_is_rejected():
    snapshot = actor_interaction_fixture()
    nodes = tuple(
        node.model_copy(update={"features": {**node.features, "activity": 8000.0}})
        if node.node_id == "actor:ada"
        else node
        for node in snapshot.nodes
    )
    # model_construct intentionally simulates an object mutated after initial validation.
    stale = GraphSnapshot.model_construct(ref=snapshot.ref, nodes=nodes, edges=snapshot.edges)
    with pytest.raises(ContractViolation, match="content hash mismatch"):
        materialize(
            stale,
            purpose="training_smoke",
            fit_node_ids=smoke_fit_node_ids("actor"),
        )


def test_inference_enforces_allowlist_and_requires_immutable_transforms():
    snapshot = actor_interaction_fixture()
    trained = materialize(
        snapshot,
        purpose="training_smoke",
        fit_node_ids=smoke_fit_node_ids("actor"),
    )
    inferred = materialize(
        snapshot,
        purpose="inference",
        transform_artifact=trained.transform_artifact,
        inference_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    assert inferred.manifest["selectedFeatureKeys"] == ["actor.activity", "actor.role"]
    assert inferred.manifest["fitScope"] == "immutable_prefit"
    with pytest.raises(ContractViolation, match="FeatureTransformArtifact"):
        materialize(
            snapshot,
            purpose="inference",
            transform_artifact=None,
            inference_at=datetime(2026, 2, 1, tzinfo=UTC),
        )


def test_inference_materializes_only_allowlisted_features():
    snapshot = actor_interaction_fixture()
    trained = materialize(
        snapshot,
        purpose="training_smoke",
        fit_node_ids=smoke_fit_node_ids("actor"),
    )
    restricted_ref = snapshot.ref.model_copy(
        update={"inference_property_allowlist": ("activity",)}
    )
    restricted = GraphSnapshot(ref=restricted_ref, nodes=snapshot.nodes, edges=snapshot.edges)
    inferred = materialize(
        restricted,
        purpose="inference",
        transform_artifact=trained.transform_artifact,
        inference_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    assert inferred.manifest["selectedFeatureKeys"] == ["actor.activity"]
    assert inferred.manifest["featureColumns"] == {"actor": ("activity",)}
    assert tuple(inferred.graph.x.shape) == (4, 1)


def test_transform_artifact_roundtrip_tamper_and_cross_snapshot_inference():
    snapshot = actor_interaction_fixture()
    trained = materialize(
        snapshot, purpose="training_smoke", fit_node_ids=smoke_fit_node_ids("actor")
    )
    encoded = trained.transform_artifact.model_dump_json(by_alias=True)
    restored = FeatureTransformArtifact.model_validate_json(encoded)
    assert restored == trained.transform_artifact

    tampered = restored.model_dump(mode="python", by_alias=True)
    tampered["states"][0]["mean"] = 999.0
    with pytest.raises(ValidationError, match="stateHash"):
        FeatureTransformArtifact.model_validate(tampered)

    nodes = tuple(
        node.model_copy(update={"features": {**node.features, "activity": 42.0}})
        if node.node_id == "actor:di"
        else node
        for node in snapshot.nodes
    )
    provisional = snapshot.model_copy(update={"nodes": nodes})
    target_ref = snapshot.ref.model_copy(
        update={
            "content_hash": provisional.payload_hash(),
            "fact_hash": provisional.fact_payload_hash(),
        }
    )
    target = GraphSnapshot(ref=target_ref, nodes=nodes, edges=snapshot.edges)
    inferred = materialize(
        target,
        purpose="inference",
        transform_artifact=restored,
        inference_at=datetime(2026, 2, 1, tzinfo=UTC),
    )
    assert inferred.transform_artifact.source_snapshot_payload_hash == snapshot.ref.content_hash
    assert target.ref.content_hash != snapshot.ref.content_hash

    incompatible_features = tuple(
        feature.model_copy(update={"dtype": "float64"}) if feature.name == "activity" else feature
        for feature in target.ref.feature_manifest
    )
    incompatible_provisional_ref = target.ref.model_copy(update={"feature_manifest": incompatible_features})
    incompatible_provisional = GraphSnapshot.model_construct(
        ref=incompatible_provisional_ref, nodes=target.nodes, edges=target.edges
    )
    incompatible_ref = incompatible_provisional_ref.model_copy(
        update={"content_hash": incompatible_provisional.payload_hash()}
    )
    incompatible = GraphSnapshot(ref=incompatible_ref, nodes=target.nodes, edges=target.edges)
    with pytest.raises(ContractViolation, match="incompatible"):
        materialize(
            incompatible,
            purpose="inference",
            transform_artifact=restored,
            inference_at=datetime(2026, 2, 1, tzinfo=UTC),
        )


def test_stable_relation_timestamp_is_warning_not_event_tensor():
    snapshot = activity_hetero_fixture()
    edges = tuple(
        edge.model_copy(update={"timestamp": datetime(2026, 2, 3, tzinfo=UTC)})
        if edge.edge_id == "h:3"
        else edge
        for edge in snapshot.edges
    )
    provisional = snapshot.model_copy(update={"edges": edges})
    ref = snapshot.ref.model_copy(
        update={
            "content_hash": provisional.payload_hash(),
            "fact_hash": provisional.fact_payload_hash(),
        }
    )
    value = materialize(
        GraphSnapshot(ref=ref, nodes=snapshot.nodes, edges=edges),
        purpose="training_smoke",
        fit_node_ids=smoke_fit_node_ids("hetero"),
    )
    assert value.events.edge_ids == ("h:1", "h:2")
    assert value.manifest["eventCount"] == 2


def test_inference_rejects_feature_unavailable_at_requested_time():
    snapshot = actor_interaction_fixture()
    features = tuple(
        feature.model_copy(update={"available_at": datetime(2027, 1, 1, tzinfo=UTC)})
        if feature.name == "activity"
        else feature
        for feature in snapshot.ref.feature_manifest
    )
    provisional_ref = snapshot.ref.model_copy(update={"feature_manifest": features})
    provisional = GraphSnapshot.model_construct(
        ref=provisional_ref, nodes=snapshot.nodes, edges=snapshot.edges
    )
    ref = provisional_ref.model_copy(update={"content_hash": provisional.payload_hash()})
    future = GraphSnapshot(ref=ref, nodes=snapshot.nodes, edges=snapshot.edges)
    with pytest.raises(ContractViolation, match="not available"):
        materialize(
            future,
            purpose="inference",
            transform_artifact=materialize(
                snapshot,
                purpose="training_smoke",
                fit_node_ids=smoke_fit_node_ids("actor"),
            ).transform_artifact,
            inference_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_undeclared_feature_and_wrong_dtype_fail_closed():
    snapshot = actor_interaction_fixture()
    nodes = list(snapshot.nodes)
    nodes[0] = NodeRecord(
        nodeId=nodes[0].node_id,
        nodeType=nodes[0].node_type,
        features={**nodes[0].features, "undeclared": "secret"},
    )
    provisional = snapshot.model_copy(update={"nodes": tuple(nodes)})
    ref = snapshot.ref.model_copy(
        update={
            "content_hash": provisional.payload_hash(),
            "fact_hash": provisional.fact_payload_hash(),
        }
    )
    with pytest.raises(ContractViolation, match="undeclared features"):
        materialize(
            GraphSnapshot(ref=ref, nodes=tuple(nodes), edges=snapshot.edges),
            purpose="training_smoke",
            fit_node_ids=smoke_fit_node_ids("actor"),
        )

    wrong_nodes = tuple(
        node.model_copy(update={"features": {**node.features, "activity": "many"}})
        if node.node_id == "actor:ada"
        else node
        for node in snapshot.nodes
    )
    provisional = snapshot.model_copy(update={"nodes": wrong_nodes})
    wrong_ref = snapshot.ref.model_copy(
        update={
            "content_hash": provisional.payload_hash(),
            "fact_hash": provisional.fact_payload_hash(),
        }
    )
    with pytest.raises(ContractViolation, match="requires numeric values"):
        materialize(
            GraphSnapshot(ref=wrong_ref, nodes=wrong_nodes, edges=snapshot.edges),
            purpose="training_smoke",
            fit_node_ids=smoke_fit_node_ids("actor"),
        )


@pytest.mark.skipif(
    not (CPU_AVAILABLE and CUDA_AVAILABLE),
    reason="both exact CPU and CUDA profiles are required",
)
def test_cpu_and_cuda_have_identical_logical_materialization_and_tensors():
    import torch

    snapshot = actor_interaction_fixture()
    fit = smoke_fit_node_ids("actor")
    cpu = materialize(snapshot, purpose="training_smoke", fit_node_ids=fit, device="cpu")
    cuda = materialize(snapshot, purpose="training_smoke", fit_node_ids=fit, device="cuda")
    assert cpu.manifest["executionDevice"] == "cpu"
    assert cuda.manifest["executionDevice"] == "cuda"
    assert cpu.manifest["materializationHash"] == cuda.manifest["materializationHash"]
    assert cpu.manifest["tensorDigests"] == cuda.manifest["tensorDigests"]
    assert cpu.transform_artifact == cuda.transform_artifact
    assert torch.equal(cpu.graph.x.cpu(), cuda.graph.x.cpu())
    assert torch.equal(cpu.graph.missing_mask.cpu(), cuda.graph.missing_mask.cpu())
    assert torch.equal(cpu.graph.edge_index.cpu(), cuda.graph.edge_index.cpu())
