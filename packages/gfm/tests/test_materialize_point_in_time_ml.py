from datetime import UTC, datetime

import pytest

from socialgraph_gfm.contracts import GraphSnapshot
from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.fixtures import (
    activity_hetero_fixture,
    actor_interaction_fixture,
    smoke_fit_node_ids,
)
from socialgraph_gfm.materialize import materialize
from socialgraph_gfm.runtime import runtime_report


CPU_AVAILABLE = runtime_report("cpu")["runtimeReady"]
pytestmark = pytest.mark.skipif(
    not CPU_AVAILABLE, reason="exact CPU Torch/PyG/OGB runtime is not installed"
)


def _rehash(snapshot: GraphSnapshot, *, edges: tuple) -> GraphSnapshot:
    provisional = snapshot.model_copy(update={"edges": edges})
    ref = snapshot.ref.model_copy(
        update={
            "content_hash": provisional.payload_hash(),
            "fact_hash": provisional.fact_payload_hash(),
        }
    )
    return GraphSnapshot(ref=ref, nodes=snapshot.nodes, edges=edges)


def test_inference_uses_one_visible_event_edge_set_for_graph_and_event_view():
    snapshot = actor_interaction_fixture()
    trained = materialize(
        snapshot,
        purpose="training_smoke",
        fit_node_ids=smoke_fit_node_ids("actor"),
        device="cpu",
    )
    inferred = materialize(
        snapshot,
        purpose="inference",
        transform_artifact=trained.transform_artifact,
        inference_at=datetime(2026, 1, 2, 12, tzinfo=UTC),
        device="cpu",
    )

    assert tuple(inferred.graph.edge_ids) == ("e:1", "e:2")
    assert inferred.events.edge_ids == ("e:1", "e:2")
    assert inferred.graph.edge_index.shape[1] == 2
    assert inferred.manifest["sourceEdgeCount"] == 5
    assert inferred.manifest["visibleEdgeCount"] == 2
    assert inferred.manifest["excludedFutureEdgeCount"] == 3


def test_heterogeneous_stable_edges_use_half_open_validity_interval():
    snapshot = activity_hetero_fixture()
    cutoff = datetime(2026, 2, 1, 12, tzinfo=UTC)
    edges = tuple(
        edge.model_copy(update={"valid_to": cutoff}) if edge.edge_id == "h:3" else edge
        for edge in snapshot.edges
    )
    target = _rehash(snapshot, edges=edges)
    trained = materialize(
        snapshot,
        purpose="training_smoke",
        fit_node_ids=smoke_fit_node_ids("hetero"),
        device="cpu",
    )
    inferred = materialize(
        target,
        purpose="inference",
        transform_artifact=trained.transform_artifact,
        inference_at=cutoff,
        device="cpu",
    )

    assert inferred.events.edge_ids == ("h:1",)
    assert inferred.graph.edge_types == [("actor", "actor_interacts_actor", "actor")]
    assert inferred.manifest["visibleEdgeCount"] == 1
    assert inferred.manifest["excludedFutureEdgeCount"] == 4


def test_formal_training_requires_observation_end_and_filters_future_edges():
    snapshot = actor_interaction_fixture()
    with pytest.raises(ContractViolation, match="observation_end"):
        materialize(
            snapshot,
            purpose="formal_training",
            fit_node_ids=smoke_fit_node_ids("actor"),
            device="cpu",
        )
    with pytest.raises(ContractViolation, match="timezone-aware"):
        materialize(
            snapshot,
            purpose="formal_training",
            fit_node_ids=smoke_fit_node_ids("actor"),
            observation_end=datetime(2026, 1, 2, 12),
            device="cpu",
        )

    value = materialize(
        snapshot,
        purpose="formal_training",
        fit_node_ids=smoke_fit_node_ids("actor"),
        observation_end=datetime(2026, 1, 2, 12, tzinfo=UTC),
        device="cpu",
    )
    assert tuple(value.graph.edge_ids) == ("e:1", "e:2")
    assert value.events.edge_ids == ("e:1", "e:2")
    assert value.manifest["fitScope"] == "explicit_train_split_nodes"
