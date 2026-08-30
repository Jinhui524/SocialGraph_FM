from datetime import UTC, datetime

from socialgraph_gfm.contracts import EdgeRecord, GraphSnapshot, NodeRecord
from socialgraph_gfm.fixtures import activity_hetero_fixture, actor_interaction_fixture
from socialgraph_gfm.profiles import check_compatibility


def replace_snapshot(snapshot, *, nodes=None, edges=None):
    return GraphSnapshot.model_construct(
        ref=snapshot.ref,
        nodes=nodes or snapshot.nodes,
        edges=edges or snapshot.edges,
    )


def test_both_canonical_profiles_are_compatible():
    assert check_compatibility(actor_interaction_fixture()).compatible
    assert check_compatibility(activity_hetero_fixture()).compatible


def test_unknown_node_type_and_relation_are_blockers():
    snapshot = actor_interaction_fixture()
    nodes = (NodeRecord(nodeId="actor:ada", nodeType="account"),) + snapshot.nodes[1:]
    edges = (
        EdgeRecord(
            edgeId="bad", source="actor:ada", target="actor:bo", relation="follows",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )
    report = check_compatibility(replace_snapshot(snapshot, nodes=nodes, edges=edges))
    assert not report.compatible
    assert {issue.code for issue in report.blockers} == {"UNKNOWN_NODE_TYPE", "AMBIGUOUS_RELATION"}


def test_event_relation_requires_timestamp():
    snapshot = activity_hetero_fixture()
    edges = list(snapshot.edges)
    edges[0] = edges[0].model_copy(update={"timestamp": None})
    report = check_compatibility(replace_snapshot(snapshot, edges=tuple(edges)))
    assert not report.compatible
    assert "EVENT_TIMESTAMP_REQUIRED" in {issue.code for issue in report.blockers}


def test_actor_interaction_relations_require_timestamp():
    snapshot = actor_interaction_fixture()
    edges = list(snapshot.edges)
    edges[0] = edges[0].model_copy(update={"timestamp": None})
    report = check_compatibility(replace_snapshot(snapshot, edges=tuple(edges)))
    assert not report.compatible
    assert "EVENT_TIMESTAMP_REQUIRED" in {issue.code for issue in report.blockers}


def test_relation_endpoint_types_are_validated():
    snapshot = activity_hetero_fixture()
    edges = list(snapshot.edges)
    edges[1] = edges[1].model_copy(update={"target": "community:core"})
    report = check_compatibility(replace_snapshot(snapshot, edges=tuple(edges)))
    assert "INVALID_RELATION_ENDPOINTS" in {issue.code for issue in report.blockers}


def test_stable_relation_with_timestamp_is_warning_not_blocker():
    snapshot = activity_hetero_fixture()
    edges = list(snapshot.edges)
    edges[2] = edges[2].model_copy(update={"timestamp": datetime(2026, 1, 1, tzinfo=UTC)})
    report = check_compatibility(replace_snapshot(snapshot, edges=tuple(edges)))
    assert report.compatible
    assert "STABLE_RELATION_EVENT_TIME_IGNORED" in {issue.code for issue in report.warnings}


def test_empty_graph_and_missing_actor_are_explicit_blockers():
    snapshot = actor_interaction_fixture()
    empty = GraphSnapshot.model_construct(ref=snapshot.ref, nodes=(), edges=())
    empty_codes = {issue.code for issue in check_compatibility(empty).blockers}
    assert {"EMPTY_GRAPH", "ACTOR_TYPE_REQUIRED", "NO_RELATION_FACTS"} <= empty_codes

    hetero = activity_hetero_fixture()
    nodes = tuple(node for node in hetero.nodes if node.node_type != "actor")
    ids = {node.node_id for node in nodes}
    edges = tuple(edge for edge in hetero.edges if edge.source in ids and edge.target in ids)
    report = check_compatibility(
        GraphSnapshot.model_construct(ref=hetero.ref, nodes=nodes, edges=edges)
    )
    assert "ACTOR_TYPE_REQUIRED" in {issue.code for issue in report.blockers}
