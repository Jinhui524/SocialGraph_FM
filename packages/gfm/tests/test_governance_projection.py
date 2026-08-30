from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from test_governance_materialize import TINY_ARTIFACT_ID, TINY_DATASET_HASH, TINY_GRAPH_HASH

from socialgraph_gfm.governance.bundle import create_tiny_contract_bundle
from socialgraph_gfm.governance.materialize import load_materialized_artifact, materialize_bundle
from socialgraph_gfm.governance.projection import ProjectionRequest, select_projection


def _data(tmp_path: Path):
    bundle = create_tiny_contract_bundle(tmp_path / "tiny.zip")
    incoming = tmp_path / "runtime" / "incoming" / TINY_ARTIFACT_ID
    incoming.mkdir(parents=True)
    shutil.copyfile(bundle, incoming / "bundle.zip")
    artifact = materialize_bundle(
        tmp_path / "runtime",
        TINY_ARTIFACT_ID,
        expected_dataset_content_hash=TINY_DATASET_HASH,
        expected_graph_version_hash=TINY_GRAPH_HASH,
        clean_self_loops=False,
    )
    return load_materialized_artifact(artifact.root)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "scores": np.asarray([0.95, 0.8, 0.65, 0.5, 0.2, 0.1], dtype=np.float32),
        "rank_order": np.arange(6, dtype=np.int32),
        "community_ids": np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
    }


def test_overview_uses_deterministic_sources_and_hard_budgets(tmp_path: Path) -> None:
    data = _data(tmp_path)
    request = ProjectionRequest.model_validate(
        {"preset": "overview", "nodeBudget": 5, "edgeBudget": 4}
    )
    first = select_projection(data, request, arrays=_arrays(), threshold=0.6)
    second = select_projection(data, request, arrays=_arrays(), threshold=0.6)
    assert first == second
    assert len(first.selected_order) == 5
    assert len(first.edges) <= 4
    assert set(first.source_counts) >= {
        "groupRepresentatives",
        "bridgeEndpoints",
        "isolates",
        "highRisk",
        "reviewRisk",
        "lowRisk",
        "rankedFill",
    }
    unscored = select_projection(data, request, arrays=None)
    assert set(unscored.source_counts) >= {
        "componentRepresentatives",
        "bridgeEndpoints",
        "isolates",
        "highDegree",
        "midDegree",
        "lowDegree",
        "rankedFill",
    }


def test_overview_accepts_the_product_visible_ceiling_without_changing_small_defaults() -> None:
    dense = ProjectionRequest.model_validate(
        {"preset": "overview", "nodeBudget": 128, "edgeBudget": 12_000}
    )
    default = ProjectionRequest.model_validate({"preset": "overview"})
    assert (dense.effective_node_budget, dense.effective_edge_budget) == (128, 12_000)
    assert (default.effective_node_budget, default.effective_edge_budget) == (120, 240)


def test_relation_and_evidence_projections_are_bounded(tmp_path: Path) -> None:
    data = _data(tmp_path)
    relation = select_projection(
        data,
        ProjectionRequest.model_validate(
            {"preset": "relation", "relation": "coRT", "nodeBudget": 4, "edgeBudget": 2}
        ),
        arrays=_arrays(),
    )
    assert len(relation.selected_order) <= 4
    assert len(relation.edges) <= 2
    evidence = select_projection(
        data,
        ProjectionRequest.model_validate(
            {
                "preset": "evidence",
                "anchorNodeIds": ["synthetic:3"],
                "nodeBudget": 4,
                "edgeBudget": 3,
            }
        ),
        arrays=_arrays(),
    )
    assert evidence.selected_order[0] == 3
    assert len(evidence.selected_order) <= 4
    assert len(evidence.edges) <= 3


def test_groups_projection_returns_renderable_supernodes_and_aggregate_edges(
    tmp_path: Path,
) -> None:
    data = _data(tmp_path)
    groups = [
        {
            "groupId": f"group-{index + 1}",
            "memberCount": 2,
            "memberNodeIds": [f"synthetic:{index * 2}", f"synthetic:{index * 2 + 1}"],
            "averageRisk": 0.5 - index * 0.1,
            "p90Risk": 0.8 - index * 0.1,
            "priority": 0.7 - index * 0.1,
            "relationCounts": {
                name: 0 for name in ("coRT", "coURL", "hashSeq", "fastRT", "tweetSim")
            },
        }
        for index in range(3)
    ]
    selection = select_projection(
        data,
        ProjectionRequest.model_validate({"preset": "groups", "groupBudget": 3}),
        arrays=_arrays(),
        groups=groups,
    )
    assert [item["id"] for item in selection.supernodes] == [
        "group-1",
        "group-2",
        "group-3",
    ]
    assert all(item["aggregate"] for item in selection.supernodes)
    assert selection.aggregate_edges
    assert all(item["count"] == item["weight"] for item in selection.aggregate_edges)


@pytest.mark.parametrize(
    "payload",
    [
        {"preset": "overview", "nodeBudget": 3_001},
        {"preset": "relation"},
        {"preset": "evidence", "edgeBudget": 121},
        {"preset": "groups", "groupBudget": 25},
    ],
)
def test_projection_contract_rejects_budget_or_preset_mismatches(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ProjectionRequest.model_validate(payload)
