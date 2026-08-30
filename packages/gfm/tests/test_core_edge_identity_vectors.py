from __future__ import annotations

import json
from pathlib import Path

from socialgraph_gfm.core.bundle import StaticEdge
from socialgraph_gfm.core.governance import RegisteredEdgeIdentity


def test_neutral_edge_identity_vectors_are_generated_by_the_live_gfm_contract() -> None:
    vectors = json.loads(
        (Path(__file__).parents[3] / "contracts" / "core-edge-identity-vectors.json").read_text(
            encoding="utf-8"
        )
    )
    assert vectors["schemaVersion"] == "socialgraph-fm.core-edge-identity-vectors/2.0"
    for case in vectors["cases"]:
        local = case["local"]
        source_id, target_id = local["source"], local["target"]
        if not case["directed"] and source_id > target_id:
            source_id, target_id = target_id, source_id
        edge = StaticEdge.model_validate(
            {
                "sourceId": source_id,
                "targetId": target_id,
                "edgeType": local["type"],
                "weight": local["weight"],
            }
        )
        actual = RegisteredEdgeIdentity.create(edge).model_dump(mode="json", by_alias=True)
        assert actual == case["registeredIdentity"], case["name"]
