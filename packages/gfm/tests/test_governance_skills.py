from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_governance_materialize import TINY_ARTIFACT_ID, TINY_DATASET_HASH, TINY_GRAPH_HASH

import socialgraph_gfm.governance.reviewed_cases as reviewed_cases_module
from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.governance.bundle import create_tiny_contract_bundle
from socialgraph_gfm.governance.knowledge import (
    KnowledgeSource,
    build_knowledge_index,
)
from socialgraph_gfm.governance.materialize import (
    OnlineInferenceData,
    load_materialized_artifact,
    materialize_bundle,
)
from socialgraph_gfm.governance.reviewed_cases import CaseKindEntry
from socialgraph_gfm.governance.skills import PUBLIC_SKILLS, GovernanceSkillExecutor


class _FakeRuntime:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "runtime"
        self.global_model_root = tmp_path / "v1"
        bundle = create_tiny_contract_bundle(tmp_path / "tiny.zip")
        incoming = self.root / "incoming" / TINY_ARTIFACT_ID
        incoming.mkdir(parents=True)
        shutil.copyfile(bundle, incoming / "bundle.zip")
        artifact = materialize_bundle(
            self.root,
            TINY_ARTIFACT_ID,
            expected_dataset_content_hash=TINY_DATASET_HASH,
            expected_graph_version_hash=TINY_GRAPH_HASH,
            clean_self_loops=False,
        )
        self.data = load_materialized_artifact(artifact.root)
        self._model = SimpleNamespace(
            model_version_id="socialgraph-fm-global/test",
            model_version_hash="1" * 64,
            model_state_hash="2" * 64,
            threshold=0.6,
            temperature=1.5,
            bias=-0.2,
            reference_metrics={"macroF1": 0.9},
            runtime_recipe_hash="3" * 64,
        )
        self.run_id = "governance-" + "4" * 32
        self.result_hash = "5" * 64
        scores = np.asarray([0.95, 0.8, 0.65, 0.5, 0.2, 0.1], dtype=np.float32)
        self.outputs = {
            "scores": scores,
            "logits": scores,
            "embeddings": np.arange(6 * 256, dtype=np.float32).reshape(6, 256) / 1000,
            "router_indices": np.ones((6, 2), dtype=np.int16),
            "router_weights": np.full((6, 2), 0.5, dtype=np.float32),
            "modality_contributions": np.full((6, 2), 0.5, dtype=np.float32),
            "modality_counts": np.arange(30, dtype=np.int32).reshape(6, 5),
            "rank_order": np.arange(6, dtype=np.int32),
            "ranks": np.arange(1, 7, dtype=np.int32),
            "community_ids": np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32),
        }
        edge_index = np.asarray(self.data.edge_index)
        pairs = [
            (int(source), int(target))
            for source, target in zip(edge_index[0], edge_index[1], strict=True)
            if int(source) < int(target)
        ]
        self.relations = {
            "source": np.asarray([pair[0] for pair in pairs], dtype=np.int32),
            "target": np.asarray([pair[1] for pair in pairs], dtype=np.int32),
            "modality_mask": np.ones(len(pairs), dtype=np.uint8),
            "priority": np.linspace(1, 0, len(pairs), dtype=np.float32),
            "endpoint_risk": np.full(len(pairs), 0.5, dtype=np.float32),
            "weight_percentile": np.full(len(pairs), 0.5, dtype=np.float32),
            "order": np.arange(len(pairs), dtype=np.int32),
        }
        self.groups = [
            {
                "groupId": "group-1",
                "memberCount": 3,
                "memberNodeIds": ["synthetic:0", "synthetic:1", "synthetic:2"],
                "averageRisk": 0.8,
                "p90Risk": 0.95,
                "priority": 0.89,
                "relationCounts": {
                    "coRT": 1,
                    "coURL": 1,
                    "hashSeq": 0,
                    "fastRT": 0,
                    "tweetSim": 0,
                },
                "derivation": "test",
                "rank": 1,
            },
            {
                "groupId": "group-2",
                "memberCount": 3,
                "memberNodeIds": ["synthetic:3", "synthetic:4", "synthetic:5"],
                "averageRisk": 0.3,
                "p90Risk": 0.5,
                "priority": 0.42,
                "relationCounts": {
                    "coRT": 1,
                    "coURL": 0,
                    "hashSeq": 1,
                    "fastRT": 1,
                    "tweetSim": 1,
                },
                "derivation": "test",
                "rank": 2,
            },
        ]
        self.links = [
            {
                "linkId": "link-0-3",
                "id": "link-0-3",
                "kind": "potential_link",
                "priority": 0.61,
                "source": "synthetic:0",
                "target": "synthetic:3",
                "nodeIds": ["synthetic:0", "synthetic:3"],
                "modalities": [],
                "factual": False,
                "scoreComponents": {"cosineSimilarity": 0.82, "jaccard": 0.25},
                "evidenceRole": "potentialLeadNotFactualEdge",
                "limitation": "Similarity lead only; not a factual edge.",
            }
        ]
        registry_logical = {
            "schemaVersion": "socialgraph-fm.global-model-registry/1.0",
            "state": "servingReady",
        }
        registry = {**registry_logical, "registryHash": canonical_sha256(registry_logical)}
        (self.global_model_root / "registry").mkdir(parents=True)
        (self.global_model_root / "registry" / "socialgraph-global.json").write_text(
            json.dumps(registry), encoding="utf-8"
        )
        (self.global_model_root / "corpus").mkdir(parents=True)
        (self.global_model_root / "corpus" / "manifest.json").write_text(
            json.dumps({"contentHash": "6" * 64}), encoding="utf-8"
        )

    def _artifact(self, artifact_id: str) -> OnlineInferenceData:
        assert artifact_id == TINY_ARTIFACT_ID
        return self.data

    def result(self, run_id: str) -> dict[str, Any]:
        assert run_id == self.run_id
        findings = [
            {
                "nodeId": self.data.node_ids[index],
                "score": float(self.outputs["scores"][index]),
                "riskBand": (
                    "high" if float(self.outputs["scores"][index]) >= self._model.threshold else "low"
                ),
            }
            for index in self.outputs["rank_order"]
        ]
        return {
            "runId": run_id,
            "artifactId": TINY_ARTIFACT_ID,
            "datasetContentHash": TINY_DATASET_HASH,
            "graphVersionHash": TINY_GRAPH_HASH,
            "modelVersionId": self._model.model_version_id,
            "modelStateHash": self._model.model_state_hash,
            "resultHash": self.result_hash,
            "distribution": {
                "low": 3,
                "review": 0,
                "high": 3,
                "predictedPositive": 3,
                "total": 6,
            },
            "findings": findings,
        }

    def evidence(self, run_id: str, node_id: str) -> dict[str, Any]:
        assert run_id == self.run_id and node_id in self.data.node_ids
        logical = {"runId": run_id, "nodeId": node_id}
        return {**logical, "evidenceHash": canonical_sha256(logical)}

    def derivations(self, run_id: str, kind: str, *, offset: int, limit: int) -> dict[str, Any]:
        assert run_id == self.run_id and kind in {"groups", "links"}
        items = self.groups if kind == "groups" else self.links
        return {
            "runId": run_id,
            "items": items[offset : offset + limit],
            "total": len(items),
            "offset": offset,
            "limit": limit,
        }

    def _outputs(self, run_id: str) -> dict[str, np.ndarray]:
        assert run_id == self.run_id
        return self.outputs

    def _analytics_arrays(self, run_id: str) -> dict[str, np.ndarray]:
        assert run_id == self.run_id
        return self.relations

    def _analytics_document(self, run_id: str) -> dict[str, Any]:
        assert run_id == self.run_id
        return {"groups": self.groups, "links": self.links}

    def _relation_derivation(
        self, data: OnlineInferenceData, arrays: dict[str, np.ndarray], index: int
    ) -> dict[str, Any]:
        source = int(arrays["source"][index])
        target = int(arrays["target"][index])
        return {
            "id": f"relation-{source}-{target}",
            "kind": "factual_relation",
            "nodeIds": [data.node_ids[source], data.node_ids[target]],
            "modalities": ["coRT"],
            "priority": float(arrays["priority"][index]),
            "factual": True,
        }

    def _finding(
        self,
        data: OnlineInferenceData,
        arrays: dict[str, np.ndarray],
        index: int,
        *,
        rank: int | None = None,
        community_sizes: np.ndarray | None = None,
    ) -> dict[str, Any]:
        del rank, community_sizes
        score = float(arrays["scores"][index])
        return {
            "nodeId": data.node_ids[index],
            "score": score,
            "riskBand": "high" if score >= self._model.threshold else "low",
        }


def _request(runtime: _FakeRuntime, command: str, params: dict[str, object], *, ordinal: int = 1):
    return {
        "schemaVersion": "socialgraph-fm.governance-command/1.0",
        "commandId": f"governance-command-{ordinal:032x}",
        "command": command,
        "graph": {
            "artifactId": TINY_ARTIFACT_ID,
            "datasetContentHash": TINY_DATASET_HASH,
            "graphVersionHash": TINY_GRAPH_HASH,
        },
        "model": {
            "modelVersionId": runtime._model.model_version_id,
            "modelStateHash": runtime._model.model_state_hash,
        },
        "params": params,
    }


def _case_params(
    runtime: _FakeRuntime,
    case_id: str,
    kind_entries: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "caseId": case_id,
        "caseHash": "a" * 64,
        "runId": runtime.run_id,
        "resultHash": runtime.result_hash,
        "kindEntries": kind_entries,
        "concludedAt": "2026-08-18T00:00:00.000000Z",
        "reviewHash": "b" * 64,
        "reviewStatus": "concluded",
    }


def test_public_skill_registry_and_read_only_commands(tmp_path: Path) -> None:
    runtime = _FakeRuntime(tmp_path)
    executor = GovernanceSkillExecutor(runtime)
    assert (
        executor.skill_names
        == PUBLIC_SKILLS
        == (
            "inspect_graph",
            "run_governance_analysis",
            "get_evidence_subgraph",
            "discover_coordination_groups",
            "rank_coordination_relations",
            "retrieve_similar_cases",
            "get_model_dataset_cards",
            "draft_review_report",
        )
    )
    inspection = executor.execute(_request(runtime, "inspect_graph", {}, ordinal=1))
    assert inspection.result["nodeCount"] == 6
    assert inspection.result["modalities"] == [
        modality
        for modality, count in inspection.result["relationCounts"].items()
        if count > 0
    ]
    assert inspection.graph.artifact_id == TINY_ARTIFACT_ID
    scoped = executor.execute(
        _request(
            runtime,
            "inspect_graph",
            {"scopeNodeIds": ["synthetic:0", "synthetic:1"]},
            ordinal=6,
        )
    )
    assert scoped.result["fusedEdgeCount"] == 1
    assert scoped.result["modalities"] == ["coRT"]
    assert scoped.result["relationCounts"] == {
        "coRT": 1,
        "coURL": 0,
        "hashSeq": 0,
        "fastRT": 0,
        "tweetSim": 0,
    }
    run_inspection = executor.execute(
        _request(
            runtime,
            "inspect_graph",
            {"runId": runtime.run_id, "candidateLimit": 5},
            ordinal=7,
        )
    )
    assert run_inspection.result["distribution"]["total"] == 6
    assert len(run_inspection.result["topCandidates"]) == 5
    assert set(run_inspection.result["topCandidates"][0]) == {
        "nodeId",
        "score",
        "riskBand",
    }
    assert all(
        forbidden not in json.dumps(run_inspection.result)
        for forbidden in ("embedding", "logit", "routes", "modalityContribution")
    )
    try:
        executor.execute(
            _request(
                runtime,
                "inspect_graph",
                {"runId": runtime.run_id, "candidateLimit": 6},
                ordinal=8,
            )
        )
    except ValueError:
        pass
    else:
        raise AssertionError("candidateLimit above five was accepted")
    plan = executor.execute(
        _request(runtime, "run_governance_analysis", {"protocol": "global", "topK": 4}, ordinal=2)
    )
    assert plan.status == "confirmation_required"
    assert plan.result["confirmationPlan"]["requiresConfirmation"] is True
    evidence = executor.execute(
        _request(
            runtime,
            "get_evidence_subgraph",
            {"runId": runtime.run_id, "nodeId": "synthetic:0"},
            ordinal=3,
        )
    )
    assert evidence.result["nodeId"] == "synthetic:0"
    groups = executor.execute(
        _request(
            runtime,
            "discover_coordination_groups",
            {"runId": runtime.run_id, "offset": 0, "limit": 10},
            ordinal=4,
        )
    )
    assert groups.result["total"] == 2
    relations = executor.execute(
        _request(
            runtime,
            "rank_coordination_relations",
            {"runId": runtime.run_id, "offset": 0, "limit": 10, "modalities": ["coRT"]},
            ordinal=5,
        )
    )
    assert relations.result["items"]
    assert relations.result["relationKind"] == "factual"
    assert all(item["kind"] == "factual_relation" and item["factual"] for item in relations.result["items"])
    potential = executor.execute(
        _request(
            runtime,
            "rank_coordination_relations",
            {
                "runId": runtime.run_id,
                "offset": 0,
                "limit": 10,
                "relationKind": "potential",
                "modalities": [],
            },
            ordinal=9,
        )
    )
    assert potential.result["relationKind"] == "potential"
    assert potential.result["modalities"] == []
    assert all(
        item["kind"] == "potential_link" and item["factual"] is False
        for item in potential.result["items"]
    )
    with pytest.raises(ValueError):
        executor.execute(
            _request(
                runtime,
                "rank_coordination_relations",
                {
                    "runId": runtime.run_id,
                    "relationKind": "potential",
                    "modalities": ["coRT"],
                },
                ordinal=10,
            )
        )
    cards = executor.execute(_request(runtime, "get_model_dataset_cards", {}, ordinal=6))
    assert set(cards.result) == {"modelCard", "datasetCard", "inputContractCard", "cardHash"}
    assert cards.result["inputContractCard"]["labelsSplitsScoresAccepted"] is False


def test_index_case_resolves_node_relation_group_and_mixed_targets(tmp_path: Path) -> None:
    runtime = _FakeRuntime(tmp_path)
    executor = GovernanceSkillExecutor(runtime)
    relation_id = f"relation-{runtime.relations['source'][0]}-{runtime.relations['target'][0]}"
    entries = [
        [{"kind": "node", "targetIds": ["synthetic:0"]}],
        [{"kind": "relation", "targetIds": [relation_id]}],
        [{"kind": "relation", "targetIds": ["link-0-3"]}],
        [{"kind": "group", "targetIds": ["group-1"]}],
        [
            {"kind": "node", "targetIds": ["synthetic:0"]},
            {"kind": "relation", "targetIds": [relation_id]},
            {"kind": "group", "targetIds": ["group-1"]},
        ],
    ]
    expected_keys = ["node", "relation", "relation", "group", "node+relation+group"]
    for ordinal, (kind_entries, expected) in enumerate(
        zip(entries, expected_keys, strict=True), 10
    ):
        case_id = f"case-{ordinal}"
        response = executor.execute(
            _request(
                runtime,
                "index_case",
                _case_params(runtime, case_id, kind_entries),
                ordinal=ordinal,
            )
        )
        assert response.result["idempotent"] is False
        record = executor._cases.record(case_id)
        assert record.kind_key == expected
        assert record.kind_entries == tuple(
            CaseKindEntry.model_validate(item) for item in kind_entries
        )
    repeated = executor.execute(
        _request(
            runtime,
            "index_case",
            _case_params(runtime, "case-10", entries[0]),
            ordinal=99,
        )
    )
    assert repeated.result["idempotent"] is True


def test_similar_cases_draft_and_knowledge_search_are_hash_bound(tmp_path: Path) -> None:
    runtime = _FakeRuntime(tmp_path)
    source = tmp_path / "guide.md"
    source.write_text(
        "Reviewed coordination evidence requires human analysis. "
        "Model scope and limitations must be checked before governance review.",
        encoding="utf-8",
    )
    build_knowledge_index(
        runtime.root / "knowledge",
        (KnowledgeSource("guide", source, "repo://guide"),),
    )
    executor = GovernanceSkillExecutor(runtime)
    entries = [{"kind": "node", "targetIds": ["synthetic:0"]}]
    for ordinal, case_id in ((20, "case-a"), (21, "case-b")):
        executor.execute(
            _request(
                runtime,
                "index_case",
                _case_params(runtime, case_id, entries),
                ordinal=ordinal,
            )
        )
    similar = executor.execute(
        _request(
            runtime,
            "retrieve_similar_cases",
            {"caseId": "case-a", "limit": 5},
            ordinal=22,
        )
    )
    assert [item["caseId"] for item in similar.result["items"]] == ["case-b"]
    assert similar.result["weights"] == {"embedding": 0.7, "structure": 0.2, "modality": 0.1}
    current_target = executor.execute(
        _request(
            runtime,
            "retrieve_similar_cases",
            {"runId": runtime.run_id, "kindEntries": entries, "limit": 5},
            ordinal=221,
        )
    )
    assert current_target.result["query"] == {
        "runId": runtime.run_id,
        "kindKey": "node",
        "kindEntries": entries,
    }
    assert [item["caseId"] for item in current_target.result["items"]] == [
        "case-a",
        "case-b",
    ]
    draft_params = {
        "caseId": "case-a",
        "caseHash": "a" * 64,
        "runId": runtime.run_id,
        "resultHash": runtime.result_hash,
        "kindEntries": entries,
        "format": "json",
        "reviewHash": "b" * 64,
    }
    first = executor.execute(
        _request(runtime, "draft_review_report", draft_params, ordinal=23)
    ).result
    second = executor.execute(
        _request(runtime, "draft_review_report", draft_params, ordinal=24)
    ).result
    assert first["draftHash"] == second["draftHash"]
    search = executor.execute(
        _request(
            runtime,
            "search_knowledge",
            {"query": "coordination evidence", "limit": 5},
            ordinal=25,
        )
    )
    assert search.result["items"][0]["sourceLabel"] == "guide"
    assert str(source) not in json.dumps(search.result)
    chinese_search = executor.execute(
        _request(
            runtime,
            "search_knowledge",
            {"query": "请说明模型的适用范围与限制", "limit": 5},
            ordinal=26,
        )
    )
    assert chinese_search.result["items"][0]["sourceLabel"] == "guide"
    no_terms = executor.execute(
        _request(
            runtime,
            "search_knowledge",
            {"query": "麻烦把这里提到的东西给我完整清楚地讲明白一些" * 20, "limit": 5},
            ordinal=27,
        )
    )
    assert no_terms.result["items"] == []
    assert no_terms.result["searchHash"] == canonical_sha256(
        {key: value for key, value in no_terms.result.items() if key != "searchHash"}
    )
    zero_hits = executor.execute(
        _request(
            runtime,
            "search_knowledge",
            {"query": "nonexistentphrase", "limit": 5},
            ordinal=28,
        )
    )
    assert zero_hits.result["items"] == []


def test_skill_binding_and_kind_entry_contract_fail_closed(tmp_path: Path) -> None:
    runtime = _FakeRuntime(tmp_path)
    executor = GovernanceSkillExecutor(runtime)
    mismatched = _request(runtime, "inspect_graph", {}, ordinal=30)
    mismatched["graph"]["graphVersionHash"] = "f" * 64
    try:
        executor.execute(mismatched)
    except ValueError as error:
        assert "binding" in str(error)
    else:
        raise AssertionError("mismatched graph binding was accepted")
    noncanonical = _case_params(
        runtime,
        "case-bad",
        [
            {"kind": "group", "targetIds": ["group-1"]},
            {"kind": "node", "targetIds": ["synthetic:0"]},
        ],
    )
    try:
        executor.execute(_request(runtime, "index_case", noncanonical, ordinal=31))
    except ValueError as error:
        assert "canonical order" in str(error)
    else:
        raise AssertionError("noncanonical kindEntries were accepted")


def test_skill_executor_restart_recovers_interrupted_case_manifest_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(tmp_path)
    executor = GovernanceSkillExecutor(runtime)
    original_atomic_json = reviewed_cases_module._atomic_json

    def interrupt_next_manifest(path: Path, value: object) -> None:
        if (
            path == executor._cases.manifest
            and isinstance(value, dict)
            and value.get("recordHashes")
        ):
            raise KeyboardInterrupt("simulated process interruption")
        original_atomic_json(path, value)  # type: ignore[arg-type]

    params = _case_params(
        runtime,
        "case-interrupted",
        [{"kind": "node", "targetIds": ["synthetic:0"]}],
    )
    monkeypatch.setattr(reviewed_cases_module, "_atomic_json", interrupt_next_manifest)
    with pytest.raises(KeyboardInterrupt, match="process interruption"):
        executor.execute(_request(runtime, "index_case", params, ordinal=40))

    monkeypatch.setattr(reviewed_cases_module, "_atomic_json", original_atomic_json)
    restarted = GovernanceSkillExecutor(runtime)
    response = restarted.execute(_request(runtime, "index_case", params, ordinal=41))
    assert response.result["caseId"] == "case-interrupted"
    assert response.result["idempotent"] is True
    assert restarted._cases.record("case-interrupted").record_hash == response.result["recordHash"]
