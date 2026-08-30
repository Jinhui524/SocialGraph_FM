from __future__ import annotations

# ruff: noqa: E402 - optional Torch dependencies are gated before package imports.

import gzip
import hashlib
import io
import json
import math
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from socialgraph_gfm.canonical import canonical_sha256, file_sha256
from socialgraph_gfm.core.adapters import BundleInputAdapter, derive_training_selection
from socialgraph_gfm.core.bundle import calculate_graph_version_hash
from socialgraph_gfm.core.datasets.parsers import parse_wiki_rfa
from socialgraph_gfm.core.model import ResearchCoreGFM
from socialgraph_gfm.research.cli import main as research_cli_main
from socialgraph_gfm.research import workflow as research_workflow
from socialgraph_gfm.research.contracts import (
    ACCOUNT_RISK_TASK,
    COLLABORATION_TASK,
    CONTENT_POLICY_TASK,
    SIGNED_RELATION_TASK,
)
from socialgraph_gfm.research.workflow import (
    _average_precision,
    _bundle_edge_index,
    _calibration_is_adequate,
    _comparison_cells,
    _comparison_claim_gate,
    _email_filtered_rankings,
    _load_exported_runtime,
    _load_graph_documents,
    _load_trained_runtime,
    _pretrain_validation_loss,
    load_comparison_manifest,
    load_corpus_manifest,
    load_registry,
    materialize_fixture_corpus,
    publish_research_model,
    readiness,
    train_research_comparison_matrix,
    train_research_model,
)
from socialgraph_gfm.research.routing import (
    SHARED_NULL_ROUTE,
    task_route_domain,
    task_route_name,
)
from socialgraph_gfm.core.structure_features import (
    StructureAlgorithmConfig,
    compute_structure_rows,
)


def _write_fixture_family(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    twitch_graphs = {}
    for language in ("DE", "EN", "ES", "FR", "PT", "RU"):
        prefix = language.lower()
        nodes = [
            {
                "id": f"{prefix}-{index:02d}",
                "features": [str(index % 5), str((index + 1) % 7)],
                "mature": index % 2,
            }
            for index in range(20)
        ]
        edges = [
            [nodes[index]["id"], nodes[(index + 1) % len(nodes)]["id"]]
            for index in range(len(nodes))
        ]
        edges.extend(
            [nodes[index]["id"], nodes[index + 2]["id"]] for index in range(0, len(nodes) - 2, 2)
        )
        twitch_graphs[language] = {"nodes": nodes, "edges": edges}
    (root / "twitch-language.json").write_text(
        json.dumps({"recipeId": "twitch-language", "graphs": twitch_graphs}),
        encoding="utf-8",
    )

    tolokers_nodes = [
        {"id": str(index), "features": [index / 20, index % 3], "banned": index % 2}
        for index in range(20)
    ]
    tolokers_edges = [[str(index), str((index + 1) % 20)] for index in range(20)]
    tolokers_edges.extend([[str(index), str(index + 2)] for index in range(18)])
    splits = []
    for fold in range(10):
        order = [(index + fold) % 20 for index in range(20)]
        splits.append({"train": order[:12], "validation": order[12:16], "test": order[16:]})
    (root / "tolokers.json").write_text(
        json.dumps(
            {
                "recipeId": "tolokers",
                "nodes": tolokers_nodes,
                "edges": tolokers_edges,
                "officialSplits": splits,
            }
        ),
        encoding="utf-8",
    )

    wiki_records = []
    for index in range(30):
        wiki_records.append(
            "\n".join(
                (
                    f"SRC:voter-{index:02d}",
                    f"TGT:candidate-{index:02d}",
                    f"VOT:{1 if index % 2 == 0 else -1}",
                    "RES:1",
                    "YEA:2013",
                    "DAT:00:00, 1 January 2013",
                    "TXT:fixture text excluded from model inputs",
                )
            )
        )
    (root / "wiki-rfa.txt").write_text("\n\n".join(wiki_records) + "\n", encoding="utf-8")

    email_nodes = [{"id": str(index), "department": str(index % 4)} for index in range(30)]
    email_edges = [[str(index), str((index + 1) % 30)] for index in range(30)]
    email_edges.extend([[str(index), str((index + 3) % 30)] for index in range(30)])
    (root / "email-eu-core.json").write_text(
        json.dumps({"recipeId": "email-eu-core", "nodes": email_nodes, "edges": email_edges}),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def fixture_files(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("research-fixtures")
    _write_fixture_family(root)
    return root


@pytest.fixture(scope="module")
def materialized_root(tmp_path_factory: pytest.TempPathFactory, fixture_files: Path) -> Path:
    root = tmp_path_factory.mktemp("research-materialized")
    materialize_fixture_corpus(root, fixture_files)
    return root


@pytest.fixture(scope="module")
def published_root(tmp_path_factory: pytest.TempPathFactory, fixture_files: Path) -> Path:
    root = tmp_path_factory.mktemp("research-published")
    assert (
        research_cli_main(
            [
                "materialize",
                "--research-root",
                str(root),
                "--fixture-root",
                str(fixture_files),
            ]
        )
        == 0
    )
    assert (
        research_cli_main(
            [
                "train",
                "--research-root",
                str(root),
                "--pretrain-epochs",
                "1",
                "--head-epochs",
                "1",
            ]
        )
        == 0
    )
    assert research_cli_main(["evaluate", "--research-root", str(root)]) == 0
    with pytest.raises(SystemExit) as export_rejected:
        research_cli_main(["export", "--research-root", str(root)])
    assert export_rejected.value.code == 2
    research_workflow.export_research_model(root, allow_test_fixture=True)
    with pytest.raises(SystemExit) as smoke_rejected:
        research_cli_main(["smoke", "--research-root", str(root)])
    assert smoke_rejected.value.code == 2
    research_workflow.smoke_research_export(root, allow_test_fixture=True)
    with pytest.raises(SystemExit) as publish_rejected:
        research_cli_main(["publish", "--research-root", str(root)])
    assert publish_rejected.value.code == 2
    publish_research_model(root, allow_test_fixture=True)
    return root


def _entry_documents(root: Path):
    manifest = load_corpus_manifest(root)
    return manifest, _load_graph_documents(root, manifest)


def test_materialization_excludes_labels_and_holds_out_wiki_email_topology(
    materialized_root: Path,
) -> None:
    manifest, documents = _entry_documents(materialized_root)
    assert manifest["graphCount"] == 9
    assert manifest["materializerVersion"] == research_workflow.MATERIALIZER_VERSION
    for entry in manifest["graphs"]:
        assert entry["parserId"]
        assert entry["parserVersion"] == "1.0.0"
        assert len(entry["parserCodeSha256"]) == 64
        assert documents[entry["graphId"]][0].split_manifest.strategy == entry["splitProtocol"]

    forbidden = {
        "twitch-DE": {"mature"},
        "tolokers": {"banned"},
        "wiki-rfa": {"TXT", "DAT", "YEA", "RES", "VOT"},
        "email-eu-core": {"department"},
    }
    for graph_id, names in forbidden.items():
        bundle = documents[graph_id][0]
        feature_names = {feature.name for feature in bundle.node_features}
        assert feature_names.isdisjoint(names)
    assert all(edge.weight == 1.0 for edge in documents["wiki-rfa"][0].edges)

    for graph_id in ("wiki-rfa", "email-eu-core"):
        bundle = documents[graph_id][0]
        selection = derive_training_selection(bundle)
        roles = {item.entity_id: item.role for item in bundle.split_manifest.assignments}
        visible_ids = {
            f"edge:{bundle.edges[index].source_id}:{bundle.edges[index].target_id}"
            for index in selection.visible_edge_indices
        }
        assert visible_ids
        assert all(roles[item] == "train" for item in visible_ids)
        assert any(role in {"validation", "test"} for role in roles.values())
        recomputed = compute_structure_rows(
            bundle,
            visible_edge_indices=selection.visible_edge_indices,
            config=StructureAlgorithmConfig.fixed(),
        )
        assert bundle.structural_features is not None
        assert torch.allclose(
            torch.tensor(recomputed),
            torch.tensor(bundle.structural_features.values),
            atol=1e-7,
            rtol=0,
        )

    for graph_id in (
        "twitch-DE",
        "twitch-EN",
        "twitch-ES",
        "twitch-FR",
        "twitch-PT",
        "twitch-RU",
        "tolokers",
    ):
        bundle = documents[graph_id][0]
        selection = derive_training_selection(bundle)
        entry = next(item for item in manifest["graphs"] if item["graphId"] == graph_id)
        assert selection.visible_edge_indices == tuple(range(len(bundle.edges)))
        assert entry["visibleTopologyEdgeCount"] == entry["edgeCount"]
        assert entry["visibleTopologyHash"] == selection.visible_topology_hash
    assert all(
        entry["visibleTopologyEdgeCount"] < entry["edgeCount"]
        for entry in manifest["graphs"]
        if entry["graphId"] in {"wiki-rfa", "email-eu-core"}
    )
    tolokers_selection = derive_training_selection(documents["tolokers"][0])
    tolokers_entry = next(
        item for item in manifest["graphs"] if item["graphId"] == "tolokers"
    )
    tolokers_folds = json.loads(
        (
            materialized_root
            / "materialized/corpus"
            / tolokers_entry["splitsPath"]
        ).read_text(encoding="utf-8")
    )["folds"]
    assert len(tolokers_folds) == 10
    assert {
        tolokers_selection.visible_topology_hash for _fold in tolokers_folds
    } == {tolokers_selection.visible_topology_hash}


def test_email_negative_roles_are_disjoint_true_edge_free_and_train_visible_hard(
    materialized_root: Path,
) -> None:
    _manifest, documents = _entry_documents(materialized_root)
    _bundle, labels, _entry = documents["email-eu-core"]
    partitions = labels["partitions"]
    true_edges = {
        tuple(sorted((item["sourceId"], item["targetId"])))
        for role in ("train", "validation", "test")
        for item in partitions[role]["positives"]
    }
    train_adjacency: dict[str, set[str]] = {}
    for item in partitions["train"]["positives"]:
        left, right = item["sourceId"], item["targetId"]
        train_adjacency.setdefault(left, set()).add(right)
        train_adjacency.setdefault(right, set()).add(left)
    observed: set[tuple[str, str]] = set()
    for role in ("train", "validation", "test"):
        counts = labels["negativeSampling"]["componentCounts"][role]
        negatives = partitions[role]["negatives"]
        assert counts["total"] == len(negatives)
        for item in negatives:
            pair = tuple(sorted((item["sourceId"], item["targetId"])))
            assert pair not in true_edges
            assert pair not in observed
            observed.add(pair)
            witness = train_adjacency.get(pair[0], set()) & train_adjacency.get(pair[1], set())
            if item["samplingComponent"] == "two-hop-hard":
                assert witness
            else:
                assert not witness
    assert labels["negativeSampling"]["roleDisjoint"] is True
    assert labels["negativeSampling"]["excludeAllTrueEdges"] is True


def test_wiki_parser_drops_self_votes(tmp_path: Path) -> None:
    path = tmp_path / "wiki.txt"
    path.write_text(
        (
            "SRC:A\nTGT:A\nVOT:1\nRES:1\nYEA:2013\nDAT:00:00\nTXT:self\n\n"
            "SRC:A\nTGT:B\nVOT:-1\nRES:-1\nYEA:2013\nDAT:00:01\nTXT:other\n"
        ),
        encoding="utf-8",
    )
    graph = parse_wiki_rfa(path)
    assert len(graph.signed_edges) == 1
    assert all(left != right for left, right, _sign in graph.signed_edges)


def test_research_heads_keep_email_symmetric_and_wiki_ordered() -> None:
    model = ResearchCoreGFM(domains=("fixture",))
    encoded = torch.zeros((2, 128))
    encoded[0, 0] = 2.0
    encoded[1, 0] = 1.0
    pairs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    collaboration = model.collaboration_head(encoded, pairs)
    assert torch.equal(collaboration[:1], collaboration[1:])
    with torch.no_grad():
        first = model.signed_edge_head.network[0]
        last = model.signed_edge_head.network[2]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 0] = 1.0
        first.weight[0, 128] = -1.0
        last.weight.zero_()
        last.bias.zero_()
        last.weight[0, 0] = 1.0
    directed = model.signed_edge_head(encoded, pairs)
    assert directed[0] != directed[1]


def test_collaboration_and_similarity_use_shared_null_route_with_nonzero_target_residual() -> None:
    domain = "email-eu-core"
    model = ResearchCoreGFM(domains=(domain,)).eval()
    key = model._domain_keys[domain]
    with torch.no_grad():
        model.domain_prompts[key].fill_(0.5)
        model.target_gates[key].fill_(8.0)
        model.target_adapters[key][-1].bias.fill_(1.0)
    features = torch.randn((5, 128), generator=torch.Generator().manual_seed(1729))
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
    )
    with torch.inference_mode():
        shared = model.encode_domain(features, edge_index, None)
        target = model.encode_domain(features, edge_index, domain)
    assert not torch.allclose(shared, target)
    assert task_route_domain(COLLABORATION_TASK, domain) is None
    assert task_route_name(COLLABORATION_TASK, domain) == SHARED_NULL_ROUTE
    assert task_route_name(SIGNED_RELATION_TASK, "wiki-rfa") == "domain:wiki-rfa"


def test_tie_group_average_precision_and_bidirectional_filtered_ranking_are_id_invariant() -> None:
    expected = _average_precision([1, 0], [0.5, 0.5])
    assert expected == pytest.approx(0.5)
    assert _average_precision([0, 1], [0.5, 0.5]) == pytest.approx(expected)

    positive_pairs = ((0, 4), (0, 5))
    visible = {(0, 1), (1, 2), (2, 3)}
    all_true = {*visible, *positive_pairs}
    calls: list[tuple[int, tuple[int, ...]]] = []

    def tied_scores(anchor: int, candidates: list[int]) -> list[float]:
        calls.append((anchor, tuple(candidates)))
        return [0.0] * len(candidates)

    groups = {0: "a", 1: "a", 2: "b", 3: "b", 4: "a", 5: "b"}
    first = _email_filtered_rankings(
        num_nodes=6,
        positive_pairs=positive_pairs,
        visible_train_pairs=visible,
        all_true_pairs=all_true,
        gfm_score_candidates=tied_scores,
        department_by_index=groups,
    )
    anchor_zero_candidates = [set(candidates) for anchor, candidates in calls if anchor == 0]
    assert len(anchor_zero_candidates) == 2
    assert all(len(candidates & {4, 5}) == 1 for candidates in anchor_zero_candidates)
    assert first["rankingExampleCount"] == 4
    assert first["directionPolicy"] == "both-endpoints/1.0"
    assert first["tiePolicy"] == "average-rank/1.0"
    assert {
        "common-neighbors-filtered-mrr",
        "common-neighbors-hits-at-10",
        "adamic-adar-filtered-mrr",
        "adamic-adar-hits-at-10",
    } <= set(first)

    permutation = {0: 3, 1: 5, 2: 0, 3: 4, 4: 2, 5: 1}
    def remap_pair(pair: tuple[int, int]) -> tuple[int, int]:
        return tuple(sorted((permutation[pair[0]], permutation[pair[1]])))
    second = _email_filtered_rankings(
        num_nodes=6,
        positive_pairs=tuple(remap_pair(pair) for pair in positive_pairs),
        visible_train_pairs={remap_pair(pair) for pair in visible},
        all_true_pairs={remap_pair(pair) for pair in all_true},
        gfm_score_candidates=lambda _anchor, candidates: [0.0] * len(candidates),
        department_by_index={permutation[index]: group for index, group in groups.items()},
    )
    compared = {
        "filtered-mrr",
        "hits-at-10",
        "common-neighbors-filtered-mrr",
        "common-neighbors-hits-at-10",
        "adamic-adar-filtered-mrr",
        "adamic-adar-hits-at-10",
    }
    assert {key: first[key] for key in compared} == {
        key: second[key] for key in compared
    }


def test_wiki_pretrain_validation_uses_bounded_negative_sampler(
    materialized_root: Path,
) -> None:
    _manifest, documents = _entry_documents(materialized_root)
    bundle, _labels, _entry = documents["wiki-rfa"]
    adapter = BundleInputAdapter(bundle, mode="training", multi_hot_buckets=256)
    model = ResearchCoreGFM(domains=("wiki-rfa",))
    calls: list[int] = []

    class FakePrepared:
        edge_index = _bundle_edge_index(bundle, visible_only=True)

        def sample_negative_pairs(self, count: int, *, generator):
            calls.append(count)
            true_pairs = {(edge.source_id, edge.target_id) for edge in bundle.edges}
            by_id = {node.id: node.index for node in bundle.nodes}
            selected = []
            for left in bundle.nodes:
                for right in bundle.nodes:
                    if left.id != right.id and (left.id, right.id) not in true_pairs:
                        selected.append((by_id[left.id], by_id[right.id]))
                        if len(selected) == count:
                            return torch.tensor(selected, dtype=torch.long)
            raise AssertionError("fixture lacks negative capacity")

    value = _pretrain_validation_loss(
        model=model,
        documents={"wiki-rfa": documents["wiki-rfa"]},
        graphs={"wiki-rfa": SimpleNamespace(graph=FakePrepared())},
        adapters={"wiki-rfa": adapter},
        device="cpu",
    )
    validation_count = sum(item.role == "validation" for item in bundle.split_manifest.assignments)
    assert calls == [validation_count]
    assert math.isfinite(value)


def test_tolokers_inventory_calibration_threshold_and_claim_gate(
    materialized_root: Path,
) -> None:
    _manifest, documents = _entry_documents(materialized_root)
    cells = _comparison_cells(materialized_root, documents)
    tolokers = [item for item in cells if item["taskId"] == ACCOUNT_RISK_TASK]
    twitch = [item for item in cells if item["taskId"] == CONTENT_POLICY_TASK]
    assert [item["fold"] for item in tolokers] == list(range(10))
    assert len(twitch) == 6
    assert _calibration_is_adequate(before_ece=0.25, after_ece=0.20, inadequacy_reason=None)
    assert not _calibration_is_adequate(before_ece=0.60, after_ece=0.50, inadequacy_reason=None)
    assert not _calibration_is_adequate(before_ece=0.10, after_ece=0.11, inadequacy_reason=None)
    aggregates = {
        task: {"sharedVsScratchDelta": delta}
        for task, delta in zip(
            (
                CONTENT_POLICY_TASK,
                ACCOUNT_RISK_TASK,
                SIGNED_RELATION_TASK,
                COLLABORATION_TASK,
            ),
            (0.1, 0.2, 0.3, -0.1),
            strict=True,
        )
    }
    assert _comparison_claim_gate(aggregates)["claimStatus"] == "observed_transfer_gain"
    aggregates[CONTENT_POLICY_TASK]["sharedVsScratchDelta"] = -1.0
    assert _comparison_claim_gate(aggregates)["claimStatus"] == "not_demonstrated"


def test_cli_six_stage_matrix_is_real_and_hash_bound(published_root: Path) -> None:
    assert all(readiness(published_root).values())
    matrix = load_comparison_manifest(published_root)
    assert matrix["cellCount"] == 18
    assert matrix["runCount"] == 54
    by_cell: dict[str, list[dict[str, object]]] = {}
    for item in matrix["runs"]:
        by_cell.setdefault(item["cellId"], []).append(item)
    assert all(len(items) == 3 for items in by_cell.values())
    for items in by_cell.values():
        seeds = {item["pretrainingReport"]["pretrainSeed"] for item in items}
        assert len(seeds) == 1
        shared = next(item for item in items if item["variant"] == "target-excluded-shared-gfm")
        single = next(item for item in items if item["variant"] == "single-domain-masked-pretrain")
        scratch = next(item for item in items if item["variant"] == "graphsage-scratch")
        assert shared["targetDomain"] not in shared["pretrainingReport"]["sourceDomains"]
        assert single["pretrainingReport"]["sourceDomains"] == [single["targetDomain"]]
        assert scratch["pretrainingReport"]["sourceDomains"] == []
    evaluation = json.loads(
        (published_root / "reports/evaluation.json").read_text(encoding="utf-8")
    )
    tolokers = evaluation["metrics"][ACCOUNT_RISK_TASK]
    assert tolokers["officialSplitCount"] == 10
    assert tolokers["officialSplitProtocol"] == "10-overlapping-official-splits/1.0"
    assert set(tolokers["metricStd"]) == {"auprc", "auroc", "macro-f1", "ece"}
    assert evaluation["comparisonMatrix"]["runCount"] == 54
    assert evaluation["advantageClaim"]["claimStatus"] in {
        "observed_transfer_gain",
        "not_demonstrated",
    }
    registry = load_registry(published_root)
    assert registry["formalReadinessUnaffected"] is True
    assert registry["corpusKind"] == "test-fixture"
    assert registry["testOnly"] is True
    export, checkpoint, _corpus, _documents, _model, _adapters = _load_exported_runtime(
        published_root, device="cpu"
    )
    assert checkpoint["schemaVersion"] == "socialgraph-fm.research-serving-checkpoint/1.0"
    model_card = json.loads(
        (published_root / "exports/research/model-card.json").read_text(encoding="utf-8")
    )
    assert model_card["modelVersionHash"] == export["modelVersionHash"]
    assert model_card["corpusKind"] == "test-fixture"
    assert model_card["testOnly"] is True
    assert model_card["dataUse"]["families"] == ["Synthetic SocialGraph-FM Research test fixtures"]
    assert model_card["tolokersProtocol"]["deployedHead"] == "split-0"
    assert model_card["offlineDepartmentEvaluation"]["usedAsModelInput"] is False
    assert model_card["routeContract"]["similarityRoute"] == SHARED_NULL_ROUTE
    assert all(item["route"] == SHARED_NULL_ROUTE for item in export["embeddings"])
    assert next(
        item for item in export["scenarios"] if item["taskId"] == COLLABORATION_TASK
    )["route"] == SHARED_NULL_ROUTE
    assert all(
        item["route"].startswith("domain:")
        for item in export["scenarios"]
        if item["taskId"] != COLLABORATION_TASK
    )
    assert checkpoint["featureContracts"]["email-eu-core"]["taskRoute"] == SHARED_NULL_ROUTE
    assert all(
        contract["similarityRoute"] == SHARED_NULL_ROUTE
        for contract in checkpoint["featureContracts"].values()
    )
    smoke = json.loads(
        (published_root / "exports/research/smoke-report.json").read_text(encoding="utf-8")
    )
    assert smoke["protocol"] == "fresh-inference-cli-http/1.0"
    assert smoke["checkpoint"]["sha256"] == export["checkpointSha256"]
    assert smoke["freshProcess"]["command"][1:3] == [
        "-m",
        "socialgraph_gfm.core.inference_cli",
    ]
    assert smoke["freshProcess"]["pid"] > 0
    assert smoke["freshProcess"]["terminationMode"] == "terminate"
    assert len(smoke["httpEvidence"]["scenarioResults"]) == 4
    assert all(
        item["repeatDeterministic"]
        for item in smoke["httpEvidence"]["scenarioResults"]
    )
    assert smoke["httpEvidence"]["similarNodes"]["repeatDeterministic"] is True
    command = smoke["freshProcess"]["command"]
    shadow_root = Path(command[command.index("--research-root") + 1])
    assert not shadow_root.exists()
    _manifest, documents = _entry_documents(published_root)
    email_bundle = documents["email-eu-core"][0]
    visible = derive_training_selection(email_bundle).visible_edge_indices
    visible_ids = {
        f"edge:{email_bundle.edges[index].source_id}:{email_bundle.edges[index].target_id}"
        for index in visible
    }
    email_preview = json.loads(
        (
            published_root
            / "exports/research/previews/email-eu-collaboration.json"
        ).read_text(encoding="utf-8")
    )
    assert {edge["id"] for edge in email_preview["edges"]} <= visible_ids
    assert email_preview["edgeCount"] == len(email_bundle.edges)
    assert email_preview["partialPreview"] is True


def test_checkpoint_model_and_registry_tampering_fail_closed(
    published_root: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "copied"
    shutil.copytree(published_root, copied)
    checkpoint = copied / "runs/shared/checkpoint.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        _load_trained_runtime(copied, device="cpu")

    copied_registry = tmp_path / "registry-copy"
    shutil.copytree(published_root, copied_registry)
    path = copied_registry / "published/registry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["modelVersionHash"] = "0" * 64
    payload["registryHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "registryHash"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="registry is stale"):
        load_registry(copied_registry)

    scenario_registry = tmp_path / "scenario-registry-copy"
    shutil.copytree(published_root, scenario_registry)
    scenario_path = scenario_registry / "published/registry.json"
    scenario_payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario_payload["scenarios"][0]["graphVersionHash"] = "0" * 64
    scenario_payload["registryHash"] = canonical_sha256(
        {key: value for key, value in scenario_payload.items() if key != "registryHash"}
    )
    scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="registry is stale"):
        load_registry(scenario_registry)

    legacy_smoke_root = tmp_path / "legacy-smoke-copy"
    shutil.copytree(published_root, legacy_smoke_root)
    smoke_path = legacy_smoke_root / "exports/research/smoke-report.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke.pop("protocol")
    smoke.pop("freshProcess")
    smoke.pop("httpEvidence")
    smoke["smokeHash"] = canonical_sha256(
        {key: value for key, value in smoke.items() if key != "smokeHash"}
    )
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind the selected export"):
        publish_research_model(legacy_smoke_root, allow_test_fixture=True)


def test_comparison_manifest_rejects_duplicate_replacement_with_valid_hash(
    published_root: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "comparison-inventory-copy"
    shutil.copytree(published_root, copied)
    path = copied / "runs/comparisons/matrix-manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runs"][0] = dict(payload["runs"][1])
    payload["matrixHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "matrixHash"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cell/variant inventory mismatch"):
        load_comparison_manifest(copied)


def test_corpus_hash_tampering_is_rejected(materialized_root: Path, tmp_path: Path) -> None:
    copied = tmp_path / "corpus-copy"
    shutil.copytree(materialized_root, copied)
    labels = copied / "materialized/corpus/graphs/email-eu-core/labels.json"
    labels.write_text(labels.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="labels hash mismatch"):
        load_corpus_manifest(copied)

    parser_copy = tmp_path / "parser-copy"
    shutil.copytree(materialized_root, parser_copy)
    manifest_path = parser_copy / "materialized/corpus/corpus-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["graphs"][0]["parserVersion"] = "9.9.9"
    payload["corpusHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "corpusHash"}
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="parser identity mismatch"):
        load_corpus_manifest(parser_copy)

    kind_copy = tmp_path / "corpus-kind-copy"
    shutil.copytree(materialized_root, kind_copy)
    kind_manifest_path = kind_copy / "materialized/corpus/corpus-manifest.json"
    kind_manifest = json.loads(kind_manifest_path.read_text(encoding="utf-8"))
    kind_manifest["corpusKind"] = "real"
    kind_manifest["testOnly"] = False
    kind_manifest["corpusHash"] = canonical_sha256(
        {key: value for key, value in kind_manifest.items() if key != "corpusHash"}
    )
    kind_manifest_path.write_text(json.dumps(kind_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash inventory"):
        load_corpus_manifest(kind_copy)


def _corpus_manifest(root: Path) -> tuple[Path, dict[str, object]]:
    path = root / "materialized/corpus/corpus-manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _write_rehashed_manifest(path: Path, payload: dict[str, object]) -> None:
    payload["corpusHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "corpusHash"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")


def _entry(payload: dict[str, object], graph_id: str) -> dict[str, object]:
    return next(item for item in payload["graphs"] if item["graphId"] == graph_id)  # type: ignore[index,union-attr]


def test_rehashed_corpus_entry_counts_hashes_and_direction_are_cross_checked(
    materialized_root: Path, tmp_path: Path
) -> None:
    mutations = {
        "nodeCount": lambda manifest, entry: entry.__setitem__(
            "nodeCount", int(entry["nodeCount"]) + 1
        ),
        "edgeCount": lambda manifest, entry: entry.__setitem__(
            "edgeCount", int(entry["edgeCount"]) + 1
        ),
        "directed": lambda manifest, entry: entry.__setitem__(
            "directed", not bool(entry["directed"])
        ),
        "labelsHash": lambda manifest, entry: entry.__setitem__("labelsHash", "0" * 64),
        "splitHash": lambda manifest, entry: entry.__setitem__("splitHash", "0" * 64),
        "totalNodeCount": lambda manifest, entry: manifest.__setitem__(
            "nodeCount", int(manifest["nodeCount"]) + 1
        ),
        "totalEdgeCount": lambda manifest, entry: manifest.__setitem__(
            "edgeCount", int(manifest["edgeCount"]) + 1
        ),
    }
    for name, mutate in mutations.items():
        copied = tmp_path / name
        shutil.copytree(materialized_root, copied)
        path, manifest = _corpus_manifest(copied)
        mutate(manifest, _entry(manifest, "twitch-DE"))
        _write_rehashed_manifest(path, manifest)
        with pytest.raises(ValueError):
            load_corpus_manifest(copied)


def test_rehashed_family_label_split_and_input_leakage_tampering_is_rejected(
    materialized_root: Path, tmp_path: Path
) -> None:
    twitch = tmp_path / "twitch-input-leak"
    shutil.copytree(materialized_root, twitch)
    manifest_path, manifest = _corpus_manifest(twitch)
    entry = _entry(manifest, "twitch-DE")
    bundle_path = twitch / "materialized/corpus" / str(entry["bundlePath"])
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["nodeFeatures"].append(
        {
            "kind": "numeric",
            "name": "mature",
            "values": [0.0] * len(bundle["nodes"]),
        }
    )
    bundle["graphVersionHash"] = calculate_graph_version_hash(bundle)
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    entry["bundleSha256"] = file_sha256(bundle_path)
    entry["graphVersionHash"] = bundle["graphVersionHash"]
    _write_rehashed_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="input/excluded-field contract"):
        load_corpus_manifest(twitch)

    tolokers = tmp_path / "tolokers-split"
    shutil.copytree(materialized_root, tolokers)
    manifest_path, manifest = _corpus_manifest(tolokers)
    entry = _entry(manifest, "tolokers")
    splits_path = tolokers / "materialized/corpus" / str(entry["splitsPath"])
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    splits["folds"][0]["train"][0], splits["folds"][0]["test"][0] = (
        splits["folds"][0]["test"][0],
        splits["folds"][0]["train"][0],
    )
    splits["splitsHash"] = canonical_sha256(
        {key: value for key, value in splits.items() if key != "splitsHash"}
    )
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    entry["splitsSha256"] = file_sha256(splits_path)
    _write_rehashed_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="split 0 differs"):
        load_corpus_manifest(tolokers)

    wiki = tmp_path / "wiki-label"
    shutil.copytree(materialized_root, wiki)
    manifest_path, manifest = _corpus_manifest(wiki)
    entry = _entry(manifest, "wiki-rfa")
    labels_path = wiki / "materialized/corpus" / str(entry["labelsPath"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    labels["targets"][0]["sourceId"] = labels["targets"][1]["sourceId"]
    labels["labelsHash"] = canonical_sha256(
        {key: value for key, value in labels.items() if key != "labelsHash"}
    )
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    entry["labelsHash"] = labels["labelsHash"]
    entry["labelsSha256"] = file_sha256(labels_path)
    _write_rehashed_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="endpoints differ"):
        load_corpus_manifest(wiki)

    email = tmp_path / "email-sampling"
    shutil.copytree(materialized_root, email)
    manifest_path, manifest = _corpus_manifest(email)
    entry = _entry(manifest, "email-eu-core")
    labels_path = email / "materialized/corpus" / str(entry["labelsPath"])
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    negative = labels["partitions"]["train"]["negatives"][0]
    old_component = negative["samplingComponent"]
    new_component = (
        "uniform-non-two-hop" if old_component == "two-hop-hard" else "two-hop-hard"
    )
    negative["samplingComponent"] = new_component
    counts = labels["negativeSampling"]["componentCounts"]["train"]
    counts["twoHopHard" if old_component == "two-hop-hard" else "uniform"] -= 1
    counts["twoHopHard" if new_component == "two-hop-hard" else "uniform"] += 1
    labels["samplingHash"] = canonical_sha256(
        {
            "partitions": labels["partitions"],
            "negativeSampling": labels["negativeSampling"],
        }
    )
    labels["labelsHash"] = canonical_sha256(
        {key: value for key, value in labels.items() if key != "labelsHash"}
    )
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    entry["labelsHash"] = labels["labelsHash"]
    entry["labelsSha256"] = file_sha256(labels_path)
    _write_rehashed_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="sampling component is inconsistent"):
        load_corpus_manifest(email)


class _FakeResponse:
    def __init__(self, payload: bytes, url: str, *, content_length: int | None = None) -> None:
        self._stream = io.BytesIO(payload)
        self._url = url
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self._stream.close()
        return False

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def geturl(self) -> str:
        return self._url


def _single_tolokers_acquisition(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(
        research_workflow,
        "RESEARCH_SOURCE_RECIPES",
        {"tolokers.npz": ("tolokers", "tolokers", "raw/tolokers/1.0.0/tolokers.npz")},
    )
    monkeypatch.setattr(
        research_workflow,
        "EXPECTED_SOURCE_HASHES",
        {"tolokers.npz": hashlib.sha256(payload).hexdigest()},
    )


def test_missing_source_acquisition_is_atomic_hash_bound_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"trusted-tolokers-fixture"
    _single_tolokers_acquisition(monkeypatch, payload)

    def open_url(request, _timeout):
        return _FakeResponse(payload, request.full_url)

    research_workflow._acquire_missing_sources(tmp_path / "ok", open_url=open_url)
    target = tmp_path / "ok/raw/tolokers/1.0.0/tolokers.npz"
    assert target.read_bytes() == payload

    wrong = tmp_path / "wrong"
    monkeypatch.setattr(
        research_workflow,
        "EXPECTED_SOURCE_HASHES",
        {"tolokers.npz": "0" * 64},
    )
    with pytest.raises(ValueError, match="downloaded source hash mismatch"):
        research_workflow._acquire_missing_sources(wrong, open_url=open_url)
    assert not (wrong / "raw/tolokers/1.0.0/tolokers.npz").exists()

    oversized = tmp_path / "oversized"
    _single_tolokers_acquisition(monkeypatch, payload)

    def oversized_url(request, _timeout):
        return _FakeResponse(payload, request.full_url, content_length=3_000_001)

    with pytest.raises(ValueError, match="Content-Length exceeds"):
        research_workflow._acquire_missing_sources(oversized, open_url=oversized_url)
    assert not (oversized / "raw/tolokers/1.0.0/tolokers.npz").exists()


def test_extracted_sources_are_rebuilt_from_verified_raw_after_tampering(tmp_path: Path) -> None:
    twitch_zip = tmp_path / "raw/twitch-language/1.0.0/twitch.zip"
    twitch_zip.parent.mkdir(parents=True)
    with zipfile.ZipFile(twitch_zip, "w") as archive:
        for member in sorted(research_workflow.TWITCH_ARCHIVE_MEMBERS):
            archive.writestr(member, f"trusted:{member}".encode())
    compressed = {
        "raw/wiki-rfa/1.0.0/wiki-RfA.txt.gz": b"trusted wiki",
        "raw/email-eu-core/1.0.0/email-Eu-core.txt.gz": b"trusted email",
        ("raw/email-eu-core/1.0.0/email-Eu-core-department-labels.txt.gz"): b"trusted departments",
    }
    for relative, payload in compressed.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as stream:
            stream.write(payload)
    research_workflow._ensure_extracted(tmp_path)
    twitch_member = tmp_path / "extracted/twitch-language/1.0.0/twitch/README.txt"
    wiki = tmp_path / "extracted/wiki-rfa/1.0.0/wiki-RfA.txt"
    twitch_member.write_bytes(b"tampered")
    wiki.write_bytes(b"tampered")
    research_workflow._ensure_extracted(tmp_path)
    assert twitch_member.read_bytes() == b"trusted:twitch/README.txt"
    assert wiki.read_bytes() == b"trusted wiki"


def test_matrix_resumes_valid_receipts_without_retraining_shared(
    materialized_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "resume"
    shutil.copytree(materialized_root, root)
    training_path = train_research_model(root, device="cpu", pretrain_epochs=1, head_epochs=1)
    training_before = training_path.read_bytes()
    original = research_workflow._fit_comparison_cell
    calls = 0

    def fail_fifth(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("injected comparison interruption")
        return original(**kwargs)

    monkeypatch.setattr(research_workflow, "_fit_comparison_cell", fail_fifth)
    with pytest.raises(RuntimeError, match="injected comparison interruption"):
        train_research_comparison_matrix(root, device="cpu", pretrain_epochs=1, downstream_epochs=1)
    assert len(tuple((root / "runs/.cmp-full/receipts").glob("*.json"))) == 4
    monkeypatch.setattr(research_workflow, "_fit_comparison_cell", original)
    assert (
        train_research_model(root, device="cpu", pretrain_epochs=1, head_epochs=1).read_bytes()
        == training_before
    )
    matrix_path = train_research_comparison_matrix(
        root, device="cpu", pretrain_epochs=1, downstream_epochs=1
    )
    assert json.loads(matrix_path.read_text(encoding="utf-8"))["runCount"] == 54


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fallback requires a CUDA runtime")
def test_shared_training_cuda_oom_restarts_complete_run_on_cpu_neighbor_fallback(
    materialized_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from socialgraph_gfm.core.trainer import CoreTrainer

    root = tmp_path / "shared-oom"
    shutil.copytree(materialized_root, root)
    original = CoreTrainer.run_steps
    injected = False

    def fail_once(self, count):
        nonlocal injected
        if not injected:
            injected = True
            raise torch.OutOfMemoryError("injected")
        return original(self, count)

    monkeypatch.setattr(CoreTrainer, "run_steps", fail_once)
    path = train_research_model(root, device="cuda", pretrain_epochs=1, head_epochs=1)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["executionMode"] == "cpu-neighbor-fallback"
    assert manifest["device"] == "cpu"
    assert manifest["headDevice"] == "cpu"


def test_matrix_oom_gate_completes_all_runs_in_cpu_neighbor_mode(
    materialized_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "matrix-oom"
    shutil.copytree(materialized_root, root)
    monkeypatch.setattr(
        research_workflow,
        "_comparison_full_attempt",
        lambda *_args, **_kwargs: (None, True),
    )
    path = train_research_comparison_matrix(
        root, device="cuda", pretrain_epochs=1, downstream_epochs=1
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["runCount"] == 54
    assert manifest["executionMode"] == "cpu-neighbor-fallback"
    assert manifest["fallbackDevice"] == "cpu"
