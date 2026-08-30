from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.io import savemat

from socialgraph_gfm.core.datasets.parsers import (
    parse_facebook100_mat,
    parse_facebook_fixture,
    parse_link_fixture,
    parse_tolokers_npz,
    parse_tolokers_fixture,
    parse_twitch_fixture,
    parse_wiki_rfa,
    prepare_github_musae,
    prepare_wiki_rfa,
)
from socialgraph_gfm.core.datasets.recipes import load_dataset_recipes


FIXTURES = Path(__file__).parent / "fixtures" / "core_datasets"


def test_facebook_fixture_preserves_profile_fields_as_categorical() -> None:
    graph = parse_facebook_fixture(FIXTURES / "facebook100.json")

    assert graph.categorical_features == {
        "gender": ("1", "2"),
        "major": ("10", "11"),
        "secondMajor": (None, "12"),
        "dorm": ("2", "3"),
        "year": ("2008", "2009"),
        "highSchool": ("20", "21"),
    }
    assert graph.numeric_features == {}
    assert load_dataset_recipes()["facebook100"].tasks["gender"].offline_benchmark_only


def test_twitch_fixture_keeps_shared_features_sparse_and_exposes_six_domains() -> None:
    graphs = parse_twitch_fixture(FIXTURES / "twitch-language.json")

    assert tuple(graphs) == ("DE", "EN", "ES", "FR", "PT", "RU")
    assert graphs["DE"].multi_hot_features["sharedAttributes"] == (("3", "8"), ("8",))
    assert graphs["DE"].numeric_features == {}
    assert graphs["DE"].targets["mature"] == (0, 1)


def test_tolokers_fixture_consumes_ten_official_splits_and_banned_target() -> None:
    graph = parse_tolokers_fixture(FIXTURES / "tolokers.json")

    assert len(graph.official_splits) == 10
    assert graph.targets == {"banned": (0, 1, 0)}
    assert graph.official_splits[0].train == (0,)
    assert graph.official_splits[0].validation == (1,)
    assert graph.official_splits[0].test == (2,)


def test_wiki_parser_drops_neutral_ties_text_and_time_then_groups_reciprocals() -> None:
    graph = parse_wiki_rfa(FIXTURES / "wiki-rfa.txt")

    assert graph.node_ids == ("A", "B")
    assert graph.signed_edges == ((0, 1, 1), (1, 0, -1))
    payload = json.dumps(graph.model_payload(), sort_keys=True)
    assert "first support" not in payload
    assert "TXT" not in payload
    assert "DAT" not in payload
    assert "YEA" not in payload

    prepared = prepare_wiki_rfa(FIXTURES / "wiki-rfa.txt", seed=17)
    split = prepared.split
    roles = [role for role in (split.train, split.validation, split.test) if role]
    assert roles == [((0, 1, 1), (1, 0, -1))]
    assert prepared.message_passing_edges == split.train


def test_github_and_email_fixtures_use_static_relation_completion_splits() -> None:
    recipes = load_dataset_recipes()
    for recipe_id in ("github-musae", "email-eu-core"):
        graph, split = parse_link_fixture(FIXTURES / f"{recipe_id}.json", seed=9)
        all_edges = set(split.train) | set(split.validation) | set(split.test)
        assert len(all_edges) == len(graph.edges)
        assert recipes[recipe_id].output_semantics == "static relation completion"
        assert split.validation or split.test

    email, _ = parse_link_fixture(FIXTURES / "email-eu-core.json", seed=9)
    assert email.offline_labels == {"department": ("7", "7", "9", "9")}


def test_github_native_parser_returns_train_only_topology_and_safe_negatives(tmp_path) -> None:
    features = tmp_path / "musae_git_features.json"
    features.write_text(json.dumps({str(i): [i % 3] for i in range(8)}), encoding="utf-8")
    edges = tmp_path / "musae_git_edges.csv"
    rows = ["id_1,id_2"] + [f"{i},{(i + 1) % 8}" for i in range(8)] + [
        "0,2", "1,3", "2,4", "3,5", "4,6", "5,7", "0,4", "1,5"
    ]
    edges.write_text("\n".join(rows) + "\n", encoding="utf-8")

    prepared = prepare_github_musae(edges_path=edges, features_path=features, seed=19)

    withheld = set(prepared.split.validation) | set(prepared.split.test)
    assert withheld
    assert not withheld & set(prepared.message_passing_edges)
    positives = set(prepared.graph.edges)
    assert not positives & set(prepared.sample_negatives(count=3, seed=23))


def _write_tolokers(path: Path, *, nodes: int = 4, feature_width: int = 10) -> None:
    train = np.zeros((10, nodes), dtype=bool)
    valid = np.zeros((10, nodes), dtype=bool)
    test = np.zeros((10, nodes), dtype=bool)
    train[:, :2] = True
    valid[:, 2:3] = True
    test[:, 3:] = True
    np.savez(
        path,
        node_features=np.zeros((nodes, feature_width), dtype=np.float32),
        node_labels=np.arange(nodes, dtype=np.int64) % 2,
        edges=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        train_masks=train,
        val_masks=valid,
        test_masks=test,
    )


def test_tolokers_native_npz_enforces_exact_dimensions(tmp_path) -> None:
    valid = tmp_path / "tolokers.npz"
    _write_tolokers(valid)
    assert len(parse_tolokers_npz(valid).official_splits) == 10

    bad = tmp_path / "bad-width.npz"
    _write_tolokers(bad, feature_width=9)
    with pytest.raises(ValueError, match="nodes, 10"):
        parse_tolokers_npz(bad)


def test_facebook_native_mat_enforces_profile_and_adjacency_dimensions(tmp_path) -> None:
    valid = tmp_path / "Reed98.mat"
    savemat(valid, {"A": sparse.eye(3), "local_info": np.zeros((3, 7))}, do_compression=True)
    assert len(parse_facebook100_mat(valid, graph_id="Reed98").node_ids) == 3

    bad = tmp_path / "bad.mat"
    savemat(bad, {"A": sparse.eye(4), "local_info": np.zeros((3, 7))}, do_compression=True)
    with pytest.raises(ValueError, match="adjacency"):
        parse_facebook100_mat(bad, graph_id="Reed98")
