from __future__ import annotations

import math

import pytest
import torch

from socialgraph_gfm.core.baselines import (
    FeatureMlp,
    GfmFamilySpec,
    SparseLinkx,
    StructureMlp,
    adamic_adar_scores,
    common_neighbors_scores,
    fixed_gfm_family_specs,
)


def test_common_neighbors_and_adamic_adar_literal_scores() -> None:
    edges = ((0, 1), (1, 2), (2, 3))
    candidates = ((0, 2), (0, 3))
    assert common_neighbors_scores(num_nodes=4, edges=edges, candidates=candidates) == (1.0, 0.0)
    scores = adamic_adar_scores(num_nodes=4, edges=edges, candidates=candidates)
    assert scores[0] == pytest.approx(1 / math.log(2))
    assert scores[1] == pytest.approx(0.0)


def test_feature_and_structure_mlps_enforce_distinct_input_contracts() -> None:
    feature = FeatureMlp(input_dim=3, output_dim=2)
    structure = StructureMlp(structure_dim=16, output_dim=2)
    assert feature(attributes=torch.ones((4, 3))).shape == (4, 2)
    assert structure(structure=torch.ones((4, 16))).shape == (4, 2)
    with pytest.raises(ValueError, match="feature width"):
        feature(attributes=torch.ones((4, 16)))
    with pytest.raises(ValueError, match="structure width"):
        structure(structure=torch.ones((4, 3)))
    with pytest.raises(TypeError, match="structure"):
        feature(structure=torch.ones((4, 3)))  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="attributes"):
        structure(attributes=torch.ones((4, 16)))  # type: ignore[call-arg]


def test_fixed_gfm_family_specs_lock_pretraining_and_target_label_isolation() -> None:
    specs = {item.method_id: item for item in fixed_gfm_family_specs()}
    assert tuple(specs) == (
        "graphsage-scratch",
        "graphmae2-single",
        "multi-graph-shared-gfm",
        "domain-aligned-gfm",
    )
    assert specs["graphsage-scratch"].encoder_initialization == "random"
    assert specs["graphsage-scratch"].pretraining_scope == "none"
    assert specs["graphmae2-single"].pretraining_scope == "single-source"
    assert specs["multi-graph-shared-gfm"].pretraining_scope == "multi-source"
    assert specs["domain-aligned-gfm"].alignment_candidates == (0.0, 0.02, 0.05)
    assert all(item.target_labels_in_pretraining is False for item in specs.values())
    assert all(
        item.target_unlabeled_adaptation == (item.method_id != "graphsage-scratch")
        for item in specs.values()
    )
    with pytest.raises(ValueError, match="fixed protocol"):
        GfmFamilySpec(
            method_id="graphsage-scratch",
            encoder_initialization="pretrained",
            pretraining_scope="none",
            field_reconstruction_weight=0.0,
            edge_reconstruction_weight=0.0,
            alignment_candidates=(0.0,),
            target_unlabeled_adaptation=False,
        )


def test_sparse_linkx_consumes_sparse_adjacency_without_dense_n_by_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = 1_000
    indices = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    with torch.sparse.check_sparse_tensor_invariants():
        adjacency = torch.sparse_coo_tensor(
            indices,
            torch.ones(indices.shape[1]),
            (nodes, nodes),
        ).coalesce()
    model = SparseLinkx(num_nodes=nodes, feature_dim=3, hidden_dim=8, output_dim=2)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dense adjacency conversion is forbidden")

    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    output = model(adjacency, torch.ones((nodes, 3)))
    assert output.shape == (nodes, 2)
    assert sum(parameter.numel() for parameter in model.parameters()) < nodes * nodes


def test_sparse_linkx_processes_topology_and_features_independently_before_fusion() -> None:
    nodes = 5
    hidden = 4
    indices = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    adjacency = torch.sparse_coo_tensor(
        indices,
        torch.ones(indices.shape[1]),
        (nodes, nodes),
    ).coalesce()
    features = torch.arange(nodes * 3, dtype=torch.float32).reshape(nodes, 3)
    model = SparseLinkx(num_nodes=nodes, feature_dim=3, hidden_dim=hidden, output_dim=2)
    branch_inputs: dict[str, tuple[int, ...]] = {}

    def capture(name: str):
        def hook(_module, inputs):
            branch_inputs[name] = tuple(inputs[0].shape)

        return hook

    handles = (
        model.adjacency_branch.register_forward_pre_hook(capture("adjacency")),
        model.feature_branch.register_forward_pre_hook(capture("features")),
        model.fusion.register_forward_pre_hook(capture("fusion")),
    )
    try:
        output = model(adjacency, features)
    finally:
        for handle in handles:
            handle.remove()

    assert output.shape == (nodes, 2)
    assert branch_inputs == {
        "adjacency": (nodes, hidden),
        "features": (nodes, 3),
        "fusion": (nodes, hidden * 2),
    }


def test_sparse_linkx_rejects_dense_or_wrong_shape_inputs() -> None:
    model = SparseLinkx(num_nodes=4, feature_dim=2, hidden_dim=4, output_dim=2)
    with pytest.raises(ValueError, match="sparse COO"):
        model(torch.eye(4), torch.ones((4, 2)))
    sparse = torch.eye(4).to_sparse()
    with pytest.raises(ValueError, match="feature shape"):
        model(sparse, torch.ones((3, 2)))
    uncoalesced = torch.sparse_coo_tensor(torch.tensor([[0, 0], [1, 1]]), torch.ones(2), (4, 4))
    with pytest.raises(ValueError, match="coalesced"):
        model(uncoalesced, torch.ones((4, 2)))
    with pytest.raises(ValueError, match="dense strided"):
        model(sparse, torch.ones((4, 2)).to_sparse())
    with pytest.raises(ValueError, match="floating point"):
        model(sparse, torch.ones((4, 2), dtype=torch.long))


@pytest.mark.parametrize(
    "edges",
    [((0.5, 1),), ((True, 1),), ((0, 1), (1, 0))],
)
def test_literal_link_baselines_reject_ambiguous_edge_inventories(edges) -> None:
    with pytest.raises(ValueError, match="integer|unique"):
        common_neighbors_scores(num_nodes=3, edges=edges, candidates=((0, 2),))
