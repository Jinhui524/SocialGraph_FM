from __future__ import annotations

import pytest
import torch

from socialgraph_gfm.global_model.model import (
    SparseTop2Router,
    GlobalModel,
    degree_bucket_one_hot,
    router_load_balancing_loss,
)


def _inputs(nodes: int = 12) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    text = torch.randn((nodes, 768), generator=generator)
    buckets = torch.arange(nodes, dtype=torch.uint8) % 128
    source = torch.arange(nodes - 1, dtype=torch.long)
    edge_index = torch.stack((torch.cat((source, source + 1)), torch.cat((source + 1, source))))
    return text, buckets, edge_index


def test_model_supports_sampled_local_edges_and_returns_sparse_explanations() -> None:
    model = GlobalModel().eval()
    text, buckets, edge_index = _inputs()
    output = model(text, buckets, edge_index, domain_id="china")
    assert output.logits.shape == (12,)
    assert output.node_embeddings.shape == (12, 256)
    assert output.fused_features.shape == (12, 256)
    assert output.modality_contributions.shape == (12, 2)
    assert output.router_indices is not None and output.router_indices.shape == (12, 2)
    assert output.router_weights is not None and output.router_weights.shape == (12, 2)
    assert len(output.expert_names) == 8
    assert torch.allclose(output.router_weights.sum(dim=1), torch.ones(12))
    assert torch.all(output.router_indices[:, 0] != output.router_indices[:, 1])
    assert torch.allclose(output.modality_contributions.sum(dim=1), torch.ones(12))

    one_hot = degree_bucket_one_hot(buckets)
    expanded = model(text, one_hot, edge_index, domain_id="china")
    assert torch.allclose(output.logits, expanded.logits)


def test_cross_modal_fusion_matches_the_official_gate_and_concatenation_order() -> None:
    model = GlobalModel().eval()
    text, buckets, edge_index = _inputs(6)
    structural = degree_bucket_one_hot(buckets)
    output = model(text, structural, edge_index, domain_id="iran")
    backbone = model.backbone
    structural_branch = backbone.struct_projector(structural) * torch.relu(
        backbone.cross_attention_to_struct(text)
    )
    text_branch = backbone.text_projector(text) * torch.relu(
        backbone.cross_attention_to_text(structural)
    )
    expected = backbone.joint_projector(torch.cat((structural_branch, text_branch), dim=-1))
    assert torch.allclose(output.fused_features, expected)


def test_router_uses_shared_domain_null_catalog_and_backpropagates_sparse_routes() -> None:
    router = SparseTop2Router(hidden_dim=256, domains=("china", "iran"), dropout=0.0)
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.bias.copy_(torch.tensor([4.0, 0.0, 3.0]))
    values = torch.randn((7, 256), requires_grad=True)
    output = router(
        values,
        domain_id="china",
        graph_stats=torch.zeros(13),
        allowed_experts=("domain:china", "null"),
    )
    assert output.expert_names == ("shared", "domain:china", "domain:iran", "null")
    assert torch.equal(output.indices, torch.tensor([[1, 3]]).expand(7, 2))
    assert torch.allclose(output.weights.sum(dim=1), torch.ones(7))
    assert torch.allclose(output.embeddings, values + router.experts[0](values))
    for country_expert in router.experts[1:3]:
        output_layer = country_expert.network[-1]
        assert torch.count_nonzero(output_layer.weight) == 0
        assert torch.count_nonzero(output_layer.bias) == 0
    loss = output.embeddings.square().mean() + router_load_balancing_loss(
        output.weights, output.indices, expert_count=4
    )
    loss.backward()
    assert values.grad is not None and bool(torch.isfinite(values.grad).all())


def test_router_graph_stats_affect_routes_and_cross_domain_masks_russia_adapter() -> None:
    router = SparseTop2Router(
        hidden_dim=256,
        domains=("china", "cuba", "iran", "russia", "UAE", "venezuela"),
        dropout=0.0,
    )
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.bias.zero_()
        router.gate.bias[6] = 1.0  # null
        router.gate.bias[3] = 100.0  # forbidden Russia adapter
        router.gate.weight[0, 256] = 2.0  # first graph statistic controls China
    values = torch.zeros((4, 256))
    cross_domain = (
        "domain:china",
        "domain:cuba",
        "domain:iran",
        "domain:UAE",
        "domain:venezuela",
        "null",
    )
    low = router(values, graph_stats=torch.zeros(13), allowed_experts=cross_domain)
    high_stats = torch.zeros(13)
    high_stats[0] = 4.0
    high = router(values, graph_stats=high_stats, allowed_experts=cross_domain)
    assert not torch.allclose(low.weights, high.weights)
    assert not bool((low.indices == 4).any())
    assert not bool((high.indices == 4).any())
    assert bool((high.indices == 1).all(dim=0).any())


def test_router_cpu_autocast_aligns_sparse_residual_accumulator_dtype() -> None:
    router = SparseTop2Router(
        hidden_dim=256,
        domains=("china", "iran"),
        dropout=0.0,
    ).train()
    values = torch.randn((8, 256), requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = router(
            values,
            graph_stats=torch.zeros(13),
            allowed_experts=("domain:china", "null"),
        )
        loss = output.embeddings.square().mean()
    assert output.weights.dtype == torch.bfloat16
    assert output.embeddings.dtype == values.dtype
    loss.backward()
    assert values.grad is not None and bool(torch.isfinite(values.grad).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_model_cuda_amp_preserves_float32_router_weights_and_backpropagates() -> None:
    model = GlobalModel().to("cuda").train()
    text, buckets, edge_index = (value.to("cuda") for value in _inputs(16))
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(
            text,
            buckets,
            edge_index,
            domain_id="china",
            graph_stats=torch.zeros(13, device="cuda"),
            allowed_experts=("domain:china", "null"),
        )
        loss = output.logits.float().square().mean()
    assert output.router_weights is not None
    assert output.router_weights.dtype == torch.float32
    assert output.node_embeddings.dtype == torch.float16
    assert torch.allclose(
        output.router_weights.sum(dim=1),
        torch.ones(16, device="cuda"),
    )
    loss.backward()
    assert model.node_head.weight.grad is not None


def test_model_rejects_global_edge_ids_in_a_sampled_batch() -> None:
    model = GlobalModel()
    text, buckets, edge_index = _inputs(5)
    edge_index[0, 0] = 5
    try:
        model(text, buckets, edge_index, domain_id="china")
    except ValueError as exc:
        assert "outside the sampled batch" in str(exc)
    else:
        raise AssertionError("out-of-range sampled edge was accepted")
