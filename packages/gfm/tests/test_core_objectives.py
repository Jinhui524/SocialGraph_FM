from __future__ import annotations

import pytest
import torch

from socialgraph_gfm.core import objectives
from socialgraph_gfm.core.objectives import (
    SourceValidationScores,
    combine_objective_losses,
    mask_feature_fields,
    mask_paired_edges,
    select_alignment_weight,
)
from socialgraph_gfm.core.training_data import PreparedGraph


def test_field_mask_rate_and_paired_reverse_edge_mask_have_no_leakage() -> None:
    generator = torch.Generator().manual_seed(23)
    field_mask = mask_feature_fields((2_000, 5), generator=generator)
    assert field_mask.dtype == torch.bool
    assert 0.27 < field_mask.float().mean().item() < 0.33

    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 4, 5], [1, 0, 2, 1, 3, 2, 5, 4]], dtype=torch.long
    )
    graph = PreparedGraph.from_edge_index(num_nodes=6, edge_index=edge_index, directed=False)
    retained, masked_pairs = mask_paired_edges(
        edge_index,
        generator=torch.Generator().manual_seed(7),
        probability=0.5,
        pair_count=graph.pair_mask_cache.pair_keys.shape[0],
        pair_inverse=graph.pair_mask_cache.inverse,
        pair_representatives=graph.pair_mask_cache.representative_edges,
        sampled=False,
    )
    retained_edges = {tuple(edge) for edge in retained.t().tolist()}
    for source, target in masked_pairs.tolist():
        assert (source, target) not in retained_edges
        assert (target, source) not in retained_edges

    sampled_edges = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)
    retained, masked_pairs = mask_paired_edges(
        sampled_edges,
        generator=torch.Generator().manual_seed(4),
        probability=0.5,
        pair_count=graph.pair_mask_cache.pair_keys.shape[0],
        pair_inverse=torch.tensor([0, 0, 2]),
        pair_representatives=torch.tensor([True, False, True]),
        sampled=True,
    )
    retained_edges = {tuple(edge) for edge in retained.t().tolist()}
    assert ((0, 1) in retained_edges) == ((1, 0) in retained_edges)
    assert all(tuple(pair) in {(0, 1), (2, 3)} for pair in masked_pairs.tolist())


def test_sampled_pair_mask_is_sampled_only_atomic_and_representative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_randint = torch.randint
    allocation_sizes: list[int] = []

    def tracked_randint(*args: object, **kwargs: object) -> torch.Tensor:
        result = original_randint(*args, **kwargs)  # type: ignore[arg-type]
        allocation_sizes.append(result.numel())
        return result

    monkeypatch.setattr(torch, "randint", tracked_randint)
    pair_id = 999_999_999_999
    retained, positives = mask_paired_edges(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        generator=torch.Generator().manual_seed(31),
        pair_count=1_000_000_000_000,
        pair_inverse=torch.tensor([pair_id, pair_id]),
        pair_representatives=torch.tensor([True, False]),
        probability=1.0,
        sampled=True,
    )
    assert allocation_sizes == [1]
    assert retained.shape == (2, 0)
    assert positives.tolist() == [[0, 1]]

    retained, positives = mask_paired_edges(
        torch.tensor([[1], [0]], dtype=torch.long),
        generator=torch.Generator().manual_seed(31),
        pair_count=1_000_000_000_000,
        pair_inverse=torch.tensor([pair_id]),
        pair_representatives=torch.tensor([False]),
        probability=1.0,
        sampled=True,
    )
    assert retained.shape == (2, 0)
    assert positives.shape == (0, 2)

    with pytest.raises(ValueError, match="pair ID"):
        mask_paired_edges(
            torch.tensor([[0], [1]], dtype=torch.long),
            generator=torch.Generator().manual_seed(31),
            pair_count=4,
            pair_inverse=torch.tensor([-1]),
            pair_representatives=torch.tensor([True]),
            sampled=True,
        )
    with pytest.raises(ValueError, match="pair ID"):
        mask_paired_edges(
            torch.tensor([[0], [1]], dtype=torch.long),
            generator=torch.Generator().manual_seed(31),
            pair_count=4,
            pair_inverse=torch.tensor([4]),
            pair_representatives=torch.tensor([True]),
            sampled=True,
        )


def test_stateless_sampled_pair_decisions_have_expected_rate() -> None:
    pair_ids = torch.arange(100_000, dtype=torch.long)
    selected = objectives.stateless_pair_decisions(
        pair_ids, nonce=123_456, probability=0.15
    )
    assert 0.145 < selected.float().mean().item() < 0.155


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable in this runtime")
def test_stateless_sampled_pair_decisions_match_cpu_and_cuda() -> None:
    pair_ids = torch.tensor([0, 1, 2, 7, 1_000_000, 2_147_483_646], dtype=torch.long)
    cpu = objectives.stateless_pair_decisions(
        pair_ids, nonce=987_654_321, probability=0.15
    )
    cuda = objectives.stateless_pair_decisions(
        pair_ids.cuda(), nonce=987_654_321, probability=0.15
    ).cpu()
    assert torch.equal(cpu, cuda)


def test_loss_weights_and_alignment_selection_use_source_validation_only() -> None:
    total = combine_objective_losses(
        field_loss=torch.tensor(2.0),
        edge_loss=torch.tensor(4.0),
        alignment_loss=torch.tensor(10.0),
        alignment_weight=0.02,
    )
    assert total.item() == pytest.approx(4.2)
    selection = select_alignment_weight(
        SourceValidationScores(weight_0=0.7, weight_002=0.8, weight_005=0.75)
    )
    assert selection.selected_weight == 0.02
    assert selection.source_scores == {0.0: 0.7, 0.02: 0.8, 0.05: 0.75}
    with pytest.raises(TypeError):
        select_alignment_weight(  # type: ignore[call-arg]
            SourceValidationScores(weight_0=0.7, weight_002=0.8, weight_005=0.75),
            target_validation={0.05: 1.0},
        )
