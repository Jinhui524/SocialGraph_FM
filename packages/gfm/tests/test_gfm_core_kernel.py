from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from socialgraph_gfm.gfm import (
    CausalExactNegativeSampler,
    CoreBatch,
    CoreModelConfig,
    CoreSampleProvenance,
    CoreTrainer,
    CoreTrainerConfig,
    DomainTransferResult,
    FixedObjectiveWeights,
    RoundRobinDomainScheduler,
    SocialGraphFMCore,
    evaluate_lodo,
    expected_calibration_error,
    ranking_metrics,
)
from socialgraph_gfm.gfm.model import TemporalNodeEncoder
from socialgraph_gfm.gfm.objectives import masked_attribute_loss, temporal_next_event_loss


def _config(*, variant: str = "moe") -> CoreModelConfig:
    return CoreModelConfig(
        modality_dims={"numeric": 4, "text": 5},
        domains=("academic", "community"),
        num_relations=3,
        hidden_channels=16,
        num_layers=2,
        domain_bottleneck=4,
        variant=variant,  # type: ignore[arg-type]
        dropout=0.0,
        pair_feature_dim=2,
        text_modality="text",
        node_class_count=3,
        graph_output_channels=2,
    )


def _batch(
    domain: str = "academic", *, future: bool = False, negatives_per_positive: int = 1
) -> CoreBatch:
    generator = torch.Generator().manual_seed(27 if domain == "academic" else 31)
    numeric = torch.randn(6, 4, generator=generator)
    text = torch.randn(6, 5, generator=generator)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 1, 4], [1, 2, 3, 4, 5, 0, 0, 2]], dtype=torch.long
    )
    edge_time = torch.tensor([1, 2, 4, 5, 6, 8, 9, 11 if future else 10], dtype=torch.float32)
    positive = torch.tensor([[0, 1, 3], [2, 4, 5]], dtype=torch.long)
    negative = torch.tensor([[0, 2, 4], [4, 5, 1]], dtype=torch.long).repeat(
        1, negatives_per_positive
    )
    return CoreBatch(
        domain_id=domain,
        modalities={"numeric": numeric, "text": text},
        modality_masks={
            "numeric": torch.tensor([True, True, False, True, True, True]),
            "text": torch.tensor([True, False, True, True, False, True]),
        },
        edge_index=edge_index,
        edge_type=torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long),
        edge_time=edge_time,
        cutoff_time=10.0,
        provenance=CoreSampleProvenance(
            domain_id=domain,
            graph_version="synthetic-core",
            cutoff=10.0,
            horizon=1.0,
            task_id="self_supervised_temporal",
            source_corpus_hash="a" * 64,
        ),
        attribute_targets={"numeric": numeric.clone()},
        attribute_masks={
            "numeric": torch.tensor([False, False, True, False, True, False])
        },
        positive_edge_index=positive,
        negative_edge_index=negative,
        positive_relation=torch.tensor([1, 2, 0], dtype=torch.long),
        positive_relation_mask=torch.tensor([True, False, True]),
        time_delta_targets=torch.tensor([2.0, 7.0, 1.0]),
        time_delta_mask=torch.tensor([True, True, False]),
        positive_pair_features=torch.randn(3, 2, generator=generator),
        negative_pair_features=torch.randn(
            3 * negatives_per_positive, 2, generator=generator
        ),
    )


@pytest.mark.parametrize("variant", ["base", "moe"])
def test_core_model_all_branches_and_seven_losses(variant: str) -> None:
    config = _config(variant=variant)
    assert config.time_channels == 32
    assert config.relation_bases == 8
    model = SocialGraphFMCore(config)
    batch = _batch()
    output = model(batch)
    assert output.node_embeddings.shape == (6, 16)
    assert output.modality_weights.shape == (6, 2)
    assert torch.allclose(output.modality_weights.sum(dim=1), torch.ones(6))
    assert output.expert_weights.shape == (6, 3)
    if variant == "base":
        assert model.moe is None
        assert torch.equal(output.expert_weights[:, 2], torch.ones(6))
        assert torch.equal(output.expert_weights[:, :2], torch.zeros(6, 2))
    else:
        assert model.moe is not None
        assert torch.allclose(output.expert_weights.sum(dim=1), torch.ones(6))
    assert model.classify_nodes(output.node_embeddings).shape == (6, 3)
    assert model.predict_graph(output.node_embeddings).shape == (1, 2)

    trainer = CoreTrainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        CoreTrainerConfig(gradient_accumulation_steps=1, amp=False),
        "cpu",
    )
    losses = trainer.forward_loss(batch)
    names = set(losses.components).difference({"total"})
    assert names == {
        "temporal_next_event",
        "masked_attribute",
        "masked_relation_type",
        "log_time_delta",
        "text_structure_alignment",
        "cross_domain_distribution_alignment",
        "moe_route_balance",
    }
    weighted = (
        losses.components["temporal_next_event"]
        + 0.5 * losses.components["masked_attribute"]
        + 0.25 * losses.components["masked_relation_type"]
        + 0.25 * losses.components["log_time_delta"]
        + 0.25 * losses.components["text_structure_alignment"]
        + 0.05 * losses.components["cross_domain_distribution_alignment"]
        + 0.01 * losses.components["moe_route_balance"]
    )
    assert torch.allclose(losses.total, weighted)
    losses.total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(bool(torch.isfinite(value).all()) for value in gradients)


def test_future_message_edge_is_rejected() -> None:
    with pytest.raises(ValueError, match="after the batch cutoff"):
        SocialGraphFMCore(_config())(_batch(future=True))


def test_historical_neighbor_attention_matches_hand_calculation() -> None:
    encoder = TemporalNodeEncoder(2)
    with torch.no_grad():
        encoder.query_projection.weight.copy_(torch.eye(2))
        encoder.key_projection.weight.zero_()
        encoder.key_projection.weight[:, :2].copy_(torch.eye(2))
        encoder.value_projection.weight.zero_()
        encoder.value_projection.weight[:, :2].copy_(torch.eye(2))
    nodes = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 0.0]])
    edges = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    encoded, attention = encoder(
        nodes,
        edges,
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        num_nodes=4,
    )
    expected = torch.softmax(torch.tensor([2.0**0.5, 0.0]), dim=0)
    assert torch.allclose(attention, expected)
    assert float(attention.sum().detach()) == pytest.approx(1.0)
    assert attention[0] > attention[1]
    assert encoded.shape == (4, 2)
    assert bool(torch.isfinite(encoded).all())


def test_text_attribute_reconstruction_uses_cosine_not_smooth_l1() -> None:
    prediction = torch.tensor([[2.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0]])
    selected = torch.tensor([True])
    numeric = masked_attribute_loss(
        {"numeric": prediction},
        {"numeric": target},
        {"numeric": selected},
        anchor=prediction,
        text_modality=None,
    )
    text = masked_attribute_loss(
        {"text": prediction},
        {"text": target},
        {"text": selected},
        anchor=prediction,
        text_modality="text",
    )
    assert numeric > 0.0
    assert float(text.detach()) == pytest.approx(0.0)


def test_temporal_sampled_softmax_supports_multiple_mixed_negatives() -> None:
    positive = torch.tensor([2.0, 1.0], requires_grad=True)
    negative = torch.tensor([[0.0, -1.0, 0.5], [1.5, 0.0, -0.5]])
    actual = temporal_next_event_loss(positive, negative)
    candidates = torch.cat((positive[:, None], negative), dim=1)
    expected = torch.nn.functional.cross_entropy(candidates, torch.zeros(2, dtype=torch.long))
    assert torch.allclose(actual, expected)
    assert torch.allclose(actual, temporal_next_event_loss(positive, negative.reshape(-1)))
    actual.backward()
    assert positive.grad is not None
    assert bool(torch.isfinite(positive.grad).all())

    trainer_model = SocialGraphFMCore(_config())
    trainer = CoreTrainer(
        trainer_model,
        torch.optim.AdamW(trainer_model.parameters(), lr=1e-3),
        CoreTrainerConfig(gradient_accumulation_steps=1, amp=False),
        "cpu",
    )
    assert torch.isfinite(trainer.forward_loss(_batch(negatives_per_positive=2)).total)


def test_fixed_objective_weights_cannot_be_overridden() -> None:
    with pytest.raises(ValueError, match="weights are frozen"):
        FixedObjectiveWeights(masked_attribute=1.0)


def test_exact_negative_sampler_is_causal_unique_and_resumable() -> None:
    visible = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    times = torch.tensor([1.0, 2.0, 3.0])
    sampler = CausalExactNegativeSampler(
        source_count=6,
        target_count=6,
        visible_edge_index=visible,
        visible_edge_time=times,
        cutoff_time=3.0,
        seed=13,
        directed=False,
    )
    positives = torch.tensor([[3], [4]], dtype=torch.long)
    first = sampler.sample(4, current_positive_edges=positives)
    assert first.shape == (2, 4)
    pairs = {tuple(sorted(pair)) for pair in first.t().tolist()}
    assert len(pairs) == 4
    assert not pairs.intersection({(0, 1), (1, 2), (2, 3), (3, 4)})
    assert all(source != target for source, target in pairs)

    state = deepcopy(sampler.state_dict())
    expected_next = sampler.sample(3, current_positive_edges=positives)
    restored = CausalExactNegativeSampler(
        source_count=6,
        target_count=6,
        visible_edge_index=visible,
        visible_edge_time=times,
        cutoff_time=3.0,
        seed=13,
        directed=False,
    )
    restored.load_state_dict(state)
    assert torch.equal(restored.sample(3, current_positive_edges=positives), expected_next)
    with pytest.raises(ValueError, match="after cutoff"):
        CausalExactNegativeSampler(
            source_count=6,
            target_count=6,
            visible_edge_index=visible,
            visible_edge_time=torch.tensor([1.0, 2.0, 4.0]),
            cutoff_time=3.0,
            seed=13,
            directed=False,
        )


def test_round_robin_and_multidomain_accumulation_train() -> None:
    scheduler = RoundRobinDomainScheduler(("a", "b", "c"))
    assert [scheduler.next_domain() for _ in range(5)] == ["a", "b", "c", "a", "b"]
    state = scheduler.state_dict()
    restored_scheduler = RoundRobinDomainScheduler(("a", "b", "c"))
    restored_scheduler.load_state_dict(state)
    assert restored_scheduler.next_domain() == "c"

    torch.manual_seed(42)
    model = SocialGraphFMCore(_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = CoreTrainer(
        model,
        optimizer,
        CoreTrainerConfig(gradient_accumulation_steps=3, amp=False),
        "cpu",
    )
    observed_parameter = next(model.link_head.parameters())
    before = observed_parameter.detach().clone()
    result = trainer.train_epoch(
        {"academic": [_batch("academic"), _batch("academic")], "community": [_batch("community")]}
    )
    assert result.batch_steps == 3
    assert result.optimizer_steps == 1
    assert result.domain_steps == {"academic": 2, "community": 1}
    assert result.mean_losses["cross_domain_distribution_alignment"] > 0.0
    assert not torch.equal(before, observed_parameter.detach())
    snapshot = deepcopy(trainer.state_dict())
    replacement = SocialGraphFMCore(_config())
    replacement_trainer = CoreTrainer(
        replacement,
        torch.optim.AdamW(replacement.parameters(), lr=1e-3),
        CoreTrainerConfig(gradient_accumulation_steps=3, amp=False),
        "cpu",
    )
    replacement_trainer.load_state_dict(snapshot)
    assert replacement_trainer.global_step == 3
    assert replacement_trainer.optimizer_step == 1
    assert all(torch.isfinite(torch.tensor(value)) for value in trainer.evaluate_batch(_batch()).values())


def test_ranking_calibration_and_lodo_evaluators() -> None:
    ranking = ranking_metrics(
        torch.tensor([0.8, 0.4]),
        torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.3, 0.2]]),
        ks=(1, 2),
    )
    assert ranking.hits_at_k == {1: 0.5, 2: 1.0}
    assert ranking.recall_at_k == ranking.hits_at_k
    assert ranking.mrr == pytest.approx(0.75)
    calibration = expected_calibration_error(
        torch.tensor([0.1, 0.8, 0.7, 0.2]), torch.tensor([0, 1, 1, 0]), bins=4
    )
    assert calibration.expected_calibration_error == pytest.approx(0.2)
    assert calibration.brier_score == pytest.approx(0.045)
    lodo = evaluate_lodo(
        [
            DomainTransferResult("academic", 0.70, 0.50, 0.60),
            DomainTransferResult("community", 0.55, 0.52, 0.57),
        ],
        primary_metric="mrr",
    )
    assert lodo.positive_transfer_domains == ("academic",)
    assert lodo.negative_transfer_domains == ("community",)
    assert lodo.mean_gain_over_random == pytest.approx(0.115)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_forward_backward_smoke() -> None:
    model = SocialGraphFMCore(_config()).cuda()
    trainer = CoreTrainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        CoreTrainerConfig(gradient_accumulation_steps=1, amp=True),
        "cuda",
    )
    losses = trainer.forward_loss(_batch().to("cuda"))
    trainer.scaler.scale(losses.total).backward()
    assert torch.isfinite(losses.total)
