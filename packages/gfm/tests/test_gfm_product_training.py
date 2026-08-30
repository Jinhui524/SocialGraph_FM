from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from socialgraph_gfm.gfm import product_training
from socialgraph_gfm.gfm.model import SocialGraphFMCore
from socialgraph_gfm.gfm.product_training import (
    ProductAdaptBatch,
    ProductTaskModule,
    ProductTrainingConfig,
    SampleProvenance,
    binary_average_precision,
    calibration_by_stratum,
    evaluate_product_predictions,
    product_multitask_loss,
    train_product_steps,
)
from socialgraph_gfm.gfm.types import CoreBatch, CoreModelConfig, CoreSampleProvenance


_CORPUS_HASH = "a" * 64


def _provenance(task: str = "newcomer") -> SampleProvenance:
    return SampleProvenance(
        domain_id="openalex",
        graph_version="openalex-product-fixture-v1",
        cutoff=4.0,
        horizon=365.0,
        task_id=task,  # type: ignore[arg-type]
        source_corpus_hash=_CORPUS_HASH,
    )


def _batch(*, task: str = "newcomer", query_offset: int = 0) -> ProductAdaptBatch:
    torch = pytest.importorskip("torch")
    core = CoreBatch(
        domain_id="openalex",
        modalities={"numeric": torch.randn(5, 4)},
        modality_masks={"numeric": torch.ones(5, dtype=torch.bool)},
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        edge_type=torch.zeros(4, dtype=torch.long),
        edge_time=torch.arange(4, dtype=torch.float32),
        cutoff_time=4.0,
        provenance=CoreSampleProvenance(
            domain_id="openalex",
            graph_version="openalex-product-fixture-v1",
            cutoff=4.0,
            horizon=365.0,
            task_id=task,
            source_corpus_hash=_CORPUS_HASH,
        ),
    )
    return ProductAdaptBatch(
        core_batch=core,
        candidate_edge_index=torch.tensor([[0, 0, 1, 1], [2, 3, 3, 4]]),
        pair_features=torch.randn(4, 8),
        pair_labels=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        query_ids=torch.tensor(
            [query_offset, query_offset, query_offset + 1, query_offset + 1]
        ),
        provenance=_provenance(task),
        participation_node_index=(torch.tensor([0, 1]) if task == "newcomer" else None),
        participation_labels=(torch.tensor([1.0, 0.0]) if task == "newcomer" else None),
    )


def test_newcomer_product_head_updates_with_finite_multitask_loss() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    config = CoreModelConfig(
        modality_dims={"numeric": 4},
        domains=("openalex",),
        num_relations=1,
        variant="base",
        hidden_channels=16,
        time_channels=8,
        relation_bases=8,
    )
    model = ProductTaskModule(SocialGraphFMCore(config), task="newcomer")
    batch = _batch()
    before = model.pair_head[0].weight.detach().clone()
    pair, participation = model(batch)
    losses = product_multitask_loss(
        task="newcomer",
        pair_logits=pair,
        batch=batch,
        participation_logits=participation,
    )
    assert torch.isfinite(losses.total)
    losses.total.backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    assert not torch.equal(before, model.pair_head[0].weight)


def test_product_report_uses_query_bootstrap_and_calibration() -> None:
    report = evaluate_product_predictions(
        task="collaboration",
        ranking_probabilities=np.asarray([0.9, 0.1, 0.8, 0.2]),
        ranking_labels=np.asarray([1, 0, 1, 0]),
        query_ids=np.asarray([0, 0, 1, 1]),
        baseline_scores=np.asarray([0.4, 0.6, 0.45, 0.55], dtype=np.float64),
        seed=20260821,
        bootstrap_samples=200,
        minimum_query_count=2,
    )
    assert report.ndcg_at_20 == 1.0
    assert report.baseline_ndcg_at_20 < report.ndcg_at_20
    assert report.bootstrap_ndcg_gain_lower > 0
    assert 0 <= report.ece <= 1
    assert report.query_count == 2
    assert report.outcome_kind == "pair_event"
    assert {"ece", "brier", "query_count", "outcome_count"} <= report.metrics().keys()


def test_stratified_calibration_requires_both_classes() -> None:
    probabilities = np.asarray([0.8, 0.2, 0.7, 0.3])
    labels = np.asarray([1, 0, 1, 0])
    values = calibration_by_stratum(
        probabilities=probabilities,
        labels=labels,
        sample_ids=np.asarray([10, 11, 12, 13]),
        provenance=_provenance(),
        strata={
            "topic_cluster_0": np.asarray([True, True, False, False]),
            "topic_cluster_1": np.asarray([False, False, True, True]),
        },
        required_partitions={
            "topic_cluster": ("topic_cluster_0", "topic_cluster_1")
        },
    )
    assert set(values) == {"ece_topic_cluster_0", "ece_topic_cluster_1"}
    with pytest.raises(ValueError, match="both outcome classes"):
        calibration_by_stratum(
            probabilities=probabilities,
            labels=labels,
            sample_ids=np.asarray([10, 11, 12, 13]),
            provenance=_provenance(),
            strata={
                "positives": np.asarray([True, False, True, False]),
                "negatives": np.asarray([False, True, False, True]),
            },
            required_partitions={"outcome": ("positives", "negatives")},
        )


def test_product_batch_rejects_query_without_negative() -> None:
    torch = pytest.importorskip("torch")
    batch = _batch()
    invalid = ProductAdaptBatch(
        **{
            **batch.__dict__,
            "pair_labels": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        }
    )
    with pytest.raises(ValueError, match="positive and negative"):
        invalid.validate(pair_feature_dim=8)


def test_newcomer_participation_only_batch_keeps_negative_cohort() -> None:
    torch = pytest.importorskip("torch")
    config = CoreModelConfig(
        modality_dims={"numeric": 4},
        domains=("openalex",),
        num_relations=1,
        variant="base",
        hidden_channels=16,
        time_channels=8,
        relation_bases=8,
    )
    model = ProductTaskModule(SocialGraphFMCore(config), task="newcomer")
    source = _batch()
    batch = ProductAdaptBatch(
        core_batch=source.core_batch,
        candidate_edge_index=torch.empty((2, 0), dtype=torch.long),
        pair_features=torch.empty((0, 8)),
        pair_labels=torch.empty(0),
        query_ids=torch.empty(0, dtype=torch.long),
        provenance=source.provenance,
        participation_node_index=torch.tensor([0, 1]),
        participation_labels=torch.tensor([0.0, 1.0]),
    )
    pair, participation = model(batch)
    losses = product_multitask_loss(
        task="newcomer",
        pair_logits=pair,
        batch=batch,
        participation_logits=participation,
    )
    assert pair.numel() == 0
    assert losses.ranking == 0
    assert torch.isfinite(losses.total)


def test_product_batch_rejects_duplicate_contradiction_and_multi_source_query() -> None:
    torch = pytest.importorskip("torch")
    source = _batch()
    duplicate = ProductAdaptBatch(
        **{
            **source.__dict__,
            "candidate_edge_index": torch.tensor([[0, 0, 0, 0], [2, 2, 3, 4]]),
            "pair_labels": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        }
    )
    with pytest.raises(ValueError, match="contradictory"):
        duplicate.validate(pair_feature_dim=8)

    multi_source = ProductAdaptBatch(
        **{
            **source.__dict__,
            "candidate_edge_index": torch.tensor([[0, 1, 1, 1], [2, 3, 2, 4]]),
        }
    )
    with pytest.raises(ValueError, match="focal source"):
        multi_source.validate(pair_feature_dim=8)


def test_evaluation_rejects_fractional_labels_and_non_integer_query_ids() -> None:
    common = {
        "task": "collaboration",
        "ranking_probabilities": np.asarray([0.9, 0.1, 0.8, 0.2]),
        "ranking_labels": np.asarray([1, 0, 1, 0]),
        "query_ids": np.asarray([0, 0, 1, 1]),
        "baseline_scores": np.asarray([0.4, 0.6, 0.45, 0.55]),
        "seed": 3,
        "bootstrap_samples": 100,
        "minimum_query_count": 2,
    }
    with pytest.raises(ValueError, match="exact binary"):
        evaluate_product_predictions(
            **{**common, "ranking_labels": np.asarray([1.0, 0.5, 1.0, 0.0])}
        )
    with pytest.raises(ValueError, match="integer dtype"):
        evaluate_product_predictions(
            **{**common, "query_ids": np.asarray([0.0, 0.0, 1.0, 1.0])}
        )


def test_evaluation_requires_each_query_class_and_formal_query_floor() -> None:
    common = {
        "task": "collaboration",
        "ranking_probabilities": np.asarray([0.9, 0.8, 0.2, 0.1]),
        "ranking_labels": np.asarray([1, 1, 1, 0]),
        "query_ids": np.asarray([0, 0, 1, 1]),
        "baseline_scores": np.asarray([0.5, 0.4, 0.3, 0.2]),
        "seed": 3,
        "bootstrap_samples": 100,
        "minimum_query_count": 2,
    }
    with pytest.raises(ValueError, match="positive and negative"):
        evaluate_product_predictions(**common)
    valid = {**common, "ranking_labels": np.asarray([1, 0, 1, 0])}
    valid.pop("minimum_query_count")
    with pytest.raises(ValueError, match="at least 100 queries"):
        evaluate_product_predictions(**valid)


def test_newcomer_metrics_use_participation_outcomes() -> None:
    report = evaluate_product_predictions(
        task="newcomer",
        ranking_probabilities=np.asarray([0.9, 0.1, 0.8, 0.2]),
        ranking_labels=np.asarray([1, 0, 1, 0]),
        query_ids=np.asarray([0, 0, 1, 1]),
        baseline_scores=np.asarray([0.4, 0.6, 0.45, 0.55]),
        participation_probabilities=np.asarray([0.9, 0.8, 0.2, 0.1]),
        participation_labels=np.asarray([0, 1, 1, 0]),
        seed=11,
        bootstrap_samples=100,
        minimum_query_count=2,
    )
    assert report.outcome_kind == "participation_outcome"
    assert report.auprc == binary_average_precision(
        np.asarray([0.9, 0.8, 0.2, 0.1]), np.asarray([0, 1, 1, 0])
    )
    assert report.auprc != binary_average_precision(
        np.asarray([0.9, 0.1, 0.8, 0.2]), np.asarray([1, 0, 1, 0])
    )


def test_average_precision_is_tie_grouped_and_order_invariant() -> None:
    first = binary_average_precision(np.asarray([0.5, 0.5]), np.asarray([1, 0]))
    second = binary_average_precision(np.asarray([0.5, 0.5]), np.asarray([0, 1]))
    assert first == second == 0.5


def test_empty_participation_supervision_is_rejected() -> None:
    torch = pytest.importorskip("torch")
    source = _batch()
    empty = ProductAdaptBatch(
        **{
            **source.__dict__,
            "participation_node_index": torch.empty(0, dtype=torch.long),
            "participation_labels": torch.empty(0),
        }
    )
    with pytest.raises(ValueError, match="participation supervision"):
        empty.validate(pair_feature_dim=8)


def test_training_returns_validation_best_model_and_resumable_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    config = CoreModelConfig(
        modality_dims={"numeric": 4},
        domains=("openalex",),
        num_relations=1,
        variant="base",
        hidden_channels=16,
        time_channels=8,
        relation_bases=8,
    )
    model = ProductTaskModule(SocialGraphFMCore(config), task="collaboration")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    validation_values = iter((0.1, 0.2))
    monkeypatch.setattr(
        product_training,
        "_validation_loss",
        lambda *args, **kwargs: next(validation_values),
    )
    result = train_product_steps(
        model,
        optimizer,
        train_batches=lambda: (_batch(task="collaboration"),),
        validation_batches=(_batch(task="collaboration", query_offset=10),),
        device="cpu",
        config=ProductTrainingConfig(
            maximum_steps=2,
            minimum_steps=0,
            evaluation_every_steps=1,
            patience_evaluations=3,
            amp=False,
        ),
    )
    assert result.best_step == 1
    assert all(
        torch.equal(model.state_dict()[name].cpu(), value)
        for name, value in result.best_state.items()
    )
    assert any(
        not torch.equal(result.resume_state.latest_model_state[name], value)
        for name, value in result.best_state.items()
    )
    assert result.resume_state.completed_steps == 2


def test_validation_rejects_cross_batch_query_collisions() -> None:
    torch = pytest.importorskip("torch")
    config = CoreModelConfig(
        modality_dims={"numeric": 4},
        domains=("openalex",),
        num_relations=1,
        variant="base",
        hidden_channels=16,
        time_channels=8,
        relation_bases=8,
    )
    model = ProductTaskModule(SocialGraphFMCore(config), task="collaboration")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="globally unique across batches"):
        train_product_steps(
            model,
            optimizer,
            train_batches=lambda: (_batch(task="collaboration"),),
            validation_batches=(
                _batch(task="collaboration", query_offset=10),
                _batch(task="collaboration", query_offset=10),
            ),
            device="cpu",
            config=ProductTrainingConfig(
                maximum_steps=1,
                minimum_steps=0,
                evaluation_every_steps=1,
                patience_evaluations=1,
                amp=False,
            ),
        )


def test_validation_factory_is_rebuilt_without_retaining_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    config = CoreModelConfig(
        modality_dims={"numeric": 4},
        domains=("openalex",),
        num_relations=1,
        variant="base",
        hidden_channels=16,
        time_channels=8,
        relation_bases=8,
    )
    model = ProductTaskModule(SocialGraphFMCore(config), task="collaboration")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    calls = 0

    def validation_factory():
        nonlocal calls
        calls += 1
        return iter((_batch(task="collaboration", query_offset=10),))

    result = train_product_steps(
        model,
        optimizer,
        train_batches=lambda: iter((_batch(task="collaboration"),)),
        validation_batches=validation_factory,
        device="cpu",
        config=ProductTrainingConfig(
            maximum_steps=2,
            minimum_steps=0,
            evaluation_every_steps=1,
            patience_evaluations=3,
            amp=False,
        ),
    )
    assert result.completed_steps == 2
    assert calls == 2


def test_progress_callback_resume_matches_uninterrupted_training() -> None:
    torch = pytest.importorskip("torch")
    config = CoreModelConfig(
        modality_dims={"numeric": 4},
        domains=("openalex",),
        num_relations=1,
        variant="base",
        hidden_channels=16,
        time_channels=8,
        relation_bases=8,
    )
    torch.manual_seed(91)
    train_batch = _batch(task="collaboration")
    validation_batch = _batch(task="collaboration", query_offset=10)
    template = ProductTaskModule(SocialGraphFMCore(config), task="collaboration")
    initial = {
        name: value.detach().clone() for name, value in template.state_dict().items()
    }
    training_config = ProductTrainingConfig(
        maximum_steps=4,
        minimum_steps=4,
        evaluation_every_steps=1,
        patience_evaluations=9,
        amp=False,
    )

    def build():
        model = ProductTaskModule(SocialGraphFMCore(config), task="collaboration")
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        return model, optimizer, scheduler

    def reset_rng() -> None:
        random.seed(808)
        np.random.seed(808)
        torch.manual_seed(808)

    reset_rng()
    full_model, full_optimizer, full_scheduler = build()
    full = train_product_steps(
        full_model,
        full_optimizer,
        train_batches=lambda: (train_batch,),
        validation_batches=lambda: (validation_batch,),
        device="cpu",
        config=training_config,
        scheduler=full_scheduler,
    )

    class Interrupted(RuntimeError):
        pass

    captured = []

    def stop_after_first(state) -> None:
        captured.append(state)
        raise Interrupted

    reset_rng()
    interrupted_model, interrupted_optimizer, interrupted_scheduler = build()
    with pytest.raises(Interrupted):
        train_product_steps(
            interrupted_model,
            interrupted_optimizer,
            train_batches=lambda: (train_batch,),
            validation_batches=lambda: (validation_batch,),
            device="cpu",
            config=training_config,
            scheduler=interrupted_scheduler,
            progress_callback=stop_after_first,
        )
    assert len(captured) == 1
    assert captured[0].completed_steps == 1

    resumed_model, resumed_optimizer, resumed_scheduler = build()
    resumed = train_product_steps(
        resumed_model,
        resumed_optimizer,
        train_batches=lambda: (train_batch,),
        validation_batches=lambda: (validation_batch,),
        device="cpu",
        config=training_config,
        scheduler=resumed_scheduler,
        resume_state=captured[0],
    )
    assert resumed.completed_steps == full.completed_steps == 4
    assert resumed.best_step == full.best_step
    assert resumed.best_validation_loss == pytest.approx(
        full.best_validation_loss, abs=0.0
    )
    assert all(
        torch.equal(resumed_model.state_dict()[name], full_model.state_dict()[name])
        for name in full_model.state_dict()
    )


def test_multibatch_resume_restores_exact_next_batch_and_dynamic_lr() -> None:
    torch = pytest.importorskip("torch")
    config = CoreModelConfig(
        modality_dims={"numeric": 4},
        domains=("openalex",),
        num_relations=1,
        variant="base",
        hidden_channels=16,
        time_channels=8,
        relation_bases=8,
        dropout=0.0,
    )
    torch.manual_seed(191)
    template = ProductTaskModule(SocialGraphFMCore(config), task="collaboration")
    initial = {
        name: value.detach().clone() for name, value in template.state_dict().items()
    }
    batches = tuple(
        _batch(task="collaboration", query_offset=offset) for offset in (0, 10, 20)
    )
    validation = _batch(task="collaboration", query_offset=100)
    training_config = ProductTrainingConfig(
        maximum_steps=5,
        minimum_steps=5,
        evaluation_every_steps=2,
        patience_evaluations=9,
        amp=False,
        train_iterator_contract_hash="b" * 64,
    )

    def build():
        model = ProductTaskModule(SocialGraphFMCore(config), task="collaboration")
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        learning_rates: list[float] = []
        optimizer.register_step_pre_hook(
            lambda selected, _args, _kwargs: learning_rates.append(
                float(selected.param_groups[0]["lr"])
            )
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0.0, 1.0 - step / 5.0)
        )
        return model, optimizer, scheduler, learning_rates

    def reset_rng() -> None:
        random.seed(1808)
        np.random.seed(1808)
        torch.manual_seed(1808)

    reset_rng()
    full_model, full_optimizer, full_scheduler, full_lrs = build()
    full = train_product_steps(
        full_model,
        full_optimizer,
        train_batches=lambda: iter(batches),
        validation_batches=lambda: (validation,),
        device="cpu",
        config=training_config,
        scheduler=full_scheduler,
    )

    class Interrupted(RuntimeError):
        pass

    captured = []

    def stop_at_first_checkpoint(state) -> None:
        captured.append(state)
        raise Interrupted

    reset_rng()
    interrupted_model, interrupted_optimizer, interrupted_scheduler, prefix_lrs = build()
    with pytest.raises(Interrupted):
        train_product_steps(
            interrupted_model,
            interrupted_optimizer,
            train_batches=lambda: iter(batches),
            validation_batches=lambda: (validation,),
            device="cpu",
            config=training_config,
            scheduler=interrupted_scheduler,
            progress_callback=stop_at_first_checkpoint,
        )
    assert len(captured) == 1
    assert captured[0].completed_steps == 2
    assert captured[0].train_epoch == 0
    assert captured[0].train_batch_offset == 2

    resumed_model, resumed_optimizer, resumed_scheduler, suffix_lrs = build()
    resumed = train_product_steps(
        resumed_model,
        resumed_optimizer,
        train_batches=lambda: iter(batches),
        validation_batches=lambda: (validation,),
        device="cpu",
        config=training_config,
        scheduler=resumed_scheduler,
        resume_state=captured[0],
    )

    assert prefix_lrs + suffix_lrs == pytest.approx(full_lrs, abs=0.0)
    assert resumed.resume_state.train_epoch == full.resume_state.train_epoch == 1
    assert resumed.resume_state.train_batch_offset == full.resume_state.train_batch_offset == 2
    assert resumed.resume_state.scheduler_state == full.resume_state.scheduler_state
    assert resumed.best_validation_loss == pytest.approx(
        full.best_validation_loss, abs=0.0
    )
    assert all(
        torch.equal(
            resumed.resume_state.latest_model_state[name],
            full.resume_state.latest_model_state[name],
        )
        for name in full.resume_state.latest_model_state
    )

    stale_scheduler = {
        **dict(captured[0].scheduler_state or {}),
        "last_epoch": 0,
    }
    stale_resume = replace(captured[0], scheduler_state=stale_scheduler)
    stale_model, stale_optimizer, stale_schedule, _ = build()
    with pytest.raises(ValueError, match="scheduler progress differs"):
        train_product_steps(
            stale_model,
            stale_optimizer,
            train_batches=lambda: iter(batches),
            validation_batches=lambda: (validation,),
            device="cpu",
            config=training_config,
            scheduler=stale_schedule,
            resume_state=stale_resume,
        )


def test_calibration_rejects_partition_gaps_and_duplicate_sample_ids() -> None:
    common = {
        "probabilities": np.asarray([0.8, 0.2, 0.7, 0.3]),
        "labels": np.asarray([1, 0, 1, 0]),
        "provenance": _provenance(),
        "strata": {
            "topic_cluster_0": np.asarray([True, True, False, False]),
            "topic_cluster_1": np.asarray([False, False, True, False]),
        },
        "required_partitions": {
            "topic_cluster": ("topic_cluster_0", "topic_cluster_1")
        },
    }
    with pytest.raises(ValueError, match="cover every sample exactly once"):
        calibration_by_stratum(
            **{**common, "sample_ids": np.asarray([10, 11, 12, 13])}
        )
    with pytest.raises(ValueError, match="globally unique"):
        calibration_by_stratum(
            **{**common, "sample_ids": np.asarray([10, 10, 12, 13])}
        )
