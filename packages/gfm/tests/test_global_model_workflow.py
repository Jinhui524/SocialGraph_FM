from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from socialgraph_gfm.global_model import config as global_model_config
from socialgraph_gfm.global_model import training, workflow
from socialgraph_gfm.global_model.config import (
    COUNTRIES,
    SOURCE_COUNTRIES,
    DatasetRef,
    ProtocolPlan,
    protocol_plan,
)
from socialgraph_gfm.global_model.contracts import TRACE_NAMES
from socialgraph_gfm.global_model.training import (
    BalancedSeedSampler,
    DomainData,
    InferenceData,
    TrainingOptions,
)


def _stage(plan: ProtocolPlan, name: str) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (reference.country, reference.variant, reference.split)
        for reference in getattr(plan, name)
    )


def test_protocols_are_pinned_and_cross_domain_is_source_only_until_frozen_test() -> None:
    in_domain = protocol_plan("in_domain")
    low_label = protocol_plan("low_label")
    cross_domain = protocol_plan("cross_domain")
    global_plan = protocol_plan("global")

    assert _stage(in_domain, "train") == (("russia", "base", "train"),)
    assert _stage(in_domain, "select") == (("russia", "base", "validation"),)
    assert _stage(in_domain, "calibrate") == (("russia", "base", "validation"),)
    assert _stage(in_domain, "evaluate") == (("russia", "base", "test"),)

    assert _stage(low_label, "train") == (("russia", "0.95U", "train"),)
    assert _stage(low_label, "select") == (("russia", "base", "validation"),)
    assert _stage(low_label, "calibrate") == (("russia", "base", "validation"),)
    assert _stage(low_label, "evaluate") == (("russia", "base", "test"),)

    assert tuple(reference.country for reference in cross_domain.train) == SOURCE_COUNTRIES
    assert tuple(reference.country for reference in cross_domain.select) == SOURCE_COUNTRIES
    assert tuple(reference.country for reference in cross_domain.calibrate) == SOURCE_COUNTRIES
    assert _stage(cross_domain, "evaluate") == (("russia", "base", "test"),)
    assert cross_domain.target_policy == "source-only-selection-then-single-frozen-target-test"
    assert all(
        reference.country != "russia"
        for stage in (cross_domain.train, cross_domain.select, cross_domain.calibrate)
        for reference in stage
    )
    assert workflow._allowed_experts("in_domain") == ("domain:russia", "null")
    assert workflow._allowed_experts("low_label") == ("domain:russia", "null")
    assert workflow._allowed_experts("cross_domain") == (
        "domain:china",
        "domain:cuba",
        "domain:iran",
        "domain:UAE",
        "domain:venezuela",
        "null",
    )
    assert "domain:russia" not in workflow._allowed_experts("cross_domain")

    for stage_name in ("train", "select", "calibrate", "evaluate"):
        assert tuple(
            reference.country for reference in getattr(global_plan, stage_name)
        ) == COUNTRIES


def test_cross_domain_contract_rejects_target_domain_before_evaluation() -> None:
    source_validation = tuple(
        DatasetRef(country=country, variant="base", split="validation")
        for country in SOURCE_COUNTRIES
    )
    invalid = ProtocolPlan(
        protocol="cross_domain",
        train=tuple(
            DatasetRef(country=country, variant="base", split="train")
            for country in SOURCE_COUNTRIES
        ),
        select=source_validation + (
            DatasetRef(country="russia", variant="base", split="validation"),
        ),
        calibrate=source_validation,
        evaluate=(DatasetRef(country="russia", variant="base", split="test"),),
        target_policy="source-only-selection-then-single-frozen-target-test",
    )

    with pytest.raises(ValueError, match="source-only"):
        global_model_config._validate_protocol(invalid)


def test_balanced_seed_sampler_is_exactly_balanced_and_restores_next_draw() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    mask = torch.ones(labels.shape, dtype=torch.bool)
    sampler = BalancedSeedSampler(labels, mask, seed=912)

    first = sampler.draw(10)
    checkpoint = sampler.state_dict()
    expected_next = sampler.draw(12)
    sampler.draw(8)
    sampler.load_state_dict(checkpoint)
    restored_next = sampler.draw(12)

    assert int((labels[first] == 0).sum()) == 5
    assert int((labels[first] == 1).sum()) == 5
    assert int((labels[restored_next] == 0).sum()) == 6
    assert int((labels[restored_next] == 1).sum()) == 6
    assert torch.equal(restored_next, expected_next)


def test_string_device_is_normalized_for_calibration_and_masked_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_devices: list[torch.device] = []

    def fake_split_logits(
        _model: Any,
        _domains: Any,
        *,
        device: Any,
        **_kwargs: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        observed_devices.append(device)
        return (
            {"russia": torch.tensor([-2.0, 2.0])},
            {"russia": torch.tensor([0.0, 1.0])},
        )

    monkeypatch.setattr(workflow, "collect_split_logits", fake_split_logits)
    calibration_model: Any = torch.nn.Identity()
    calibration_domains: Any = {"russia": SimpleNamespace()}
    report, _calibrator, _validation = workflow._calibration_report(
        calibration_model,
        calibration_domains,
        options=TrainingOptions(max_steps=1, min_steps=1, amp=False),
        device="cpu",
        allowed_experts=("domain:russia", "null"),
    )
    assert observed_devices == [torch.device("cpu")]
    assert report["countries"] == ["russia"]

    domain = InferenceData(
        country="russia",
        edge_index=torch.empty((2, 0), dtype=torch.long),
        text_features=torch.zeros((2, 768)),
        structural_features=torch.zeros(2, dtype=torch.uint8),
        labels=torch.tensor([0.0, 1.0]),
        mask=torch.ones(2, dtype=torch.bool),
        structure_missing=torch.tensor([True, False]),
        graph_stats=torch.zeros(13),
        source_hashes={"fixture": "russia"},
        split_hash="russia-test",
    )
    batch = SimpleNamespace(
        batch_size=2,
        n_id=torch.tensor([0, 1]),
        y=torch.tensor([0.0, 1.0]),
        structure_missing=torch.tensor([True, False]),
    )
    monkeypatch.setattr(
        training,
        "_pyg_inference_data",
        lambda _domain: SimpleNamespace(selected_mask=torch.ones(2, dtype=torch.bool)),
    )
    monkeypatch.setattr(training, "_neighbor_loader", lambda *_args, **_kwargs: [batch])

    def fake_forward(
        _model: Any,
        _batch: Any,
        _domain_id: Any,
        _graph_stats: Any,
        _allowed_experts: Any,
        device: Any,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        observed_devices.append(device)
        return SimpleNamespace(
            logits=torch.tensor([-1.0, 1.0]),
            router_indices=torch.tensor([[4, 7], [4, 7]]),
            router_weights=torch.tensor([[0.6, 0.4], [0.7, 0.3]]),
            modality_contributions=torch.ones((2, 2)),
            expert_names=(
                "domain:china",
                "domain:cuba",
                "domain:iran",
                "domain:UAE",
                "domain:russia",
                "domain:venezuela",
                "shared",
                "null",
            ),
        )

    monkeypatch.setattr(training, "_forward", fake_forward)
    outputs = training.collect_masked_outputs(
        torch.nn.Identity(),
        {"russia": domain},
        options=TrainingOptions(max_steps=1, min_steps=1, seed_batch_size=2, amp=False),
        device="cpu",
        phase="frozen-test",
        allowed_experts=("domain:russia", "null"),
    )
    assert observed_devices[-1] == torch.device("cpu")
    assert outputs["russia"].node_ids.tolist() == [0, 1]


def _domain(country: str) -> DomainData:
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    return DomainData(
        country=country,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        text_features=torch.zeros((4, 768)),
        structural_features=torch.zeros(4, dtype=torch.uint8),
        labels=labels,
        train_mask=torch.ones(4, dtype=torch.bool),
        validation_mask=torch.tensor([True, True, False, False]),
        structure_missing=torch.tensor([False, False, True, True]),
        graph_stats=torch.zeros(13),
        source_hashes={"fixture": country},
        train_split_hash=f"{country}-train",
        validation_split_hash=f"{country}-validation",
    )


def test_two_domains_use_one_optimizer_step_per_logical_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    domains = {country: _domain(country) for country in ("china", "cuba")}
    backward_countries: list[str] = []
    optimizer_steps: list[int] = []

    monkeypatch.setattr(
        training,
        "_pyg_data",
        lambda domain: SimpleNamespace(y=domain.labels, train_mask=domain.train_mask),
    )
    monkeypatch.setattr(
        training,
        "_memory_smoke",
        lambda *_args, **_kwargs: (
            4,
            {
                "selectedSeedBatchSize": 4,
                "oomFallbacks": [],
                "memorySmokeHash": "fixture",
            },
        ),
    )

    def fake_backward(
        selected_model: torch.nn.Module,
        _data_by_country: Any,
        selected_domains: dict[str, DomainData],
        _samplers: Any,
        _neighbor_generators: Any,
        **_kwargs: Any,
    ) -> dict[str, float]:
        losses: dict[str, float] = {}
        assert isinstance(selected_model, torch.nn.Linear)
        for country in selected_domains:
            backward_countries.append(country)
            loss = torch.square(selected_model.weight).sum() / len(selected_domains)
            loss.backward()
            losses[country] = float(loss.detach())
        return losses

    monkeypatch.setattr(training, "_domain_update_backward", fake_backward)
    monkeypatch.setattr(
        training,
        "collect_split_logits",
        lambda _model, selected_domains, **_kwargs: (
            {country: torch.tensor([-1.0, 1.0]) for country in selected_domains},
            {country: torch.tensor([0.0, 1.0]) for country in selected_domains},
        ),
    )
    original_step = torch.optim.Adam.step

    def counted_step(optimizer: torch.optim.Adam, closure: Any = None):
        optimizer_steps.append(1)
        return original_step(optimizer, closure)

    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    options = TrainingOptions(
        max_steps=2,
        min_steps=1,
        eval_every_steps=1,
        patience_evals=2,
        seed_batch_size=4,
        num_neighbors=(1, 1),
        memory_smoke_batch_sizes=(4, 2),
        checkpoint_every_steps=1,
        amp=False,
    )

    outcome = training.train_balanced_neighbor_model(
        model,
        domains,
        run_dir=tmp_path / "run",
        protocol="cross_domain",
        identity={"fixture": "identity"},
        options=options,
        allowed_experts=("domain:china", "domain:cuba", "null"),
        device="cpu",
    )

    assert len(optimizer_steps) == options.max_steps
    assert backward_countries == ["china", "cuba", "china", "cuba"]
    assert [row["optimizerSteps"] for row in outcome.history] == [1, 2]
    assert all(row["domainBackwardPasses"] == 2 for row in outcome.history)


def test_production_memory_smoke_runs_fifty_dry_updates_and_restores_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = _domain("russia")
    sampler = BalancedSeedSampler(domain.labels, domain.train_mask, seed=17)
    neighbor_generator = torch.Generator(device="cpu")
    neighbor_generator.manual_seed(23)
    sampler_before = sampler.state_dict()
    neighbor_before = neighbor_generator.get_state().clone()
    calls = 0

    def fake_backward(*_args: Any, **_kwargs: Any) -> dict[str, float]:
        nonlocal calls
        calls += 1
        sampler.draw(4)
        torch.rand(1, generator=neighbor_generator)
        return {"russia": 1.0}

    monkeypatch.setattr(training, "_domain_update_backward", fake_backward)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    options = TrainingOptions(
        max_steps=1000,
        min_steps=100,
        seed_batch_size=4,
        memory_smoke_batch_sizes=(4, 2),
        amp=False,
    )

    selected, report = training._memory_smoke(
        model,
        optimizer,
        scaler,
        {"russia": SimpleNamespace()},
        {"russia": domain},
        {"russia": sampler},
        {"russia": neighbor_generator},
        allowed_experts=("domain:russia", "null"),
        options=options,
        device=torch.device("cpu"),
    )

    assert selected == 4
    assert report["smokeSteps"] == 50
    assert calls == 51  # one candidate probe plus the fixed 50-step dry run
    assert report["throughputSeedsPerSecond"] > 0
    assert report["estimatedTrainingSeconds"] > 0
    restored = sampler.state_dict()
    assert torch.equal(restored["negativeOrder"], sampler_before["negativeOrder"])
    assert torch.equal(restored["positiveOrder"], sampler_before["positiveOrder"])
    assert restored["negativeCursor"] == sampler_before["negativeCursor"]
    assert restored["positiveCursor"] == sampler_before["positiveCursor"]
    assert torch.equal(restored["generatorState"], sampler_before["generatorState"])
    assert torch.equal(neighbor_generator.get_state(), neighbor_before)


def test_checkpoint_resume_restores_exact_sampler_rng_and_optimizer_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    domains = {"russia": _domain("russia")}
    monkeypatch.setattr(
        training,
        "_pyg_data",
        lambda domain: SimpleNamespace(y=domain.labels, train_mask=domain.train_mask),
    )
    monkeypatch.setattr(
        training,
        "_memory_smoke",
        lambda *_args, **_kwargs: (
            4,
            {
                "selectedSeedBatchSize": 4,
                "smokeSteps": 2,
                "oomFallbacks": [],
                "memorySmokeHash": "fixture",
            },
        ),
    )

    def deterministic_backward(
        model: torch.nn.Module,
        _data_by_country: Any,
        selected_domains: dict[str, DomainData],
        samplers: dict[str, BalancedSeedSampler],
        neighbor_generators: dict[str, torch.Generator],
        *,
        batch_size: int,
        **_kwargs: Any,
    ) -> dict[str, float]:
        losses: dict[str, float] = {}
        for country in selected_domains:
            seeds = samplers[country].draw(batch_size).float()
            noise = torch.rand((), generator=neighbor_generators[country])
            coefficient = seeds.mean() + noise
            parameter = next(model.parameters())
            loss = torch.square(parameter * coefficient).sum()
            loss.backward()
            losses[country] = float(loss.detach())
        return losses

    monkeypatch.setattr(training, "_domain_update_backward", deterministic_backward)
    monkeypatch.setattr(
        training,
        "collect_split_logits",
        lambda model, _domains, **_kwargs: (
            {
                "russia": torch.tensor(
                    [-1.0, float(next(model.parameters()).detach().mean())]
                )
            },
            {"russia": torch.tensor([0.0, 1.0])},
        ),
    )
    options = TrainingOptions(
        max_steps=3,
        min_steps=1,
        eval_every_steps=1,
        patience_evals=3,
        seed_batch_size=4,
        num_neighbors=(1, 1),
        memory_smoke_batch_sizes=(4, 2),
        checkpoint_every_steps=1,
        amp=False,
    )
    initial = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(initial.weight, 0.5)

    interrupted = torch.nn.Linear(1, 1, bias=False)
    interrupted.load_state_dict(initial.state_dict())

    def stop_after_first_step(step: int) -> None:
        if step == 1:
            raise RuntimeError("injected interruption")

    with pytest.raises(RuntimeError, match="injected interruption"):
        training.train_balanced_neighbor_model(
            interrupted,
            domains,
            run_dir=tmp_path / "resumed",
            protocol="in_domain",
            identity={"fixture": "identity"},
            options=options,
            allowed_experts=("domain:russia", "null"),
            device="cpu",
            on_step_complete=stop_after_first_step,
        )

    resumed_model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(resumed_model.weight, -99.0)
    resumed = training.train_balanced_neighbor_model(
        resumed_model,
        domains,
        run_dir=tmp_path / "resumed",
        protocol="in_domain",
        identity={"fixture": "identity"},
        options=options,
        allowed_experts=("domain:russia", "null"),
        device="cpu",
    )
    uninterrupted_model = torch.nn.Linear(1, 1, bias=False)
    uninterrupted_model.load_state_dict(initial.state_dict())
    uninterrupted = training.train_balanced_neighbor_model(
        uninterrupted_model,
        domains,
        run_dir=tmp_path / "uninterrupted",
        protocol="in_domain",
        identity={"fixture": "identity"},
        options=options,
        allowed_experts=("domain:russia", "null"),
        device="cpu",
    )
    resumed_latest = torch.load(
        resumed.checkpoint_path, map_location="cpu", weights_only=True
    )
    uninterrupted_latest = torch.load(
        uninterrupted.checkpoint_path, map_location="cpu", weights_only=True
    )

    assert resumed.resumed_from_step == 1
    assert resumed_latest["stepCompleted"] == options.max_steps
    assert uninterrupted_latest["stepCompleted"] == options.max_steps
    assert training.tensor_state_hash(resumed_latest["modelState"]) == training.tensor_state_hash(
        uninterrupted_latest["modelState"]
    )
    assert resumed_latest["optimizerState"] == uninterrupted_latest["optimizerState"]
    assert torch.equal(
        resumed_latest["rngState"]["cpuRngState"],
        uninterrupted_latest["rngState"]["cpuRngState"],
    )


class _InjectedTargetReadFailure(RuntimeError):
    pass


def test_cross_domain_target_access_is_claimed_before_read_and_cannot_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "global-model"
    run_dir = root / "runs" / "cross_domain-fixture"
    run_dir.mkdir(parents=True)
    training_manifest = run_dir / "training-manifest.json"
    plan = protocol_plan("cross_domain")
    options = TrainingOptions(max_steps=1, min_steps=1, seed_batch_size=2)
    identity = {"releaseIdentityHash": "a" * 64}
    frozen_model = torch.nn.Linear(1, 1).eval()
    frozen_model.requires_grad_(False)
    index = SimpleNamespace(
        manifest=SimpleNamespace(content_hash="c" * 64),
        entries={
            "russia": SimpleNamespace(split_hashes={"full-fold-0": "d" * 64})
        },
    )
    training_record = {
        "runId": "cross_domain-fixture",
        "trainingHash": "e" * 64,
        "identity": identity,
        "targetAccess": {
            "labelsOrMasksPassedToTrainer": False,
            "labelsOrMasksUsedForSelection": False,
            "labelsOrMasksUsedForCalibration": False,
            "frozenTargetEvaluations": 0,
        },
        "artifacts": {"modelStateHash": "b" * 64},
    }
    input_calls = 0

    monkeypatch.setattr(
        workflow,
        "_training_manifest_path",
        lambda *_args, **_kwargs: (
            training_manifest,
            index,
            {},
            plan,
            options,
            identity,
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_load_trained_model",
        lambda *_args, **_kwargs: (
            frozen_model,
            training_record,
            index,
            plan,
            options,
            SimpleNamespace(),
        ),
    )

    def fail_after_claim(*_args: Any, **_kwargs: Any):
        nonlocal input_calls
        input_calls += 1
        access_path = run_dir / "test-access.json"
        assert access_path.is_file()
        access = json.loads(access_path.read_text(encoding="utf-8"))
        assert access["schemaVersion"] == "socialgraph-fm.global-model-test-access/1.0"
        assert access["protocol"] == "cross_domain"
        assert access["runId"] == "cross_domain-fixture"
        assert access["state"] == "claimed"
        assert access["modelStateHash"] == "b" * 64
        raise _InjectedTargetReadFailure("failure after target access was claimed")

    monkeypatch.setattr(workflow, "_evaluation_inputs", fail_after_claim)

    with pytest.raises(_InjectedTargetReadFailure, match="after target access"):
        workflow.evaluate_global_model_protocol(root, protocol="cross_domain", device="cpu")
    with pytest.raises(RuntimeError, match="claimed|target access|already"):
        workflow.evaluate_global_model_protocol(root, protocol="cross_domain", device="cpu")
    assert input_calls == 1
    assert not (run_dir / "evaluation.json").exists()


def test_modality_counts_use_relation_csr_edges_not_trace_membership() -> None:
    relation_degrees = {
        "coRT": [1, 0, 2, 0],
        "coURL": [0, 3, 0, 1],
        "hashSeq": [2, 1, 0, 0],
        "fastRT": [0, 0, 1, 2],
        "tweetSim": [4, 0, 0, 1],
    }

    def relation(name: str) -> SimpleNamespace:
        degrees = np.asarray(relation_degrees[name], dtype=np.int64)
        return SimpleNamespace(
            indptr=np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(degrees)))
        )

    corpus = SimpleNamespace(
        manifest=SimpleNamespace(node_count=4),
        relation=relation,
        # Deliberately contradictory: membership is node presence, never edge evidence.
        trace_membership=np.ones((4, len(TRACE_NAMES)), dtype=np.bool_),
        edge_index=np.asarray([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64),
    )
    expected = np.asarray(
        [[relation_degrees[name][node] for name in TRACE_NAMES] for node in range(4)],
        dtype=np.int32,
    )

    observed = workflow._modality_counts(corpus)  # type: ignore[arg-type]

    assert observed.dtype == np.int32
    assert np.array_equal(observed, expected)


def test_export_verify_smoke_and_publish_bind_four_protocol_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "global-model"
    root.mkdir()
    corpus_hash = "c" * 64
    code_hash = "d" * 64
    lock_hash = "e" * 64
    graph_hash = "f" * 64
    expert_names = [
        "shared",
        "domain:china",
        "domain:cuba",
        "domain:iran",
        "domain:russia",
        "domain:UAE",
        "domain:venezuela",
        "null",
    ]
    trainings: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    node_ids = np.arange(716, dtype=np.int64)
    scores = np.linspace(0.0, 1.0, 716, dtype=np.float32)
    metrics = {
        "sampleCount": 100,
        "positiveCount": 50,
        "macroF1": 0.75,
        "prAuc": 0.8,
        "rocAuc": 0.82,
        "accuracy": 0.76,
        "confusion": {"tp": 40, "tn": 36, "fp": 14, "fn": 10},
    }
    for index, protocol in enumerate(workflow.PROTOCOLS, start=1):
        allowed = list(workflow._allowed_experts(protocol))
        route_indices = [expert_names.index(name) for name in allowed[:2]]
        run_dir = root / "runs" / protocol
        checkpoint = run_dir / "checkpoint-best.pt"
        checkpoint.parent.mkdir(parents=True)
        state = {"weight": torch.tensor([float(index)])}
        torch.save({"modelState": state}, checkpoint)
        state_hash = training.tensor_state_hash(state)
        result_npz = run_dir / "results" / "russia.npz"
        workflow._atomic_npz(
            result_npz,
            node_ids=node_ids,
            scores=scores,
            logits=np.linspace(-2.0, 2.0, 716, dtype=np.float32),
            structure_missing=np.zeros(716, dtype=np.bool_),
            router_indices=np.tile(
                np.asarray([route_indices], dtype=np.int64), (716, 1)
            ),
            router_weights=np.tile(np.asarray([[0.6, 0.4]], dtype=np.float32), (716, 1)),
            modality_counts=np.zeros((716, 5), dtype=np.int32),
        )
        result_json = run_dir / "results" / "russia.json"
        workflow._write_hashed(
            result_json,
            {
                "schemaVersion": "socialgraph-fm.global-model-result/1.0",
                "releaseId": "socialgraph-fm",
                "taskId": "coordination_risk",
                "protocol": protocol,
                "country": "russia",
                "expertNames": expert_names,
                "metrics": metrics,
            },
            "resultHash",
        )
        trainings[protocol] = {
            "trainingHash": f"{index:x}" * 64,
            "identity": {"configHash": "a" * 64},
            "artifacts": {
                "checkpointPath": checkpoint.relative_to(root).as_posix(),
                "checkpointSha256": workflow.file_sha256(checkpoint),
                "modelStateHash": state_hash,
            },
            "allowedExperts": allowed,
            "expertNames": expert_names,
            "calibration": {
                "temperature": 1.0,
                "bias": 0.0,
                "threshold": 0.5,
                "countryBalancedMacroF1": 0.75,
            },
            "metrics": {"validationCountryBalancedMacroF1": 0.75},
            "labelledTrainNodeCount": 100,
            "labelledTrainNodes": {"russia": 100},
        }
        evaluations[protocol] = {
            "evaluationHash": f"{index + 4:x}" * 64,
            "corpusHash": corpus_hash,
            "codeHash": code_hash,
            "runtimeLockHash": lock_hash,
            "metrics": {
                "perCountry": {"russia": metrics},
                "countryBalancedMacroF1": metrics["macroF1"],
                "countryBalancedPrAuc": metrics["prAuc"],
            },
            "splitHashes": {"russia": "9" * 64},
            "targetAccess": {"modelFrozenBeforeTargetAccess": True},
            "artifacts": {
                "resultPaths": {
                    "russia": {
                        "npzPath": result_npz.relative_to(root).as_posix(),
                        "npzSha256": workflow.file_sha256(result_npz),
                        "jsonPath": result_json.relative_to(root).as_posix(),
                        "jsonSha256": workflow.file_sha256(result_json),
                    }
                }
            },
        }

    empty_indptr = np.zeros(717, dtype=np.int64)
    empty_indices = np.empty(0, dtype=np.int64)
    fake_corpus = SimpleNamespace(
        manifest=SimpleNamespace(node_count=716, edge_count=0, content_hash=graph_hash),
        edge_index=np.empty((2, 0), dtype=np.int64),
        fused_csr=SimpleNamespace(indptr=empty_indptr, indices=empty_indices),
        relation=lambda _name: SimpleNamespace(
            indptr=empty_indptr,
            indices=empty_indices,
            weights=np.empty(0, dtype=np.float64),
        ),
    )
    fake_index = SimpleNamespace(
        manifest=SimpleNamespace(content_hash=corpus_hash),
        entries={
            country: SimpleNamespace(
                source_hashes={"fixture": "1" * 64},
                split_hashes={"full-fold-0": "2" * 64},
            )
            for country in COUNTRIES
        },
        load_country=lambda *_args, **_kwargs: fake_corpus,
        country_root=lambda country: root / "corpus" / "countries" / country,
    )
    monkeypatch.setattr(
        workflow, "_load_evaluations", lambda *_args, **_kwargs: (evaluations, trainings)
    )
    monkeypatch.setattr(workflow, "load_corpus_index", lambda *_args, **_kwargs: fake_index)
    monkeypatch.setattr(
        workflow,
        "read_country_manifest",
        lambda _path: SimpleNamespace(node_count=716),
    )

    manifest_path = workflow.export_global_model_release(root)
    verified = workflow.verify_global_model_export(root)
    smoke_path = workflow.smoke_global_model_export(root, fresh_process=True)
    registry_path = workflow.publish_global_model_release(root)

    export = workflow._read_hashed(
        manifest_path, schema=workflow.EXPORT_SCHEMA, hash_field="exportHash"
    )
    registry = workflow._read_hashed(
        registry_path, schema=workflow.REGISTRY_SCHEMA, hash_field="registryHash"
    )
    assert verified["passed"] is True
    assert smoke_path.is_file()
    assert len(
        {
            artifact["protocolModelVersionId"]
            for artifact in export["protocolArtifacts"].values()
        }
    ) == 4
    assert export["modelVersionId"] == export["protocolModels"]["global"]["modelVersionId"]
    assert export["checkpointPath"] == export["protocolArtifacts"]["global"]["checkpointPath"]
    for protocol in workflow.PROTOCOLS:
        artifact = export["protocolArtifacts"][protocol]
        assert artifact["checkpointPath"].endswith(f"checkpoints/{protocol}.pt")
        result = workflow._read_hashed(
            root / artifact["resultPaths"]["russia"]["jsonPath"],
            schema="socialgraph-fm.global-model-result/1.0",
            hash_field="resultHash",
        )
        assert result["modelVersionId"] == artifact["protocolModelVersionId"]
        assert result["modelVersionHash"] == artifact["protocolModelVersionHash"]
    expected_documents = {
        "model-card.json",
        "model-config.json",
        "preprocess-config.json",
        "experts.json",
        "calibration.json",
        "metrics.json",
        "environment-lock.json",
        "source-files.json",
        "example-input.json",
        "example-output.json",
    }
    assert expected_documents.issubset(export["artifacts"])
    assert registry["state"] == "servingReady"
    assert registry["protocolModels"]["global"]["state"] == "servingReady"
    assert all(
        registry["protocolModels"][protocol]["state"] == "frozenDemo"
        for protocol in ("in_domain", "low_label", "cross_domain")
    )
    assert registry["modelCardPath"] == export["modelCardPath"]
    assert registry["modelCardSha256"] == export["modelCardSha256"]
