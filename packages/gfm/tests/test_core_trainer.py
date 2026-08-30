from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.adapters import BundleInputAdapter
from socialgraph_gfm.core.bundle import CoreGraphBundle, calculate_graph_version_hash
from socialgraph_gfm.core.checkpoint import (
    CheckpointBindings,
    load_checkpoint,
    publish_checkpoint,
)
from socialgraph_gfm.core.config import TrainingConfig
from socialgraph_gfm.core.model import CoreGFM
from socialgraph_gfm.core.objectives import SourceValidationScores
from socialgraph_gfm.core.trainer import (
    CoreTrainer,
    TrainingGraph,
    ValidationContract,
)
from socialgraph_gfm.core import training_data
from socialgraph_gfm.core.training_data import ExecutionPolicy, PreparedGraph


def _graphs() -> dict[str, TrainingGraph]:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )
    generator = torch.Generator().manual_seed(101)
    return {
        domain: TrainingGraph(
            features=torch.randn(4, 128, generator=generator),
            graph=PreparedGraph.from_edge_index(
                num_nodes=4, edge_index=edge_index.clone(), directed=False
            ),
        )
        for domain in ("alpha", "beta")
    }


def _resume_graphs(device: torch.device) -> dict[str, TrainingGraph]:
    policy = ExecutionPolicy(
        full_batch_edge_threshold=1,
        node_batch_size=2,
        edge_batch_size=2,
        fanout=(1, 0, 0),
    )
    return {
        domain: TrainingGraph.from_bundle(
            adapter=_bundle_adapter().to(device),
            graph=value.graph.to(device),
            execution_policy=policy,
        )
        for domain, value in _graphs().items()
    }


def _bundle_adapter() -> BundleInputAdapter:
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [{"id": str(index), "index": index} for index in range(4)],
        "edges": [
            {"sourceId": "0", "targetId": "1", "edgeType": "edge"},
            {"sourceId": "1", "targetId": "2", "edgeType": "edge"},
            {"sourceId": "2", "targetId": "3", "edgeType": "edge"},
            {"sourceId": "0", "targetId": "3", "edgeType": "edge"},
        ],
        "nodeFeatures": [
            {"kind": "numeric", "name": "score", "values": [0.0, 1.0, 2.0, 3.0]},
            {
                "kind": "multiHot",
                "name": "tags",
                "rowOffsets": [0, 1, 2, 3, 4],
                "values": ["a", "b", "c", "d"],
            },
        ],
        "structuralFeatures": None,
        "source": {"sourceName": "fixture", "sourceSha256": "e" * 64},
        "splitManifest": {"strategy": "official", "assignments": []},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return BundleInputAdapter(
        CoreGraphBundle.model_validate(payload), multi_hot_buckets=16, mode="training"
    )


def _many_field_adapter(field_count: int) -> BundleInputAdapter:
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [{"id": "0", "index": 0}, {"id": "1", "index": 1}],
        "edges": [{"sourceId": "0", "targetId": "1", "edgeType": "edge"}],
        "nodeFeatures": [
            {
                "kind": "multiHot",
                "name": f"tags-{index}",
                "rowOffsets": [0, 1, 2],
                "values": [f"a-{index}", f"b-{index}"],
            }
            for index in range(field_count)
        ],
        "structuralFeatures": None,
        "source": {"sourceName": "fixture", "sourceSha256": "e" * 64},
        "splitManifest": {"strategy": "official", "assignments": []},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return BundleInputAdapter(CoreGraphBundle.model_validate(payload), mode="training")


def _bindings(suffix: str = "a") -> CheckpointBindings:
    return CheckpointBindings(
        config_hash=("a" * 63) + suffix,
        data_hash="b" * 64,
        code_hash="c" * 64,
        environment_hash="d" * 64,
    )


def _validation_contract() -> ValidationContract:
    return ValidationContract.from_artifacts(
        protocol={"name": "fixture-validation-protocol", "version": 1},
        data={"graphVersionHash": "a" * 64, "labelsHash": "b" * 64},
        partition={"role": "validation", "partitionHash": "c" * 64},
        callback={"implementation": "test-core-trainer", "version": 1},
    )


class _ValidationCallback:
    def __init__(
        self,
        *,
        by_step: dict[int, float] | None = None,
        constant: float | None = None,
        consume_torch_rng: bool = False,
    ) -> None:
        self._by_step = by_step
        self._constant = constant
        self._consume_torch_rng = consume_torch_rng
        if (by_step is None) == (constant is None):
            raise ValueError("validation fixture requires exactly one score source")

    def __call__(self, model: CoreGFM) -> float:
        if self._consume_torch_rng:
            torch.rand(())
        if self._by_step is not None:
            step = int(model.node_head.bias.detach()[0].item())
            return self._by_step[step]
        if self._constant is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("constant validation score is unavailable")
        return self._constant


def _bound_best_path(latest: Path, best_base: Path) -> Path:
    payload = load_checkpoint(latest, expected_bindings=_bindings())
    name = payload["trainer"]["fitState"]["bestCheckpointName"]
    assert isinstance(name, str)
    return best_base.parent / name


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def _digest_nested(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                digest.update(repr(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                update(child)
        else:
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def _step_summary(trainer: CoreTrainer, result: Any) -> dict[str, Any]:
    return {
        "domain": result.domain,
        "batch": result.batch_index,
        "ordinal": result.batch_ordinal,
        "loss": result.loss,
        "objective": result.objective_signature,
        "state": _digest_nested(trainer.state_dict()),
    }


def _fresh_process_step_summary(checkpoint: str, device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    graphs = _resume_graphs(device)
    config = replace(TrainingConfig.smoke(max_steps=6), gradient_accumulation=2)
    trainer = CoreTrainer(CoreGFM(node_classes=2).to(device), graphs, config=config, seed=999)
    trainer.load_checkpoint(Path(checkpoint), bindings=_bindings())
    return _step_summary(trainer, trainer.run_steps(1)[0])


def test_resume_produces_exact_next_batch_loss_model_and_optimizer(tmp_path: Path) -> None:
    torch.manual_seed(77)
    initial = copy.deepcopy(CoreGFM(node_classes=2).state_dict())
    config = replace(TrainingConfig.smoke(max_steps=6), gradient_accumulation=2)

    torch.manual_seed(900)
    reference_model = CoreGFM(node_classes=2)
    reference_model.load_state_dict(initial)
    reference = CoreTrainer(
        reference_model, _resume_graphs(torch.device("cpu")), config=config, seed=19
    )
    reference.run_steps(6)
    expected_result = reference.history[-1]

    torch.manual_seed(900)
    interrupted_model = CoreGFM(node_classes=2)
    interrupted_model.load_state_dict(initial)
    interrupted = CoreTrainer(
        interrupted_model, _resume_graphs(torch.device("cpu")), config=config, seed=19
    )
    interrupted.run_steps(5)
    assert interrupted.state_dict()["gradientAccumulationCursor"] == 0
    checkpoint = tmp_path / "resume.pt"
    interrupted.save_checkpoint(checkpoint, bindings=_bindings())

    resumed = CoreTrainer(
        CoreGFM(node_classes=2),
        _resume_graphs(torch.device("cpu")),
        config=config,
        seed=999,
    )
    resumed.load_checkpoint(checkpoint, bindings=_bindings())
    resumed.run_steps(1)

    assert resumed.history[-1].domain == expected_result.domain
    assert resumed.history[-1].batch_index == expected_result.batch_index
    assert resumed.history[-1].loss == pytest.approx(expected_result.loss, abs=0.0)
    assert resumed.history[-1].objective_signature == expected_result.objective_signature
    _assert_nested_equal(reference.model.state_dict(), resumed.model.state_dict())
    _assert_nested_equal(reference.optimizer.state_dict(), resumed.optimizer.state_dict())
    _assert_nested_equal(reference.scheduler.state_dict(), resumed.scheduler.state_dict())

    code = (
        "import json,runpy,sys; ns=runpy.run_path(sys.argv[2]); "
        "print(json.dumps(ns['_fresh_process_step_summary'](sys.argv[1], 'cpu'),sort_keys=True))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(checkpoint), str(Path(__file__))],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == _step_summary(reference, expected_result)


def test_restore_is_transactional_when_late_sampler_state_is_corrupt() -> None:
    trainer = CoreTrainer(
        CoreGFM(node_classes=2), _graphs(), config=TrainingConfig.smoke(max_steps=2), seed=19
    )
    trainer.run_steps(1)
    before = copy.deepcopy(trainer.state_dict())
    corrupt = copy.deepcopy(before)
    first_parameter = next(iter(corrupt["model"]))
    corrupt["model"][first_parameter].add_(100.0)
    corrupt["domainSampler"]["version"] = "corrupt-late-component"

    with pytest.raises(ValueError, match="domain sampler state"):
        trainer.load_state_dict(corrupt)

    after = trainer.state_dict()
    _assert_nested_equal(before, after)


def test_restore_persists_the_exact_training_seed() -> None:
    trainer = CoreTrainer(
        CoreGFM(node_classes=2), _graphs(), config=TrainingConfig.smoke(max_steps=2), seed=19
    )
    state = trainer.state_dict()
    restored = CoreTrainer(
        CoreGFM(node_classes=2), _graphs(), config=TrainingConfig.smoke(max_steps=2), seed=999
    )
    restored.load_state_dict(state)

    assert restored.state_dict()["trainingSeed"] == 19


def test_checkpoint_rejects_hash_mismatch_and_loads_in_fresh_process(tmp_path: Path) -> None:
    config = TrainingConfig.smoke(max_steps=1)
    trainer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=3)
    trainer.run_steps(1)
    checkpoint = tmp_path / "fixed.pt"
    trainer.save_checkpoint(checkpoint, bindings=_bindings())

    for field, changed in {
        "config hash": CheckpointBindings("1" * 64, "b" * 64, "c" * 64, "d" * 64),
        "data hash": CheckpointBindings("a" * 64, "1" * 64, "c" * 64, "d" * 64),
        "code hash": CheckpointBindings("a" * 64, "b" * 64, "1" * 64, "d" * 64),
        "environment hash": CheckpointBindings("a" * 64, "b" * 64, "c" * 64, "1" * 64),
    }.items():
        with pytest.raises(ValueError, match=f"{field} mismatch"):
            load_checkpoint(checkpoint, expected_bindings=changed)

    code = (
        "import json,sys; from pathlib import Path; "
        "from socialgraph_gfm.core.checkpoint import CheckpointBindings,load_checkpoint; "
        "b=CheckpointBindings(config_hash='a'*64,data_hash='b'*64,code_hash='c'*64,"
        "environment_hash='d'*64); p=load_checkpoint(Path(sys.argv[1]),expected_bindings=b); "
        "print(json.dumps({'step':p['trainer']['optimizerStep'],'schema':p['schemaVersion']}))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(checkpoint)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == {
        "schema": "socialgraph-fm.core-checkpoint/1.0",
        "step": 1,
    }


def test_cpu_twenty_step_smoke_and_cuda_smoke_when_available() -> None:
    trainer = CoreTrainer(
        CoreGFM(node_classes=2), _graphs(), config=TrainingConfig.smoke(), seed=31
    )
    trainer.run_steps(20)
    assert trainer.optimizer_step == 20
    assert all(torch.isfinite(torch.tensor(result.loss)) for result in trainer.history)


def test_edge_objective_uses_one_representative_and_only_catches_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = PreparedGraph.from_edge_index(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        directed=False,
    )

    def make_trainer() -> CoreTrainer:
        return CoreTrainer(
            CoreGFM(node_classes=2),
            {"only": TrainingGraph(graph=graph, features=torch.randn(3, 128))},
            config=TrainingConfig.smoke(max_steps=1),
            seed=17,
        )

    requests: list[int] = []
    representative = make_trainer()

    def record_negative_request(_graph: PreparedGraph, count: int) -> torch.Tensor:
        requests.append(count)
        return torch.tensor([[0, 2]], dtype=torch.long)

    monkeypatch.setattr(representative, "_sample_negatives", record_negative_request)
    representative.run_steps(1)
    assert requests == [1]

    unrelated = make_trainer()

    def raise_unrelated(_graph: PreparedGraph, _count: int) -> torch.Tensor:
        raise ValueError("unrelated negative validation")

    monkeypatch.setattr(unrelated, "_sample_negatives", raise_unrelated)
    with pytest.raises(ValueError, match="unrelated negative validation"):
        unrelated.run_steps(1)

    capacity = make_trainer()

    def raise_capacity(_graph: PreparedGraph, _count: int) -> torch.Tensor:
        raise training_data.InsufficientNegativeCapacityError("insufficient unique negative edges")

    monkeypatch.setattr(capacity, "_sample_negatives", raise_capacity)
    result = capacity.run_steps(1)[0]
    assert torch.isfinite(torch.tensor(result.loss))


def test_timeout_checkpoint_is_explicitly_non_promotable(tmp_path: Path) -> None:
    config = replace(TrainingConfig.smoke(max_steps=1), timeout_seconds=0.0)
    trainer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=5)
    checkpoint = tmp_path / "timeout-latest.pt"
    report = trainer.fit(
        latest_checkpoint_path=checkpoint,
        best_checkpoint_path=tmp_path / "timeout-best.pt",
        bindings=_bindings(),
        validate=lambda _model: 1.0,
        validation_contract=_validation_contract(),
    )
    assert report.status == "timeout-non-promotable"
    payload = load_checkpoint(checkpoint, expected_bindings=_bindings())
    assert payload["status"] == "timeout-non-promotable"
    assert payload["promotable"] is False


def test_fresh_and_resumed_legacy_single_checkpoint_fit_are_fail_closed(
    tmp_path: Path,
) -> None:
    trainer = CoreTrainer(
        CoreGFM(node_classes=2),
        _graphs(),
        config=TrainingConfig.smoke(max_steps=1),
        seed=5,
    )
    checkpoint = tmp_path / "legacy.pt"
    with pytest.raises(ValueError, match="legacy single-checkpoint fit is unsupported"):
        trainer.fit(
            checkpoint,
            bindings=_bindings(),
            validate=lambda _model: 1.0,
            validation_contract=_validation_contract(),
        )
    assert not checkpoint.exists()

    latest = tmp_path / "valid-latest.pt"
    best = tmp_path / "valid-best.pt"
    evaluator = _ValidationCallback(constant=0.5)
    producer = CoreTrainer(
        CoreGFM(node_classes=2),
        _graphs(),
        config=TrainingConfig.smoke(max_steps=1),
        seed=5,
    )
    producer.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=best,
        bindings=_bindings(),
        validate=evaluator,
        validation_contract=_validation_contract(),
    )
    latest_before = latest.read_bytes()
    resumed = CoreTrainer(
        CoreGFM(node_classes=2),
        _graphs(),
        config=TrainingConfig.smoke(max_steps=1),
        seed=9,
    )
    resumed.load_checkpoint(latest, bindings=_bindings())
    with pytest.raises(ValueError, match="legacy single-checkpoint fit is unsupported"):
        resumed.fit(
            latest,
            bindings=_bindings(),
            validate=_ValidationCallback(constant=0.5),
            validation_contract=_validation_contract(),
        )
    assert latest.read_bytes() == latest_before


def test_fit_retains_distinct_validation_best_and_latest_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = CoreTrainer(
        CoreGFM(node_classes=2),
        _graphs(),
        config=TrainingConfig.dev(max_steps=750),
        seed=5,
    )

    def advance(count: int):
        trainer.optimizer_step += count
        trainer.model.node_head.bias.data.fill_(float(trainer.optimizer_step))
        return []

    monkeypatch.setattr(trainer, "run_steps", advance)
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    report = trainer.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=best,
        bindings=_bindings(),
        validate=_ValidationCallback(by_step={250: 0.4, 500: 0.8, 750: 0.6}),
        validation_contract=_validation_contract(),
    )

    assert report.status == "complete"
    assert report.latest_step == 750
    assert report.best_step == 500
    latest_payload = load_checkpoint(latest, expected_bindings=_bindings())
    assert report.best_checkpoint_path is not None
    best_payload = load_checkpoint(report.best_checkpoint_path, expected_bindings=_bindings())
    assert latest_payload["trainer"]["optimizerStep"] == 750
    assert best_payload["trainer"]["optimizerStep"] == 500
    assert latest_payload["trainer"]["model"]["node_head.bias"].tolist() == [750, 750]
    assert best_payload["trainer"]["model"]["node_head.bias"].tolist() == [500, 500]


def test_timeout_before_validation_reports_no_best_checkpoint(tmp_path: Path) -> None:
    config = replace(TrainingConfig.smoke(max_steps=1), timeout_seconds=0.0)
    trainer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=5)
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    report = trainer.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=best,
        bindings=_bindings(),
        validate=lambda _model: 1.0,
        validation_contract=_validation_contract(),
    )
    assert report.status == "timeout-non-promotable"
    assert report.best_step is None
    assert report.best_checkpoint_path is None
    assert latest.exists()
    assert not best.exists()


def test_formal_best_checkpoint_is_selected_only_after_minimum_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = CoreTrainer(
        CoreGFM(node_classes=2),
        _graphs(),
        config=TrainingConfig.formal(max_steps=2_250, min_steps=2_000),
        seed=5,
    )

    def advance(count: int):
        trainer.optimizer_step += count
        trainer.model.node_head.bias.data.fill_(float(trainer.optimizer_step))
        return []

    monkeypatch.setattr(trainer, "run_steps", advance)
    latest = tmp_path / "formal-latest.pt"
    best = tmp_path / "formal-best.pt"
    report = trainer.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=best,
        bindings=_bindings(),
        validate=_ValidationCallback(
            by_step={
                250: 0.99,
                500: 0.98,
                750: 0.97,
                1000: 0.96,
                1250: 0.95,
                1500: 0.94,
                1750: 0.93,
                2000: 0.10,
                2250: 0.0,
            }
        ),
        validation_contract=_validation_contract(),
    )
    assert report.best_step == 2_000
    assert report.best_metric == pytest.approx(0.10)
    assert report.best_checkpoint_path is not None
    payload = load_checkpoint(report.best_checkpoint_path, expected_bindings=_bindings())
    assert payload["trainer"]["optimizerStep"] == 2_000
    assert payload["promotable"] is False
    assert load_checkpoint(latest, expected_bindings=_bindings())["promotable"] is True


def test_resumed_fit_retains_prior_validation_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = TrainingConfig.dev(max_steps=750)
    producer = CoreTrainer(
        CoreGFM(node_classes=2),
        _graphs(),
        config=config,
        seed=5,
    )
    latest = tmp_path / "resume-latest.pt"
    best = tmp_path / "resume-best.pt"

    def advance_producer(_count: int):
        if producer.optimizer_step >= 500:
            raise RuntimeError("intentional interruption")
        producer.optimizer_step += 250
        producer.model.node_head.bias.data.fill_(float(producer.optimizer_step))
        return []

    monkeypatch.setattr(producer, "run_steps", advance_producer)
    producer_scores = _ValidationCallback(by_step={250: 0.8, 500: 0.9, 750: 0.8})
    with pytest.raises(RuntimeError, match="intentional interruption"):
        producer.fit(
            latest_checkpoint_path=latest,
            best_checkpoint_path=best,
            bindings=_bindings(),
            validate=producer_scores,
            validation_contract=_validation_contract(),
        )
    bound_best = _bound_best_path(latest, best)
    best_before = bound_best.read_bytes()

    trainer = CoreTrainer(
        CoreGFM(node_classes=2),
        _graphs(),
        config=config,
        seed=9,
    )
    trainer.load_checkpoint(latest, bindings=_bindings())

    def advance(count: int):
        trainer.optimizer_step += count * 250
        trainer.model.node_head.bias.data.fill_(float(trainer.optimizer_step))
        return []

    monkeypatch.setattr(trainer, "run_steps", advance)

    score = _ValidationCallback(by_step={250: 0.8, 500: 0.9, 750: 0.8})

    report = trainer.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=best,
        bindings=_bindings(),
        validate=score,
        validation_contract=_validation_contract(),
    )
    assert report.best_step == 500
    assert report.best_metric == pytest.approx(0.9)
    assert report.best_checkpoint_path == bound_best
    assert bound_best.read_bytes() == best_before
    assert load_checkpoint(latest, expected_bindings=_bindings())["trainer"]["optimizerStep"] == 750


def test_fit_rejects_latest_best_path_alias_before_training(tmp_path: Path) -> None:
    trainer = CoreTrainer(
        CoreGFM(node_classes=2),
        _graphs(),
        config=TrainingConfig.smoke(max_steps=1),
        seed=5,
    )
    same = tmp_path / "same.pt"
    alias = tmp_path / "alias" / ".." / "same.pt"

    with pytest.raises(ValueError, match="physically distinct"):
        trainer.fit(
            latest_checkpoint_path=same,
            best_checkpoint_path=alias,
            bindings=_bindings(),
            validate=lambda _model: 1.0,
            validation_contract=_validation_contract(),
        )
    assert trainer.optimizer_step == 0


def test_fit_rejects_existing_hardlinked_checkpoint_destinations(tmp_path: Path) -> None:
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    latest.write_bytes(b"stale")
    os.link(latest, best)
    trainer = CoreTrainer(
        CoreGFM(node_classes=2),
        _graphs(),
        config=TrainingConfig.smoke(max_steps=1),
        seed=5,
    )

    with pytest.raises(ValueError, match="physically distinct"):
        trainer.fit(
            latest_checkpoint_path=latest,
            best_checkpoint_path=best,
            bindings=_bindings(),
            validate=lambda _model: 1.0,
            validation_contract=_validation_contract(),
        )


@pytest.mark.parametrize("crossing_phase", ["optimizer-step", "validation"])
def test_fit_timeout_crossing_never_publishes_promotable_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crossing_phase: str,
) -> None:
    config = replace(
        TrainingConfig.smoke(max_steps=1),
        timeout_seconds=1.0,
    )
    trainer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=5)
    latest = tmp_path / f"{crossing_phase}-latest.pt"
    best = tmp_path / f"{crossing_phase}-best.pt"
    clock = iter((0.0, 0.1, 2.0) if crossing_phase == "optimizer-step" else (0.0, 0.1, 0.2, 2.0))
    monkeypatch.setattr("socialgraph_gfm.core.trainer.time.monotonic", lambda: next(clock))
    report = trainer.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=best,
        bindings=_bindings(),
        validate=_ValidationCallback(constant=1.0),
        validation_contract=_validation_contract(),
    )

    assert report.status == "timeout-non-promotable"
    assert report.best_step is None
    assert not best.exists()
    assert load_checkpoint(latest, expected_bindings=_bindings())["promotable"] is False


def test_resumed_fit_uses_persisted_best_evidence_without_consuming_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = TrainingConfig.dev(max_steps=250)
    producer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=5)

    def advance(count: int):
        producer.optimizer_step += count
        return []

    monkeypatch.setattr(producer, "run_steps", advance)
    latest = tmp_path / "rng-latest.pt"
    best = tmp_path / "rng-best.pt"
    initial_validation = _ValidationCallback(constant=0.75, consume_torch_rng=True)
    producer.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=best,
        bindings=_bindings(),
        validate=initial_validation,
        validation_contract=_validation_contract(),
    )

    resumed = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=999)
    resumed.load_checkpoint(latest, bindings=_bindings())
    state = torch.random.get_rng_state()
    expected_next = torch.rand(())
    torch.random.set_rng_state(state)
    random_validation = _ValidationCallback(constant=0.75, consume_torch_rng=True)

    report = resumed.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=best,
        bindings=_bindings(),
        validate=random_validation,
        validation_contract=_validation_contract(),
    )

    assert report.best_step == 250
    assert report.best_metric == pytest.approx(0.75)
    assert torch.equal(torch.rand(()), expected_next)


def test_nonlegacy_resume_requires_hash_valid_persisted_fit_state(tmp_path: Path) -> None:
    config = TrainingConfig.smoke(max_steps=2)
    producer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=5)
    producer.run_steps(1)
    legacy_state = producer.state_dict()
    legacy_state.pop("fitState")
    checkpoint = tmp_path / "legacy-resume.pt"
    publish_checkpoint(
        checkpoint,
        trainer_state=legacy_state,
        bindings=_bindings(),
        status="training",
        promotable=False,
    )
    resumed = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=9)
    resumed.load_checkpoint(checkpoint, bindings=_bindings())
    with pytest.raises(ValueError, match="persisted fit validation state"):
        resumed.fit(
            latest_checkpoint_path=tmp_path / "latest.pt",
            best_checkpoint_path=tmp_path / "best.pt",
            bindings=_bindings(),
            validate=lambda _model: 1.0,
            validation_contract=_validation_contract(),
        )

    tampered_state = producer.state_dict()
    tampered_state["fitState"]["bestMetric"] = 0.9
    tampered = tmp_path / "tampered-fit-state.pt"
    publish_checkpoint(
        tampered,
        trainer_state=tampered_state,
        bindings=_bindings(),
        status="training",
        promotable=False,
    )
    with pytest.raises(ValueError, match="fit state hash"):
        resumed.load_checkpoint(tampered, bindings=_bindings())


def test_fresh_fit_rejects_stale_destinations_before_timeout(tmp_path: Path) -> None:
    latest = tmp_path / "stale-latest.pt"
    best = tmp_path / "stale-best.pt"
    latest.write_bytes(b"STALE-LATEST")
    best.write_bytes(b"STALE-BEST")
    config = replace(TrainingConfig.smoke(max_steps=1), timeout_seconds=0.0)
    trainer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=5)

    with pytest.raises(FileExistsError, match="fresh fit checkpoint destination"):
        trainer.fit(
            latest_checkpoint_path=latest,
            best_checkpoint_path=best,
            bindings=_bindings(),
            validate=lambda _model: 1.0,
            validation_contract=_validation_contract(),
        )
    assert latest.read_bytes() == b"STALE-LATEST"
    assert best.read_bytes() == b"STALE-BEST"


def test_public_checkpoint_api_cannot_claim_validation_or_acceptance(
    tmp_path: Path,
) -> None:
    trainer = CoreTrainer(
        CoreGFM(node_classes=2),
        _graphs(),
        config=TrainingConfig.formal(max_steps=2_000),
        seed=5,
    )

    for status in ("validated", "accepted"):
        with pytest.raises(ValueError, match="fit-internal validation evidence"):
            trainer.save_checkpoint(
                tmp_path / f"{status}.pt",
                bindings=_bindings(),
                status=status,
                promotable=True,
            )


def test_failed_best_publication_does_not_advance_fit_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = CoreTrainer(
        CoreGFM(node_classes=2),
        _graphs(),
        config=TrainingConfig.dev(max_steps=500),
        seed=5,
    )

    def advance(count: int):
        trainer.optimizer_step += count
        trainer.model.node_head.bias.data.fill_(float(trainer.optimizer_step))
        return []

    monkeypatch.setattr(trainer, "run_steps", advance)
    latest = tmp_path / "failed-best-latest.pt"
    best = tmp_path / "failed-best.pt"
    original_publish = trainer._publish_fit_checkpoint

    def fail_second_best(*args, **kwargs):
        if kwargs.get("best_step") == 500 and kwargs.get("status") == "validated":
            raise OSError("best publication failed")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(trainer, "_publish_fit_checkpoint", fail_second_best)
    with pytest.raises(OSError, match="best publication failed"):
        trainer.fit(
            latest_checkpoint_path=latest,
            best_checkpoint_path=best,
            bindings=_bindings(),
            validate=_ValidationCallback(by_step={250: 0.4, 500: 0.8}),
            validation_contract=_validation_contract(),
        )

    assert trainer.fit_best_step == 250
    assert trainer.fit_best_metric == pytest.approx(0.4)
    prior_best = _bound_best_path(latest, best)
    assert (
        load_checkpoint(prior_best, expected_bindings=_bindings())["trainer"]["optimizerStep"]
        == 250
    )


def test_fit_state_rejects_rehashed_impossible_validation_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = TrainingConfig.dev(max_steps=500)
    trainer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=5)

    def advance(count: int):
        trainer.optimizer_step += count
        trainer.model.node_head.bias.data.fill_(float(trainer.optimizer_step))
        return []

    monkeypatch.setattr(trainer, "run_steps", advance)
    latest = tmp_path / "history-latest.pt"
    report = trainer.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=tmp_path / "history-best.pt",
        bindings=_bindings(),
        validate=_ValidationCallback(by_step={250: 0.8, 500: 0.6}),
        validation_contract=_validation_contract(),
    )
    assert report.best_metric == pytest.approx(0.8)
    payload = load_checkpoint(latest, expected_bindings=_bindings())
    fit_state = payload["trainer"]["fitState"]
    fit_state["lastValidationMetric"] = 9.0
    fit_state["lastValidationHash"] = canonical_sha256(
        {
            "optimizerStep": fit_state["lastValidationStep"],
            "validationMetric": 9.0,
            "modelStateHash": fit_state["lastModelStateHash"],
            "validationContextHash": fit_state["validationContextHash"],
        }
    )
    fit_state["stateHash"] = canonical_sha256(
        {key: value for key, value in fit_state.items() if key != "stateHash"}
    )
    tampered = tmp_path / "impossible-history.pt"
    publish_checkpoint(
        tampered,
        trainer_state=payload["trainer"],
        bindings=_bindings(),
        status="validated",
        promotable=True,
    )
    resumed = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=9)
    with pytest.raises(ValueError, match="historical maximum"):
        resumed.load_checkpoint(tampered, bindings=_bindings())


def test_resume_recomputes_best_metric_and_rejects_coherent_checkpoint_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = TrainingConfig.dev(max_steps=500)
    trainer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=5)

    def advance(count: int):
        trainer.optimizer_step += count
        trainer.model.node_head.bias.data.fill_(float(trainer.optimizer_step))
        return []

    monkeypatch.setattr(trainer, "run_steps", advance)
    latest = tmp_path / "coherent-latest.pt"
    best_base = tmp_path / "coherent-best.pt"
    evaluator = _ValidationCallback(by_step={250: 0.8, 500: 0.6})
    trainer.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=best_base,
        bindings=_bindings(),
        validate=evaluator,
        validation_contract=_validation_contract(),
    )
    best_path = _bound_best_path(latest, best_base)
    best_payload = load_checkpoint(best_path, expected_bindings=_bindings())
    best_fit = best_payload["trainer"]["fitState"]
    for step_field, metric_field, model_field, identity_field in (
        ("bestStep", "bestMetric", "bestModelStateHash", "bestValidationHash"),
        (
            "lastValidationStep",
            "lastValidationMetric",
            "lastModelStateHash",
            "lastValidationHash",
        ),
    ):
        best_fit[metric_field] = 9.0
        best_fit[identity_field] = canonical_sha256(
            {
                "optimizerStep": best_fit[step_field],
                "validationMetric": 9.0,
                "modelStateHash": best_fit[model_field],
                "validationContextHash": best_fit["validationContextHash"],
            }
        )
    best_fit["stateHash"] = canonical_sha256(
        {key: value for key, value in best_fit.items() if key != "stateHash"}
    )
    publish_checkpoint(
        best_path,
        trainer_state=best_payload["trainer"],
        bindings=_bindings(),
        status="validated",
        promotable=False,
    )

    latest_payload = load_checkpoint(latest, expected_bindings=_bindings())
    latest_fit = latest_payload["trainer"]["fitState"]
    latest_fit["bestMetric"] = 9.0
    latest_fit["bestValidationHash"] = canonical_sha256(
        {
            "optimizerStep": latest_fit["bestStep"],
            "validationMetric": 9.0,
            "modelStateHash": latest_fit["bestModelStateHash"],
            "validationContextHash": latest_fit["validationContextHash"],
        }
    )
    latest_fit["bestCheckpointSha256"] = hashlib.sha256(best_path.read_bytes()).hexdigest()
    latest_fit["stateHash"] = canonical_sha256(
        {key: value for key, value in latest_fit.items() if key != "stateHash"}
    )
    publish_checkpoint(
        latest,
        trainer_state=latest_payload["trainer"],
        bindings=_bindings(),
        status="validated",
        promotable=True,
    )

    resumed = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=9)
    resumed.load_checkpoint(latest, bindings=_bindings())
    with pytest.raises(ValueError, match="not reproduced"):
        resumed.fit(
            latest_checkpoint_path=latest,
            best_checkpoint_path=best_base,
            bindings=_bindings(),
            validate=_ValidationCallback(by_step={250: 0.8, 500: 0.6}),
            validation_contract=_validation_contract(),
        )


def test_checkpoint_commit_deadline_never_exposes_promotable_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(TrainingConfig.smoke(max_steps=1), timeout_seconds=1.0)
    trainer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=5)
    clock = iter((0.0, 0.1, 0.2, 0.3, 0.4, 2.0))
    monkeypatch.setattr("socialgraph_gfm.core.trainer.time.monotonic", lambda: next(clock))
    latest = tmp_path / "commit-timeout-latest.pt"
    report = trainer.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=tmp_path / "commit-timeout-best.pt",
        bindings=_bindings(),
        validate=_ValidationCallback(constant=1.0),
        validation_contract=_validation_contract(),
    )
    assert report.status == "timeout-non-promotable"
    assert report.best_step is None
    assert load_checkpoint(latest, expected_bindings=_bindings())["promotable"] is False
    checkpoint_files = [path for path in tmp_path.iterdir() if path.suffix == ".pt"]
    assert checkpoint_files
    assert all(
        load_checkpoint(path, expected_bindings=_bindings())["promotable"] is False
        for path in checkpoint_files
    )


def test_fit_return_boundary_timeout_downgrades_latest_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(TrainingConfig.smoke(max_steps=1), timeout_seconds=1.0)
    trainer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=5)
    clock = iter((0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 2.0))
    monkeypatch.setattr("socialgraph_gfm.core.trainer.time.monotonic", lambda: next(clock))
    latest = tmp_path / "return-timeout-latest.pt"
    report = trainer.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=tmp_path / "return-timeout-best.pt",
        bindings=_bindings(),
        validate=_ValidationCallback(constant=1.0),
        validation_contract=_validation_contract(),
    )
    assert report.status == "timeout-non-promotable"
    assert report.best_step is None
    assert load_checkpoint(latest, expected_bindings=_bindings())["promotable"] is False


def test_latest_failure_preserves_prior_resumable_best_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = TrainingConfig.dev(max_steps=500)
    producer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=5)

    def advance_producer(count: int):
        producer.optimizer_step += count
        producer.model.node_head.bias.data.fill_(float(producer.optimizer_step))
        return []

    monkeypatch.setattr(producer, "run_steps", advance_producer)
    latest = tmp_path / "transaction-latest.pt"
    best = tmp_path / "transaction-best.pt"
    original_publish = producer._publish_fit_checkpoint

    def fail_latest_after_new_best(path: Path, *args, **kwargs):
        if path == latest and kwargs.get("best_step") == 500:
            raise OSError("latest publication failed")
        return original_publish(path, *args, **kwargs)

    monkeypatch.setattr(producer, "_publish_fit_checkpoint", fail_latest_after_new_best)
    with pytest.raises(OSError, match="latest publication failed"):
        producer.fit(
            latest_checkpoint_path=latest,
            best_checkpoint_path=best,
            bindings=_bindings(),
            validate=_ValidationCallback(by_step={250: 0.4, 500: 0.8}),
            validation_contract=_validation_contract(),
        )
    prior_latest = load_checkpoint(latest, expected_bindings=_bindings())
    assert prior_latest["trainer"]["optimizerStep"] == 250
    prior_best = _bound_best_path(latest, best)
    assert prior_best.exists()

    resumed = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=9)
    resumed.load_checkpoint(latest, bindings=_bindings())

    def advance_resumed(count: int):
        resumed.optimizer_step += count
        resumed.model.node_head.bias.data.fill_(float(resumed.optimizer_step))
        return []

    monkeypatch.setattr(resumed, "run_steps", advance_resumed)
    report = resumed.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=best,
        bindings=_bindings(),
        validate=_ValidationCallback(by_step={250: 0.4, 500: 0.8}),
        validation_contract=_validation_contract(),
    )
    assert report.status == "complete"
    assert report.best_step == 500
    assert report.best_checkpoint_path is not None
    assert report.best_checkpoint_path != prior_best


def test_resume_rejects_changed_validation_callback_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = TrainingConfig.dev(max_steps=250)
    producer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=5)

    def advance(count: int):
        producer.optimizer_step += count * 250
        return []

    monkeypatch.setattr(producer, "run_steps", advance)
    latest = tmp_path / "callback-latest.pt"
    best = tmp_path / "callback-best.pt"
    producer.fit(
        latest_checkpoint_path=latest,
        best_checkpoint_path=best,
        bindings=_bindings(),
        validate=_ValidationCallback(constant=0.5),
        validation_contract=_validation_contract(),
    )
    resumed = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=9)
    resumed.load_checkpoint(latest, bindings=_bindings())

    with pytest.raises(ValueError, match="callback identity changed"):
        resumed.fit(
            latest_checkpoint_path=latest,
            best_checkpoint_path=best,
            bindings=_bindings(),
            validate=_ValidationCallback(constant=0.6),
            validation_contract=_validation_contract(),
        )


def test_fit_rejects_caller_fabricated_validation_contract(
    tmp_path: Path,
) -> None:
    fabricated = object.__new__(ValidationContract)
    for name, value in (
        ("protocol_hash", "1" * 64),
        ("data_hash", "2" * 64),
        ("partition_hash", "3" * 64),
        ("callback_artifact_hash", "4" * 64),
        ("contract_hash", "5" * 64),
        ("_seal", object()),
    ):
        object.__setattr__(fabricated, name, value)
    trainer = CoreTrainer(
        CoreGFM(node_classes=2),
        _graphs(),
        config=TrainingConfig.smoke(max_steps=1),
        seed=5,
    )
    with pytest.raises(TypeError, match="artifact factory"):
        trainer.fit(
            latest_checkpoint_path=tmp_path / "fabricated-latest.pt",
            best_checkpoint_path=tmp_path / "fabricated-best.pt",
            bindings=_bindings(),
            validate=_ValidationCallback(constant=0.5),
            validation_contract=fabricated,
        )


@pytest.mark.parametrize(
    ("status", "promotable"),
    [("training", True), ("timeout-non-promotable", True), ("accepted", False)],
)
def test_checkpoint_writer_rejects_contradictory_status(
    tmp_path: Path, status: str, promotable: bool
) -> None:
    with pytest.raises(ValueError, match="status/promotable"):
        publish_checkpoint(
            tmp_path / "bad.pt",
            trainer_state={"optimizerStep": 0},
            bindings=_bindings(),
            status=status,  # type: ignore[arg-type]
            promotable=promotable,
        )


def test_immutable_checkpoint_publication_never_clobbers_existing_bytes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "immutable.pt"
    destination.write_bytes(b"COMPETITOR")
    with pytest.raises(FileExistsError):
        publish_checkpoint(
            destination,
            trainer_state={"optimizerStep": 0},
            bindings=_bindings(),
            status="training",
            promotable=False,
            replace_existing=False,
        )
    assert destination.read_bytes() == b"COMPETITOR"


def test_checkpoint_reader_rejects_corrupt_status_matrix(tmp_path: Path) -> None:
    trainer = CoreTrainer(
        CoreGFM(node_classes=2), _graphs(), config=TrainingConfig.smoke(max_steps=1), seed=4
    )
    checkpoint = tmp_path / "corrupt-status.pt"
    trainer.save_checkpoint(checkpoint, bindings=_bindings())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["promotable"] = True
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="status/promotable"):
        load_checkpoint(checkpoint, expected_bindings=_bindings())


def test_source_alignment_selection_persists_in_config_checkpoint_and_result(
    tmp_path: Path,
) -> None:
    config = TrainingConfig.smoke_from_source_validation(
        SourceValidationScores(weight_0=0.6, weight_002=0.9, weight_005=0.7), max_steps=1
    )
    trainer = CoreTrainer(CoreGFM(node_classes=2), _graphs(), config=config, seed=12)
    result = trainer.run_steps(1)[0]
    assert result.alignment_weight == 0.02
    checkpoint = tmp_path / "alignment.pt"
    trainer.save_checkpoint(checkpoint, bindings=_bindings())
    payload = load_checkpoint(checkpoint, expected_bindings=_bindings())
    assert payload["trainer"]["config"]["alignment_weight"] == 0.02
    assert payload["trainer"]["config"]["alignment_source_scores"] == (0.6, 0.9, 0.7)
    with pytest.raises(ValueError, match="selected source-validation candidate"):
        TrainingConfig(
            preset="smoke",
            min_steps=0,
            max_steps=1,
            alignment_weight=0.05,
            alignment_source_scores=(0.6, 0.9, 0.7),
        )


def test_bundle_adapter_is_optimized_and_checkpointed_with_model(tmp_path: Path) -> None:
    adapter = _bundle_adapter()
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )
    graph = TrainingGraph.from_bundle(
        adapter=adapter,
        graph=PreparedGraph.from_edge_index(num_nodes=4, edge_index=edge_index, directed=False),
    )
    before = adapter.adapters["field_0"].projection.weight.detach().clone()
    trainer = CoreTrainer(
        CoreGFM(node_classes=2),
        {"bundle": graph},
        config=TrainingConfig.smoke(max_steps=1),
        seed=8,
    )
    trainer.run_steps(1)
    assert not torch.equal(before, adapter.adapters["field_0"].projection.weight)
    checkpoint = tmp_path / "adapter.pt"
    trainer.save_checkpoint(checkpoint, bindings=_bindings())
    payload = load_checkpoint(checkpoint, expected_bindings=_bindings())
    assert payload["trainer"]["adapterSchemas"]["bundle"]["schemaVersion"] == (
        "socialgraph-fm.core-adapter-schema/1.1"
    )
    assert all(not name.startswith("_field_") for name in payload["trainer"]["adapters"]["bundle"])

    replacement = _bundle_adapter()
    resumed = CoreTrainer(
        CoreGFM(node_classes=2),
        {"bundle": TrainingGraph.from_bundle(adapter=replacement, graph=graph.graph)},
        config=TrainingConfig.smoke(max_steps=1),
        seed=9,
    )
    resumed.load_checkpoint(checkpoint, bindings=_bindings())
    _assert_nested_equal(adapter.state_dict(), replacement.state_dict())


def test_training_resume_accepts_legacy_adapter_rows_but_keeps_current_graph_rows() -> None:
    adapter = _bundle_adapter()
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )
    graph = TrainingGraph.from_bundle(
        adapter=adapter,
        graph=PreparedGraph.from_edge_index(num_nodes=4, edge_index=edge_index, directed=False),
    )
    trainer = CoreTrainer(
        CoreGFM(node_classes=2),
        {"bundle": graph},
        config=TrainingConfig.smoke(max_steps=1),
        seed=8,
    )
    state = copy.deepcopy(trainer.state_dict())
    state.pop("adapterSchemas")
    state["adapters"]["bundle"]["_field_0_values"] = torch.full((99, 1), 777.0)
    replacement = _bundle_adapter()
    current_rows = replacement._field_0_values.clone()
    resumed = CoreTrainer(
        CoreGFM(node_classes=2),
        {"bundle": TrainingGraph.from_bundle(adapter=replacement, graph=graph.graph)},
        config=TrainingConfig.smoke(max_steps=1),
        seed=9,
    )

    resumed.load_state_dict(state)

    assert torch.equal(replacement._field_0_values, current_rows)


def test_neighbor_batched_bundle_adapter_keeps_field_objective_gradients() -> None:
    adapter = _bundle_adapter()
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )
    policy = ExecutionPolicy(
        full_batch_edge_threshold=1,
        node_batch_size=2,
        edge_batch_size=2,
        fanout=(1, 0, 0),
    )
    graph = TrainingGraph.from_bundle(
        adapter=adapter,
        graph=PreparedGraph.from_edge_index(num_nodes=4, edge_index=edge_index, directed=False),
        execution_policy=policy,
    )
    before = adapter.adapters["field_0"].projection.weight.detach().clone()
    trainer = CoreTrainer(
        CoreGFM(node_classes=2),
        {"bundle": graph},
        config=TrainingConfig.smoke(max_steps=1),
        seed=8,
    )
    result = trainer.run_steps(1)[0]
    assert result.execution_mode == "neighbor"
    assert not torch.equal(before, adapter.adapters["field_0"].projection.weight)


def test_masked_bundle_loss_ignores_unselected_field_targets() -> None:
    adapter = _bundle_adapter()
    altered = _bundle_adapter()
    altered._field_1_indices.fill_(7)
    altered.load_state_dict(
        {
            key: value
            for key, value in adapter.state_dict().items()
            if "_field_1_indices" not in key
        },
        strict=False,
    )
    decoded = torch.randn(4, 2, 128)
    field_mask = torch.tensor(
        [[True, False], [True, False], [False, False], [False, False]], dtype=torch.bool
    )
    generator = torch.Generator().manual_seed(5)
    first = adapter.reconstruction_loss(decoded, field_mask, generator=generator)
    generator.manual_seed(5)
    second = altered.reconstruction_loss(decoded, field_mask, generator=generator)
    assert torch.equal(first, second)


def test_aggregate_parameter_budget_covers_many_domains_and_fields() -> None:
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    prepared = PreparedGraph.from_edge_index(num_nodes=2, edge_index=edge_index, directed=False)
    graphs = {
        f"domain-{index}": TrainingGraph.from_bundle(
            adapter=_many_field_adapter(10), graph=prepared
        )
        for index in range(10)
    }
    trainer = CoreTrainer(
        CoreGFM(node_classes=2), graphs, config=TrainingConfig.smoke(max_steps=1), seed=4
    )
    assert trainer.trainable_parameter_count < 5_000_000

    too_many = {
        f"domain-{index}": TrainingGraph.from_bundle(
            adapter=_many_field_adapter(10), graph=prepared
        )
        for index in range(20)
    }
    with pytest.raises(ValueError, match="5,000,000"):
        CoreTrainer(
            CoreGFM(node_classes=2),
            too_many,
            config=TrainingConfig.smoke(max_steps=1),
            seed=4,
        )


@pytest.mark.parametrize("loader_kind", ["node", "link"])
def test_trainer_consumes_interleaved_neighbor_batches_and_restores_cursor(
    loader_kind: str,
) -> None:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]],
        dtype=torch.long,
    )
    policy = ExecutionPolicy(
        full_batch_edge_threshold=1,
        node_batch_size=2,
        edge_batch_size=2,
        fanout=(1, 0, 0),
    )
    graphs = {
        domain: TrainingGraph(
            graph=PreparedGraph.from_edge_index(
                num_nodes=6, edge_index=edge_index.clone(), directed=False
            ),
            features=torch.randn(6, 128, generator=torch.Generator().manual_seed(index)),
            execution_policy=policy,
            loader_kind=loader_kind,  # type: ignore[arg-type]
            edge_label_index=edge_index[:, :6] if loader_kind == "link" else None,
        )
        for index, domain in enumerate(("alpha", "beta"))
    }
    trainer = CoreTrainer(
        CoreGFM(node_classes=2), graphs, config=TrainingConfig.smoke(max_steps=4), seed=6
    )
    trainer.run_steps(3)
    assert [result.domain for result in trainer.history[:2]] in (
        ["alpha", "beta"],
        ["beta", "alpha"],
    )
    assert all(result.execution_mode == "neighbor" for result in trainer.history)
    assert all(result.batch_num_nodes < 6 for result in trainer.history)
    state = copy.deepcopy(trainer.state_dict())
    expected = trainer.run_steps(1)[0]

    resumed = CoreTrainer(
        CoreGFM(node_classes=2), graphs, config=TrainingConfig.smoke(max_steps=4), seed=999
    )
    resumed.load_state_dict(state)
    actual = resumed.run_steps(1)[0]
    assert (actual.domain, actual.batch_index, actual.batch_ordinal) == (
        expected.domain,
        expected.batch_index,
        expected.batch_ordinal,
    )
    assert actual.objective_signature == expected.objective_signature


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable in this runtime")
def test_cuda_loader_smoke_and_fresh_process_resume_stay_below_memory_target(
    tmp_path: Path,
    recwarn: pytest.WarningsRecorder,
) -> None:
    device = torch.device("cuda")
    resume_config = replace(TrainingConfig.smoke(max_steps=6), gradient_accumulation=2)

    torch.manual_seed(900)
    reference = CoreTrainer(
        CoreGFM(node_classes=2).to(device),
        _resume_graphs(device),
        config=resume_config,
        seed=31,
    )
    reference.run_steps(2)
    expected = _step_summary(reference, reference.history[-1])

    torch.manual_seed(900)
    interrupted = CoreTrainer(
        CoreGFM(node_classes=2).to(device),
        _resume_graphs(device),
        config=resume_config,
        seed=31,
    )
    interrupted.run_steps(1)
    checkpoint = tmp_path / "cuda-resume.pt"
    interrupted.save_checkpoint(checkpoint, bindings=_bindings())
    code = (
        "import json,runpy,sys; ns=runpy.run_path(sys.argv[2]); "
        "print(json.dumps(ns['_fresh_process_step_summary'](sys.argv[1], 'cuda'),sort_keys=True))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(checkpoint), str(Path(__file__))],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == expected

    loader_policy = ExecutionPolicy(
        full_batch_edge_threshold=1,
        node_batch_size=2,
        edge_batch_size=2,
        fanout=(1, 0, 0),
    )
    loader_graphs = {
        domain: TrainingGraph(
            features=value.features.to(device),
            graph=value.graph.to(device),
            execution_policy=loader_policy,
        )
        for domain, value in _graphs().items()
        if value.features is not None
    }
    torch.cuda.reset_peak_memory_stats(device)
    trainer = CoreTrainer(
        CoreGFM(node_classes=2).to(device),
        loader_graphs,
        config=TrainingConfig.smoke(max_steps=2),
        seed=31,
    )
    trainer.run_steps(2)
    assert all(result.execution_mode == "neighbor" for result in trainer.history)
    assert torch.cuda.max_memory_allocated(device) < 6.5 * 1024**3
    assert not any("lr_scheduler.step() before" in str(item.message) for item in recwarn)
