from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable, cast

import pytest
import torch

import socialgraph_gfm.core.formal_preflight as preflight_module
from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.baselines import fixed_gfm_family_specs
from socialgraph_gfm.core.experiments import (
    ExperimentExecutionConfigEvidence,
    ExperimentArtifactRef,
    ExperimentLedger,
    ExperimentProtocol,
    ExperimentRecipeEvidence,
    ExperimentRunRecord,
    ExperimentTrainingDataEvidence,
    ResourceTelemetryEvidence,
    ResourceTelemetrySample,
    aggregate_experiment,
    build_experiment_matrix,
    derive_transfer_advantage,
)
from socialgraph_gfm.core.config import TrainingConfig
from socialgraph_gfm.core.metrics import TaskMetricSet
from socialgraph_gfm.core.resource_telemetry import ResourceTelemetryRecorder
from socialgraph_gfm.core.telemetry_receipt import (
    OperatorTelemetryCapability,
    TelemetryReceipt,
    TelemetryReceiptSigner,
)
from socialgraph_gfm.tensor_digest import canonical_tensor_digest


_TEST_OPERATOR = OperatorTelemetryCapability.from_secret(
    key_id="experiment-runner-unit", secret=bytes(range(32))
)
_TEST_SIGNER = TelemetryReceiptSigner(_TEST_OPERATOR)


def _issue_receipt(telemetry, cell, *, fold_id: str = "cell-run") -> TelemetryReceipt:
    issue = cast(Callable[..., TelemetryReceipt], _TEST_SIGNER.issue)
    arguments: dict[str, Any] = {
        "telemetry": telemetry,
        "fold_id": fold_id,
    }
    if "composite_state_hash" in inspect.signature(_TEST_SIGNER.issue).parameters:
        arguments.update(
            {
                "composite_state_hash": canonical_sha256(
                    {"cellId": cell.cell_id, "role": "unit-composite-state"}
                ),
                "recovery_state_hash": canonical_sha256(
                    {"cellId": cell.cell_id, "role": "unit-recovery-state"}
                ),
            }
        )
    return issue(**arguments)


def _artifacts(
    cell,
    metrics,
    *,
    latest: str | None,
    best: str | None,
    recipe_hash: str,
    config_hash: str,
    telemetry_hash: str,
    training_data_hash: str,
    head_data_hash: str | None,
    fold_inventory_hash: str | None = None,
    telemetry_receipt_hash: str | None = None,
):
    semantics = {
        "adapter-schema": "b" * 64,
        "code": "8" * 64,
        "configuration": config_hash,
        "dataset-manifest": "5" * 64,
        "environment": "9" * 64,
        "experiment-recipe": recipe_hash,
        "labels": "c" * 64,
        "predictions": metrics.prediction_hash,
        "resource-telemetry": telemetry_hash,
        "split-manifest": "6" * 64,
        "split-inventory": "0" * 64,
        "structure-cache": "a" * 64,
        "targets": metrics.target_hash,
        "training-data": training_data_hash,
    }
    if metrics.threshold_hash is not None:
        semantics["threshold"] = metrics.threshold_hash
    if fold_inventory_hash is not None:
        semantics["fold-evaluation-inventory"] = fold_inventory_hash
    if telemetry_receipt_hash is not None:
        semantics["telemetry-receipt"] = telemetry_receipt_hash
    if cell.trainable:
        assert head_data_hash is not None
        semantics["head-data"] = head_data_hash
        semantics["head-report"] = "d" * 64
        if cell.task_id != "penn94.community-resilience":
            semantics["calibration-report"] = "e" * 64
        assert latest is not None and best is not None
        semantics["latest-checkpoint"] = latest
        semantics["best-checkpoint"] = best
    return tuple(
        ExperimentArtifactRef(
            role=role,
            relativePath=f"artifacts/{cell.cell_id}/{role}.bin",
            byteSha256=(
                semantic
                if role in {"latest-checkpoint", "best-checkpoint"}
                else hashlib.sha256(role.encode()).hexdigest()
            ),
            semanticHash=semantic,
            sizeBytes=1,
        )
        for role, semantic in sorted(semantics.items())
    )


def _record(
    cell,
    *,
    value: float,
    elapsed: float = 10.0,
    wait: float = 1.0,
    best: str | None = None,
    bound: bool = True,
    with_receipt: bool = True,
    bind_receipt_artifact: bool = True,
    tamper_receipt: bool = False,
    mismatched_receipt: bool = False,
    receipt_checkpoint_mismatch: bool = False,
    receipt_fold_id: str = "cell-run",
):
    trainable = cell.trainable
    manifest_hashes = {
        graph_id: hashlib.sha256(graph_id.encode()).hexdigest()
        for graph_id in (
            *cell.pretraining_graph_ids,
            cell.target_graph_id,
            *((cell.validation_graph_id,) if cell.validation_graph_id is not None else ()),
        )
    }
    recipe = ExperimentRecipeEvidence.create(cell=cell, manifest_hashes=manifest_hashes)
    execution = ExperimentExecutionConfigEvidence.create(
        cell=cell,
        recipe=recipe,
        training_config=(
            TrainingConfig.formal(max_steps=2_000, min_steps=2_000) if trainable else None
        ),
    )
    head_data_hash = "f" * 64 if trainable else None
    training_data = ExperimentTrainingDataEvidence.create(
        cell=cell,
        recipe=recipe,
        target_split_inventory_hash="0" * 64,
        head_data_hash=head_data_hash,
    )
    checkpoint = f"{cell.seed:064x}"[-64:] if trainable else None
    best_checkpoint = best or checkpoint
    fold_inventory_hash = canonical_sha256({"cellId": cell.cell_id, "fixture": "unit"})
    if elapsed == 10.0 and wait == 1.0 and trainable:
        state = {"weight": torch.tensor([float(cell.seed)])}
        model_hash = canonical_sha256(
            {name: canonical_tensor_digest(value) for name, value in state.items()}
        )

        def fit_state(step: int):
            payload = {
                "optimizerStep": step,
                "checkpointModelStateHash": model_hash,
            }
            payload["stateHash"] = canonical_sha256(payload)
            return payload

        def sealed(run_id: str, latest_hash: str, best_hash: str):
            recorder = ResourceTelemetryRecorder(
                cell_id=cell.cell_id,
                run_id=run_id,
                phase="formal",
                config_hash=execution.config_hash,
                data_hash=training_data.inventory_hash,
                code_hash="8" * 64,
                environment_hash="9" * 64,
            )
            recorder.record_start(model_state=state, fit_state=fit_state(0))
            for step in range(250, 2_000, 250):
                recorder.record_checkpoint(
                    optimizer_step=step,
                    model_state=state,
                    fit_state=fit_state(step),
                )
            return recorder.finish(
                final_optimizer_step=2_000,
                model_state=state,
                fit_state=fit_state(2_000),
                latest_checkpoint_semantic_hash=latest_hash,
                best_checkpoint_semantic_hash=best_hash,
            )

        assert checkpoint is not None and best_checkpoint is not None
        telemetry = sealed(
            f"unit-{cell.cell_id[:16]}", checkpoint, best_checkpoint
        )
        receipt_telemetry = telemetry
        if mismatched_receipt:
            receipt_telemetry = sealed(
                f"mismatch-{cell.cell_id[:16]}", checkpoint, best_checkpoint
            )
        elif receipt_checkpoint_mismatch:
            receipt_telemetry = sealed(
                f"checkpoint-mismatch-{cell.cell_id[:8]}", "a" * 64, "0" * 64
            )
        receipt = (
            _issue_receipt(receipt_telemetry, cell, fold_id=receipt_fold_id)
            if with_receipt
            else None
        )
        if tamper_receipt:
            assert receipt is not None
            receipt = receipt.model_copy(update={"fold_id": "caller-tampered-fold"})
    else:
        telemetry = ResourceTelemetryEvidence.create(
            cell_id=cell.cell_id,
            phase="formal",
            samples=(
                ResourceTelemetrySample(
                    monotonicSeconds=10.0,
                    cumulativeDataWaitSeconds=0.0,
                    optimizerStep=0,
                    cudaAllocatedBytes=0,
                ),
                ResourceTelemetrySample(
                    monotonicSeconds=10.0 + elapsed,
                    cumulativeDataWaitSeconds=wait,
                    optimizerStep=2_000 if trainable else 0,
                    cudaAllocatedBytes=1_024,
                ),
            ),
        )
        receipt = None
    metrics = TaskMetricSet.create(
        task_id=cell.task_id,
        metrics={name: value for name in cell.required_metrics},
        prediction_hash="1" * 64,
        target_hash="2" * 64,
        threshold_hash="3" * 64,
    )
    return ExperimentRunRecord.create(
        cell=cell,
        phase="formal",
        preflight_evidence_hash="4" * 64,
        dataset_manifest_hash="5" * 64,
        split_manifest_hash="6" * 64,
        split_inventory_hash="0" * 64,
        evaluation_fold_ids=("primary",),
        recipe_hash=recipe.recipe_hash,
        config_hash=execution.config_hash,
        training_data_hash=training_data.inventory_hash,
        head_data_hash=head_data_hash,
        code_hash="8" * 64,
        environment_hash="9" * 64,
        structure_cache_hash="a" * 64,
        adapter_schema_hash="b" * 64,
        label_artifact_hash="c" * 64,
        head_report_hash="d" * 64 if trainable else None,
        calibration_hash="e" * 64 if trainable else None,
        checkpoint_sha256=checkpoint,
        best_checkpoint_sha256=best_checkpoint,
        telemetry=telemetry,
        telemetry_receipt=receipt,
        fold_evaluation_inventory_hash=fold_inventory_hash,
        metrics=metrics,
        artifacts=(
            _artifacts(
                cell,
                metrics,
                latest=checkpoint,
                best=best_checkpoint,
                recipe_hash=recipe.recipe_hash,
                config_hash=execution.config_hash,
                telemetry_hash=telemetry.telemetry_hash,
                training_data_hash=training_data.inventory_hash,
                head_data_hash=head_data_hash,
                fold_inventory_hash=fold_inventory_hash,
                telemetry_receipt_hash=(
                    receipt.receipt_hash
                    if receipt is not None and bind_receipt_artifact
                    else None
                ),
            )
            if bound
            else ()
        ),
    )


def _cells(protocol: ExperimentProtocol, task: str, method: str, budget: str):
    return tuple(
        cell
        for cell in build_experiment_matrix(protocol)
        if cell.task_id == task and cell.method_id == method and cell.label_budget == budget
    )


def test_recipe_and_resource_evidence_are_derived_from_fixed_cell_contract() -> None:
    protocol = ExperimentProtocol.fixed()
    cell = _cells(protocol, "tolokers.risk", "domain-aligned-gfm", "full")[0]
    manifest_hashes = {
        graph_id: hashlib.sha256(graph_id.encode()).hexdigest()
        for graph_id in (*cell.pretraining_graph_ids, cell.target_graph_id)
    }
    recipe = ExperimentRecipeEvidence.create(cell=cell, manifest_hashes=manifest_hashes)
    assert tuple(item.graph_id for item in recipe.source_manifests) == cell.pretraining_graph_ids
    assert recipe.target_manifest.graph_id == cell.target_graph_id
    assert recipe.family is not None
    assert recipe.family.alignment_candidates == (0.0, 0.02, 0.05)
    config = TrainingConfig.formal(max_steps=2_000, min_steps=2_000)
    execution = ExperimentExecutionConfigEvidence.create(
        cell=cell,
        recipe=recipe,
        training_config=config,
    )
    assert execution.training_config == config.to_dict()

    telemetry = ResourceTelemetryEvidence.create(
        cell_id=cell.cell_id,
        phase="formal",
        samples=(
            ResourceTelemetrySample(
                monotonicSeconds=10.0,
                cumulativeDataWaitSeconds=2.0,
                optimizerStep=0,
                cudaAllocatedBytes=100,
            ),
            ResourceTelemetrySample(
                monotonicSeconds=20.0,
                cumulativeDataWaitSeconds=3.0,
                optimizerStep=2_000,
                cudaAllocatedBytes=1_024,
            ),
        ),
    )
    assert telemetry.elapsed_seconds == 10.0
    assert telemetry.data_wait_seconds == 1.0
    assert telemetry.optimizer_steps == 2_000
    assert telemetry.peak_cuda_bytes == 1_024

    forged = telemetry.model_dump(mode="json", by_alias=True)
    forged["elapsedSeconds"] = 9.0
    forged["telemetryHash"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "telemetryHash"}
    )
    with pytest.raises(ValueError, match="derived"):
        ResourceTelemetryEvidence.model_validate(forged)


def test_fixed_protocol_matrix_is_deterministic_graph_disjoint_and_label_safe() -> None:
    protocol = ExperimentProtocol.fixed()
    assert protocol.seeds == (20260821, 20260822, 20260823, 20260824, 20260825)
    assert protocol.label_budgets == ("1", "5", "20", "full")
    assert {method.method_id for method in protocol.methods} == {
        "attribute-mlp",
        "structure-only",
        "common-neighbors",
        "adamic-adar",
        "graphsage-scratch",
        "linkx",
        "graphmae2-single",
        "multi-graph-shared-gfm",
        "domain-aligned-gfm",
    }
    first = build_experiment_matrix(protocol)
    assert first == build_experiment_matrix(ExperimentProtocol.fixed())
    assert len({cell.cell_id for cell in first}) == len(first)
    assert all(cell.target_graph_id not in cell.pretraining_graph_ids for cell in first)
    assert all(
        cell.validation_graph_id is None
        or cell.validation_graph_id not in cell.pretraining_graph_ids
        for cell in first
    )
    assert all(cell.target_labels_in_pretraining is False for cell in first)
    assert all(
        cell.target_unlabeled_adaptation
        == (cell.method_id in {"graphmae2-single", "multi-graph-shared-gfm", "domain-aligned-gfm"})
        for cell in first
    )
    assert not _cells(protocol, "email.relation-completion", "attribute-mlp", "1")
    assert _cells(protocol, "email.relation-completion", "common-neighbors", "1")
    assert len([task for task in protocol.tasks if task.task_id.startswith("twitch.")]) == 6
    methods = {method.method_id: method for method in protocol.methods}
    for family in fixed_gfm_family_specs():
        assert (
            methods[family.method_id].target_unlabeled_adaptation
            == family.target_unlabeled_adaptation
        )

    forged = protocol.model_dump(mode="json", by_alias=True)
    removed = forged["methods"].pop()["methodId"]
    for task in forged["tasks"]:
        task["applicableMethods"] = [
            method for method in task["applicableMethods"] if method != removed
        ]
    forged["protocolHash"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "protocolHash"}
    )
    with pytest.raises(ValueError, match="fixed formal protocol"):
        ExperimentProtocol.model_validate(forged)


def test_ledger_is_immutable_exact_replay_and_resource_failures_are_preserved(
    tmp_path: Path,
) -> None:
    protocol = ExperimentProtocol.fixed()
    cell = _cells(protocol, "tolokers.risk", "graphsage-scratch", "1")[0]
    good = _record(cell, value=0.7)
    ledger = ExperimentLedger(tmp_path)
    path = ledger.publish_run(good)
    assert ledger.publish_run(good) == path
    assert ledger.load_run(cell.cell_id) == good

    conflict = _record(cell, value=0.8)
    with pytest.raises(FileExistsError, match="conflicting"):
        ledger.publish_run(conflict)
    assert ledger.load_run(cell.cell_id) == good

    slow_cell = _cells(protocol, "tolokers.risk", "graphsage-scratch", "5")[0]
    slow = _record(slow_cell, value=0.8, elapsed=21_601.0, wait=1.0)
    assert slow.promotable is False
    assert "elapsed-time" in slow.failed_gates
    assert ledger.publish_run(slow).exists()

    wait_cell = _cells(protocol, "tolokers.risk", "graphsage-scratch", "20")[0]
    waiting = _record(wait_cell, value=0.8, elapsed=10.0, wait=2.0)
    assert waiting.promotable is False
    assert "data-wait-ratio" in waiting.failed_gates

    partial_metrics = TaskMetricSet.create(
        task_id=cell.task_id,
        metrics={cell.primary_metric: 0.7},
        prediction_hash="1" * 64,
        target_hash="2" * 64,
        threshold_hash="3" * 64,
    )
    partial = ExperimentRunRecord.create(
        cell=cell,
        phase="formal",
        preflight_evidence_hash="4" * 64,
        dataset_manifest_hash="5" * 64,
        split_manifest_hash="6" * 64,
        split_inventory_hash="0" * 64,
        evaluation_fold_ids=("primary",),
        recipe_hash=_record(cell, value=0.7).recipe_hash,
        config_hash=_record(cell, value=0.7).config_hash,
        training_data_hash=_record(cell, value=0.7).training_data_hash,
        head_data_hash="f" * 64,
        code_hash="8" * 64,
        environment_hash="9" * 64,
        structure_cache_hash="a" * 64,
        adapter_schema_hash="b" * 64,
        label_artifact_hash="c" * 64,
        head_report_hash="d" * 64,
        calibration_hash="e" * 64,
        checkpoint_sha256=f"{cell.seed:064x}"[-64:],
        best_checkpoint_sha256=f"{cell.seed:064x}"[-64:],
        telemetry=ResourceTelemetryEvidence.create(
            cell_id=cell.cell_id,
            phase="formal",
            samples=(
                ResourceTelemetrySample(
                    monotonicSeconds=0.0,
                    cumulativeDataWaitSeconds=0.0,
                    optimizerStep=0,
                    cudaAllocatedBytes=0,
                ),
                ResourceTelemetrySample(
                    monotonicSeconds=10.0,
                    cumulativeDataWaitSeconds=1.0,
                    optimizerStep=2_000,
                    cudaAllocatedBytes=1_024,
                ),
            ),
        ),
        metrics=partial_metrics,
        artifacts=_artifacts(
            cell,
            partial_metrics,
            latest=f"{cell.seed:064x}"[-64:],
            best=f"{cell.seed:064x}"[-64:],
            recipe_hash=_record(cell, value=0.7).recipe_hash,
            config_hash=_record(cell, value=0.7).config_hash,
            telemetry_hash=ResourceTelemetryEvidence.create(
                cell_id=cell.cell_id,
                phase="formal",
                samples=(
                    ResourceTelemetrySample(
                        monotonicSeconds=0.0,
                        cumulativeDataWaitSeconds=0.0,
                        optimizerStep=0,
                        cudaAllocatedBytes=0,
                    ),
                    ResourceTelemetrySample(
                        monotonicSeconds=10.0,
                        cumulativeDataWaitSeconds=1.0,
                        optimizerStep=2_000,
                        cudaAllocatedBytes=1_024,
                    ),
                ),
            ).telemetry_hash,
            training_data_hash=_record(cell, value=0.7).training_data_hash,
            head_data_hash="f" * 64,
        ),
    )
    assert partial.promotable is False
    assert "metric-inventory" in partial.failed_gates


def test_ledger_precommit_failure_never_publishes_partial_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = ExperimentProtocol.fixed()
    cell = _cells(protocol, "tolokers.risk", "graphsage-scratch", "1")[0]
    record = _record(cell, value=0.7)
    ledger = ExperimentLedger(tmp_path)

    def fail_before_commit(kind: str, _target: Path) -> None:
        if kind == "evidence":
            raise OSError("injected precommit failure")

    monkeypatch.setattr(preflight_module, "_PUBLICATION_SEAM", fail_before_commit)
    with pytest.raises(OSError, match="injected"):
        ledger.publish_run(record)
    assert not ledger._path(cell.cell_id).exists()
    assert not tuple(ledger.root.glob(f".{cell.cell_id}.json.*"))


def test_latest_control_and_immutable_best_are_distinct_bound_artifacts() -> None:
    protocol = ExperimentProtocol.fixed()
    cell = _cells(protocol, "tolokers.risk", "graphsage-scratch", "1")[0]
    best = "0" * 64
    record = _record(cell, value=0.7, best=best)
    assert record.checkpoint_sha256 != record.best_checkpoint_sha256
    assert record.promotable is True
    refs = {item.role: item for item in record.artifacts}
    assert refs["latest-checkpoint"].byte_sha256 == record.checkpoint_sha256
    assert refs["best-checkpoint"].byte_sha256 == best


def test_legacy_self_reported_formal_telemetry_is_never_promotable() -> None:
    protocol = ExperimentProtocol.fixed()
    cell = _cells(protocol, "tolokers.risk", "graphsage-scratch", "1")[0]

    record = _record(cell, value=0.7, elapsed=10.0, wait=0.5)

    assert record.telemetry_evidence_scope == "legacy-unverified"
    assert record.promotable is False
    assert "resource-telemetry-unverified" in record.failed_gates
    assert "resource-telemetry-receipt" in record.failed_gates


def test_sealed_formal_telemetry_without_operator_receipt_is_not_promotable() -> None:
    protocol = ExperimentProtocol.fixed()
    cell = _cells(protocol, "tolokers.risk", "graphsage-scratch", "1")[0]

    record = _record(cell, value=0.7, with_receipt=False)

    assert record.telemetry_evidence_scope == "sealed-runtime"
    assert record.telemetry_receipt_hash is None
    assert record.promotable is False
    assert record.failed_gates == ("resource-telemetry-receipt",)


def test_receipt_requires_its_exact_artifact_inventory_reference() -> None:
    protocol = ExperimentProtocol.fixed()
    cell = _cells(protocol, "tolokers.risk", "graphsage-scratch", "1")[0]

    record = _record(cell, value=0.7, bind_receipt_artifact=False)

    assert record.telemetry_receipt_hash is not None
    assert "artifact-inventory" in record.failed_gates
    assert record.promotable is False


def test_model_copy_cannot_bypass_exact_receipt_revalidation() -> None:
    protocol = ExperimentProtocol.fixed()
    cell = _cells(protocol, "tolokers.risk", "graphsage-scratch", "1")[0]

    with pytest.raises(ValueError, match="receiptHash"):
        _record(cell, value=0.7, tamper_receipt=True)


def test_top_level_run_rejects_a_valid_non_cell_receipt() -> None:
    protocol = ExperimentProtocol.fixed()
    cell = _cells(protocol, "tolokers.risk", "graphsage-scratch", "1")[0]

    with pytest.raises(ValueError, match="does not bind"):
        _record(cell, value=0.7, receipt_fold_id="tolokers::official-00")


@pytest.mark.parametrize(
    "record_kwargs",
    [{"mismatched_receipt": True}, {"receipt_checkpoint_mismatch": True}],
    ids=["canonical-telemetry", "checkpoint-identities"],
)
def test_receipt_must_bind_the_exact_run_and_checkpoint_identities(record_kwargs) -> None:
    protocol = ExperimentProtocol.fixed()
    cell = _cells(protocol, "tolokers.risk", "graphsage-scratch", "1")[0]

    with pytest.raises(ValueError, match="does not bind"):
        _record(cell, value=0.7, **record_kwargs)


def test_aggregate_requires_exact_five_seeds_and_is_deterministic() -> None:
    protocol = ExperimentProtocol.fixed()
    cells = _cells(protocol, "tolokers.risk", "graphsage-scratch", "1")
    records = tuple(_record(cell, value=0.60 + index * 0.01) for index, cell in enumerate(cells))
    aggregate = aggregate_experiment(protocol, records)
    assert aggregate.seeds == protocol.seeds
    assert aggregate.mean == pytest.approx(0.62)
    assert aggregate.sample_std > 0
    assert aggregate == aggregate_experiment(protocol, tuple(reversed(records)))
    with pytest.raises(ValueError, match="five seeds"):
        aggregate_experiment(protocol, records[:-1])
    unbound = tuple(_record(cell, value=0.7, bound=False) for cell in cells)
    assert all(not record.promotable for record in unbound)
    with pytest.raises(ValueError, match="hash-bound"):
        aggregate_experiment(protocol, unbound)


def test_transfer_gate_requires_three_budgets_point_zero_two_and_positive_ci() -> None:
    protocol = ExperimentProtocol.fixed()
    scratch = []
    shared = []
    improvements = {"1": 0.04, "5": 0.03, "20": 0.02, "full": -0.005}
    for budget in protocol.label_budgets:
        for scratch_cell, shared_cell in zip(
            _cells(protocol, "tolokers.risk", "graphsage-scratch", budget),
            _cells(protocol, "tolokers.risk", "multi-graph-shared-gfm", budget),
            strict=True,
        ):
            scratch.append(_record(scratch_cell, value=0.60))
            shared.append(_record(shared_cell, value=0.60 + improvements[budget]))
    decision = derive_transfer_advantage(protocol, scratch, shared)
    assert decision.winning_budget_count == 3
    assert decision.aggregate_improvement >= 0.02
    assert decision.ci_lower > 0
    assert decision.transfer_advantage is True
    assert decision == derive_transfer_advantage(
        protocol, list(reversed(scratch)), list(reversed(shared))
    )

    weak = [_record(record.cell, value=0.605) for record in shared]
    rejected = derive_transfer_advantage(protocol, scratch, weak)
    assert rejected.transfer_advantage is False


def test_transfer_bootstrap_resamples_four_budget_vector_by_seed() -> None:
    protocol = ExperimentProtocol.fixed()
    scratch = []
    shared = []
    # Four budgets are perfectly correlated within seed.  Flattening the 20
    # cells makes the interval spuriously narrow; resampling five seed vectors
    # correctly retains the one adverse seed in the lower tail.
    gains = dict(zip(protocol.seeds, (-0.04, 0.04, 0.04, 0.04, 0.04), strict=True))
    for budget in protocol.label_budgets:
        for scratch_cell, shared_cell in zip(
            _cells(protocol, "tolokers.risk", "graphsage-scratch", budget),
            _cells(protocol, "tolokers.risk", "multi-graph-shared-gfm", budget),
            strict=True,
        ):
            scratch.append(_record(scratch_cell, value=0.60))
            shared.append(_record(shared_cell, value=0.60 + gains[shared_cell.seed]))
    decision = derive_transfer_advantage(protocol, scratch, shared)
    assert decision.winning_budget_count == 4
    assert decision.aggregate_improvement == pytest.approx(0.024)
    assert decision.ci_lower < 0.0
    assert decision.transfer_advantage is False
