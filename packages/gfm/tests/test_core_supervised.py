from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.calibration import (
    BinaryScoreSemantics,
    CalibrationProtocol,
    ValidationScoreBatch,
    VerifiedValidationScores,
    derive_validation_scores,
    fit_score_calibration,
    fit_score_calibration_report,
)
from socialgraph_gfm.core.adapters import BundleInputAdapter
from socialgraph_gfm.core.bundle import (
    CoreGraphBundle,
    calculate_graph_version_hash,
)
from socialgraph_gfm.core.model import CoreGFM
from socialgraph_gfm.core.inference_contracts import GfmRunRequest
from socialgraph_gfm.core.serving_registry import ScoreCalibration
from socialgraph_gfm.core.serving_head import CoreServingHead
from socialgraph_gfm.core.supervised import (
    EncodedGraphProvenance,
    HeadTrainingConfig,
    HeadTrainingReport,
    SupervisedPartition,
    SupervisedTestSet,
    SupervisedTrainValidation,
    VerifiedEncodedGraph,
    VerifiedHeadTrainingReport,
    encode_supervised_graph,
    fit_supervised_head,
    validate_supervised_test_isolation,
)


def _bundle(*, variant: bool = False, directed: bool = False) -> CoreGraphBundle:
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": directed,
        "nodes": [{"id": str(index), "index": index} for index in range(5)],
        "edges": [
            {"sourceId": "0", "targetId": "1", "edgeType": "edge"},
            {"sourceId": "1", "targetId": "2", "edgeType": "edge"},
            {"sourceId": "2", "targetId": "3", "edgeType": "edge"},
            {"sourceId": "3", "targetId": "4", "edgeType": "edge"},
            *([{"sourceId": "0", "targetId": "4", "edgeType": "edge"}] if variant else []),
        ],
        "nodeFeatures": [
            {
                "kind": "numeric",
                "name": "score",
                "values": [0.0, 1.0, 2.0, 3.0, 4.0],
            }
        ],
        "structuralFeatures": None,
        "source": {"sourceName": "fixture", "sourceSha256": "e" * 64},
        "splitManifest": {"strategy": "official", "assignments": []},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _official_node_bundle() -> CoreGraphBundle:
    payload = _bundle().model_dump(mode="json", by_alias=True)
    payload["splitManifest"] = {
        "strategy": "official",
        "assignments": [
            {"entityId": "0", "role": "train"},
            {"entityId": "1", "role": "validation"},
            {"entityId": "2", "role": "validation"},
            {"entityId": "3", "role": "test"},
            {"entityId": "4", "role": "test"},
        ],
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _encoding(
    model: CoreGFM, *, variant: bool = False, directed: bool = False
) -> VerifiedEncodedGraph:
    bundle = _bundle(variant=variant, directed=directed)
    adapter = BundleInputAdapter(bundle, mode="training")
    return encode_supervised_graph(model, bundle, adapter)


def _partition(
    _prefix: str,
    *,
    node_indices: tuple[int, ...] = (),
    edge_pairs: tuple[tuple[int, int], ...] = (),
    targets: tuple[int | float, ...],
) -> SupervisedPartition:
    entity_ids = (
        tuple(str(index) for index in node_indices)
        if node_indices
        else tuple(f"edge:{min(left, right)}:{max(left, right)}" for left, right in edge_pairs)
    )
    return SupervisedPartition(
        entityIds=entity_ids,
        nodeIndices=node_indices,
        edgePairs=edge_pairs,
        targets=targets,
    )


def _data(task_kind: str, provenance: EncodedGraphProvenance) -> SupervisedTrainValidation:
    if task_kind in {"node-binary", "node-multiclass", "resilience-regression"}:
        train_targets: tuple[int | float, ...]
        validation_targets: tuple[int | float, ...]
        if task_kind == "resilience-regression":
            train_targets = (1.0, 1.0)
            validation_targets = (-1.0, -1.0)
        else:
            train_targets = (0, 1)
            validation_targets = (0, 1)
        train = _partition("train", node_indices=(0, 1), targets=train_targets)
        validation = _partition("validation", node_indices=(2, 3), targets=validation_targets)
    else:
        train = _partition("train", edge_pairs=((0, 1), (1, 2)), targets=(1, 1))
        validation = _partition("validation", edge_pairs=((2, 3), (3, 4)), targets=(0, 0))
    return SupervisedTrainValidation.create(
        task_kind=task_kind,  # type: ignore[arg-type]
        provenance=provenance,
        train=train,
        validation=validation,
    )


def test_train_validation_contract_has_no_test_labels_and_rejects_overlap() -> None:
    encoding = _encoding(CoreGFM(node_classes=2))
    data = _data("node-binary", encoding.provenance)
    assert not hasattr(data, "test")
    assert data.train.entity_ids == ("0", "1")
    payload = data.model_dump(mode="json", by_alias=True)
    payload["validation"]["entityIds"][0] = "0"
    payload["dataHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "dataHash"}
    )
    with pytest.raises(ValidationError, match="disjoint"):
        SupervisedTrainValidation.model_validate(payload)

    test = SupervisedTestSet.create(
        task_kind="node-binary",
        provenance=encoding.provenance,
        test=_partition("test", node_indices=(4,), targets=(1,)),
    )
    assert test.test.entity_ids == ("4",)
    assert validate_supervised_test_isolation(data, test, bundle=encoding.bundle)


def test_signed_reciprocal_pairs_cannot_cross_train_validation_or_test() -> None:
    encoding = _encoding(CoreGFM(node_classes=2), directed=True)
    train = SupervisedPartition(entityIds=("edge:0:1",), edgePairs=((0, 1),), targets=(1,))
    validation = SupervisedPartition(entityIds=("edge:1:0",), edgePairs=((1, 0),), targets=(0,))
    with pytest.raises(ValidationError, match="unordered edge pairs"):
        SupervisedTrainValidation.create(
            task_kind="signed-edge",
            provenance=encoding.provenance,
            train=train,
            validation=validation,
        )

    data = _data("signed-edge", encoding.provenance)
    overlapping_test = SupervisedTestSet.create(
        task_kind="signed-edge",
        provenance=encoding.provenance,
        test=SupervisedPartition(entityIds=("edge:1:0",), edgePairs=((1, 0),), targets=(0,)),
    )
    with pytest.raises(ValueError, match="unordered edge pairs"):
        validate_supervised_test_isolation(data, overlapping_test, bundle=encoding.bundle)


@pytest.mark.parametrize(
    ("node_indices", "edge_pairs"),
    [((0, 0), ()), ((), ((0, 1), (0, 1)))],
)
def test_partition_rejects_duplicate_exact_locators(
    node_indices: tuple[int, ...],
    edge_pairs: tuple[tuple[int, int], ...],
) -> None:
    with pytest.raises(ValidationError, match="locators must be unique"):
        SupervisedPartition(
            entityIds=("entity-0", "entity-1"),
            nodeIndices=node_indices,
            edgePairs=edge_pairs,
            targets=(0, 1),
        )


@pytest.mark.parametrize(
    ("task_kind", "head_name"),
    [
        ("node-binary", "node_head"),
        ("edge-binary", "binary_link_head"),
        ("signed-edge", "signed_edge_head"),
        ("resilience-regression", "resilience_head"),
    ],
)
def test_real_governance_head_is_optimized_and_encoder_stays_frozen(
    task_kind: str, head_name: str
) -> None:
    torch.manual_seed(4)
    model = CoreGFM(node_classes=2)
    selected = getattr(model, head_name)
    before_head = copy.deepcopy(selected.state_dict())
    before_encoder = copy.deepcopy(model.encoder.state_dict())
    encoded = _encoding(model)
    report = fit_supervised_head(
        model,
        encoded,
        _data(task_kind, encoded.provenance),
        config=HeadTrainingConfig.smoke(max_steps=4, learning_rate=0.05),
    )

    assert report.head_name == head_name
    assert report.encoded_tensor_hash == encoded.provenance.encoded_tensor_hash
    assert report.graph_version_hash == encoded.bundle.graph_version_hash
    assert report.model_identity_hash == encoded.provenance.model_identity_hash
    assert report.encoding_artifact_hash == encoded.provenance.artifact_hash
    assert report.best_step in {1, 2, 3, 4}
    assert any(
        not torch.equal(before_head[name], value) for name, value in selected.state_dict().items()
    )
    assert all(
        torch.equal(before_encoder[name], value)
        for name, value in model.encoder.state_dict().items()
    )


def test_head_training_restores_validation_best_not_last_state() -> None:
    model = CoreGFM(node_classes=2)
    torch.nn.init.zeros_(model.node_head.weight)
    torch.nn.init.zeros_(model.node_head.bias)
    encoded = _encoding(model)
    report = fit_supervised_head(
        model,
        encoded,
        _data("node-binary", encoded.provenance),
        config=HeadTrainingConfig.smoke(max_steps=6, learning_rate=0.2),
    )

    assert report.best_step < 6
    assert report.best_metric == max(item.validation_metric for item in report.history)
    assert report.head_state_hash == report.calculate_current_head_hash(model)


def _trained_binary():
    torch.manual_seed(17)
    model = CoreGFM(node_classes=2)
    encoded = _encoding(model)
    data = _data("node-binary", encoded.provenance)
    report = fit_supervised_head(
        model,
        encoded,
        data,
        config=HeadTrainingConfig.smoke(max_steps=4, learning_rate=0.05),
    )
    semantics = BinaryScoreSemantics.for_task(data.task_kind)
    scores = derive_validation_scores(
        model,
        encoded,
        data,
        report,
        semantics=semantics,
    )
    return model, encoded, data, report, scores


def test_head_training_rejects_unverified_tensor_and_identity_substitution() -> None:
    model = CoreGFM(node_classes=2)
    encoded = _encoding(model)
    data = _data("node-binary", encoded.provenance)

    with pytest.raises(TypeError, match="VerifiedEncodedGraph"):
        fit_supervised_head(
            model,
            encoded.tensor,
            data,
            config=HeadTrainingConfig.smoke(max_steps=1),
        )

    replacement = CoreGFM(node_classes=2)
    before = copy.deepcopy(replacement.node_head.state_dict())
    with pytest.raises(ValueError, match="encoder identity"):
        fit_supervised_head(
            replacement,
            encoded,
            data,
            config=HeadTrainingConfig.smoke(max_steps=1),
        )
    assert all(
        torch.equal(before[name], value)
        for name, value in replacement.node_head.state_dict().items()
    )


@pytest.mark.parametrize("mutation", ["encoded", "encoder"])
def test_head_training_rejects_mutated_encoding_provenance(mutation: str) -> None:
    model = CoreGFM(node_classes=2)
    encoded = _encoding(model)
    data = _data("node-binary", encoded.provenance)
    if mutation == "encoded":
        encoded.tensor[0, 0] += 1.0
    else:
        next(model.encoder.parameters()).data.add_(1.0)

    with pytest.raises(ValueError, match="identity|encoded tensor"):
        fit_supervised_head(
            model,
            encoded,
            data,
            config=HeadTrainingConfig.smoke(max_steps=1),
        )


def test_head_training_rejects_adapter_and_graph_substitution() -> None:
    model = CoreGFM(node_classes=2)
    encoded = _encoding(model)
    data = _data("node-binary", encoded.provenance)
    next(encoded.adapter.parameters()).data.add_(1.0)
    with pytest.raises(ValueError, match="adapter state identity"):
        fit_supervised_head(
            model,
            encoded,
            data,
            config=HeadTrainingConfig.smoke(max_steps=1),
        )

    encoded = _encoding(model)
    other = _encoding(model, variant=True)
    other_data = _data("node-binary", other.provenance)
    with pytest.raises(ValueError, match="supervised data identity"):
        fit_supervised_head(
            model,
            encoded,
            other_data,
            config=HeadTrainingConfig.smoke(max_steps=1),
        )


def test_score_calibration_is_validation_only_and_serving_compatible() -> None:
    _model, _encoded, data, _head_report, scores = _trained_binary()
    protocol = CalibrationProtocol.fixed(scores)
    report = fit_score_calibration_report(scores, protocol=protocol)
    calibration = fit_score_calibration(scores, protocol=protocol)

    assert calibration == report.calibration
    assert report.after_nll <= report.before_nll + 1e-12
    assert report.validation_logits_hash
    assert report.validation_targets_hash
    assert (
        ScoreCalibration.model_validate(calibration.model_dump(mode="json", by_alias=True))
        == calibration
    )
    serving_logits = (scores.logits + calibration.bias) / calibration.temperature
    serving_nll = torch.nn.functional.binary_cross_entropy_with_logits(
        serving_logits, scores.targets
    ).item()
    assert report.after_nll == pytest.approx(serving_nll)
    held_out = SupervisedTestSet.create(
        task_kind="node-binary",
        provenance=scores.provenance,
        test=_partition("test", node_indices=(4,), targets=(1,)),
    )
    assert held_out.test_hash
    assert fit_score_calibration(scores, protocol=protocol) == calibration
    assert data.validation.partition_hash == report.validation_partition_hash


def test_node_binary_score_semantics_are_positive_minus_negative_logits() -> None:
    model, encoded, data, _report, scores = _trained_binary()
    locator = torch.tensor(data.validation.node_indices, dtype=torch.long)
    with torch.no_grad():
        logits = model.node_head(encoded.tensor[locator])
    assert torch.equal(scores.logits, (logits[:, 1] - logits[:, 0]).double())
    assert (
        scores.record.score_semantics_hash
        == BinaryScoreSemantics.for_task("node-binary").semantics_hash
    )
    with pytest.raises(ValueError, match="unavailable"):
        BinaryScoreSemantics.for_task("resilience-regression")


def test_serving_node_score_uses_the_same_selected_class_log_odds() -> None:
    bundle = _bundle()
    model = CoreGFM(node_classes=2)
    torch.nn.init.zeros_(model.node_head.weight)
    model.node_head.bias.data.copy_(torch.tensor([-1.0, 2.0]))
    request = GfmRunRequest.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-run-request/2.0",
            "graphVersionId": "graph-v1",
            "taskId": "core.risk_and_trust_review",
            "targetScope": {"kind": "risk-review", "nodeIds": ["0"], "edgeIds": []},
            "modelVersionId": "model-v1",
            "parameters": {"kind": "risk-and-trust", "topKSimilarCases": 1},
        }
    )
    model_record = SimpleNamespace(
        model_version_id="model-v1",
        model_version_hash="f" * 64,
    )
    head = SimpleNamespace(node_output_index=1)

    score = CoreServingHead._scores(
        request,
        bundle,
        model_record,
        head,
        model,
        torch.zeros((5, 128)),
    )[0]

    assert score.score == pytest.approx(3.0)


def test_score_calibration_rejects_protocol_for_different_validation_binding() -> None:
    model, encoded, data, head_report, scores = _trained_binary()
    changed_model = CoreGFM(node_classes=2)
    changed_encoded = _encoding(changed_model)
    changed_validation = SupervisedPartition(
        entityIds=("2", "3"),
        nodeIndices=(2, 3),
        targets=(1, 0),
    )
    changed_data = SupervisedTrainValidation.create(
        task_kind="node-binary",
        provenance=changed_encoded.provenance,
        train=_data("node-binary", changed_encoded.provenance).train,
        validation=changed_validation,
    )
    changed_report = fit_supervised_head(
        changed_model,
        changed_encoded,
        changed_data,
        config=HeadTrainingConfig.smoke(max_steps=1),
    )
    changed_scores = derive_validation_scores(
        changed_model,
        changed_encoded,
        changed_data,
        changed_report,
        semantics=BinaryScoreSemantics.for_task("node-binary"),
    )
    protocol = CalibrationProtocol.fixed(changed_scores)

    with pytest.raises(ValueError, match="validation score binding"):
        fit_score_calibration_report(scores, protocol=protocol)

    original_protocol = CalibrationProtocol.fixed(scores)
    next(model.node_head.parameters()).data.add_(1.0)
    with pytest.raises(ValueError, match="head state"):
        fit_score_calibration_report(scores, protocol=original_protocol)
    with pytest.raises(ValueError, match="head state"):
        derive_validation_scores(
            model,
            encoded,
            data,
            head_report,
            semantics=BinaryScoreSemantics.for_task("node-binary"),
        )


@pytest.mark.parametrize(
    "field",
    [
        "adapterSchemaHash",
        "adapterStateHash",
        "topologyHash",
        "encodedTensorHash",
        "trainPartitionHash",
    ],
)
def test_calibration_rederives_every_head_report_provenance_field(field: str) -> None:
    model, encoded, data, report, _scores = _trained_binary()
    payload = report.record.model_dump(mode="json", by_alias=True)
    payload[field] = "a" * 64
    payload["reportHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "reportHash"}
    )
    forged = HeadTrainingReport.model_validate(payload)
    object.__setattr__(report, "record", forged)

    with pytest.raises(ValueError, match="runtime seal"):
        derive_validation_scores(
            model,
            encoded,
            data,
            report,
            semantics=BinaryScoreSemantics.for_task("node-binary"),
        )


def test_validation_score_batch_rederives_task_and_head_fields() -> None:
    _model, _encoded, _data_record, _report, scores = _trained_binary()
    payload = scores.record.model_dump(mode="json", by_alias=True)
    payload["taskKind"] = "edge-binary"
    payload["headName"] = "binary_link_head"
    payload["batchHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "batchHash"}
    )
    forged_record = ValidationScoreBatch.model_validate(payload)
    forged = object.__new__(VerifiedValidationScores)
    for name, value in (
        ("logits", scores.logits),
        ("targets", scores.targets),
        ("record", forged_record),
        ("model", scores.model),
        ("encoded", scores.encoded),
        ("data", scores.data),
        ("head_report", scores.head_report),
        ("semantics", scores.semantics),
    ):
        object.__setattr__(forged, name, value)

    with pytest.raises(ValueError, match="validation score binding"):
        VerifiedValidationScores.verify(forged)


def test_verified_graph_is_factory_sealed_and_subclass_verify_cannot_bypass() -> None:
    model = CoreGFM(node_classes=2)
    encoded = _encoding(model)

    with pytest.raises(TypeError):
        VerifiedEncodedGraph(
            tensor=encoded.tensor,
            provenance=encoded.provenance,
            bundle=encoded.bundle,
            adapter=encoded.adapter,
        )

    class BypassVerifiedGraph(VerifiedEncodedGraph):
        def verify(self, model: CoreGFM) -> None:
            return None

    bypass = object.__new__(BypassVerifiedGraph)
    for name in ("tensor", "provenance", "bundle", "adapter"):
        object.__setattr__(bypass, name, getattr(encoded, name))
    with pytest.raises(TypeError, match="exact VerifiedEncodedGraph"):
        fit_supervised_head(
            model,
            bypass,
            _data("node-binary", encoded.provenance),
            config=HeadTrainingConfig.smoke(max_steps=1),
        )


def test_validation_scores_are_factory_sealed_and_duck_types_cannot_calibrate() -> None:
    _model, _encoded, _data_record, _report, scores = _trained_binary()
    protocol = CalibrationProtocol.fixed(scores)

    with pytest.raises(TypeError):
        VerifiedValidationScores(
            logits=scores.logits,
            targets=scores.targets,
            record=scores.record,
            model=scores.model,
            encoded=scores.encoded,
            data=scores.data,
            head_report=scores.head_report,
            semantics=scores.semantics,
        )

    duck = SimpleNamespace(
        verify=lambda: None,
        logits=scores.logits,
        targets=scores.targets,
        record=scores.record,
    )
    with pytest.raises(TypeError, match="exact VerifiedValidationScores"):
        fit_score_calibration_report(duck, protocol=protocol)

    class BypassValidationScores(VerifiedValidationScores):
        def verify(self) -> None:
            return None

    bypass = object.__new__(BypassValidationScores)
    for name in (
        "logits",
        "targets",
        "record",
        "model",
        "encoded",
        "data",
        "head_report",
        "semantics",
    ):
        object.__setattr__(bypass, name, getattr(scores, name))
    with pytest.raises(TypeError, match="exact VerifiedValidationScores"):
        CalibrationProtocol.fixed(bypass)


def test_encoded_graph_rederives_num_nodes_from_bundle_and_tensor() -> None:
    model = CoreGFM(node_classes=2)
    encoded = _encoding(model)
    payload = encoded.provenance.model_dump(mode="json", by_alias=True)
    payload["numNodes"] += 1
    payload["artifactHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifactHash"}
    )
    forged_provenance = EncodedGraphProvenance.model_validate(payload)
    forged = object.__new__(VerifiedEncodedGraph)
    object.__setattr__(forged, "tensor", encoded.tensor)
    object.__setattr__(forged, "provenance", forged_provenance)
    object.__setattr__(forged, "bundle", encoded.bundle)
    object.__setattr__(forged, "adapter", encoded.adapter)

    with pytest.raises(ValueError, match="numNodes"):
        VerifiedEncodedGraph.verify(forged, model)


def test_node_binary_training_rejects_non_two_class_head_before_mutation() -> None:
    model = CoreGFM(node_classes=3)
    encoded = _encoding(model)
    before = copy.deepcopy(model.node_head.state_dict())

    with pytest.raises(ValueError, match="exactly two class logits"):
        fit_supervised_head(
            model,
            encoded,
            _data("node-binary", encoded.provenance),
            config=HeadTrainingConfig.smoke(max_steps=1),
        )
    assert all(
        torch.equal(before[name], value) for name, value in model.node_head.state_dict().items()
    )


def test_official_train_and_validation_must_follow_stable_authoritative_roles() -> None:
    model = CoreGFM(node_classes=2)
    bundle = _official_node_bundle()
    encoded = encode_supervised_graph(model, bundle, BundleInputAdapter(bundle, mode="training"))
    poisoned = SupervisedTrainValidation.create(
        task_kind="node-binary",
        provenance=encoded.provenance,
        train=SupervisedPartition(
            entityIds=("0", "3"),
            nodeIndices=(0, 3),
            targets=(0, 1),
        ),
        validation=SupervisedPartition(
            entityIds=("1", "2"),
            nodeIndices=(1, 2),
            targets=(0, 1),
        ),
    )

    with pytest.raises(ValueError, match="authoritative train role"):
        fit_supervised_head(
            model,
            encoded,
            poisoned,
            config=HeadTrainingConfig.smoke(max_steps=1),
        )

    wrong_identity = SupervisedTrainValidation.create(
        task_kind="node-binary",
        provenance=encoded.provenance,
        train=SupervisedPartition(
            entityIds=("not-node-zero",),
            nodeIndices=(0,),
            targets=(0,),
        ),
        validation=SupervisedPartition(
            entityIds=("1", "2"),
            nodeIndices=(1, 2),
            targets=(0, 1),
        ),
    )
    with pytest.raises(ValueError, match="stable node IDs"):
        fit_supervised_head(
            model,
            encoded,
            wrong_identity,
            config=HeadTrainingConfig.smoke(max_steps=1),
        )


def test_official_validation_and_test_use_exact_roles_and_never_cross_roles() -> None:
    model = CoreGFM(node_classes=2)
    bundle = _official_node_bundle()
    encoded = encode_supervised_graph(model, bundle, BundleInputAdapter(bundle, mode="training"))
    incomplete_validation = SupervisedTrainValidation.create(
        task_kind="node-binary",
        provenance=encoded.provenance,
        train=SupervisedPartition(entityIds=("0",), nodeIndices=(0,), targets=(0,)),
        validation=SupervisedPartition(entityIds=("1",), nodeIndices=(1,), targets=(0,)),
    )
    with pytest.raises(ValueError, match="exact authoritative validation role"):
        fit_supervised_head(
            model,
            encoded,
            incomplete_validation,
            config=HeadTrainingConfig.smoke(max_steps=1),
        )

    correct = SupervisedTrainValidation.create(
        task_kind="node-binary",
        provenance=encoded.provenance,
        train=SupervisedPartition(entityIds=("0",), nodeIndices=(0,), targets=(0,)),
        validation=SupervisedPartition(entityIds=("1", "2"), nodeIndices=(1, 2), targets=(0, 1)),
    )
    incomplete_test = SupervisedTestSet.create(
        task_kind="node-binary",
        provenance=encoded.provenance,
        test=SupervisedPartition(entityIds=("3",), nodeIndices=(3,), targets=(0,)),
    )
    with pytest.raises(ValueError, match="exact authoritative test role"):
        validate_supervised_test_isolation(correct, incomplete_test, bundle=bundle)

    validation_as_test = SupervisedTestSet.create(
        task_kind="node-binary",
        provenance=encoded.provenance,
        test=SupervisedPartition(entityIds=("1", "2"), nodeIndices=(1, 2), targets=(0, 1)),
    )
    with pytest.raises(ValueError, match="authoritative test role"):
        validate_supervised_test_isolation(correct, validation_as_test, bundle=bundle)


def test_empty_split_evidence_is_smoke_only_and_permanently_non_promotable() -> None:
    model, encoded, _data_record, _report, _scores = _trained_binary()
    data = _data("node-binary", encoded.provenance)
    report = fit_supervised_head(
        model,
        encoded,
        data,
        config=HeadTrainingConfig.smoke(max_steps=1),
    )
    scores = derive_validation_scores(
        model,
        encoded,
        data,
        report,
        semantics=BinaryScoreSemantics.for_task("node-binary"),
    )
    protocol = CalibrationProtocol.fixed(scores)
    calibration_report = fit_score_calibration_report(scores, protocol=protocol)

    assert report.promotion_eligible is False
    assert scores.record.promotion_eligible is False
    assert protocol.promotion_eligible is False
    assert calibration_report.promotion_eligible is False

    with pytest.raises(ValueError, match="formal head training requires authoritative split"):
        fit_supervised_head(
            model,
            encoded,
            data,
            config=HeadTrainingConfig.formal(max_steps=1),
        )


def test_reparsed_coherently_forged_history_cannot_enter_validation_scoring() -> None:
    model, encoded, data, report, _scores = _trained_binary()
    with pytest.raises(TypeError):
        VerifiedHeadTrainingReport(
            record=report.record,
            _sealed_report_hash=report.report_hash,
            _factory_seal=object(),
        )

    class BypassVerifiedHeadReport(VerifiedHeadTrainingReport):
        pass

    bypass = object.__new__(BypassVerifiedHeadReport)
    for name in ("record", "_sealed_report_hash", "_factory_seal"):
        object.__setattr__(bypass, name, getattr(report, name))
    with pytest.raises(TypeError, match="VerifiedHeadTrainingReport"):
        derive_validation_scores(
            model,
            encoded,
            data,
            bypass,
            semantics=BinaryScoreSemantics.for_task("node-binary"),
        )

    payload = report.record.model_dump(mode="json", by_alias=True)
    for index, point in enumerate(payload["history"], start=1):
        point["validationMetric"] = float(index + 100)
        point["validationLoss"] = float(index + 200)
    payload["bestStep"] = payload["history"][-1]["step"]
    payload["bestMetric"] = payload["history"][-1]["validationMetric"]
    payload["reportHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "reportHash"}
    )
    forged = HeadTrainingReport.model_validate(payload)

    with pytest.raises(TypeError, match="VerifiedHeadTrainingReport"):
        derive_validation_scores(
            model,
            encoded,
            data,
            forged,
            semantics=BinaryScoreSemantics.for_task("node-binary"),
        )


def test_calibration_rebuilds_score_record_and_rejects_mutated_role() -> None:
    _model, _encoded, _data_record, _report, scores = _trained_binary()
    object.__setattr__(scores.record, "role", "test")
    payload = scores.record.model_dump(mode="json", by_alias=True)
    payload.pop("batchHash")
    object.__setattr__(scores.record, "batch_hash", canonical_sha256(payload))

    with pytest.raises(ValueError, match="validation score binding"):
        CalibrationProtocol.fixed(scores)
