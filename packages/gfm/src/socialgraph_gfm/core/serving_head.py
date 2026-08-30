"""Real core inference over Task 4 checkpoint state and bundle tensors."""

from __future__ import annotations

import json
from typing import cast

import torch

from .adapters import AdapterSchema, BundleInputAdapter
from .bundle import CoreGraphBundle
from .governance import (
    CalibratedConfidence,
    ConfidenceEvidence,
    GovernanceFinding,
    ModelScore,
    RegressionConfidenceInterval,
    RegisteredEdgeIdentity,
    build_collaboration_findings,
    build_community_resilience_findings,
    build_risk_and_trust_findings,
)
from .inference_contracts import GfmRunRequest
from .model import CoreGFM
from .serving_registry import (
    ConfidenceArtifact,
    RegressionConfidenceArtifact,
    ScoreCalibration,
    ServingModel,
    VerifiedCheckpoint,
)


def _edge_index(bundle: CoreGraphBundle) -> torch.Tensor:
    by_id = {node.id: node.index for node in bundle.nodes}
    edges = [(by_id[edge.source_id], by_id[edge.target_id]) for edge in bundle.edges]
    if not bundle.directed:
        edges.extend((target, source) for source, target in tuple(edges))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class CoreServingHead:
    def execute(
        self,
        request: GfmRunRequest,
        bundle: CoreGraphBundle,
        model_record: ServingModel,
        checkpoint: VerifiedCheckpoint,
        calibrations: dict[str, ConfidenceArtifact],
    ) -> tuple[GovernanceFinding, ...]:
        head = model_record.task_head(request.task_id)
        requested_entity = self._requested_entity_type(request)
        entity_binding = head.calibration(requested_entity)
        trainer = checkpoint.payload["trainer"]
        schema = AdapterSchema.model_validate_json(
            json.dumps(
                trainer["adapterSchemas"][entity_binding.adapter_domain],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        adapter = BundleInputAdapter(bundle, schema=schema, mode="inference")
        model = CoreGFM(node_classes=model_record.checkpoint.node_classes)
        adapter.load_state_dict(trainer["adapters"][entity_binding.adapter_domain], strict=True)
        model.load_state_dict(trainer["model"], strict=True)
        adapter.eval()
        model.eval()
        with torch.inference_mode():
            encoded = model.encode(adapter(), _edge_index(bundle))
            scored = self._scores(request, bundle, model_record, head, model, encoded)
        calibrated_items: list[tuple[ModelScore, ConfidenceEvidence]] = []
        for score in scored:
            calibration = calibrations.get(score.entity_type)
            if calibration is None:
                raise ValueError("task output is missing its entity-specific calibration")
            if isinstance(calibration, ScoreCalibration):
                confidence: ConfidenceEvidence = CalibratedConfidence.create(
                    score=score,
                    value=float(
                        torch.sigmoid(
                            torch.tensor((score.score + calibration.bias) / calibration.temperature)
                        ).item()
                    ),
                    calibration_version=calibration.calibration_version,
                    method=calibration.method,
                    calibration_artifact_hash=calibration.artifact_hash,
                    calibration_protocol_hash=calibration.protocol_hash,
                )
            elif isinstance(calibration, RegressionConfidenceArtifact):
                confidence = RegressionConfidenceInterval.create(
                    score=score,
                    lower_bound=score.score - calibration.residual_quantile,
                    upper_bound=score.score + calibration.residual_quantile,
                    coverage=calibration.coverage,
                    validation_count=calibration.validation_count,
                    confidence_version=calibration.confidence_version,
                    method=calibration.method,
                    confidence_artifact_hash=calibration.artifact_hash,
                    confidence_protocol_hash=calibration.protocol_hash,
                )
            else:  # pragma: no cover - the confidence artifact union is closed
                raise TypeError("unsupported serving confidence artifact")
            calibrated_items.append((score, confidence))
        calibrated = tuple(calibrated_items)
        if request.task_id == "core.community_resilience_review":
            if any(
                not isinstance(confidence, RegressionConfidenceInterval)
                for _score, confidence in calibrated
            ):
                raise ValueError("community resilience requires regression interval evidence")
            return build_community_resilience_findings(
                bundle,
                scored_candidates=cast(
                    tuple[tuple[ModelScore, RegressionConfidenceInterval], ...], calibrated
                ),
            )
        if request.task_id == "core.risk_and_trust_review":
            if any(
                not isinstance(confidence, CalibratedConfidence)
                for _score, confidence in calibrated
            ):
                raise ValueError("risk and trust review requires binary calibration evidence")
            return build_risk_and_trust_findings(
                bundle,
                scored_candidates=cast(
                    tuple[tuple[ModelScore, CalibratedConfidence], ...], calibrated
                ),
            )
        if any(
            not isinstance(confidence, CalibratedConfidence) for _score, confidence in calibrated
        ):
            raise ValueError("collaboration completion requires binary calibration evidence")
        return build_collaboration_findings(
            bundle,
            scored_candidates=cast(tuple[tuple[ModelScore, CalibratedConfidence], ...], calibrated),
            top_k=request.parameters.candidate_limit,  # type: ignore[union-attr]
        )

    @staticmethod
    def _requested_entity_type(request: GfmRunRequest) -> str:
        if request.task_id == "core.community_resilience_review":
            return "community"
        if request.task_id == "core.collaboration_completion":
            return "node-pair"
        return "node" if request.target_scope.node_ids else "edge"  # type: ignore[union-attr]

    @staticmethod
    def _scores(
        request: GfmRunRequest,
        bundle: CoreGraphBundle,
        model_record: ServingModel,
        head,
        model: CoreGFM,
        encoded: torch.Tensor,
    ) -> tuple[ModelScore, ...]:
        by_id = {node.id: node.index for node in bundle.nodes}
        scores: list[ModelScore] = []

        def create(entity_type, entity_ids, value, edge_identity=None):
            return ModelScore.create(
                task_id=request.task_id,
                entity_type=entity_type,
                entity_ids=entity_ids,
                score=float(value),
                graph_version_hash=bundle.graph_version_hash,
                model_version=model_record.model_version_id,
                model_version_hash=model_record.model_version_hash,
                edge_identity=edge_identity,
            )

        if request.task_id == "core.community_resilience_review":
            logits = model.resilience_head(encoded)
            for identifier in request.target_scope.community_ids:  # type: ignore[union-attr]
                if identifier not in by_id:
                    raise ValueError("community target is not present in bundle")
                scores.append(create("community", (identifier,), logits[by_id[identifier]].item()))
        elif request.task_id == "core.risk_and_trust_review":
            node_logits = model.node_head(encoded)
            output_index = head.node_output_index
            if output_index != 1 or node_logits.shape[1] != 2:
                raise ValueError("risk task requires binary logits with positive class 1")
            for identifier in request.target_scope.node_ids:  # type: ignore[union-attr]
                if identifier not in by_id:
                    raise ValueError("risk node target is not present in bundle")
                scores.append(
                    create(
                        "node",
                        (identifier,),
                        (
                            node_logits[by_id[identifier], 1] - node_logits[by_id[identifier], 0]
                        ).item(),
                    )
                )
            edge_by_hash = {
                RegisteredEdgeIdentity.create(edge).edge_hash: edge for edge in bundle.edges
            }
            for edge_hash in request.target_scope.edge_ids:  # type: ignore[union-attr]
                edge = edge_by_hash.get(edge_hash)
                if edge is None:
                    raise ValueError("risk edge target is not present in bundle")
                identity = RegisteredEdgeIdentity.create(edge)
                pairs = torch.tensor(
                    [[by_id[edge.source_id], by_id[edge.target_id]]], dtype=torch.long
                )
                value = model.signed_edge_head(encoded, pairs)[0].item()
                scores.append(
                    create(
                        "edge",
                        (edge.source_id, edge.target_id),
                        value,
                        edge_identity=identity,
                    )
                )
        else:
            target_pairs = request.target_scope.pairs  # type: ignore[union-attr]
            for source, target in target_pairs:
                if source not in by_id or target not in by_id:
                    raise ValueError("collaboration target is not present in bundle")
            pair_tensor = torch.tensor(
                [[by_id[source], by_id[target]] for source, target in target_pairs],
                dtype=torch.long,
            )
            logits = model.binary_link_head(encoded, pair_tensor)
            scores.extend(
                create("node-pair", pair, logits[index].item())
                for index, pair in enumerate(target_pairs)
            )
        return tuple(scores)


__all__ = ["CoreServingHead"]
