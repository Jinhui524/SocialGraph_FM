"""Bounded, no-LLM execution registry for SocialGraph-FM Governance product skills."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from socialgraph_gfm.canonical import canonical_json, canonical_sha256, file_sha256

from .contracts import (
    INPUT_SCHEMA_VERSION,
    MAX_EVIDENCE_EDGES,
    MAX_EVIDENCE_NODES,
    MAX_NODES,
    MAX_PREVIEW_EDGES,
    MAX_PREVIEW_NODES,
    MAX_RELATION_ROWS,
    MODALITIES,
)
from .knowledge import KnowledgeIndex
from .materialize import OnlineInferenceData
from .reviewed_cases import CaseKindEntry, CaseVectors, ReviewedCaseIndex
from .skill_contracts import (
    PUBLIC_SKILLS,
    CommandProvenance,
    CommandRequest,
    CommandResponse,
    DraftReportParams,
    EmptyParams,
    EvidenceParams,
    IndexCaseParams,
    InspectGraphParams,
    KnowledgeSearchParams,
    PageParams,
    RelationParams,
    RunGovernanceAnalysisParams,
    SimilarCaseParams,
    SkillRuntimeProtocol,
    _components,
    _kind_key,
    _utc_now,
)


class GovernanceSkillExecutor:
    """Closed, read-only public registry plus two explicit internal persistence commands."""

    def __init__(self, runtime: SkillRuntimeProtocol) -> None:
        self._runtime = runtime
        self._cases = ReviewedCaseIndex(runtime.root / "reviewed-cases")

    @property
    def skill_names(self) -> tuple[str, ...]:
        return PUBLIC_SKILLS

    def _binding(self, request: CommandRequest) -> OnlineInferenceData:
        data = self._runtime._artifact(request.graph.artifact_id)
        model = self._runtime._model
        if (
            data.artifact.dataset_content_hash != request.graph.dataset_content_hash
            or data.artifact.graph_version_hash != request.graph.graph_version_hash
            or model is None
            or model.model_version_id != request.model.model_version_id
            or model.model_state_hash != request.model.model_state_hash
        ):
            raise ValueError("skill graph or model binding does not match the GFM runtime")
        return data

    def _run(self, request: CommandRequest, run_id: str) -> dict[str, Any]:
        result = self._runtime.result(run_id)
        if (
            result.get("artifactId") != request.graph.artifact_id
            or result.get("datasetContentHash") != request.graph.dataset_content_hash
            or result.get("graphVersionHash") != request.graph.graph_version_hash
            or result.get("modelVersionId") != request.model.model_version_id
            or result.get("modelStateHash") != request.model.model_state_hash
        ):
            raise ValueError("skill request is not bound to the persisted run")
        return result

    def _resolved_nodes(
        self,
        request: CommandRequest,
        run_id: str,
        kind_entries: Sequence[CaseKindEntry],
    ) -> tuple[str, ...]:
        self._run(request, run_id)
        data = self._binding(request)
        node_inventory = set(data.node_ids)
        resolved: set[str] = set()
        relation_lookup: dict[str, tuple[str, str]] | None = None
        group_lookup: dict[str, tuple[str, ...]] | None = None
        for entry in kind_entries:
            if entry.kind == "node":
                if not set(entry.target_ids) <= node_inventory:
                    raise ValueError("node kind contains a target outside the bound graph")
                resolved.update(entry.target_ids)
            elif entry.kind == "relation":
                if relation_lookup is None:
                    arrays = self._runtime._analytics_arrays(run_id)
                    relation_lookup = {
                        f"relation-{int(source)}-{int(target)}": (
                            data.node_ids[int(source)],
                            data.node_ids[int(target)],
                        )
                        for source, target in zip(arrays["source"], arrays["target"], strict=True)
                    }
                    document = self._runtime._analytics_document(run_id)
                    for item in document["links"]:
                        link_id = str(item["linkId"])
                        endpoints = (str(item["source"]), str(item["target"]))
                        if (
                            link_id in relation_lookup
                            or endpoints[0] not in node_inventory
                            or endpoints[1] not in node_inventory
                            or endpoints[0] == endpoints[1]
                        ):
                            raise ValueError("potential-link derivation inventory is invalid")
                        relation_lookup[link_id] = endpoints
                try:
                    for target_id in entry.target_ids:
                        resolved.update(relation_lookup[target_id])
                except KeyError as error:
                    raise ValueError(
                        "relation target does not name a persisted derivation"
                    ) from error
            else:
                if group_lookup is None:
                    document = self._runtime._analytics_document(run_id)
                    group_lookup = {
                        str(item["groupId"]): tuple(str(value) for value in item["memberNodeIds"])
                        for item in document["groups"]
                    }
                try:
                    for target_id in entry.target_ids:
                        resolved.update(group_lookup[target_id])
                except KeyError as error:
                    raise ValueError("group target does not name a persisted derivation") from error
        if not resolved:
            raise ValueError("kindEntries resolve to no graph nodes")
        return tuple(sorted(resolved))

    def _vectors(
        self,
        request: CommandRequest,
        run_id: str,
        kind_entries: Sequence[CaseKindEntry],
    ) -> CaseVectors:
        data = self._binding(request)
        target_ids = self._resolved_nodes(request, run_id, kind_entries)
        lookup = {node_id: index for index, node_id in enumerate(data.node_ids)}
        selected = np.asarray([lookup[value] for value in target_ids], dtype=np.int64)
        outputs = self._runtime._outputs(run_id)
        embedding = np.asarray(outputs["embeddings"][selected], dtype=np.float32).mean(axis=0)
        embedding_norm = float(np.linalg.norm(embedding))
        if embedding_norm:
            embedding /= embedding_norm
        degrees = np.diff(np.asarray(data.arrays["fused_indptr"]))[selected].astype(np.float64)
        scaled = np.log1p(degrees) / math.log1p(MAX_NODES)
        selected_set = {int(value) for value in selected}
        edge_index = np.asarray(data.edge_index)
        internal = sum(
            int(source) < int(target)
            and int(source) in selected_set
            and int(target) in selected_set
            for source, target in zip(edge_index[0], edge_index[1], strict=True)
        )
        possible = len(selected_set) * (len(selected_set) - 1) / 2
        structure = np.asarray(
            [
                float(scaled.mean()),
                float(scaled.std()),
                float(scaled.max()),
                float(np.asarray(data.structure_missing)[selected].mean()),
                float(np.asarray(data.degree_bucket)[selected].mean() / 127.0),
                0.0 if not possible else float(internal / possible),
            ],
            dtype=np.float32,
        )
        modality = np.log1p(
            np.asarray(outputs["modality_counts"][selected], dtype=np.float64).sum(axis=0)
        ).astype(np.float32)
        modality_norm = float(np.linalg.norm(modality))
        if modality_norm:
            modality /= modality_norm
        return CaseVectors(embedding=embedding, structure=structure, modality=modality)

    def execute(self, raw: Mapping[str, Any]) -> CommandResponse:
        request = CommandRequest.model_validate(raw)
        self._binding(request)
        input_hash = canonical_sha256(request.model_dump(mode="json", by_alias=True))
        result, status, warnings = self._dispatch(request)
        response = CommandResponse(
            commandId=request.command_id,
            command=request.command,
            status=status,
            graph=request.graph,
            model=request.model,
            result=result,
            provenance=CommandProvenance(generatedAt=_utc_now(), inputHash=input_hash),
            warnings=warnings,
        )
        if response.graph != request.graph or response.model != request.model:
            raise ValueError("skill response binding changed during execution")
        return response

    def _dispatch(
        self, request: CommandRequest
    ) -> tuple[dict[str, Any], Literal["completed", "confirmation_required"], tuple[str, ...]]:
        command = request.command
        if command == "inspect_graph":
            return self._inspect(request), "completed", ()
        if command == "run_governance_analysis":
            return (
                self._run_plan(request),
                "confirmation_required",
                ("No inference was started; explicit confirmation is required.",),
            )
        if command == "get_evidence_subgraph":
            evidence_params = EvidenceParams.model_validate(request.params)
            self._run(request, evidence_params.run_id)
            return (
                self._runtime.evidence(evidence_params.run_id, evidence_params.node_id),
                "completed",
                (),
            )
        if command == "discover_coordination_groups":
            page_params = PageParams.model_validate(request.params)
            self._run(request, page_params.run_id)
            return (
                self._runtime.derivations(
                    page_params.run_id,
                    "groups",
                    offset=page_params.offset,
                    limit=page_params.limit,
                ),
                "completed",
                ("Communities are analyst leads, not proof of coordination.",),
            )
        if command == "rank_coordination_relations":
            return (
                self._relations(request),
                "completed",
                ("Relation priority is explanation-only and non-causal.",),
            )
        if command == "retrieve_similar_cases":
            return (
                self._similar(request),
                "completed",
                ("Similarity is non-causal retrieval over reviewed cases only.",),
            )
        if command == "get_model_dataset_cards":
            return self._cards(request), "completed", ()
        if command == "draft_review_report":
            return (
                self._draft(request),
                "completed",
                ("The report is a deterministic draft and requires human review.",),
            )
        if command == "index_case":
            return self._index_case(request), "completed", ()
        if command == "search_knowledge":
            return (
                self._search_knowledge(request),
                "completed",
                ("Retrieved text is evidence context and does not change model outputs.",),
            )
        raise ValueError("unsupported Governance skill command")

    def _inspect(self, request: CommandRequest) -> dict[str, Any]:
        params = InspectGraphParams.model_validate(request.params)
        data = self._binding(request)
        lookup = {node_id: index for index, node_id in enumerate(data.node_ids)}
        if params.scope_node_ids:
            if tuple(sorted(set(params.scope_node_ids))) != params.scope_node_ids:
                raise ValueError("scopeNodeIds must be unique and canonically sorted")
            try:
                scope = {lookup[value] for value in params.scope_node_ids}
            except KeyError as error:
                raise ValueError("scopeNodeIds contains an unknown node") from error
        else:
            scope = set(range(len(data.node_ids)))
        edges = tuple(
            (int(source), int(target))
            for source, target in zip(data.edge_index[0], data.edge_index[1], strict=True)
            if int(source) < int(target) and int(source) in scope and int(target) in scope
        )
        relation_counts: dict[str, int] = {}
        for modality in MODALITIES:
            indptr = np.asarray(data.arrays[f"relation_{modality.lower()}_indptr"])
            indices = np.asarray(data.arrays[f"relation_{modality.lower()}_indices"])
            relation_counts[modality] = sum(
                source in scope and int(target) in scope and source < int(target)
                for source in scope
                for target in indices[int(indptr[source]) : int(indptr[source + 1])]
            )
        ordered_scope = sorted(scope)
        local_index = {value: index for index, value in enumerate(ordered_scope)}
        local_edges = tuple((local_index[source], local_index[target]) for source, target in edges)
        payload = {
            "nodeCount": len(scope),
            "fusedEdgeCount": len(edges),
            "componentCount": _components(len(scope), local_edges),
            "isolateCount": sum(bool(data.structure_missing[index]) for index in scope),
            "modalities": [
                modality for modality in MODALITIES if relation_counts[modality] > 0
            ],
            "relationCounts": relation_counts,
            "scopeNodeIds": [data.node_ids[index] for index in ordered_scope],
        }
        if params.run_id is not None:
            run_result = self._run(request, params.run_id)
            distribution = run_result.get("distribution")
            findings = run_result.get("findings")
            if not isinstance(distribution, dict) or not isinstance(findings, list):
                raise ValueError("persisted run does not contain bounded result summaries")
            required_distribution = {"low", "review", "high", "predictedPositive", "total"}
            if set(distribution) != required_distribution or any(
                not isinstance(distribution[key], int) or distribution[key] < 0
                for key in required_distribution
            ):
                raise ValueError("persisted run distribution is invalid")
            candidates: list[dict[str, Any]] = []
            allowed = {
                "nodeId",
                "label",
                "score",
                "rank",
                "riskBand",
                "structureMissing",
                "communityId",
            }
            for finding in findings[: params.candidate_limit]:
                if not isinstance(finding, dict):
                    raise TypeError("persisted run finding summary is invalid")
                candidate = {key: finding[key] for key in allowed if key in finding}
                if not {"nodeId", "score", "riskBand"} <= set(candidate):
                    raise ValueError("persisted run finding summary is incomplete")
                candidates.append(candidate)
            payload.update(
                {
                    "runId": params.run_id,
                    "distribution": distribution,
                    "topCandidates": candidates,
                    "candidateLimit": params.candidate_limit,
                }
            )
        payload["inspectionHash"] = canonical_sha256(payload)
        return payload

    def _run_plan(self, request: CommandRequest) -> dict[str, Any]:
        params = RunGovernanceAnalysisParams.model_validate(request.params)
        data = self._binding(request)
        if params.top_k > len(data.node_ids):
            raise ValueError("topK exceeds the bound graph node count")
        execution = {
            "schemaVersion": "socialgraph-fm.gfm-governance/2.0",
            "protocol": "global",
            "artifactId": request.graph.artifact_id,
            "datasetContentHash": request.graph.dataset_content_hash,
            "graphVersionHash": request.graph.graph_version_hash,
            "modelVersionId": request.model.model_version_id,
            "modelStateHash": request.model.model_state_hash,
            "topK": params.top_k,
        }
        return {
            "confirmationPlan": {
                "action": "run_governance_analysis",
                "requiresConfirmation": True,
                "requestDigest": canonical_sha256(execution),
                "steps": ["validate", "materialize", "enqueue", "review"],
                "estimatedScope": {
                    "nodeCount": len(data.node_ids),
                    "relationRowCount": int(data.artifact.document["relationRowCount"]),
                    "topK": params.top_k,
                },
                "executionRequest": execution,
            }
        }

    def _relations(self, request: CommandRequest) -> dict[str, Any]:
        params = RelationParams.model_validate(request.params)
        result = self._run(request, params.run_id)
        if params.relation_kind == "potential":
            derived = self._runtime.derivations(
                params.run_id, "links", offset=params.offset, limit=params.limit
            )
            items = list(derived["items"])
            if any(
                item.get("kind") != "potential_link"
                or item.get("factual") is not False
                or item.get("modalities") not in ([], ())
                for item in items
            ):
                raise ValueError("potential relation result contains a factual or untyped relation")
            total = int(derived["total"])
        else:
            data = self._runtime._artifact(str(result["artifactId"]))
            arrays = self._runtime._analytics_arrays(params.run_id)
            accepted: list[int] = []
            requested = set(params.modalities)
            for index in arrays["order"]:
                value = int(index)
                mask = int(arrays["modality_mask"][value])
                present = {name for column, name in enumerate(MODALITIES) if mask & (1 << column)}
                if not requested or requested & present:
                    accepted.append(value)
            selected = accepted[params.offset : params.offset + params.limit]
            items = [self._runtime._relation_derivation(data, arrays, index) for index in selected]
            if any(
                item.get("kind") != "factual_relation" or item.get("factual") is not True
                for item in items
            ):
                raise ValueError("factual relation result contains a potential or untyped relation")
            total = len(accepted)
        payload = {
            "runId": params.run_id,
            "items": items,
            "total": total,
            "offset": params.offset,
            "limit": params.limit,
            "relationKind": params.relation_kind,
            "modalities": list(params.modalities),
        }
        payload["pageHash"] = canonical_sha256(payload)
        return payload

    def _similar(self, request: CommandRequest) -> dict[str, Any]:
        params = SimilarCaseParams.model_validate(request.params)
        query: dict[str, Any]
        if params.case_id is not None:
            record = self._cases.record(params.case_id)
            if (
                record.artifact_id != request.graph.artifact_id
                or record.dataset_content_hash != request.graph.dataset_content_hash
                or record.graph_version_hash != request.graph.graph_version_hash
                or record.model_state_hash != request.model.model_state_hash
            ):
                raise ValueError("caseId is not bound to the requested graph/model")
            vectors = self._cases.vectors(record)
            kind_key = record.kind_key
            exclude = record.case_id
            query = {"caseId": record.case_id}
        else:
            assert params.run_id is not None
            vectors = self._vectors(request, params.run_id, params.kind_entries)
            kind_key = _kind_key(params.kind_entries)
            exclude = None
            query = {
                "runId": params.run_id,
                "kindKey": kind_key,
                "kindEntries": [
                    item.model_dump(mode="json", by_alias=True) for item in params.kind_entries
                ],
            }
        matches = self._cases.query(
            vectors,
            model_state_hash=request.model.model_state_hash,
            kind_key=kind_key,
            limit=params.limit,
            exclude_case_id=exclude,
        )
        payload = {
            "query": query,
            "items": [
                {
                    "caseId": item.record.case_id,
                    "score": item.score,
                    "components": {
                        "embedding": item.embedding_score,
                        "structure": item.structure_score,
                        "modality": item.modality_score,
                    },
                    "graphVersionHash": item.record.graph_version_hash,
                    "modelStateHash": item.record.model_state_hash,
                    "kindKey": item.record.kind_key,
                    "kindEntries": [
                        value.model_dump(mode="json", by_alias=True)
                        for value in item.record.kind_entries
                    ],
                    "concludedAt": item.record.concluded_at,
                    "recordHash": item.record.record_hash,
                }
                for item in matches
            ],
            "weights": {"embedding": 0.7, "structure": 0.2, "modality": 0.1},
            "indexHash": self._cases.index_hash,
        }
        payload["retrievalHash"] = canonical_sha256(payload)
        return payload

    @staticmethod
    def _verified_json(path: Path, *, hash_field: str | None = None) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("card source must be a JSON object")
        if hash_field is not None:
            logical = {key: item for key, item in value.items() if key != hash_field}
            if value.get(hash_field) != canonical_sha256(logical):
                raise ValueError("card source hash is invalid")
        return value

    def _cards(self, request: CommandRequest) -> dict[str, Any]:
        EmptyParams.model_validate(request.params)
        data = self._binding(request)
        model = self._runtime._model
        registry_path = self._runtime.global_model_root / "registry" / "socialgraph-global.json"
        registry = self._verified_json(registry_path, hash_field="registryHash")
        corpus_path = self._runtime.global_model_root / "corpus" / "manifest.json"
        corpus = self._verified_json(corpus_path)
        model_card = {
            "modelVersionId": model.model_version_id,
            "modelVersionHash": model.model_version_hash,
            "modelStateHash": model.model_state_hash,
            "threshold": model.threshold,
            "calibration": {"temperature": model.temperature, "bias": model.bias},
            "referenceMetrics": dict(model.reference_metrics),
            "runtimeRecipeHash": model.runtime_recipe_hash,
            "registryHash": registry["registryHash"],
            "limitations": [
                "Only the published Global checkpoint is online-serving ready.",
                "External graph scores remain out-of-domain unverified rankings.",
            ],
        }
        dataset_card = {
            "datasetId": data.artifact.document["datasetId"],
            "displayName": data.artifact.document["displayName"],
            "artifactId": data.artifact.artifact_id,
            "artifactHash": data.artifact.document["artifactHash"],
            "datasetContentHash": data.artifact.dataset_content_hash,
            "graphVersionHash": data.artifact.graph_version_hash,
            "nodeCount": len(data.node_ids),
            "relationRowCount": data.artifact.document["relationRowCount"],
            "fusedUndirectedEdgeCount": data.artifact.document["fusedUndirectedEdgeCount"],
            "modalities": list(data.artifact.document["modalities"]),
            "license": data.artifact.document["license"],
            "sourceUri": data.artifact.document["sourceUri"],
            "corpusContentHash": corpus.get("contentHash"),
            "corpusManifestSha256": file_sha256(corpus_path),
        }
        input_card = {
            "schemaVersion": INPUT_SCHEMA_VERSION,
            "featureDimension": 768,
            "modalities": list(MODALITIES),
            "requiredMembers": ["manifest.json", "nodes.csv", "relations.csv", "features.npz"],
            "labelsSplitsScoresAccepted": False,
            "limits": {
                "maxNodes": MAX_NODES,
                "maxRelationRows": MAX_RELATION_ROWS,
                "maxEvidenceNodes": MAX_EVIDENCE_NODES,
                "maxEvidenceEdges": MAX_EVIDENCE_EDGES,
                "maxPreviewNodes": MAX_PREVIEW_NODES,
                "maxPreviewEdges": MAX_PREVIEW_EDGES,
            },
        }
        input_card["contractHash"] = canonical_sha256(input_card)
        payload: dict[str, Any] = {
            "modelCard": model_card,
            "datasetCard": dataset_card,
            "inputContractCard": input_card,
        }
        payload["cardHash"] = canonical_sha256(payload)
        return payload

    def _draft(self, request: CommandRequest) -> dict[str, Any]:
        params = DraftReportParams.model_validate(request.params)
        result = self._run(request, params.run_id)
        if result["resultHash"] != params.result_hash:
            raise ValueError("resultHash does not match the persisted run result")
        if params.review_hash is not None:
            record = self._cases.record(params.case_id)
            if any(
                (
                    record.case_hash != params.case_hash,
                    record.review_hash != params.review_hash,
                    record.run_id != params.run_id,
                    record.result_hash != params.result_hash,
                    record.kind_entries != params.kind_entries,
                )
            ):
                raise ValueError("reviewHash is not bound to the indexed case")
        data = self._binding(request)
        outputs = self._runtime._outputs(params.run_id)
        resolved_nodes = self._resolved_nodes(request, params.run_id, params.kind_entries)
        lookup = {node_id: index for index, node_id in enumerate(data.node_ids)}
        reported_nodes = sorted(
            resolved_nodes,
            key=lambda node_id: (-float(outputs["scores"][lookup[node_id]]), node_id),
        )[:20]
        indices = [lookup[value] for value in reported_nodes]
        community_sizes = np.bincount(outputs["community_ids"].astype(np.int64))
        findings = [
            self._runtime._finding(data, outputs, index, community_sizes=community_sizes)
            for index in indices
        ]
        evidence_hashes = [
            self._runtime.evidence(params.run_id, node_id)["evidenceHash"]
            for node_id in reported_nodes
        ]
        report_data = {
            "caseId": params.case_id,
            "caseHash": params.case_hash,
            "reviewHash": params.review_hash,
            "runId": params.run_id,
            "resultHash": params.result_hash,
            "graphVersionHash": request.graph.graph_version_hash,
            "modelStateHash": request.model.model_state_hash,
            "kindKey": _kind_key(params.kind_entries),
            "kindEntries": [
                item.model_dump(mode="json", by_alias=True) for item in params.kind_entries
            ],
            "resolvedNodeCount": len(resolved_nodes),
            "reportedNodeCount": len(reported_nodes),
            "findings": findings,
            "evidenceHashes": evidence_hashes,
            "limitations": [
                "Deterministic no-LLM draft; a human reviewer owns the conclusion.",
                "Scores and factual relations do not prove intent or coordination.",
            ],
            "generatedWithoutLlm": True,
        }
        if params.format == "json":
            content = canonical_json(report_data)
        else:
            lines = [
                "# Governance Review Draft",
                "",
                f"- Case: `{json.dumps(params.case_id, ensure_ascii=False)}`",
                f"- Run: `{params.run_id}`",
                f"- Result hash: `{params.result_hash}`",
                f"- Graph hash: `{request.graph.graph_version_hash}`",
                f"- Model state hash: `{request.model.model_state_hash}`",
            ]
            for finding, evidence_hash in zip(findings, evidence_hashes, strict=True):
                lines.extend(
                    [
                        "",
                        f"## {json.dumps(str(finding['nodeId']), ensure_ascii=False)}",
                        "",
                        f"- Score: {float(finding['score']):.9g}",
                        f"- Risk band: `{finding['riskBand']}`",
                        f"- Evidence hash: `{evidence_hash}`",
                    ]
                )
            lines.extend(
                [
                    "",
                    "Human review is required. Scores and relations do not prove intent or coordination.",
                ]
            )
            content = "\n".join(lines) + "\n"
        payload = {
            "format": params.format,
            "content": content,
            "caseId": params.case_id,
            "citedHashes": [params.case_hash, params.result_hash, *evidence_hashes],
            "generatedWithoutLlm": True,
        }
        payload["draftHash"] = canonical_sha256(payload)
        return payload

    def _index_case(self, request: CommandRequest) -> dict[str, Any]:
        params = IndexCaseParams.model_validate(request.params)
        result = self._run(request, params.run_id)
        if result["resultHash"] != params.result_hash:
            raise ValueError("resultHash does not match the persisted run result")
        vectors = self._vectors(request, params.run_id, params.kind_entries)
        kind_key = _kind_key(params.kind_entries)
        metadata = {
            "caseId": params.case_id,
            "caseHash": params.case_hash,
            "runId": params.run_id,
            "resultHash": params.result_hash,
            "kindKey": kind_key,
            "kindEntries": [
                item.model_dump(mode="json", by_alias=True) for item in params.kind_entries
            ],
            "concludedAt": params.concluded_at,
            "reviewHash": params.review_hash,
            "reviewStatus": params.review_status,
            "artifactId": request.graph.artifact_id,
            "datasetContentHash": request.graph.dataset_content_hash,
            "graphVersionHash": request.graph.graph_version_hash,
            "modelVersionId": request.model.model_version_id,
            "modelStateHash": request.model.model_state_hash,
        }
        record, idempotent, index_hash = self._cases.index(
            metadata,
            vectors,
            indexed_at=_utc_now(),
            source_request_hash=canonical_sha256(
                {
                    "command": request.command,
                    "graph": request.graph,
                    "model": request.model,
                    "params": params,
                }
            ),
        )
        return {
            "caseId": record.case_id,
            "recordHash": record.record_hash,
            "indexHash": index_hash,
            "indexedAt": record.indexed_at,
            "idempotent": idempotent,
        }

    def _search_knowledge(self, request: CommandRequest) -> dict[str, Any]:
        params = KnowledgeSearchParams.model_validate(request.params)
        index = KnowledgeIndex(self._runtime.root / "knowledge")
        items = index.search(params.query, limit=params.limit)
        payload = {
            "items": [
                {
                    "sourceLabel": item.source_label,
                    "sourceUri": item.source_uri,
                    "contentHash": item.content_hash,
                    "chunkHash": item.chunk_hash,
                    "text": item.text,
                    "rank": item.rank,
                }
                for item in items
            ],
            "indexHash": index.verify(),
        }
        payload["searchHash"] = canonical_sha256(payload)
        return payload


__all__ = [
    "GovernanceSkillExecutor",
]
