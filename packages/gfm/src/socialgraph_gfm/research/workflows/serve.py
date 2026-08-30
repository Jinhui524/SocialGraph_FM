"""Serving export construction, runtime loading, and fresh-process smoke validation."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from socialgraph_gfm.canonical import canonical_json, canonical_sha256, file_sha256

from ...core.bundle import CoreGraphBundle
from ..contracts import (
    ACCOUNT_RISK_TASK,
    COLLABORATION_TASK,
    CONTENT_POLICY_TASK,
    RELEASE_ID,
    RESEARCH_SEED,
    SIGNED_RELATION_TASK,
)
from ..routing import SHARED_NULL_ROUTE, route_contract, task_route_name
from .common import (
    EVALUATION_SCHEMA,
    EXPORT_SCHEMA,
    FEATURE_CONTRACT_SCHEMA,
    FRESH_HTTP_RUN_TIMEOUT_SECONDS,
    FRESH_HTTP_STARTUP_TIMEOUT_SECONDS,
    SMOKE_SCHEMA,
    _atomic_json,
    _domain_task_id,
    _read_hashed_document,
    _route_contract_hash,
    _safe_root,
    load_research_config,
)
from .materialize import _require_publishable_corpus
from .runtime import _checkpoint_runtime, _load_trained_runtime
from .train import (
    _bundle_edge_index,
    _role_ids,
    _tensor_state_hash,
    _torch_atomic_save,
)


def _scenario_rows(documents, evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    definitions = (
        (
            "twitch-content-policy",
            "twitch-language",
            "twitch-EN",
            CONTENT_POLICY_TASK,
            "Content policy review",
            "macro-f1",
        ),
        (
            "tolokers-account-risk",
            "tolokers",
            "tolokers",
            ACCOUNT_RISK_TASK,
            "Historical account status review",
            "auprc",
        ),
        (
            "wiki-rfa-signed-relation",
            "wiki-rfa",
            "wiki-rfa",
            SIGNED_RELATION_TASK,
            "Governance relation stance review",
            "negative-auprc",
        ),
        (
            "email-eu-collaboration",
            "email-eu-core",
            "email-eu-core",
            COLLABORATION_TASK,
            "Collaboration relation candidates",
            "filtered-mrr",
        ),
    )
    rows: list[dict[str, Any]] = []
    for scenario_id, dataset_id, domain, task_id, title, primary_metric in definitions:
        bundle, labels, _entry = documents[domain]
        target: dict[str, Any]
        if task_id in {CONTENT_POLICY_TASK, ACCOUNT_RISK_TASK}:
            label_ids = {item["entityId"] for item in labels["targets"]}
            selected = [item for item in _role_ids(bundle, "test") if item in label_ids]
            if not selected:
                selected = [item for item in _role_ids(bundle, "validation") if item in label_ids]
            if not selected:
                selected = [item for item in _role_ids(bundle, "train") if item in label_ids]
            target = {"kind": "nodes", "nodeIds": selected[:10]}
        elif task_id == SIGNED_RELATION_TASK:
            label_by_id = {item["entityId"]: item for item in labels["targets"]}
            selected = [item for item in _role_ids(bundle, "test") if item in label_by_id]
            if not selected:
                selected = [item for item in _role_ids(bundle, "validation") if item in label_by_id]
            if not selected:
                selected = [item for item in _role_ids(bundle, "train") if item in label_by_id]
            target = {
                "kind": "directed-node-pairs",
                "pairs": [
                    [label_by_id[item]["sourceId"], label_by_id[item]["targetId"]]
                    for item in selected[:10]
                ],
            }
        else:
            target = {
                "kind": "collaboration-candidates",
                "anchorNodeId": bundle.nodes[0].id,
                "topK": 10,
            }
        if target.get("nodeIds") == [] or target.get("pairs") == []:
            raise ValueError(f"scenario {scenario_id} has no usable default target")
        rows.append(
            {
                "scenarioId": scenario_id,
                "datasetId": dataset_id,
                "domain": domain,
                "route": task_route_name(task_id, domain),
                "title": title,
                "taskId": task_id,
                "graphVersionId": f"research:{dataset_id}",
                "graphVersionHash": bundle.graph_version_hash,
                "defaultTargetScope": target,
                "primaryMetric": {
                    "name": primary_metric,
                    "value": float(evaluation["metrics"][task_id][primary_metric]),
                },
                "scratchDelta": (
                    None
                    if evaluation.get("comparisonMatrix") is None
                    else float(
                        evaluation["comparisonMatrix"]["aggregates"][task_id][
                            "sharedVsScratchDelta"
                        ]
                    )
                ),
            }
        )
    return rows


def _preview_payload(
    *,
    scenario: Mapping[str, Any],
    bundle: CoreGraphBundle,
    model_version_id: str,
    model_version_hash: str,
) -> dict[str, Any]:
    from ...core.adapters import derive_training_selection

    visible_indices = derive_training_selection(bundle).visible_edge_indices
    visible_edges = tuple(bundle.edges[index] for index in visible_indices)
    adjacency: dict[str, set[str]] = {node.id: set() for node in bundle.nodes}
    for edge in visible_edges:
        adjacency[edge.source_id].add(edge.target_id)
        adjacency[edge.target_id].add(edge.source_id)
    target = scenario["defaultTargetScope"]
    seeds = (
        list(target.get("nodeIds", ()))
        or [endpoint for pair in target.get("pairs", ()) for endpoint in pair]
        or [target.get("anchorNodeId")]
    )
    queue = [item for item in seeds if item in adjacency]
    if not queue:
        queue = [bundle.nodes[0].id]
    selected: list[str] = []
    seen: set[str] = set()
    while queue and len(selected) < 800:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        selected.append(node)
        queue.extend(item for item in sorted(adjacency[node]) if item not in seen)
    if len(selected) < min(800, len(bundle.nodes)):
        selected.extend(
            node.id
            for node in bundle.nodes
            if node.id not in seen and len(selected) < min(800, len(bundle.nodes))
        )
    selected_set = set(selected)
    edges = [
        {
            "id": f"edge:{edge.source_id}:{edge.target_id}",
            "source": edge.source_id,
            "target": edge.target_id,
            "directed": bundle.directed,
        }
        for edge in visible_edges
        if edge.source_id in selected_set and edge.target_id in selected_set
    ][:2500]
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.research/1.0",
        "scenarioId": scenario["scenarioId"],
        "graphVersionId": scenario["graphVersionId"],
        "graphVersionHash": bundle.graph_version_hash,
        "modelVersionId": model_version_id,
        "modelVersionHash": model_version_hash,
        "nodes": [{"id": node_id, "label": node_id} for node_id in selected],
        "edges": edges,
        "partialPreview": (
            len(selected) < len(bundle.nodes)
            or len(edges) < len(visible_edges)
            or len(visible_edges) < len(bundle.edges)
        ),
        "nodeCount": len(bundle.nodes),
        "edgeCount": len(bundle.edges),
    }
    payload["previewHash"] = canonical_sha256(payload)
    return payload


def _feature_contracts(documents, adapters) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for domain, (bundle, _labels, entry) in sorted(documents.items()):
        payload: dict[str, Any] = {
            "schemaVersion": FEATURE_CONTRACT_SCHEMA,
            "graphVersionHash": bundle.graph_version_hash,
            "nodeFeatures": [
                {"name": feature.name, "kind": feature.kind} for feature in bundle.node_features
            ],
            "structuralFeatureNames": list(
                bundle.structural_features.names if bundle.structural_features else ()
            ),
            "adapterSchemaHash": adapters[domain].schema.adapter_schema_hash,
            "excludedInputFields": list(entry["excludedInputFields"]),
            "taskId": _domain_task_id(domain),
            "taskRoute": task_route_name(_domain_task_id(domain), domain),
            "similarityRoute": SHARED_NULL_ROUTE,
        }
        payload["featureContractHash"] = canonical_sha256(payload)
        contracts[domain] = payload
    return contracts


def _serving_checkpoint_projection(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "modelVersionId",
        "modelVersionHash",
        "researchConfigSha256",
        "corpusHash",
        "corpusKind",
        "testOnly",
        "trainingHash",
        "comparisonEvaluationHash",
        "evaluationHash",
        "modelStateHash",
        "taskIds",
        "graphVersionHashes",
        "splitHashes",
        "visibleTopologyHashes",
        "adapterSchemaHashes",
        "featureContractHashes",
        "routeContract",
        "routeContractHash",
        "parserContractHash",
        "metricHashes",
    )
    return {field: checkpoint[field] for field in fields}


def export_research_model(
    research_root: str | Path, *, allow_test_fixture: bool = False
) -> Path:
    import numpy as np
    import torch

    root = _safe_root(research_root)
    evaluation = _read_hashed_document(
        root / "reports/evaluation.json",
        schema=EVALUATION_SCHEMA,
        hash_field="evaluationHash",
    )
    comparison = evaluation.get("comparisonMatrix")
    if (
        not isinstance(comparison, dict)
        or comparison.get("runCount") != 54
        or comparison.get("testRole") != "evaluate-only"
    ):
        raise ValueError(
            "SocialGraph-FM Research export requires the complete evaluated 54-run comparison matrix"
        )
    training, checkpoint, corpus, documents, model, adapters = _load_trained_runtime(
        root, device="cpu"
    )
    _require_publishable_corpus(
        corpus, allow_test_fixture=allow_test_fixture, stage="export"
    )
    if evaluation["trainingHash"] != training["trainingHash"]:
        raise ValueError("research evaluation does not bind the selected checkpoint")
    target = root / "exports/research"
    if target.exists():
        raise FileExistsError(f"research export already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".research.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        research_config_sha = load_research_config()["configSha256"]
        task_ids = [
            CONTENT_POLICY_TASK,
            ACCOUNT_RISK_TASK,
            SIGNED_RELATION_TASK,
            COLLABORATION_TASK,
        ]
        graph_version_hashes = {
            domain: bundle.graph_version_hash
            for domain, (bundle, _labels, _entry) in sorted(documents.items())
        }
        split_hashes = {entry["graphId"]: entry["splitHash"] for entry in corpus["graphs"]}
        visible_topology_hashes = {
            entry["graphId"]: entry["visibleTopologyHash"] for entry in corpus["graphs"]
        }
        feature_contracts = _feature_contracts(documents, adapters)
        feature_contract_hashes = {
            domain: contract["featureContractHash"]
            for domain, contract in feature_contracts.items()
        }
        adapter_schema_hashes = {
            domain: adapters[domain].schema.adapter_schema_hash for domain in sorted(adapters)
        }
        metric_hashes = {
            "taskMetricsHash": canonical_sha256(evaluation["metrics"]),
            "comparisonEvaluationHash": comparison["comparisonEvaluationHash"],
            "evaluationHash": evaluation["evaluationHash"],
        }
        parser_contracts = {
            entry["graphId"]: {
                "parserId": entry["parserId"],
                "parserVersion": entry["parserVersion"],
                "parserCodeSha256": entry["parserCodeSha256"],
            }
            for entry in corpus["graphs"]
        }
        parser_contract_hash = canonical_sha256(parser_contracts)
        model_version_hash = canonical_sha256(
            {
                "schemaVersion": "socialgraph-fm.research-model-identity/1.0",
                "researchConfigSha256": research_config_sha,
                "modelStateHash": training["modelStateHash"],
                "corpusHash": corpus["corpusHash"],
                "corpusKind": corpus["corpusKind"],
                "testOnly": corpus["testOnly"],
                "trainingHash": training["trainingHash"],
                "evaluationHash": evaluation["evaluationHash"],
                "comparisonEvaluationHash": comparison["comparisonEvaluationHash"],
                "graphVersionHashes": graph_version_hashes,
                "splitHashes": split_hashes,
                "adapterSchemaHashes": adapter_schema_hashes,
                "featureContractHashes": feature_contract_hashes,
                "routeContractHash": _route_contract_hash(),
                "parserContractHash": parser_contract_hash,
                "seed": RESEARCH_SEED,
            }
        )
        model_version_id = f"socialgraph-fm-research/{model_version_hash[:16]}"
        serving_checkpoint: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.research-serving-checkpoint/1.0",
            "releaseId": RELEASE_ID,
            "seed": RESEARCH_SEED,
            "preliminary": True,
            "transductive": True,
            "modelVersionId": model_version_id,
            "modelVersionHash": model_version_hash,
            "researchConfigSha256": research_config_sha,
            "corpusHash": corpus["corpusHash"],
            "corpusKind": corpus["corpusKind"],
            "testOnly": corpus["testOnly"],
            "trainingHash": training["trainingHash"],
            "comparisonEvaluationHash": comparison["comparisonEvaluationHash"],
            "evaluationHash": evaluation["evaluationHash"],
            "metricHashes": metric_hashes,
            "taskIds": task_ids,
            "domains": tuple(sorted(documents)),
            "graphVersionHashes": graph_version_hashes,
            "splitHashes": split_hashes,
            "visibleTopologyHashes": visible_topology_hashes,
            "graphArtifactBindings": {
                entry["graphId"]: {
                    "bundlePath": entry["bundlePath"],
                    "bundleSha256": entry["bundleSha256"],
                    "labelsPath": entry["labelsPath"],
                    "labelsSha256": entry["labelsSha256"],
                    "graphVersionHash": entry["graphVersionHash"],
                    "splitHash": entry["splitHash"],
                }
                for entry in corpus["graphs"]
            },
            "parserContracts": parser_contracts,
            "parserContractHash": parser_contract_hash,
            "featureContracts": feature_contracts,
            "featureContractHashes": feature_contract_hashes,
            "routeContract": route_contract(),
            "routeContractHash": _route_contract_hash(),
            "modelState": model.state_dict(),
            "modelStateHash": training["modelStateHash"],
            "headStateHashes": {
                "contentPolicy": _tensor_state_hash(model.content_policy_head.state_dict()),
                "accountRisk": _tensor_state_hash(model.account_risk_head.state_dict()),
                "signedRelation": _tensor_state_hash(model.signed_edge_head.state_dict()),
                "collaboration": _tensor_state_hash(model.collaboration_head.state_dict()),
            },
            "adapterSchemas": {
                domain: adapters[domain].schema.model_dump(mode="json", by_alias=True)
                for domain in sorted(adapters)
            },
            "adapterSchemaHashes": adapter_schema_hashes,
            "adapterStates": {domain: adapters[domain].state_dict() for domain in sorted(adapters)},
            "adapterStateHashes": {
                domain: _tensor_state_hash(adapters[domain].state_dict())
                for domain in sorted(adapters)
            },
            "calibrators": checkpoint["calibrators"],
            "tolokersFoldHeadStates": checkpoint["tolokersFoldHeadStates"],
            "tolokersFoldCalibrators": checkpoint["tolokersFoldCalibrators"],
            "trainingConfig": checkpoint["trainingConfig"],
            "sourceTrainingCheckpointSha256": training["checkpointSha256"],
        }
        projection = _serving_checkpoint_projection(serving_checkpoint)
        serving_checkpoint["exportProjectionHash"] = canonical_sha256(projection)
        exported_checkpoint = staging / "checkpoint.pt"
        _torch_atomic_save(exported_checkpoint, serving_checkpoint)
        checkpoint_sha = file_sha256(exported_checkpoint)
        embedding_entries: list[dict[str, Any]] = []
        with torch.inference_mode():
            for domain, (bundle, _labels, _entry) in sorted(documents.items()):
                values = model.encode_domain(
                    adapters[domain](),
                    _bundle_edge_index(bundle, visible_only=True),
                    None,
                )
                values = torch.nn.functional.normalize(values, dim=-1).cpu().numpy().astype("<f4")
                path = staging / "embeddings" / f"{domain}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path,
                    embeddings=values,
                    node_ids=np.asarray([node.id for node in bundle.nodes], dtype="U500"),
                )
                embedding_entries.append(
                    {
                        "domain": domain,
                        "route": SHARED_NULL_ROUTE,
                        "graphVersionHash": bundle.graph_version_hash,
                        "path": f"embeddings/{domain}.npz",
                        "sha256": file_sha256(path),
                        "nodeCount": len(bundle.nodes),
                        "width": int(values.shape[1]),
                    }
                )
        scenarios = _scenario_rows(documents, evaluation)
        for scenario in scenarios:
            preview = _preview_payload(
                scenario=scenario,
                bundle=documents[scenario["domain"]][0],
                model_version_id=model_version_id,
                model_version_hash=model_version_hash,
            )
            preview_path = staging / "previews" / f"{scenario['scenarioId']}.json"
            _atomic_json(preview_path, preview)
            scenario["previewPath"] = f"previews/{scenario['scenarioId']}.json"
            scenario["previewSha256"] = file_sha256(preview_path)
        model_card: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.research-model-card/1.0",
            "releaseId": RELEASE_ID,
            "releaseLabel": "SocialGraph-FM Research",
            "modelVersionId": model_version_id,
            "modelVersionHash": model_version_hash,
            "researchConfigSha256": research_config_sha,
            "seed": RESEARCH_SEED,
            "corpusKind": corpus["corpusKind"],
            "testOnly": corpus["testOnly"],
            "preliminary": True,
            "transductive": True,
            "singleRunInitialResult": True,
            "formalReadinessUnaffected": True,
            "routeContract": route_contract(),
            "routeContractHash": _route_contract_hash(),
            "taskSemantics": {
                CONTENT_POLICY_TASK: (
                    "Ranks explicit-language label review; not illegal or harmful content detection."
                ),
                ACCOUNT_RISK_TASK: (
                    "Ranks historical ban-label review; not an automatic ban or early warning."
                ),
                SIGNED_RELATION_TASK: (
                    "Predicts support/opposition relation sign; not toxicity or objective trust."
                ),
                COLLABORATION_TASK: (
                    "Completes unobserved static relations; not future-link forecasting."
                ),
            },
            "tolokersProtocol": {
                "evaluation": (
                    "arithmetic mean and standard deviation over 10 official splits; "
                    "test memberships may overlap"
                ),
                "deployedHead": "split-0",
            },
            "emailRankingProtocol": {
                "directions": "both endpoints",
                "candidateFilter": "all known true neighbors except the current target",
                "heuristicTopology": "training-visible edges only",
                "tiePolicy": "average rank",
            },
            "calibrationStatus": evaluation["calibrationStatus"],
            "taskCalibrationStatus": evaluation["taskCalibrationStatus"],
            "claimGate": evaluation["advantageClaim"],
            "uploadedGraphUse": {
                "classificationHeadsAvailable": False,
                "structuralSimilarity": "ranking-only",
                "collaborationCompletion": "ranking-only when graph contract is compatible",
            },
            "offlineDepartmentEvaluation": {
                "usedAsModelInput": False,
                "groupMetrics": evaluation["metrics"][COLLABORATION_TASK][
                    "offlineDepartmentGroupMetrics"
                ],
            },
            "dataUse": (
                {
                    "families": [
                        "Twitch Language",
                        "Tolokers",
                        "Wiki-RfA",
                        "Email-EU-core",
                    ],
                    "rawDataRedistributed": False,
                    "licenseNotice": (
                        "Raw files remain local; verify SNAP, Wikimedia, and upstream dataset terms."
                    ),
                }
                if corpus["testOnly"] is False
                else {
                    "families": ["Synthetic SocialGraph-FM Research test fixtures"],
                    "rawDataRedistributed": False,
                    "licenseNotice": "Test-only synthetic artifacts; not a dataset or model claim.",
                }
            ),
            "artifactBindings": {
                "checkpointSha256": checkpoint_sha,
                "corpusHash": corpus["corpusHash"],
                "trainingHash": training["trainingHash"],
                "comparisonEvaluationHash": comparison["comparisonEvaluationHash"],
                "evaluationHash": evaluation["evaluationHash"],
                "exportProjectionHash": serving_checkpoint["exportProjectionHash"],
            },
            "parserContracts": parser_contracts,
            "materializerVersion": corpus["materializerVersion"],
        }
        model_card["modelCardHash"] = canonical_sha256(model_card)
        model_card_path = staging / "model-card.json"
        _atomic_json(model_card_path, model_card)
        export: dict[str, Any] = {
            "schemaVersion": EXPORT_SCHEMA,
            "releaseId": RELEASE_ID,
            "releaseLabel": "SocialGraph-FM Research",
            "seed": RESEARCH_SEED,
            "preliminary": True,
            "formalReadinessUnaffected": True,
            "transductive": True,
            "modelVersionId": model_version_id,
            "modelVersionHash": model_version_hash,
            "taskIds": task_ids,
            "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
            "maxNodes": 50_000,
            "maxEdges": 1_500_000,
            "checkpointPath": "checkpoint.pt",
            "checkpointSha256": checkpoint_sha,
            "modelCardPath": "model-card.json",
            "modelCardSha256": file_sha256(model_card_path),
            "modelCardHash": model_card["modelCardHash"],
            "exportProjectionHash": serving_checkpoint["exportProjectionHash"],
            "researchConfigSha256": research_config_sha,
            "corpusKind": corpus["corpusKind"],
            "testOnly": corpus["testOnly"],
            "modelStateHash": training["modelStateHash"],
            "corpusHash": corpus["corpusHash"],
            "trainingHash": training["trainingHash"],
            "comparisonEvaluationHash": comparison["comparisonEvaluationHash"],
            "evaluationHash": evaluation["evaluationHash"],
            "metricHashes": metric_hashes,
            "graphVersionHashes": graph_version_hashes,
            "splitHashes": split_hashes,
            "visibleTopologyHashes": visible_topology_hashes,
            "adapterSchemaHashes": adapter_schema_hashes,
            "featureContractHashes": feature_contract_hashes,
            "routeContract": route_contract(),
            "routeContractHash": _route_contract_hash(),
            "parserContractHash": parser_contract_hash,
            "claimStatus": evaluation["advantageClaim"]["claimStatus"],
            "calibrationStatus": evaluation["calibrationStatus"],
            "taskCalibrationStatus": evaluation["taskCalibrationStatus"],
            "calibrators": checkpoint["calibrators"],
            "scenarios": scenarios,
            "embeddings": embedding_entries,
        }
        export["artifactHash"] = canonical_sha256(export)
        _atomic_json(staging / "export-manifest.json", export)
        os.replace(staging, target)
        return target / "export-manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_export_manifest(research_root: str | Path) -> dict[str, Any]:
    root = _safe_root(research_root)
    export = _read_hashed_document(
        root / "exports/research/export-manifest.json",
        schema=EXPORT_SCHEMA,
        hash_field="artifactHash",
    )
    if export.get("researchConfigSha256") != load_research_config()["configSha256"]:
        raise ValueError("research export configuration identity mismatch")
    if (
        export.get("routeContract") != route_contract()
        or export.get("routeContractHash") != _route_contract_hash()
    ):
        raise ValueError("research export route contract mismatch")
    if (export.get("corpusKind"), export.get("testOnly")) not in {
        ("real", False),
        ("test-fixture", True),
    }:
        raise ValueError("research export corpus kind/test-only identity is invalid")
    checkpoint = root / "exports/research" / export["checkpointPath"]
    if file_sha256(checkpoint) != export["checkpointSha256"]:
        raise ValueError("research export checkpoint hash mismatch")
    model_card_path = (root / "exports/research" / export["modelCardPath"]).resolve()
    if not model_card_path.is_relative_to((root / "exports/research").resolve()):
        raise ValueError("research model card path escapes export root")
    if file_sha256(model_card_path) != export["modelCardSha256"]:
        raise ValueError("research model card artifact hash mismatch")
    model_card = _read_hashed_document(
        model_card_path,
        schema="socialgraph-fm.research-model-card/1.0",
        hash_field="modelCardHash",
    )
    if (
        model_card["modelCardHash"] != export["modelCardHash"]
        or model_card["modelVersionHash"] != export["modelVersionHash"]
        or model_card.get("corpusKind") != export["corpusKind"]
        or model_card.get("testOnly") is not export["testOnly"]
        or model_card.get("routeContract") != route_contract()
        or model_card.get("routeContractHash") != _route_contract_hash()
        or model_card["artifactBindings"]["checkpointSha256"] != export["checkpointSha256"]
        or model_card["artifactBindings"]["evaluationHash"] != export["evaluationHash"]
        or model_card["artifactBindings"]["exportProjectionHash"] != export["exportProjectionHash"]
    ):
        raise ValueError("research model card binding mismatch")
    for entry in export["embeddings"]:
        if entry.get("route") != SHARED_NULL_ROUTE:
            raise ValueError("research similarity embedding route mismatch")
        path = (root / "exports/research" / entry["path"]).resolve()
        if not path.is_relative_to((root / "exports/research").resolve()):
            raise ValueError("research embedding path escapes export root")
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("research embedding artifact hash mismatch")
    for scenario in export["scenarios"]:
        if scenario.get("route") != task_route_name(
            scenario.get("taskId"), scenario.get("domain")
        ):
            raise ValueError("research scenario task route mismatch")
        path = (root / "exports/research" / scenario["previewPath"]).resolve()
        if not path.is_relative_to((root / "exports/research").resolve()):
            raise ValueError("research preview path escapes export root")
        if file_sha256(path) != scenario["previewSha256"]:
            raise ValueError("research preview artifact hash mismatch")
        preview = _read_hashed_document(
            path,
            schema="socialgraph-fm.research/1.0",
            hash_field="previewHash",
        )
        if (
            preview["scenarioId"] != scenario["scenarioId"]
            or preview["graphVersionHash"] != scenario["graphVersionHash"]
            or preview["modelVersionId"] != export["modelVersionId"]
            or preview["modelVersionHash"] != export["modelVersionHash"]
        ):
            raise ValueError("research preview binding mismatch")
    return export


def _load_exported_runtime(root: Path, *, device: str):
    """Reconstruct serving state from the exported checkpoint, never the training run."""

    import torch

    export = load_export_manifest(root)
    checkpoint_path = root / "exports/research" / export["checkpointPath"]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("schemaVersion") != "socialgraph-fm.research-serving-checkpoint/1.0":
        raise ValueError("unsupported SocialGraph-FM Research serving checkpoint schema")
    checkpoint_projection = _serving_checkpoint_projection(checkpoint)
    export_projection = _serving_checkpoint_projection(export)
    projection_hash = canonical_sha256(checkpoint_projection)
    if (
        checkpoint_projection != export_projection
        or checkpoint.get("exportProjectionHash") != projection_hash
        or export.get("exportProjectionHash") != projection_hash
    ):
        raise ValueError("research serving checkpoint export binding mismatch")
    corpus, documents, model, adapters = _checkpoint_runtime(root, checkpoint, device=device)
    graph_version_hashes = {
        domain: bundle.graph_version_hash
        for domain, (bundle, _labels, _entry) in sorted(documents.items())
    }
    split_hashes = {entry["graphId"]: entry["splitHash"] for entry in corpus["graphs"]}
    visible_hashes = {entry["graphId"]: entry["visibleTopologyHash"] for entry in corpus["graphs"]}
    if (
        checkpoint["graphVersionHashes"] != graph_version_hashes
        or checkpoint["splitHashes"] != split_hashes
        or checkpoint["visibleTopologyHashes"] != visible_hashes
    ):
        raise ValueError("research serving checkpoint graph/split identity mismatch")
    bindings = checkpoint.get("graphArtifactBindings")
    expected_bindings = {
        entry["graphId"]: {
            "bundlePath": entry["bundlePath"],
            "bundleSha256": entry["bundleSha256"],
            "labelsPath": entry["labelsPath"],
            "labelsSha256": entry["labelsSha256"],
            "graphVersionHash": entry["graphVersionHash"],
            "splitHash": entry["splitHash"],
        }
        for entry in corpus["graphs"]
    }
    if bindings != expected_bindings:
        raise ValueError("research serving checkpoint graph artifact binding mismatch")
    parser_contracts = {
        entry["graphId"]: {
            "parserId": entry["parserId"],
            "parserVersion": entry["parserVersion"],
            "parserCodeSha256": entry["parserCodeSha256"],
        }
        for entry in corpus["graphs"]
    }
    if checkpoint.get("parserContracts") != parser_contracts or checkpoint.get(
        "parserContractHash"
    ) != canonical_sha256(parser_contracts):
        raise ValueError("research serving checkpoint parser contract mismatch")
    feature_contracts = checkpoint.get("featureContracts")
    if not isinstance(feature_contracts, Mapping):
        raise ValueError(  # noqa: TRY004 - persisted artifact validation is a value error
            "research serving checkpoint lacks feature contracts"
        )
    for domain, contract in feature_contracts.items():
        observed = contract.get("featureContractHash")
        expected = canonical_sha256(
            {key: value for key, value in contract.items() if key != "featureContractHash"}
        )
        if (
            contract.get("schemaVersion") != FEATURE_CONTRACT_SCHEMA
            or observed != expected
            or checkpoint["featureContractHashes"].get(domain) != observed
            or contract.get("taskId") != _domain_task_id(domain)
            or contract.get("taskRoute")
            != task_route_name(_domain_task_id(domain), domain)
            or contract.get("similarityRoute") != SHARED_NULL_ROUTE
            or adapters[domain].schema.adapter_schema_hash
            != checkpoint["adapterSchemaHashes"].get(domain)
        ):
            raise ValueError("research serving checkpoint feature contract mismatch")
    head_hashes = {
        "contentPolicy": _tensor_state_hash(model.content_policy_head.state_dict()),
        "accountRisk": _tensor_state_hash(model.account_risk_head.state_dict()),
        "signedRelation": _tensor_state_hash(model.signed_edge_head.state_dict()),
        "collaboration": _tensor_state_hash(model.collaboration_head.state_dict()),
    }
    if checkpoint.get("headStateHashes") != head_hashes:
        raise ValueError("research serving checkpoint task head identity mismatch")
    return export, checkpoint, corpus, documents, model, adapters


def _formal_contract_source(name: str) -> Path:
    packaged = resources.files("socialgraph_gfm").joinpath(f"resources/{name}")
    if packaged.is_file():
        return Path(str(packaged))
    source = Path(__file__).resolve().parents[4] / "contracts" / name
    if not source.is_file():
        raise FileNotFoundError(f"the packaged core serving contract is unavailable: {name}")
    return source


def _clone_read_only_tree(source: Path, destination: Path) -> None:
    if not source.is_dir() or destination.exists():
        raise ValueError("fresh HTTP smoke tree source or destination is invalid")
    destination.mkdir(parents=True)
    for item in sorted(source.rglob("*")):
        if item.is_symlink():
            raise ValueError("fresh HTTP smoke refuses linked export artifacts")
        target = destination / item.relative_to(source)
        if item.is_dir():
            target.mkdir()
            continue
        if not item.is_file():
            raise ValueError("fresh HTTP smoke encountered a non-regular export artifact")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(item, target)
        except OSError:
            shutil.copy2(item, target)


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _http_json(
    *,
    port: int,
    token: str,
    method: str,
    path: str,
    payload: Mapping[str, Any] | None = None,
    expected_status: int = 200,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    body = None
    if payload is not None:
        body = canonical_json(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        encoded = response.read(16 * 1024 * 1024 + 1)
    finally:
        connection.close()
    if len(encoded) > 16 * 1024 * 1024:
        raise RuntimeError(f"fresh SocialGraph-FM Research HTTP response is oversized: {path}")
    try:
        document = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"fresh SocialGraph-FM Research HTTP response is not JSON: {path}") from error
    if response.status != expected_status:
        code = document.get("error", {}).get("code") if isinstance(document, dict) else None
        raise RuntimeError(
            f"fresh SocialGraph-FM Research HTTP request failed: {method} {path} "
            f"returned {response.status} ({code or 'unknown'})"
        )
    if not isinstance(document, dict):
        raise RuntimeError(  # noqa: TRY004 - invalid remote response, not caller type misuse
            f"fresh SocialGraph-FM Research HTTP response is not an object: {path}"
        )
    return document


def _verify_wire_hash(document: Mapping[str, Any], field: str, *, context: str) -> None:
    observed = document.get(field)
    expected = canonical_sha256({key: value for key, value in document.items() if key != field})
    if observed != expected:
        raise RuntimeError(f"fresh SocialGraph-FM Research {context} hash mismatch")


def _wait_for_research_http(
    process: subprocess.Popen[str], *, token_file: Path, port: int
) -> tuple[str, dict[str, Any]]:
    deadline = time.monotonic() + FRESH_HTTP_STARTUP_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"fresh inference_cli exited before serving SocialGraph-FM Research (exit {process.returncode})"
            )
        try:
            token = token_file.read_text(encoding="utf-8").strip()
            if token:
                capabilities = _http_json(
                    port=port,
                    token=token,
                    method="GET",
                    path="/internal/research/capabilities",
                )
                return token, capabilities
        except (FileNotFoundError, OSError, RuntimeError) as error:
            last_error = error
        time.sleep(0.05)
    raise TimeoutError(
        "fresh inference_cli did not expose SocialGraph-FM Research within "
        f"{FRESH_HTTP_STARTUP_TIMEOUT_SECONDS} seconds"
    ) from last_error


def _exercise_research_http(
    *, port: int, token: str, capabilities: dict[str, Any], export: Mapping[str, Any]
) -> dict[str, Any]:
    wire_schema = "socialgraph-fm.research/1.0"
    expected_scenarios = (
        "twitch-content-policy",
        "tolokers-account-risk",
        "wiki-rfa-signed-relation",
        "email-eu-collaboration",
    )
    _verify_wire_hash(capabilities, "capabilityHash", context="capabilities")
    model = capabilities.get("model")
    if (
        capabilities.get("schemaVersion") != wire_schema
        or capabilities.get("researchServingReady") is not True
        or not isinstance(model, dict)
        or model.get("modelVersionId") != export["modelVersionId"]
        or model.get("modelVersionHash") != export["modelVersionHash"]
        or model.get("artifactHash") != export["artifactHash"]
        or model.get("taskIds") != export["taskIds"]
    ):
        raise RuntimeError("fresh SocialGraph-FM Research capabilities do not bind the selected export")
    scenarios = _http_json(
        port=port,
        token=token,
        method="GET",
        path="/internal/research/scenarios",
    )
    _verify_wire_hash(scenarios, "scenariosHash", context="scenarios")
    rows = scenarios.get("scenarios")
    if not isinstance(rows, list) or tuple(item.get("scenarioId") for item in rows) != expected_scenarios:
        raise RuntimeError("fresh SocialGraph-FM Research scenario inventory is not canonical")
    expected_by_id = {item["scenarioId"]: item for item in export["scenarios"]}
    scenario_results: list[dict[str, Any]] = []
    for scenario in rows:
        expected = expected_by_id.get(scenario["scenarioId"])
        if (
            expected is None
            or scenario.get("enabled") is not True
            or scenario.get("graphVersionHash") != expected["graphVersionHash"]
            or scenario.get("modelVersionId") != export["modelVersionId"]
            or scenario.get("taskId") != expected["taskId"]
        ):
            raise RuntimeError(f"fresh SocialGraph-FM Research scenario binding failed: {scenario['scenarioId']}")
        request = {
            "schemaVersion": wire_schema,
            "graphVersionId": scenario["graphVersionId"],
            "taskId": scenario["taskId"],
            "modelVersionId": export["modelVersionId"],
            "targetScope": scenario["defaultTargetScope"],
            "scenarioId": scenario["scenarioId"],
            "parameters": {"candidateLimit": 20},
        }
        envelope = {
            "schemaVersion": wire_schema,
            "request": request,
            "graphReference": {
                "kind": "registered-scenario",
                "graphVersionId": scenario["graphVersionId"],
                "graphVersionHash": scenario["graphVersionHash"],
                "nodeCount": 0,
                "edgeCount": 0,
            },
            "expectedModel": model,
        }
        status = _http_json(
            port=port,
            token=token,
            method="POST",
            path="/internal/research/runs",
            payload=envelope,
            expected_status=202,
        )
        _verify_wire_hash(status, "stateHash", context="run status")
        expected_request_hash = canonical_sha256(request)
        if status.get("requestHash") != expected_request_hash:
            raise RuntimeError("fresh SocialGraph-FM Research run request hash mismatch")
        deadline = time.monotonic() + FRESH_HTTP_RUN_TIMEOUT_SECONDS
        while status.get("status") not in {"succeeded", "failed"}:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"fresh SocialGraph-FM Research run timed out: {status['runId']}")
            time.sleep(0.02)
            status = _http_json(
                port=port,
                token=token,
                method="GET",
                path=f"/internal/research/runs/{status['runId']}",
            )
            _verify_wire_hash(status, "stateHash", context="run status")
        if status["status"] != "succeeded":
            raise RuntimeError(
                f"fresh SocialGraph-FM Research run failed: {scenario['scenarioId']} ({status.get('errorCode')})"
            )
        result = _http_json(
            port=port,
            token=token,
            method="GET",
            path=f"/internal/research/runs/{status['runId']}/result",
        )
        _verify_wire_hash(result, "resultHash", context="run result")
        if (
            result.get("requestHash") != expected_request_hash
            or result.get("graphVersionHash") != scenario["graphVersionHash"]
            or result.get("modelVersionHash") != export["modelVersionHash"]
            or not result.get("findings")
        ):
            raise RuntimeError(f"fresh SocialGraph-FM Research result binding failed: {scenario['scenarioId']}")
        repeated_status = _http_json(
            port=port,
            token=token,
            method="POST",
            path="/internal/research/runs",
            payload=envelope,
            expected_status=202,
        )
        repeated_result = _http_json(
            port=port,
            token=token,
            method="GET",
            path=f"/internal/research/runs/{status['runId']}/result",
        )
        if repeated_status != status or repeated_result != result:
            raise RuntimeError(
                f"fresh SocialGraph-FM Research repeated request is not deterministic: {scenario['scenarioId']}"
            )
        scenario_results.append(
            {
                "scenarioId": scenario["scenarioId"],
                "taskId": scenario["taskId"],
                "runId": status["runId"],
                "requestHash": expected_request_hash,
                "stateHash": status["stateHash"],
                "resultHash": result["resultHash"],
                "findingCount": len(result["findings"]),
                "repeatDeterministic": True,
            }
        )
    similarity_scenario = rows[0]
    source_node = similarity_scenario["defaultTargetScope"]["nodeIds"][0]
    similarity_request = {
        "schemaVersion": wire_schema,
        "graphVersionId": similarity_scenario["graphVersionId"],
        "nodeId": source_node,
        "topK": 3,
        "modelVersionId": export["modelVersionId"],
    }
    similarity_envelope = {
        "schemaVersion": wire_schema,
        "request": similarity_request,
        "graphReference": {
            "kind": "registered-scenario",
            "graphVersionId": similarity_scenario["graphVersionId"],
            "graphVersionHash": similarity_scenario["graphVersionHash"],
            "nodeCount": 0,
            "edgeCount": 0,
        },
        "expectedModel": model,
    }
    similarity = _http_json(
        port=port,
        token=token,
        method="POST",
        path="/internal/research/similar-nodes",
        payload=similarity_envelope,
    )
    repeated_similarity = _http_json(
        port=port,
        token=token,
        method="POST",
        path="/internal/research/similar-nodes",
        payload=similarity_envelope,
    )
    _verify_wire_hash(similarity, "resultHash", context="similar-nodes result")
    if (
        similarity != repeated_similarity
        or similarity.get("modelVersionHash") != export["modelVersionHash"]
        or not similarity.get("matches")
    ):
        raise RuntimeError("fresh SocialGraph-FM Research similar-nodes result is not deterministic")
    return {
        "capabilityHash": capabilities["capabilityHash"],
        "scenariosHash": scenarios["scenariosHash"],
        "endpointInventory": [
            "GET /internal/research/capabilities",
            "GET /internal/research/scenarios",
            "POST /internal/research/runs",
            "GET /internal/research/runs/{runId}",
            "GET /internal/research/runs/{runId}/result",
            "POST /internal/research/similar-nodes",
        ],
        "scenarioResults": scenario_results,
        "similarNodes": {
            "graphVersionId": similarity_request["graphVersionId"],
            "nodeId": source_node,
            "resultHash": similarity["resultHash"],
            "matchCount": len(similarity["matches"]),
            "repeatDeterministic": True,
        },
        "allRequestsAuthenticated": True,
        "loopbackOnly": True,
        "startupTimeoutSeconds": FRESH_HTTP_STARTUP_TIMEOUT_SECONDS,
        "perRunTimeoutSeconds": FRESH_HTTP_RUN_TIMEOUT_SECONDS,
    }


def smoke_research_export(
    research_root: str | Path, *, allow_test_fixture: bool = False
) -> Path:
    from .publish import _build_registry_payload

    root = _safe_root(research_root)
    path = root / "exports/research/smoke-report.json"
    if path.exists():
        raise FileExistsError(f"research smoke report already exists: {path}")
    export = load_export_manifest(root)
    _require_publishable_corpus(
        export, allow_test_fixture=allow_test_fixture, stage="smoke"
    )
    checkpoint_path = root / "exports/research/checkpoint.pt"
    if file_sha256(checkpoint_path) != export["checkpointSha256"]:
        raise ValueError("research export checkpoint hash mismatch before fresh HTTP smoke")
    temporary_parent = root / ".smoke"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    shadow_root = Path(tempfile.mkdtemp(prefix="r1-", dir=temporary_parent)).resolve()
    if shadow_root.parent != temporary_parent.resolve():
        raise ValueError("fresh HTTP smoke temporary root escaped its parent")
    process: subprocess.Popen[str] | None = None
    process_stdout = ""
    process_stderr = ""
    termination_mode = "not-started"
    candidate_registry_hash = ""
    command: list[str] = []
    http_evidence: dict[str, Any] | None = None
    smoke_error: Exception | None = None
    port = _reserve_loopback_port()
    try:
        _clone_read_only_tree(
            root / "materialized/corpus", shadow_root / "materialized/corpus"
        )
        _clone_read_only_tree(
            root / "exports/research", shadow_root / "exports/research"
        )
        candidate_registry = _build_registry_payload(
            shadow_root, export, smoke_hash="0" * 64
        )
        candidate_registry_hash = candidate_registry["registryHash"]
        _atomic_json(shadow_root / "published/registry.json", candidate_registry)
        formal_root = shadow_root / "formal"
        formal_root.mkdir()
        for name in (
            "core-serving-control.json",
            "core-serving-registry.json",
            "core-serving-graph-catalog.json",
        ):
            shutil.copy2(_formal_contract_source(name), formal_root / name)
        artifact_root = formal_root / "artifacts"
        artifact_root.mkdir()
        token_file = formal_root / "session.token"
        command = [
            sys.executable,
            "-m",
            "socialgraph_gfm.core.inference_cli",
            "--runtime-root",
            str(formal_root),
            "--serving-control",
            str(formal_root / "core-serving-control.json"),
            "--artifact-root",
            str(artifact_root),
            "--token-file",
            str(token_file),
            "--research-root",
            str(shadow_root),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        package_root = Path(__file__).resolve().parents[4]
        environment = dict(os.environ)
        source_root = str(package_root / "src")
        environment["PYTHONPATH"] = source_root + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
        environment["PYTHONUNBUFFERED"] = "1"
        if export["testOnly"] is True:
            environment["SOCIALGRAPH_FM_INTERNAL_TEST_FIXTURE"] = "1"
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            cwd=package_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creation_flags,
        )
        token, capabilities = _wait_for_research_http(
            process, token_file=token_file, port=port
        )
        http_evidence = _exercise_research_http(
            port=port, token=token, capabilities=capabilities, export=export
        )
        if process.poll() is not None:
            raise RuntimeError("fresh inference_cli exited during SocialGraph-FM Research HTTP smoke")
    except Exception as error:  # noqa: BLE001 - cleanup must preserve any smoke failure
        smoke_error = error
    finally:
        if process is not None:
            if process.poll() is None:
                termination_mode = "terminate"
                process.terminate()
                try:
                    process_stdout, process_stderr = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    termination_mode = "kill-after-timeout"
                    process.kill()
                    process_stdout, process_stderr = process.communicate(timeout=10)
            else:
                termination_mode = "already-exited"
                process_stdout, process_stderr = process.communicate(timeout=10)
        shutil.rmtree(shadow_root, ignore_errors=True)
        try:
            temporary_parent.rmdir()
        except OSError:
            pass
    if smoke_error is not None:
        diagnostic = process_stderr[-4_000:].strip()
        message = f"fresh SocialGraph-FM Research HTTP smoke failed: {smoke_error}"
        if diagnostic:
            message += f"; child diagnostic: {diagnostic}"
        raise RuntimeError(message) from smoke_error
    if process is None or http_evidence is None:
        raise RuntimeError("fresh SocialGraph-FM Research HTTP smoke did not complete")
    payload: dict[str, Any] = {
        "schemaVersion": SMOKE_SCHEMA,
        "releaseId": RELEASE_ID,
        "protocol": "fresh-inference-cli-http/1.0",
        "modelVersionId": export["modelVersionId"],
        "modelVersionHash": export["modelVersionHash"],
        "artifactHash": export["artifactHash"],
        "corpusKind": export["corpusKind"],
        "testOnly": export["testOnly"],
        "checkpoint": {
            "relativePath": "exports/research/checkpoint.pt",
            "sha256": export["checkpointSha256"],
        },
        "candidateRegistryHash": candidate_registry_hash,
        "freshProcess": {
            "command": command,
            "commandHash": canonical_sha256(command),
            "pythonExecutable": str(Path(sys.executable).resolve()),
            "pythonExecutableSha256": file_sha256(Path(sys.executable)),
            "pid": process.pid,
            "host": "127.0.0.1",
            "port": port,
            "exitCode": process.returncode,
            "terminationMode": termination_mode,
            "stdoutSha256": hashlib.sha256(process_stdout.encode("utf-8")).hexdigest(),
            "stderrSha256": hashlib.sha256(process_stderr.encode("utf-8")).hexdigest(),
        },
        "httpEvidence": http_evidence,
        "passed": True,
    }
    payload["smokeHash"] = canonical_sha256(payload)
    _atomic_json(path, payload)
    return path

COMPAT_EXPORTS = (
    '_scenario_rows',
    '_preview_payload',
    '_feature_contracts',
    '_serving_checkpoint_projection',
    'export_research_model',
    'load_export_manifest',
    '_load_exported_runtime',
    '_formal_contract_source',
    '_clone_read_only_tree',
    '_reserve_loopback_port',
    '_http_json',
    '_verify_wire_hash',
    '_wait_for_research_http',
    '_exercise_research_http',
    'smoke_research_export',
)

__all__ = [
    'export_research_model',
    'load_export_manifest',
    'smoke_research_export',
]
