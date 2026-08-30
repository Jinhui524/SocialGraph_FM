from __future__ import annotations

import http.client
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from socialgraph_gfm.canonical import canonical_sha256, file_sha256
from socialgraph_gfm.global_model.contracts import (
    COUNTRY_IDS,
    GRAPH_STAT_NAMES,
    TRACE_ARRAY_TOKENS,
    GlobalCorpusEntry,
    GlobalCorpusManifest,
    GlobalCountryManifest,
    GlobalSplitDescriptor,
    atomic_write_contract,
)
from socialgraph_gfm.global_model.converter import write_array_artifact
from socialgraph_gfm.global_model.service import PROTOCOLS, GlobalServingRuntime


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_hashed_json(path: Path, payload: dict, field: str) -> None:
    _write_json(path, {**payload, field: canonical_sha256(payload)})


def _safe_corpus(root: Path) -> tuple[str, str, np.ndarray]:
    node_count = 716
    country_root = root / "corpus/countries/russia"
    fused_indptr = np.concatenate(
        (
            np.asarray([0, 1, 3, 4], dtype=np.int64),
            np.full(node_count - 3, 4, dtype=np.int64),
        )
    )
    structure_missing = np.diff(fused_indptr) == 0
    arrays: dict[str, np.ndarray] = {
        "edge_index": np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        "fused_indptr": fused_indptr,
        "fused_indices": np.asarray([1, 0, 2, 1], dtype=np.int64),
        "text_features": np.zeros((node_count, 768), dtype=np.float32),
        "degree_bucket": np.asarray([1, 2, 1, *([0] * (node_count - 3))], dtype=np.uint8),
        "structure_missing": structure_missing,
        "graph_stats": np.zeros(len(GRAPH_STAT_NAMES), dtype=np.float32),
        "labels": (np.arange(node_count) % 2).astype(np.uint8),
        "trace_membership": np.zeros((node_count, 5), dtype=np.bool_),
        "split_full_0_train": np.arange(node_count) == 0,
        "split_full_0_validation": np.arange(node_count) == 1,
        "split_full_0_test": np.arange(node_count) == 2,
    }
    arrays["trace_membership"][[0, 1], 0] = True
    arrays["trace_membership"][[1, 2], 4] = True
    for trace_name, token in TRACE_ARRAY_TOKENS.items():
        if trace_name == "coRT":
            indptr = np.concatenate(
                (np.asarray([0, 1, 2], dtype=np.int64), np.full(node_count - 2, 2))
            )
            indices = np.asarray([1, 0], dtype=np.int64)
            weights = np.asarray([2.5, 2.5], dtype=np.float64)
        elif trace_name == "tweetSim":
            indptr = np.concatenate(
                (np.asarray([0, 0, 1, 2], dtype=np.int64), np.full(node_count - 3, 2))
            )
            indices = np.asarray([2, 1], dtype=np.int64)
            weights = np.asarray([0.75, 0.75], dtype=np.float64)
        else:
            indptr = np.zeros(node_count + 1, dtype=np.int64)
            indices = np.empty(0, dtype=np.int64)
            weights = np.empty(0, dtype=np.float64)
        arrays[f"relation_{token}_indptr"] = indptr
        arrays[f"relation_{token}_indices"] = indices
        arrays[f"relation_{token}_weights"] = weights
    descriptors = tuple(
        write_array_artifact(country_root, name=name, array=array)
        for name, array in arrays.items()
    )
    split = GlobalSplitDescriptor.create(
        split_id="full-fold-0",
        regime="full",
        fold=0,
        train_array="split_full_0_train",
        validation_array="split_full_0_validation",
        test_array="split_full_0_test",
    )
    source_hashes = {"pickle:full": "1" * 64, "textTensor": "2" * 64}
    country = GlobalCountryManifest.create(
        country_id="russia",
        node_count=node_count,
        edge_count=4,
        arrays=descriptors,
        splits=(split,),
        source_hashes=source_hashes,
        relation_edge_counts={
            trace_name: 2 if trace_name in {"coRT", "tweetSim"} else 0
            for trace_name in TRACE_ARRAY_TOKENS
        },
        preprocessing={"fixture": "safe-npy-csr"},
    )
    atomic_write_contract(country_root / "manifest.json", country)
    entries = []
    for index, country_id in enumerate(COUNTRY_IDS, start=1):
        if country_id == "russia":
            entries.append(
                GlobalCorpusEntry.from_country_manifest(
                    country,
                    manifest_path="countries/russia/manifest.json",
                )
            )
        else:
            entries.append(
                GlobalCorpusEntry(
                    countryId=country_id,
                    manifestPath=f"countries/{country_id}/manifest.json",
                    manifestHash=f"{index}" * 64,
                    sourceHashes=source_hashes,
                    splitHashes={split.split_id: split.split_hash},
                )
            )
    corpus = GlobalCorpusManifest.create(entries)
    atomic_write_contract(root / "corpus/manifest.json", corpus)
    return corpus.content_hash, country.content_hash, structure_missing


def _published_root(tmp_path: Path) -> Path:
    corpus_hash, graph_hash, structure_missing = _safe_corpus(tmp_path)
    preview = tmp_path / "exports/socialgraph-global/previews/russia.json"
    _write_hashed_json(
        preview,
        {
            "schemaVersion": "socialgraph-fm.global-model-preview/1.0",
            "releaseId": "socialgraph-fm",
            "datasetVersionId": "socialgraph-fm:russia",
            "graphVersionHash": graph_hash,
            "nodes": [
                {"id": 0, "label": "Account 0", "degree": 1, "structureMissing": False},
                {"id": 1, "label": "Account 1", "degree": 2, "structureMissing": False},
                {"id": 2, "label": "Account 2", "degree": 1, "structureMissing": False},
            ],
            "edges": [
                {"id": "0:1", "source": 0, "target": 1, "modality": "coRT"},
                {"id": "1:2", "source": 1, "target": 2, "modality": "tweetSim"},
            ],
            "nodeCount": 716,
            "edgeCount": 2,
            "partialPreview": True,
            "traceNames": list(TRACE_ARRAY_TOKENS),
        },
        "previewHash",
    )
    protocol_artifacts = {}
    protocol_models = {}
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
    for protocol_index, protocol in enumerate(PROTOCOLS, start=1):
        checkpoint = tmp_path / f"exports/socialgraph-global/checkpoints/{protocol}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"{protocol}-weights-only-placeholder".encode())
        result_dir = tmp_path / "exports/socialgraph-global/results"
        result_dir.mkdir(parents=True, exist_ok=True)
        npz_path = result_dir / f"{protocol}-russia.npz"
        node_ids = np.arange(716, dtype=np.int64)
        scores = np.linspace(1.0, 0.0, 716, dtype=np.float32)
        np.savez_compressed(
            npz_path,
            node_ids=node_ids,
            scores=scores,
            logits=np.log(np.clip(scores, 1e-5, 1 - 1e-5) / np.clip(1 - scores, 1e-5, 1)),
            structure_missing=structure_missing,
            router_indices=np.tile(np.asarray([[4, 7]], dtype=np.int64), (716, 1)),
            router_weights=np.tile(np.asarray([[0.7, 0.3]], dtype=np.float32), (716, 1)),
            modality_counts=np.column_stack(
                (
                    np.asarray([1, 1, *([0] * 714)], dtype=np.int32),
                    np.zeros(716, dtype=np.int32),
                    np.zeros(716, dtype=np.int32),
                    np.zeros(716, dtype=np.int32),
                    np.asarray([0, 1, 1, *([0] * 713)], dtype=np.int32),
                )
            ),
        )
        json_path = result_dir / f"{protocol}-russia.json"
        model_version_hash = (
            "a" * 64 if protocol == "global" else f"{protocol_index}" * 64
        )
        model_state_hash = f"{protocol_index + 4}" * 64
        model_version_id = f"socialgraph-fm-{protocol.replace('_', '-')}/test"
        result_metadata = {
            "schemaVersion": "socialgraph-fm.global-model-result/1.0",
            "releaseId": "socialgraph-fm",
            "taskId": "coordination_risk",
            "protocol": protocol,
            "country": "russia",
            "nodeCount": 716,
            "graphVersionHash": graph_hash,
            "corpusHash": corpus_hash,
            "splitHash": "f" * 64,
            "threshold": 0.5,
            "expertNames": expert_names,
            "modelVersionId": model_version_id,
            "modelVersionHash": model_version_hash,
            "modelStateHash": model_state_hash,
            "npzPath": npz_path.relative_to(tmp_path).as_posix(),
            "npzSha256": file_sha256(npz_path),
        }
        _write_hashed_json(json_path, result_metadata, "resultHash")
        protocol_artifacts[protocol] = {
            "resultPaths": {
                "russia": {
                    "jsonPath": json_path.relative_to(tmp_path).as_posix(),
                    "jsonSha256": file_sha256(json_path),
                    "npzPath": npz_path.relative_to(tmp_path).as_posix(),
                    "npzSha256": file_sha256(npz_path),
                }
            },
            "splitHash": "f" * 64,
            "threshold": 0.5,
            "temperature": 1.0,
            "bias": 0.0,
            "metrics": {"macroF1": 0.8, "prAuc": 0.7},
            "labelledTrainNodes": 429,
            "protocolModelVersionId": model_version_id,
            "protocolModelVersionHash": model_version_hash,
            "modelStateHash": model_state_hash,
            "checkpointPath": checkpoint.relative_to(tmp_path).as_posix(),
            "checkpointSha256": file_sha256(checkpoint),
        }
        protocol_models[protocol] = {
            "modelVersionId": model_version_id,
            "modelVersionHash": model_version_hash,
            "modelStateHash": model_state_hash,
            "state": "servingReady" if protocol == "global" else "frozenDemo",
        }
    model_card_path = tmp_path / "exports/socialgraph-global/model-card.json"
    model_card = {
        "schemaVersion": "socialgraph-fm.global-model-card/1.0",
        "releaseId": "socialgraph-fm",
        "modelVersionId": "socialgraph-fm-global/test",
        "modelVersionHash": "a" * 64,
        "taskId": "coordination_risk",
        "architecture": {
            "name": "Global cross-modal GraphSAGE with sparse routing",
            "textFeatures": "anonymous precomputed 768-dimensional embeddings",
            "structuralFeatures": "factual 128-bucket node degree",
            "gnnLayers": 2,
            "hiddenDim": 256,
            "router": "shared residual plus top-2 domain/null adapters",
        },
        "protocols": protocol_models,
        "trainingData": {
            "countries": list(COUNTRY_IDS),
            "nodeCount": 716 * len(COUNTRY_IDS),
            "nodeCountByCountry": {country: 716 for country in COUNTRY_IDS},
            "content": "anonymous graph data with no raw text",
        },
        "intendedUse": ["analyst-facing prioritization with human review"],
        "outOfScope": ["automatic enforcement"],
        "limitations": ["frozen static research snapshot"],
        "ethics": ["preserve anonymity and require human review"],
        "licenses": [
            {"name": "Information-operations dataset", "license": "CC-BY-4.0", "url": "https://zenodo.org/records/13357621"},
            {
                "name": ("InfoOps" + "GFM") + " code",
                "license": "MIT",
                "url": "https://github.com/mminici/" + ("InfoOps" + "GFM"),
            },
        ],
        "sourceAttribution": {
            "kind": "inspired",
            "paperUrl": "https://proceedings.mlr.press/v267/yuan25h.html",
            "completeReproduction": False,
        },
        "metrics": {protocol: {"countryBalancedMacroF1": 0.8} for protocol in PROTOCOLS},
        "artifactHash": "b" * 64,
    }
    _write_hashed_json(model_card_path, model_card, "modelCardHash")
    global_checkpoint = tmp_path / "exports/socialgraph-global/checkpoints/global.pt"
    registry = {
        "schemaVersion": "socialgraph-fm.global-model-registry/1.0",
        "modelVersionId": "socialgraph-fm-global/test",
        "modelVersionHash": "a" * 64,
        "artifactHash": "b" * 64,
        "corpusHash": corpus_hash,
        "sourceCodeHash": "d" * 64,
        "state": "servingReady",
        "protocols": list(PROTOCOLS),
        "graphVersionHash": graph_hash,
        "checkpointPath": global_checkpoint.relative_to(tmp_path).as_posix(),
        "checkpointSha256": file_sha256(global_checkpoint),
        "russiaPreviewPath": preview.relative_to(tmp_path).as_posix(),
        "russiaPreviewSha256": file_sha256(preview),
        "modelCardPath": model_card_path.relative_to(tmp_path).as_posix(),
        "modelCardSha256": file_sha256(model_card_path),
        "expertNames": expert_names,
        "protocolArtifacts": protocol_artifacts,
        "protocolModels": protocol_models,
    }
    _write_hashed_json(tmp_path / "registry/socialgraph-global.json", registry, "registryHash")
    return tmp_path


def test_global_model_service_uses_frozen_predictions_and_hash_bound_evidence(tmp_path: Path) -> None:
    runtime = GlobalServingRuntime(_published_root(tmp_path))
    try:
        capabilities = runtime.capabilities()
        assert capabilities["servingReady"] is True
        assert capabilities["capabilityHash"] == canonical_sha256(
            {key: value for key, value in capabilities.items() if key != "capabilityHash"}
        )
        scenario = runtime.scenario()
        health = runtime.health()
        assert health["servingReady"] is True
        assert health["modelVersionHash"] == capabilities["model"]["modelVersionHash"]
        assert runtime.model_card()["modelCardHash"]
        assert scenario["edgeCount"] == 2
        request = {
            "schemaVersion": "socialgraph-fm.gfm-global-model/1.0",
            "taskId": "coordination_risk",
            "datasetVersionId": "socialgraph-fm:russia",
            "protocol": "global",
            "modelVersionId": capabilities["model"]["protocolModels"]["global"][
                "modelVersionId"
            ],
            "topK": 5,
        }
        status = runtime.create_run(
            {
                "schemaVersion": "socialgraph-fm.gfm-global-model/1.0",
                "request": request,
                "expectedModel": capabilities["model"],
                "datasetBinding": {
                    "datasetVersionId": "socialgraph-fm:russia",
                    "graphVersionHash": scenario["graphVersionHash"],
                },
            }
        )
        for _ in range(100):
            status = runtime.get_run(status["runId"])
            if status["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        assert status["status"] == "succeeded", status
        result = runtime.get_result(status["runId"])
        assert len(result["findings"]) == 5
        assert result["findings"][0]["nodeId"] == "russia:0"
        assert result["findings"][0]["modalityEvidence"]["coRT"] == 1
        evidence = runtime.evidence(status["runId"], "russia:0")
        assert evidence["neighbors"][0]["nodeId"] == "russia:1"
        assert evidence["neighbors"][0]["relations"] == [
            {"modality": "coRT", "rawWeight": 2.5}
        ]
        assert evidence["neighbors"][0]["riskBand"] == "high"
        assert evidence["structuralSignals"]["fusedDegree"] == 1
        assert evidence["evidenceSubgraph"]["nodeCount"] == 3
        assert evidence["evidenceSubgraph"]["edgeCount"] == 2
        assert evidence["resultHash"] == result["resultHash"]
        assert evidence["graphVersionHash"] == result["graphVersionHash"]
        assert evidence["modelVersionHash"] == result["modelVersionHash"]
        assert evidence["evidenceHash"] == canonical_sha256(
            {key: value for key, value in evidence.items() if key != "evidenceHash"}
        )
    finally:
        runtime.close()


def test_loopback_listener_exposes_hash_bound_global_model_health_and_model_card(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from socialgraph_gfm.core.inference_service import create_server

    runtime = GlobalServingRuntime(_published_root(tmp_path))
    token = "session-" + "x" * 64
    server = create_server(
        "127.0.0.1",
        0,
        token=token,
        runtime=SimpleNamespace(),  # type: ignore[arg-type]
        global_model_runtime=runtime,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        for path, hash_field in (
            ("/internal/global-model/health", "healthHash"),
            ("/internal/global-model/model-card", "modelCardHash"),
        ):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request(
                "GET",
                path,
                headers={"Authorization": f"Bearer {token}"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            assert response.status == 200
            assert len(payload[hash_field]) == 64
            connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_global_model_service_is_explicitly_unavailable_without_registry(tmp_path: Path) -> None:
    runtime = GlobalServingRuntime(tmp_path)
    try:
        assert runtime.capabilities()["servingReady"] is False
        assert runtime.health()["servingReady"] is False
        assert runtime.scenario()["enabled"] is False
    finally:
        runtime.close()


def test_global_model_service_close_releases_serving_corpus_mmaps(tmp_path: Path) -> None:
    root = _published_root(tmp_path)
    runtime = GlobalServingRuntime(root)
    array_path = next((root / "corpus/countries/russia").rglob("*.npy"))

    runtime.close()

    moved = array_path.with_suffix(".released")
    os.replace(array_path, moved)
    os.replace(moved, array_path)


def test_global_model_registry_fails_closed_without_safe_corpus_or_valid_model_card(
    tmp_path: Path,
) -> None:
    root = _published_root(tmp_path / "missing-corpus")
    (root / "corpus/manifest.json").unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        GlobalServingRuntime(root)

    root = _published_root(tmp_path / "tampered-card")
    card = root / "exports/socialgraph-global/model-card.json"
    payload = json.loads(card.read_text(encoding="utf-8"))
    payload["modelVersionHash"] = "0" * 64
    _write_json(card, payload)
    registry_path = root / "registry/socialgraph-global.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["modelCardSha256"] = file_sha256(card)
    registry.pop("registryHash")
    _write_hashed_json(registry_path, registry, "registryHash")
    with pytest.raises(ValueError, match="model.?[Cc]ard"):
        GlobalServingRuntime(root)

    root = _published_root(tmp_path / "tampered-registry")
    registry_path = root / "registry/socialgraph-global.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["seed"] = 999
    _write_json(registry_path, registry)
    with pytest.raises(ValueError, match="registryHash"):
        GlobalServingRuntime(root)
