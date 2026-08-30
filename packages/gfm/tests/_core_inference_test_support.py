"""Private test-only constructors for isolated core inference fixtures."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.tensor_digest import canonical_tensor_digest
from socialgraph_gfm.core.adapters import BundleInputAdapter
from socialgraph_gfm.core.artifact_catalog import ArtifactCatalog
from socialgraph_gfm.core.bundle import calculate_graph_version_hash
from socialgraph_gfm.core.checkpoint import CheckpointBindings, publish_checkpoint
from socialgraph_gfm.core.governance import GovernanceFinding
from socialgraph_gfm.core.inference_contracts import (
    AuthorizedGraphReference,
    GfmRunRequest,
    InternalCreateRunRequest,
)
from socialgraph_gfm.core.inference_service import RunLease, RunStore
from socialgraph_gfm.core.model import CoreGFM
from socialgraph_gfm.core.serving_control import ServingControlStore
from socialgraph_gfm.core.serving_registry import ServingModel, ServingRegistry

_TestExecutor = Callable[
    [GfmRunRequest, AuthorizedGraphReference, ServingModel],
    Sequence[GovernanceFinding | dict[str, object]],
]

HASHES = {letter: letter * 64 for letter in "123456789abcdef"}
FEATURE_DESCRIPTOR = {
    "schemaVersion": "socialgraph-fm.core-graph-feature-contract/2.0",
    "nodeFeatures": [{"kind": "numeric", "name": "score"}],
    "structuralFeatureNames": ["degree"],
}


def _tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    records = []
    for name, value in sorted(state.items()):
        records.append({"name": name, **canonical_tensor_digest(value)})
    return canonical_sha256(records)


def _bundle_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [{"id": "a", "index": 0}, {"id": "b", "index": 1}],
        "edges": [
            {
                "sourceId": "a",
                "targetId": "b",
                "edgeType": "supports",
                "weight": 1.0,
            }
        ],
        "nodeFeatures": [
            {"kind": "numeric", "name": "score", "values": [0.25, 0.75]}
        ],
        "structuralFeatures": {"names": ["degree"], "values": [[1.0], [1.0]]},
        "source": {"sourceName": "fix-round1", "sourceSha256": HASHES["1"]},
        "splitManifest": {"strategy": "official", "assignments": []},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return payload


def _catalog(tmp_path: Path) -> tuple[ArtifactCatalog, AuthorizedGraphReference, Path]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True)
    bundle_path = artifact_root / "bundle.json"
    bundle_path.write_text(
        json.dumps(_bundle_payload(), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    bundle_sha256 = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    artifact_hash = HASHES["f"]
    source_graph_fact_hash = HASHES["e"]
    graph_hash = str(_bundle_payload()["graphVersionHash"])
    feature_hash = canonical_sha256(FEATURE_DESCRIPTOR)
    catalog_path = tmp_path / "artifact-catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.core-serving-graph-catalog/1.0",
                "generation": 1,
                "artifacts": [
                    {
                        "artifactId": "artifact-v1",
                        "artifactHash": artifact_hash,
                        "bundleSha256": bundle_sha256,
                        "relativePath": "bundle.json",
                        "graphVersionId": "graph-v1",
                        "sourceGraphFactHash": source_graph_fact_hash,
                        "graphVersionHash": graph_hash,
                        "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
                        "featureContract": FEATURE_DESCRIPTOR,
                        "featureContractHash": feature_hash,
                        "nodeCount": 2,
                        "edgeCount": 1,
                    }
                ],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    reference = AuthorizedGraphReference.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-authorized-graph-reference/2.1",
            "graphVersionId": "graph-v1",
            "sourceGraphFactHash": source_graph_fact_hash,
            "graphVersionHash": graph_hash,
            "artifactId": "artifact-v1",
            "artifactHash": artifact_hash,
            "bundleSha256": bundle_sha256,
            "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
            "featureContractHash": feature_hash,
            "nodeCount": 2,
            "edgeCount": 1,
        }
    )
    return ArtifactCatalog.load(catalog_path, artifact_root=artifact_root), reference, bundle_path


def _serving_registry(
    tmp_path: Path,
    *,
    catalog: ArtifactCatalog,
    checkpoint_status: str = "accepted",
    promotable: bool = True,
    include_adapter_schema: bool = True,
    invalid_state: str | None = None,
) -> ServingRegistry:
    bundle = catalog.resolve(
        AuthorizedGraphReference.model_validate(
            {
                "schemaVersion": "socialgraph-fm.core-authorized-graph-reference/2.1",
                **{
                    key: value
                    for key, value in catalog.document.artifacts[0]
                    .model_dump(mode="json", by_alias=True)
                    .items()
                    if key
                    in {
                        "artifactId",
                        "artifactHash",
                        "bundleSha256",
                        "graphVersionId",
                        "sourceGraphFactHash",
                        "graphVersionHash",
                        "graphSchemaVersion",
                        "featureContractHash",
                        "nodeCount",
                        "edgeCount",
                    }
                },
            }
        )
    )
    runtime_root = tmp_path / "gfm-runtime"
    checkpoint_path = runtime_root / "checkpoints" / "model.pt"
    bindings = CheckpointBindings(
        config_hash=HASHES["2"],
        data_hash=HASHES["3"],
        code_hash=HASHES["4"],
        environment_hash=HASHES["5"],
    )
    model = CoreGFM(node_classes=2)
    adapter = BundleInputAdapter(bundle, multi_hot_buckets=32, mode="training")
    model_state = model.state_dict()
    adapter_state = adapter.state_dict()
    if invalid_state == "missing-adapter":
        adapter_state.pop(next(iter(adapter_state)))
    elif invalid_state == "unexpected-adapter":
        adapter_state["unexpected.weight"] = torch.zeros(1)
    elif invalid_state == "bad-adapter-shape":
        first = next(iter(adapter_state))
        adapter_state[first] = adapter_state[first].reshape(-1)[:1]
    elif invalid_state == "legacy-row-buffer":
        adapter_state["_field_0_values"] = torch.zeros(2, 1)
    elif invalid_state == "missing-model":
        model_state.pop("encoder.layers.0.lin_l.weight")
    trainer_state: dict[str, object] = {
        "model": model_state,
        "adapters": {"serving": adapter_state},
    }
    if include_adapter_schema:
        trainer_state["adapterSchemas"] = {
            "serving": adapter.schema.model_dump(mode="json", by_alias=True)
        }
    publish_checkpoint(
        checkpoint_path,
        trainer_state=trainer_state,
        bindings=bindings,
        status=checkpoint_status,  # type: ignore[arg-type]
        promotable=promotable,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    calibration_bindings: list[dict[str, object]] = []
    for entity_type, version, relative_path, protocol_hash, temperature, bias in (
        ("node", "node-calibration/1", "calibration/risk-node.json", HASHES["b"], 1.0, 0.0),
        ("edge", "edge-calibration/1", "calibration/risk-edge.json", HASHES["c"], 2.0, 0.25),
    ):
        calibration_payload: dict[str, object] = {
            "schemaVersion": "socialgraph-fm.core-score-calibration/2.0",
            "calibrationVersion": version,
            "method": "sigmoid",
            "temperature": temperature,
            "bias": bias,
            "protocolHash": protocol_hash,
        }
        calibration_payload["artifactHash"] = canonical_sha256(calibration_payload)
        calibration_path = runtime_root / relative_path
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(
            json.dumps(calibration_payload, separators=(",", ":")), encoding="utf-8"
        )
        calibration_bindings.append(
            {
                "entityType": entity_type,
                "confidenceKind": "binary-calibration",
                "calibrationVersion": version,
                "calibrationMethod": "sigmoid",
                "calibrationArtifactHash": calibration_payload["artifactHash"],
                "calibrationRelativePath": relative_path,
                "calibrationSha256": hashlib.sha256(
                    calibration_path.read_bytes()
                ).hexdigest(),
                "calibrationProtocolHash": protocol_hash,
                "adapterDomain": "serving",
                "adapterSchemaHash": adapter.schema.adapter_schema_hash,
                "adapterStateHash": _tensor_state_hash(adapter_state),
                "graphFeatureContractHash": catalog.document.artifacts[
                    0
                ].feature_contract_hash,
            }
        )
    task_heads = [
        {
            "taskId": "core.risk_and_trust_review",
            "kind": "risk-and-trust",
            "nodeOutputIndex": 1,
            "calibrations": calibration_bindings,
        }
    ]
    serving_manifest = {
        "schemaVersion": "socialgraph-fm.core-serving-checkpoint-manifest/1.1",
        "task4CheckpointSha256": checkpoint_sha256,
        "accepted": True,
        "promotable": True,
        "modelStateHash": _tensor_state_hash(model_state),
        "adapterStateHash": _tensor_state_hash(adapter_state),
        "adapterSchemaHash": adapter.schema.adapter_schema_hash,
        "adapterDomain": "serving",
        "nodeClasses": 2,
        "multiHotBuckets": 32,
        "adapterBindings": [
            {
                "adapterDomain": "serving",
                "adapterSchemaHash": adapter.schema.adapter_schema_hash,
                "adapterStateHash": _tensor_state_hash(adapter_state),
                "multiHotBuckets": 32,
            }
        ],
        "taskHeads": task_heads,
    }
    manifest_path = runtime_root / "checkpoints" / "model.serving.json"
    manifest_path.write_text(
        json.dumps(serving_manifest, separators=(",", ":")), encoding="utf-8"
    )
    checkpoint = {
        "relativePath": "checkpoints/model.pt",
        "sha256": checkpoint_sha256,
        "servingManifestRelativePath": "checkpoints/model.serving.json",
        "servingManifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "bindings": {
            "configHash": bindings.config_hash,
            "dataHash": bindings.data_hash,
            "codeHash": bindings.code_hash,
            "environmentHash": bindings.environment_hash,
        },
        "adapterDomain": "serving",
        "nodeClasses": 2,
        "multiHotBuckets": 32,
    }
    model_payload: dict[str, object] = {
        "modelVersionId": "socialgraph-fm-core/review",
        "state": "servingReady",
        "checkpoint": checkpoint,
        "taskHeads": task_heads,
        "tasks": ["core.risk_and_trust_review"],
        "graphSchemaVersions": ["socialgraph-fm.core-graph-bundle/2.0"],
        "graphFeatureContractHash": canonical_sha256(
            [
                {
                    "taskId": head["taskId"],
                    "entityType": binding["entityType"],
                    "featureContractHash": binding["graphFeatureContractHash"],
                }
                for head in task_heads
                for binding in head["calibrations"]
            ]
        ),
        "maxNodes": 1000,
        "maxEdges": 5000,
    }
    model_payload["modelVersionHash"] = canonical_sha256(
        {key: value for key, value in model_payload.items() if key != "state"}
    )
    registry_path = runtime_root / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.core-serving-registry/2.0",
                "generation": 1,
                "models": [model_payload],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return ServingRegistry.load(registry_path, runtime_root=runtime_root)


def _wait_terminal(store: RunStore, run_id: str) -> None:
    deadline = time.monotonic() + 10
    while store.get(run_id).status not in {"succeeded", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.02)


def _make_test_serving_control(
    root: Path,
    registry: ServingRegistry,
    catalog: ArtifactCatalog,
) -> ServingControlStore:
    """Create a coherent control file from Python fixture objects, never CLI input."""

    sources = root / "test-serving-control-sources"
    sources.mkdir(parents=True, exist_ok=True)
    registry_path = sources / "registry.json"
    catalog_path = sources / "catalog.json"
    registry_bytes = registry.path.read_bytes()
    catalog_bytes = catalog.path.read_bytes()
    registry_path.write_bytes(registry_bytes)
    catalog_path.write_bytes(catalog_bytes)
    registry_document = json.loads(registry_bytes)
    catalog_document = json.loads(catalog_bytes)
    payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-serving-control/1.0",
        "generation": 1,
        "registry": {
            "relativePath": registry_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(registry_bytes).hexdigest(),
            "semanticHash": canonical_sha256(registry_document),
            "generation": registry_document["generation"],
        },
        "catalog": {
            "relativePath": catalog_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "semanticHash": canonical_sha256(catalog_document),
            "generation": catalog_document["generation"],
        },
    }
    payload["controlHash"] = canonical_sha256(payload)
    path = root / "test-serving-control.json"
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return ServingControlStore.load(path, high_water_root=root / "test-high-water")


def _make_test_internal_create_request(
    reference: AuthorizedGraphReference,
    control: ServingControlStore,
) -> InternalCreateRunRequest:
    snapshot = control.capture()
    return InternalCreateRunRequest.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-internal-create-run/2.1",
            "request": {
                "schemaVersion": "socialgraph-fm.core-run-request/2.0",
                "graphVersionId": "graph-v1",
                "taskId": "core.risk_and_trust_review",
                "targetScope": {"kind": "risk-review", "nodeIds": ["a"], "edgeIds": []},
                "modelVersionId": "socialgraph-fm-core/review",
                "parameters": {"kind": "risk-and-trust", "topKSimilarCases": 0},
            },
            "graphReference": reference.model_dump(mode="json", by_alias=True),
            "expectedServingControl": {
                "controlHash": snapshot.document.control_hash,
                "controlGeneration": snapshot.document.generation,
                "registryHash": snapshot.registry_hash,
                "registryGeneration": snapshot.registry_document.generation,
                "catalogHash": snapshot.catalog_hash,
                "catalogGeneration": snapshot.catalog_document.generation,
                "modelVersionHash": snapshot.registry_document.models[0].model_version_hash,
            },
        }
    )


class _SyntheticTestRunStore(RunStore):
    def __init__(self, *args: object, executor: _TestExecutor, **kwargs: object) -> None:
        self._executor = executor
        super().__init__(*args, **kwargs)

    def _production_execute(
        self, request: GfmRunRequest, lease: RunLease
    ) -> Sequence[GovernanceFinding | dict[str, object]]:
        model, _checkpoint, _calibrations, _schema_hash = lease.model.materialize()
        return self._executor(request, lease.graph.reference, model)


def _make_test_only_run_store(
    root: Path,
    *,
    registry: ServingRegistry,
    artifact_catalog: ArtifactCatalog,
    serving_control: ServingControlStore,
    executor: _TestExecutor,
) -> RunStore:
    """Inject synthetic execution through a private Python-only test factory."""

    return _SyntheticTestRunStore(
        root,
        registry=registry,
        artifact_catalog=artifact_catalog,
        serving_control=serving_control,
        executor=executor,
    )
