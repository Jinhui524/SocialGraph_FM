"""Private test-only constructors for isolated core inference fixtures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.artifact_catalog import ArtifactCatalog
from socialgraph_gfm.core.governance import GovernanceFinding
from socialgraph_gfm.core.inference_contracts import (
    AuthorizedGraphReference,
    GfmRunRequest,
    InternalCreateRunRequest,
)
from socialgraph_gfm.core.inference_service import RunLease, RunStore
from socialgraph_gfm.core.serving_control import ServingControlStore
from socialgraph_gfm.core.serving_registry import ServingModel, ServingRegistry

_TestExecutor = Callable[
    [GfmRunRequest, AuthorizedGraphReference, ServingModel],
    Sequence[GovernanceFinding | dict[str, object]],
]


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
