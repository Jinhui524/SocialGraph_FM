from __future__ import annotations

import os
import json
import threading
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

import socialgraph_gfm.core.safe_paths as safe_paths
import socialgraph_gfm.core.inference_service as inference_service
from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.bundle import calculate_graph_version_hash
from socialgraph_gfm.core.inference_contracts import (
    AuthorizedGraphReference,
    InternalCreateRunRequest,
)
from socialgraph_gfm.core.inference_service import RunStore
from tests.test_core_inference_fix_round1 import (
    _catalog,
    _serving_registry,
    _wait_terminal,
    FEATURE_DESCRIPTOR,
    HASHES,
)
from _core_inference_test_support import (
    _make_test_internal_create_request,
    _make_test_serving_control,
)


def test_confined_snapshot_rejects_parent_identity_swap_at_handle_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "authorized"
    parent = root / "nested"
    parent.mkdir(parents=True)
    (parent / "payload.json").write_bytes(b'{"trusted":true}')
    original = root / "original"

    def swap(_final: Path) -> None:
        os.replace(parent, original)
        parent.mkdir()
        (parent / "payload.json").write_bytes(b'{"trusted":false}')

    monkeypatch.setattr(safe_paths, "_PATH_WALK_SEAM", swap)

    with pytest.raises(ValueError, match="changed during handle walk"):
        safe_paths.read_confined_snapshot(root, "nested/payload.json", max_bytes=1024)


def test_confined_snapshot_reads_exact_bytes_through_held_handles(tmp_path: Path) -> None:
    root = tmp_path / "authorized"
    root.mkdir()
    payload = root / "payload.json"
    payload.write_bytes(b'{"trusted":true}')

    assert safe_paths.read_confined_snapshot(root, "payload.json", max_bytes=1024) == (
        b'{"trusted":true}'
    )


def test_recovery_committed_result_wins_after_success_state_write_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _make_test_serving_control(tmp_path, registry, catalog)
    runtime = tmp_path / "gfm-runtime" / "inference"
    crashed = False

    def crash_after_marker(stage: str, _run_id: str) -> None:
        nonlocal crashed
        if stage == "after-success-marker" and not crashed:
            crashed = True
            raise OSError("injected success-state publication crash")

    monkeypatch.setattr(inference_service, "_PUBLICATION_SEAM", crash_after_marker)
    store = RunStore(runtime, registry=registry, artifact_catalog=catalog, serving_control=control)
    created = store.create(_make_test_internal_create_request(reference, control))
    _wait_terminal(store, created.status.run_id)
    assert store.get(created.status.run_id).status == "failed"
    assert (runtime / "runs" / created.status.run_id / "success.json").is_file()

    recovered = RunStore(runtime, registry=registry, artifact_catalog=catalog, serving_control=control)

    assert recovered.get(created.status.run_id).status == "succeeded"
    assert recovered.get_result(created.status.run_id).run_id == created.status.run_id


def test_run_executes_only_captured_bytes_after_registry_and_catalog_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _make_test_serving_control(tmp_path, registry, catalog)
    runtime = tmp_path / "gfm-runtime" / "inference"
    entered = threading.Event()
    release = threading.Event()

    def pause(stage: str, _run_id: str) -> None:
        if stage == "before-production-materialize":
            entered.set()
            assert release.wait(5)

    monkeypatch.setattr(inference_service, "_EXECUTION_SEAM", pause)
    store = RunStore(runtime, registry=registry, artifact_catalog=catalog, serving_control=control)
    created = store.create(_make_test_internal_create_request(reference, control))
    assert entered.wait(5)

    replacement_registry = registry.path.with_suffix(".replacement")
    replacement_registry.write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.core-serving-registry/2.0",
                "generation": 2,
                "models": [],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    os.replace(replacement_registry, registry.path)
    replacement_catalog = catalog.path.with_suffix(".replacement")
    replacement_catalog.write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.core-serving-graph-catalog/1.0",
                "generation": 2,
                "artifacts": [],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    os.replace(replacement_catalog, catalog.path)
    release.set()

    _wait_terminal(store, created.status.run_id)
    assert store.get(created.status.run_id).status == "succeeded"
    assert store.get_result(created.status.run_id).run_id == created.status.run_id
    # Subsequent creates continue to use the immutable accepted control snapshot,
    # rather than mutable registry/catalog paths replaced after capture.
    store.create(_make_test_internal_create_request(reference, control))


def test_fresh_process_serves_same_feature_contract_with_different_node_count(
    tmp_path: Path,
) -> None:
    catalog, _source_reference, bundle_path = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    target: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [
            {"id": "a", "index": 0},
            {"id": "b", "index": 1},
            {"id": "c", "index": 2},
        ],
        "edges": [
            {"sourceId": "a", "targetId": "b", "edgeType": "supports", "weight": 1.0},
            {"sourceId": "b", "targetId": "c", "edgeType": "supports", "weight": 1.0},
        ],
        "nodeFeatures": [
            {"kind": "numeric", "name": "score", "values": [-10.0, 0.5, 10.0]}
        ],
        "structuralFeatures": {
            "names": ["degree"],
            "values": [[999.0], [999.0], [999.0]],
        },
        "source": {"sourceName": "round2-target", "sourceSha256": HASHES["1"]},
        "splitManifest": {"strategy": "official", "assignments": []},
    }
    target["graphVersionHash"] = calculate_graph_version_hash(target)
    bundle_path.write_text(json.dumps(target, separators=(",", ":")), encoding="utf-8")
    bundle_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    feature_hash = canonical_sha256(FEATURE_DESCRIPTOR)
    catalog.path.write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.core-serving-graph-catalog/1.0",
                "generation": 2,
                "artifacts": [{
                    "artifactId": "artifact-target",
                    "artifactHash": HASHES["d"],
                    "bundleSha256": bundle_sha,
                    "relativePath": "bundle.json",
                    "graphVersionId": "graph-target",
                    "sourceGraphFactHash": HASHES["c"],
                    "graphVersionHash": target["graphVersionHash"],
                    "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
                    "featureContract": FEATURE_DESCRIPTOR,
                    "featureContractHash": feature_hash,
                    "nodeCount": 3,
                    "edgeCount": 2,
                }],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    reference = AuthorizedGraphReference.model_validate({
        "schemaVersion": "socialgraph-fm.core-authorized-graph-reference/2.1",
        "graphVersionId": "graph-target",
        "sourceGraphFactHash": HASHES["c"],
        "graphVersionHash": target["graphVersionHash"],
        "artifactId": "artifact-target",
        "artifactHash": HASHES["d"],
        "bundleSha256": bundle_sha,
        "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "featureContractHash": feature_hash,
        "nodeCount": 3,
        "edgeCount": 2,
    })
    control = _make_test_serving_control(registry.runtime_root, registry, catalog)
    envelope = _make_test_internal_create_request(
        reference.model_copy(update={"graph_version_id": "graph-v1"}), control
    ).model_dump(mode="json", by_alias=True)
    envelope["request"]["graphVersionId"] = "graph-target"
    envelope["request"]["targetScope"] = {
        "kind": "risk-review", "nodeIds": ["c"], "edgeIds": []
    }
    envelope["graphReference"] = reference.model_dump(mode="json", by_alias=True)
    validated = InternalCreateRunRequest.model_validate(envelope)
    envelope_path = tmp_path / "target-envelope.json"
    envelope_path.write_text(
        validated.model_dump_json(by_alias=True), encoding="utf-8"
    )
    code = """
import json,sys,time
from pathlib import Path
from socialgraph_gfm.core.artifact_catalog import ArtifactCatalog
from socialgraph_gfm.core.inference_contracts import InternalCreateRunRequest
from socialgraph_gfm.core.inference_service import RunStore
from socialgraph_gfm.core.serving_control import ServingControlStore
from socialgraph_gfm.core.serving_registry import ServingRegistry
registry=ServingRegistry.load(sys.argv[1],runtime_root=sys.argv[2])
catalog=ArtifactCatalog.load(sys.argv[3],artifact_root=sys.argv[4])
control=ServingControlStore.load(sys.argv[7],high_water_root=Path(sys.argv[5])/'serving-control')
store=RunStore(sys.argv[5],registry=registry,artifact_catalog=catalog,serving_control=control)
request=InternalCreateRunRequest.model_validate_json(Path(sys.argv[6]).read_bytes())
created=store.create(request)
deadline=time.monotonic()+10
while store.get(created.status.run_id).status not in {'succeeded','failed'}:
    assert time.monotonic()<deadline
    time.sleep(.02)
result=store.get_result(created.status.run_id)
print(json.dumps({'status':store.get(created.status.run_id).status,'findings':len(result.findings)}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(registry.path),
            str(registry.runtime_root),
            str(catalog.path),
            str(catalog.artifact_root),
            str(tmp_path / "fresh-runtime"),
            str(envelope_path),
            str(control.path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {
        "status": "succeeded",
        "findings": 1,
    }


def test_legacy_training_recovery_checkpoint_never_becomes_serving_ready(
    tmp_path: Path,
) -> None:
    catalog, _reference, _ = _catalog(tmp_path)
    registry = _serving_registry(
        tmp_path, catalog=catalog, include_adapter_schema=False
    )

    with pytest.raises(ValueError, match="fitted schema"):
        registry.capabilities()
