from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from socialgraph_gfm.canonical import canonical_sha256  # noqa: E402
from socialgraph_gfm.global_model.model import GlobalModel, GlobalModelConfig  # noqa: E402
from socialgraph_gfm.governance.analytics import derive_analytics  # noqa: E402
from socialgraph_gfm.governance.bundle import create_tiny_contract_bundle  # noqa: E402
from socialgraph_gfm.governance.inference import (  # noqa: E402
    InferenceCancelled,
    LoadedGlobalModel,
    OnlineInferenceOutputs,
)
from socialgraph_gfm.governance.materialize import (  # noqa: E402
    _dataset_content_hash,
    _graph_version_hash,
    load_materialized_artifact,
    materialize_bundle,
)
from socialgraph_gfm.governance import service as service_module  # noqa: E402

from test_governance_materialize import (  # noqa: E402
    TINY_ARTIFACT_ID,
    TINY_DATASET_HASH,
    TINY_GRAPH_HASH,
    _install_bundle,
)


def _loaded_model() -> LoadedGlobalModel:
    model = GlobalModel(GlobalModelConfig(dropout=0.0)).eval()
    model.requires_grad_(False)
    return LoadedGlobalModel(
        model=model,
        device=torch.device("cpu"),
        device_name="cpu",
        dtype_name="float32",
        model_version_id="socialgraph-fm-global/test",
        model_version_hash="1" * 64,
        model_state_hash="2" * 64,
        allowed_experts=(
            "domain:china",
            "domain:cuba",
            "domain:iran",
            "domain:russia",
            "domain:UAE",
            "domain:venezuela",
            "null",
        ),
        expert_names=(
            "shared",
            "domain:china",
            "domain:cuba",
            "domain:iran",
            "domain:russia",
            "domain:UAE",
            "domain:venezuela",
            "null",
        ),
        temperature=1.5,
        bias=-0.2,
        threshold=0.6,
        reference_metrics={"macroF1": 0.9},
        loaded_at="2026-08-18T00:00:00.000000Z",
        execution_environment_hash="4" * 64,
        runtime_recipe_hash="3" * 64,
    )


def _direct_forward(data, loaded, *, progress, cancelled) -> OnlineInferenceOutputs:
    if cancelled():
        raise InferenceCancelled
    output = loaded.model(
        torch.tensor(np.asarray(data.text_features), dtype=torch.float32),
        torch.tensor(np.asarray(data.degree_bucket), dtype=torch.long),
        torch.tensor(np.asarray(data.edge_index), dtype=torch.long),
        domain_id=None,
        graph_stats=torch.tensor(np.asarray(data.graph_stats), dtype=torch.float32),
        allowed_experts=loaded.allowed_experts,
    )
    progress(1.0)
    assert output.router_indices is not None and output.router_weights is not None
    logits = output.logits.detach().numpy().astype(np.float32)
    scores = torch.sigmoid(output.logits / loaded.temperature + loaded.bias).detach().numpy()
    counts = np.column_stack(
        [
            np.diff(np.asarray(data.arrays[f"relation_{name.lower()}_indptr"]))
            for name in ("coRT", "coURL", "hashSeq", "fastRT", "tweetSim")
        ]
    ).astype(np.int32)
    return OnlineInferenceOutputs(
        logits=logits,
        scores=scores.astype(np.float32),
        embeddings=output.node_embeddings.detach().numpy().astype(np.float32),
        router_indices=output.router_indices.detach().numpy().astype(np.int16),
        router_weights=output.router_weights.detach().numpy().astype(np.float32),
        modality_contributions=output.modality_contributions.detach().numpy().astype(np.float32),
        modality_counts=counts,
        batch_size=128,
        peak_memory_mib=None,
        seed=123,
    )


def _runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "v2"
    bundle = create_tiny_contract_bundle(tmp_path / "tiny.zip")
    _install_bundle(root, bundle)
    materialize_bundle(
        root,
        TINY_ARTIFACT_ID,
        expected_dataset_content_hash=TINY_DATASET_HASH,
        expected_graph_version_hash=TINY_GRAPH_HASH,
        clean_self_loops=False,
    )
    loaded = _loaded_model()
    monkeypatch.setattr(service_module, "load_global_model", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(service_module, "run_online_inference", _direct_forward)
    runtime = service_module.GovernanceServingRuntime(
        root, global_model_root=tmp_path / "v1", device="cpu"
    )
    return runtime, loaded


def _request(loaded: LoadedGlobalModel) -> dict[str, Any]:
    return {
        "schemaVersion": "socialgraph-fm.gfm-governance/2.0",
        "protocol": "global",
        "artifactId": TINY_ARTIFACT_ID,
        "datasetContentHash": TINY_DATASET_HASH,
        "graphVersionHash": TINY_GRAPH_HASH,
        "modelVersionId": loaded.model_version_id,
        "modelStateHash": loaded.model_state_hash,
        "topK": 6,
    }


def _wait(runtime, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = runtime.dispatch_get(f"/internal/governance/runs/{run_id}")
        if state["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            return state
        time.sleep(0.01)
    raise AssertionError("online run did not reach a terminal state")


def _rewrite_hashed_json(path: Path, document: dict[str, Any], hash_key: str) -> None:
    logical = {key: value for key, value in document.items() if key != hash_key}
    document[hash_key] = canonical_sha256(logical)
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_run_request_rejects_a_different_model_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, loaded = _runtime(tmp_path, monkeypatch)
    try:
        request = _request(loaded)
        request["modelStateHash"] = "0" * 64
        with pytest.raises(service_module.GovernanceInvalid):
            runtime.create_run(request)
        assert not list((tmp_path / "v2" / "runs").glob("governance-*"))
    finally:
        runtime.close()


def test_runtime_calls_model_forward_and_freezes_all_node_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, loaded = _runtime(tmp_path, monkeypatch)
    original = loaded.model.forward
    calls = 0

    def counted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    loaded.model.forward = counted_forward
    try:
        status = runtime.create_run(_request(loaded))
        terminal = _wait(runtime, status["runId"])
        assert terminal["status"] == "succeeded"
        assert calls == 1
        result = runtime.result(status["runId"])
        assert result["datasetMetrics"] is None
        assert result["calibration"]["applicability"] == "out_of_domain_unverified"
        assert result["totalFindings"] == 6
        assert len(result["findings"]) == 6
        output_path = runtime._run_dir(status["runId"]) / "outputs.npz"
        with np.load(output_path, allow_pickle=False) as archive:
            assert archive["embeddings"].shape == (6, 256)
            assert archive["embeddings"].dtype == np.float16
        manifest = runtime._run_manifest(status["runId"])
        assert manifest["fanout"] == [20, 10]
        assert manifest["runtimeRecipeHash"] == loaded.runtime_recipe_hash
        assert manifest["modelStateHash"] == loaded.model_state_hash
        assert manifest["datasetContentHash"] == TINY_DATASET_HASH
        assert runtime.derivations(status["runId"], "groups", offset=0, limit=10)["total"]
        assert runtime.run_preview(status["runId"])["resultHash"] == result["resultHash"]
        evidence = runtime.evidence(status["runId"], result["findings"][0]["nodeId"])
        assert evidence["evidenceSubgraph"]["nodeCount"] <= 300
        assert evidence["evidenceSubgraph"]["edgeCount"] <= 1000
    finally:
        runtime.close()


def test_retry_creates_a_new_run_and_preserves_failed_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, loaded = _runtime(tmp_path, monkeypatch)

    def fail(*_args, **_kwargs):
        raise RuntimeError("expected test failure")

    monkeypatch.setattr(service_module, "run_online_inference", fail)
    try:
        failed = runtime.create_run(_request(loaded))
        assert _wait(runtime, failed["runId"])["status"] == "failed"
        monkeypatch.setattr(service_module, "run_online_inference", _direct_forward)
        retried = runtime.retry_run(failed["runId"])
        assert retried["runId"] != failed["runId"]
        assert _wait(runtime, retried["runId"])["status"] == "succeeded"
        assert runtime._read_state(failed["runId"])["status"] == "failed"
    finally:
        runtime.close()


def test_retry_rejects_a_rehashed_request_that_no_longer_matches_the_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, loaded = _runtime(tmp_path, monkeypatch)

    def fail(*_args, **_kwargs):
        raise RuntimeError("expected test failure")

    monkeypatch.setattr(service_module, "run_online_inference", fail)
    try:
        failed = runtime.create_run(_request(loaded))
        assert _wait(runtime, failed["runId"])["status"] == "failed"
        request_path = runtime._run_dir(failed["runId"]) / "request.json"
        document = json.loads(request_path.read_text(encoding="utf-8"))
        document["request"]["topK"] = 5
        document["requestHash"] = canonical_sha256(document["request"])
        _rewrite_hashed_json(request_path, document, "requestDocumentHash")

        with pytest.raises(ValueError, match="request identity mismatch"):
            runtime.retry_run(failed["runId"])
    finally:
        runtime.close()


def test_execution_fails_closed_when_request_identity_changes_before_result_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, loaded = _runtime(tmp_path, monkeypatch)

    def tamper_then_forward(data, model, *, progress, cancelled):
        request_path = next(runtime.run_root.glob("governance-*/request.json"))
        document = json.loads(request_path.read_text(encoding="utf-8"))
        document["request"]["modelStateHash"] = "0" * 64
        document["requestHash"] = canonical_sha256(document["request"])
        _rewrite_hashed_json(request_path, document, "requestDocumentHash")
        return _direct_forward(data, model, progress=progress, cancelled=cancelled)

    monkeypatch.setattr(service_module, "run_online_inference", tamper_then_forward)
    try:
        status = runtime.create_run(_request(loaded))
        terminal = _wait(runtime, status["runId"])
        assert terminal["status"] == "failed"
        assert terminal["errorCode"] == "GFM_GOVERNANCE_EXECUTION_FAILED"
    finally:
        runtime.close()


def test_analytics_is_deterministic_for_the_same_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, loaded = _runtime(tmp_path, monkeypatch)
    try:
        data = runtime._artifact(TINY_ARTIFACT_ID)
        outputs = _direct_forward(
            data, loaded, progress=lambda _value: None, cancelled=lambda: False
        )
        repeated = _direct_forward(
            data, loaded, progress=lambda _value: None, cancelled=lambda: False
        )
        assert np.array_equal(outputs.logits, repeated.logits)
        assert np.array_equal(outputs.embeddings, repeated.embeddings)
        assert np.array_equal(outputs.router_indices, repeated.router_indices)
        assert np.array_equal(outputs.router_weights, repeated.router_weights)
        first = derive_analytics(data, outputs, seed=123)
        second = derive_analytics(data, outputs, seed=123)
        assert first.groups == second.groups
        assert first.links == second.links
        assert all(
            np.array_equal(first.relation_arrays[name], second.relation_arrays[name])
            for name in first.relation_arrays
        )
    finally:
        runtime.close()


def test_cancel_and_restart_interruption_are_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, loaded = _runtime(tmp_path, monkeypatch)

    def wait_for_cancel(_data, _loaded, *, progress, cancelled):
        progress(0.1)
        for _ in range(200):
            if cancelled():
                raise InferenceCancelled
            time.sleep(0.005)
        raise AssertionError("cancel was not observed")

    monkeypatch.setattr(service_module, "run_online_inference", wait_for_cancel)
    try:
        status = runtime.create_run(_request(loaded))
        for _ in range(100):
            current = runtime._read_state(status["runId"])
            if current["status"] == "running":
                break
            time.sleep(0.005)
        runtime.cancel_run(status["runId"])
        cancelled = _wait(runtime, status["runId"])
        assert cancelled["status"] == "cancelled"
        assert cancelled["cancelRequested"] is True

        interrupted_id = "governance-" + "f" * 32
        directory = runtime._run_dir(interrupted_id)
        directory.mkdir()
        now = "2026-08-18T00:00:00.000000Z"
        runtime._write_state(
            interrupted_id,
            {
                "schemaVersion": "socialgraph-fm.gfm-governance/2.0",
                "runId": interrupted_id,
                "requestHash": "4" * 64,
                "artifactId": TINY_ARTIFACT_ID,
                "datasetContentHash": TINY_DATASET_HASH,
                "graphVersionHash": TINY_GRAPH_HASH,
                "modelVersionId": loaded.model_version_id,
                "modelVersionHash": loaded.model_version_hash,
                "modelStateHash": loaded.model_state_hash,
                "status": "running",
                "stage": "inferencing",
                "progress": 50,
                "createdAt": now,
                "updatedAt": now,
                "errorCode": None,
                "cancelRequested": False,
            },
        )
    finally:
        runtime.close()
    recovered = service_module.GovernanceServingRuntime(
        tmp_path / "v2", global_model_root=tmp_path / "v1", device="cpu"
    )
    try:
        state = recovered._read_state(interrupted_id)
        assert state["status"] == "interrupted"
        assert state["errorCode"] == "GFM_GOVERNANCE_INTERRUPTED"
    finally:
        recovered.close()


def test_result_rejects_post_success_output_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, loaded = _runtime(tmp_path, monkeypatch)
    try:
        status = runtime.create_run(_request(loaded))
        assert _wait(runtime, status["runId"])["status"] == "succeeded"
        runtime.result(status["runId"])
        output_path = runtime._run_dir(status["runId"]) / "outputs.npz"
        payload = bytearray(output_path.read_bytes())
        payload[-1] ^= 1
        output_path.write_bytes(payload)
        with pytest.raises(ValueError, match="NPZ identity"):
            runtime.result(status["runId"])
    finally:
        runtime.close()


def test_result_rejects_a_rehashed_manifest_with_a_different_model_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, loaded = _runtime(tmp_path, monkeypatch)
    try:
        status = runtime.create_run(_request(loaded))
        assert _wait(runtime, status["runId"])["status"] == "succeeded"
        manifest_path = runtime._run_dir(status["runId"]) / "run-artifacts.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["modelStateHash"] = "0" * 64
        _rewrite_hashed_json(manifest_path, manifest, "runArtifactHash")
        runtime._verified_runs.discard(status["runId"])

        with pytest.raises(ValueError, match="run identity mismatch"):
            runtime.result(status["runId"])
    finally:
        runtime.close()


def test_result_rejects_a_rehashed_result_with_a_different_request_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, loaded = _runtime(tmp_path, monkeypatch)
    try:
        status = runtime.create_run(_request(loaded))
        assert _wait(runtime, status["runId"])["status"] == "succeeded"
        result_path = runtime._run_dir(status["runId"]) / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["requestHash"] = "0" * 64
        _rewrite_hashed_json(result_path, result, "resultHash")

        with pytest.raises(ValueError, match="run identity mismatch"):
            runtime.result(status["runId"])
    finally:
        runtime.close()


def test_run_listing_skips_legacy_states_without_model_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, loaded = _runtime(tmp_path, monkeypatch)
    try:
        status = runtime.create_run(_request(loaded))
        assert _wait(runtime, status["runId"])["status"] == "succeeded"
        state_path = runtime._run_dir(status["runId"]) / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.pop("modelStateHash")
        _rewrite_hashed_json(state_path, state, "statusHash")

        listed = runtime.list_runs(offset=0, limit=100)

        assert all(item["runId"] != status["runId"] for item in listed["items"])
        with pytest.raises(ValueError, match="status identity mismatch"):
            runtime._read_state(status["runId"])
    finally:
        runtime.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing retry contract")
def test_state_reads_retry_transient_windows_sharing_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, loaded = _runtime(tmp_path, monkeypatch)
    try:
        status = runtime.create_run(_request(loaded))
        assert _wait(runtime, status["runId"])["status"] == "succeeded"
        state_path = runtime._run_dir(status["runId"]) / "state.json"
        original_read_text = Path.read_text
        attempts = 0

        def flaky_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
            nonlocal attempts
            if path == state_path and attempts < 2:
                attempts += 1
                error = PermissionError(13, "transient sharing violation", str(path))
                error.winerror = 5  # type: ignore[attr-defined]
                raise error
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", flaky_read_text)

        assert runtime._read_state(status["runId"])["runId"] == status["runId"]
        assert attempts == 2
    finally:
        runtime.close()


def test_relation_change_updates_hashes_and_model_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, loaded = _runtime(tmp_path, monkeypatch)
    changed_root = tmp_path / "changed-v2"
    changed_bundle = tmp_path / "changed.zip"
    source_bundle = tmp_path / "tiny.zip"
    with zipfile.ZipFile(source_bundle) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    relations = entries["relations.csv"].decode().replace(
        "synthetic:0,synthetic:1,coRT", "synthetic:0,synthetic:2,coRT", 1
    ).encode()
    manifest = json.loads(entries["manifest.json"])
    manifest["files"]["relations.csv"] = {
        "sha256": hashlib.sha256(relations).hexdigest(),
        "bytes": len(relations),
    }
    entries["relations.csv"] = relations
    entries["manifest.json"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    with zipfile.ZipFile(changed_bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    file_digests = {
        name: hashlib.sha256(entries[name]).hexdigest()
        for name in ("features.npz", "nodes.csv", "relations.csv")
    }
    dataset_hash = _dataset_content_hash(
        manifest_hash=hashlib.sha256(entries["manifest.json"]).hexdigest(),
        file_digests=file_digests,
        clean=False,
        removed=0,
    )
    graph_hash = _graph_version_hash(
        tuple(f"synthetic:{index}" for index in range(6)),
        ((0, 2), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)),
    )
    artifact_id = f"governance-artifact-{dataset_hash[:32]}"
    _install_bundle(changed_root, changed_bundle, artifact_id)
    changed_artifact = materialize_bundle(
        changed_root,
        artifact_id,
        expected_dataset_content_hash=dataset_hash,
        expected_graph_version_hash=graph_hash,
        clean_self_loops=False,
    )
    try:
        baseline = runtime._artifact(TINY_ARTIFACT_ID)
        changed = load_materialized_artifact(changed_artifact.root)
        first = _direct_forward(
            baseline, loaded, progress=lambda _value: None, cancelled=lambda: False
        )
        second = _direct_forward(
            changed, loaded, progress=lambda _value: None, cancelled=lambda: False
        )
        assert dataset_hash != TINY_DATASET_HASH
        assert graph_hash != TINY_GRAPH_HASH
        assert not np.array_equal(first.logits, second.logits)
        assert not np.array_equal(first.embeddings, second.embeddings)
    finally:
        runtime.close()
