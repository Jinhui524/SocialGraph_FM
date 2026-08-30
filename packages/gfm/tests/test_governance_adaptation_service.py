from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from socialgraph_gfm.governance.service import GovernanceServingRuntime
from socialgraph_gfm.governance.errors import GovernanceInvalid, GovernanceServiceError
from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.governance.adaptation import AdaptationBinding


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _imported_sidecar_request() -> tuple[dict[str, object], AdaptationBinding, SimpleNamespace]:
    node_ids = tuple(f"node-{index:03d}" for index in range(128))
    degrees = np.arange(1, 129, dtype=np.int64)
    indptr = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(degrees)))
    binding = AdaptationBinding.model_validate(
        {
            "artifactId": "governance-artifact-" + "c" * 32,
            "datasetContentHash": _hash("sidecar-dataset"),
            "graphVersionHash": _hash("sidecar-graph"),
            "runId": "governance-" + "d" * 32,
            "requestHash": _hash("sidecar-request"),
            "resultHash": _hash("sidecar-result"),
            "runArtifactHash": _hash("sidecar-run-artifact"),
            "modelVersionId": "socialgraph-fm-global/test",
            "modelVersionHash": _hash("sidecar-model"),
            "modelStateHash": _hash("sidecar-state"),
            "recipeHash": _hash("sidecar-recipe"),
            "codeHash": _hash("sidecar-code"),
            "seed": 1729,
        }
    )
    rows: list[dict[str, object]] = []
    for label, offset in (("io", 0), ("control", 2)):
        for stratum in range(4):
            for within in range(2):
                index = stratum * 32 + offset + within
                rows.append(
                    {
                        "nodeId": node_ids[index],
                        "label": label,
                        "structuralStratum": stratum,
                        "fusedDegree": int(degrees[index]),
                    }
                )
    rows.sort(key=lambda item: str(item["nodeId"]))
    label_selection = {
        "version": "graph-fused-degree-quartile-stable-hash-v2",
        "stratification": "graph-fused-degree-rank-quartile",
        "structuralStrata": 4,
        "labelsPerClass": 8,
        "labelsPerClassPerStratum": 2,
        "scoreInputs": [],
    }
    label_document = {
        "schemaVersion": "socialgraph-fm.governance-target-label-recipe/1.1",
        "datasetId": "thailand-authorized",
        "bundleSha256": _hash("sidecar-bundle"),
        "selectionRecipe": label_selection,
        "labels": rows,
    }
    labels_bytes = json.dumps(
        label_document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    receipt_logical: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-target-package-receipt/1.1",
        "datasetId": "thailand-authorized",
        "sourceSchemaVersion": "socialgraph-fm.anonymized-posts/1.0",
        "sourceSha256": _hash("sidecar-source"),
        "authorizationReference": "fixture-approval-2026-08-20",
        "bundleSha256": _hash("sidecar-bundle"),
        "labelsSha256": hashlib.sha256(labels_bytes).hexdigest(),
        "encoder": {
            "modelId": "fixture/deterministic-encoder",
            "revision": "fixture-v1",
            "cacheSha256": "1" * 64,
            "compatibility": "dimension-only-unverified",
            "dimension": 768,
        },
        "selectionRecipe": {
            "version": "connected-structural-hash-v2",
            "nodeCount": 128,
            "requiredIo": 16,
            "requiredControls": 64,
            "minimumNonemptyModalities": 4,
            "scoreInputs": [],
            "groupRelations": {"maxGroupAccounts": 256, "totalPotentialPairBudget": 50_000},
            "fastRT": {"windowSeconds": 10, "pairBudget": 50_000, "algorithm": "sorted-sliding-window-v1"},
            "tweetSim": {"mutualTopK": 5, "cosineThreshold": 0.8, "pairBudget": 10_000},
        },
        "labelSelectionRecipe": label_selection,
        "coverage": {
            "nodeCount": 128,
            "ioCount": 32,
            "controlCount": 96,
            "nonemptyModalities": ["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"],
            "connected": True,
        },
    }
    receipt = {**receipt_logical, "receiptHash": canonical_sha256(receipt_logical)}
    labels = []
    for row in rows:
        source_hash = canonical_sha256(
            {
                "schemaVersion": label_document["schemaVersion"],
                "datasetId": label_document["datasetId"],
                "bundleSha256": label_document["bundleSha256"],
                "labelsSha256": receipt["labelsSha256"],
                "receiptHash": receipt["receiptHash"],
                **row,
            }
        )
        labels.append(
            {
                "nodeId": row["nodeId"],
                "label": "positive" if row["label"] == "io" else "negative",
                "sourceType": "imported_sidecar",
                "sourceRecordId": f"thailand-authorized:{row['nodeId']}",
                "sourceRecordHash": source_hash,
                "reviewEventHash": None,
                "structuralStratum": row["structuralStratum"],
                "fusedDegree": row["fusedDegree"],
                "labelsSha256": receipt["labelsSha256"],
                "receiptHash": receipt["receiptHash"],
            }
        )
    request: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-target-label-set/1.1",
        "runId": binding.run_id,
        "resultHash": binding.result_hash,
        "sidecarReceipt": receipt,
        "labels": labels,
    }
    artifact = SimpleNamespace(
        node_ids=node_ids,
        arrays={"fused_indptr": indptr},
        artifact=SimpleNamespace(
            document={
                "artifactId": binding.artifact_id,
                "datasetContentHash": binding.dataset_content_hash,
                "graphVersionHash": binding.graph_version_hash,
                "bundleSha256": receipt["bundleSha256"],
                "nodeCount": 128,
            }
        ),
    )
    return request, binding, artifact


def test_imported_sidecar_is_revalidated_against_receipt_bundle_and_current_fused_graph(
    tmp_path: Path,
) -> None:
    request, binding, artifact = _imported_sidecar_request()
    runtime = object.__new__(GovernanceServingRuntime)
    runtime.root = tmp_path
    runtime._lock = threading.RLock()
    runtime._adaptation_binding = MethodType(lambda _self, _run_id: binding, runtime)
    runtime._artifact = MethodType(lambda _self, _artifact_id: artifact, runtime)

    created = runtime.create_adaptation_label_set(request)
    assert created["schemaVersion"] == "socialgraph-fm.governance-target-label-set/1.1"
    assert created["sidecarReceipt"]["receiptHash"] == request["sidecarReceipt"]["receiptHash"]

    artifact.arrays["fused_indptr"] = np.asarray(artifact.arrays["fused_indptr"]).copy()
    artifact.arrays["fused_indptr"][1:] += 1
    with pytest.raises(ValueError, match="fused degree|structural stratum"):
        runtime._label_set(created["labelSetHash"])


@pytest.mark.parametrize("tamper", ["receipt", "labels", "degree", "stratum", "bundle", "graph"])
def test_imported_sidecar_tampering_fails_before_label_set_persistence(
    tmp_path: Path, tamper: str
) -> None:
    request, binding, artifact = _imported_sidecar_request()
    request = copy.deepcopy(request)
    if tamper == "receipt":
        request["sidecarReceipt"]["receiptHash"] = "f" * 64
    elif tamper == "labels":
        request["sidecarReceipt"]["labelsSha256"] = "e" * 64
    elif tamper == "degree":
        request["labels"][0]["fusedDegree"] += 1
    elif tamper == "stratum":
        request["labels"][0]["structuralStratum"] = 3
    elif tamper == "bundle":
        artifact.artifact.document["bundleSha256"] = "d" * 64
    else:
        artifact.artifact.document["graphVersionHash"] = "c" * 64
    runtime = object.__new__(GovernanceServingRuntime)
    runtime.root = tmp_path
    runtime._lock = threading.RLock()
    runtime._adaptation_binding = MethodType(lambda _self, _run_id: binding, runtime)
    runtime._artifact = MethodType(lambda _self, _artifact_id: artifact, runtime)

    with pytest.raises((GovernanceInvalid, ValueError)):
        runtime.create_adaptation_label_set(request)
    assert not (tmp_path / "adaptations" / "label-sets").exists()


def test_serving_runtime_owns_fitting_and_returns_only_bounded_metadata(
    tmp_path: Path,
) -> None:
    # Catches an API-side fit, missing internal routes, mutable base artifacts, and vector leaks.
    runtime = object.__new__(GovernanceServingRuntime)
    runtime.root = tmp_path
    runtime.run_root = tmp_path / "runs"
    runtime.run_root.mkdir()
    runtime._lock = threading.RLock()
    run_id = "governance-" + "b" * 32
    run_dir = runtime.run_root / run_id
    run_dir.mkdir()
    node_ids = [f"node-{index}" for index in range(12)]
    logits = np.asarray([0.1, -0.2, 0.0, 0.2, 0.3, -0.1, 0.05, -0.3, 0.7, -0.6, 0.4, -0.4])
    embeddings = np.zeros((12, 256), dtype=np.float32)
    for index in range(12):
        embeddings[index, index % 4] = 1.0 if index < 4 else -1.0
        embeddings[index, 8 + index] = 0.05
    scores = np.asarray(
        [0.51, 0.93, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61],
        dtype=np.float32,
    )
    ranks = np.asarray([12, 1, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2], dtype=np.int32)
    checkpoint_path = tmp_path / "model" / "checkpoint.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"immutable checkpoint fixture")
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    result: dict[str, object] = {
        "runId": run_id,
        "requestHash": _hash("request"),
        "resultHash": _hash("result"),
        "artifactId": "governance-artifact-" + "a" * 32,
        "datasetContentHash": _hash("dataset"),
        "graphVersionHash": _hash("graph"),
        "modelVersionId": "socialgraph-fm-global/test",
        "modelVersionHash": _hash("model"),
        "modelStateHash": checkpoint_hash,
    }
    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    output_path = run_dir / "outputs.npz"
    np.savez_compressed(
        output_path,
        logits=logits,
        scores=scores,
        ranks=ranks,
        embeddings=embeddings,
    )
    manifest: dict[str, object] = {
        "runtimeRecipeHash": _hash("recipe"),
        "inferenceSeed": 1729,
        "outputsSha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    manifest["runArtifactHash"] = _hash(json.dumps(manifest, sort_keys=True))
    manifest_path = run_dir / "run-artifacts.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    review_path = tmp_path / "reviews.sqlite3"
    review_hashes = [_hash(f"review-{index}") for index in range(8)]
    with sqlite3.connect(review_path) as connection:
        connection.execute(
            "CREATE TABLE review_events (node_id TEXT, decision TEXT, event_hash TEXT)"
        )
        connection.executemany(
            "INSERT INTO review_events VALUES (?, ?, ?)",
            [
                (
                    f"node-{index}",
                    "confirmed" if index < 4 else "rejected",
                    review_hashes[index],
                )
                for index in range(8)
            ],
        )
    immutable_paths = (
        output_path,
        result_path,
        manifest_path,
        checkpoint_path,
        review_path,
    )
    before = {path: (path.read_bytes(), hashlib.sha256(path.read_bytes()).hexdigest()) for path in immutable_paths}

    runtime.result = MethodType(
        lambda _self, _run_id: json.loads(result_path.read_text(encoding="utf-8")),
        runtime,
    )
    runtime._run_manifest = MethodType(
        lambda _self, _run_id: json.loads(manifest_path.read_text(encoding="utf-8")),
        runtime,
    )

    def load_outputs(_self, _run_id):  # type: ignore[no-untyped-def]
        with np.load(output_path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}

    runtime._outputs = MethodType(load_outputs, runtime)
    runtime._artifact = MethodType(
        lambda _self, _artifact_id: SimpleNamespace(node_ids=tuple(node_ids)), runtime
    )
    request = {
        "schemaVersion": "socialgraph-fm.governance-target-label-set/1.0",
        "runId": run_id,
        "resultHash": result["resultHash"],
        "labels": [
            {
                "nodeId": f"node-{index}",
                "label": "positive" if index < 4 else "negative",
                "sourceType": "concluded_review",
                "sourceRecordId": f"event-{index:032x}",
                "sourceRecordHash": review_hashes[index],
                "reviewEventHash": review_hashes[index],
            }
            for index in range(8)
        ],
    }

    created = runtime.dispatch_post(
        "/internal/governance/adaptations/label-sets", request
    )
    label_set_root = tmp_path / "adaptations" / "label-sets"
    persisted_before_oversize = tuple(label_set_root.iterdir())
    oversized = copy.deepcopy(request)
    oversized["labels"] = [
        {
            "nodeId": f"oversized-{index}",
            "label": "positive" if index < 128 else "negative",
            "sourceType": "imported_sidecar",
            "sourceRecordId": f"oversized-{index}",
            "sourceRecordHash": _hash(f"oversized-{index}"),
            "reviewEventHash": None,
        }
        for index in range(257)
    ]
    with pytest.raises(GovernanceInvalid):
        runtime.dispatch_post(
            "/internal/governance/adaptations/label-sets", oversized
        )
    assert tuple(label_set_root.iterdir()) == persisted_before_oversize
    fitted = runtime.dispatch_post(
        f"/internal/governance/adaptations/label-sets/{created['labelSetHash']}/policies"
    )
    read = runtime.dispatch_get(
        f"/internal/governance/adaptations/policies/{fitted['policyHash']}"
    )
    page = runtime.dispatch_get(
        f"/internal/governance/adaptations/runs/{run_id}/policies/"
        f"{fitted['policyHash']}/comparison?offset=0&limit=5"
    )

    assert read == fitted
    assert len(page["rows"]) == 5
    assert page["total"] == len(node_ids)
    by_node = {row["nodeId"]: row for row in page["rows"]}
    assert by_node["node-1"]["baseScore"] == float(scores[1])
    assert by_node["node-1"]["baseRank"] == int(ranks[1])
    for public_payload in (created, fitted, page):
        encoded = json.dumps(public_payload).lower()
        assert '"embeddings"' not in encoded
        assert '"positivecentroid":' not in encoded
        assert '"negativecentroid":' not in encoded
    for path, identity in before.items():
        assert (path.read_bytes(), hashlib.sha256(path.read_bytes()).hexdigest()) == identity
    assert result["modelStateHash"] == checkpoint_hash


def test_v2_adaptation_contracts_are_exposed_by_the_existing_internal_routes(
    tmp_path: Path,
) -> None:
    """Catches Task 2's v2 fitter remaining unreachable across loopback HTTP."""
    runtime = object.__new__(GovernanceServingRuntime)
    runtime.root = tmp_path
    runtime._lock = threading.RLock()
    node_ids = tuple(f"target-{index}" for index in range(10))
    binding = AdaptationBinding.model_validate(
        {
            "artifactId": "governance-artifact-" + "a" * 32,
            "datasetContentHash": _hash("v2-dataset"),
            "graphVersionHash": _hash("v2-graph"),
            "runId": "governance-" + "b" * 32,
            "requestHash": _hash("v2-request"),
            "resultHash": _hash("v2-result"),
            "runArtifactHash": _hash("v2-run-artifact"),
            "modelVersionId": "socialgraph-fm-global/test",
            "modelVersionHash": _hash("v2-model"),
            "modelStateHash": _hash("v2-model-state"),
            "recipeHash": _hash("v2-recipe"),
            "codeHash": _hash("v2-code"),
            "seed": 1729,
        }
    )
    runtime._adaptation_binding = MethodType(
        lambda _self, _run_id: binding, runtime
    )
    runtime._artifact = MethodType(
        lambda _self, _artifact_id: SimpleNamespace(
            node_ids=node_ids,
            artifact=SimpleNamespace(
                document={"bundleSha256": _hash("v2-inference")}
            ),
        ),
        runtime,
    )
    logits = np.linspace(-0.4, 0.5, len(node_ids), dtype=np.float32)
    embeddings = np.eye(len(node_ids), 256, dtype=np.float32)
    scores = np.linspace(0.1, 0.9, len(node_ids), dtype=np.float32)
    ranks = np.arange(1, len(node_ids) + 1, dtype=np.int32)
    runtime._outputs = MethodType(
        lambda _self, _run_id: {
            "logits": logits,
            "embeddings": embeddings,
            "scores": scores,
            "ranks": ranks,
        },
        runtime,
    )
    labels = [
        {
            "nodeId": node_ids[index],
            "label": "positive" if index < 4 else "negative",
            "structuralStratum": index % 4,
            "fusedDegree": index + 1,
        }
        for index in range(8)
    ]
    request = {
        "schemaVersion": "socialgraph-fm.governance-target-label-set/2.0",
        "taskId": "generic-target",
        "inferenceSha256": _hash("v2-inference"),
        "runId": binding.run_id,
        "resultHash": binding.result_hash,
        "labels": labels,
    }

    created = runtime.dispatch_post(
        "/internal/governance/adaptations/label-sets", request
    )
    fitted = runtime.dispatch_post(
        f"/internal/governance/adaptations/label-sets/{created['labelSetHash']}/policies"
    )
    read = runtime.dispatch_get(
        f"/internal/governance/adaptations/policies/{fitted['policyHash']}"
    )
    page = runtime.dispatch_get(
        f"/internal/governance/adaptations/runs/{binding.run_id}/policies/"
        f"{fitted['policyHash']}/comparison"
    )

    assert created["schemaVersion"] == "socialgraph-fm.governance-target-label-set/2.0"
    assert fitted["schemaVersion"] == "socialgraph-fm.governance-target-review-policy/2.0"
    assert read == fitted
    assert page["schemaVersion"] == "socialgraph-fm.governance-adaptation-comparison/2.0"
    assert page["baseOutputsImmutable"] is True
    assert len(page["rows"]) == len(node_ids)
    assert "embedding" not in json.dumps((created, fitted, page)).lower()

    stale_binding = binding.model_copy(update={"model_state_hash": _hash("stale-state")})
    runtime._adaptation_binding = MethodType(
        lambda _self, _run_id: stale_binding, runtime
    )
    with pytest.raises(GovernanceServiceError) as stale:
        runtime.dispatch_get(
            f"/internal/governance/adaptations/policies/{fitted['policyHash']}"
        )
    assert stale.value.status == 409


def test_v2_same_label_content_binds_two_runs_and_exact_repeats_are_idempotent(
    tmp_path: Path,
) -> None:
    runtime = object.__new__(GovernanceServingRuntime)
    runtime.root = tmp_path
    runtime._lock = threading.RLock()
    node_ids = tuple(f"target-{index}" for index in range(10))
    registration_id = "target-task-" + "7" * 32
    inference_hash = _hash("shared-few-shot-inference")

    def binding_for(token: str) -> AdaptationBinding:
        return AdaptationBinding.model_validate(
            {
                "artifactId": "governance-artifact-" + "a" * 32,
                "datasetContentHash": _hash("shared-dataset"),
                "graphVersionHash": _hash("shared-graph"),
                "runId": "governance-" + token * 32,
                "requestHash": _hash(f"request-{token}"),
                "resultHash": _hash(f"result-{token}"),
                "runArtifactHash": _hash(f"artifact-{token}"),
                "modelVersionId": "socialgraph-fm-global/test",
                "modelVersionHash": _hash("model"),
                "modelStateHash": _hash("model-state"),
                "recipeHash": _hash("recipe"),
                "codeHash": _hash("code"),
                "seed": 1729,
            }
        )

    bindings = {binding.run_id: binding for binding in (binding_for("1"), binding_for("2"))}
    runtime._adaptation_binding = MethodType(
        lambda _self, run_id: bindings[run_id], runtime
    )
    runtime._artifact = MethodType(
        lambda _self, _artifact_id: SimpleNamespace(
            node_ids=node_ids,
            artifact=SimpleNamespace(document={"bundleSha256": inference_hash}),
        ),
        runtime,
    )
    logits = np.linspace(-0.4, 0.5, len(node_ids), dtype=np.float32)
    embeddings = np.eye(len(node_ids), 256, dtype=np.float32)
    scores = np.linspace(0.1, 0.9, len(node_ids), dtype=np.float32)
    ranks = np.arange(1, len(node_ids) + 1, dtype=np.int32)
    runtime._outputs = MethodType(
        lambda _self, _run_id: {
            "logits": logits,
            "embeddings": embeddings,
            "scores": scores,
            "ranks": ranks,
        },
        runtime,
    )
    labels = [
        {
            "nodeId": node_ids[index],
            "label": "positive" if index < 4 else "negative",
            "structuralStratum": index % 4,
            "fusedDegree": index + 1,
        }
        for index in range(8)
    ]

    created: list[dict[str, object]] = []
    policies: list[dict[str, object]] = []
    for binding in bindings.values():
        create_request = {
            "schemaVersion": "socialgraph-fm.governance-target-label-set/2.0",
            "taskId": "generic-target",
            "inferenceSha256": inference_hash,
            "targetTaskRegistrationId": registration_id,
            "runId": binding.run_id,
            "resultHash": binding.result_hash,
            "labels": labels,
        }
        first = runtime.create_adaptation_label_set(create_request)
        assert runtime.create_adaptation_label_set(create_request) == first
        created.append(first)
        fit_request = {
            "schemaVersion": "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
            "targetTaskRegistrationId": registration_id,
            "runId": binding.run_id,
            "resultHash": binding.result_hash,
        }
        policy = runtime.fit_adaptation_policy(first["labelSetHash"], fit_request)
        assert runtime.fit_adaptation_policy(first["labelSetHash"], fit_request) == policy
        assert policy["binding"]["runId"] == binding.run_id
        policies.append(policy)

    assert created[0] == created[1]
    assert policies[0]["policyHash"] != policies[1]["policyHash"]
    assert len(list((tmp_path / "adaptations" / "label-set-bindings").iterdir())) == 2
    with pytest.raises(GovernanceInvalid):
        runtime.fit_adaptation_policy(created[0]["labelSetHash"])
    with pytest.raises((GovernanceInvalid, GovernanceServiceError)):
        runtime.fit_adaptation_policy(
            created[0]["labelSetHash"],
            {
                "schemaVersion": "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
                "targetTaskRegistrationId": registration_id,
                "runId": next(iter(bindings)),
                "resultHash": _hash("tampered-result"),
            },
        )
    first_binding = next(iter(bindings.values()))
    with pytest.raises((GovernanceInvalid, GovernanceServiceError)):
        runtime.fit_adaptation_policy(
            created[0]["labelSetHash"],
            {
                "schemaVersion": "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
                "targetTaskRegistrationId": "target-task-" + "8" * 32,
                "runId": first_binding.run_id,
                "resultHash": first_binding.result_hash,
            },
        )
    persisted_binding_path = next(
        path
        for path in (tmp_path / "adaptations" / "label-set-bindings").iterdir()
        if json.loads(path.read_text(encoding="utf-8"))["runId"]
        == first_binding.run_id
    )
    tampered_document = json.loads(persisted_binding_path.read_text(encoding="utf-8"))
    tampered_document["binding"]["resultHash"] = _hash("tampered-persisted-binding")
    persisted_binding_path.write_text(
        json.dumps(tampered_document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(GovernanceServiceError):
        runtime.fit_adaptation_policy(
            created[0]["labelSetHash"],
            {
                "schemaVersion": "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
                "targetTaskRegistrationId": registration_id,
                "runId": first_binding.run_id,
                "resultHash": first_binding.result_hash,
            },
        )


def test_v2_internal_comparison_rejects_an_insufficient_signal_policy(
    tmp_path: Path,
) -> None:
    """A zero-lambda policy is inspectable, but it must never publish a comparison."""
    runtime = object.__new__(GovernanceServingRuntime)
    runtime.root = tmp_path
    runtime._lock = threading.RLock()
    binding = AdaptationBinding.model_validate(
        {
            "artifactId": "governance-artifact-" + "a" * 32,
            "datasetContentHash": _hash("insufficient-dataset"),
            "graphVersionHash": _hash("insufficient-graph"),
            "runId": "governance-" + "b" * 32,
            "requestHash": _hash("insufficient-request"),
            "resultHash": _hash("insufficient-result"),
            "runArtifactHash": _hash("insufficient-run-artifact"),
            "modelVersionId": "socialgraph-fm-global/test",
            "modelVersionHash": _hash("insufficient-model"),
            "modelStateHash": _hash("insufficient-model-state"),
            "recipeHash": _hash("insufficient-recipe"),
            "codeHash": _hash("insufficient-code"),
            "seed": 1729,
        }
    )
    runtime._adaptation_binding = MethodType(
        lambda _self, _run_id: binding, runtime
    )
    policy_payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-target-review-policy/2.0",
        "binding": binding.model_dump(mode="json", by_alias=True),
        "labelSetHash": _hash("insufficient-label-set"),
        "status": "insufficient_signal",
        "selectedLambda": 0.0,
        "eligibleLabelCount": 8,
        "positiveCount": 4,
        "negativeCount": 4,
        "fittingRecipe": "l2-centroids+run-zscore+loo-balanced-log-loss-v1",
        "baseOutputsImmutable": True,
        "adaptedOutputFields": ["adaptedReviewPriority", "adaptedRank"],
    }
    policy_payload["policyHash"] = canonical_sha256(policy_payload)
    comparison_payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-adaptation-comparison/2.0",
        "binding": binding.model_dump(mode="json", by_alias=True),
        "policyHash": policy_payload["policyHash"],
        "total": 1,
        "baseOutputsImmutable": True,
        "rows": [
            {
                "nodeId": "target-0",
                "baseScore": 0.5,
                "baseRank": 1,
                "adaptedReviewPriority": 0.5,
                "adaptedRank": 1,
                "rankDelta": 0,
            }
        ],
    }
    comparison_payload["comparisonHash"] = canonical_sha256(comparison_payload)
    runtime._persist_immutable_adaptation(
        runtime._adaptation_path("policies", str(policy_payload["policyHash"])),
        policy_payload,
    )
    runtime._persist_immutable_adaptation(
        runtime._adaptation_path("comparisons", str(policy_payload["policyHash"])),
        comparison_payload,
    )

    assert runtime.adaptation_policy(str(policy_payload["policyHash"]))["status"] == (
        "insufficient_signal"
    )
    for requested_run_id in (binding.run_id, "governance-" + "c" * 32):
        with pytest.raises(GovernanceServiceError) as rejected:
            runtime.dispatch_get(
                f"/internal/governance/adaptations/runs/{requested_run_id}/policies/"
                f"{policy_payload['policyHash']}/comparison"
            )
        assert rejected.value.status == 409
        assert rejected.value.code == "GFM_GOVERNANCE_ADAPTATION_POLICY_NOT_READY"
        assert str(policy_payload["policyHash"]) not in str(rejected.value)
