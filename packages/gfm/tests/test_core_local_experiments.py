from __future__ import annotations

import gzip
import hashlib
import io
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import torch
import torch_geometric
from pydantic import ValidationError
from scipy import sparse

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.core import local_experiments
from socialgraph_gfm.core import formal_preflight
from socialgraph_gfm.core.bundle import CoreGraphBundle, calculate_graph_version_hash
from socialgraph_gfm.core.config import TrainingConfig
from socialgraph_gfm.core.datasets.materialize import materialize_email_from_files
from socialgraph_gfm.core.datasets import parsers, penn94_conversion
from socialgraph_gfm.core.local_experiments import (
    LocalExperimentRun,
    run_local_nonpromotable_experiment,
)

_REAL_LOAD_EMAIL_LOCAL_INPUTS = local_experiments.load_email_local_inputs
_REAL_LOAD_PENN94_LOCAL_INPUTS = local_experiments.load_penn94_local_inputs


def _node_bundle() -> CoreGraphBundle:
    payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [{"id": str(index), "index": index} for index in range(6)],
        "edges": [
            {"sourceId": str(left), "targetId": str(right), "edgeType": "social"}
            for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5))
        ],
        "nodeFeatures": [
            {
                "kind": "numeric",
                "name": "profile",
                "values": [0.0, 1.0, 0.5, 1.5, 2.0, 2.5],
            }
        ],
        "structuralFeatures": None,
        "source": {"sourceName": "local-fixture", "sourceSha256": "a" * 64},
        "splitManifest": {
            "strategy": "official",
            "assignments": [
                {"entityId": "5", "role": "test"},
                {"entityId": "3", "role": "validation"},
                {"entityId": "1", "role": "train"},
                {"entityId": "4", "role": "test"},
                {"entityId": "2", "role": "validation"},
                {"entityId": "0", "role": "train"},
            ],
        },
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _penn_split_inventory(
    bundle: CoreGraphBundle,
) -> local_experiments.LocalSplitInventory:
    return local_experiments.LocalSplitInventory.create(
        dataset_id="penn94",
        manifests=(bundle.split_manifest,) * 5,
        selected_fold_id="fold-0",
    )


def _source_inventory(
    bundle: CoreGraphBundle, *, dataset_id: str = "penn94"
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-local-source-inventory/1.0",
        "datasetId": dataset_id,
        "sourceSha256": bundle.source.source_sha256,
        "scope": "test-fixture",
        "files": (),
    }
    if dataset_id == "penn94":
        payload["conversionManifestHash"] = "b" * 64
        payload["splitInventoryHash"] = _penn_split_inventory(bundle).inventory_hash
    payload["inventoryHash"] = canonical_sha256(payload)
    return payload


def _email_bundle() -> CoreGraphBundle:
    payload = _node_bundle().model_dump(mode="json", by_alias=True)
    edge_ids = [
        "edge:0:1",
        "edge:1:2",
        "edge:2:3",
        "edge:3:4",
        "edge:4:5",
        "edge:0:5",
    ]
    payload["splitManifest"] = {
        "strategy": "official",
        "assignments": [
            *({"entityId": identifier, "role": "train"} for identifier in edge_ids[:2]),
            *({"entityId": identifier, "role": "validation"} for identifier in edge_ids[2:4]),
            *({"entityId": identifier, "role": "test"} for identifier in edge_ids[4:]),
        ],
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _formal_hash(runtime: Path) -> str:
    path = runtime / "experiments-core" / "formal-preflight-v2-current.json"
    if path.is_file():
        return formal_preflight.FormalPreflightEvidence.model_validate_json(
            path.read_bytes()
        ).evidence_hash
    evidence = formal_preflight.run_formal_preflight(runtime, publish_to=path)
    assert evidence.formal_ready is False
    assert evidence.promotable is False
    return evidence.evidence_hash


def _local_fixture_arguments(runtime: Path, output: Path) -> dict[str, object]:
    bundle = _node_bundle()
    return {
        "bundle": bundle,
        "dataset_id": "penn94",
        "phase": "dev",
        "task_kind": "node-binary",
        "targets_by_entity": {str(index): index % 2 for index in range(6)},
        "split_inventory": _penn_split_inventory(bundle),
        "source_inventory": _source_inventory(bundle),
        "runtime_root": runtime,
        "output_path": output,
        "seed": 20260821,
        "optimizer_steps": 1,
        "head_steps": 1,
        "device_name": "cpu",
        "formal_preflight_evidence_hash": _formal_hash(runtime),
        "formal_ready": False,
    }


def _synthetic_penn_local_inputs(_runtime: Path) -> local_experiments.LocalDatasetInputs:
    bundle = _node_bundle()
    return local_experiments.LocalDatasetInputs(
        bundle=bundle,
        targets_by_entity={str(index): index % 2 for index in range(6)},
        split_inventory=_penn_split_inventory(bundle),
        source_inventory=_source_inventory(bundle),
    )


def _synthetic_email_local_inputs(_runtime: Path) -> local_experiments.LocalDatasetInputs:
    bundle = _email_bundle()
    return local_experiments.LocalDatasetInputs(
        bundle=bundle,
        targets_by_entity={
            "edge:0:1": 0,
            "edge:1:2": 1,
            "edge:2:3": 0,
            "edge:3:4": 1,
            "edge:4:5": 0,
            "edge:0:5": 1,
        },
        split_inventory=local_experiments.LocalSplitInventory.create(
            dataset_id="email-eu-core",
            manifests=(bundle.split_manifest,),
            selected_fold_id="fold-0",
        ),
        source_inventory=_source_inventory(bundle, dataset_id="email-eu-core"),
    )


@pytest.fixture(autouse=True)
def _fixed_synthetic_dataset_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit runs small while exercising the production authority boundary."""

    monkeypatch.setattr(
        local_experiments,
        "load_penn94_local_inputs",
        _synthetic_penn_local_inputs,
    )
    monkeypatch.setattr(
        local_experiments,
        "load_email_local_inputs",
        _synthetic_email_local_inputs,
    )


def test_real_email_loader_rederives_a_tiny_materialization_from_held_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    raw = runtime / "raw" / "email-eu-core" / "1.0.0"
    raw.mkdir(parents=True)
    plain = tmp_path / "plain"
    plain.mkdir()
    edges_plain = plain / "email-Eu-core.txt"
    departments_plain = plain / "email-Eu-core-department-labels.txt"
    edges_plain.write_text(
        "".join(f"{left} {right}\n" for left in range(6) for right in range(left + 1, 6)),
        encoding="utf-8",
    )
    departments_plain.write_text(
        "".join(f"{index} {index % 3}\n" for index in range(6)),
        encoding="utf-8",
    )
    edges_raw = raw / "email-Eu-core.txt.gz"
    departments_raw = raw / "email-Eu-core-department-labels.txt.gz"
    with gzip.open(edges_raw, "wb") as stream:
        stream.write(edges_plain.read_bytes())
    with gzip.open(departments_raw, "wb") as stream:
        stream.write(departments_plain.read_bytes())
    raw_hashes = {
        "edges": hashlib.sha256(edges_raw.read_bytes()).hexdigest(),
        "departments": hashlib.sha256(departments_raw.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(local_experiments, "_EMAIL_RAW_SOURCE_SHA256", raw_hashes)
    materialize_email_from_files(
        edges_path=edges_plain,
        departments_path=departments_plain,
        raw_source_paths={"edges": edges_raw, "departments": departments_raw},
        runtime_root=runtime,
        seed=1729,
    )

    loaded = _REAL_LOAD_EMAIL_LOCAL_INPUTS(runtime)

    assert len(loaded.bundle.nodes) == 6
    assert len(loaded.bundle.edges) == 15
    assert len(loaded.targets_by_entity) == 15
    assert loaded.split_inventory.fold_ids == ("fold-0",)
    assert tuple(item["kind"] for item in loaded.source_inventory["files"]) == (
        "raw-edges",
        "raw-labels",
        "materialized-bundle",
        "materialization-manifest",
        "offline-labels",
    )


def test_email_snapshot_decompression_rejects_an_expansion_bomb_before_parsing() -> None:
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb") as stream:
        stream.write(b"x" * 2_000_001)

    with pytest.raises(ValueError, match="expanded-size"):
        local_experiments._bounded_gzip_snapshot(
            compressed.getvalue(),
            max_expanded_bytes=2_000_000,
        )


def test_held_local_source_snapshot_prevents_or_detects_visible_replacement(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    source = runtime / "source.bin"
    source.write_bytes(b"owned source snapshot")
    replacement = runtime / "replacement.bin"
    competitor_bytes = b"competitor replacement"
    replacement.write_bytes(competitor_bytes)
    held = local_experiments._hold_local_source_snapshot(
        runtime,
        source,
        kind="test-source",
        max_bytes=1024,
    )
    replaced = False
    try:
        try:
            os.replace(replacement, source)
            replaced = True
        except OSError:
            held.assert_visible_binding()
        if replaced:
            with pytest.raises(ValueError, match="identity changed"):
                held.assert_visible_binding()
            assert source.read_bytes() == competitor_bytes
    finally:
        held.close()


def test_authoritative_source_lease_spans_training_and_artifact_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "held-authority.json"
    source = runtime / "authority.bin"
    source_bytes = b"fixed authoritative source"
    source.write_bytes(source_bytes)
    replacement = runtime / "authority-replacement.bin"
    competitor_bytes = b"competitor source replacement"
    replacement.write_bytes(competitor_bytes)
    bundle = _node_bundle()
    source_inventory = _source_inventory(bundle)
    source_inventory["files"] = [
        {
            "kind": "test-authority",
            "relativePath": source.relative_to(runtime).as_posix(),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "sizeBytes": len(source_bytes),
        }
    ]
    source_inventory.pop("inventoryHash")
    source_inventory["inventoryHash"] = canonical_sha256(source_inventory)
    authority = local_experiments.LocalDatasetInputs(
        bundle=bundle,
        targets_by_entity={str(index): index % 2 for index in range(6)},
        split_inventory=_penn_split_inventory(bundle),
        source_inventory=source_inventory,
    )
    monkeypatch.setattr(local_experiments, "load_penn94_local_inputs", lambda _root: authority)
    replaced: list[bool] = []

    def replace_authority(stage: str, _target: Path) -> None:
        if stage != "authority-before-artifact-commit":
            return
        try:
            os.replace(replacement, source)
            replaced.append(True)
        except OSError:
            replaced.append(False)

    monkeypatch.setattr(local_experiments, "_LOCAL_PUBLICATION_SEAM", replace_authority)
    arguments = _local_fixture_arguments(runtime, output)
    arguments["source_inventory"] = source_inventory
    if os.name == "nt":
        report = run_local_nonpromotable_experiment(**arguments)
        assert report.promotable is False
        assert replaced == [False]
        assert source.read_bytes() == source_bytes
        assert replacement.read_bytes() == competitor_bytes
    else:
        with pytest.raises(ValueError, match="identity changed"):
            run_local_nonpromotable_experiment(**arguments)
        assert replaced == [True]
        assert source.read_bytes() == competitor_bytes
        assert not output.exists()
        assert not output.with_suffix(".artifacts").exists()


def test_real_penn_loader_parses_one_held_mat_and_safe_split_snapshot_into_all_five_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    raw = runtime / "raw" / "facebook100" / "1.0.0"
    derived = runtime / "derived" / "facebook100" / "penn94-official-splits" / "1.0.0"
    raw.mkdir(parents=True)
    derived.mkdir(parents=True)
    mat_bytes = b"fixed tiny Penn94 MAT snapshot"
    raw_split_bytes = b"fixed tiny raw split snapshot"
    mat_path = raw / "Penn94.mat"
    raw_split_path = raw / "fb100-Penn94-splits.npy"
    safe_split_path = derived / "penn94-official-splits-safe.npz"
    manifest_path = derived / "conversion-manifest.json"
    mat_path.write_bytes(mat_bytes)
    raw_split_path.write_bytes(raw_split_bytes)
    mat_sha256 = hashlib.sha256(mat_bytes).hexdigest()
    raw_split_sha256 = hashlib.sha256(raw_split_bytes).hexdigest()
    role_counts = {"train": 2, "valid": 2, "test": 2}
    monkeypatch.setattr(local_experiments, "PENN94_DATA_SHA256", mat_sha256)
    monkeypatch.setattr(local_experiments, "PENN94_RAW_SPLIT_SHA256", raw_split_sha256)
    monkeypatch.setattr(local_experiments, "PENN94_LABELED_NODE_COUNT", 6)
    monkeypatch.setattr(local_experiments, "PENN94_SPLIT_COUNTS", role_counts)
    monkeypatch.setattr(penn94_conversion, "PENN94_LABELED_NODE_COUNT", 6)
    monkeypatch.setattr(penn94_conversion, "PENN94_SPLIT_COUNTS", role_counts)
    split_rows = {
        "train": np.asarray(((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)), dtype=np.int64),
        "valid": np.asarray(((2, 3), (3, 4), (4, 5), (5, 0), (0, 1)), dtype=np.int64),
        "test": np.asarray(((4, 5), (5, 0), (0, 1), (1, 2), (2, 3)), dtype=np.int64),
    }
    safe_split_sha256 = penn94_conversion.write_deterministic_safe_splits(
        safe_split_path,
        split_rows,
    )
    monkeypatch.setattr(
        local_experiments,
        "_PENN94_SAFE_SPLIT_SHA256",
        safe_split_sha256,
    )

    def fake_safe_load_mat_arrays(
        path: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        assert path.read_bytes() == mat_bytes
        adjacency = sparse.coo_matrix(
            (
                np.ones(12, dtype=np.float64),
                (
                    np.asarray((0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0)),
                    np.asarray((1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5)),
                ),
            ),
            shape=(6, 6),
        )
        profile = np.zeros((6, 7), dtype=np.int64)
        profile[:, 1] = np.asarray((1, 2, 1, 2, 1, 2))
        return {"A": adjacency, "local_info": profile}

    monkeypatch.setattr(parsers, "safe_load_mat_arrays", fake_safe_load_mat_arrays)
    recipe = local_experiments.load_dataset_recipes()["facebook100"]
    converter_module_path = Path(
        sys.modules[local_experiments.verify_penn94_raw_split.__module__].__file__ or ""
    ).resolve()
    manifest_without_hash = {
        "schemaVersion": "socialgraph-fm.core-penn94-split-conversion/1.0",
        "sourceCommit": local_experiments.PENN94_LINKX_COMMIT,
        "sourceUrl": local_experiments.PENN94_RAW_SPLIT_URL,
        "sourceSha256": raw_split_sha256,
        "penn94DataUrl": next(
            source.url for source in recipe.sources if source.source_id == "Penn94"
        ),
        "penn94DataObservedSha256": mat_sha256,
        "derivedFormat": "npz with primitive little-endian int64 NPY members",
        "derivedSha256": safe_split_sha256,
        "converterVersion": local_experiments.PENN94_CONVERTER_VERSION,
        "converterCodeSha256": hashlib.sha256(converter_module_path.read_bytes()).hexdigest(),
        "splitCount": 5,
        "labeledNodeCount": 6,
        "roleCounts": role_counts,
        "recipeSha256": recipe.recipe_sha256,
    }
    manifest = {
        **manifest_without_hash,
        "manifestSha256": canonical_sha256(manifest_without_hash),
    }
    manifest_path.write_bytes((canonical_json(manifest) + "\n").encode())

    loaded = _REAL_LOAD_PENN94_LOCAL_INPUTS(runtime)

    assert loaded.split_inventory.fold_ids == tuple(f"fold-{index}" for index in range(5))
    assert loaded.split_inventory.selected_fold_id == "fold-0"
    assert len(loaded.targets_by_entity) == 6
    assert loaded.source_inventory["splitInventoryHash"] == loaded.split_inventory.inventory_hash

    attacker_rows = {name: values.copy() for name, values in split_rows.items()}
    attacker_rows["train"][0] = np.asarray((0, 2))
    attacker_rows["valid"][0] = np.asarray((1, 3))
    attacker_path = derived / "attacker-safe-splits.npz"
    attacker_sha256 = penn94_conversion.write_deterministic_safe_splits(
        attacker_path,
        attacker_rows,
    )
    os.replace(attacker_path, safe_split_path)
    attacker_manifest = dict(manifest)
    attacker_manifest["derivedSha256"] = attacker_sha256
    attacker_manifest.pop("manifestSha256")
    attacker_manifest["manifestSha256"] = canonical_sha256(attacker_manifest)
    manifest_path.write_bytes((canonical_json(attacker_manifest) + "\n").encode())
    with pytest.raises(ValueError, match="fixed hash-locked conversion"):
        _REAL_LOAD_PENN94_LOCAL_INPUTS(runtime)


def _reseal_local_report(document: dict[str, object]) -> None:
    evidence_names = (
        "codeInventoryEvidence",
        "environmentEvidence",
        "sourceInventoryEvidence",
        "splitInventoryEvidence",
        "targetsEvidence",
        "baseBundleEvidence",
        "adapterEvidence",
        "headReportEvidence",
        "calibrationEvidence",
        "formalPreflightEvidence",
        "recoveryBundleEvidence",
        "recoveryRequestEvidence",
        "recoveryEvaluationEvidence",
        "recoveryReceiptEvidence",
        "checkpointEvidence",
        "headArtifactEvidence",
        "structureManifestEvidence",
        "structureNpzEvidence",
        "artifactInventoryEvidence",
    )
    identity: dict[str, object] = {
        "datasetId": document["datasetId"],
        "phase": document["phase"],
        "taskKind": document["taskKind"],
        "seed": document["seed"],
        "configHash": document["configHash"],
        "dataHash": document["dataHash"],
        "codeHash": document["codeHash"],
        "environmentHash": document["environmentHash"],
        "targetsHash": document["targetsHash"],
        "formalPreflightEvidenceHash": document["formalPreflightEvidenceHash"],
        "splitInventoryHash": document["splitInventoryHash"],
        "selectedFoldId": document["selectedFoldId"],
    }
    for name in evidence_names:
        if name in document:
            identity[name] = document[name]
    document["runId"] = canonical_sha256(identity)
    document.pop("reportHash", None)
    document["reportHash"] = canonical_sha256(document)


def _write_local_report_pair(output: Path, artifacts: Path, document: dict[str, object]) -> None:
    _reseal_local_report(document)
    serialized = (canonical_json(document) + "\n").encode()
    output.write_bytes(serialized)
    (artifacts / "report.json").write_bytes(serialized)


def _refresh_artifact_inventory_for_attack(
    runtime: Path,
    document: dict[str, object],
) -> None:
    reference = document["artifactInventoryEvidence"]
    assert isinstance(reference, dict)
    path = runtime / str(reference["relativePath"])
    inventory = json.loads(path.read_bytes())
    artifact_directory = path.parent
    for entry in inventory["payloadFiles"]:
        payload_path = artifact_directory / entry["name"]
        entry["sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        entry["sizeBytes"] = payload_path.stat().st_size
    inventory.pop("inventoryHash")
    inventory["inventoryHash"] = canonical_sha256(inventory)
    path.write_text(canonical_json(inventory) + "\n", encoding="utf-8")
    changed_reference = dict(reference)
    changed_reference["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    changed_reference["semanticHash"] = inventory["inventoryHash"]
    document["artifactInventoryEvidence"] = changed_reference


def _hard_exit_local_publisher(
    runtime_text: str,
    output_text: str,
    exit_stage: str = "artifacts-before-rename",
) -> None:
    runtime = Path(runtime_text)
    output = Path(output_text)

    def hard_exit(stage: str, _path: Path) -> None:
        if stage == exit_stage:
            os._exit(73)

    local_experiments.load_penn94_local_inputs = _synthetic_penn_local_inputs
    local_experiments.load_email_local_inputs = _synthetic_email_local_inputs
    local_experiments._LOCAL_PUBLICATION_SEAM = hard_exit
    run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))


def test_email_local_smoke_executes_real_pipeline_and_remains_nonpromotable(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    bundle = _email_bundle()
    output = runtime / "email-smoke.json"
    report = run_local_nonpromotable_experiment(
        bundle=bundle,
        dataset_id="email-eu-core",
        phase="smoke",
        task_kind="edge-binary",
        targets_by_entity={
            "edge:0:1": 0,
            "edge:1:2": 1,
            "edge:2:3": 0,
            "edge:3:4": 1,
            "edge:4:5": 0,
            "edge:0:5": 1,
        },
        split_inventory=local_experiments.LocalSplitInventory.create(
            dataset_id="email-eu-core",
            manifests=(bundle.split_manifest,),
            selected_fold_id="fold-0",
        ),
        source_inventory=_source_inventory(bundle, dataset_id="email-eu-core"),
        runtime_root=runtime,
        output_path=output,
        seed=20260821,
        optimizer_steps=1,
        head_steps=1,
        device_name="cpu",
        formal_preflight_evidence_hash=_formal_hash(runtime),
        formal_ready=False,
    )

    assert report.dataset_id == "email-eu-core"
    assert report.phase == "smoke"
    assert report.formal_ready is False
    assert report.promotable is False
    assert report.calibration_report_hash is not None
    assert (
        local_experiments.reopen_local_experiment_evidence(output, runtime_root=runtime) == report
    )


def test_real_local_smoke_trains_recovers_head_and_calibration_but_never_promotes(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "experiments-core" / "local" / "fixture.json"

    report = run_local_nonpromotable_experiment(
        bundle=_node_bundle(),
        dataset_id="penn94",
        phase="smoke",
        task_kind="node-binary",
        targets_by_entity={str(index): index % 2 for index in range(6)},
        split_inventory=_penn_split_inventory(_node_bundle()),
        source_inventory=_source_inventory(_node_bundle()),
        runtime_root=runtime,
        output_path=output,
        seed=20260821,
        optimizer_steps=2,
        head_steps=2,
        device_name="cpu",
        formal_preflight_evidence_hash=_formal_hash(runtime),
        formal_ready=False,
    )

    reopened = LocalExperimentRun.model_validate_json(output.read_bytes())
    assert reopened == report
    assert report.phase == "smoke"
    assert report.promotable is False
    assert report.formal_ready is False
    assert report.failed_gates == ("phase-not-formal", "formal-corpus-not-ready")
    assert report.optimizer_steps == 2
    assert report.checkpoint_status == "training"
    assert report.checkpoint_promotable is False
    assert report.fresh_recovery_state_hash
    assert report.recovery_process_id != os.getpid()
    assert report.recovery_parent_process_id == os.getpid()
    assert report.recovery_device == "cpu"
    assert report.recovery_receipt_sha256
    receipt_document = json.loads((runtime / report.recovery_receipt_relative_path).read_bytes())
    expected_interpreter = (
        Path(getattr(sys, "_base_executable", sys.executable)).resolve()
        if os.name == "nt"
        else Path(sys.executable).resolve()
    )
    interpreter = Path(receipt_document["recoveryInterpreterPath"]).resolve(strict=True)
    assert interpreter == expected_interpreter
    assert receipt_document["recoveryInterpreterPath"] == str(interpreter)
    assert (
        receipt_document["recoveryInterpreterSha256"]
        == hashlib.sha256(interpreter.read_bytes()).hexdigest()
    )
    assert receipt_document["recoveryProcessId"] == report.recovery_process_id
    assert receipt_document["recoveryProcessId"] != os.getpid()
    assert receipt_document["recoveryParentProcessId"] == os.getpid()
    assert receipt_document["recoveryDevice"] == "cpu"
    expected_recovery_environment = local_experiments.local_environment_inventory("cpu").model_dump(
        mode="json", by_alias=True
    )
    assert receipt_document["recoveryEnvironmentInventory"] == expected_recovery_environment
    assert (
        receipt_document["recoveryEnvironmentHash"]
        == expected_recovery_environment["inventoryHash"]
    )
    assert report.head_promotion_eligible is False
    assert report.calibration_report_hash is not None
    assert report.calibration_promotion_eligible is False
    assert len(report.report_hash) == 64
    assert (
        "This local dev run selects official fold-0 only; formal evaluation consumes all "
        "five official folds." in report.limitations
    )
    reopened_evidence = local_experiments.reopen_local_experiment_evidence(
        output, runtime_root=runtime
    )
    assert reopened_evidence == report
    for reference in (
        report.code_inventory_evidence,
        report.environment_evidence,
        report.source_inventory_evidence,
        report.split_inventory_evidence,
        report.adapter_evidence,
        report.head_report_evidence,
        report.calibration_evidence,
    ):
        assert (runtime / reference.relative_path).is_file()
        assert reference.sha256
        assert reference.semantic_hash

    checkpoint_path = runtime / report.checkpoint_relative_path
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_path.write_bytes(checkpoint_bytes + b"post-report-mutation")
    with pytest.raises(ValueError, match="checkpoint"):
        local_experiments.reopen_local_experiment_evidence(output, runtime_root=runtime)
    checkpoint_path.write_bytes(checkpoint_bytes)

    calibration_path = runtime / report.calibration_evidence.relative_path
    calibration_path.write_bytes(calibration_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="evidence bytes"):
        local_experiments.reopen_local_experiment_evidence(output, runtime_root=runtime)

    tampered = report.model_dump(mode="json", by_alias=True)
    tampered["optimizerSteps"] = 1
    with pytest.raises(ValidationError, match="reportHash"):
        LocalExperimentRun.model_validate(tampered)


def test_local_report_references_every_recoverable_behavior_artifact(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "artifact-closure.json"
    report = run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))
    document = report.model_dump(mode="json", by_alias=True)

    assert document["schemaVersion"] == "socialgraph-fm.core-local-run/3.0"
    for field in (
        "targetsEvidence",
        "baseBundleEvidence",
        "recoveryBundleEvidence",
        "recoveryRequestEvidence",
        "recoveryEvaluationEvidence",
        "recoveryReceiptEvidence",
        "checkpointEvidence",
        "headArtifactEvidence",
        "structureManifestEvidence",
        "structureNpzEvidence",
        "formalPreflightEvidence",
        "artifactInventoryEvidence",
    ):
        reference = document[field]
        assert isinstance(reference, dict)
        assert (runtime / str(reference["relativePath"])).is_file()

    artifacts = output.with_suffix(".artifacts")
    assert (artifacts / "target-inventory.json").is_file()
    assert (artifacts / "recovery-bundle.json").is_file()
    assert (artifacts / "artifact-inventory.json").is_file()


def test_reopen_requires_the_fixed_dataset_loader_to_rederive_every_data_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "raw-authority.json"
    report = run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))
    baseline = local_experiments.LocalDatasetInputs(
        bundle=_node_bundle(),
        targets_by_entity={str(index): index % 2 for index in range(6)},
        split_inventory=_penn_split_inventory(_node_bundle()),
        source_inventory=_source_inventory(_node_bundle()),
    )

    changed_bundle_document = baseline.bundle.model_dump(mode="json", by_alias=True)
    changed_bundle_document["nodeFeatures"][0]["values"][0] = 99.0
    changed_bundle_document["graphVersionHash"] = calculate_graph_version_hash(
        changed_bundle_document
    )
    changed_bundle = CoreGraphBundle.model_validate(changed_bundle_document)

    split_documents = [
        fold.split_manifest.model_dump(mode="json", by_alias=True)
        for fold in baseline.split_inventory.folds
    ]
    assignments = split_documents[1]["assignments"]
    left = next(index for index, item in enumerate(assignments) if item["role"] == "train")
    right = next(index for index, item in enumerate(assignments) if item["role"] == "validation")
    assignments[left]["role"], assignments[right]["role"] = (
        assignments[right]["role"],
        assignments[left]["role"],
    )
    changed_split = local_experiments.LocalSplitInventory.create(
        dataset_id="penn94",
        manifests=tuple(split_documents),
        selected_fold_id="fold-0",
    )

    changed_source = dict(baseline.source_inventory)
    changed_source["scope"] = "attacker"
    changed_source["files"] = ()
    changed_source.pop("inventoryHash")
    changed_source["inventoryHash"] = canonical_sha256(changed_source)

    attacks = (
        local_experiments.LocalDatasetInputs(
            bundle=changed_bundle,
            targets_by_entity=baseline.targets_by_entity,
            split_inventory=baseline.split_inventory,
            source_inventory=baseline.source_inventory,
        ),
        local_experiments.LocalDatasetInputs(
            bundle=baseline.bundle,
            targets_by_entity={str(index): (index + 1) % 2 for index in range(6)},
            split_inventory=baseline.split_inventory,
            source_inventory=baseline.source_inventory,
        ),
        local_experiments.LocalDatasetInputs(
            bundle=baseline.bundle,
            targets_by_entity=baseline.targets_by_entity,
            split_inventory=changed_split,
            source_inventory=baseline.source_inventory,
        ),
        local_experiments.LocalDatasetInputs(
            bundle=baseline.bundle,
            targets_by_entity=baseline.targets_by_entity,
            split_inventory=baseline.split_inventory,
            source_inventory=changed_source,
        ),
    )
    for attack in attacks:
        monkeypatch.setattr(local_experiments, "load_penn94_local_inputs", lambda _root: attack)
        with pytest.raises(ValueError, match="authoritative|raw|dataset"):
            local_experiments.reopen_local_experiment_evidence(output, runtime_root=runtime)

    monkeypatch.setattr(local_experiments, "load_penn94_local_inputs", lambda _root: baseline)
    assert (
        local_experiments.reopen_local_experiment_evidence(output, runtime_root=runtime) == report
    )


@pytest.mark.parametrize("component", ("bundle", "targets", "fold-1", "source"))
def test_run_rejects_non_authoritative_inputs_before_training_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / f"non-authoritative-{component}.json"
    arguments = _local_fixture_arguments(runtime, output)
    baseline = _synthetic_penn_local_inputs(runtime)

    if component == "bundle":
        changed_bundle_document = baseline.bundle.model_dump(mode="json", by_alias=True)
        changed_bundle_document["nodeFeatures"][0]["values"][0] = 99.0
        changed_bundle_document["graphVersionHash"] = calculate_graph_version_hash(
            changed_bundle_document
        )
        arguments["bundle"] = CoreGraphBundle.model_validate(changed_bundle_document)
    elif component == "targets":
        arguments["targets_by_entity"] = {str(index): (index + 1) % 2 for index in range(6)}
    elif component == "fold-1":
        split_documents = [
            fold.split_manifest.model_dump(mode="json", by_alias=True)
            for fold in baseline.split_inventory.folds
        ]
        assignments = split_documents[1]["assignments"]
        left = next(index for index, item in enumerate(assignments) if item["role"] == "train")
        right = next(
            index for index, item in enumerate(assignments) if item["role"] == "validation"
        )
        assignments[left]["role"], assignments[right]["role"] = (
            assignments[right]["role"],
            assignments[left]["role"],
        )
        arguments["split_inventory"] = local_experiments.LocalSplitInventory.create(
            dataset_id="penn94",
            manifests=tuple(split_documents),
            selected_fold_id="fold-0",
        )
    else:
        changed_source = dict(baseline.source_inventory)
        changed_source["scope"] = "attacker"
        changed_source["files"] = ()
        changed_source.pop("inventoryHash")
        changed_source["inventoryHash"] = canonical_sha256(changed_source)
        arguments["source_inventory"] = changed_source

    trained: list[bool] = []

    def training_bomb(**_kwargs: object) -> LocalExperimentRun:
        trained.append(True)
        raise AssertionError("training must not start for non-authoritative inputs")

    monkeypatch.setattr(
        local_experiments,
        "_run_local_nonpromotable_experiment_unpublished",
        training_bomb,
    )
    with pytest.raises(ValueError, match="authoritative|raw|dataset"):
        run_local_nonpromotable_experiment(**arguments)

    assert not trained
    assert not output.exists()
    assert not output.with_suffix(".artifacts").exists()
    assert not tuple(runtime.glob(".*.staging"))
    assert not tuple(runtime.glob(".*.local-experiment.lock"))
    assert not tuple(runtime.glob(".*.recovery-journal.json"))
    assert not (runtime / "experiments-core" / "cache").exists()


def test_reopen_rejects_unknown_fields_even_after_coherent_document_rehash(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "exact-evidence-keys.json"
    report = run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))
    pristine_report = report.model_dump(mode="json", by_alias=True)
    artifacts = output.with_suffix(".artifacts")
    inventory_path = artifacts / "artifact-inventory.json"
    pristine_inventory = inventory_path.read_bytes()

    attacks = (
        ("codeInventoryEvidence", "inventoryHash"),
        ("environmentEvidence", "inventoryHash"),
        ("sourceInventoryEvidence", "inventoryHash"),
        ("splitInventoryEvidence", "inventoryHash"),
        ("targetsEvidence", "inventoryHash"),
        ("adapterEvidence", "evidenceHash"),
        ("recoveryRequestEvidence", "requestHash"),
        ("recoveryReceiptEvidence", "receiptHash"),
    )
    for field, hash_field in attacks:
        reference = pristine_report[field]
        assert isinstance(reference, dict)
        path = runtime / str(reference["relativePath"])
        pristine = path.read_bytes()
        document = json.loads(pristine)
        document["attackerUnknown"] = True
        document.pop(hash_field)
        document[hash_field] = canonical_sha256(document)
        path.write_bytes((canonical_json(document) + "\n").encode())
        changed = dict(pristine_report)
        changed_reference = dict(reference)
        changed_reference["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        changed_reference["semanticHash"] = document[hash_field]
        changed[field] = changed_reference
        mirrored_hashes = {
            "codeInventoryEvidence": "codeHash",
            "environmentEvidence": "environmentHash",
            "splitInventoryEvidence": "splitInventoryHash",
            "targetsEvidence": "targetsHash",
            "recoveryReceiptEvidence": "recoveryReceiptHash",
        }
        mirrored_hash = mirrored_hashes.get(field)
        if mirrored_hash is not None:
            changed[mirrored_hash] = document[hash_field]
        if field == "recoveryReceiptEvidence":
            changed["recoveryReceiptSha256"] = changed_reference["sha256"]
        _refresh_artifact_inventory_for_attack(runtime, changed)
        _write_local_report_pair(output, artifacts, changed)
        with pytest.raises(
            (ValueError, ValidationError),
            match="inventory|contract|extra|request|references",
        ):
            local_experiments.reopen_local_experiment_evidence(output, runtime_root=runtime)
        path.write_bytes(pristine)
        inventory_path.write_bytes(pristine_inventory)
        _write_local_report_pair(output, artifacts, dict(pristine_report))

    inventory_document = json.loads(pristine_inventory)
    inventory_document["attackerUnknown"] = True
    inventory_document.pop("inventoryHash")
    inventory_document["inventoryHash"] = canonical_sha256(inventory_document)
    inventory_path.write_bytes((canonical_json(inventory_document) + "\n").encode())
    changed = dict(pristine_report)
    changed_reference = dict(pristine_report["artifactInventoryEvidence"])
    changed_reference["sha256"] = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    changed_reference["semanticHash"] = inventory_document["inventoryHash"]
    changed["artifactInventoryEvidence"] = changed_reference
    _write_local_report_pair(output, artifacts, changed)
    with pytest.raises(ValueError, match="artifact inventory"):
        local_experiments.reopen_local_experiment_evidence(output, runtime_root=runtime)


def test_reopen_rejects_coherent_claim_and_artifact_rehash_attacks(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "coherent-attacks.json"
    report = run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))
    artifacts = output.with_suffix(".artifacts")
    pristine_report = report.model_dump(mode="json", by_alias=True)
    artifact_inventory_reference = pristine_report["artifactInventoryEvidence"]
    assert isinstance(artifact_inventory_reference, dict)
    artifact_inventory_path = runtime / str(artifact_inventory_reference["relativePath"])
    artifact_inventory_before = artifact_inventory_path.read_bytes()
    accepted: list[str] = []

    def expect_rejected(name: str) -> None:
        try:
            local_experiments.reopen_local_experiment_evidence(output, runtime_root=runtime)
        except (ValueError, ValidationError):
            return
        accepted.append(name)

    for field, replacement in (
        ("optimizerSteps", 2),
        ("headSteps", 2),
        ("supervisedDataHash", "c" * 64),
        ("encodedArtifactHash", "c" * 64),
        ("baseGraphVersionHash", "c" * 64),
        ("enrichedGraphVersionHash", "c" * 64),
        ("sourceSha256", "c" * 64),
        ("nodeCount", 7),
        ("edgeCount", 7),
        ("device", "cuda"),
    ):
        changed = dict(pristine_report)
        changed[field] = replacement
        _write_local_report_pair(output, artifacts, changed)
        expect_rejected(f"report:{field}")
    _write_local_report_pair(output, artifacts, dict(pristine_report))

    source_reference = pristine_report["sourceInventoryEvidence"]
    assert isinstance(source_reference, dict)
    source_path = runtime / str(source_reference["relativePath"])
    source_before = source_path.read_bytes()
    source_document = json.loads(source_before)
    source_document["scope"] = "coherently-rehashed-attacker-scope"
    source_document.pop("inventoryHash")
    source_document["inventoryHash"] = canonical_sha256(source_document)
    source_path.write_text(canonical_json(source_document) + "\n", encoding="utf-8")
    changed = dict(pristine_report)
    changed_reference = dict(source_reference)
    changed_reference["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    changed_reference["semanticHash"] = source_document["inventoryHash"]
    changed["sourceInventoryEvidence"] = changed_reference
    _refresh_artifact_inventory_for_attack(runtime, changed)
    _write_local_report_pair(output, artifacts, changed)
    expect_rejected("source-inventory")
    source_path.write_bytes(source_before)
    artifact_inventory_path.write_bytes(artifact_inventory_before)
    _write_local_report_pair(output, artifacts, dict(pristine_report))

    targets_reference = pristine_report["targetsEvidence"]
    assert isinstance(targets_reference, dict)
    targets_path = runtime / str(targets_reference["relativePath"])
    targets_before = targets_path.read_bytes()
    targets_document = json.loads(targets_before)
    targets_document["targets"][0]["target"] = 1 - targets_document["targets"][0]["target"]
    targets_document.pop("inventoryHash")
    targets_document["inventoryHash"] = canonical_sha256(targets_document)
    targets_path.write_text(canonical_json(targets_document) + "\n", encoding="utf-8")
    changed = dict(pristine_report)
    changed_reference = dict(targets_reference)
    changed_reference["sha256"] = hashlib.sha256(targets_path.read_bytes()).hexdigest()
    changed_reference["semanticHash"] = targets_document["inventoryHash"]
    changed["targetsEvidence"] = changed_reference
    changed["targetsHash"] = targets_document["inventoryHash"]
    _refresh_artifact_inventory_for_attack(runtime, changed)
    _write_local_report_pair(output, artifacts, changed)
    expect_rejected("target-entity-mapping")
    targets_path.write_bytes(targets_before)
    artifact_inventory_path.write_bytes(artifact_inventory_before)
    _write_local_report_pair(output, artifacts, dict(pristine_report))

    formal_reference = pristine_report["formalPreflightEvidence"]
    assert isinstance(formal_reference, dict)
    formal_path = runtime / str(formal_reference["relativePath"])
    formal_before = formal_path.read_bytes()
    formal_document = json.loads(formal_before)
    formal_document["requirementsHash"] = "c" * 64
    formal_document.pop("evidenceHash")
    formal_document["evidenceHash"] = canonical_sha256(formal_document)
    formal_path.write_bytes((canonical_json(formal_document) + "\n").encode())
    changed = dict(pristine_report)
    changed_reference = dict(formal_reference)
    changed_reference["sha256"] = hashlib.sha256(formal_path.read_bytes()).hexdigest()
    changed_reference["semanticHash"] = formal_document["evidenceHash"]
    changed["formalPreflightEvidence"] = changed_reference
    changed["formalPreflightEvidenceHash"] = formal_document["evidenceHash"]
    _write_local_report_pair(output, artifacts, changed)
    expect_rejected("formal-preflight-rederivation")
    formal_path.write_bytes(formal_before)
    _write_local_report_pair(output, artifacts, dict(pristine_report))

    receipt_path = runtime / report.recovery_receipt_relative_path
    receipt_before = receipt_path.read_bytes()
    receipt_document = json.loads(receipt_before)
    receipt_document["trainerStateHash"] = "c" * 64
    receipt_document.pop("receiptHash")
    receipt_document["receiptHash"] = canonical_sha256(receipt_document)
    receipt_path.write_text(canonical_json(receipt_document) + "\n", encoding="utf-8")
    changed = dict(pristine_report)
    changed["recoveryReceiptSha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    changed["recoveryReceiptHash"] = receipt_document["receiptHash"]
    changed_receipt_reference = dict(pristine_report["recoveryReceiptEvidence"])
    changed_receipt_reference["sha256"] = changed["recoveryReceiptSha256"]
    changed_receipt_reference["semanticHash"] = changed["recoveryReceiptHash"]
    changed["recoveryReceiptEvidence"] = changed_receipt_reference
    _refresh_artifact_inventory_for_attack(runtime, changed)
    _write_local_report_pair(output, artifacts, changed)
    expect_rejected("recovery-receipt-state")
    receipt_path.write_bytes(receipt_before)
    artifact_inventory_path.write_bytes(artifact_inventory_before)
    _write_local_report_pair(output, artifacts, dict(pristine_report))

    calibration_reference = pristine_report["calibrationEvidence"]
    assert isinstance(calibration_reference, dict)
    calibration_path = runtime / str(calibration_reference["relativePath"])
    calibration_before = calibration_path.read_bytes()
    calibration_document = json.loads(calibration_before)
    calibration_document["headTrainingReportHash"] = "c" * 64
    calibration_document.pop("reportHash")
    calibration_document["reportHash"] = canonical_sha256(calibration_document)
    calibration_path.write_text(canonical_json(calibration_document) + "\n", encoding="utf-8")
    changed = dict(pristine_report)
    changed_reference = dict(calibration_reference)
    changed_reference["sha256"] = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    changed_reference["semanticHash"] = calibration_document["reportHash"]
    changed["calibrationEvidence"] = changed_reference
    changed["calibrationReportHash"] = calibration_document["reportHash"]
    _refresh_artifact_inventory_for_attack(runtime, changed)
    _write_local_report_pair(output, artifacts, changed)
    expect_rejected("calibration-head-binding")
    calibration_path.write_bytes(calibration_before)
    artifact_inventory_path.write_bytes(artifact_inventory_before)
    _write_local_report_pair(output, artifacts, dict(pristine_report))

    head_path = runtime / report.head_artifact_relative_path
    head_before = head_path.read_bytes()
    head_payload = torch.load(head_path, map_location="cpu", weights_only=True)
    encoder_key = next(name for name in head_payload["model"] if name.startswith("encoder."))
    head_payload["model"][encoder_key] = head_payload["model"][encoder_key].clone()
    head_payload["model"][encoder_key].view(-1)[0] += 1
    torch.save(head_payload, head_path)
    adapter_reference = pristine_report["adapterEvidence"]
    assert isinstance(adapter_reference, dict)
    adapter_path = runtime / str(adapter_reference["relativePath"])
    adapter_before = adapter_path.read_bytes()
    adapter_document = json.loads(adapter_before)
    adapter_document["headArtifactSha256"] = hashlib.sha256(head_path.read_bytes()).hexdigest()
    adapter_document.pop("evidenceHash")
    adapter_document["evidenceHash"] = canonical_sha256(adapter_document)
    adapter_path.write_text(canonical_json(adapter_document) + "\n", encoding="utf-8")
    changed = dict(pristine_report)
    changed_adapter_reference = dict(adapter_reference)
    changed_adapter_reference["sha256"] = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    changed_adapter_reference["semanticHash"] = adapter_document["evidenceHash"]
    changed["adapterEvidence"] = changed_adapter_reference
    changed["headArtifactSha256"] = adapter_document["headArtifactSha256"]
    changed_head_reference = dict(pristine_report["headArtifactEvidence"])
    changed_head_reference["sha256"] = adapter_document["headArtifactSha256"]
    changed_head_reference["semanticHash"] = canonical_sha256(
        {
            "schemaVersion": "socialgraph-fm.core-local-head-artifact/1.0",
            "modelStateHash": local_experiments._state_hash(head_payload["model"]),
            "adapterStateHash": local_experiments._state_hash(head_payload["adapter"]),
            "headReportHash": head_payload["headReportHash"],
        }
    )
    changed["headArtifactEvidence"] = changed_head_reference
    adapter_document["headArtifactSemanticHash"] = changed_head_reference["semanticHash"]
    adapter_document.pop("evidenceHash")
    adapter_document["evidenceHash"] = canonical_sha256(adapter_document)
    adapter_path.write_text(canonical_json(adapter_document) + "\n", encoding="utf-8")
    changed_adapter_reference["sha256"] = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    changed_adapter_reference["semanticHash"] = adapter_document["evidenceHash"]
    changed["adapterEvidence"] = changed_adapter_reference
    _refresh_artifact_inventory_for_attack(runtime, changed)
    _write_local_report_pair(output, artifacts, changed)
    expect_rejected("head-nonselected-model-state")
    head_path.write_bytes(head_before)
    adapter_path.write_bytes(adapter_before)
    artifact_inventory_path.write_bytes(artifact_inventory_before)
    _write_local_report_pair(output, artifacts, dict(pristine_report))

    for path in (
        artifacts / "recovery-bundle.json",
        artifacts / "report.json",
        next((runtime / "experiments-core" / "cache").glob("*/manifest.json")),
        next((runtime / "experiments-core" / "cache").glob("*/structure.npz")),
    ):
        before = path.read_bytes()
        path.unlink()
        expect_rejected(f"missing:{path.name}")
        path.write_bytes(before)

    extra = artifacts / "attacker-extra.bin"
    extra.write_bytes(b"unbound extra artifact")
    expect_rejected("extra-artifact")
    extra.unlink()
    _write_local_report_pair(output, artifacts, dict(pristine_report))

    assert accepted == []


def test_recovery_child_rejects_a_forged_parent_process_id(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    bundle = _node_bundle()
    output = runtime / "parent-binding.json"
    report = run_local_nonpromotable_experiment(
        bundle=bundle,
        dataset_id="penn94",
        phase="dev",
        task_kind="node-binary",
        targets_by_entity={str(index): index % 2 for index in range(6)},
        split_inventory=_penn_split_inventory(bundle),
        source_inventory=_source_inventory(bundle),
        runtime_root=runtime,
        output_path=output,
        seed=20260821,
        optimizer_steps=1,
        head_steps=1,
        device_name="cpu",
        formal_preflight_evidence_hash=_formal_hash(runtime),
        formal_ready=False,
    )
    artifacts = runtime / Path(report.recovery_receipt_relative_path).parent
    request = json.loads((artifacts / "recovery-request.json").read_bytes())
    assert request["trainingEnvironmentInventory"]["torchGeometric"] == torch_geometric.__version__
    drift_interpreter, drift_environment = local_experiments._recovery_interpreter()
    for package_field in ("torch", "numpy", "scipy", "pydantic", "torchGeometric"):
        environment_drift = json.loads((artifacts / "recovery-request.json").read_bytes())
        environment_drift["trainingEnvironmentInventory"][package_field] = "0.attacker"
        environment_drift["trainingEnvironmentInventory"].pop("inventoryHash")
        environment_drift["trainingEnvironmentInventory"]["inventoryHash"] = canonical_sha256(
            environment_drift["trainingEnvironmentInventory"]
        )
        environment_drift["bindings"]["environment_hash"] = environment_drift[
            "trainingEnvironmentInventory"
        ]["inventoryHash"]
        slug = package_field.lower()
        environment_drift["evaluationArtifactName"] = f"drift-{slug}-evaluation.pt"
        environment_drift["receiptName"] = f"drift-{slug}-receipt.json"
        environment_drift.pop("requestHash")
        environment_drift["requestHash"] = canonical_sha256(environment_drift)
        drift_request = artifacts / f"drift-{slug}-request.json"
        drift_request.write_bytes((canonical_json(environment_drift) + "\n").encode())
        drift_completed = subprocess.run(
            [
                str(drift_interpreter),
                "-m",
                "socialgraph_gfm.core.local_recovery",
                "--request",
                str(drift_request),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=drift_environment,
        )
        assert drift_completed.returncode != 0
        assert "environment inventory" in drift_completed.stderr
        assert not (artifacts / f"drift-{slug}-evaluation.pt").exists()
        assert not (artifacts / f"drift-{slug}-receipt.json").exists()

    request["parentProcessId"] = os.getpid() + 100_000
    request["evaluationArtifactName"] = "forged-evaluation.pt"
    request["receiptName"] = "forged-receipt.json"
    request.pop("requestHash")
    request["requestHash"] = canonical_sha256(request)
    forged_request = artifacts / "forged-request.json"
    forged_request.write_bytes((canonical_json(request) + "\n").encode())

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "socialgraph_gfm.core.local_recovery",
            "--request",
            str(forged_request),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "parent process identity" in completed.stderr
    assert not (artifacts / "forged-evaluation.pt").exists()
    assert not (artifacts / "forged-receipt.json").exists()


def test_local_run_publication_is_exact_idempotent_and_conflicts_fail_closed(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "local.json"
    arguments = dict(
        bundle=_node_bundle(),
        dataset_id="penn94",
        phase="dev",
        task_kind="node-binary",
        targets_by_entity={str(index): index % 2 for index in range(6)},
        split_inventory=_penn_split_inventory(_node_bundle()),
        source_inventory=_source_inventory(_node_bundle()),
        runtime_root=runtime,
        output_path=output,
        seed=7,
        optimizer_steps=1,
        head_steps=1,
        device_name="cpu",
        formal_preflight_evidence_hash=_formal_hash(runtime),
        formal_ready=False,
    )
    first = run_local_nonpromotable_experiment(**arguments)
    before = output.read_bytes()
    second = run_local_nonpromotable_experiment(**arguments)
    assert second == first
    conflicting = {**arguments, "seed": 8}
    with pytest.raises(FileExistsError, match="different local experiment"):
        run_local_nonpromotable_experiment(**conflicting)
    conflicting_targets = {
        **arguments,
        "targets_by_entity": {str(index): (index + 1) % 2 for index in range(6)},
    }
    with pytest.raises(FileExistsError, match="different local experiment"):
        run_local_nonpromotable_experiment(**conflicting_targets)
    assert output.read_bytes() == before
    assert first.promotable is False


def test_exact_artifacts_without_recovery_ownership_are_never_adopted(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "unowned-artifacts.json"
    arguments = _local_fixture_arguments(runtime, output)
    run_local_nonpromotable_experiment(**arguments)
    artifacts = output.with_suffix(".artifacts")
    internal_before = (artifacts / "report.json").read_bytes()
    output.unlink()
    assert not tuple(runtime.glob(".*.local-experiment.lock"))
    assert not tuple(runtime.glob(".*.recovery-journal.json"))

    with pytest.raises(FileExistsError, match="lack exact stale-publication ownership"):
        run_local_nonpromotable_experiment(**arguments)

    assert not output.exists()
    assert (artifacts / "report.json").read_bytes() == internal_before


def test_local_artifact_crash_before_final_publication_preserves_audit_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "crash.json"

    def fail_before_final(stage: str, _path: Path) -> None:
        if stage == "artifacts-before-rename":
            raise RuntimeError("injected crash")

    monkeypatch.setattr(local_experiments, "_LOCAL_PUBLICATION_SEAM", fail_before_final)
    with pytest.raises(RuntimeError, match="injected crash"):
        run_local_nonpromotable_experiment(
            bundle=_node_bundle(),
            dataset_id="penn94",
            phase="dev",
            task_kind="node-binary",
            targets_by_entity={str(index): index % 2 for index in range(6)},
            split_inventory=_penn_split_inventory(_node_bundle()),
            source_inventory=_source_inventory(_node_bundle()),
            runtime_root=runtime,
            output_path=output,
            seed=7,
            optimizer_steps=1,
            head_steps=1,
            device_name="cpu",
            formal_preflight_evidence_hash=_formal_hash(runtime),
            formal_ready=False,
        )

    assert not output.exists()
    assert not output.with_suffix(".artifacts").exists()
    assert len(tuple(runtime.glob(".*.staging"))) == 1
    if os.name == "nt":
        assert not tuple(runtime.glob(".*.recovery-journal.json"))
        assert not tuple(runtime.glob(".*.local-experiment.lock"))


def test_local_artifact_never_claims_a_child_added_before_the_proof_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "pre-seal-extra-child.json"
    competitor_bytes = b"attacker child predates the proof snapshot"

    def add_pre_seal_child(stage: str, target: Path) -> None:
        if stage != "artifacts-before-seal":
            return
        staging = next(target.parent.glob(f".{target.name}.*.staging"))
        (staging / "attacker-pre-seal.bin").write_bytes(competitor_bytes)

    monkeypatch.setattr(local_experiments, "_LOCAL_PUBLICATION_SEAM", add_pre_seal_child)
    with pytest.raises(ValueError, match="not exact before sealing"):
        run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))

    staging = next(runtime.glob(f".{output.with_suffix('.artifacts').name}.*.staging"))
    assert (staging / "attacker-pre-seal.bin").read_bytes() == competitor_bytes
    assert not output.exists()
    assert not output.with_suffix(".artifacts").exists()


def test_local_artifact_rollback_never_claims_a_child_added_after_sealing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "extra-child-race.json"
    competitor_bytes = b"attacker owns this extra child"

    def add_extra_child(stage: str, target: Path) -> None:
        if stage != "artifacts-before-rename":
            return
        staging = next(target.parent.glob(f".{target.name}.*.staging"))
        (staging / "attacker-extra.bin").write_bytes(competitor_bytes)
        raise RuntimeError("injected post-seal failure")

    monkeypatch.setattr(local_experiments, "_LOCAL_PUBLICATION_SEAM", add_extra_child)
    with pytest.raises(RuntimeError, match="post-seal failure"):
        run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))

    staging = next(runtime.glob(f".{output.with_suffix('.artifacts').name}.*.staging"))
    assert (staging / "attacker-extra.bin").read_bytes() == competitor_bytes
    assert not output.exists()
    assert not output.with_suffix(".artifacts").exists()


def test_local_artifact_rollback_never_deletes_a_child_replaced_after_sealing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "replaced-child-race.json"
    competitor_bytes = b"attacker owns this replacement child"
    replacement_succeeded: list[bool] = []

    def replace_child(stage: str, target: Path) -> None:
        if stage != "artifacts-after-child-release":
            return
        staging = next(target.parent.glob(f".{target.name}.*.staging"))
        competitor = runtime / "attacker-replacement.bin"
        competitor.write_bytes(competitor_bytes)
        try:
            os.replace(competitor, staging / "report.json")
            replacement_succeeded.append(True)
        except OSError:
            replacement_succeeded.append(False)
        raise RuntimeError("injected replacement failure")

    monkeypatch.setattr(local_experiments, "_LOCAL_PUBLICATION_SEAM", replace_child)
    with pytest.raises(RuntimeError, match="replacement failure"):
        run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))

    if replacement_succeeded == [True]:
        staging = next(runtime.glob(f".{output.with_suffix('.artifacts').name}.*.staging"))
        assert (staging / "report.json").read_bytes() == competitor_bytes
    else:
        assert (runtime / "attacker-replacement.bin").read_bytes() == competitor_bytes
    assert not output.exists()
    assert not output.with_suffix(".artifacts").exists()


def test_published_artifacts_remain_held_until_external_report_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "held-published-artifacts.json"
    competitor_bytes = b"competitor artifact after internal reopen"
    inserted: list[bool] = []

    def add_after_internal_reopen(stage: str, target: Path) -> None:
        if stage != "artifacts-before-external-report":
            return
        try:
            (target / "attacker-after-reopen.bin").write_bytes(competitor_bytes)
            inserted.append(True)
        except OSError:
            inserted.append(False)

    monkeypatch.setattr(
        local_experiments,
        "_LOCAL_PUBLICATION_SEAM",
        add_after_internal_reopen,
    )
    try:
        report = run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))
    except ValueError as error:
        assert "inventory" in str(error) or "known file" in str(error)
        assert inserted == [True]
        assert not output.exists()
        assert (
            output.with_suffix(".artifacts") / "attacker-after-reopen.bin"
        ).read_bytes() == competitor_bytes
    else:
        assert report.promotable is False
        assert inserted == [False]
        assert output.is_file()


def test_local_artifact_hard_exit_recovers_only_exact_owned_staging(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "hard-exit.json"
    process = multiprocessing.get_context("spawn").Process(
        target=_hard_exit_local_publisher,
        args=(str(runtime), str(output)),
    )
    process.start()
    process.join(timeout=120)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("hard-exit local publisher did not reach the injected seam")
    assert process.exitcode == 73
    assert not output.exists()
    assert not output.with_suffix(".artifacts").exists()
    assert len(tuple(runtime.glob(".*.staging"))) == 1
    assert len(tuple(runtime.glob(".*.local-experiment.lock"))) == 1

    report = run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))

    assert report.promotable is False
    assert output.is_file()
    assert output.with_suffix(".artifacts").is_dir()
    assert not tuple(runtime.glob(".*.staging"))
    if os.name == "nt":
        assert not tuple(runtime.glob(".*.recovery-journal.json"))
        assert not tuple(runtime.glob(".*.local-experiment.lock"))
    else:
        assert len(tuple(runtime.glob(".*.recovery-journal.json"))) == 1
        assert len(tuple(runtime.glob(".*.local-experiment.lock"))) == 1


def test_local_artifact_hard_exit_after_rename_recovers_and_cleans_ownership_files(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "hard-exit-post-rename.json"
    process = multiprocessing.get_context("spawn").Process(
        target=_hard_exit_local_publisher,
        args=(str(runtime), str(output), "artifacts-post-rename"),
    )
    process.start()
    process.join(timeout=120)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("post-rename hard-exit publisher did not reach the injected seam")
    assert process.exitcode == 73
    assert not output.exists()
    assert output.with_suffix(".artifacts").is_dir()
    assert len(tuple(runtime.glob(".*.recovery-journal.json"))) == 1
    assert len(tuple(runtime.glob(".*.local-experiment.lock"))) == 1

    report = run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))

    assert (
        local_experiments.reopen_local_experiment_evidence(output, runtime_root=runtime) == report
    )
    assert not tuple(runtime.glob(".*.staging"))
    if os.name == "nt":
        assert not tuple(runtime.glob(".*.recovery-journal.json"))
        assert not tuple(runtime.glob(".*.local-experiment.lock"))
    else:
        assert len(tuple(runtime.glob(".*.recovery-journal.json"))) == 1
        assert len(tuple(runtime.glob(".*.local-experiment.lock"))) == 1


def test_fresh_recovery_binds_the_receipt_to_the_direct_child_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "replaced-child.json"
    real_popen = subprocess.Popen

    class _PidSubstitution:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._process = real_popen(*args, **kwargs)
            self.args = self._process.args
            self.pid = self._process.pid + 100_000

        @property
        def returncode(self) -> int | None:
            return self._process.returncode

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            self._process.__exit__(*args)

        def communicate(self, *args: object, **kwargs: object):
            return self._process.communicate(*args, **kwargs)

        def poll(self) -> int | None:
            return self._process.poll()

        def wait(self, *args: object, **kwargs: object) -> int:
            return self._process.wait(*args, **kwargs)

        def kill(self) -> None:
            self._process.kill()

    monkeypatch.setattr(local_experiments.subprocess, "Popen", _PidSubstitution)
    with pytest.raises(ValueError, match="direct child"):
        run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))
    assert not output.exists()
    assert not output.with_suffix(".artifacts").exists()


def test_fresh_recovery_interpreter_imports_the_exact_bound_source() -> None:
    interpreter, environment = local_experiments._recovery_interpreter()
    expected_source_root = Path(local_experiments.__file__).resolve().parents[2]
    expected_module = expected_source_root / "socialgraph_gfm/core/local_recovery.py"

    if environment is not None:
        pythonpath = environment["PYTHONPATH"].split(os.pathsep)
        assert Path(pythonpath[0]).resolve(strict=True) == expected_source_root
        assert Path(pythonpath[1]).resolve(strict=True) == Path(
            local_experiments.sysconfig.get_path("purelib")
        ).resolve(strict=True)

    completed = subprocess.run(
        [
            str(interpreter),
            "-c",
            (
                "from pathlib import Path; "
                "from socialgraph_gfm.core import local_recovery; "
                "print(Path(local_recovery.__file__).resolve())"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert Path(completed.stdout.strip()).resolve(strict=True) == expected_module


def test_fresh_recovery_rejects_an_interposed_extra_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "interposed-child.json"
    real_popen = subprocess.Popen

    def interpose(command: list[str], **kwargs: object):
        wrapper = "import subprocess,sys; raise SystemExit(subprocess.run(sys.argv[1:]).returncode)"
        return real_popen([command[0], "-c", wrapper, *command], **kwargs)

    monkeypatch.setattr(local_experiments.subprocess, "Popen", interpose)
    with pytest.raises(ValueError, match="parent process identity"):
        run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))
    assert not output.exists()
    assert not output.with_suffix(".artifacts").exists()


@pytest.mark.parametrize("substitution", ("lock", "staging", "journal"))
def test_local_hard_exit_recovery_never_removes_a_competitor_substitution(
    tmp_path: Path,
    substitution: str,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / f"competitor-{substitution}.json"
    process = multiprocessing.get_context("spawn").Process(
        target=_hard_exit_local_publisher,
        args=(str(runtime), str(output)),
    )
    process.start()
    process.join(timeout=120)
    assert process.exitcode == 73
    lock = next(runtime.glob(".*.local-experiment.lock"))
    staging = next(runtime.glob(".*.staging"))
    journal = next(runtime.glob(".*.recovery-journal.json"))
    competitor_bytes = f"competitor-{substitution}-survives".encode()

    if substitution == "staging":
        displaced = runtime / "displaced-owned-staging"
        os.replace(staging, displaced)
        staging.mkdir()
        competitor = staging / "competitor.bin"
        competitor.write_bytes(competitor_bytes)
    else:
        target = lock if substitution == "lock" else journal
        displaced = runtime / f"displaced-owned-{substitution}"
        os.replace(target, displaced)
        target.write_bytes(competitor_bytes)
        competitor = target

    with pytest.raises((RuntimeError, ValueError, FileExistsError)):
        run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))

    assert competitor.read_bytes() == competitor_bytes
    assert not output.exists()
    assert not output.with_suffix(".artifacts").exists()


def test_local_hard_exit_conflicting_or_live_owner_retry_preserves_exact_recovery(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "stale-conflict.json"
    process = multiprocessing.get_context("spawn").Process(
        target=_hard_exit_local_publisher,
        args=(str(runtime), str(output)),
    )
    process.start()
    process.join(timeout=120)
    assert process.exitcode == 73
    staging = next(runtime.glob(".*.staging"))
    lock = next(runtime.glob(".*.local-experiment.lock"))
    journal = next(runtime.glob(".*.recovery-journal.json"))
    staging_identity = local_experiments._path_identity(staging)
    lock_identity = local_experiments._path_identity(lock)
    journal_before = journal.read_bytes()

    conflicting = _local_fixture_arguments(runtime, output)
    conflicting["seed"] = 20260822
    with pytest.raises(ValueError, match="journal identity differs"):
        run_local_nonpromotable_experiment(**conflicting)
    assert local_experiments._path_identity(staging) == staging_identity
    assert local_experiments._path_identity(lock) == lock_identity

    live_owner_document = json.loads(journal_before)
    live_owner = local_experiments._process_identity(os.getpid())
    assert live_owner is not None
    live_owner_document["ownerProcessIdentity"] = live_owner
    live_owner_document.pop("journalHash")
    live_owner_document["journalHash"] = canonical_sha256(live_owner_document)
    journal.write_bytes((canonical_json(live_owner_document) + "\n").encode())
    with pytest.raises(RuntimeError, match="active publisher"):
        run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))
    assert local_experiments._path_identity(staging) == staging_identity
    assert local_experiments._path_identity(lock) == lock_identity

    journal.write_bytes(journal_before)
    report = run_local_nonpromotable_experiment(**_local_fixture_arguments(runtime, output))
    assert report.promotable is False
    assert not tuple(runtime.glob(".*.staging"))


def test_local_publication_rejects_an_ancestor_link_before_training(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    real_parent = runtime / "real-parent"
    real_parent.mkdir()
    linked_parent = runtime / "linked-parent"
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(linked_parent), str(real_parent)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip("directory junction creation is unavailable")
    else:
        try:
            linked_parent.symlink_to(real_parent, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(ValueError, match="link|reparse"):
        run_local_nonpromotable_experiment(
            bundle=_node_bundle(),
            dataset_id="penn94",
            phase="dev",
            task_kind="node-binary",
            targets_by_entity={str(index): index % 2 for index in range(6)},
            split_inventory=_penn_split_inventory(_node_bundle()),
            source_inventory=_source_inventory(_node_bundle()),
            runtime_root=runtime,
            output_path=linked_parent / "linked.json",
            seed=7,
            optimizer_steps=1,
            head_steps=1,
            device_name="cpu",
            formal_preflight_evidence_hash=_formal_hash(runtime),
            formal_ready=False,
        )

    assert not (real_parent / "linked.json").exists()


def test_local_artifact_parent_replacement_is_prevented_or_detected_without_deleting_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    publication_parent = runtime / "nested"
    output = publication_parent / "parent-race.json"
    displaced = runtime / "displaced-parent"
    competitor_bytes = b"competitor parent survives"
    prevented: list[OSError] = []

    def replace_parent(stage: str, target: Path) -> None:
        if stage != "artifacts-before-rename":
            return
        try:
            os.replace(publication_parent, displaced)
        except OSError as error:
            prevented.append(error)
            return
        publication_parent.mkdir()
        (publication_parent / "competitor.txt").write_bytes(competitor_bytes)

    monkeypatch.setattr(local_experiments, "_LOCAL_PUBLICATION_SEAM", replace_parent)
    arguments = dict(
        bundle=_node_bundle(),
        dataset_id="penn94",
        phase="dev",
        task_kind="node-binary",
        targets_by_entity={str(index): index % 2 for index in range(6)},
        split_inventory=_penn_split_inventory(_node_bundle()),
        source_inventory=_source_inventory(_node_bundle()),
        runtime_root=runtime,
        output_path=output,
        seed=7,
        optimizer_steps=1,
        head_steps=1,
        device_name="cpu",
        formal_preflight_evidence_hash=_formal_hash(runtime),
        formal_ready=False,
    )
    if os.name == "nt":
        report = run_local_nonpromotable_experiment(**arguments)
        assert prevented
        assert report.promotable is False
        assert output.is_file()
    else:
        with pytest.raises(ValueError, match="ancestor identity changed"):
            run_local_nonpromotable_experiment(**arguments)
        assert (publication_parent / "competitor.txt").read_bytes() == competitor_bytes
        assert not output.exists()


def test_local_artifact_staging_substitution_is_prevented_or_detected_without_deleting_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "staging-race.json"
    artifacts = output.with_suffix(".artifacts")
    displaced = runtime / "displaced-staging"
    competitor_bytes = b"competitor staging survives"
    prevented: list[OSError] = []

    def replace_staging(stage: str, target: Path) -> None:
        if stage != "artifacts-before-rename":
            return
        staging = next(target.parent.glob(f".{target.name}.*.staging"))
        try:
            os.replace(staging, displaced)
        except OSError as error:
            prevented.append(error)
            return
        staging.mkdir()
        (staging / "competitor.txt").write_bytes(competitor_bytes)

    monkeypatch.setattr(local_experiments, "_LOCAL_PUBLICATION_SEAM", replace_staging)
    arguments = dict(
        bundle=_node_bundle(),
        dataset_id="penn94",
        phase="dev",
        task_kind="node-binary",
        targets_by_entity={str(index): index % 2 for index in range(6)},
        split_inventory=_penn_split_inventory(_node_bundle()),
        source_inventory=_source_inventory(_node_bundle()),
        runtime_root=runtime,
        output_path=output,
        seed=7,
        optimizer_steps=1,
        head_steps=1,
        device_name="cpu",
        formal_preflight_evidence_hash=_formal_hash(runtime),
        formal_ready=False,
    )
    if os.name == "nt":
        report = run_local_nonpromotable_experiment(**arguments)
        assert prevented
        assert report.promotable is False
        assert artifacts.is_dir()
    else:
        with pytest.raises(ValueError, match="no longer published|identity changed"):
            run_local_nonpromotable_experiment(**arguments)
        competitor = next(runtime.glob(f".{artifacts.name}.*.staging"))
        assert (competitor / "competitor.txt").read_bytes() == competitor_bytes
        assert not artifacts.exists()
        assert not output.exists()


def test_local_report_post_link_substitution_fails_without_deleting_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "substituted.json"
    competitor_bytes = b'{"competitor":true}\n'

    def substitute(stage: str, target: Path) -> None:
        if stage == "evidence-post-link" and target == output:
            competitor = runtime / "competitor.json"
            competitor.write_bytes(competitor_bytes)
            os.replace(competitor, target)

    monkeypatch.setattr(formal_preflight, "_PUBLICATION_SEAM", substitute)
    with pytest.raises(ValueError, match="identity changed"):
        run_local_nonpromotable_experiment(
            bundle=_node_bundle(),
            dataset_id="penn94",
            phase="dev",
            task_kind="node-binary",
            targets_by_entity={str(index): index % 2 for index in range(6)},
            split_inventory=_penn_split_inventory(_node_bundle()),
            source_inventory=_source_inventory(_node_bundle()),
            runtime_root=runtime,
            output_path=output,
            seed=7,
            optimizer_steps=1,
            head_steps=1,
            device_name="cpu",
            formal_preflight_evidence_hash=_formal_hash(runtime),
            formal_ready=False,
        )

    assert output.read_bytes() == competitor_bytes


def test_local_run_rejects_a_true_formal_readiness_override_before_writing(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "forbidden-formal-claim.json"

    with pytest.raises(ValueError, match="local runs are never formal-ready"):
        run_local_nonpromotable_experiment(
            bundle=_node_bundle(),
            dataset_id="penn94",
            phase="dev",
            task_kind="node-binary",
            targets_by_entity={str(index): index % 2 for index in range(6)},
            split_inventory=_penn_split_inventory(_node_bundle()),
            source_inventory=_source_inventory(_node_bundle()),
            runtime_root=runtime,
            output_path=output,
            seed=7,
            optimizer_steps=1,
            head_steps=1,
            device_name="cpu",
            formal_preflight_evidence_hash=_formal_hash(runtime),
            formal_ready=True,
        )

    assert not output.exists()
    assert not output.with_suffix(".artifacts").exists()


def test_code_inventory_hash_binds_every_behavior_defining_category(tmp_path: Path) -> None:
    relative_paths = (
        "__init__.py",
        "canonical.py",
        "contracts.py",
        "errors.py",
        "public_contracts.py",
        "tensor_digest.py",
        "core/__init__.py",
        "core/adapters.py",
        "core/bundle.py",
        "core/calibration.py",
        "core/checkpoint.py",
        "core/config.py",
        "core/experiment_cli.py",
        "core/experiment_data.py",
        "core/fold_recovery.py",
        "core/formal_materialization.py",
        "core/formal_preflight.py",
        "core/graph_ops.py",
        "core/local_experiments.py",
        "core/local_recovery.py",
        "core/model.py",
        "core/objectives.py",
        "core/safe_paths.py",
        "core/serving_registry.py",
        "core/splits.py",
        "core/structure_features.py",
        "core/supervised.py",
        "core/trainer.py",
        "core/training_data.py",
        "core/datasets/__init__.py",
        "core/datasets/acquire.py",
        "core/datasets/mat_worker.py",
        "core/datasets/materialize.py",
        "core/datasets/parsers.py",
        "core/datasets/penn94_conversion.py",
        "core/datasets/recipes.py",
        "core/datasets/recipes.json",
    )
    assert local_experiments._CODE_INVENTORY_RELATIVE_PATHS == relative_paths
    source_root = Path(local_experiments.__file__).resolve().parents[1]
    for relative in local_experiments._CODE_INVENTORY_RELATIVE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source_root / relative).read_bytes())

    baseline = local_experiments._code_inventory(tmp_path, relative_paths=relative_paths)
    assert baseline["schemaVersion"] == "socialgraph-fm.core-local-code-inventory/2.0"
    assert tuple(item["relativePath"] for item in baseline["files"]) == relative_paths
    local_experiments._validate_code_inventory_document(
        baseline,
        root=tmp_path,
        relative_paths=relative_paths,
    )

    for relative in relative_paths:
        path = tmp_path / relative
        original = path.read_bytes()
        path.write_bytes(original + b":changed")
        with pytest.raises(ValueError, match="actual behavior file"):
            local_experiments._validate_code_inventory_document(
                baseline,
                root=tmp_path,
                relative_paths=relative_paths,
            )
        path.write_bytes(original)


def test_local_self_hashed_inventory_validators_reject_unknown_fields() -> None:
    code = local_experiments._code_inventory()
    code["attackerUnknown"] = True
    code.pop("inventoryHash")
    code["inventoryHash"] = canonical_sha256(code)
    with pytest.raises(ValueError, match="code inventory contract"):
        local_experiments._validate_code_inventory_document(code)

    targets = local_experiments._target_inventory({"0": 0, "1": 1})
    targets["attackerUnknown"] = True
    targets.pop("inventoryHash")
    targets["inventoryHash"] = canonical_sha256(targets)
    with pytest.raises(ValueError, match="target inventory"):
        local_experiments._validate_target_inventory(targets)

    source = _source_inventory(_node_bundle())
    source["attackerUnknown"] = True
    source.pop("inventoryHash")
    source["inventoryHash"] = canonical_sha256(source)
    with pytest.raises(ValueError, match="source inventory"):
        local_experiments._validate_source_inventory(
            source,
            dataset_id="penn94",
            source_sha256=_node_bundle().source.source_sha256,
        )

    environment = local_experiments.local_environment_inventory("cpu").model_dump(
        mode="python", by_alias=True
    )
    environment["attackerUnknown"] = True
    environment.pop("inventoryHash")
    environment["inventoryHash"] = canonical_sha256(environment)
    with pytest.raises(ValidationError, match="extra"):
        local_experiments.validate_local_environment_inventory(environment)

    unavailable_calibration = {
        "schemaVersion": "socialgraph-fm.core-local-calibration-evidence/1.0",
        "status": "unavailable-single-class-validation",
    }
    unavailable_calibration["evidenceHash"] = canonical_sha256(unavailable_calibration)
    unavailable_calibration["attackerUnknown"] = True
    unavailable_calibration.pop("evidenceHash")
    unavailable_calibration["evidenceHash"] = canonical_sha256(unavailable_calibration)
    with pytest.raises(ValueError, match="unavailable calibration evidence is not exact"):
        local_experiments._validate_unavailable_calibration_document(unavailable_calibration)


def test_code_inventory_reopen_rehashes_the_actual_behavior_files(tmp_path: Path) -> None:
    relative_paths = ("training_data.py", "datasets/parsers.py")
    for relative in relative_paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"original:{relative}".encode())
    inventory = local_experiments._code_inventory(tmp_path, relative_paths=relative_paths)

    local_experiments._validate_code_inventory_document(
        inventory,
        root=tmp_path,
        relative_paths=relative_paths,
    )
    (tmp_path / relative_paths[0]).write_bytes(b"post-report mutation")

    with pytest.raises(ValueError, match="actual behavior file"):
        local_experiments._validate_code_inventory_document(
            inventory,
            root=tmp_path,
            relative_paths=relative_paths,
        )


def test_penn_local_split_inventory_rejects_reordering_and_replacement() -> None:
    manifests = []
    for fold_index in range(5):
        payload = _node_bundle().split_manifest.model_dump(mode="json", by_alias=True)
        assignments = payload["assignments"]
        assert isinstance(assignments, list)
        assignments[0]["role"] = "train" if fold_index % 2 else "test"
        manifests.append(payload)

    inventory = local_experiments.LocalSplitInventory.create(
        dataset_id="penn94",
        manifests=tuple(manifests),
        selected_fold_id="fold-0",
    )
    assert inventory.fold_ids == ("fold-0", "fold-1", "fold-2", "fold-3", "fold-4")
    assert inventory.selected_fold_id == "fold-0"

    reordered = inventory.model_dump(mode="json", by_alias=True)
    reordered["folds"] = list(reversed(reordered["folds"]))
    with pytest.raises(ValidationError, match="ordered fold-0 through fold-4"):
        local_experiments.LocalSplitInventory.model_validate(reordered)

    replaced = inventory.model_dump(mode="json", by_alias=True)
    replaced["folds"][2]["splitManifest"]["assignments"][0]["role"] = "unlabeled"
    with pytest.raises(ValidationError, match="splitManifestHash"):
        local_experiments.LocalSplitInventory.model_validate(replaced)


def test_local_recovery_binds_the_exact_config_derived_neighbor_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovery must recreate the non-default neighbor policy, not silently default it."""

    def _dev_with_policy(cls: type[TrainingConfig], *, max_steps: int = 2_000) -> TrainingConfig:
        return cls(
            preset="dev",
            min_steps=0,
            max_steps=max_steps,
            full_batch_edge_threshold=1,
            node_batch_size=3,
            edge_batch_size=3,
            fanout=(2, 1, 0),
        )

    monkeypatch.setattr(TrainingConfig, "dev", classmethod(_dev_with_policy))
    runtime = tmp_path / "runtime"
    runtime.mkdir()

    report = run_local_nonpromotable_experiment(
        bundle=_node_bundle(),
        dataset_id="penn94",
        phase="dev",
        task_kind="node-binary",
        targets_by_entity={str(index): index % 2 for index in range(6)},
        split_inventory=_penn_split_inventory(_node_bundle()),
        source_inventory=_source_inventory(_node_bundle()),
        runtime_root=runtime,
        output_path=runtime / "neighbor-policy.json",
        seed=20260821,
        optimizer_steps=1,
        head_steps=1,
        device_name="cpu",
        formal_preflight_evidence_hash=_formal_hash(runtime),
        formal_ready=False,
    )

    assert report.optimizer_steps == 1
    assert report.fresh_recovery_state_hash
    assert report.config_hash == canonical_sha256(
        {
            "phase": "dev",
            "seed": 20260821,
            "training": TrainingConfig.dev(max_steps=1).to_dict(),
        }
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_neighbor_training_keeps_membership_and_candidates_on_one_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        TrainingConfig,
        "dev",
        classmethod(
            lambda cls, *, max_steps=2_000: cls(
                preset="dev",
                min_steps=0,
                max_steps=max_steps,
                full_batch_edge_threshold=1,
                node_batch_size=3,
                edge_batch_size=3,
                fanout=(2, 1, 0),
            )
        ),
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    report = run_local_nonpromotable_experiment(
        bundle=_node_bundle(),
        dataset_id="penn94",
        phase="dev",
        task_kind="node-binary",
        targets_by_entity={str(index): index % 2 for index in range(6)},
        split_inventory=_penn_split_inventory(_node_bundle()),
        source_inventory=_source_inventory(_node_bundle()),
        runtime_root=runtime,
        output_path=runtime / "cuda-dev.json",
        seed=41,
        optimizer_steps=1,
        head_steps=1,
        device_name="cuda",
        formal_preflight_evidence_hash=_formal_hash(runtime),
        formal_ready=False,
    )

    assert report.device == "cuda"
    assert report.evaluation_device == "cpu"
    assert report.checkpoint_promotable is False
    assert report.encoded_artifact_hash
