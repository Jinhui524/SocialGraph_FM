from __future__ import annotations

import json
from pathlib import Path

import pytest

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.core.bundle import (
    CoreGraphBundle,
    calculate_graph_version_hash,
)
from socialgraph_gfm.core.experiment_cli import main
from socialgraph_gfm.core.formal_preflight import FormalPreflightEvidence
from socialgraph_gfm.core.local_experiments import (
    LocalDatasetInputs,
    LocalExperimentRun,
    LocalSplitInventory,
)
from socialgraph_gfm.core.structure_features import StructureCacheManifest


def _tiny_bundle() -> CoreGraphBundle:
    payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [
            {"id": "a", "index": 0},
            {"id": "b", "index": 1},
            {"id": "c", "index": 2},
        ],
        "edges": [
            {"sourceId": "a", "targetId": "b", "edgeType": "social", "weight": 1.0},
            {"sourceId": "b", "targetId": "c", "edgeType": "social", "weight": 1.0},
        ],
        "nodeFeatures": [],
        "structuralFeatures": None,
        "source": {
            "sourceName": "tiny-cli-fixture",
            "sourceSha256": "a" * 64,
        },
        "splitManifest": {
            "strategy": "official",
            "assignments": [
                {"entityId": "a", "role": "train"},
                {"entityId": "b", "role": "validation"},
                {"entityId": "c", "role": "test"},
            ],
        },
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def test_preflight_command_publishes_real_nonready_evidence_without_downloading(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = runtime / "experiments-core" / "preflight.json"

    assert (
        main(
            [
                "preflight",
                "--runtime-root",
                str(runtime),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    evidence = FormalPreflightEvidence.model_validate_json(output.read_bytes())
    assert evidence.formal_ready is False
    assert evidence.promotable is False
    assert not (runtime / "raw").exists()
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "evidenceHash": evidence.evidence_hash,
        "formalReady": False,
        "output": str(output.resolve()),
        "promotable": False,
    }


def test_every_public_command_requires_an_explicit_runtime_root() -> None:
    for command in (
        "preflight",
        "structure-cache",
        "email-smoke",
        "penn-dev",
        "aggregate",
        "accept",
        "serving-smoke",
        "promote",
        "readiness",
    ):
        with pytest.raises(SystemExit):
            main([command])


def test_email_and_penn_commands_fail_closed_without_local_assets(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for command in ("email-smoke", "penn-dev"):
        with pytest.raises(SystemExit, match="local dataset assets"):
            main(
                [
                    command,
                    "--runtime-root",
                    str(runtime),
                    "--output",
                    str(runtime / f"{command}.json"),
                ]
            )
    assert tuple(runtime.iterdir()) == ()


def test_structure_cache_command_builds_and_exact_reopens_hash_bound_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    bundle = _tiny_bundle()
    bundle_path = runtime / "bundle.json"
    bundle_path.write_text(canonical_json(bundle) + "\n", encoding="utf-8")

    assert (
        main(
            [
                "structure-cache",
                "--runtime-root",
                str(runtime),
                "--bundle",
                str(bundle_path),
                "--role",
                "training",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    manifest_path = Path(summary["manifestPath"])
    manifest = StructureCacheManifest.model_validate_json(manifest_path.read_bytes())
    assert summary == {
        "artifactId": manifest.artifact_id,
        "baseGraphVersionHash": bundle.graph_version_hash,
        "manifestHash": manifest.manifest_hash,
        "manifestPath": str(manifest_path.resolve()),
        "role": "training",
    }
    assert manifest_path.is_relative_to(runtime.resolve())


def test_penn_dev_command_runs_real_nonpromotable_pipeline_from_bound_local_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    preflight = runtime / "experiments-core" / "formal-preflight-v2-current.json"
    assert main(["preflight", "--runtime-root", str(runtime), "--output", str(preflight)]) == 0
    capsys.readouterr()
    bundle = _tiny_bundle().model_dump(mode="json", by_alias=True)
    bundle["splitManifest"] = {
        "strategy": "official",
        "assignments": [
            {"entityId": "a", "role": "train"},
            {"entityId": "b", "role": "train"},
            {"entityId": "c", "role": "validation"},
        ],
    }
    bundle["graphVersionHash"] = calculate_graph_version_hash(bundle)
    local_bundle = CoreGraphBundle.model_validate(bundle)
    split_inventory = LocalSplitInventory.create(
        dataset_id="penn94",
        manifests=(local_bundle.split_manifest,) * 5,
        selected_fold_id="fold-0",
    )
    source_inventory: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-local-source-inventory/1.0",
        "datasetId": "penn94",
        "sourceSha256": local_bundle.source.source_sha256,
        "scope": "test-fixture",
        "files": (),
        "conversionManifestHash": "b" * 64,
        "splitInventoryHash": split_inventory.inventory_hash,
    }
    source_inventory["inventoryHash"] = canonical_sha256(source_inventory)
    local_inputs = LocalDatasetInputs(
        bundle=local_bundle,
        targets_by_entity={"a": 0, "b": 1, "c": 0},
        split_inventory=split_inventory,
        source_inventory=source_inventory,
    )
    monkeypatch.setattr(
        "socialgraph_gfm.core.experiment_cli._require_local_run_assets",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "socialgraph_gfm.core.experiment_cli.load_penn94_local_inputs",
        lambda _root: local_inputs,
    )
    monkeypatch.setattr(
        "socialgraph_gfm.core.local_experiments.load_penn94_local_inputs",
        lambda _root: local_inputs,
    )
    output = runtime / "experiments-core" / "local" / "penn-dev.json"

    assert (
        main(
            [
                "penn-dev",
                "--runtime-root",
                str(runtime),
                "--output",
                str(output),
                "--optimizer-steps",
                "1",
                "--head-steps",
                "1",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    report = LocalExperimentRun.model_validate_json(output.read_bytes())
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "datasetId": "penn94",
        "device": "cpu",
        "formalReady": False,
        "output": str(output.resolve()),
        "phase": "dev",
        "promotable": False,
        "reportHash": report.report_hash,
    }
