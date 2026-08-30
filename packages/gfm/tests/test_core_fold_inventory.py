from __future__ import annotations

import pytest

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.fold_inventory import (
    CellFoldEvaluationInventory,
    FoldEvaluationBinding,
    FoldRuntimeArtifactRef,
)


def _hash(value: int) -> str:
    return f"{value:064x}"


def _ref(role: str, fold_id: str, index: int) -> FoldRuntimeArtifactRef:
    byte_hash = _hash(index)
    return FoldRuntimeArtifactRef(
        role=role,
        relativePath=f"fold-runs/cell/{fold_id}/{role}.bin",
        byteSha256=byte_hash,
        semanticHash=(
            byte_hash if role in {"latest-checkpoint", "best-checkpoint"} else _hash(index + 100)
        ),
        sizeBytes=index + 1,
    )


def _binding(fold_id: str, *, binary: bool = True) -> FoldEvaluationBinding:
    roles = [
        "best-checkpoint",
        "head-data",
        "head-report",
        "latest-checkpoint",
        "predictions",
        "resource-telemetry",
        "telemetry-receipt",
    ]
    if binary:
        roles.extend(("calibration-report", "threshold"))
    refs = tuple(_ref(role, fold_id, index + 1) for index, role in enumerate(sorted(roles)))
    payload = {
        "schemaVersion": "socialgraph-fm.core-fold-evaluation-binding/1.0",
        "foldId": fold_id,
        "runtimeKind": "core-gfm",
        "taskKind": "node-binary" if binary else "resilience-regression",
        "splitManifestHash": _hash(50 + int(fold_id[-1])),
        "preparedGraphVersionHash": _hash(60 + int(fold_id[-1])),
        "foldDataHash": _hash(70 + int(fold_id[-1])),
        "adapterDomain": f"tolokers::{fold_id}",
        "artifacts": [item.model_dump(mode="python", by_alias=True) for item in refs],
    }
    payload["bindingHash"] = canonical_sha256(payload)
    return FoldEvaluationBinding.model_validate(payload)


def test_inventory_binds_exact_fold_runtime_and_is_order_deterministic() -> None:
    folds = (_binding("official-00"), _binding("official-01"))
    payload = {
        "schemaVersion": "socialgraph-fm.core-cell-fold-evaluation/1.0",
        "cellId": _hash(1),
        "taskId": "tolokers.risk",
        "datasetManifestHash": _hash(2),
        "splitInventoryHash": _hash(3),
        "labelsHash": _hash(4),
        "targetName": "banned",
        "foldIds": ["official-00", "official-01"],
        "folds": [item.model_dump(mode="python", by_alias=True) for item in folds],
    }
    payload["inventoryHash"] = canonical_sha256(payload)

    inventory = CellFoldEvaluationInventory.model_validate(payload)

    assert tuple(item.fold_id for item in inventory.folds) == inventory.fold_ids
    assert inventory.folds[0].artifacts[0].role == "best-checkpoint"


@pytest.mark.parametrize("mutation", ["missing-role", "duplicate-path", "wrong-fold-order"])
def test_inventory_rejects_incomplete_or_ambiguous_fold_runtime(mutation: str) -> None:
    first = _binding("official-00")
    second = _binding("official-01")
    if mutation == "missing-role":
        raw = first.model_dump(mode="python", by_alias=True)
        raw["artifacts"] = raw["artifacts"][:-1]
        raw["bindingHash"] = canonical_sha256(
            {key: value for key, value in raw.items() if key != "bindingHash"}
        )
        with pytest.raises(ValueError, match="artifact inventory"):
            FoldEvaluationBinding.model_validate(raw)
        return
    if mutation == "duplicate-path":
        raw = first.model_dump(mode="python", by_alias=True)
        raw["artifacts"][1]["relativePath"] = raw["artifacts"][0]["relativePath"]
        raw["bindingHash"] = canonical_sha256(
            {key: value for key, value in raw.items() if key != "bindingHash"}
        )
        with pytest.raises(ValueError, match="artifact inventory"):
            FoldEvaluationBinding.model_validate(raw)
        return
    payload = {
        "schemaVersion": "socialgraph-fm.core-cell-fold-evaluation/1.0",
        "cellId": _hash(1),
        "taskId": "tolokers.risk",
        "datasetManifestHash": _hash(2),
        "splitInventoryHash": _hash(3),
        "labelsHash": _hash(4),
        "targetName": "banned",
        "foldIds": ["official-00", "official-01"],
        "folds": [
            second.model_dump(mode="python", by_alias=True),
            first.model_dump(mode="python", by_alias=True),
        ],
    }
    payload["inventoryHash"] = canonical_sha256(payload)
    with pytest.raises(ValueError, match="fold inventory"):
        CellFoldEvaluationInventory.model_validate(payload)


def test_resilience_binding_rejects_binary_only_artifacts() -> None:
    binding = _binding("official-00", binary=False)
    raw = binding.model_dump(mode="python", by_alias=True)
    raw["artifacts"] = list(raw["artifacts"])
    raw["artifacts"].append(
        _ref("calibration-report", "official-00", 20).model_dump(mode="python", by_alias=True)
    )
    raw["artifacts"].sort(key=lambda item: item["role"])
    raw["bindingHash"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "bindingHash"}
    )
    with pytest.raises(ValueError, match="artifact inventory"):
        FoldEvaluationBinding.model_validate(raw)
