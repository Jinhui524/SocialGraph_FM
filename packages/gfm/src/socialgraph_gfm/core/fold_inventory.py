"""Immutable per-fold runtime inventory for formal core evaluation."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_sha256

from .fold_evaluation import FoldTaskKind


_HASH = r"^[0-9a-f]{64}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


def _safe_relative_path(value: str) -> str:
    if "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("fold artifact path must use safe POSIX-relative syntax")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError("fold artifact path must be a safe relative path")
    return parsed.as_posix()


class FoldRuntimeArtifactRef(_StrictModel):
    role: Literal[
        "baseline-input",
        "baseline-report",
        "best-checkpoint",
        "calibration-report",
        "head-data",
        "head-report",
        "latest-checkpoint",
        "predictions",
        "resource-telemetry",
        "telemetry-receipt",
        "threshold",
    ]
    relative_path: str = Field(alias="relativePath")
    byte_sha256: str = Field(alias="byteSha256", pattern=_HASH)
    semantic_hash: str = Field(alias="semanticHash", pattern=_HASH)
    size_bytes: int = Field(alias="sizeBytes", gt=0)

    @model_validator(mode="after")
    def validate_reference(self):
        if _safe_relative_path(self.relative_path) != self.relative_path:
            raise ValueError("fold artifact path is not canonical")
        if self.role in {"latest-checkpoint", "best-checkpoint"} and (
            self.semantic_hash != self.byte_sha256
        ):
            raise ValueError("fold checkpoint semantic identity must equal its exact bytes")
        return self


class FoldEvaluationBinding(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-fold-evaluation-binding/1.0"] = Field(
        alias="schemaVersion"
    )
    fold_id: str = Field(alias="foldId", min_length=1)
    runtime_kind: Literal["core-gfm", "baseline", "heuristic"] = Field(alias="runtimeKind")
    task_kind: FoldTaskKind = Field(alias="taskKind")
    split_manifest_hash: str = Field(alias="splitManifestHash", pattern=_HASH)
    prepared_graph_version_hash: str = Field(alias="preparedGraphVersionHash", pattern=_HASH)
    fold_data_hash: str = Field(alias="foldDataHash", pattern=_HASH)
    adapter_domain: str | None = Field(default=None, alias="adapterDomain", min_length=1)
    artifacts: tuple[FoldRuntimeArtifactRef, ...] = Field(strict=False, min_length=1)
    binding_hash: str = Field(alias="bindingHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_binding(self):
        roles = tuple(item.role for item in self.artifacts)
        expected: set[str]
        if self.runtime_kind == "core-gfm":
            expected = {
                "best-checkpoint",
                "head-data",
                "head-report",
                "latest-checkpoint",
                "predictions",
                "resource-telemetry",
                "telemetry-receipt",
            }
            if self.task_kind != "resilience-regression":
                expected.add("calibration-report")
            if self.task_kind in {"node-binary", "signed-edge"}:
                expected.add("threshold")
            if self.adapter_domain is None:
                raise ValueError("core-gfm fold runtime requires an adapter domain")
        elif self.runtime_kind == "baseline":
            expected = {
                "baseline-input",
                "baseline-report",
                "best-checkpoint",
                "latest-checkpoint",
                "predictions",
                "resource-telemetry",
                "telemetry-receipt",
            }
            if self.task_kind in {"node-binary", "signed-edge"}:
                expected.add("threshold")
            if self.adapter_domain is not None:
                raise ValueError("baseline fold runtime cannot claim a GFM adapter domain")
        else:
            expected = {"predictions"}
            if self.adapter_domain is not None:
                raise ValueError("heuristic fold runtime cannot claim a GFM adapter domain")
        if (
            roles != tuple(sorted(expected))
            or len(set(roles)) != len(roles)
            or len({item.relative_path for item in self.artifacts}) != len(self.artifacts)
        ):
            raise ValueError("fold runtime artifact inventory is incomplete or ambiguous")
        expected_hash = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"binding_hash"})
        )
        if self.binding_hash != expected_hash:
            raise ValueError("bindingHash does not match the fold runtime inventory")
        return self


class CellFoldEvaluationInventory(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-cell-fold-evaluation/1.0"] = Field(
        alias="schemaVersion"
    )
    cell_id: str = Field(alias="cellId", pattern=_HASH)
    task_id: str = Field(alias="taskId", min_length=1)
    dataset_manifest_hash: str = Field(alias="datasetManifestHash", pattern=_HASH)
    split_inventory_hash: str = Field(alias="splitInventoryHash", pattern=_HASH)
    labels_hash: str = Field(alias="labelsHash", pattern=_HASH)
    target_name: str = Field(alias="targetName", min_length=1)
    fold_ids: tuple[str, ...] = Field(alias="foldIds", strict=False, min_length=1)
    folds: tuple[FoldEvaluationBinding, ...] = Field(strict=False, min_length=1)
    inventory_hash: str = Field(alias="inventoryHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_inventory(self):
        observed_ids = tuple(item.fold_id for item in self.folds)
        if (
            self.fold_ids != tuple(sorted(set(self.fold_ids)))
            or observed_ids != self.fold_ids
            or len({item.binding_hash for item in self.folds}) != len(self.folds)
            or len({ref.relative_path for item in self.folds for ref in item.artifacts})
            != sum(len(item.artifacts) for item in self.folds)
            or len({item.task_kind for item in self.folds}) != 1
            or len({item.runtime_kind for item in self.folds}) != 1
        ):
            raise ValueError("cell fold inventory is incomplete, unordered, or ambiguous")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"inventory_hash"})
        )
        if self.inventory_hash != expected:
            raise ValueError("inventoryHash does not match cell fold evaluation")
        return self


__all__ = [
    "CellFoldEvaluationInventory",
    "FoldEvaluationBinding",
    "FoldRuntimeArtifactRef",
]
