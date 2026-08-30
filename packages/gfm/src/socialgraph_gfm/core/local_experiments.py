"""Real, deliberately non-promotable Email/Penn local experiment evidence."""

from __future__ import annotations

import hashlib
import json
import ctypes
import gzip
import os
import random
import subprocess
import sys
import sysconfig
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.tensor_digest import canonical_tensor_digest

from . import formal_preflight as _formal_fs
from .adapters import AdapterSchema, BundleInputAdapter, derive_training_selection
from .bundle import (
    SourceProvenance,
    SplitManifest,
    CoreGraphBundle,
    calculate_graph_version_hash,
    load_core_graph_bundle_json,
)
from .calibration import (
    BinaryScoreSemantics,
    CalibrationFitReport,
    CalibrationProtocol,
    derive_validation_scores,
    fit_score_calibration_report,
)
from .checkpoint import CheckpointBindings, load_checkpoint
from .config import TrainingConfig
from .datasets.parsers import parse_email_files, parse_facebook100_mat
from .datasets.penn94_conversion import (
    PENN94_CONVERTER_VERSION,
    PENN94_DATA_SHA256,
    PENN94_LABELED_NODE_COUNT,
    PENN94_LINKX_COMMIT,
    PENN94_RAW_SPLIT_MAX_BYTES,
    PENN94_RAW_SPLIT_SHA256,
    PENN94_RAW_SPLIT_URL,
    PENN94_SPLIT_COUNTS,
    verify_penn94_raw_split,
)
from .datasets.recipes import load_dataset_recipes
from .experiment_data import bundle_from_parsed_graph
from .local_recovery import (
    LOCAL_CODE_INVENTORY_RELATIVE_PATHS,
    LocalRecoveryReceipt,
    local_code_inventory,
    local_environment_inventory,
    validate_local_code_inventory,
    validate_local_environment_inventory,
)
from .model import CoreGFM
from .formal_preflight import (
    FormalPreflightEvidence,
    _OwnedChildProof,
    _OwnedDirectoryLease,
    _OwnedFileLease,
    _PublicationParentLease,
    _PublisherLock,
    _path_identity,
    _publish_exact,
    _rename_directory_no_replace,
    load_formal_preflight,
)
from .fold_recovery import _composite_hash, _state_hash as _trainer_state_hash
from .safe_paths import read_confined_snapshot, reject_link_components, secure_existing_root
from .structure_features import (
    StructureCacheManifest,
    build_structure_cache,
    enrich_bundle_with_structure,
    load_structure_cache,
)
from .splits import spanning_forest_link_split
from .supervised import (
    HeadTrainingReport,
    HeadTrainingConfig,
    SupervisedPartition,
    SupervisedTrainValidation,
    _new_verified_head_training_report,
    derive_encoder_identity,
    encode_supervised_graph,
    fit_supervised_head,
    verify_head_training_report,
)
from .trainer import CoreTrainer, TrainingGraph
from .training_data import ExecutionPolicy, PreparedGraph


_SCHEMA = "socialgraph-fm.core-local-run/3.0"
_HEAD_ARTIFACT_SCHEMA = "socialgraph-fm.core-local-head-artifact/1.0"
_CODE_INVENTORY_RELATIVE_PATHS = LOCAL_CODE_INVENTORY_RELATIVE_PATHS
_EMAIL_RAW_SOURCE_SHA256 = {
    "edges": "4b47acdb80197b085fe63c819c357ae488131ee904ed93d1b219a68b0f9e245f",
    "departments": "e5abe5b4581a480032a63adcf2576c161785f45692642c6ebb0b1276f0f33669",
}
_PENN94_SAFE_SPLIT_SHA256 = "46ead6a2c5ba5987e63502e543a643be29f0e467493bb63323989d55f8ee0139"
_LOCAL_ARTIFACT_PAYLOAD_NAMES = (
    "adapter-evidence.json",
    "base-bundle.json",
    "calibration-report.json",
    "code-inventory.json",
    "core-training.pt",
    "cpu-evaluation-state.pt",
    "environment-inventory.json",
    "head-evaluation.pt",
    "head-training-report.json",
    "publication-control.json",
    "recovery-bundle.json",
    "recovery-receipt.json",
    "recovery-request.json",
    "source-inventory.json",
    "split-inventory.json",
    "target-inventory.json",
)
_LOCAL_ARTIFACT_CONTROL_NAMES = ("artifact-inventory.json", "report.json")


def _LOCAL_PUBLICATION_SEAM(_stage: str, _path: Path) -> None:
    return


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


@dataclass(frozen=True)
class LocalDatasetInputs:
    bundle: CoreGraphBundle
    targets_by_entity: Mapping[str, int]
    split_inventory: LocalSplitInventory
    source_inventory: Mapping[str, Any]


class LocalSplitFold(_StrictModel):
    fold_id: str = Field(alias="foldId", pattern=r"^fold-[0-9]+$")
    split_manifest: SplitManifest = Field(alias="splitManifest")
    split_manifest_hash: str = Field(alias="splitManifestHash", pattern=r"^[0-9a-f]{64}$")
    role_counts: dict[Literal["train", "validation", "test", "unlabeled"], int] = Field(
        alias="roleCounts"
    )

    @model_validator(mode="after")
    def validate_fold(self):
        expected_hash = canonical_sha256(
            self.split_manifest.model_dump(mode="python", by_alias=True)
        )
        if self.split_manifest_hash != expected_hash:
            raise ValueError("splitManifestHash does not match the official split manifest")
        expected_counts = {role: 0 for role in ("train", "validation", "test", "unlabeled")}
        for assignment in self.split_manifest.assignments:
            expected_counts[assignment.role] += 1
        if self.role_counts != expected_counts:
            raise ValueError("roleCounts do not match the official split manifest")
        return self


class LocalSplitInventory(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-local-split-inventory/1.0"] = Field(
        alias="schemaVersion"
    )
    dataset_id: Literal["email-eu-core", "penn94"] = Field(alias="datasetId")
    folds: tuple[LocalSplitFold, ...] = Field(strict=False, min_length=1, max_length=5)
    selected_fold_id: str = Field(alias="selectedFoldId", pattern=r"^fold-[0-9]+$")
    inventory_hash: str = Field(alias="inventoryHash", pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        dataset_id: Literal["email-eu-core", "penn94"],
        manifests: tuple[SplitManifest | Mapping[str, Any], ...],
        selected_fold_id: str,
    ) -> LocalSplitInventory:
        folds = []
        for index, raw_manifest in enumerate(manifests):
            manifest = (
                raw_manifest
                if isinstance(raw_manifest, SplitManifest)
                else SplitManifest.model_validate(raw_manifest)
            )
            counts = {role: 0 for role in ("train", "validation", "test", "unlabeled")}
            for assignment in manifest.assignments:
                counts[assignment.role] += 1
            folds.append(
                {
                    "foldId": f"fold-{index}",
                    "splitManifest": manifest.model_dump(mode="python", by_alias=True),
                    "splitManifestHash": canonical_sha256(
                        manifest.model_dump(mode="python", by_alias=True)
                    ),
                    "roleCounts": counts,
                }
            )
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-local-split-inventory/1.0",
            "datasetId": dataset_id,
            "folds": folds,
            "selectedFoldId": selected_fold_id,
        }
        payload["inventoryHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @property
    def fold_ids(self) -> tuple[str, ...]:
        return tuple(fold.fold_id for fold in self.folds)

    @model_validator(mode="after")
    def validate_inventory(self):
        expected_ids = (
            ("fold-0", "fold-1", "fold-2", "fold-3", "fold-4")
            if self.dataset_id == "penn94"
            else ("fold-0",)
        )
        if self.fold_ids != expected_ids:
            raise ValueError("Penn inventory requires ordered fold-0 through fold-4")
        if self.selected_fold_id != "fold-0":
            raise ValueError("local smoke/dev may select only explicit fold-0")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"inventory_hash"})
        )
        if self.inventory_hash != expected:
            raise ValueError("inventoryHash does not match the official five-fold inventory")
        return self


class EvidenceReference(_StrictModel):
    relative_path: str = Field(alias="relativePath", min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_hash: str = Field(alias="semanticHash", pattern=r"^[0-9a-f]{64}$")


class LocalExperimentRun(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-local-run/3.0"] = Field(alias="schemaVersion")
    run_id: str = Field(alias="runId", pattern=r"^[0-9a-f]{64}$")
    dataset_id: Literal["email-eu-core", "penn94"] = Field(alias="datasetId")
    phase: Literal["smoke", "dev"]
    task_kind: Literal["node-binary", "edge-binary"] = Field(alias="taskKind")
    seed: int
    device: Literal["cpu", "cuda"]
    evaluation_device: Literal["cpu"] = Field(alias="evaluationDevice")
    optimizer_steps: int = Field(alias="optimizerSteps", ge=1, le=2_000)
    head_steps: int = Field(alias="headSteps", ge=1, le=20)
    node_count: int = Field(alias="nodeCount", ge=1)
    edge_count: int = Field(alias="edgeCount", ge=1)
    base_graph_version_hash: str = Field(alias="baseGraphVersionHash", pattern=r"^[0-9a-f]{64}$")
    enriched_graph_version_hash: str = Field(
        alias="enrichedGraphVersionHash", pattern=r"^[0-9a-f]{64}$"
    )
    source_sha256: str = Field(alias="sourceSha256", pattern=r"^[0-9a-f]{64}$")
    targets_hash: str = Field(alias="targetsHash", pattern=r"^[0-9a-f]{64}$")
    split_inventory_hash: str = Field(alias="splitInventoryHash", pattern=r"^[0-9a-f]{64}$")
    fold_ids: tuple[str, ...] = Field(alias="foldIds", strict=False, min_length=1)
    selected_fold_id: str = Field(alias="selectedFoldId", min_length=1)
    selected_split_manifest_hash: str = Field(
        alias="selectedSplitManifestHash", pattern=r"^[0-9a-f]{64}$"
    )
    selected_split_role_counts: dict[str, int] = Field(alias="selectedSplitRoleCounts")
    structure_manifest_hash: str = Field(alias="structureManifestHash", pattern=r"^[0-9a-f]{64}$")
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(alias="configHash", pattern=r"^[0-9a-f]{64}$")
    data_hash: str = Field(alias="dataHash", pattern=r"^[0-9a-f]{64}$")
    code_hash: str = Field(alias="codeHash", pattern=r"^[0-9a-f]{64}$")
    environment_hash: str = Field(alias="environmentHash", pattern=r"^[0-9a-f]{64}$")
    code_inventory_evidence: EvidenceReference = Field(alias="codeInventoryEvidence")
    environment_evidence: EvidenceReference = Field(alias="environmentEvidence")
    source_inventory_evidence: EvidenceReference = Field(alias="sourceInventoryEvidence")
    split_inventory_evidence: EvidenceReference = Field(alias="splitInventoryEvidence")
    targets_evidence: EvidenceReference = Field(alias="targetsEvidence")
    base_bundle_evidence: EvidenceReference = Field(alias="baseBundleEvidence")
    adapter_evidence: EvidenceReference = Field(alias="adapterEvidence")
    head_report_evidence: EvidenceReference = Field(alias="headReportEvidence")
    calibration_evidence: EvidenceReference = Field(alias="calibrationEvidence")
    formal_preflight_evidence: EvidenceReference = Field(alias="formalPreflightEvidence")
    recovery_bundle_evidence: EvidenceReference = Field(alias="recoveryBundleEvidence")
    recovery_request_evidence: EvidenceReference = Field(alias="recoveryRequestEvidence")
    recovery_evaluation_evidence: EvidenceReference = Field(alias="recoveryEvaluationEvidence")
    recovery_receipt_evidence: EvidenceReference = Field(alias="recoveryReceiptEvidence")
    checkpoint_evidence: EvidenceReference = Field(alias="checkpointEvidence")
    head_artifact_evidence: EvidenceReference = Field(alias="headArtifactEvidence")
    structure_manifest_evidence: EvidenceReference = Field(alias="structureManifestEvidence")
    structure_npz_evidence: EvidenceReference = Field(alias="structureNpzEvidence")
    artifact_inventory_evidence: EvidenceReference = Field(alias="artifactInventoryEvidence")
    checkpoint_relative_path: str = Field(alias="checkpointRelativePath", min_length=1)
    checkpoint_sha256: str = Field(alias="checkpointSha256", pattern=r"^[0-9a-f]{64}$")
    checkpoint_status: Literal["training"] = Field(alias="checkpointStatus")
    checkpoint_promotable: Literal[False] = Field(alias="checkpointPromotable")
    fresh_composite_state_hash: str = Field(
        alias="freshCompositeStateHash", pattern=r"^[0-9a-f]{64}$"
    )
    fresh_recovery_state_hash: str = Field(
        alias="freshRecoveryStateHash", pattern=r"^[0-9a-f]{64}$"
    )
    recovery_process_id: int = Field(alias="recoveryProcessId", ge=1)
    recovery_parent_process_id: int = Field(alias="recoveryParentProcessId", ge=1)
    recovery_device: Literal["cpu"] = Field(alias="recoveryDevice")
    recovery_receipt_relative_path: str = Field(alias="recoveryReceiptRelativePath", min_length=1)
    recovery_receipt_sha256: str = Field(alias="recoveryReceiptSha256", pattern=r"^[0-9a-f]{64}$")
    recovery_receipt_hash: str = Field(alias="recoveryReceiptHash", pattern=r"^[0-9a-f]{64}$")
    supervised_data_hash: str = Field(alias="supervisedDataHash", pattern=r"^[0-9a-f]{64}$")
    encoded_artifact_hash: str = Field(alias="encodedArtifactHash", pattern=r"^[0-9a-f]{64}$")
    head_artifact_relative_path: str = Field(alias="headArtifactRelativePath", min_length=1)
    head_artifact_sha256: str = Field(alias="headArtifactSha256", pattern=r"^[0-9a-f]{64}$")
    head_report_hash: str = Field(alias="headReportHash", pattern=r"^[0-9a-f]{64}$")
    head_state_hash: str = Field(alias="headStateHash", pattern=r"^[0-9a-f]{64}$")
    head_promotion_eligible: Literal[False] = Field(alias="headPromotionEligible")
    calibration_report_hash: str | None = Field(
        default=None, alias="calibrationReportHash", pattern=r"^[0-9a-f]{64}$"
    )
    calibration_promotion_eligible: Literal[False] | None = Field(
        default=None, alias="calibrationPromotionEligible"
    )
    formal_preflight_evidence_hash: str = Field(
        alias="formalPreflightEvidenceHash", pattern=r"^[0-9a-f]{64}$"
    )
    formal_ready: Literal[False] = Field(alias="formalReady")
    promotable: Literal[False]
    failed_gates: tuple[Literal["phase-not-formal", "formal-corpus-not-ready"], ...] = Field(
        alias="failedGates", strict=False, min_length=1
    )
    limitations: tuple[str, ...] = Field(strict=False, min_length=2)
    report_hash: str = Field(alias="reportHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self):
        expected_gates = ("phase-not-formal", "formal-corpus-not-ready")
        if self.failed_gates != expected_gates:
            raise ValueError("failedGates do not match local/formal readiness evidence")
        if (self.calibration_report_hash is None) != (self.calibration_promotion_eligible is None):
            raise ValueError("calibration report and eligibility must be present together")
        if (
            self.targets_evidence.semantic_hash != self.targets_hash
            or self.formal_preflight_evidence.semantic_hash != self.formal_preflight_evidence_hash
            or self.structure_manifest_evidence.semantic_hash != self.structure_manifest_hash
            or self.checkpoint_evidence.relative_path != self.checkpoint_relative_path
            or self.checkpoint_evidence.sha256 != self.checkpoint_sha256
            or self.recovery_receipt_evidence.relative_path != self.recovery_receipt_relative_path
            or self.recovery_receipt_evidence.sha256 != self.recovery_receipt_sha256
            or self.recovery_receipt_evidence.semantic_hash != self.recovery_receipt_hash
            or self.head_artifact_evidence.relative_path != self.head_artifact_relative_path
            or self.head_artifact_evidence.sha256 != self.head_artifact_sha256
        ):
            raise ValueError("local artifact references differ from mirrored report fields")
        expected_run_identity = {
            "datasetId": self.dataset_id,
            "phase": self.phase,
            "taskKind": self.task_kind,
            "seed": self.seed,
            "configHash": self.config_hash,
            "dataHash": self.data_hash,
            "codeHash": self.code_hash,
            "environmentHash": self.environment_hash,
            "targetsHash": self.targets_hash,
            "codeInventoryEvidence": self.code_inventory_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "environmentEvidence": self.environment_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "sourceInventoryEvidence": self.source_inventory_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "splitInventoryEvidence": self.split_inventory_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "targetsEvidence": self.targets_evidence.model_dump(mode="python", by_alias=True),
            "baseBundleEvidence": self.base_bundle_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "adapterEvidence": self.adapter_evidence.model_dump(mode="python", by_alias=True),
            "headReportEvidence": self.head_report_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "calibrationEvidence": self.calibration_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "formalPreflightEvidence": self.formal_preflight_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "recoveryBundleEvidence": self.recovery_bundle_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "recoveryRequestEvidence": self.recovery_request_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "recoveryEvaluationEvidence": self.recovery_evaluation_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "recoveryReceiptEvidence": self.recovery_receipt_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "checkpointEvidence": self.checkpoint_evidence.model_dump(mode="python", by_alias=True),
            "headArtifactEvidence": self.head_artifact_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "structureManifestEvidence": self.structure_manifest_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "structureNpzEvidence": self.structure_npz_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "artifactInventoryEvidence": self.artifact_inventory_evidence.model_dump(
                mode="python", by_alias=True
            ),
            "formalPreflightEvidenceHash": self.formal_preflight_evidence_hash,
            "splitInventoryHash": self.split_inventory_hash,
            "selectedFoldId": self.selected_fold_id,
        }
        if self.run_id != canonical_sha256(expected_run_identity):
            raise ValueError("runId does not match exact local run identity")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"report_hash"})
        )
        if self.report_hash != expected:
            raise ValueError("reportHash does not match local experiment evidence")
        return self


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class _HeldLocalSourceSnapshot:
    runtime: Path
    path: Path
    kind: str
    payload: bytes
    parent: _PublicationParentLease
    lease: _OwnedFileLease

    def inventory_entry(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "relativePath": self.path.relative_to(self.runtime).as_posix(),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "sizeBytes": len(self.payload),
        }

    def assert_visible_binding(self) -> None:
        self.lease.assert_visible_binding()
        self.parent.assert_confined()

    def close(self) -> None:
        try:
            self.lease.close()
        finally:
            self.parent.close()


def _hold_local_source_snapshot(
    runtime: Path,
    path: Path,
    *,
    kind: str,
    max_bytes: int,
) -> _HeldLocalSourceSnapshot:
    lexical = reject_link_components(path)
    try:
        lexical.relative_to(runtime)
    except ValueError as error:
        raise ValueError("local source path escapes the authorized runtime") from error
    parent = _PublicationParentLease(runtime, lexical.parent, create=False)
    lease: _OwnedFileLease | None = None
    try:
        lease = _OwnedFileLease(
            lexical,
            _path_identity(lexical),
            deletable=False,
            parent_lease=parent,
        )
        payload = lease.read(max_bytes=max_bytes)
        return _HeldLocalSourceSnapshot(
            runtime=runtime,
            path=lexical,
            kind=kind,
            payload=payload,
            parent=parent,
            lease=lease,
        )
    except Exception:
        if lease is not None:
            lease.close()
        parent.close()
        raise


def _canonical_json_snapshot(snapshot: bytes, *, label: str) -> dict[str, Any]:
    document = json.loads(snapshot)
    if not isinstance(document, dict) or snapshot != (canonical_json(document) + "\n").encode():
        raise ValueError(f"{label} is not an exact canonical JSON object")
    return document


def _bounded_gzip_snapshot(snapshot: bytes, *, max_expanded_bytes: int) -> bytes:
    with gzip.GzipFile(fileobj=BytesIO(snapshot), mode="rb") as stream:
        expanded = stream.read(max_expanded_bytes + 1)
    if not expanded or len(expanded) > max_expanded_bytes:
        raise ValueError("Email raw source exceeds the fixed expanded-size maximum")
    return expanded


def _seal_source_inventory(payload: dict[str, Any]) -> dict[str, Any]:
    payload["inventoryHash"] = canonical_sha256(payload)
    return payload


def load_email_local_inputs(runtime_root: Path) -> LocalDatasetInputs:
    """Reparse fixed Email raw bytes and match the published materialization exactly."""

    runtime = secure_existing_root(runtime_root)
    target = runtime / "materialized" / "email-eu-core" / "1.0.0"
    raw = runtime / "raw" / "email-eu-core" / "1.0.0"
    recipe = load_dataset_recipes()["email-eu-core"]
    source_by_id = {source.source_id: source for source in recipe.sources}
    specifications = (
        (
            "edges",
            raw / "email-Eu-core.txt.gz",
            "raw-edges",
            int(source_by_id["edges"].max_bytes),
        ),
        (
            "departments",
            raw / "email-Eu-core-department-labels.txt.gz",
            "raw-labels",
            int(source_by_id["departments"].max_bytes),
        ),
        ("bundle", target / "bundle.json", "materialized-bundle", 4 * 1024 * 1024),
        (
            "manifest",
            target / "materialization-manifest.json",
            "materialization-manifest",
            64 * 1024,
        ),
        (
            "offline",
            target / "offline-community-labels.json",
            "offline-labels",
            256 * 1024,
        ),
    )
    snapshots: list[_HeldLocalSourceSnapshot] = []
    try:
        by_name: dict[str, _HeldLocalSourceSnapshot] = {}
        for name, path, kind, max_bytes in specifications:
            held = _hold_local_source_snapshot(
                runtime,
                path,
                kind=kind,
                max_bytes=max_bytes,
            )
            snapshots.append(held)
            by_name[name] = held

        published_bundle = load_core_graph_bundle_json(by_name["bundle"].payload)
        manifest = _canonical_json_snapshot(
            by_name["manifest"].payload,
            label="Email materialization manifest",
        )
        offline = _canonical_json_snapshot(
            by_name["offline"].payload,
            label="Email offline labels",
        )
        raw_hashes = {
            source_id: hashlib.sha256(by_name[source_id].payload).hexdigest()
            for source_id in ("edges", "departments")
        }
        expected_keys = {
            "schemaVersion",
            "recipeId",
            "recipeVersion",
            "recipeSha256",
            "observedRawSha256",
            "expectedRawSha256",
            "combinedSourceSha256",
            "graphVersionHash",
            "offlineLabelsSha256",
            "splitSeed",
            "outputSemantics",
            "manifestSha256",
        }
        without_manifest_hash = {
            key: value for key, value in manifest.items() if key != "manifestSha256"
        }
        expected_raw = {source.source_id: source.expected_sha256 for source in recipe.sources}
        if (
            set(manifest) != expected_keys
            or manifest.get("manifestSha256") != canonical_sha256(without_manifest_hash)
            or manifest.get("schemaVersion") != "socialgraph-fm.core-dataset-materialization/1.0"
            or manifest.get("recipeId") != recipe.recipe_id
            or manifest.get("recipeVersion") != recipe.recipe_version
            or manifest.get("recipeSha256") != recipe.recipe_sha256
            or manifest.get("outputSemantics") != recipe.output_semantics
            or manifest.get("expectedRawSha256") != expected_raw
            or manifest.get("observedRawSha256") != raw_hashes
            or raw_hashes != _EMAIL_RAW_SOURCE_SHA256
            or type(manifest.get("splitSeed")) is not int
        ):
            raise ValueError("Email materialization manifest identity is invalid")
        combined_source_sha = canonical_sha256(dict(sorted(raw_hashes.items())))
        labels = offline.get("labels")
        if (
            manifest.get("combinedSourceSha256") != combined_source_sha
            or manifest.get("graphVersionHash") != published_bundle.graph_version_hash
            or published_bundle.source.source_sha256 != combined_source_sha
            or set(offline) != {"schemaVersion", "graphId", "labels", "labelsSha256"}
            or offline.get("schemaVersion") != "socialgraph-fm.core-offline-community-labels/1.0"
            or offline.get("graphId") != "email-eu-core"
            or not isinstance(labels, dict)
            or set(labels) != {"department"}
            or offline.get("labelsSha256") != canonical_sha256(labels)
            or manifest.get("offlineLabelsSha256") != offline.get("labelsSha256")
        ):
            raise ValueError("Email materialization evidence identity is invalid")

        edges_bytes = _bounded_gzip_snapshot(
            by_name["edges"].payload,
            max_expanded_bytes=2_000_000,
        )
        departments_bytes = _bounded_gzip_snapshot(
            by_name["departments"].payload,
            max_expanded_bytes=2_000_000,
        )
        with tempfile.TemporaryDirectory(prefix="socialgraph-local-email-") as temporary:
            temporary_root = Path(temporary)
            edges_path = temporary_root / "edges.txt"
            departments_path = temporary_root / "departments.txt"
            edges_path.write_bytes(edges_bytes)
            departments_path.write_bytes(departments_bytes)
            parsed = parse_email_files(edges_path, departments_path)

        split = spanning_forest_link_split(
            num_nodes=len(parsed.node_ids),
            edges=parsed.edges,
            seed=int(manifest["splitSeed"]),
        )
        role_by_edge = {
            edge: role
            for role, role_edges in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            )
            for edge in role_edges
        }
        bundle_payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
            "directed": False,
            "nodes": [
                {"id": identifier, "index": index}
                for index, identifier in enumerate(parsed.node_ids)
            ],
            "edges": [
                {
                    "sourceId": parsed.node_ids[source],
                    "targetId": parsed.node_ids[target_index],
                    "edgeType": "email",
                    "weight": 1.0,
                }
                for source, target_index in parsed.edges
            ],
            "nodeFeatures": [],
            "structuralFeatures": None,
            "source": {
                "sourceName": "SNAP Email-Eu-core",
                "sourceUri": "https://snap.stanford.edu/data/email-Eu-core.html",
                "citation": recipe.citation,
                "sourceSha256": combined_source_sha,
            },
            "splitManifest": {
                "strategy": "spanning-forest-80-10-10",
                "assignments": [
                    {
                        "entityId": (
                            f"edge:{parsed.node_ids[source]}:{parsed.node_ids[target_index]}"
                        ),
                        "role": role_by_edge[(source, target_index)],
                    }
                    for source, target_index in parsed.edges
                ],
            },
        }
        bundle_payload["graphVersionHash"] = calculate_graph_version_hash(bundle_payload)
        bundle = CoreGraphBundle.model_validate(bundle_payload)
        authoritative_departments = {
            node_id: value
            for node_id, value in zip(
                parsed.node_ids,
                parsed.offline_labels["department"],
                strict=True,
            )
        }
        if (
            bundle != published_bundle
            or labels["department"] != authoritative_departments
            or set(authoritative_departments) != {node.id for node in bundle.nodes}
        ):
            raise ValueError("Email raw derivation differs from published materialization")
        targets = {f"edge:{edge.source_id}:{edge.target_id}": 1 for edge in bundle.edges}
        split_inventory = LocalSplitInventory.create(
            dataset_id="email-eu-core",
            manifests=(bundle.split_manifest,),
            selected_fold_id="fold-0",
        )
        source_inventory = _seal_source_inventory(
            {
                "schemaVersion": "socialgraph-fm.core-local-source-inventory/1.0",
                "datasetId": "email-eu-core",
                "sourceSha256": bundle.source.source_sha256,
                "scope": "validated-email-materialization-and-raw-sources",
                "files": tuple(snapshot.inventory_entry() for snapshot in snapshots),
            }
        )
        for snapshot in snapshots:
            snapshot.assert_visible_binding()
        return LocalDatasetInputs(
            bundle=bundle,
            targets_by_entity=targets,
            split_inventory=split_inventory,
            source_inventory=source_inventory,
        )
    finally:
        for snapshot in reversed(snapshots):
            snapshot.close()


def _penn_split_manifest(node_ids: tuple[str, ...], split: Any) -> SplitManifest:
    assigned = set(split.train) | set(split.validation) | set(split.test)
    assignments = [
        {"entityId": node_ids[index], "role": role}
        for role, indices in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
        for index in indices
    ]
    assignments.extend(
        {"entityId": node_ids[index], "role": "unlabeled"}
        for index in range(len(node_ids))
        if index not in assigned
    )
    return SplitManifest.model_validate(
        {
            "strategy": "official",
            "assignments": sorted(assignments, key=lambda item: item["entityId"]),
        }
    )


def load_penn94_local_inputs(runtime_root: Path) -> LocalDatasetInputs:
    """Reparse fixed Penn94 bytes and its one-time hash-locked safe split conversion."""

    runtime = secure_existing_root(runtime_root)
    raw = runtime / "raw" / "facebook100" / "1.0.0"
    mat_path = raw / "Penn94.mat"
    raw_split = raw / "fb100-Penn94-splits.npy"
    derived = runtime / "derived" / "facebook100" / "penn94-official-splits" / "1.0.0"
    safe_split = derived / "penn94-official-splits-safe.npz"
    manifest_path = derived / "conversion-manifest.json"
    recipe = load_dataset_recipes()["facebook100"]
    sources = {source.source_id: source for source in recipe.sources}
    specifications = (
        ("mat", mat_path, "raw-mat", int(sources["Penn94"].max_bytes)),
        (
            "raw-split",
            raw_split,
            "raw-official-splits",
            PENN94_RAW_SPLIT_MAX_BYTES,
        ),
        (
            "safe-split",
            safe_split,
            "safe-official-splits",
            2 * 1024 * 1024,
        ),
        (
            "manifest",
            manifest_path,
            "conversion-manifest",
            1024 * 1024,
        ),
    )
    snapshots: list[_HeldLocalSourceSnapshot] = []
    try:
        by_name: dict[str, _HeldLocalSourceSnapshot] = {}
        for name, path, kind, max_bytes in specifications:
            held = _hold_local_source_snapshot(
                runtime,
                path,
                kind=kind,
                max_bytes=max_bytes,
            )
            snapshots.append(held)
            by_name[name] = held
        mat_sha256 = hashlib.sha256(by_name["mat"].payload).hexdigest()
        raw_split_sha256 = hashlib.sha256(by_name["raw-split"].payload).hexdigest()
        safe_split_sha256 = hashlib.sha256(by_name["safe-split"].payload).hexdigest()
        if mat_sha256 != PENN94_DATA_SHA256:
            raise ValueError("Penn94 MAT bytes do not match the fixed observed SHA-256")
        if raw_split_sha256 != PENN94_RAW_SPLIT_SHA256:
            raise ValueError("Penn94 split does not match the fixed raw SHA-256")
        if safe_split_sha256 != _PENN94_SAFE_SPLIT_SHA256:
            raise ValueError("Penn94 safe split does not match the fixed hash-locked conversion")

        manifest = _canonical_json_snapshot(
            by_name["manifest"].payload,
            label="Penn94 conversion manifest",
        )
        expected_keys = {
            "schemaVersion",
            "sourceCommit",
            "sourceUrl",
            "sourceSha256",
            "penn94DataUrl",
            "penn94DataObservedSha256",
            "derivedFormat",
            "derivedSha256",
            "converterVersion",
            "converterCodeSha256",
            "splitCount",
            "labeledNodeCount",
            "roleCounts",
            "recipeSha256",
            "manifestSha256",
        }
        if set(manifest) != expected_keys:
            raise ValueError("Penn94 conversion manifest inventory is invalid")
        without_hash = {key: value for key, value in manifest.items() if key != "manifestSha256"}
        converter_module_path = sys.modules[verify_penn94_raw_split.__module__].__file__
        if converter_module_path is None:
            raise ValueError("Penn94 converter module path is unavailable")
        converter_path = Path(converter_module_path).resolve()
        expected = {
            "schemaVersion": "socialgraph-fm.core-penn94-split-conversion/1.0",
            "sourceCommit": PENN94_LINKX_COMMIT,
            "sourceUrl": PENN94_RAW_SPLIT_URL,
            "sourceSha256": PENN94_RAW_SPLIT_SHA256,
            "penn94DataUrl": sources["Penn94"].url,
            "penn94DataObservedSha256": PENN94_DATA_SHA256,
            "derivedFormat": "npz with primitive little-endian int64 NPY members",
            "derivedSha256": safe_split_sha256,
            "converterVersion": PENN94_CONVERTER_VERSION,
            "converterCodeSha256": _hash_file(converter_path),
            "splitCount": 5,
            "labeledNodeCount": PENN94_LABELED_NODE_COUNT,
            "roleCounts": PENN94_SPLIT_COUNTS,
            "recipeSha256": recipe.recipe_sha256,
        }
        if manifest.get("manifestSha256") != canonical_sha256(without_hash) or any(
            manifest.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("Penn94 conversion manifest provenance is invalid")

        with tempfile.TemporaryDirectory(prefix="socialgraph-local-penn94-") as temporary:
            temporary_root = Path(temporary)
            snapshot_mat = temporary_root / "Penn94.mat"
            snapshot_safe_split = temporary_root / "penn94-official-splits-safe.npz"
            snapshot_mat.write_bytes(by_name["mat"].payload)
            snapshot_safe_split.write_bytes(by_name["safe-split"].payload)
            parsed = parse_facebook100_mat(
                snapshot_mat,
                graph_id="Penn94",
                official_splits_path=snapshot_safe_split,
            )
        if len(parsed.official_splits) != 5:
            raise ValueError("Penn94 local dev requires all five official splits")
        combined_source_sha = canonical_sha256(
            {
                "Penn94": mat_sha256,
                "Penn94-official-splits": raw_split_sha256,
            }
        )
        bundle = bundle_from_parsed_graph(
            parsed,
            source=SourceProvenance(
                sourceName="facebook100:Penn94",
                sourceUri=sources["Penn94"].url,
                citation=recipe.citation,
                sourceSha256=combined_source_sha,
            ),
            split=parsed.official_splits[0],
            excluded_feature_names={"gender"},
        )
        targets = {
            node_id: int(value) - 1
            for node_id, value in zip(parsed.node_ids, parsed.targets["gender"], strict=True)
            if int(value) in {1, 2}
        }
        if any(value not in {0, 1} for value in targets.values()):
            raise ValueError("Penn94 gender benchmark labels did not normalize to two classes")
        split_inventory = LocalSplitInventory.create(
            dataset_id="penn94",
            manifests=tuple(
                _penn_split_manifest(parsed.node_ids, split) for split in parsed.official_splits
            ),
            selected_fold_id="fold-0",
        )
        if split_inventory.folds[0].split_manifest != bundle.split_manifest:
            raise ValueError("Penn94 selected fold-0 differs from the training bundle")
        source_inventory = _seal_source_inventory(
            {
                "schemaVersion": "socialgraph-fm.core-local-source-inventory/1.0",
                "datasetId": "penn94",
                "sourceSha256": bundle.source.source_sha256,
                "scope": "hash-locked-penn94-raw-conversion-safe-split-and-five-folds",
                "conversionManifestHash": manifest["manifestSha256"],
                "splitInventoryHash": split_inventory.inventory_hash,
                "files": tuple(snapshot.inventory_entry() for snapshot in snapshots),
            }
        )
        for snapshot in snapshots:
            snapshot.assert_visible_binding()
        return LocalDatasetInputs(
            bundle=bundle,
            targets_by_entity=targets,
            split_inventory=split_inventory,
            source_inventory=source_inventory,
        )
    finally:
        for snapshot in reversed(snapshots):
            snapshot.close()


def _state_hash(state: Mapping[str, Tensor]) -> str:
    return canonical_sha256(
        {name: canonical_tensor_digest(value) for name, value in sorted(state.items())}
    )


def _target_inventory(targets_by_entity: Mapping[str, int]) -> dict[str, Any]:
    if any(
        not isinstance(identifier, str)
        or not identifier
        or type(target) is not int
        or target not in {0, 1}
        for identifier, target in targets_by_entity.items()
    ):
        raise ValueError("local binary targets require nonempty IDs and integer zero/one labels")
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-local-target-inventory/1.0",
        "targets": tuple(
            {"entityId": identifier, "target": targets_by_entity[identifier]}
            for identifier in sorted(targets_by_entity)
        ),
    }
    payload["inventoryHash"] = canonical_sha256(payload)
    return payload


def _targets_hash(targets_by_entity: Mapping[str, int]) -> str:
    return str(_target_inventory(targets_by_entity)["inventoryHash"])


def _code_inventory(
    root: Path | None = None,
    *,
    relative_paths: tuple[str, ...] = _CODE_INVENTORY_RELATIVE_PATHS,
) -> dict[str, Any]:
    return local_code_inventory(root, relative_paths=relative_paths)


def _validate_code_inventory_document(
    document: Mapping[str, Any],
    *,
    root: Path | None = None,
    relative_paths: tuple[str, ...] = _CODE_INVENTORY_RELATIVE_PATHS,
) -> dict[str, Any]:
    return validate_local_code_inventory(dict(document), root=root, relative_paths=relative_paths)


def _code_hash() -> str:
    return str(_code_inventory()["inventoryHash"])


def _environment(device: torch.device) -> tuple[dict[str, Any], str]:
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("local environment supports only CPU or CUDA execution")
    device_type: Literal["cpu", "cuda"] = "cuda" if device.type == "cuda" else "cpu"
    inventory = local_environment_inventory(device_type)
    payload = inventory.model_dump(mode="python", by_alias=True)
    return payload, inventory.inventory_hash


def _edge_index(bundle: CoreGraphBundle) -> Tensor:
    selection = derive_training_selection(bundle)
    node_index = {node.id: node.index for node in bundle.nodes}
    pairs: list[tuple[int, int]] = []
    for ordinal in selection.visible_edge_indices:
        edge = bundle.edges[ordinal]
        left = node_index[edge.source_id]
        right = node_index[edge.target_id]
        pairs.append((left, right))
        if not bundle.directed:
            pairs.append((right, left))
    if not pairs:
        raise ValueError("local training requires at least one train-visible edge")
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _partition(
    bundle: CoreGraphBundle,
    *,
    task_kind: Literal["node-binary", "edge-binary"],
    role: Literal["train", "validation"],
    targets_by_entity: Mapping[str, int],
) -> SupervisedPartition:
    assignments = tuple(
        sorted(
            assignment.entity_id
            for assignment in bundle.split_manifest.assignments
            if assignment.role == role
        )
    )
    if not assignments:
        raise ValueError(f"local supervised {role} role is empty")
    if any(identifier not in targets_by_entity for identifier in assignments):
        raise ValueError(f"local supervised {role} targets are incomplete")
    if task_kind == "node-binary":
        index_by_id = {node.id: node.index for node in bundle.nodes}
        indices = tuple(index_by_id[identifier] for identifier in assignments)
        return SupervisedPartition(
            entityIds=assignments,
            nodeIndices=indices,
            targets=tuple(targets_by_entity[identifier] for identifier in assignments),
        )
    node_index = {node.id: node.index for node in bundle.nodes}
    edge_by_id = {
        f"edge:{edge.source_id}:{edge.target_id}": (
            node_index[edge.source_id],
            node_index[edge.target_id],
        )
        for edge in bundle.edges
    }
    return SupervisedPartition(
        entityIds=assignments,
        edgePairs=tuple(edge_by_id[identifier] for identifier in assignments),
        targets=tuple(targets_by_entity[identifier] for identifier in assignments),
    )


def _resolve_device(device_name: Literal["auto", "cpu", "cuda"]) -> torch.device:
    selected = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    if selected == "auto":
        selected = "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(selected)


def _relative(runtime: Path, path: Path) -> str:
    return path.resolve().relative_to(runtime).as_posix()


def _publish_head_artifact(
    path: Path,
    *,
    model: CoreGFM,
    bundle: CoreGraphBundle,
    adapter: BundleInputAdapter,
    head_report_hash: str,
) -> tuple[str, str]:
    payload = {
        "schemaVersion": _HEAD_ARTIFACT_SCHEMA,
        "model": model.state_dict(),
        "adapter": adapter.state_dict(),
        "headReportHash": head_report_hash,
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    observed = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(observed, dict) or observed.get("schemaVersion") != _HEAD_ARTIFACT_SCHEMA:
        raise ValueError("local head artifact fresh load failed")
    fresh_model = CoreGFM(node_classes=2)
    fresh_adapter = BundleInputAdapter(bundle, mode="training", schema=adapter.schema)
    fresh_model.load_state_dict(observed["model"], strict=True)
    fresh_adapter.load_state_dict(observed["adapter"], strict=True)
    if (
        _state_hash(fresh_model.state_dict()) != _state_hash(model.state_dict())
        or _state_hash(fresh_adapter.state_dict()) != _state_hash(adapter.state_dict())
        or observed.get("headReportHash") != head_report_hash
    ):
        raise ValueError("local head artifact differs after fresh strict load")
    semantic_hash = canonical_sha256(
        {
            "schemaVersion": _HEAD_ARTIFACT_SCHEMA,
            "modelStateHash": _state_hash(fresh_model.state_dict()),
            "adapterStateHash": _state_hash(fresh_adapter.state_dict()),
            "headReportHash": head_report_hash,
        }
    )
    return _hash_file(path), semantic_hash


def _write_new_bytes(path: Path, serialized: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


class _LocalOwnedControlFile:
    """Create and retain one exact control-file identity without pathname adoption."""

    def __init__(
        self,
        parent: _PublicationParentLease,
        name: str,
        serialized: bytes,
    ) -> None:
        if not serialized:
            raise ValueError("local control file must be nonempty")
        if Path(name).name != name or any(character in name for character in ("/", "\\", ":")):
            raise ValueError("local control filename is unsafe")
        self.parent = parent
        self.path = parent.parent / name
        self._handle: int | None = None
        self._descriptor: int | None = None
        parent.assert_confined()
        if os.name == "nt":
            create_file = getattr(_formal_fs, "_CreateFileW")
            handle = create_file(
                str(self.path),
                0x80000000 | 0x40000000 | 0x00010000,
                0x0001,
                None,
                1,
                0x0080 | 0x00200000,
                None,
            )
            if handle == getattr(_formal_fs, "_INVALID_HANDLE_VALUE"):
                code = ctypes.get_last_error()
                if code in {80, 183}:
                    raise FileExistsError(self.path)
                raise OSError(code, "failed to create local control file")
            self._handle = int(handle)
            self.identity = tuple(getattr(_formal_fs, "_win_identity")(self._handle))
        else:
            descriptor = parent.open_file(name, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            self._descriptor = descriptor
            details = os.fstat(descriptor)
            self.identity = (int(details.st_dev), int(details.st_ino))
        try:
            self.replace_payload(serialized)
            self.assert_visible_binding()
        except Exception:
            self.close()
            raise

    def replace_payload(self, serialized: bytes) -> None:
        if not serialized or len(serialized) > 4 * 1024 * 1024:
            raise ValueError("local control payload size is outside the authorized bound")
        self.parent.assert_confined()
        if self._handle is not None:
            from ctypes import wintypes

            set_pointer = getattr(_formal_fs, "_SetFilePointerEx")
            if not set_pointer(self._handle, ctypes.c_longlong(0), None, 0):
                raise OSError(ctypes.get_last_error(), "failed to rewind local control file")
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            set_end = kernel32.SetEndOfFile
            set_end.argtypes = [wintypes.HANDLE]
            set_end.restype = wintypes.BOOL
            if not set_end(self._handle):
                raise OSError(ctypes.get_last_error(), "failed to truncate local control file")
            write_file = kernel32.WriteFile
            write_file.argtypes = [
                wintypes.HANDLE,
                wintypes.LPCVOID,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                wintypes.LPVOID,
            ]
            write_file.restype = wintypes.BOOL
            written = wintypes.DWORD()
            buffer = ctypes.create_string_buffer(serialized)
            if not write_file(
                self._handle,
                buffer,
                len(serialized),
                ctypes.byref(written),
                None,
            ) or int(written.value) != len(serialized):
                raise OSError(ctypes.get_last_error(), "failed to write local control file")
            if not getattr(_formal_fs, "_FlushFileBuffers")(self._handle):
                raise OSError(ctypes.get_last_error(), "failed to flush local control file")
            return
        if self._descriptor is None:
            raise RuntimeError("local control file is closed")
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        os.ftruncate(self._descriptor, 0)
        view = memoryview(serialized)
        while view:
            posix_written = os.write(self._descriptor, view)
            if posix_written < 1:
                raise OSError("failed to write local control file")
            view = view[posix_written:]
        os.fsync(self._descriptor)

    def read(self, *, max_bytes: int) -> bytes:
        if self._handle is not None:
            details = getattr(_formal_fs, "_win_info")(self._handle)
            size = (int(details.nFileSizeHigh) << 32) | int(details.nFileSizeLow)
            if size < 1 or size > max_bytes:
                raise ValueError("local control payload size is outside the authorized bound")
            if not getattr(_formal_fs, "_SetFilePointerEx")(
                self._handle, ctypes.c_longlong(0), None, 0
            ):
                raise OSError(ctypes.get_last_error(), "failed to rewind local control file")
            chunks: list[bytes] = []
            remaining = size
            while remaining:
                length = min(1024 * 1024, remaining)
                buffer = ctypes.create_string_buffer(length)
                from ctypes import wintypes

                observed = wintypes.DWORD()
                if (
                    not getattr(_formal_fs, "_ReadFile")(
                        self._handle,
                        buffer,
                        length,
                        ctypes.byref(observed),
                        None,
                    )
                    or not observed.value
                ):
                    raise OSError(ctypes.get_last_error(), "failed to read local control file")
                chunks.append(buffer.raw[: observed.value])
                remaining -= int(observed.value)
            return b"".join(chunks)
        if self._descriptor is None:
            raise RuntimeError("local control file is closed")
        details = os.fstat(self._descriptor)
        if details.st_size < 1 or details.st_size > max_bytes:
            raise ValueError("local control payload size is outside the authorized bound")
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        chunks = []
        remaining = details.st_size
        while remaining:
            chunk = os.read(self._descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("local control file changed while held")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def assert_visible_binding(self) -> None:
        self.parent.assert_confined()
        if _path_identity(self.path) != self.identity:
            raise ValueError("local control file identity changed")

    def remove_owned_link(self) -> bool:
        self.assert_visible_binding()
        if self._handle is not None:
            disposition_type = getattr(_formal_fs, "_FILE_DISPOSITION_INFO")
            disposition = disposition_type(True)
            set_information = getattr(_formal_fs, "_SetFileInformationByHandle")
            if not set_information(
                self._handle,
                4,
                ctypes.byref(disposition),
                ctypes.sizeof(disposition),
            ):
                raise OSError(ctypes.get_last_error(), "failed to remove local control file")
            self.close()
            return not self.path.exists() and not self.path.is_symlink()
        self.close()
        return False

    def close(self) -> None:
        if self._handle is not None:
            handle, self._handle = self._handle, None
            getattr(_formal_fs, "_close_win_handle")(handle)
        if self._descriptor is not None:
            descriptor, self._descriptor = self._descriptor, None
            os.close(descriptor)


class _LocalPublisherLock:
    def __init__(
        self,
        parent: _PublicationParentLease,
        name: str,
        *,
        request_hash: str,
        token: str,
    ) -> None:
        self._owned: _LocalOwnedControlFile | None = None
        self._posix: _PublisherLock | None = None
        self.path = parent.parent / name
        if os.name == "nt":
            payload: dict[str, Any] = {
                "schemaVersion": "socialgraph-fm.core-local-publisher-lock/1.0",
                "requestHash": request_hash,
                "token": token,
            }
            payload["lockHash"] = canonical_sha256(payload)
            self._owned = _LocalOwnedControlFile(
                parent,
                name,
                (canonical_json(payload) + "\n").encode(),
            )
            self.identity = self._owned.identity
        else:
            self._posix = _PublisherLock(
                parent,
                name,
                active_message="local experiment already has an active publisher",
            )
            self.identity = _path_identity(self.path)

    def close(self) -> None:
        if self._owned is not None:
            owned, self._owned = self._owned, None
            owned.remove_owned_link()
        if self._posix is not None:
            lock, self._posix = self._posix, None
            lock.close()


def _publish_json_evidence(
    runtime: Path,
    path: Path,
    document: Mapping[str, Any] | BaseModel,
    *,
    semantic_hash: str,
    published_path: Path | None = None,
) -> EvidenceReference:
    serialized = (canonical_json(document) + "\n").encode()
    _write_new_bytes(path, serialized)
    return EvidenceReference(
        relativePath=_relative(runtime, path if published_path is None else published_path),
        sha256=_hash_file(path),
        semanticHash=semantic_hash,
    )


def _existing_file_evidence(
    runtime: Path,
    path: Path,
    *,
    semantic_hash: str,
    published_path: Path | None = None,
) -> EvidenceReference:
    return EvidenceReference(
        relativePath=_relative(runtime, path if published_path is None else published_path),
        sha256=_hash_file(path),
        semanticHash=semantic_hash,
    )


def _artifact_inventory_document(artifacts: Path) -> dict[str, Any]:
    observed = tuple(sorted(artifacts.iterdir(), key=lambda item: item.name))
    if tuple(entry.name for entry in observed) != _LOCAL_ARTIFACT_PAYLOAD_NAMES:
        raise ValueError("local artifact payload inventory is not exact before sealing")
    entries = []
    for entry in observed:
        if not entry.is_file() or entry.is_symlink():
            raise ValueError("local artifact inventory requires regular payload files only")
        entries.append(
            {
                "name": entry.name,
                "sha256": _hash_file(entry),
                "sizeBytes": entry.stat().st_size,
            }
        )
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-local-artifact-inventory/1.0",
        "payloadFiles": tuple(entries),
        "requiredControlFiles": _LOCAL_ARTIFACT_CONTROL_NAMES,
    }
    payload["inventoryHash"] = canonical_sha256(payload)
    return payload


def _validate_target_inventory(document: Mapping[str, Any]) -> dict[str, int]:
    payload = dict(document)
    observed_hash = payload.pop("inventoryHash", None)
    targets = payload.get("targets")
    if (
        set(document) != {"schemaVersion", "targets", "inventoryHash"}
        or payload.get("schemaVersion") != "socialgraph-fm.core-local-target-inventory/1.0"
        or not isinstance(targets, (list, tuple))
        or observed_hash != canonical_sha256(payload)
    ):
        raise ValueError("local target inventory identity is invalid")
    result: dict[str, int] = {}
    for item in targets:
        if not isinstance(item, dict) or set(item) != {"entityId", "target"}:
            raise ValueError("local target inventory entry is invalid")
        identifier = item["entityId"]
        target = item["target"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or type(target) is not int
            or target not in {0, 1}
            or identifier in result
        ):
            raise ValueError("local target inventory entry is invalid")
        result[identifier] = target
    if tuple(result) != tuple(sorted(result)) or not result:
        raise ValueError("local target inventory must be nonempty, unique, and sorted")
    return result


def _validate_artifact_inventory(
    root: Path,
    reference: EvidenceReference,
) -> tuple[Path, dict[str, Any]]:
    document = json.loads(_evidence_bytes(root, reference))
    if not isinstance(document, dict):
        raise ValueError("local artifact inventory must be a JSON object")
    payload = dict(document)
    observed_hash = payload.pop("inventoryHash", None)
    entries = payload.get("payloadFiles")
    if (
        set(document) != {"schemaVersion", "payloadFiles", "requiredControlFiles", "inventoryHash"}
        or payload.get("schemaVersion") != "socialgraph-fm.core-local-artifact-inventory/1.0"
        or payload.get("requiredControlFiles")
        not in (
            ("artifact-inventory.json", "report.json"),
            ["artifact-inventory.json", "report.json"],
        )
        or not isinstance(entries, (list, tuple))
        or observed_hash != canonical_sha256(payload)
        or observed_hash != reference.semantic_hash
    ):
        raise ValueError("local artifact inventory identity is invalid")
    inventory_relative = Path(reference.relative_path)
    if inventory_relative.name != "artifact-inventory.json":
        raise ValueError("local artifact inventory path is not canonical")
    artifact_relative = inventory_relative.parent
    artifact_directory = reject_link_components(root / artifact_relative)
    if not artifact_directory.is_dir():
        raise ValueError("local artifact directory is unavailable")
    expected_payload_names = set(_LOCAL_ARTIFACT_PAYLOAD_NAMES)
    observed_entries: dict[str, dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict) or set(item) != {"name", "sha256", "sizeBytes"}:
            raise ValueError("local artifact inventory entry is invalid")
        name = item["name"]
        size_bytes = item["sizeBytes"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in observed_entries
            or not isinstance(item["sha256"], str)
            or type(size_bytes) is not int
            or size_bytes < 1
        ):
            raise ValueError("local artifact inventory entry is invalid")
        observed_entries[name] = item
    if set(observed_entries) != expected_payload_names:
        raise ValueError("local artifact payload inventory is not exact")
    actual_names = {entry.name for entry in artifact_directory.iterdir()}
    if actual_names != expected_payload_names | {"artifact-inventory.json", "report.json"}:
        raise ValueError("local artifact directory contains missing or extra files")
    for name, item in observed_entries.items():
        label = "checkpoint" if name == "core-training.pt" else "evidence bytes"
        try:
            snapshot = read_confined_snapshot(
                root,
                (artifact_relative / name).as_posix(),
                max_bytes=int(item["sizeBytes"]),
            )
        except ValueError as error:
            raise ValueError(f"local {label} differ from exact artifact inventory") from error
        if (
            len(snapshot) != item["sizeBytes"]
            or hashlib.sha256(snapshot).hexdigest() != item["sha256"]
        ):
            raise ValueError(f"local {label} differ from exact artifact inventory")
    document["inventoryHash"] = observed_hash
    return artifact_relative, document


def _validate_source_inventory(
    inventory: Mapping[str, Any],
    *,
    dataset_id: Literal["email-eu-core", "penn94"],
    source_sha256: str,
) -> dict[str, Any]:
    payload = dict(inventory)
    observed_hash = payload.pop("inventoryHash", None)
    expected_keys = {
        "schemaVersion",
        "datasetId",
        "sourceSha256",
        "scope",
        "files",
        "inventoryHash",
    }
    if dataset_id == "penn94":
        expected_keys |= {"conversionManifestHash", "splitInventoryHash"}
    files = payload.get("files")
    if (
        set(inventory) != expected_keys
        or payload.get("schemaVersion") != "socialgraph-fm.core-local-source-inventory/1.0"
        or payload.get("datasetId") != dataset_id
        or payload.get("sourceSha256") != source_sha256
        or not isinstance(payload.get("scope"), str)
        or not payload["scope"]
        or not isinstance(files, (list, tuple))
        or any(
            not isinstance(item, dict)
            or set(item) != {"kind", "relativePath", "sha256", "sizeBytes"}
            or not isinstance(item["kind"], str)
            or not item["kind"]
            or not isinstance(item["relativePath"], str)
            or not item["relativePath"]
            or not isinstance(item["sha256"], str)
            or len(item["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in item["sha256"])
            or type(item["sizeBytes"]) is not int
            or item["sizeBytes"] < 1
            for item in files
        )
        or len({item["kind"] for item in files}) != len(files)
        or (
            dataset_id == "penn94"
            and any(
                not isinstance(payload.get(field), str)
                or len(payload[field]) != 64
                or any(character not in "0123456789abcdef" for character in payload[field])
                for field in ("conversionManifestHash", "splitInventoryHash")
            )
        )
        or observed_hash != canonical_sha256(payload)
    ):
        raise ValueError("local source inventory identity is invalid")
    payload["inventoryHash"] = observed_hash
    return payload


def _assert_authoritative_local_inputs(
    *,
    runtime: Path,
    dataset_id: Literal["email-eu-core", "penn94"],
    bundle: CoreGraphBundle,
    targets_by_entity: Mapping[str, int],
    split_inventory: LocalSplitInventory,
    source_inventory: Mapping[str, Any],
) -> tuple[_HeldLocalSourceSnapshot, ...]:
    observed_source = _validate_source_inventory(
        json.loads(canonical_json(source_inventory)),
        dataset_id=dataset_id,
        source_sha256=bundle.source.source_sha256,
    )
    authoritative = (
        load_email_local_inputs(runtime)
        if dataset_id == "email-eu-core"
        else load_penn94_local_inputs(runtime)
    )
    authoritative_source = _validate_source_inventory(
        json.loads(canonical_json(authoritative.source_inventory)),
        dataset_id=dataset_id,
        source_sha256=authoritative.bundle.source.source_sha256,
    )
    if (
        authoritative.bundle != bundle
        or dict(authoritative.targets_by_entity) != dict(targets_by_entity)
        or authoritative.split_inventory != split_inventory
        or authoritative_source != observed_source
    ):
        raise ValueError("local authoritative raw dataset derivation differs from inputs")

    held: list[_HeldLocalSourceSnapshot] = []
    try:
        for item in authoritative_source["files"]:
            snapshot = _hold_local_source_snapshot(
                runtime,
                runtime / item["relativePath"],
                kind=item["kind"],
                max_bytes=item["sizeBytes"],
            )
            held.append(snapshot)
            if snapshot.inventory_entry() != item:
                raise ValueError("local authoritative raw dataset changed after fixed derivation")
        for snapshot in held:
            snapshot.assert_visible_binding()
        return tuple(held)
    except Exception:
        for snapshot in reversed(held):
            snapshot.close()
        raise


def _assert_held_authoritative_sources(
    snapshots: tuple[_HeldLocalSourceSnapshot, ...],
) -> None:
    for snapshot in snapshots:
        snapshot.assert_visible_binding()
        if snapshot.lease.read(max_bytes=len(snapshot.payload)) != snapshot.payload:
            raise ValueError("local authoritative raw dataset bytes changed during publication")


def _close_held_authoritative_sources(
    snapshots: tuple[_HeldLocalSourceSnapshot, ...],
) -> None:
    for snapshot in reversed(snapshots):
        snapshot.close()


def _validate_adapter_evidence_document(document: Mapping[str, Any]) -> str:
    expected_keys = {
        "schemaVersion",
        "adapterSchema",
        "adapterSchemaHash",
        "adapterStateHash",
        "headArtifactRelativePath",
        "headArtifactSha256",
        "headArtifactSemanticHash",
        "evidenceHash",
    }
    if (
        set(document) != expected_keys
        or document.get("schemaVersion") != "socialgraph-fm.core-local-adapter-evidence/1.0"
    ):
        raise ValueError("local adapter evidence inventory is invalid")
    return _recompute_document_hash(dict(document), "evidenceHash")


def _validate_unavailable_calibration_document(document: Mapping[str, Any]) -> str:
    if (
        set(document) != {"schemaVersion", "status", "evidenceHash"}
        or document.get("schemaVersion")
        != "socialgraph-fm.core-local-calibration-evidence/1.0"
        or document.get("status") != "unavailable-single-class-validation"
    ):
        raise ValueError("local unavailable calibration evidence is not exact")
    return _recompute_document_hash(dict(document), "evidenceHash")


def _validate_publication_control_document(
    document: Mapping[str, Any],
    *,
    request_hash: str | None = None,
    target_name: str | None = None,
) -> dict[str, Any]:
    expected_keys = {
        "schemaVersion",
        "requestHash",
        "targetName",
        "stagingName",
        "stagingIdentity",
        "lockName",
        "lockIdentity",
        "journalName",
        "journalIdentity",
        "controlHash",
    }
    payload = dict(document)
    observed_hash = payload.pop("controlHash", None)
    identities = (
        payload.get("stagingIdentity"),
        payload.get("lockIdentity"),
        payload.get("journalIdentity"),
    )
    names = (
        payload.get("targetName"),
        payload.get("stagingName"),
        payload.get("lockName"),
        payload.get("journalName"),
    )
    if (
        set(document) != expected_keys
        or payload.get("schemaVersion") != "socialgraph-fm.core-local-publication-control/1.0"
        or not isinstance(payload.get("requestHash"), str)
        or len(payload["requestHash"]) != 64
        or any(character not in "0123456789abcdef" for character in payload["requestHash"])
        or any(not isinstance(name, str) or not name or Path(name).name != name for name in names)
        or any(
            not isinstance(identity, (list, tuple))
            or len(identity) != 2
            or not all(type(value) is int for value in identity)
            for identity in identities
        )
        or observed_hash != canonical_sha256(payload)
        or (request_hash is not None and payload["requestHash"] != request_hash)
        or (target_name is not None and payload["targetName"] != target_name)
    ):
        raise ValueError("local publication control identity is invalid")
    payload["controlHash"] = observed_hash
    return payload


def _process_identity(process_id: int) -> dict[str, Any] | None:
    """Return a PID-reuse-safe native process identity, or None when it is gone."""

    if type(process_id) is not int or process_id < 1:
        raise ValueError("local publisher process ID is invalid")
    if os.name == "nt":

        class _FileTime(ctypes.Structure):
            _fields_ = (("low", ctypes.c_uint32), ("high", ctypes.c_uint32))

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        open_process.restype = ctypes.c_void_p
        handle = open_process(0x1000, 0, process_id)
        if not handle:
            code = ctypes.get_last_error()
            if code == 87:
                return None
            raise OSError(code, "unable to establish local publisher process identity")
        try:
            creation = _FileTime()
            exit_time = _FileTime()
            kernel_time = _FileTime()
            user_time = _FileTime()
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                raise OSError(ctypes.get_last_error(), "unable to read publisher start time")
            if (int(exit_time.high) << 32) | int(exit_time.low):
                return None
            buffer = ctypes.create_unicode_buffer(32768)
            length = ctypes.c_uint32(len(buffer))
            if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)):
                raise OSError(ctypes.get_last_error(), "unable to read publisher executable")
            executable = Path(buffer.value[: length.value]).resolve(strict=True)
            start_token = (int(creation.high) << 32) | int(creation.low)
        finally:
            kernel32.CloseHandle(handle)
    else:
        proc = Path("/proc") / str(process_id)
        try:
            stat_text = (proc / "stat").read_text(encoding="utf-8")
            after_name = stat_text[stat_text.rfind(")") + 2 :].split()
            start_token = int(after_name[19])
            executable = (proc / "exe").resolve(strict=True)
        except FileNotFoundError:
            return None
        except (IndexError, OSError, ValueError) as error:
            raise ValueError("unable to establish local publisher process identity") from error
    return {
        "processId": process_id,
        "startToken": start_token,
        "executablePath": str(executable),
        "executableSha256": _hash_file(executable),
    }


def _local_publication_request_hash(
    *,
    bundle: CoreGraphBundle,
    dataset_id: Literal["email-eu-core", "penn94"],
    phase: Literal["smoke", "dev"],
    task_kind: Literal["node-binary", "edge-binary"],
    targets_by_entity: Mapping[str, int],
    split_inventory: LocalSplitInventory,
    source_inventory: Mapping[str, Any],
    runtime: Path,
    output: Path,
    seed: int,
    optimizer_steps: int,
    head_steps: int,
    device_name: Literal["auto", "cpu", "cuda"],
    formal_preflight_evidence_hash: str,
) -> str:
    device = _resolve_device(device_name)
    _environment_document, environment_hash = _environment(device)
    preflight_path = runtime / "experiments-core" / "formal-preflight-v2-current.json"
    preflight = load_formal_preflight(preflight_path, runtime_root=runtime)
    if (
        preflight.evidence_hash != formal_preflight_evidence_hash
        or preflight.formal_ready
        or preflight.promotable
    ):
        raise ValueError("local run formal preflight evidence is not exact and non-ready")
    source_hash = _validate_source_inventory(
        source_inventory,
        dataset_id=dataset_id,
        source_sha256=bundle.source.source_sha256,
    )["inventoryHash"]
    return canonical_sha256(
        {
            "schemaVersion": "socialgraph-fm.core-local-publication-request/1.0",
            "outputRelativePath": _relative(runtime, output),
            "datasetId": dataset_id,
            "phase": phase,
            "taskKind": task_kind,
            "baseGraphVersionHash": bundle.graph_version_hash,
            "targetsHash": _targets_hash(targets_by_entity),
            "splitInventoryHash": split_inventory.inventory_hash,
            "sourceInventoryHash": source_hash,
            "seed": seed,
            "optimizerSteps": optimizer_steps,
            "headSteps": head_steps,
            "device": device.type,
            "formalPreflightEvidenceHash": formal_preflight_evidence_hash,
            "codeHash": _code_hash(),
            "environmentHash": environment_hash,
            "formalReady": False,
        }
    )


class _LocalArtifactPublisher:
    """Own one hidden artifact directory until an atomic no-clobber rename."""

    def __init__(self, runtime: Path, target: Path, *, request_hash: str) -> None:
        self.runtime = runtime
        self.target = target
        self.request_hash = request_hash
        self.parent = _PublicationParentLease(runtime, target.parent, create=True)
        self.lock_name = f".{target.stem}.local-experiment.lock"
        self.token = uuid.uuid4().hex
        try:
            self.lock = _LocalPublisherLock(
                self.parent,
                self.lock_name,
                request_hash=request_hash,
                token=self.token,
            )
        except Exception:
            self.parent.close()
            raise
        self.staging = self.parent.parent / f".{target.name}.{self.token}.staging"
        self.journal_path = (
            self.parent.parent / f".{target.name}.{self.token}.recovery-journal.json"
        )
        self.lease: _OwnedDirectoryLease | None = None
        self.journal_lease: _LocalOwnedControlFile | None = None
        self.published_lease: _OwnedDirectoryLease | None = None
        self.published_children: tuple[_OwnedFileLease, ...] = ()
        self.published_proofs: tuple[_OwnedChildProof, ...] = ()
        self.published = False
        try:
            self.parent.make_directory(self.staging.name)
            self.identity = _path_identity(self.staging)
            self.lease = _OwnedDirectoryLease(
                self.staging,
                self.identity,
                parent_lease=self.parent,
                mutable=True,
            )
            pending_journal: dict[str, Any] = {
                "schemaVersion": "socialgraph-fm.core-local-recovery-journal-pending/1.0",
                "requestHash": request_hash,
                "ownerProcessId": os.getpid(),
            }
            pending_journal["pendingHash"] = canonical_sha256(pending_journal)
            self.journal_lease = _LocalOwnedControlFile(
                self.parent,
                self.journal_path.name,
                (canonical_json(pending_journal) + "\n").encode(),
            )
            control: dict[str, Any] = {
                "schemaVersion": "socialgraph-fm.core-local-publication-control/1.0",
                "requestHash": request_hash,
                "targetName": target.name,
                "stagingName": self.staging.name,
                "stagingIdentity": self.identity,
                "lockName": self.lock_name,
                "lockIdentity": self.lock.identity,
                "journalName": self.journal_path.name,
                "journalIdentity": self.journal_lease.identity,
            }
            control["controlHash"] = canonical_sha256(control)
            _write_new_bytes(
                self.staging / "publication-control.json",
                (canonical_json(control) + "\n").encode(),
            )
        except Exception:
            if self.journal_lease is not None:
                journal, self.journal_lease = self.journal_lease, None
                journal.remove_owned_link()
            if self.lease is not None:
                self.lease.close()
                self.lease = None
            self.lock.close()
            self.parent.close()
            raise

    def _proofs(self) -> tuple[_OwnedChildProof, ...]:
        if self.lease is None:
            raise RuntimeError("local artifact staging lease is closed")
        self.lease.assert_identity()
        entries = tuple(sorted(self.staging.iterdir(), key=lambda item: item.name))
        expected_names = _LOCAL_ARTIFACT_PAYLOAD_NAMES + _LOCAL_ARTIFACT_CONTROL_NAMES
        if tuple(entry.name for entry in entries) != tuple(sorted(expected_names)):
            raise ValueError("local artifact staging inventory is not exact before sealing")
        proofs = []
        for entry in entries:
            if not entry.is_file() or entry.is_symlink():
                raise ValueError("local artifact staging inventory must contain regular files only")
            proofs.append(
                _OwnedChildProof(
                    name=entry.name,
                    identity=_path_identity(entry),
                    sha256=_hash_file(entry),
                    size_bytes=entry.stat().st_size,
                )
            )
        return tuple(proofs)

    def commit(self, serialized_report: bytes) -> None:
        if self.lease is None or self.journal_lease is None:
            raise RuntimeError("local artifact staging lease is closed")
        _write_new_bytes(self.staging / "report.json", serialized_report)
        _LOCAL_PUBLICATION_SEAM("artifacts-before-seal", self.target)
        proofs = self._proofs()
        children = self.lease.hold_known_files(proofs)
        try:
            self.lease.flush()
            self.lease.verify_known_files(children, proofs)
            control_child = next(
                (child for child in children if child.target.name == "publication-control.json"),
                None,
            )
            if control_child is None:
                raise ValueError("local publication control is absent from sealed artifacts")
            control_bytes = control_child.read(max_bytes=4 * 1024 * 1024)
            control_document = json.loads(control_bytes)
            if (
                not isinstance(control_document, dict)
                or control_bytes != (canonical_json(control_document) + "\n").encode()
            ):
                raise ValueError("local publication control is not exact canonical JSON")
            validated_control = _validate_publication_control_document(
                control_document,
                request_hash=self.request_hash,
                target_name=self.target.name,
            )
            if (
                tuple(validated_control["stagingIdentity"]) != self.identity
                or validated_control["stagingName"] != self.staging.name
                or tuple(validated_control["lockIdentity"]) != self.lock.identity
                or validated_control["lockName"] != self.lock_name
                or tuple(validated_control["journalIdentity"]) != self.journal_lease.identity
                or validated_control["journalName"] != self.journal_path.name
            ):
                raise ValueError("local publication control differs from held ownership")
            owner = _process_identity(os.getpid())
            if owner is None:  # pragma: no cover - the current process must exist
                raise RuntimeError("local publisher process identity disappeared")
            journal_payload: dict[str, Any] = {
                "schemaVersion": "socialgraph-fm.core-local-recovery-journal/1.0",
                "requestHash": self.request_hash,
                "targetName": self.target.name,
                "stagingName": self.staging.name,
                "stagingIdentity": self.identity,
                "lockName": self.lock_name,
                "lockIdentity": self.lock.identity,
                "journalIdentity": self.journal_lease.identity,
                "ownerProcessIdentity": owner,
                "reportSha256": hashlib.sha256(serialized_report).hexdigest(),
                "payloadProofs": tuple(
                    {
                        "name": proof.name,
                        "identity": proof.identity,
                        "sha256": proof.sha256,
                        "sizeBytes": proof.size_bytes,
                    }
                    for proof in proofs
                ),
            }
            journal_payload["journalHash"] = canonical_sha256(journal_payload)
            journal_serialized = (canonical_json(journal_payload) + "\n").encode()
            self.journal_lease.replace_payload(journal_serialized)
            if self.journal_lease.read(max_bytes=4 * 1024 * 1024) != journal_serialized:
                raise ValueError("local recovery journal changed after sealing")
            _LOCAL_PUBLICATION_SEAM("artifacts-before-rename", self.target)
            self.lease.verify_known_files(children, proofs)
        finally:
            for child in children:
                child.close()
        _LOCAL_PUBLICATION_SEAM("artifacts-after-child-release", self.target)
        self.lease.close()
        self.lease = None
        _rename_directory_no_replace(self.staging, self.target)
        self.published = True
        published = _OwnedDirectoryLease(self.target, self.identity, parent_lease=self.parent)
        held = published.hold_known_files(proofs)
        self.published_lease = published
        self.published_children = held
        self.published_proofs = proofs
        try:
            _LOCAL_PUBLICATION_SEAM("artifacts-post-rename", self.target)
            published.verify_known_files(held, proofs)
            self.parent.flush()
            published.verify_known_files(held, proofs)
        except Exception:
            for child in held:
                child.close()
            self.published_children = ()
            published.close()
            self.published_lease = None
            raise

    def verify_published(self) -> None:
        if self.published_lease is None or not self.published_children:
            raise RuntimeError("local artifact publication is not held")
        self.published_lease.verify_known_files(
            self.published_children,
            self.published_proofs,
        )
        self.parent.flush()
        self.published_lease.verify_known_files(
            self.published_children,
            self.published_proofs,
        )

    def close(self, *, failed: bool) -> None:
        try:
            for child in self.published_children:
                child.close()
            self.published_children = ()
            if self.published_lease is not None:
                self.published_lease.close()
                self.published_lease = None
            if self.lease is not None:
                self.lease.close()
                self.lease = None
        finally:
            try:
                self.lock.close()
                self.parent.flush()
            finally:
                try:
                    if self.journal_lease is not None:
                        journal_lease, self.journal_lease = self.journal_lease, None
                        journal_lease.remove_owned_link()
                        self.parent.flush()
                finally:
                    self.parent.close()


@dataclass
class _RecoveredLocalPublication:
    parent: _PublicationParentLease
    published: _OwnedDirectoryLease
    children: tuple[_OwnedFileLease, ...]
    proofs: tuple[_OwnedChildProof, ...]
    journal_cleanup: _OwnedFileLease
    lock_cleanup: _OwnedFileLease | None
    posix_lock: _PublisherLock | None

    def verify(self) -> None:
        self.published.verify_known_files(self.children, self.proofs)
        self.parent.flush()
        self.published.verify_known_files(self.children, self.proofs)

    def close(self, *, cleanup_controls: bool) -> None:
        for child in self.children:
            child.close()
        self.children = ()
        self.published.close()
        try:
            if self.lock_cleanup is not None:
                lock_cleanup, self.lock_cleanup = self.lock_cleanup, None
                if cleanup_controls:
                    removed = lock_cleanup.remove_owned_link()
                    if os.name == "nt" and not removed:
                        raise RuntimeError("owned local recovery lock was not removed")
                else:
                    lock_cleanup.close()
            if self.posix_lock is not None:
                posix_lock, self.posix_lock = self.posix_lock, None
                posix_lock.close()
            if cleanup_controls:
                self.parent.flush()
                removed = self.journal_cleanup.remove_owned_link()
                if os.name == "nt" and not removed:
                    raise RuntimeError("owned local recovery journal was not removed")
            else:
                self.journal_cleanup.close()
            self.parent.flush()
        finally:
            self.parent.close()


def _recover_stale_local_publication(
    *,
    runtime: Path,
    artifacts: Path,
    request_hash: str,
    bundle: CoreGraphBundle,
    dataset_id: Literal["email-eu-core", "penn94"],
    phase: Literal["smoke", "dev"],
    task_kind: Literal["node-binary", "edge-binary"],
    targets_by_entity: Mapping[str, int],
    split_inventory: LocalSplitInventory,
    source_inventory: Mapping[str, Any],
    seed: int,
    optimizer_steps: int,
    head_steps: int,
    device_name: Literal["auto", "cpu", "cuda"],
    formal_preflight_evidence_hash: str,
) -> _RecoveredLocalPublication | None:
    """Resume one exact sealed staging tree left by a dead publisher."""

    lock_name = f".{artifacts.stem}.local-experiment.lock"
    lock_path = artifacts.parent / lock_name
    if not lock_path.exists() and not lock_path.is_symlink():
        return None
    parent = _PublicationParentLease(runtime, artifacts.parent, create=False)
    journal_read: _OwnedFileLease | None = None
    journal_cleanup: _OwnedFileLease | None = None
    lock_cleanup: _OwnedFileLease | None = None
    posix_lock: _PublisherLock | None = None
    staging_lease: _OwnedDirectoryLease | None = None
    held_children: tuple[_OwnedFileLease, ...] = ()
    published_lease: _OwnedDirectoryLease | None = None
    published_children: tuple[_OwnedFileLease, ...] = ()
    transferred = False
    try:
        journal_paths = tuple(
            sorted(
                parent.parent.glob(f".{artifacts.name}.*.recovery-journal.json"),
                key=lambda item: item.name,
            )
        )
        if len(journal_paths) != 1:
            raise RuntimeError("local experiment stale lock has no unique recovery journal")
        journal_path = journal_paths[0]
        journal_identity = _path_identity(journal_path)
        journal_read = _OwnedFileLease(
            journal_path,
            journal_identity,
            deletable=False,
            parent_lease=parent,
        )
        journal_bytes = journal_read.read(max_bytes=4 * 1024 * 1024)
        journal = json.loads(journal_bytes)
        if not isinstance(journal, dict):
            raise ValueError("local recovery journal is not a JSON object")
        expected_journal_keys = {
            "schemaVersion",
            "requestHash",
            "targetName",
            "stagingName",
            "stagingIdentity",
            "lockName",
            "lockIdentity",
            "journalIdentity",
            "ownerProcessIdentity",
            "reportSha256",
            "payloadProofs",
            "journalHash",
        }
        journal_without_hash = {
            key: value for key, value in journal.items() if key != "journalHash"
        }
        if (
            set(journal) != expected_journal_keys
            or journal.get("schemaVersion") != "socialgraph-fm.core-local-recovery-journal/1.0"
            or journal.get("journalHash") != canonical_sha256(journal_without_hash)
            or journal_bytes != (canonical_json(journal) + "\n").encode()
            or journal.get("requestHash") != request_hash
            or journal.get("targetName") != artifacts.name
            or journal.get("lockName") != lock_name
        ):
            raise ValueError("local recovery journal identity differs from exact request")
        owner = journal.get("ownerProcessIdentity")
        if not isinstance(owner, dict) or set(owner) != {
            "processId",
            "startToken",
            "executablePath",
            "executableSha256",
        }:
            raise ValueError("local recovery journal owner identity is invalid")
        owner_process_id = owner.get("processId")
        if type(owner_process_id) is not int:
            raise ValueError("local recovery journal owner PID is invalid")
        live_owner = _process_identity(owner_process_id)
        if live_owner == owner:
            raise RuntimeError("local experiment already has an active publisher")

        lock_identity_raw = journal.get("lockIdentity")
        staging_identity_raw = journal.get("stagingIdentity")
        journal_identity_raw = journal.get("journalIdentity")
        if (
            not isinstance(lock_identity_raw, (list, tuple))
            or len(lock_identity_raw) != 2
            or not all(type(value) is int for value in lock_identity_raw)
            or not isinstance(staging_identity_raw, (list, tuple))
            or len(staging_identity_raw) != 2
            or not all(type(value) is int for value in staging_identity_raw)
            or not isinstance(journal_identity_raw, (list, tuple))
            or len(journal_identity_raw) != 2
            or not all(type(value) is int for value in journal_identity_raw)
        ):
            raise ValueError("local recovery journal filesystem identity is invalid")
        lock_identity = (lock_identity_raw[0], lock_identity_raw[1])
        staging_identity = (staging_identity_raw[0], staging_identity_raw[1])
        recorded_journal_identity = (journal_identity_raw[0], journal_identity_raw[1])
        if journal_identity != recorded_journal_identity:
            raise ValueError("local recovery journal was replaced by a competitor")
        if _path_identity(lock_path) != lock_identity:
            raise ValueError("local recovery lock was replaced by a competitor")
        if os.name == "nt":
            lock_cleanup = _OwnedFileLease(
                lock_path,
                lock_identity,
                parent_lease=parent,
            )
        else:
            posix_lock = _PublisherLock(
                parent,
                lock_name,
                active_message="local experiment already has an active publisher",
            )
            if _path_identity(lock_path) != lock_identity:
                raise ValueError("local recovery lock changed while claiming exclusion")

        staging_name = journal.get("stagingName")
        if (
            not isinstance(staging_name, str)
            or Path(staging_name).name != staging_name
            or not staging_name.startswith(f".{artifacts.name}.")
            or not staging_name.endswith(".staging")
        ):
            raise ValueError("local recovery staging name is invalid")
        staging = parent.parent / staging_name
        staging_present = staging.exists() or staging.is_symlink()
        artifacts_present = artifacts.exists() or artifacts.is_symlink()
        if staging_present == artifacts_present:
            raise ValueError(
                "local recovery requires exactly one sealed staging or published artifact tree"
            )
        owned_directory = staging if staging_present else artifacts
        staging_lease = _OwnedDirectoryLease(
            owned_directory,
            staging_identity,
            parent_lease=parent,
            mutable=True,
            exclusive=True,
        )
        raw_proofs = journal.get("payloadProofs")
        if not isinstance(raw_proofs, (list, tuple)):
            raise ValueError("local recovery payload proof inventory is invalid")
        proofs: list[_OwnedChildProof] = []
        for item in raw_proofs:
            if not isinstance(item, dict) or set(item) != {
                "name",
                "identity",
                "sha256",
                "sizeBytes",
            }:
                raise ValueError("local recovery payload proof is invalid")
            identity = item["identity"]
            if (
                not isinstance(item["name"], str)
                or Path(item["name"]).name != item["name"]
                or not isinstance(identity, (list, tuple))
                or len(identity) != 2
                or not all(type(value) is int for value in identity)
                or not isinstance(item["sha256"], str)
                or type(item["sizeBytes"]) is not int
                or item["sizeBytes"] < 1
            ):
                raise ValueError("local recovery payload proof is invalid")
            proofs.append(
                _OwnedChildProof(
                    name=item["name"],
                    identity=(identity[0], identity[1]),
                    sha256=item["sha256"],
                    size_bytes=item["sizeBytes"],
                )
            )
        proof_tuple = tuple(proofs)
        held_children = staging_lease.hold_known_files(proof_tuple)
        staging_lease.verify_known_files(held_children, proof_tuple)
        child_by_name = {child.target.name: child for child in held_children}
        report_lease = child_by_name.get("report.json")
        inventory_lease = child_by_name.get("artifact-inventory.json")
        publication_control_lease = child_by_name.get("publication-control.json")
        if report_lease is None or inventory_lease is None or publication_control_lease is None:
            raise ValueError("local recovery staging omits control evidence")
        publication_control_bytes = publication_control_lease.read(max_bytes=4 * 1024 * 1024)
        publication_control_document = json.loads(publication_control_bytes)
        if (
            not isinstance(publication_control_document, dict)
            or publication_control_bytes
            != (canonical_json(publication_control_document) + "\n").encode()
        ):
            raise ValueError("local recovery publication control is not canonical")
        publication_control = _validate_publication_control_document(
            publication_control_document,
            request_hash=request_hash,
            target_name=artifacts.name,
        )
        if (
            publication_control["stagingName"] != staging_name
            or tuple(publication_control["stagingIdentity"]) != staging_identity
            or publication_control["lockName"] != lock_name
            or tuple(publication_control["lockIdentity"]) != lock_identity
            or publication_control["journalName"] != journal_path.name
            or tuple(publication_control["journalIdentity"]) != recorded_journal_identity
        ):
            raise ValueError("local recovery controls differ from sealed artifact anchor")
        journal_read.close()
        journal_read = None
        journal_cleanup = _OwnedFileLease(
            journal_path,
            recorded_journal_identity,
            parent_lease=parent,
        )
        if journal_cleanup.read(max_bytes=4 * 1024 * 1024) != journal_bytes:
            raise ValueError("local recovery journal changed while claiming exclusion")
        report_bytes = report_lease.read(max_bytes=4 * 1024 * 1024)
        if hashlib.sha256(report_bytes).hexdigest() != journal.get("reportSha256"):
            raise ValueError("local recovery report differs from sealed journal")
        recovered_report = LocalExperimentRun.model_validate_json(report_bytes)
        if report_bytes != (canonical_json(recovered_report) + "\n").encode():
            raise ValueError("local recovery report is not canonical")
        _assert_exact_local_request(
            recovered_report,
            bundle=bundle,
            dataset_id=dataset_id,
            phase=phase,
            task_kind=task_kind,
            targets_by_entity=targets_by_entity,
            split_inventory=split_inventory,
            source_inventory=source_inventory,
            seed=seed,
            optimizer_steps=optimizer_steps,
            head_steps=head_steps,
            device_name=device_name,
            formal_preflight_evidence_hash=formal_preflight_evidence_hash,
        )
        if recovered_report.code_hash != _code_hash():
            raise FileExistsError("stale local experiment code identity differs")

        proof_by_name = {proof.name: proof for proof in proof_tuple}
        inventory_bytes = inventory_lease.read(max_bytes=4 * 1024 * 1024)
        if hashlib.sha256(inventory_bytes).hexdigest() != (
            recovered_report.artifact_inventory_evidence.sha256
        ):
            raise ValueError("local recovery artifact inventory bytes differ from report")
        inventory_document = json.loads(inventory_bytes)
        inventory_without_hash = {
            key: value for key, value in inventory_document.items() if key != "inventoryHash"
        }
        if (
            set(inventory_document)
            != {"schemaVersion", "payloadFiles", "requiredControlFiles", "inventoryHash"}
            or inventory_document.get("schemaVersion")
            != "socialgraph-fm.core-local-artifact-inventory/1.0"
            or inventory_document.get("requiredControlFiles")
            != ["artifact-inventory.json", "report.json"]
            or inventory_document.get("inventoryHash") != canonical_sha256(inventory_without_hash)
            or inventory_document.get("inventoryHash")
            != recovered_report.artifact_inventory_evidence.semantic_hash
        ):
            raise ValueError("local recovery artifact inventory hash differs")
        inventory_entries = inventory_document.get("payloadFiles")
        if not isinstance(inventory_entries, list):
            raise ValueError("local recovery artifact inventory entries are invalid")
        inventory_names: set[str] = set()
        for item in inventory_entries:
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "sha256", "sizeBytes"}
                or not isinstance(item.get("name"), str)
            ):
                raise ValueError("local recovery artifact inventory entry is invalid")
            name = item["name"]
            inventory_names.add(name)
            proof = proof_by_name.get(name)
            if (
                proof is None
                or item.get("sha256") != proof.sha256
                or item.get("sizeBytes") != proof.size_bytes
            ):
                raise ValueError("local recovery artifact inventory differs from proofs")
        if set(proof_by_name) != inventory_names | {"artifact-inventory.json", "report.json"}:
            raise ValueError("local recovery staging file inventory is not exact")
        report_artifact_references = (
            recovered_report.adapter_evidence,
            recovered_report.base_bundle_evidence,
            recovered_report.calibration_evidence,
            recovered_report.code_inventory_evidence,
            recovered_report.checkpoint_evidence,
            recovered_report.recovery_evaluation_evidence,
            recovered_report.environment_evidence,
            recovered_report.head_artifact_evidence,
            recovered_report.head_report_evidence,
            recovered_report.recovery_bundle_evidence,
            recovered_report.recovery_receipt_evidence,
            recovered_report.recovery_request_evidence,
            recovered_report.source_inventory_evidence,
            recovered_report.split_inventory_evidence,
            recovered_report.targets_evidence,
        )
        for reference in report_artifact_references:
            relative = Path(reference.relative_path)
            proof = proof_by_name.get(relative.name)
            if relative.parent != artifacts.relative_to(runtime) or (
                proof is None or proof.sha256 != reference.sha256
            ):
                raise ValueError("local recovery report reference differs from staging proofs")

        for child in held_children:
            child.close()
        held_children = ()
        staging_lease.close()
        staging_lease = None
        if staging_present:
            _rename_directory_no_replace(staging, artifacts)
        published_lease = _OwnedDirectoryLease(artifacts, staging_identity, parent_lease=parent)
        published_children = published_lease.hold_known_files(proof_tuple)
        published_lease.verify_known_files(published_children, proof_tuple)
        parent.flush()
        published_lease.verify_known_files(published_children, proof_tuple)
        reopened = reopen_local_experiment_evidence(artifacts / "report.json", runtime_root=runtime)
        _assert_exact_local_request(
            reopened,
            bundle=bundle,
            dataset_id=dataset_id,
            phase=phase,
            task_kind=task_kind,
            targets_by_entity=targets_by_entity,
            split_inventory=split_inventory,
            source_inventory=source_inventory,
            seed=seed,
            optimizer_steps=optimizer_steps,
            head_steps=head_steps,
            device_name=device_name,
            formal_preflight_evidence_hash=formal_preflight_evidence_hash,
        )
        if journal_cleanup is None or published_lease is None:
            raise RuntimeError("local recovery ownership transfer is incomplete")
        recovered_publication = _RecoveredLocalPublication(
            parent=parent,
            published=published_lease,
            children=published_children,
            proofs=proof_tuple,
            journal_cleanup=journal_cleanup,
            lock_cleanup=lock_cleanup,
            posix_lock=posix_lock,
        )
        published_lease = None
        published_children = ()
        journal_cleanup = None
        lock_cleanup = None
        posix_lock = None
        transferred = True
        return recovered_publication
    finally:
        for child in published_children:
            child.close()
        if published_lease is not None:
            published_lease.close()
        for child in held_children:
            child.close()
        if staging_lease is not None:
            staging_lease.close()
        if journal_read is not None:
            journal_read.close()
        if journal_cleanup is not None:
            journal_cleanup.close()
        if lock_cleanup is not None:
            lock_cleanup.close()
        if posix_lock is not None:
            posix_lock.close()
        if not transferred:
            parent.close()


def _recovery_interpreter() -> tuple[Path, dict[str, str] | None]:
    interpreter = Path(sys.executable).resolve(strict=True)
    if os.name != "nt":
        return interpreter, None
    base_executable = Path(getattr(sys, "_base_executable", sys.executable)).resolve(strict=True)
    if base_executable == interpreter:
        return interpreter, None
    environment = os.environ.copy()
    source_root = Path(__file__).resolve(strict=True).parents[2]
    purelib = sysconfig.get_path("purelib")
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), purelib, inherited_pythonpath) if part
    )
    return base_executable, environment


def _fresh_cpu_recovery(
    *,
    artifacts: Path,
    bundle: CoreGraphBundle,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    bindings: CheckpointBindings,
    config: TrainingConfig,
    domain: str,
    seed: int,
    code_inventory: Mapping[str, Any],
    training_environment: Mapping[str, Any],
) -> tuple[LocalRecoveryReceipt, CoreGFM, BundleInputAdapter]:
    bundle_path = artifacts / "recovery-bundle.json"
    _write_new_bytes(bundle_path, (canonical_json(bundle) + "\n").encode())
    request_path = artifacts / "recovery-request.json"
    evaluation_path = artifacts / "cpu-evaluation-state.pt"
    receipt_path = artifacts / "recovery-receipt.json"
    interpreter, recovery_environment = _recovery_interpreter()
    request_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-local-recovery-request/4.0",
        "checkpointName": checkpoint_path.name,
        "bundleName": bundle_path.name,
        "evaluationArtifactName": evaluation_path.name,
        "receiptName": receipt_path.name,
        "checkpointSha256": checkpoint_sha256,
        "bindings": asdict(bindings),
        "codeInventory": dict(code_inventory),
        "trainingEnvironmentInventory": dict(training_environment),
        "interpreterPath": str(interpreter),
        "interpreterSha256": _hash_file(interpreter),
        "trainingConfig": config.to_dict(),
        "trainingSeed": seed,
        "adapterDomain": domain,
        "graphVersionHash": bundle.graph_version_hash,
        "parentProcessId": os.getpid(),
    }
    request_payload["requestHash"] = canonical_sha256(request_payload)
    _write_new_bytes(request_path, (canonical_json(request_payload) + "\n").encode())
    child = subprocess.Popen(
        [
            str(interpreter),
            "-m",
            "socialgraph_gfm.core.local_recovery",
            "--request",
            str(request_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=recovery_environment,
    )
    _stdout, stderr = child.communicate()
    if child.returncode != 0:
        raise ValueError(f"fresh CPU recovery subprocess failed: {stderr.strip()}")
    receipt = LocalRecoveryReceipt.model_validate_json(receipt_path.read_bytes())
    expected_bindings = asdict(bindings)
    expected_recovery_environment = local_environment_inventory("cpu")
    if (
        receipt.recovery_process_id != child.pid
        or receipt.recovery_process_id == os.getpid()
        or receipt.recovery_parent_process_id != os.getpid()
        or receipt.recovery_device != "cpu"
        or receipt.recovery_interpreter_path != str(interpreter)
        or receipt.recovery_interpreter_sha256 != _hash_file(interpreter)
        or receipt.checkpoint_sha256 != checkpoint_sha256
        or receipt.config_hash != expected_bindings["config_hash"]
        or receipt.data_hash != expected_bindings["data_hash"]
        or receipt.code_hash != expected_bindings["code_hash"]
        or receipt.environment_hash != expected_bindings["environment_hash"]
        or receipt.recovery_environment_inventory != expected_recovery_environment
        or receipt.recovery_environment_hash != expected_recovery_environment.inventory_hash
        or receipt.evaluation_artifact_sha256 != _hash_file(evaluation_path)
    ):
        raise ValueError("fresh CPU recovery receipt differs from the direct child identities")
    observed = torch.load(evaluation_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(observed, dict)
        or observed.get("schemaVersion")
        != "socialgraph-fm.core-local-cpu-evaluation-state/1.0"
        or observed.get("requestHash") != receipt.request_hash
    ):
        raise ValueError("fresh CPU evaluation artifact is invalid")
    recovered_model = CoreGFM(node_classes=2)
    recovered_adapter = BundleInputAdapter(
        bundle,
        mode="training",
        schema=AdapterSchema.model_validate(observed["adapterSchema"], strict=False),
    )
    recovered_model.load_state_dict(observed["model"], strict=True)
    recovered_adapter.load_state_dict(observed["adapter"], strict=True)
    if (
        _state_hash(recovered_model.state_dict()) != receipt.model_state_hash
        or _state_hash(recovered_adapter.state_dict()) != receipt.adapter_state_hash
        or recovered_adapter.schema.adapter_schema_hash != receipt.adapter_schema_hash
        or next(recovered_model.parameters()).device.type != "cpu"
    ):
        raise ValueError("fresh CPU evaluation state differs from recovery receipt")
    return receipt, recovered_model, recovered_adapter


def _run_local_nonpromotable_experiment_unpublished(
    *,
    bundle: CoreGraphBundle,
    dataset_id: Literal["email-eu-core", "penn94"],
    phase: Literal["smoke", "dev"],
    task_kind: Literal["node-binary", "edge-binary"],
    targets_by_entity: Mapping[str, int],
    split_inventory: LocalSplitInventory,
    source_inventory: Mapping[str, Any],
    runtime_root: Path,
    output_path: Path,
    seed: int,
    optimizer_steps: int,
    head_steps: int,
    device_name: Literal["auto", "cpu", "cuda"],
    formal_preflight_evidence_hash: str,
    formal_ready: bool,
    _artifacts: Path,
    _published_artifacts: Path,
) -> LocalExperimentRun:
    """Run real local training while structurally preventing formal promotion."""

    if formal_ready:
        raise ValueError("local runs are never formal-ready")
    targets_document = _target_inventory(targets_by_entity)
    targets_hash = str(targets_document["inventoryHash"])
    if split_inventory.dataset_id != dataset_id:
        raise ValueError("local split inventory belongs to a different dataset")
    if split_inventory.selected_fold_id != "fold-0":
        raise ValueError("local smoke/dev must explicitly select fold-0")
    selected_fold = split_inventory.folds[0]
    if selected_fold.split_manifest != bundle.split_manifest:
        raise ValueError("selected fold-0 differs from the supplied training bundle")
    if (dataset_id, phase, task_kind) not in {
        ("email-eu-core", "smoke", "edge-binary"),
        ("penn94", "smoke", "node-binary"),
        ("penn94", "dev", "node-binary"),
    }:
        raise ValueError("local dataset, phase, and task kind combination is not approved")
    if phase == "smoke" and not 1 <= optimizer_steps <= 20:
        raise ValueError("smoke optimizer steps must be between 1 and 20")
    if phase == "dev" and not 1 <= optimizer_steps <= 2_000:
        raise ValueError("dev optimizer steps must be between 1 and 2,000")
    if not 1 <= head_steps <= 20:
        raise ValueError("local head steps must be between 1 and 20")
    runtime = runtime_root.resolve(strict=True)
    output = output_path.resolve()
    try:
        output.relative_to(runtime)
    except ValueError as error:
        raise ValueError("local report output must remain inside runtime root") from error
    artifacts = _artifacts
    published_artifacts = _published_artifacts

    device = _resolve_device(device_name)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    cache = build_structure_cache(
        bundle,
        cache_root=runtime / "experiments-core" / "cache",
        role="training",
    )
    enriched = enrich_bundle_with_structure(bundle, cache)
    config = (
        TrainingConfig.smoke(max_steps=optimizer_steps)
        if phase == "smoke"
        else TrainingConfig.dev(max_steps=optimizer_steps)
    )
    adapter = BundleInputAdapter(enriched, mode="training")
    edge_index = _edge_index(enriched)
    policy = ExecutionPolicy(
        full_batch_edge_threshold=config.full_batch_edge_threshold,
        node_batch_size=config.node_batch_size,
        edge_batch_size=config.edge_batch_size,
        fanout=config.fanout,
    )
    prepared = PreparedGraph.from_edge_index(
        num_nodes=len(enriched.nodes),
        edge_index=edge_index,
        directed=enriched.directed,
    )
    if policy.mode(edge_count=edge_index.shape[1]) == "full-batch":
        prepared = prepared.to(device)
    code_inventory = _code_inventory()
    code_hash = str(code_inventory["inventoryHash"])
    environment, environment_hash = _environment(device)
    source_inventory_document = _validate_source_inventory(
        source_inventory,
        dataset_id=dataset_id,
        source_sha256=bundle.source.source_sha256,
    )
    code_inventory_evidence = _publish_json_evidence(
        runtime,
        artifacts / "code-inventory.json",
        code_inventory,
        semantic_hash=code_hash,
        published_path=published_artifacts / "code-inventory.json",
    )
    environment_evidence = _publish_json_evidence(
        runtime,
        artifacts / "environment-inventory.json",
        environment,
        semantic_hash=environment_hash,
        published_path=published_artifacts / "environment-inventory.json",
    )
    source_inventory_evidence = _publish_json_evidence(
        runtime,
        artifacts / "source-inventory.json",
        source_inventory_document,
        semantic_hash=str(source_inventory_document["inventoryHash"]),
        published_path=published_artifacts / "source-inventory.json",
    )
    split_inventory_evidence = _publish_json_evidence(
        runtime,
        artifacts / "split-inventory.json",
        split_inventory,
        semantic_hash=split_inventory.inventory_hash,
        published_path=published_artifacts / "split-inventory.json",
    )
    targets_evidence = _publish_json_evidence(
        runtime,
        artifacts / "target-inventory.json",
        targets_document,
        semantic_hash=targets_hash,
        published_path=published_artifacts / "target-inventory.json",
    )
    base_bundle_evidence = _publish_json_evidence(
        runtime,
        artifacts / "base-bundle.json",
        bundle,
        semantic_hash=bundle.graph_version_hash,
        published_path=published_artifacts / "base-bundle.json",
    )
    formal_preflight_path = runtime / "experiments-core" / "formal-preflight-v2-current.json"
    formal_preflight = load_formal_preflight(formal_preflight_path, runtime_root=runtime)
    if (
        formal_preflight.evidence_hash != formal_preflight_evidence_hash
        or formal_preflight.formal_ready
        or formal_preflight.promotable
    ):
        raise ValueError("local run formal preflight evidence is not exact and non-ready")
    formal_preflight_evidence = _existing_file_evidence(
        runtime,
        formal_preflight_path,
        semantic_hash=formal_preflight_evidence_hash,
    )
    structure_manifest_evidence = _existing_file_evidence(
        runtime,
        cache.manifest_path,
        semantic_hash=cache.manifest.manifest_hash,
    )
    structure_npz_evidence = _existing_file_evidence(
        runtime,
        cache.npz_path,
        semantic_hash=cache.manifest.npz_sha256,
    )
    config_hash = canonical_sha256({"phase": phase, "seed": seed, "training": config.to_dict()})
    data_hash = canonical_sha256(
        {
            "baseGraphVersionHash": bundle.graph_version_hash,
            "enrichedGraphVersionHash": enriched.graph_version_hash,
            "structureManifestHash": cache.manifest.manifest_hash,
            "taskKind": task_kind,
            "targetsHash": targets_hash,
            "splitInventoryHash": split_inventory.inventory_hash,
            "selectedFoldId": split_inventory.selected_fold_id,
            "sourceInventoryHash": source_inventory_document["inventoryHash"],
        }
    )
    bindings = CheckpointBindings(
        config_hash=config_hash,
        data_hash=data_hash,
        code_hash=code_hash,
        environment_hash=environment_hash,
    )
    domain = f"{dataset_id}::{phase}"
    model = CoreGFM(node_classes=2).to(device)
    trainer = CoreTrainer(
        model,
        {
            domain: TrainingGraph.from_bundle(
                adapter=adapter,
                graph=prepared,
                execution_policy=policy,
            )
        },
        config=config,
        seed=seed,
    )
    trainer.run_steps(optimizer_steps)
    checkpoint_path = artifacts / "core-training.pt"
    trainer.save_checkpoint(checkpoint_path, bindings=bindings)
    checkpoint_sha256 = _hash_file(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, expected_bindings=bindings)
    if checkpoint.get("status") != "training" or checkpoint.get("promotable") is not False:
        raise ValueError("local checkpoint unexpectedly claims promotion eligibility")
    recovery, recovered_model, recovered_adapter = _fresh_cpu_recovery(
        artifacts=artifacts,
        bundle=enriched,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        bindings=bindings,
        config=config,
        domain=domain,
        seed=seed,
        code_inventory=code_inventory,
        training_environment=environment,
    )
    recovery_bundle_evidence = _existing_file_evidence(
        runtime,
        artifacts / "recovery-bundle.json",
        semantic_hash=enriched.graph_version_hash,
        published_path=published_artifacts / "recovery-bundle.json",
    )
    recovery_request_evidence = _existing_file_evidence(
        runtime,
        artifacts / "recovery-request.json",
        semantic_hash=recovery.request_hash,
        published_path=published_artifacts / "recovery-request.json",
    )
    recovery_evaluation_evidence = _existing_file_evidence(
        runtime,
        artifacts / "cpu-evaluation-state.pt",
        semantic_hash=recovery.recovery_state_hash,
        published_path=published_artifacts / "cpu-evaluation-state.pt",
    )
    recovery_receipt_evidence = _existing_file_evidence(
        runtime,
        artifacts / "recovery-receipt.json",
        semantic_hash=recovery.receipt_hash,
        published_path=published_artifacts / "recovery-receipt.json",
    )
    checkpoint_evidence = _existing_file_evidence(
        runtime,
        checkpoint_path,
        semantic_hash=recovery.trainer_state_hash,
        published_path=published_artifacts / checkpoint_path.name,
    )
    evaluation_device = torch.device("cpu")
    encoded = encode_supervised_graph(recovered_model, enriched, recovered_adapter)
    train = _partition(
        enriched,
        task_kind=task_kind,
        role="train",
        targets_by_entity=targets_by_entity,
    )
    validation = _partition(
        enriched,
        task_kind=task_kind,
        role="validation",
        targets_by_entity=targets_by_entity,
    )
    supervised = SupervisedTrainValidation.create(
        task_kind=task_kind,
        provenance=encoded.provenance,
        train=train,
        validation=validation,
    )
    head_report = fit_supervised_head(
        recovered_model,
        encoded,
        supervised,
        config=HeadTrainingConfig.smoke(max_steps=head_steps),
    )
    calibration_report_hash: str | None = None
    calibration_promotion_eligible: Literal[False] | None = None
    calibration: CalibrationFitReport | None = None
    if set(validation.targets) == {0, 1}:
        scores = derive_validation_scores(
            recovered_model,
            encoded,
            supervised,
            head_report,
            semantics=BinaryScoreSemantics.for_task(task_kind),
        )
        calibration = fit_score_calibration_report(
            scores,
            protocol=CalibrationProtocol.fixed(scores),
        )
        calibration_report_hash = calibration.report_hash
        if calibration.promotion_eligible:
            raise ValueError("local calibration unexpectedly claims promotion eligibility")
        calibration_promotion_eligible = False
    head_artifact_path = artifacts / "head-evaluation.pt"
    head_artifact_hash, head_artifact_semantic_hash = _publish_head_artifact(
        head_artifact_path,
        model=recovered_model,
        bundle=enriched,
        adapter=recovered_adapter,
        head_report_hash=head_report.report_hash,
    )
    head_artifact_evidence = _existing_file_evidence(
        runtime,
        head_artifact_path,
        semantic_hash=head_artifact_semantic_hash,
        published_path=published_artifacts / head_artifact_path.name,
    )
    head_report_evidence = _publish_json_evidence(
        runtime,
        artifacts / "head-training-report.json",
        head_report.record,
        semantic_hash=head_report.report_hash,
        published_path=published_artifacts / "head-training-report.json",
    )
    if calibration is None:
        calibration_document: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-local-calibration-evidence/1.0",
            "status": "unavailable-single-class-validation",
        }
        calibration_document["evidenceHash"] = canonical_sha256(calibration_document)
        calibration_semantic_hash = str(calibration_document["evidenceHash"])
    else:
        calibration_document = calibration.model_dump(mode="python", by_alias=True)
        calibration_semantic_hash = calibration.report_hash
    calibration_evidence = _publish_json_evidence(
        runtime,
        artifacts / "calibration-report.json",
        calibration_document,
        semantic_hash=calibration_semantic_hash,
        published_path=published_artifacts / "calibration-report.json",
    )
    adapter_document: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-local-adapter-evidence/1.0",
        "adapterSchema": recovered_adapter.schema.model_dump(mode="python", by_alias=True),
        "adapterSchemaHash": recovered_adapter.schema.adapter_schema_hash,
        "adapterStateHash": _state_hash(recovered_adapter.state_dict()),
        "headArtifactRelativePath": _relative(
            runtime, published_artifacts / head_artifact_path.name
        ),
        "headArtifactSha256": head_artifact_hash,
        "headArtifactSemanticHash": head_artifact_semantic_hash,
    }
    adapter_document["evidenceHash"] = canonical_sha256(adapter_document)
    adapter_evidence = _publish_json_evidence(
        runtime,
        artifacts / "adapter-evidence.json",
        adapter_document,
        semantic_hash=str(adapter_document["evidenceHash"]),
        published_path=published_artifacts / "adapter-evidence.json",
    )
    artifact_inventory_document = _artifact_inventory_document(artifacts)
    artifact_inventory_evidence = _publish_json_evidence(
        runtime,
        artifacts / "artifact-inventory.json",
        artifact_inventory_document,
        semantic_hash=str(artifact_inventory_document["inventoryHash"]),
        published_path=published_artifacts / "artifact-inventory.json",
    )
    limitations = [
        "Local smoke/dev evidence only; it is excluded from formal acceptance and promotion.",
        "Outputs are research evidence for human review and never automatic governance action.",
        "Automatic crash-recovery cleanup of ownership journals and lock pathnames is "
        "Windows-only; POSIX preserves identity-safe audit orphans for operator quarantine.",
        "Failed staging trees are preserved rather than expanding deletion ownership from a "
        "post-failure directory scan.",
    ]
    if dataset_id == "email-eu-core":
        limitations.append(
            "The Email sanity head uses known positive edges only; it is not a calibrated relation-completion metric."
        )
    else:
        limitations.append(
            "Penn94 gender is an offline benchmark only and is excluded from governance serving."
        )
        limitations.append(
            "This local dev run selects official fold-0 only; formal evaluation consumes all five "
            "official folds."
        )
    run_identity = {
        "datasetId": dataset_id,
        "phase": phase,
        "taskKind": task_kind,
        "seed": seed,
        "configHash": config_hash,
        "dataHash": data_hash,
        "codeHash": code_hash,
        "environmentHash": environment_hash,
        "targetsHash": targets_hash,
        "codeInventoryEvidence": code_inventory_evidence.model_dump(mode="python", by_alias=True),
        "environmentEvidence": environment_evidence.model_dump(mode="python", by_alias=True),
        "sourceInventoryEvidence": source_inventory_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "splitInventoryEvidence": split_inventory_evidence.model_dump(mode="python", by_alias=True),
        "targetsEvidence": targets_evidence.model_dump(mode="python", by_alias=True),
        "baseBundleEvidence": base_bundle_evidence.model_dump(mode="python", by_alias=True),
        "adapterEvidence": adapter_evidence.model_dump(mode="python", by_alias=True),
        "headReportEvidence": head_report_evidence.model_dump(mode="python", by_alias=True),
        "calibrationEvidence": calibration_evidence.model_dump(mode="python", by_alias=True),
        "formalPreflightEvidence": formal_preflight_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "recoveryBundleEvidence": recovery_bundle_evidence.model_dump(mode="python", by_alias=True),
        "recoveryRequestEvidence": recovery_request_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "recoveryEvaluationEvidence": recovery_evaluation_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "recoveryReceiptEvidence": recovery_receipt_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "checkpointEvidence": checkpoint_evidence.model_dump(mode="python", by_alias=True),
        "headArtifactEvidence": head_artifact_evidence.model_dump(mode="python", by_alias=True),
        "structureManifestEvidence": structure_manifest_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "structureNpzEvidence": structure_npz_evidence.model_dump(mode="python", by_alias=True),
        "artifactInventoryEvidence": artifact_inventory_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "formalPreflightEvidenceHash": formal_preflight_evidence_hash,
        "splitInventoryHash": split_inventory.inventory_hash,
        "selectedFoldId": split_inventory.selected_fold_id,
    }
    payload: dict[str, Any] = {
        "schemaVersion": _SCHEMA,
        "runId": canonical_sha256(run_identity),
        "datasetId": dataset_id,
        "phase": phase,
        "taskKind": task_kind,
        "seed": seed,
        "device": device.type,
        "evaluationDevice": evaluation_device.type,
        "optimizerSteps": trainer.optimizer_step,
        "headSteps": head_steps,
        "nodeCount": len(enriched.nodes),
        "edgeCount": len(enriched.edges),
        "baseGraphVersionHash": bundle.graph_version_hash,
        "enrichedGraphVersionHash": enriched.graph_version_hash,
        "sourceSha256": bundle.source.source_sha256,
        "targetsHash": targets_hash,
        "splitInventoryHash": split_inventory.inventory_hash,
        "foldIds": split_inventory.fold_ids,
        "selectedFoldId": split_inventory.selected_fold_id,
        "selectedSplitManifestHash": selected_fold.split_manifest_hash,
        "selectedSplitRoleCounts": selected_fold.role_counts,
        "structureManifestHash": cache.manifest.manifest_hash,
        "adapterSchemaHash": recovered_adapter.schema.adapter_schema_hash,
        "configHash": config_hash,
        "dataHash": data_hash,
        "codeHash": code_hash,
        "environmentHash": environment_hash,
        "codeInventoryEvidence": code_inventory_evidence.model_dump(mode="python", by_alias=True),
        "environmentEvidence": environment_evidence.model_dump(mode="python", by_alias=True),
        "sourceInventoryEvidence": source_inventory_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "splitInventoryEvidence": split_inventory_evidence.model_dump(mode="python", by_alias=True),
        "targetsEvidence": targets_evidence.model_dump(mode="python", by_alias=True),
        "baseBundleEvidence": base_bundle_evidence.model_dump(mode="python", by_alias=True),
        "adapterEvidence": adapter_evidence.model_dump(mode="python", by_alias=True),
        "headReportEvidence": head_report_evidence.model_dump(mode="python", by_alias=True),
        "calibrationEvidence": calibration_evidence.model_dump(mode="python", by_alias=True),
        "formalPreflightEvidence": formal_preflight_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "recoveryBundleEvidence": recovery_bundle_evidence.model_dump(mode="python", by_alias=True),
        "recoveryRequestEvidence": recovery_request_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "recoveryEvaluationEvidence": recovery_evaluation_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "recoveryReceiptEvidence": recovery_receipt_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "checkpointEvidence": checkpoint_evidence.model_dump(mode="python", by_alias=True),
        "headArtifactEvidence": head_artifact_evidence.model_dump(mode="python", by_alias=True),
        "structureManifestEvidence": structure_manifest_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "structureNpzEvidence": structure_npz_evidence.model_dump(mode="python", by_alias=True),
        "artifactInventoryEvidence": artifact_inventory_evidence.model_dump(
            mode="python", by_alias=True
        ),
        "checkpointRelativePath": _relative(runtime, published_artifacts / checkpoint_path.name),
        "checkpointSha256": checkpoint_sha256,
        "checkpointStatus": "training",
        "checkpointPromotable": False,
        "freshCompositeStateHash": recovery.composite_state_hash,
        "freshRecoveryStateHash": recovery.recovery_state_hash,
        "recoveryProcessId": recovery.recovery_process_id,
        "recoveryParentProcessId": recovery.recovery_parent_process_id,
        "recoveryDevice": recovery.recovery_device,
        "recoveryReceiptRelativePath": _relative(
            runtime, published_artifacts / "recovery-receipt.json"
        ),
        "recoveryReceiptSha256": _hash_file(artifacts / "recovery-receipt.json"),
        "recoveryReceiptHash": recovery.receipt_hash,
        "supervisedDataHash": supervised.data_hash,
        "encodedArtifactHash": encoded.provenance.artifact_hash,
        "headArtifactRelativePath": _relative(
            runtime, published_artifacts / head_artifact_path.name
        ),
        "headArtifactSha256": head_artifact_hash,
        "headReportHash": head_report.report_hash,
        "headStateHash": head_report.head_state_hash,
        "headPromotionEligible": False,
        "calibrationReportHash": calibration_report_hash,
        "calibrationPromotionEligible": calibration_promotion_eligible,
        "formalPreflightEvidenceHash": formal_preflight_evidence_hash,
        "formalReady": False,
        "promotable": False,
        "failedGates": ["phase-not-formal", "formal-corpus-not-ready"],
        "limitations": limitations,
    }
    payload["reportHash"] = canonical_sha256(payload)
    report = LocalExperimentRun.model_validate(payload)
    _ = environment  # The hash-bound payload is intentionally not duplicated in the report.
    return report


def _assert_exact_local_request(
    report: LocalExperimentRun,
    *,
    bundle: CoreGraphBundle,
    dataset_id: Literal["email-eu-core", "penn94"],
    phase: Literal["smoke", "dev"],
    task_kind: Literal["node-binary", "edge-binary"],
    targets_by_entity: Mapping[str, int],
    split_inventory: LocalSplitInventory,
    source_inventory: Mapping[str, Any],
    seed: int,
    optimizer_steps: int,
    head_steps: int,
    device_name: Literal["auto", "cpu", "cuda"],
    formal_preflight_evidence_hash: str,
) -> None:
    requested_device = _resolve_device(device_name).type
    source_inventory_hash = _validate_source_inventory(
        source_inventory,
        dataset_id=dataset_id,
        source_sha256=bundle.source.source_sha256,
    )["inventoryHash"]
    if (
        report.dataset_id != dataset_id
        or report.phase != phase
        or report.task_kind != task_kind
        or report.base_graph_version_hash != bundle.graph_version_hash
        or report.split_inventory_hash != split_inventory.inventory_hash
        or report.seed != seed
        or report.optimizer_steps != optimizer_steps
        or report.head_steps != head_steps
        or report.device != requested_device
        or report.formal_preflight_evidence_hash != formal_preflight_evidence_hash
        or report.source_inventory_evidence.semantic_hash != source_inventory_hash
        or report.targets_hash != _targets_hash(targets_by_entity)
    ):
        raise FileExistsError("different local experiment evidence already exists")


def _run_local_nonpromotable_experiment_guarded(
    *,
    bundle: CoreGraphBundle,
    dataset_id: Literal["email-eu-core", "penn94"],
    phase: Literal["smoke", "dev"],
    task_kind: Literal["node-binary", "edge-binary"],
    targets_by_entity: Mapping[str, int],
    split_inventory: LocalSplitInventory,
    source_inventory: Mapping[str, Any],
    runtime_root: Path,
    output_path: Path,
    seed: int,
    optimizer_steps: int,
    head_steps: int,
    device_name: Literal["auto", "cpu", "cuda"],
    formal_preflight_evidence_hash: str,
    formal_ready: bool,
    _authority_snapshots: tuple[_HeldLocalSourceSnapshot, ...],
) -> LocalExperimentRun:
    if formal_ready:
        raise ValueError("local runs are never formal-ready")
    _targets_hash(targets_by_entity)
    runtime = secure_existing_root(runtime_root)
    output = reject_link_components(output_path)
    try:
        output.relative_to(runtime)
    except ValueError as error:
        raise ValueError("local report output must remain inside runtime root") from error
    artifacts = output.with_suffix(".artifacts")
    publication_request_hash = _local_publication_request_hash(
        bundle=bundle,
        dataset_id=dataset_id,
        phase=phase,
        task_kind=task_kind,
        targets_by_entity=targets_by_entity,
        split_inventory=split_inventory,
        source_inventory=source_inventory,
        runtime=runtime,
        output=output,
        seed=seed,
        optimizer_steps=optimizer_steps,
        head_steps=head_steps,
        device_name=device_name,
        formal_preflight_evidence_hash=formal_preflight_evidence_hash,
    )
    stale_lock = artifacts.parent / f".{artifacts.stem}.local-experiment.lock"
    recovered_publication: _RecoveredLocalPublication | None = None
    if (
        not output.exists()
        and not output.is_symlink()
        and (stale_lock.exists() or stale_lock.is_symlink())
    ):
        recovered_publication = _recover_stale_local_publication(
            runtime=runtime,
            artifacts=artifacts,
            request_hash=publication_request_hash,
            bundle=bundle,
            dataset_id=dataset_id,
            phase=phase,
            task_kind=task_kind,
            targets_by_entity=targets_by_entity,
            split_inventory=split_inventory,
            source_inventory=source_inventory,
            seed=seed,
            optimizer_steps=optimizer_steps,
            head_steps=head_steps,
            device_name=device_name,
            formal_preflight_evidence_hash=formal_preflight_evidence_hash,
        )
    if output.exists() or output.is_symlink():
        try:
            report = reopen_local_experiment_evidence(output, runtime_root=runtime)
            _assert_exact_local_request(
                report,
                bundle=bundle,
                dataset_id=dataset_id,
                phase=phase,
                task_kind=task_kind,
                targets_by_entity=targets_by_entity,
                split_inventory=split_inventory,
                source_inventory=source_inventory,
                seed=seed,
                optimizer_steps=optimizer_steps,
                head_steps=head_steps,
                device_name=device_name,
                formal_preflight_evidence_hash=formal_preflight_evidence_hash,
            )
            return report
        except FileExistsError:
            raise
        except Exception as error:
            raise FileExistsError("different local experiment evidence already exists") from error
    if artifacts.exists() or artifacts.is_symlink():
        if recovered_publication is None:
            raise FileExistsError(
                "local experiment artifacts lack exact stale-publication ownership"
            )
        internal = artifacts / "report.json"
        cleanup_recovery_controls = False
        try:
            report = reopen_local_experiment_evidence(internal, runtime_root=runtime)
            _assert_exact_local_request(
                report,
                bundle=bundle,
                dataset_id=dataset_id,
                phase=phase,
                task_kind=task_kind,
                targets_by_entity=targets_by_entity,
                split_inventory=split_inventory,
                source_inventory=source_inventory,
                seed=seed,
                optimizer_steps=optimizer_steps,
                head_steps=head_steps,
                device_name=device_name,
                formal_preflight_evidence_hash=formal_preflight_evidence_hash,
            )
            _assert_held_authoritative_sources(_authority_snapshots)
            recovered_publication.verify()
            serialized = (canonical_json(report) + "\n").encode()
            _LOCAL_PUBLICATION_SEAM("artifacts-before-external-report", artifacts)
            recovered_publication.verify()
            _publish_exact(
                runtime,
                output,
                serialized,
                conflict_message="different local experiment evidence already exists",
            )
            reopened = reopen_local_experiment_evidence(output, runtime_root=runtime)
            if reopened != report:
                raise ValueError("local recovered report differs after exact publication")
            _assert_held_authoritative_sources(_authority_snapshots)
            recovered_publication.verify()
            cleanup_recovery_controls = True
            return reopened
        except FileExistsError:
            raise
        except Exception as error:
            raise FileExistsError("different local experiment artifacts already exist") from error
        finally:
            recovered_publication.close(cleanup_controls=cleanup_recovery_controls)
    publisher = _LocalArtifactPublisher(
        runtime,
        artifacts,
        request_hash=publication_request_hash,
    )
    failed = True
    try:
        report = _run_local_nonpromotable_experiment_unpublished(
            bundle=bundle,
            dataset_id=dataset_id,
            phase=phase,
            task_kind=task_kind,
            targets_by_entity=targets_by_entity,
            split_inventory=split_inventory,
            source_inventory=source_inventory,
            runtime_root=runtime,
            output_path=output,
            seed=seed,
            optimizer_steps=optimizer_steps,
            head_steps=head_steps,
            device_name=device_name,
            formal_preflight_evidence_hash=formal_preflight_evidence_hash,
            formal_ready=False,
            _artifacts=publisher.staging,
            _published_artifacts=artifacts,
        )
        serialized = (canonical_json(report) + "\n").encode()
        _LOCAL_PUBLICATION_SEAM("authority-before-artifact-commit", artifacts)
        _assert_held_authoritative_sources(_authority_snapshots)
        publisher.commit(serialized)
        publisher.verify_published()
        internal_reopened = reopen_local_experiment_evidence(
            artifacts / "report.json", runtime_root=runtime
        )
        if internal_reopened != report:
            raise ValueError("local internal report differs after artifact publication")
        _assert_held_authoritative_sources(_authority_snapshots)
        _LOCAL_PUBLICATION_SEAM("artifacts-before-external-report", artifacts)
        publisher.verify_published()
        _publish_exact(
            runtime,
            output,
            serialized,
            conflict_message="different local experiment evidence already exists",
        )
        reopened = reopen_local_experiment_evidence(output, runtime_root=runtime)
        if reopened != report:
            raise ValueError("local report differs after hardened exact publication")
        _assert_held_authoritative_sources(_authority_snapshots)
        publisher.verify_published()
        failed = False
        return reopened
    finally:
        publisher.close(failed=failed)


def run_local_nonpromotable_experiment(
    *,
    bundle: CoreGraphBundle,
    dataset_id: Literal["email-eu-core", "penn94"],
    phase: Literal["smoke", "dev"],
    task_kind: Literal["node-binary", "edge-binary"],
    targets_by_entity: Mapping[str, int],
    split_inventory: LocalSplitInventory,
    source_inventory: Mapping[str, Any],
    runtime_root: Path,
    output_path: Path,
    seed: int,
    optimizer_steps: int,
    head_steps: int,
    device_name: Literal["auto", "cpu", "cuda"],
    formal_preflight_evidence_hash: str,
    formal_ready: bool,
) -> LocalExperimentRun:
    """Publish one exact local run, or return its exact validated replay."""

    if formal_ready:
        raise ValueError("local runs are never formal-ready")
    runtime = secure_existing_root(runtime_root)
    output = reject_link_components(output_path)
    try:
        output.relative_to(runtime)
    except ValueError as error:
        raise ValueError("local report output must remain inside runtime root") from error
    artifacts = output.with_suffix(".artifacts")
    try:
        authority_snapshots = _assert_authoritative_local_inputs(
            runtime=runtime,
            dataset_id=dataset_id,
            bundle=bundle,
            targets_by_entity=targets_by_entity,
            split_inventory=split_inventory,
            source_inventory=source_inventory,
        )
    except ValueError as error:
        if output.exists() or output.is_symlink() or artifacts.exists() or artifacts.is_symlink():
            raise FileExistsError("different local experiment evidence already exists") from error
        raise
    try:
        return _run_local_nonpromotable_experiment_guarded(
            bundle=bundle,
            dataset_id=dataset_id,
            phase=phase,
            task_kind=task_kind,
            targets_by_entity=targets_by_entity,
            split_inventory=split_inventory,
            source_inventory=source_inventory,
            runtime_root=runtime,
            output_path=output,
            seed=seed,
            optimizer_steps=optimizer_steps,
            head_steps=head_steps,
            device_name=device_name,
            formal_preflight_evidence_hash=formal_preflight_evidence_hash,
            formal_ready=False,
            _authority_snapshots=authority_snapshots,
        )
    finally:
        _close_held_authoritative_sources(authority_snapshots)


def _evidence_bytes(
    root: Path,
    reference: EvidenceReference,
    *,
    max_bytes: int = 64 * 1024 * 1024,
) -> bytes:
    snapshot = read_confined_snapshot(root, reference.relative_path, max_bytes=max_bytes)
    if hashlib.sha256(snapshot).hexdigest() != reference.sha256:
        raise ValueError("local evidence bytes differ from report SHA-256")
    return snapshot


def _recompute_document_hash(document: dict[str, Any], field: str) -> str:
    observed = document.get(field)
    if not isinstance(observed, str):
        raise ValueError(f"local evidence {field} is missing")
    without_hash = {key: value for key, value in document.items() if key != field}
    if canonical_sha256(without_hash) != observed:
        raise ValueError(f"local evidence {field} does not match its document")
    return observed


def _reopen_local_experiment_evidence_guarded(
    report_path: Path,
    *,
    runtime_root: Path,
    _authority_snapshot_sink: list[_HeldLocalSourceSnapshot],
) -> LocalExperimentRun:
    """Reopen every immutable local-evidence file and recompute its semantic identity."""

    root = secure_existing_root(runtime_root)
    report_lexical = reject_link_components(report_path)
    try:
        relative_report = report_lexical.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("local report must remain inside the authorized runtime") from error
    report_snapshot = read_confined_snapshot(root, relative_report, max_bytes=4 * 1024 * 1024)
    report = LocalExperimentRun.model_validate_json(report_snapshot)
    canonical_report = (canonical_json(report) + "\n").encode()
    if report_snapshot != canonical_report:
        raise ValueError("local report bytes are not the canonical immutable encoding")
    artifact_relative, _artifact_inventory = _validate_artifact_inventory(
        root, report.artifact_inventory_evidence
    )
    publication_control_snapshot = read_confined_snapshot(
        root,
        (artifact_relative / "publication-control.json").as_posix(),
        max_bytes=4 * 1024 * 1024,
    )
    publication_control_document = json.loads(publication_control_snapshot)
    if (
        not isinstance(publication_control_document, dict)
        or publication_control_snapshot
        != (canonical_json(publication_control_document) + "\n").encode()
    ):
        raise ValueError("local publication control is not exact canonical JSON")
    _validate_publication_control_document(
        publication_control_document,
        target_name=artifact_relative.name,
    )
    internal_report = read_confined_snapshot(
        root, (artifact_relative / "report.json").as_posix(), max_bytes=4 * 1024 * 1024
    )
    if internal_report != canonical_report:
        raise ValueError("local internal report mirror differs from canonical report bytes")

    artifact_references = {
        "adapter-evidence.json": report.adapter_evidence,
        "base-bundle.json": report.base_bundle_evidence,
        "calibration-report.json": report.calibration_evidence,
        "code-inventory.json": report.code_inventory_evidence,
        "core-training.pt": report.checkpoint_evidence,
        "cpu-evaluation-state.pt": report.recovery_evaluation_evidence,
        "environment-inventory.json": report.environment_evidence,
        "head-evaluation.pt": report.head_artifact_evidence,
        "head-training-report.json": report.head_report_evidence,
        "recovery-bundle.json": report.recovery_bundle_evidence,
        "recovery-receipt.json": report.recovery_receipt_evidence,
        "recovery-request.json": report.recovery_request_evidence,
        "source-inventory.json": report.source_inventory_evidence,
        "split-inventory.json": report.split_inventory_evidence,
        "target-inventory.json": report.targets_evidence,
    }
    if any(
        Path(reference.relative_path).parent != artifact_relative
        or Path(reference.relative_path).name != name
        for name, reference in artifact_references.items()
    ):
        raise ValueError("local report artifact references are not canonically confined")

    documents: dict[str, dict[str, Any]] = {}
    for name, reference in (
        ("code", report.code_inventory_evidence),
        ("environment", report.environment_evidence),
        ("source", report.source_inventory_evidence),
        ("split", report.split_inventory_evidence),
        ("targets", report.targets_evidence),
        ("adapter", report.adapter_evidence),
        ("head", report.head_report_evidence),
        ("calibration", report.calibration_evidence),
        ("formal", report.formal_preflight_evidence),
    ):
        evidence_snapshot = _evidence_bytes(root, reference)
        parsed = json.loads(evidence_snapshot)
        if not isinstance(parsed, dict):
            raise ValueError(f"local {name} evidence must be a JSON object")
        if evidence_snapshot != (canonical_json(parsed) + "\n").encode():
            raise ValueError(f"local {name} evidence is not exact canonical JSON")
        documents[name] = parsed
    validated_code = _validate_code_inventory_document(documents["code"])
    observed_environment = validate_local_environment_inventory(
        documents["environment"],
        expected_device_type=report.device,
        rederive=True,
    )
    source_document = _validate_source_inventory(
        documents["source"],
        dataset_id=report.dataset_id,
        source_sha256=report.source_sha256,
    )
    targets = _validate_target_inventory(documents["targets"])
    targets_hash = str(documents["targets"].get("inventoryHash"))
    if (
        validated_code["inventoryHash"] != report.code_inventory_evidence.semantic_hash
        or report.code_hash != report.code_inventory_evidence.semantic_hash
        or observed_environment.inventory_hash != report.environment_evidence.semantic_hash
        or report.environment_hash != report.environment_evidence.semantic_hash
        or report.environment_hash != observed_environment.inventory_hash
        or source_document["inventoryHash"] != report.source_inventory_evidence.semantic_hash
        or targets_hash != report.targets_evidence.semantic_hash
        or targets_hash != report.targets_hash
    ):
        raise ValueError("local inventory semantic identity differs from report")
    formal_document = documents["formal"]
    formal_preflight = FormalPreflightEvidence.model_validate(formal_document, strict=False)
    formal_preflight_path = root / report.formal_preflight_evidence.relative_path
    if (
        formal_preflight.evidence_hash != report.formal_preflight_evidence.semantic_hash
        or formal_preflight.evidence_hash != report.formal_preflight_evidence_hash
        or formal_preflight.formal_ready
        or formal_preflight.promotable
        or load_formal_preflight(formal_preflight_path, runtime_root=root) != formal_preflight
    ):
        raise ValueError("local formal preflight evidence does not exact-reopen")
    source_files = documents["source"].get("files")
    if not isinstance(source_files, (list, tuple)):
        raise ValueError("local source inventory file list is invalid")
    for item in source_files:
        if not isinstance(item, dict) or not isinstance(item.get("relativePath"), str):
            raise ValueError("local source inventory entry is invalid")
        size_bytes = item.get("sizeBytes")
        if not isinstance(size_bytes, int) or size_bytes < 1:
            raise ValueError("local source inventory size is invalid")
        source_snapshot = read_confined_snapshot(root, item["relativePath"], max_bytes=size_bytes)
        if len(source_snapshot) != size_bytes or hashlib.sha256(
            source_snapshot
        ).hexdigest() != item.get("sha256"):
            raise ValueError("local source file bytes differ from immutable inventory")
    split = LocalSplitInventory.model_validate(documents["split"], strict=False)

    base_bundle = load_core_graph_bundle_json(
        _evidence_bytes(root, report.base_bundle_evidence, max_bytes=512 * 1024 * 1024)
    )
    recovery_bundle = load_core_graph_bundle_json(
        _evidence_bytes(root, report.recovery_bundle_evidence, max_bytes=512 * 1024 * 1024)
    )
    if (
        report.base_bundle_evidence.semantic_hash != base_bundle.graph_version_hash
        or report.recovery_bundle_evidence.semantic_hash != recovery_bundle.graph_version_hash
        or base_bundle.graph_version_hash != report.base_graph_version_hash
        or recovery_bundle.graph_version_hash != report.enriched_graph_version_hash
        or base_bundle.source.source_sha256 != report.source_sha256
        or len(recovery_bundle.nodes) != report.node_count
        or len(recovery_bundle.edges) != report.edge_count
        or base_bundle.split_manifest != split.folds[0].split_manifest
    ):
        raise ValueError("local graph bundle identities differ from report")
    if (
        split.inventory_hash != report.split_inventory_evidence.semantic_hash
        or split.inventory_hash != report.split_inventory_hash
        or split.fold_ids != report.fold_ids
        or split.selected_fold_id != report.selected_fold_id
        or split.folds[0].split_manifest_hash != report.selected_split_manifest_hash
        or split.folds[0].role_counts != report.selected_split_role_counts
    ):
        raise ValueError("local split inventory semantic identity differs from report")

    _authority_snapshot_sink.extend(
        _assert_authoritative_local_inputs(
            runtime=root,
            dataset_id=report.dataset_id,
            bundle=base_bundle,
            targets_by_entity=targets,
            split_inventory=split,
            source_inventory=source_document,
        )
    )

    structure_manifest_bytes = _evidence_bytes(root, report.structure_manifest_evidence)
    structure_npz_bytes = _evidence_bytes(
        root, report.structure_npz_evidence, max_bytes=512 * 1024 * 1024
    )
    structure_manifest = StructureCacheManifest.model_validate_json(
        structure_manifest_bytes, strict=False
    )
    manifest_relative = Path(report.structure_manifest_evidence.relative_path)
    npz_relative = Path(report.structure_npz_evidence.relative_path)
    if (
        manifest_relative.name != "manifest.json"
        or npz_relative.name != "structure.npz"
        or manifest_relative.parent != npz_relative.parent
        or hashlib.sha256(structure_npz_bytes).hexdigest() != structure_manifest.npz_sha256
        or structure_manifest.manifest_hash != report.structure_manifest_evidence.semantic_hash
        or structure_manifest.npz_sha256 != report.structure_npz_evidence.semantic_hash
        or structure_manifest.manifest_hash != report.structure_manifest_hash
        or structure_manifest.base_graph_version_hash != base_bundle.graph_version_hash
        or structure_manifest.enriched_graph_version_hash != recovery_bundle.graph_version_hash
    ):
        raise ValueError("local structure artifacts differ from report")
    structure_cache = load_structure_cache(
        base_bundle,
        cache_root=root / manifest_relative.parent.parent,
        role="training",
    )
    if structure_cache.manifest != structure_manifest:
        raise ValueError("local structure cache does not reopen to the bound manifest")
    rederived_recovery_bundle = enrich_bundle_with_structure(base_bundle, structure_cache)
    if rederived_recovery_bundle != recovery_bundle:
        raise ValueError("local recovery bundle is not derived from the bound structure cache")

    expected_data_hash = canonical_sha256(
        {
            "baseGraphVersionHash": base_bundle.graph_version_hash,
            "enrichedGraphVersionHash": recovery_bundle.graph_version_hash,
            "structureManifestHash": structure_manifest.manifest_hash,
            "taskKind": report.task_kind,
            "targetsHash": targets_hash,
            "splitInventoryHash": split.inventory_hash,
            "selectedFoldId": split.selected_fold_id,
            "sourceInventoryHash": source_document["inventoryHash"],
        }
    )
    if expected_data_hash != report.data_hash:
        raise ValueError("local dataHash is not derived from reopened data evidence")

    head = HeadTrainingReport.model_validate(documents["head"], strict=False)
    if (
        head.report_hash != report.head_report_evidence.semantic_hash
        or head.report_hash != report.head_report_hash
    ):
        raise ValueError("local head report semantic identity differs from report")
    calibration_document = documents["calibration"]
    if calibration_document.get("schemaVersion") == "socialgraph-fm.core-calibration-fit/1.0":
        calibration = CalibrationFitReport.model_validate(calibration_document, strict=False)
        calibration_hash = calibration.report_hash
    else:
        calibration_hash = _validate_unavailable_calibration_document(calibration_document)
    if calibration_hash != report.calibration_evidence.semantic_hash or (
        report.calibration_report_hash is not None
        and calibration_hash != report.calibration_report_hash
    ):
        raise ValueError("local calibration semantic identity differs from report")
    adapter_document = documents["adapter"]
    adapter_hash = _validate_adapter_evidence_document(adapter_document)
    adapter_schema = AdapterSchema.model_validate(
        adapter_document.get("adapterSchema"), strict=False
    )
    if (
        adapter_hash != report.adapter_evidence.semantic_hash
        or adapter_schema.adapter_schema_hash != report.adapter_schema_hash
        or adapter_document.get("adapterSchemaHash") != report.adapter_schema_hash
    ):
        raise ValueError("local adapter semantic identity differs from report")
    head_artifact_relative = adapter_document.get("headArtifactRelativePath")
    if (
        not isinstance(head_artifact_relative, str)
        or head_artifact_relative != report.head_artifact_relative_path
    ):
        raise ValueError("local adapter evidence omits head artifact path")
    head_artifact = _evidence_bytes(root, report.head_artifact_evidence, max_bytes=64 * 1024 * 1024)
    head_artifact_sha256 = hashlib.sha256(head_artifact).hexdigest()
    if (
        head_artifact_sha256 != adapter_document.get("headArtifactSha256")
        or head_artifact_sha256 != report.head_artifact_sha256
    ):
        raise ValueError("local adapter tensor artifact bytes differ")
    tensor_payload = torch.load(BytesIO(head_artifact), map_location="cpu", weights_only=True)
    if not isinstance(tensor_payload, dict) or set(tensor_payload) != {
        "schemaVersion",
        "model",
        "adapter",
        "headReportHash",
    }:
        raise ValueError("local head tensor artifact inventory is invalid")

    checkpoint_snapshot = _evidence_bytes(
        root, report.checkpoint_evidence, max_bytes=512 * 1024 * 1024
    )
    if hashlib.sha256(checkpoint_snapshot).hexdigest() != report.checkpoint_sha256:
        raise ValueError("local checkpoint bytes differ from report SHA-256")
    bindings = CheckpointBindings(
        config_hash=report.config_hash,
        data_hash=report.data_hash,
        code_hash=report.code_hash,
        environment_hash=report.environment_hash,
    )
    checkpoint = load_checkpoint(checkpoint_snapshot, expected_bindings=bindings)
    if (
        checkpoint.get("status") != report.checkpoint_status
        or checkpoint.get("promotable") is not report.checkpoint_promotable
    ):
        raise ValueError("local checkpoint state differs from report")
    checkpoint_state = checkpoint["trainer"]
    expected_config = (
        TrainingConfig.smoke(max_steps=report.optimizer_steps)
        if report.phase == "smoke"
        else TrainingConfig.dev(max_steps=report.optimizer_steps)
    )
    expected_config_hash = canonical_sha256(
        {"phase": report.phase, "seed": report.seed, "training": expected_config.to_dict()}
    )
    expected_config_document = json.loads(canonical_json(expected_config.to_dict()))
    if (
        checkpoint_state.get("config") != expected_config.to_dict()
        or checkpoint_state.get("trainingSeed") != report.seed
        or checkpoint_state.get("optimizerStep") != report.optimizer_steps
        or expected_config_hash != report.config_hash
    ):
        raise ValueError("local optimizer/config claims differ from checkpoint state")

    recovery_snapshot = _evidence_bytes(root, report.recovery_receipt_evidence)
    if hashlib.sha256(recovery_snapshot).hexdigest() != report.recovery_receipt_sha256:
        raise ValueError("local CPU recovery receipt bytes differ")
    recovery = LocalRecoveryReceipt.model_validate_json(recovery_snapshot)
    if recovery_snapshot != (canonical_json(recovery) + "\n").encode():
        raise ValueError("local CPU recovery receipt is not exact canonical JSON")
    interpreter, _recovery_environment = _recovery_interpreter()
    expected_recovery_environment = local_environment_inventory("cpu")
    if (
        recovery.receipt_hash != report.recovery_receipt_hash
        or recovery.recovery_process_id != report.recovery_process_id
        or recovery.recovery_parent_process_id != report.recovery_parent_process_id
        or recovery.recovery_device != report.recovery_device
        or recovery.recovery_interpreter_path != str(interpreter)
        or recovery.recovery_interpreter_sha256 != _hash_file(interpreter)
        or recovery.checkpoint_sha256 != report.checkpoint_sha256
        or recovery.config_hash != report.config_hash
        or recovery.data_hash != report.data_hash
        or recovery.code_hash != report.code_hash
        or recovery.environment_hash != report.environment_hash
        or recovery.recovery_environment_inventory != expected_recovery_environment
        or recovery.recovery_environment_hash != expected_recovery_environment.inventory_hash
        or recovery.composite_state_hash != report.fresh_composite_state_hash
        or recovery.recovery_state_hash != report.fresh_recovery_state_hash
    ):
        raise ValueError("local CPU recovery receipt semantic identity differs")

    trainer_state_hash = _trainer_state_hash(checkpoint_state)
    composite_state_hash = _composite_hash(checkpoint_state)
    if (
        trainer_state_hash != recovery.trainer_state_hash
        or trainer_state_hash != report.checkpoint_evidence.semantic_hash
        or composite_state_hash != recovery.composite_state_hash
    ):
        raise ValueError("local recovery receipt state hashes are not checkpoint-derived")
    checkpoint_adapters = checkpoint_state.get("adapters")
    checkpoint_schemas = checkpoint_state.get("adapterSchemas")
    if (
        not isinstance(checkpoint_state.get("model"), dict)
        or not isinstance(checkpoint_adapters, dict)
        or len(checkpoint_adapters) != 1
        or not isinstance(checkpoint_schemas, dict)
        or len(checkpoint_schemas) != 1
        or _state_hash(checkpoint_state["model"]) != recovery.model_state_hash
        or _state_hash(next(iter(checkpoint_adapters.values()))) != recovery.adapter_state_hash
        or AdapterSchema.model_validate(
            next(iter(checkpoint_schemas.values())), strict=False
        ).adapter_schema_hash
        != recovery.adapter_schema_hash
        or recovery.adapter_state_hash != adapter_document.get("adapterStateHash")
    ):
        raise ValueError("local checkpoint model/adapter identity differs from recovery receipt")
    request_snapshot = _evidence_bytes(root, report.recovery_request_evidence)
    request_document = json.loads(request_snapshot)
    if not isinstance(request_document, dict):
        raise ValueError("local CPU recovery request is invalid")
    if request_snapshot != (canonical_json(request_document) + "\n").encode():
        raise ValueError("local CPU recovery request is not exact canonical JSON")
    request_without_hash = {
        key: value for key, value in request_document.items() if key != "requestHash"
    }
    request_hash = request_document.get("requestHash")
    expected_request_keys = {
        "schemaVersion",
        "checkpointName",
        "bundleName",
        "evaluationArtifactName",
        "receiptName",
        "checkpointSha256",
        "bindings",
        "codeInventory",
        "trainingEnvironmentInventory",
        "interpreterPath",
        "interpreterSha256",
        "trainingConfig",
        "trainingSeed",
        "adapterDomain",
        "graphVersionHash",
        "parentProcessId",
        "requestHash",
    }
    domains = checkpoint_state.get("adapters")
    domain = next(iter(domains)) if isinstance(domains, dict) and len(domains) == 1 else None
    if (
        set(request_document) != expected_request_keys
        or request_document.get("schemaVersion")
        != "socialgraph-fm.core-local-recovery-request/4.0"
        or request_hash != canonical_sha256(request_without_hash)
        or request_hash != recovery.request_hash
        or request_hash != report.recovery_request_evidence.semantic_hash
        or request_document.get("checkpointName") != Path(report.checkpoint_relative_path).name
        or request_document.get("bundleName")
        != Path(report.recovery_bundle_evidence.relative_path).name
        or request_document.get("evaluationArtifactName")
        != Path(report.recovery_evaluation_evidence.relative_path).name
        or request_document.get("receiptName")
        != Path(report.recovery_receipt_evidence.relative_path).name
        or request_document.get("checkpointSha256") != report.checkpoint_sha256
        or request_document.get("bindings") != asdict(bindings)
        or request_document.get("trainingEnvironmentInventory")
        != observed_environment.model_dump(mode="python", by_alias=True)
        or request_document.get("interpreterPath") != str(interpreter)
        or request_document.get("interpreterSha256") != _hash_file(interpreter)
        or request_document.get("trainingConfig") != expected_config_document
        or request_document.get("trainingSeed") != report.seed
        or request_document.get("adapterDomain") != domain
        or request_document.get("graphVersionHash") != recovery_bundle.graph_version_hash
        or request_document.get("parentProcessId") != recovery.recovery_parent_process_id
        or recovery.recovery_process_id == recovery.recovery_parent_process_id
        or not isinstance(request_document.get("codeInventory"), dict)
        or _validate_code_inventory_document(request_document["codeInventory"])["inventoryHash"]
        != report.code_hash
    ):
        raise ValueError("local CPU recovery request identity differs from receipt")
    evaluation_snapshot = _evidence_bytes(
        root, report.recovery_evaluation_evidence, max_bytes=512 * 1024 * 1024
    )
    if hashlib.sha256(evaluation_snapshot).hexdigest() != recovery.evaluation_artifact_sha256:
        raise ValueError("local CPU evaluation artifact differs from recovery receipt")
    evaluation = torch.load(BytesIO(evaluation_snapshot), map_location="cpu", weights_only=True)
    if (
        not isinstance(evaluation, dict)
        or set(evaluation) != {"schemaVersion", "requestHash", "model", "adapterSchema", "adapter"}
        or evaluation.get("schemaVersion")
        != "socialgraph-fm.core-local-cpu-evaluation-state/1.0"
        or evaluation.get("requestHash") != recovery.request_hash
        or _state_hash(evaluation["model"]) != recovery.model_state_hash
        or _state_hash(evaluation["adapter"]) != recovery.adapter_state_hash
        or AdapterSchema.model_validate(
            evaluation.get("adapterSchema"), strict=False
        ).adapter_schema_hash
        != recovery.adapter_schema_hash
    ):
        raise ValueError("local CPU evaluation state differs from recovery receipt")

    expected_recovery_state_hash = canonical_sha256(
        {
            "trainerStateHash": trainer_state_hash,
            "requestHash": request_hash,
            "recoveryDevice": "cpu",
            "modelStateHash": recovery.model_state_hash,
            "adapterStateHash": recovery.adapter_state_hash,
        }
    )
    if (
        expected_recovery_state_hash != recovery.recovery_state_hash
        or expected_recovery_state_hash != report.recovery_evaluation_evidence.semantic_hash
    ):
        raise ValueError("local recovery state hash is not derived from exact artifacts")

    evaluation_model = CoreGFM(node_classes=2)
    evaluation_model.load_state_dict(evaluation["model"], strict=True)
    evaluation_adapter = BundleInputAdapter(
        recovery_bundle,
        mode="training",
        schema=AdapterSchema.model_validate(evaluation["adapterSchema"], strict=False),
    )
    evaluation_adapter.load_state_dict(evaluation["adapter"], strict=True)
    head_model = CoreGFM(node_classes=2)
    head_model.load_state_dict(tensor_payload["model"], strict=True)
    head_adapter = BundleInputAdapter(
        recovery_bundle,
        mode="training",
        schema=evaluation_adapter.schema,
    )
    head_adapter.load_state_dict(tensor_payload["adapter"], strict=True)
    selected_prefix = f"{head.head_name}."
    evaluation_model_state = evaluation_model.state_dict()
    head_model_state = head_model.state_dict()
    if any(
        canonical_tensor_digest(head_model_state[name])
        != canonical_tensor_digest(evaluation_model_state[name])
        for name in head_model_state
        if not name.startswith(selected_prefix)
    ):
        raise ValueError("local head artifact changed encoder or non-selected head state")
    head_artifact_semantic_hash = canonical_sha256(
        {
            "schemaVersion": _HEAD_ARTIFACT_SCHEMA,
            "modelStateHash": _state_hash(head_model_state),
            "adapterStateHash": _state_hash(head_adapter.state_dict()),
            "headReportHash": head.report_hash,
        }
    )
    if (
        tensor_payload.get("schemaVersion") != _HEAD_ARTIFACT_SCHEMA
        or tensor_payload.get("headReportHash") != head.report_hash
        or _state_hash(head_adapter.state_dict()) != recovery.adapter_state_hash
        or _state_hash(head_adapter.state_dict()) != adapter_document.get("adapterStateHash")
        or head_artifact_semantic_hash != report.head_artifact_evidence.semantic_hash
        or head_artifact_semantic_hash != adapter_document.get("headArtifactSemanticHash")
        or head.calculate_current_head_hash(head_model) != report.head_state_hash
    ):
        raise ValueError("local full head artifact identity differs from evidence")

    encoded = encode_supervised_graph(head_model, recovery_bundle, head_adapter)
    train = _partition(
        recovery_bundle,
        task_kind=report.task_kind,
        role="train",
        targets_by_entity=targets,
    )
    validation = _partition(
        recovery_bundle,
        task_kind=report.task_kind,
        role="validation",
        targets_by_entity=targets,
    )
    supervised = SupervisedTrainValidation.create(
        task_kind=report.task_kind,
        provenance=encoded.provenance,
        train=train,
        validation=validation,
    )
    if (
        supervised.data_hash != report.supervised_data_hash
        or encoded.provenance.artifact_hash != report.encoded_artifact_hash
        or head.task_kind != report.task_kind
        or head.graph_version_hash != recovery_bundle.graph_version_hash
        or head.model_identity_hash != derive_encoder_identity(head_model)
        or head.encoding_artifact_hash != encoded.provenance.artifact_hash
        or head.adapter_schema_hash != head_adapter.schema.adapter_schema_hash
        or head.adapter_state_hash != _state_hash(head_adapter.state_dict())
        or head.data_hash != supervised.data_hash
        or head.train_partition_hash != train.partition_hash
        or head.validation_partition_hash != validation.partition_hash
        or head.training_config.max_steps != report.head_steps
        or head.num_nodes != len(recovery_bundle.nodes)
        or head.head_state_hash != report.head_state_hash
    ):
        raise ValueError("local supervised/head claims are not rederived from exact evidence")
    verified_head = _new_verified_head_training_report(head)
    verify_head_training_report(head_model, encoded, supervised, verified_head)

    if calibration_document.get("schemaVersion") == "socialgraph-fm.core-calibration-fit/1.0":
        observed_calibration = CalibrationFitReport.model_validate(
            calibration_document, strict=False
        )
        scores = derive_validation_scores(
            head_model,
            encoded,
            supervised,
            verified_head,
            semantics=BinaryScoreSemantics.for_task(report.task_kind),
        )
        expected_calibration = fit_score_calibration_report(
            scores,
            protocol=CalibrationProtocol.fixed(scores),
        )
        if observed_calibration != expected_calibration:
            raise ValueError("local calibration is not rederived from head/validation scores")
    elif calibration_document != {
        "schemaVersion": "socialgraph-fm.core-local-calibration-evidence/1.0",
        "status": "unavailable-single-class-validation",
        "evidenceHash": calibration_hash,
    }:
        raise ValueError("local unavailable calibration evidence is not exact")
    return report


def reopen_local_experiment_evidence(
    report_path: Path, *, runtime_root: Path
) -> LocalExperimentRun:
    authority_snapshots: list[_HeldLocalSourceSnapshot] = []
    try:
        return _reopen_local_experiment_evidence_guarded(
            report_path,
            runtime_root=runtime_root,
            _authority_snapshot_sink=authority_snapshots,
        )
    finally:
        _close_held_authoritative_sources(tuple(authority_snapshots))


__all__ = [
    "LocalDatasetInputs",
    "LocalExperimentRun",
    "LocalSplitInventory",
    "load_email_local_inputs",
    "load_penn94_local_inputs",
    "run_local_nonpromotable_experiment",
    "reopen_local_experiment_evidence",
]
