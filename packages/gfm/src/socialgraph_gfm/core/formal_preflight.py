"""Fail-closed, hash-bound inventory checks for the core formal corpus.

The preflight is deliberately diagnostic only: it never downloads or repairs a
dataset and it cannot promote a model.  A graph becomes ``ready`` only through
the versioned experiment-dataset manifest published by this module.
"""

from __future__ import annotations

import hashlib
import gzip
import io
import importlib
import ctypes
import errno
import os
import stat
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from socialgraph_gfm.canonical import canonical_json, canonical_sha256

from .bundle import SplitManifest, CoreGraphBundle, load_core_graph_bundle_json
from .datasets.recipes import SourceRecipe, load_dataset_recipes
from .formal_materialization import (
    derive_registered_formal_dataset,
    formal_materializer_binding,
)
from .safe_paths import read_confined_snapshot, reject_link_components, secure_existing_root


_HASH = r"^[0-9a-f]{64}$"
_MANIFEST_SCHEMA = "socialgraph-fm.core-experiment-dataset/1.2"
_LABEL_SCHEMA = "socialgraph-fm.core-experiment-labels/1.0"
_SPLIT_INVENTORY_SCHEMA = "socialgraph-fm.core-experiment-splits/1.0"
_PREFLIGHT_SCHEMA = "socialgraph-fm.core-formal-preflight/1.0"
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_LABEL_BYTES = 64 * 1024 * 1024
_MAX_SPLIT_INVENTORY_BYTES = 128 * 1024 * 1024
_MAX_BUNDLE_BYTES = 512 * 1024 * 1024
_STREAM_CHUNK_BYTES = 1024 * 1024
_GZIP_EXPANDED_LIMITS = {
    ("email-eu-core", "edges"): 64 * 1024 * 1024,
    ("email-eu-core", "departments"): 16 * 1024 * 1024,
    ("wiki-rfa", "wiki-rfa"): 512 * 1024 * 1024,
}


if os.name == "nt":  # pragma: win32 cover
    from ctypes import wintypes

    class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    class _FILE_DISPOSITION_INFO(ctypes.Structure):
        _fields_ = [("DeleteFile", wintypes.BOOL)]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CreateFileW = _kernel32.CreateFileW
    _CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _CreateFileW.restype = wintypes.HANDLE
    _GetFileInformationByHandle = _kernel32.GetFileInformationByHandle
    _GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION),
    ]
    _GetFileInformationByHandle.restype = wintypes.BOOL
    _GetFinalPathNameByHandleW = _kernel32.GetFinalPathNameByHandleW
    _GetFinalPathNameByHandleW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    _GetFinalPathNameByHandleW.restype = wintypes.DWORD
    _FlushFileBuffers = _kernel32.FlushFileBuffers
    _FlushFileBuffers.argtypes = [wintypes.HANDLE]
    _FlushFileBuffers.restype = wintypes.BOOL
    _SetFileInformationByHandle = _kernel32.SetFileInformationByHandle
    _SetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    _SetFileInformationByHandle.restype = wintypes.BOOL
    _SetFilePointerEx = _kernel32.SetFilePointerEx
    _SetFilePointerEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.DWORD,
    ]
    _SetFilePointerEx.restype = wintypes.BOOL
    _ReadFile = _kernel32.ReadFile
    _ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    _ReadFile.restype = wintypes.BOOL
    _CloseHandle = _kernel32.CloseHandle
    _CloseHandle.argtypes = [wintypes.HANDLE]
    _CloseHandle.restype = wintypes.BOOL

    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _DELETE = 0x00010000
    _FILE_READ_ATTRIBUTES = 0x0080
    _FILE_SHARE_READ = 0x0001
    _FILE_SHARE_WRITE = 0x0002
    _FILE_SHARE_DELETE = 0x0004
    _OPEN_EXISTING = 3
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_ATTRIBUTE_DIRECTORY = 0x0010
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
    _FILE_DISPOSITION_INFO_CLASS = 4
    _FILE_BEGIN = 0
    # Win32 reports these two explicit capability failures for filesystems that
    # do not implement a directory-buffer flush. Access/sharing failures remain fatal.
    _UNSUPPORTED_DIRECTORY_FLUSH_ERRORS = frozenset({1, 50})


def _PUBLICATION_SEAM(_kind: str, _target: Path) -> None:
    """Test seam immediately before a no-clobber publication commit."""

    return


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        protected_namespaces=("model_dump",),
        strict=True,
    )


def _safe_relative_path(value: str) -> str:
    if not value or "\\" in value or ":" in value:
        raise ValueError("path must be a safe relative POSIX path")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("path must be a safe relative POSIX path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed.as_posix() != value:
        raise ValueError("path must be a safe relative POSIX path")
    return value


class FormalCorpusRequirement(_StrictModel):
    requirement_id: str = Field(alias="requirementId", min_length=1)
    recipe_id: str = Field(alias="recipeId", min_length=1)
    graph_id: str = Field(alias="graphId", min_length=1)
    corpus_role: Literal[
        "near-domain-source",
        "offline-target",
        "cross-domain-source",
        "governance-target",
        "relation-target",
    ] = Field(alias="corpusRole")
    required_usage_scope: Literal["public-serving-eligible", "local-research-demo-only"] = Field(
        alias="requiredUsageScope"
    )
    expected_split_policy: str = Field(alias="expectedSplitPolicy", min_length=1)
    experiment_split_policy: str = Field(alias="experimentSplitPolicy", min_length=1)
    official_split_count: int | None = Field(alias="officialSplitCount", ge=1)
    manifest_relative_path: str = Field(alias="manifestRelativePath")
    raw_source_ids: tuple[str, ...] = Field(alias="rawSourceIds", strict=False)
    raw_relative_paths: tuple[str, ...] = Field(alias="rawRelativePaths", strict=False)
    audit_relative_paths: tuple[str, ...] = Field(alias="auditRelativePaths", strict=False)

    @field_validator("manifest_relative_path", "raw_relative_paths", "audit_relative_paths")
    @classmethod
    def validate_paths(cls, value: str | tuple[str, ...]) -> str | tuple[str, ...]:
        if isinstance(value, str):
            return _safe_relative_path(value)
        return tuple(_safe_relative_path(item) for item in value)

    @model_validator(mode="after")
    def validate_path_inventory(self):
        paths = (self.manifest_relative_path, *self.raw_relative_paths)
        if len(paths) != len(set(paths)):
            raise ValueError("requirement paths must not contain duplicates")
        if (
            len(self.raw_source_ids) != len(self.raw_relative_paths)
            or len(self.raw_source_ids) != len(set(self.raw_source_ids))
            or any(not source_id for source_id in self.raw_source_ids)
        ):
            raise ValueError("raw source IDs must uniquely align with raw paths")
        if (self.expected_split_policy == "official") != (self.official_split_count is not None):
            raise ValueError("official split count must be explicit only for official requirements")
        return self


ScalarLabel = int | float | str


class LabelValue(_StrictModel):
    entity_id: str = Field(alias="entityId", min_length=1)
    value: ScalarLabel

    @model_validator(mode="after")
    def validate_value(self):
        if isinstance(self.value, bool):
            raise ValueError("Boolean labels are not supported")
        return self


class LabelTarget(_StrictModel):
    name: str = Field(min_length=1)
    values: tuple[LabelValue, ...] = Field(strict=False)

    @model_validator(mode="after")
    def validate_values(self):
        identifiers = tuple(item.entity_id for item in self.values)
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(set(identifiers)):
            raise ValueError("label entity records must be unique and sorted")
        return self


class ExperimentLabels(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-experiment-labels/1.0"] = Field(
        alias="schemaVersion"
    )
    requirement_id: str = Field(alias="requirementId", min_length=1)
    targets: tuple[LabelTarget, ...] = Field(strict=False)
    labels_hash: str = Field(alias="labelsHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_labels(self):
        names = tuple(target.name for target in self.targets)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("label targets must be unique and sorted")
        if self.labels_hash != canonical_sha256(self.targets):
            raise ValueError("labelsHash does not match canonical targets")
        return self


class ExperimentSplitFold(_StrictModel):
    fold_id: str = Field(alias="foldId", min_length=1)
    split_manifest: SplitManifest = Field(alias="splitManifest")
    split_manifest_hash: str = Field(alias="splitManifestHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_split_hash(self):
        expected = canonical_sha256(self.split_manifest.model_dump(mode="python", by_alias=True))
        if self.split_manifest_hash != expected:
            raise ValueError("splitManifestHash does not match the fold manifest")
        return self


class ExperimentSplitInventory(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-experiment-splits/1.0"] = Field(
        alias="schemaVersion"
    )
    requirement_id: str = Field(alias="requirementId", min_length=1)
    folds: tuple[ExperimentSplitFold, ...] = Field(strict=False, min_length=1)
    inventory_hash: str = Field(alias="inventoryHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_inventory(self):
        fold_ids = tuple(fold.fold_id for fold in self.folds)
        if fold_ids != tuple(sorted(fold_ids)) or len(fold_ids) != len(set(fold_ids)):
            raise ValueError("split fold IDs must be unique and sorted")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"inventory_hash"})
        )
        if self.inventory_hash != expected:
            raise ValueError("inventoryHash does not match the split inventory")
        return self


class ExperimentDatasetManifest(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-experiment-dataset/1.2"] = Field(
        alias="schemaVersion"
    )
    requirement_id: str = Field(alias="requirementId", min_length=1)
    recipe_id: str = Field(alias="recipeId", min_length=1)
    recipe_version: str = Field(alias="recipeVersion", min_length=1)
    recipe_sha256: str = Field(alias="recipeSha256", pattern=_HASH)
    graph_id: str = Field(alias="graphId", min_length=1)
    phase_eligibility: Literal["smoke", "dev", "formal"] = Field(alias="phaseEligibility")
    usage_scope: Literal["public-serving-eligible", "local-research-demo-only"] = Field(
        alias="usageScope"
    )
    split_policy: str = Field(alias="splitPolicy", min_length=1)
    experiment_split_policy: str = Field(alias="experimentSplitPolicy", min_length=1)
    materializer_id: str = Field(alias="materializerId", min_length=1)
    materializer_version: str = Field(alias="materializerVersion", min_length=1)
    materializer_code_sha256: str = Field(alias="materializerCodeSha256", pattern=_HASH)
    materialization_protocol_hash: str = Field(alias="materializationProtocolHash", pattern=_HASH)
    manifest_relative_path: str = Field(alias="manifestRelativePath")
    bundle_relative_path: str = Field(alias="bundleRelativePath")
    bundle_sha256: str = Field(alias="bundleSha256", pattern=_HASH)
    labels_relative_path: str = Field(alias="labelsRelativePath")
    labels_sha256: str = Field(alias="labelsSha256", pattern=_HASH)
    label_names: tuple[str, ...] = Field(alias="labelNames", strict=False)
    split_inventory_relative_path: str = Field(alias="splitInventoryRelativePath")
    split_inventory_sha256: str = Field(alias="splitInventorySha256", pattern=_HASH)
    split_count: int = Field(alias="splitCount", ge=1)
    split_ids: tuple[str, ...] = Field(alias="splitIds", strict=False)
    split_manifest_hashes: tuple[str, ...] = Field(alias="splitManifestHashes", strict=False)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH)
    source_sha256: str = Field(alias="sourceSha256", pattern=_HASH)
    split_manifest_hash: str = Field(alias="splitManifestHash", pattern=_HASH)
    manifest_hash: str = Field(alias="manifestHash", pattern=_HASH)

    @field_validator(
        "manifest_relative_path",
        "bundle_relative_path",
        "labels_relative_path",
        "split_inventory_relative_path",
    )
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _safe_relative_path(value)

    @model_validator(mode="after")
    def validate_manifest(self):
        expected_prefix = f"experiment-corpus/{self.requirement_id}/"
        paths = (
            self.manifest_relative_path,
            self.bundle_relative_path,
            self.labels_relative_path,
            self.split_inventory_relative_path,
        )
        if any(not path.startswith(expected_prefix) for path in paths):
            raise ValueError("experiment artifacts must remain in the requirement directory")
        if len(paths) != len(set(paths)):
            raise ValueError("experiment artifact paths must be distinct")
        if tuple(sorted(set(self.label_names))) != self.label_names:
            raise ValueError("labelNames must be unique and sorted")
        if (
            self.split_ids != tuple(sorted(self.split_ids))
            or len(self.split_ids) != self.split_count
            or len(set(self.split_ids)) != self.split_count
            or len(self.split_manifest_hashes) != self.split_count
        ):
            raise ValueError("split metadata must bind every unique sorted fold")
        if self.split_manifest_hash != self.split_manifest_hashes[0]:
            raise ValueError("the bundle split must be the first split inventory fold")
        expected_hash = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"manifest_hash"})
        )
        if self.manifest_hash != expected_hash:
            raise ValueError("manifestHash does not match canonical manifest")
        return self


class FileEvidence(_StrictModel):
    relative_path: str = Field(alias="relativePath")
    sha256: str = Field(pattern=_HASH)
    size_bytes: int = Field(alias="sizeBytes", gt=0)
    purpose: Literal["manifest", "bundle", "labels", "split-inventory", "raw", "audit"]

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _safe_relative_path(value)


class FormalCorpusObservation(_StrictModel):
    requirement_id: str = Field(alias="requirementId", min_length=1)
    status: Literal["ready", "missing", "raw-only", "audit-only", "invalid", "usage-ineligible"]
    reason_code: str = Field(alias="reasonCode", min_length=1)
    manifest_hash: str | None = Field(default=None, alias="manifestHash", pattern=_HASH)
    graph_version_hash: str | None = Field(default=None, alias="graphVersionHash", pattern=_HASH)
    split_manifest_hash: str | None = Field(default=None, alias="splitManifestHash", pattern=_HASH)
    files: tuple[FileEvidence, ...] = Field(strict=False)

    @model_validator(mode="after")
    def validate_status_evidence(self):
        paths = tuple(item.relative_path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("observation file evidence must be unique and sorted")
        purposes = tuple(item.purpose for item in self.files)
        hashes = (
            self.manifest_hash,
            self.graph_version_hash,
            self.split_manifest_hash,
        )
        if self.status in {"ready", "usage-ineligible"}:
            if any(value is None for value in hashes):
                raise ValueError("validated observation requires all semantic hashes")
            if not {"manifest", "bundle", "labels", "split-inventory"} <= set(purposes):
                raise ValueError(
                    "validated observation requires manifest, bundle, labels, and splits"
                )
        elif self.status == "missing":
            if self.files or any(value is not None for value in hashes):
                raise ValueError("missing observation cannot claim files or semantic hashes")
        elif self.status == "raw-only":
            if (
                not self.files
                or set(purposes) != {"raw"}
                or any(value is not None for value in hashes)
            ):
                raise ValueError("raw-only observation must contain only raw evidence")
        elif self.status == "audit-only":
            if (
                not self.files
                or set(purposes) != {"audit"}
                or any(value is not None for value in hashes)
            ):
                raise ValueError("audit-only observation must contain only audit evidence")
        if self.status == "ready":
            if self.reason_code != "validated-formal-dataset" or "raw" not in purposes:
                raise ValueError("ready observation requires validated raw source evidence")
        return self


class FormalPreflightEvidence(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-formal-preflight/1.0"] = Field(
        alias="schemaVersion"
    )
    requirements_hash: str = Field(alias="requirementsHash", pattern=_HASH)
    recipe_catalog_hash: str = Field(alias="recipeCatalogHash", pattern=_HASH)
    observations: tuple[FormalCorpusObservation, ...] = Field(strict=False)
    formal_ready: bool = Field(alias="formalReady")
    promotable: bool
    evidence_hash: str = Field(alias="evidenceHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_evidence(self):
        if self.requirements_hash != _requirements_hash():
            raise ValueError("requirementsHash does not match fixed formal inventory")
        if self.recipe_catalog_hash != _recipe_catalog_hash():
            raise ValueError("recipeCatalogHash does not match packaged recipes")
        expected_ids = tuple(item.requirement_id for item in FORMAL_CORPUS_REQUIREMENTS)
        observed_ids = tuple(item.requirement_id for item in self.observations)
        if observed_ids != expected_ids:
            raise ValueError("formal observations do not match fixed requirement order")
        for requirement, observation in zip(
            FORMAL_CORPUS_REQUIREMENTS, self.observations, strict=True
        ):
            if observation.status == "ready":
                by_purpose = {
                    purpose: {
                        item.relative_path for item in observation.files if item.purpose == purpose
                    }
                    for purpose in {item.purpose for item in observation.files}
                }
                if by_purpose.get("manifest") != {requirement.manifest_relative_path}:
                    raise ValueError("ready observation manifest path is not authoritative")
                if by_purpose.get("raw") != set(requirement.raw_relative_paths):
                    raise ValueError("ready observation raw inventory is not authoritative")
        ready = bool(self.observations) and all(
            item.status == "ready" for item in self.observations
        )
        if self.formal_ready != ready or self.promotable != ready:
            raise ValueError("formalReady/promotable must be derived from all observations")
        expected_hash = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"evidence_hash"})
        )
        if self.evidence_hash != expected_hash:
            raise ValueError("evidenceHash does not match canonical preflight evidence")
        return self


def _requirement(
    requirement_id: str,
    recipe_id: str,
    graph_id: str,
    corpus_role: Literal[
        "near-domain-source",
        "offline-target",
        "cross-domain-source",
        "governance-target",
        "relation-target",
    ],
    raw_source_ids: tuple[str, ...],
    raw_relative_paths: tuple[str, ...],
    *,
    audit_relative_paths: tuple[str, ...] = (),
    bundle_split_policy: str | None = None,
    official_split_count: int | None = None,
) -> FormalCorpusRequirement:
    recipe = load_dataset_recipes()[recipe_id]
    if not set(raw_source_ids) <= {source.source_id for source in recipe.sources}:
        raise ValueError("formal requirement references an unknown recipe source")
    return FormalCorpusRequirement(
        requirementId=requirement_id,
        recipeId=recipe_id,
        graphId=graph_id,
        corpusRole=corpus_role,
        requiredUsageScope=recipe.graph_usage_scopes[graph_id],
        expectedSplitPolicy=bundle_split_policy or recipe.split_policy,
        experimentSplitPolicy=recipe.split_policy,
        officialSplitCount=official_split_count,
        manifestRelativePath=f"experiment-corpus/{requirement_id}/dataset-manifest.json",
        rawSourceIds=raw_source_ids,
        rawRelativePaths=raw_relative_paths,
        auditRelativePaths=audit_relative_paths,
    )


_FB_RAW = "raw/facebook100/1.0.0"
_TWITCH_RAW = ("raw/twitch-language/1.0.0/twitch.zip",)

FORMAL_CORPUS_REQUIREMENTS: tuple[FormalCorpusRequirement, ...] = (
    _requirement(
        "facebook100.reed98",
        "facebook100",
        "Reed98",
        "near-domain-source",
        ("Reed98",),
        (f"{_FB_RAW}/Reed98.mat",),
        bundle_split_policy="all-visible-training",
    ),
    _requirement(
        "facebook100.amherst41",
        "facebook100",
        "Amherst41",
        "near-domain-source",
        ("Amherst41",),
        (f"{_FB_RAW}/Amherst41.mat",),
        bundle_split_policy="all-visible-training",
    ),
    _requirement(
        "facebook100.johns-hopkins55",
        "facebook100",
        "Johns Hopkins55",
        "near-domain-source",
        ("Johns Hopkins55",),
        (f"{_FB_RAW}/Johns Hopkins55.mat",),
        bundle_split_policy="all-visible-training",
    ),
    _requirement(
        "facebook100.cornell5",
        "facebook100",
        "Cornell5",
        "near-domain-source",
        ("Cornell5",),
        (f"{_FB_RAW}/Cornell5.mat",),
        bundle_split_policy="all-visible-training",
    ),
    _requirement(
        "facebook100.penn94",
        "facebook100",
        "Penn94",
        "offline-target",
        ("Penn94", "Penn94-official-splits"),
        (f"{_FB_RAW}/Penn94.mat", f"{_FB_RAW}/fb100-Penn94-splits.npy"),
        official_split_count=5,
    ),
    *(
        _requirement(
            f"twitch.{domain.lower()}",
            "twitch-language",
            domain,
            "cross-domain-source",
            ("twitch",),
            _TWITCH_RAW,
            audit_relative_paths=(".inventory-audit/twitch.zip",),
            bundle_split_policy="all-visible-training",
        )
        for domain in ("DE", "EN", "ES", "FR", "PT", "RU")
    ),
    _requirement(
        "tolokers",
        "tolokers",
        "tolokers",
        "governance-target",
        ("tolokers",),
        ("raw/tolokers/1.0.0/tolokers.npz",),
        official_split_count=10,
    ),
    _requirement(
        "wiki-rfa",
        "wiki-rfa",
        "wiki-rfa",
        "governance-target",
        ("wiki-rfa",),
        ("raw/wiki-rfa/1.0.0/wiki-RfA.txt.gz",),
    ),
    _requirement(
        "github-musae",
        "github-musae",
        "github-musae",
        "relation-target",
        ("github",),
        ("raw/github-musae/1.0.0/git_web_ml.zip",),
        audit_relative_paths=(".inventory-audit/git_web_ml.zip",),
    ),
    _requirement(
        "email-eu-core",
        "email-eu-core",
        "email-eu-core",
        "relation-target",
        ("edges", "departments"),
        (
            "raw/email-eu-core/1.0.0/email-Eu-core.txt.gz",
            "raw/email-eu-core/1.0.0/email-Eu-core-department-labels.txt.gz",
        ),
    ),
)

if len({item.requirement_id for item in FORMAL_CORPUS_REQUIREMENTS}) != len(
    FORMAL_CORPUS_REQUIREMENTS
):  # pragma: no cover - module invariant
    raise RuntimeError("formal corpus requirement IDs must be unique")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: BaseModel | dict[str, Any]) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _source_pairs(
    requirement: FormalCorpusRequirement,
) -> tuple[tuple[SourceRecipe, str], ...]:
    recipe = load_dataset_recipes()[requirement.recipe_id]
    source_by_id = {source.source_id: source for source in recipe.sources}
    return tuple(
        (source_by_id[source_id], relative_path)
        for source_id, relative_path in zip(
            requirement.raw_source_ids, requirement.raw_relative_paths, strict=True
        )
    )


def _validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("raw archive member path is unsafe")


def _validate_raw_format(
    data: bytes, source: SourceRecipe, *, expanded_max_bytes: int | None = None
) -> None:
    if source.archive_type == "zip":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = [item for item in archive.infolist() if not item.is_dir()]
                names = [item.filename for item in members]
                for name in names:
                    _validate_member_name(name)
                if (
                    len(names) != len(set(names))
                    or set(names) != set(source.inventory)
                    or any(
                        item.flag_bits & 0x1
                        or stat.S_ISLNK(item.external_attr >> 16)
                        or item.file_size > _MAX_BUNDLE_BYTES
                        for item in members
                    )
                ):
                    raise ValueError("raw ZIP inventory or member metadata is invalid")
        except (OSError, zipfile.BadZipFile) as error:
            raise ValueError("raw ZIP source is invalid") from error
    elif source.archive_type == "gzip":
        if len(source.inventory) != 1:
            raise ValueError("gzip recipe must declare one output")
        if expanded_max_bytes is None or expanded_max_bytes <= 0:
            raise ValueError("gzip recipe requires a dataset-specific expansion limit")
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                expanded_size = 0
                while chunk := stream.read(_STREAM_CHUNK_BYTES):
                    expanded_size += len(chunk)
                    if expanded_size > expanded_max_bytes:
                        raise ValueError("raw gzip expanded size is invalid")
        except (OSError, EOFError) as error:
            raise ValueError("raw gzip source is invalid") from error
        if expanded_size == 0:
            raise ValueError("raw gzip expanded size is invalid")
    elif source.archive_type == "mat":
        if len(data) < 128 or not data.startswith(b"MATLAB"):
            raise ValueError("raw MAT source header is invalid")
    elif source.archive_type == "npy":
        if len(data) < 10 or not data.startswith(b"\x93NUMPY"):
            raise ValueError("raw NPY source header is invalid")
    elif source.archive_type == "npz":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = [item for item in archive.infolist() if not item.is_dir()]
                names = [item.filename for item in members]
                if (
                    not names
                    or len(names) != len(set(names))
                    or any(
                        "/" in name
                        or "\\" in name
                        or not name.endswith(".npy")
                        or item.flag_bits & 0x1
                        or item.file_size > _MAX_BUNDLE_BYTES
                        for name, item in zip(names, members, strict=True)
                    )
                ):
                    raise ValueError("raw NPZ inventory is invalid")
        except (OSError, zipfile.BadZipFile) as error:
            raise ValueError("raw NPZ source is invalid") from error
    elif source.archive_type == "plain" and not data:
        raise ValueError("raw plain source must not be empty")


def _read_raw_inventory(
    root: Path,
    requirement: FormalCorpusRequirement,
    *,
    require_all: bool,
) -> tuple[tuple[FileEvidence, ...], dict[str, str], dict[str, bytes]]:
    files: list[FileEvidence] = []
    hashes: dict[str, str] = {}
    snapshots: dict[str, bytes] = {}
    for source, relative_path in _source_pairs(requirement):
        data, present = _read_if_present(root, relative_path, max_bytes=source.max_bytes)
        if not present or data is None:
            if require_all:
                raise FileNotFoundError("formal raw source is missing")
            continue
        expanded_max = _GZIP_EXPANDED_LIMITS.get((requirement.recipe_id, source.source_id))
        _validate_raw_format(data, source, expanded_max_bytes=expanded_max)
        observed_hash = _sha256(data)
        if source.expected_sha256 is not None and observed_hash != source.expected_sha256:
            raise ValueError("raw source SHA-256 does not match locked recipe")
        hashes[source.source_id] = observed_hash
        snapshots[source.source_id] = data
        files.append(_file_evidence(relative_path, data, "raw"))
    return (
        tuple(sorted(files, key=lambda item: item.relative_path)),
        dict(sorted(hashes.items())),
        dict(sorted(snapshots.items())),
    )


def _validate_formal_semantics(
    requirement: FormalCorpusRequirement,
    bundle: CoreGraphBundle,
    labels: ExperimentLabels,
    split_inventory: ExperimentSplitInventory,
) -> None:
    if not bundle.nodes or not bundle.edges:
        raise ValueError("formal graph must contain nodes and edges")
    expected_directed = requirement.recipe_id == "wiki-rfa"
    if bundle.directed != expected_directed:
        raise ValueError("formal graph directionality does not match recipe")
    if bundle.split_manifest.strategy != requirement.expected_split_policy:
        raise ValueError("formal graph split strategy does not match recipe")
    expected_count = requirement.official_split_count or 1
    expected_ids = (
        tuple(f"official-{index:02d}" for index in range(expected_count))
        if requirement.official_split_count is not None
        else ("primary",)
    )
    if (
        split_inventory.requirement_id != requirement.requirement_id
        or len(split_inventory.folds) != expected_count
        or tuple(fold.fold_id for fold in split_inventory.folds) != expected_ids
        or split_inventory.folds[0].split_manifest != bundle.split_manifest
        or any(
            fold.split_manifest.strategy != requirement.expected_split_policy
            for fold in split_inventory.folds
        )
    ):
        raise ValueError("formal split inventory is incomplete or non-authoritative")
    recipe = load_dataset_recipes()[requirement.recipe_id]
    target_fields = {task.target_field for task in recipe.tasks.values()}
    feature_names = {feature.name for feature in bundle.node_features}
    leaked = target_fields & feature_names
    if leaked:
        raise ValueError("formal graph exposes a target field as model input")
    label_names = tuple(target.name for target in labels.targets)
    if not set(label_names) <= set(recipe.tasks):
        raise ValueError("formal label inventory references an unknown recipe task")
    if bundle.directed and requirement.recipe_id == "wiki-rfa":
        if {edge.edge_type for edge in bundle.edges} - {"support", "oppose"}:
            raise ValueError("Wiki-RfA formal edges must use support/oppose types")


def _read_if_present(
    root: Path, relative_path: str, *, max_bytes: int
) -> tuple[bytes | None, bool]:
    target = root.joinpath(*relative_path.split("/"))
    if not target.exists() and not target.is_symlink():
        return None, False
    return read_confined_snapshot(root, relative_path, max_bytes=max_bytes), True


def _file_evidence(relative_path: str, data: bytes, purpose: str) -> FileEvidence:
    return FileEvidence.model_validate(
        {
            "relativePath": relative_path,
            "sha256": _sha256(data),
            "sizeBytes": len(data),
            "purpose": purpose,
        }
    )


def _observation_from_manifest(
    root: Path, requirement: FormalCorpusRequirement, manifest_bytes: bytes
) -> FormalCorpusObservation:
    try:
        manifest = ExperimentDatasetManifest.model_validate_json(manifest_bytes)
        recipes = load_dataset_recipes()
        recipe = recipes[requirement.recipe_id]
        expected_identity = (
            requirement.requirement_id,
            requirement.recipe_id,
            recipe.recipe_version,
            recipe.recipe_sha256,
            requirement.graph_id,
            requirement.required_usage_scope,
            requirement.expected_split_policy,
            requirement.experiment_split_policy,
            requirement.official_split_count,
            requirement.manifest_relative_path,
        )
        observed_identity = (
            manifest.requirement_id,
            manifest.recipe_id,
            manifest.recipe_version,
            manifest.recipe_sha256,
            manifest.graph_id,
            manifest.usage_scope,
            manifest.split_policy,
            manifest.experiment_split_policy,
            manifest.split_count if requirement.official_split_count is not None else None,
            manifest.manifest_relative_path,
        )
        if observed_identity != expected_identity:
            raise ValueError("manifest identity does not match fixed requirement")
    except Exception:
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="invalid",
            reasonCode="manifest-invalid",
            files=(_file_evidence(requirement.manifest_relative_path, manifest_bytes, "manifest"),),
        )

    manifest_file = _file_evidence(requirement.manifest_relative_path, manifest_bytes, "manifest")
    try:
        bundle_bytes = read_confined_snapshot(
            root, manifest.bundle_relative_path, max_bytes=_MAX_BUNDLE_BYTES
        )
        labels_bytes = read_confined_snapshot(
            root, manifest.labels_relative_path, max_bytes=_MAX_LABEL_BYTES
        )
        split_inventory_bytes = read_confined_snapshot(
            root,
            manifest.split_inventory_relative_path,
            max_bytes=_MAX_SPLIT_INVENTORY_BYTES,
        )
        if _sha256(bundle_bytes) != manifest.bundle_sha256:
            raise ValueError("bundle byte hash mismatch")
        if _sha256(labels_bytes) != manifest.labels_sha256:
            raise ValueError("labels byte hash mismatch")
        if _sha256(split_inventory_bytes) != manifest.split_inventory_sha256:
            raise ValueError("split inventory byte hash mismatch")
        bundle = load_core_graph_bundle_json(bundle_bytes)
        labels = ExperimentLabels.model_validate_json(labels_bytes)
        split_inventory = ExperimentSplitInventory.model_validate_json(split_inventory_bytes)
        if (
            bundle_bytes != _canonical_bytes(bundle)
            or labels_bytes != _canonical_bytes(labels)
            or split_inventory_bytes != _canonical_bytes(split_inventory)
        ):
            raise ValueError("experiment artifacts are not canonical JSON plus newline")
        split_hash = canonical_sha256(
            bundle.split_manifest.model_dump(mode="python", by_alias=True)
        )
        if (
            bundle.graph_version_hash != manifest.graph_version_hash
            or bundle.source.source_sha256 != manifest.source_sha256
            or split_hash != manifest.split_manifest_hash
            or labels.requirement_id != requirement.requirement_id
            or tuple(target.name for target in labels.targets) != manifest.label_names
            or labels.labels_hash != canonical_sha256(labels.targets)
            or split_inventory.requirement_id != requirement.requirement_id
            or len(split_inventory.folds) != manifest.split_count
            or tuple(fold.fold_id for fold in split_inventory.folds) != manifest.split_ids
            or tuple(fold.split_manifest_hash for fold in split_inventory.folds)
            != manifest.split_manifest_hashes
        ):
            raise ValueError("experiment artifact semantic binding mismatch")
    except Exception:
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="invalid",
            reasonCode="artifact-validation-failed",
            manifestHash=manifest.manifest_hash,
            files=(manifest_file,),
        )

    core_files = tuple(
        sorted(
            (
                manifest_file,
                _file_evidence(manifest.bundle_relative_path, bundle_bytes, "bundle"),
                _file_evidence(manifest.labels_relative_path, labels_bytes, "labels"),
                _file_evidence(
                    manifest.split_inventory_relative_path,
                    split_inventory_bytes,
                    "split-inventory",
                ),
            ),
            key=lambda item: item.relative_path,
        )
    )
    if manifest.phase_eligibility != "formal":
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="usage-ineligible",
            reasonCode="phase-not-formal",
            manifestHash=manifest.manifest_hash,
            graphVersionHash=bundle.graph_version_hash,
            splitManifestHash=manifest.split_manifest_hash,
            files=core_files,
        )
    sources = tuple(source for source, _path in _source_pairs(requirement))
    if any(source.expected_sha256 is None for source in sources):
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="usage-ineligible",
            reasonCode="source-hash-unlocked",
            manifestHash=manifest.manifest_hash,
            graphVersionHash=bundle.graph_version_hash,
            splitManifestHash=manifest.split_manifest_hash,
            files=core_files,
        )
    binding = formal_materializer_binding(
        requirement_id=requirement.requirement_id,
        recipe=recipe,
        graph_id=requirement.graph_id,
        bundle_split_policy=requirement.expected_split_policy,
        experiment_split_policy=requirement.experiment_split_policy,
    )
    if binding is None:
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="usage-ineligible",
            reasonCode="formal-materializer-unavailable",
            manifestHash=manifest.manifest_hash,
            graphVersionHash=bundle.graph_version_hash,
            splitManifestHash=manifest.split_manifest_hash,
            files=core_files,
        )
    if (
        manifest.materializer_id,
        manifest.materializer_version,
        manifest.materializer_code_sha256,
        manifest.materialization_protocol_hash,
    ) != (
        binding.materializer_id,
        binding.materializer_version,
        binding.materializer_code_sha256,
        binding.materialization_protocol_hash,
    ):
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="invalid",
            reasonCode="formal-materializer-identity-mismatch",
            manifestHash=manifest.manifest_hash,
            graphVersionHash=bundle.graph_version_hash,
            splitManifestHash=manifest.split_manifest_hash,
            files=core_files,
        )
    try:
        raw_files, raw_hashes, raw_snapshots = _read_raw_inventory(
            root, requirement, require_all=True
        )
    except FileNotFoundError:
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="usage-ineligible",
            reasonCode="formal-raw-source-missing",
            manifestHash=manifest.manifest_hash,
            graphVersionHash=bundle.graph_version_hash,
            splitManifestHash=manifest.split_manifest_hash,
            files=core_files,
        )
    except Exception:
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="invalid",
            reasonCode="raw-source-invalid",
            manifestHash=manifest.manifest_hash,
            graphVersionHash=bundle.graph_version_hash,
            splitManifestHash=manifest.split_manifest_hash,
            files=core_files,
        )
    combined_source_hash = canonical_sha256(raw_hashes)
    try:
        derived = derive_registered_formal_dataset(
            requirement_id=requirement.requirement_id,
            recipe=recipe,
            graph_id=requirement.graph_id,
            raw_sources=raw_snapshots,
            combined_source_sha256=combined_source_hash,
            bundle_split_policy=requirement.expected_split_policy,
        )
        expected_labels = _labels_document(requirement.requirement_id, derived.labels)
    except Exception:
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="invalid",
            reasonCode="formal-materialization-validation-failed",
            manifestHash=manifest.manifest_hash,
            graphVersionHash=bundle.graph_version_hash,
            splitManifestHash=manifest.split_manifest_hash,
            files=tuple(sorted((*core_files, *raw_files), key=lambda item: item.relative_path)),
        )
    expected_splits = _split_inventory_document(requirement, derived.split_manifests)
    if bundle != derived.bundle or labels != expected_labels or split_inventory != expected_splits:
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="invalid",
            reasonCode="formal-materialization-mismatch",
            manifestHash=manifest.manifest_hash,
            graphVersionHash=bundle.graph_version_hash,
            splitManifestHash=manifest.split_manifest_hash,
            files=tuple(sorted((*core_files, *raw_files), key=lambda item: item.relative_path)),
        )
    try:
        _validate_formal_semantics(requirement, bundle, labels, split_inventory)
    except Exception:
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="invalid",
            reasonCode="formal-semantic-validation-failed",
            manifestHash=manifest.manifest_hash,
            graphVersionHash=bundle.graph_version_hash,
            splitManifestHash=manifest.split_manifest_hash,
            files=tuple(sorted((*core_files, *raw_files), key=lambda item: item.relative_path)),
        )
    files = tuple(sorted((*core_files, *raw_files), key=lambda item: item.relative_path))
    return FormalCorpusObservation(
        requirementId=requirement.requirement_id,
        status="ready",
        reasonCode="validated-formal-dataset",
        manifestHash=manifest.manifest_hash,
        graphVersionHash=bundle.graph_version_hash,
        splitManifestHash=manifest.split_manifest_hash,
        files=files,
    )


def _observe_requirement(
    root: Path, requirement: FormalCorpusRequirement
) -> FormalCorpusObservation:
    try:
        manifest_bytes, present = _read_if_present(
            root, requirement.manifest_relative_path, max_bytes=_MAX_MANIFEST_BYTES
        )
    except Exception:
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="invalid",
            reasonCode="manifest-invalid",
            files=(),
        )
    if present and manifest_bytes is not None:
        return _observation_from_manifest(root, requirement, manifest_bytes)

    try:
        raw_files, _raw_hashes, _raw_snapshots = _read_raw_inventory(
            root, requirement, require_all=False
        )
    except Exception:
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="invalid",
            reasonCode="raw-source-invalid",
            files=(),
        )
    if raw_files:
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="raw-only",
            reasonCode="raw-source-not-materialized",
            files=tuple(sorted(raw_files, key=lambda item: item.relative_path)),
        )

    audit_files: list[FileEvidence] = []
    audit_invalid = False
    for relative_path in requirement.audit_relative_paths:
        try:
            data, present = _read_if_present(root, relative_path, max_bytes=_MAX_BUNDLE_BYTES)
            if present and data is not None:
                audit_files.append(_file_evidence(relative_path, data, "audit"))
        except Exception:
            audit_invalid = True
    if audit_invalid:
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="invalid",
            reasonCode="audit-artifact-invalid",
            files=tuple(sorted(audit_files, key=lambda item: item.relative_path)),
        )
    if audit_files:
        return FormalCorpusObservation(
            requirementId=requirement.requirement_id,
            status="audit-only",
            reasonCode="audit-artifact-ineligible",
            files=tuple(sorted(audit_files, key=lambda item: item.relative_path)),
        )
    return FormalCorpusObservation(
        requirementId=requirement.requirement_id,
        status="missing",
        reasonCode="formal-dataset-manifest-missing",
        files=(),
    )


def _requirements_hash() -> str:
    return canonical_sha256(FORMAL_CORPUS_REQUIREMENTS)


def _recipe_catalog_hash() -> str:
    recipes = load_dataset_recipes()
    return canonical_sha256(
        {
            recipe_id: recipe.model_dump(mode="python", by_alias=True)
            for recipe_id, recipe in sorted(recipes.items())
        }
    )


def _hold_exact_evidence(
    parent_lease: _PublicationParentLease,
    target: Path,
    serialized: bytes,
    *,
    max_bytes: int,
) -> _OwnedFileLease | None:
    held = _OwnedFileLease(
        target,
        _path_identity(target),
        deletable=False,
        parent_lease=parent_lease,
    )
    try:
        if held.read(max_bytes=max_bytes) != serialized:
            held.close()
            return None
        return held
    except Exception:
        held.close()
        raise


def _publish_exact(
    authorized_root: Path,
    path: Path,
    serialized: bytes,
    *,
    conflict_message: str,
) -> None:
    path = reject_link_components(path)
    parent_lease = _PublicationParentLease(authorized_root, path.parent, create=True)
    parent = parent_lease.parent
    target = parent / path.name
    try:
        lock = _PublisherLock(
            parent_lease,
            f".{path.name}.publisher.lock",
            active_message="preflight evidence already has an active publisher",
        )
    except Exception:
        parent_lease.close()
        raise
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary_identity: tuple[int, int] | None = None
    lease: _OwnedFileLease | None = None
    exact_publication_verified = False
    maximum = max(len(serialized), _MAX_MANIFEST_BYTES)
    try:
        if target.exists() or target.is_symlink():
            lease = _hold_exact_evidence(
                parent_lease,
                target,
                serialized,
                max_bytes=maximum,
            )
            if lease is None:
                raise FileExistsError(conflict_message)
            parent_lease.flush()
            if lease.read(max_bytes=maximum) != serialized:
                raise ValueError("existing evidence changed after directory flush")
            exact_publication_verified = True
            return
        descriptor = parent_lease.open_file(temporary.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        expected_identity = _path_identity(temporary)
        temporary_identity = expected_identity
        _PUBLICATION_SEAM("evidence", target)
        try:
            parent_lease.commit_file_no_replace(temporary.name, target.name)
        except FileExistsError as error:
            lease = _hold_exact_evidence(
                parent_lease,
                target,
                serialized,
                max_bytes=maximum,
            )
            if lease is None:
                raise FileExistsError(conflict_message) from error
            parent_lease.flush()
            if lease.read(max_bytes=maximum) != serialized:
                raise ValueError("racing exact evidence changed after directory flush")
            exact_publication_verified = True
            return
        _PUBLICATION_SEAM("evidence-post-link", target)
        lease = _OwnedFileLease(target, expected_identity, parent_lease=parent_lease)
        try:
            parent_lease.flush()
            if _verify_owned_evidence(lease, max_bytes=maximum) != serialized:
                raise ValueError("atomic evidence publication verification failed")
        except Exception:
            # A transient reopen failure may leave an exact, pinned immutable link.
            # A byte mismatch on the owned inode is removed through that held handle;
            # an identity replacement is never deleted by this publisher.
            if lease.read(max_bytes=maximum) != serialized:
                removed = lease.remove_owned_link()
                if removed:
                    parent_lease.flush()
            raise
        exact_publication_verified = True
    finally:
        try:
            lock.close()
        finally:
            try:
                if temporary_identity is not None:
                    _remove_owned_file_path(
                        temporary,
                        temporary_identity,
                        parent_lease=parent_lease,
                    )
                if exact_publication_verified and lease is not None:
                    lease.assert_visible_binding()
                    if lease.read(max_bytes=maximum) != serialized:
                        raise ValueError("evidence changed during publication cleanup")
                    lease.assert_visible_binding()
            finally:
                try:
                    if lease is not None:
                        lease.close()
                finally:
                    parent_lease.close()


def run_formal_preflight(
    runtime_root: Path, *, publish_to: Path | None = None
) -> FormalPreflightEvidence:
    """Inspect a fixed formal corpus without downloading or mutating serving state."""

    root = secure_existing_root(runtime_root)
    observations = tuple(
        _observe_requirement(root, requirement) for requirement in FORMAL_CORPUS_REQUIREMENTS
    )
    formal_ready = all(item.status == "ready" for item in observations)
    payload: dict[str, Any] = {
        "schemaVersion": _PREFLIGHT_SCHEMA,
        "requirementsHash": _requirements_hash(),
        "recipeCatalogHash": _recipe_catalog_hash(),
        "observations": [item.model_dump(mode="python", by_alias=True) for item in observations],
        "formalReady": formal_ready,
        "promotable": formal_ready,
    }
    payload["evidenceHash"] = canonical_sha256(payload)
    evidence = FormalPreflightEvidence.model_validate(payload)
    if publish_to is not None:
        _publish_exact(
            root,
            Path(publish_to),
            _canonical_bytes(evidence),
            conflict_message="conflicting formal preflight evidence already exists",
        )
    return evidence


def load_formal_preflight(
    path: Path, *, runtime_root: Path | None = None
) -> FormalPreflightEvidence:
    """Reload canonical preflight evidence and verify its semantic hash."""

    lexical = reject_link_components(path)
    root = secure_existing_root(lexical.parent)
    serialized = read_confined_snapshot(root, lexical.name, max_bytes=_MAX_MANIFEST_BYTES)
    evidence = FormalPreflightEvidence.model_validate_json(serialized)
    if serialized != _canonical_bytes(evidence):
        raise ValueError("formal preflight evidence is not canonical JSON plus newline")
    if evidence.formal_ready and runtime_root is None:
        raise ValueError("formal-ready evidence requires runtime byte revalidation")
    if runtime_root is not None and run_formal_preflight(runtime_root) != evidence:
        raise ValueError("formal preflight evidence does not match current runtime bytes")
    return evidence


def _label_records(
    labels: dict[str, dict[str, ScalarLabel]],
) -> tuple[LabelTarget, ...]:
    return tuple(
        LabelTarget(
            name=name,
            values=tuple(
                LabelValue(entityId=entity_id, value=value)
                for entity_id, value in sorted(values.items())
            ),
        )
        for name, values in sorted(labels.items())
    )


def _labels_document(
    requirement_id: str, labels: dict[str, dict[str, ScalarLabel]]
) -> ExperimentLabels:
    records = _label_records(labels)
    return ExperimentLabels.model_validate(
        {
            "schemaVersion": _LABEL_SCHEMA,
            "requirementId": requirement_id,
            "targets": [item.model_dump(mode="python", by_alias=True) for item in records],
            "labelsHash": canonical_sha256(records),
        }
    )


def _split_inventory_document(
    requirement: FormalCorpusRequirement,
    split_manifests: tuple[SplitManifest, ...],
) -> ExperimentSplitInventory:
    expected_count = requirement.official_split_count or 1
    if len(split_manifests) != expected_count:
        raise ValueError("split inventory does not match the fixed official split count")
    fold_ids = (
        tuple(f"official-{index:02d}" for index in range(expected_count))
        if requirement.official_split_count is not None
        else ("primary",)
    )
    folds = tuple(
        ExperimentSplitFold(
            foldId=fold_id,
            splitManifest=split_manifest,
            splitManifestHash=canonical_sha256(
                split_manifest.model_dump(mode="python", by_alias=True)
            ),
        )
        for fold_id, split_manifest in zip(fold_ids, split_manifests, strict=True)
    )
    payload: dict[str, Any] = {
        "schemaVersion": _SPLIT_INVENTORY_SCHEMA,
        "requirementId": requirement.requirement_id,
        "folds": [fold.model_dump(mode="python", by_alias=True) for fold in folds],
    }
    payload["inventoryHash"] = canonical_sha256(payload)
    return ExperimentSplitInventory.model_validate(payload)


def _formal_source_hashes(
    root: Path, requirement: FormalCorpusRequirement
) -> tuple[dict[str, str], dict[str, bytes]]:
    sources = tuple(source for source, _path in _source_pairs(requirement))
    if any(source.expected_sha256 is None for source in sources):
        raise ValueError("every formal source SHA-256 must be locked in the recipe")
    _files, hashes, snapshots = _read_raw_inventory(root, requirement, require_all=True)
    return hashes, snapshots


def _win_open_native(path: Path, *, access: int, share: int, directory: bool) -> int:
    if os.name != "nt":  # pragma: no cover - guarded by callers
        raise RuntimeError("Win32 handle operations are unavailable")
    flags = _FILE_FLAG_OPEN_REPARSE_POINT
    if directory:
        flags |= _FILE_FLAG_BACKUP_SEMANTICS
    handle = _CreateFileW(str(path), access, share, None, _OPEN_EXISTING, flags, None)
    if handle == _INVALID_HANDLE_VALUE:
        raise OSError(ctypes.get_last_error(), "failed to open held filesystem handle")
    return int(handle)


def _win_info(handle: int) -> Any:
    details = _BY_HANDLE_FILE_INFORMATION()
    if not _GetFileInformationByHandle(handle, ctypes.byref(details)):
        raise OSError(ctypes.get_last_error(), "failed to inspect held filesystem handle")
    return details


def _win_identity(handle: int) -> tuple[int, int]:
    details = _win_info(handle)
    return (
        int(details.dwVolumeSerialNumber),
        (int(details.nFileIndexHigh) << 32) | int(details.nFileIndexLow),
    )


def _close_win_handle(handle: int) -> None:
    if not _CloseHandle(handle):
        raise OSError(ctypes.get_last_error(), "failed to close held filesystem handle")


def _win_final_path(handle: int) -> Path:
    if os.name != "nt":  # pragma: no cover - guarded by callers
        raise RuntimeError("Win32 handle operations are unavailable")
    required = _GetFinalPathNameByHandleW(handle, None, 0, 0)
    if not required:
        raise OSError(ctypes.get_last_error(), "failed to size held filesystem path")
    buffer = ctypes.create_unicode_buffer(required + 1)
    if not _GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0):
        raise OSError(ctypes.get_last_error(), "failed to resolve held filesystem path")
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)


def _path_identity(path: Path) -> tuple[int, int]:
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or bool(
        getattr(details, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    ):
        raise ValueError("owned publication path became a link or reparse point")
    if os.name != "nt":
        return int(details.st_dev), int(details.st_ino)
    directory = stat.S_ISDIR(details.st_mode)
    handle = _win_open_native(
        path,
        access=_FILE_READ_ATTRIBUTES,
        share=_FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        directory=directory,
    )
    try:
        opened = _win_info(handle)
        if bool(opened.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError("owned publication path became a reparse point")
        return _win_identity(handle)
    finally:
        _close_win_handle(handle)


class _PublicationParentLease:
    """Pin the authorized root-to-parent chain for one publication lifetime."""

    def __init__(self, authorized_root: Path, parent: Path, *, create: bool) -> None:
        self.root = secure_existing_root(authorized_root)
        self.parent = reject_link_components(parent)
        try:
            relative_parts = self.parent.relative_to(self.root).parts
        except ValueError as error:
            raise ValueError("publication parent escapes the authorized runtime") from error
        self._paths = tuple(
            [self.root]
            + [
                self.root.joinpath(*relative_parts[:index])
                for index in range(1, len(relative_parts) + 1)
            ]
        )
        self._handles: list[int] = []
        self._descriptors: list[int] = []
        self._identities: list[tuple[int, int]] = []
        try:
            if os.name == "nt":
                self._open_windows(create=create)
            else:
                self._open_posix(relative_parts, create=create)
            self.assert_confined()
        except Exception:
            self.close()
            raise

    def _open_windows(self, *, create: bool) -> None:
        for index, path in enumerate(self._paths):
            if index and not path.exists() and not path.is_symlink():
                if not create:
                    raise FileNotFoundError(path)
                path.mkdir()
            handle = _win_open_native(
                path,
                access=_FILE_READ_ATTRIBUTES
                | (_GENERIC_WRITE if index == len(self._paths) - 1 else 0),
                # Ancestors deny both write and delete sharing.  The immediate
                # publication parent must permit directory-entry mutation by
                # this process, but still denies delete sharing so it cannot be
                # renamed or replaced by a junction during publication.
                share=_FILE_SHARE_READ
                | (_FILE_SHARE_WRITE if index == len(self._paths) - 1 else 0),
                directory=True,
            )
            self._handles.append(handle)
            details = _win_info(handle)
            if (
                not details.dwFileAttributes & _FILE_ATTRIBUTE_DIRECTORY
                or details.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT
            ):
                raise ValueError("publication path contains a reparse component")
            self._identities.append(_win_identity(handle))

    def _open_posix(self, relative_parts: tuple[str, ...], *, create: bool) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        root_descriptor = os.open(self.root, flags)
        self._descriptors.append(root_descriptor)
        opened = os.fstat(root_descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            raise ValueError("authorized publication root is not a directory")
        self._identities.append((int(opened.st_dev), int(opened.st_ino)))
        for part in relative_parts:
            try:
                descriptor = os.open(part, flags, dir_fd=self._descriptors[-1])
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd=self._descriptors[-1])
                descriptor = os.open(part, flags, dir_fd=self._descriptors[-1])
            self._descriptors.append(descriptor)
            opened = os.fstat(descriptor)
            if not stat.S_ISDIR(opened.st_mode):
                raise ValueError("publication path contains a non-directory component")
            self._identities.append((int(opened.st_dev), int(opened.st_ino)))

    @property
    def descriptor(self) -> int:
        if not self._descriptors:
            raise RuntimeError("POSIX publication parent descriptor is unavailable")
        return self._descriptors[-1]

    @staticmethod
    def _validate_child_name(name: str) -> None:
        if not name or name in {".", ".."} or "/" in name or "\\" in name or ":" in name:
            raise ValueError("publication child name is unsafe")

    def assert_confined(self) -> None:
        if os.name == "nt":
            trusted = os.path.normcase(os.path.abspath(self.root))
            for path, handle, expected_identity in zip(
                self._paths, self._handles, self._identities, strict=True
            ):
                details = _win_info(handle)
                if (
                    not details.dwFileAttributes & _FILE_ATTRIBUTE_DIRECTORY
                    or details.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT
                    or _win_identity(handle) != expected_identity
                ):
                    raise ValueError("held publication path identity changed")
                observed = os.path.normcase(os.path.abspath(_win_final_path(handle)))
                expected = os.path.normcase(os.path.abspath(path))
                if observed != expected or os.path.commonpath((observed, trusted)) != trusted:
                    raise ValueError("held publication path escaped the authorized runtime")
            return
        root_details = os.stat(self.root, follow_symlinks=False)
        if (
            int(root_details.st_dev),
            int(root_details.st_ino),
        ) != self._identities[0] or not stat.S_ISDIR(root_details.st_mode):
            raise ValueError("authorized publication root identity changed")
        for index, part in enumerate(self.parent.relative_to(self.root).parts, start=1):
            details = os.stat(part, dir_fd=self._descriptors[index - 1], follow_symlinks=False)
            if (
                int(details.st_dev),
                int(details.st_ino),
            ) != self._identities[index] or not stat.S_ISDIR(details.st_mode):
                raise ValueError("held publication ancestor identity changed")
        for descriptor, expected_identity in zip(self._descriptors, self._identities, strict=True):
            details = os.fstat(descriptor)
            if (
                int(details.st_dev),
                int(details.st_ino),
            ) != expected_identity or not stat.S_ISDIR(details.st_mode):
                raise ValueError("held publication directory identity changed")

    def open_file(self, name: str, flags: int, mode: int = 0o600) -> int:
        self._validate_child_name(name)
        self.assert_confined()
        if os.name == "nt":
            return os.open(self.parent / name, flags, mode)
        return os.open(name, flags, mode, dir_fd=self.descriptor)

    def make_directory(self, name: str) -> None:
        self._validate_child_name(name)
        self.assert_confined()
        if os.name == "nt":
            (self.parent / name).mkdir()
        else:
            os.mkdir(name, 0o700, dir_fd=self.descriptor)

    def commit_file_no_replace(self, source_name: str, target_name: str) -> None:
        self._validate_child_name(source_name)
        self._validate_child_name(target_name)
        self.assert_confined()
        if os.name == "nt":
            # Windows rename is atomic and fails when the destination already
            # exists.  Moving the temporary file avoids a second hard-link that
            # would otherwise have to be removed after the target lease closes.
            os.rename(self.parent / source_name, self.parent / target_name)
            return
        if not sys.platform.startswith("linux"):
            raise RuntimeError("atomic no-clobber file publication is unavailable")
        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("atomic no-clobber file publication is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            self.descriptor,
            os.fsencode(source_name),
            self.descriptor,
            os.fsencode(target_name),
            1,
        )
        if result != 0:
            code = ctypes.get_errno()
            if code in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(target_name)
            raise OSError(code, os.strerror(code), self.parent / target_name)

    def flush(self) -> None:
        self.assert_confined()
        if os.name == "nt":
            handle = self._handles[-1]
            if not _FlushFileBuffers(handle):
                code = ctypes.get_last_error()
                if code not in _UNSUPPORTED_DIRECTORY_FLUSH_ERRORS:
                    raise OSError(code, "failed to flush held publication directory")
            return
        os.fsync(self.descriptor)

    def close(self) -> None:
        while self._handles:
            _close_win_handle(self._handles.pop())
        while self._descriptors:
            os.close(self._descriptors.pop())


class _OwnedFileLease:
    """Pin one just-linked evidence inode and delete only that exact link."""

    def __init__(
        self,
        target: Path,
        expected_identity: tuple[int, int],
        *,
        deletable: bool = True,
        parent_lease: _PublicationParentLease | None = None,
        directory_lease: _OwnedDirectoryLease | None = None,
    ) -> None:
        self.target = target
        self.identity = expected_identity
        self.deletable = deletable
        self.parent_lease = parent_lease
        self.directory_lease = directory_lease
        self._handle: int | None = None
        self._descriptor: int | None = None
        self._assert_parent()
        if os.name == "nt":
            handle = _win_open_native(
                target,
                access=_GENERIC_READ | (_DELETE if deletable else 0),
                share=_FILE_SHARE_READ,
                directory=False,
            )
            self._handle = handle
            details = _win_info(handle)
            if (
                details.dwFileAttributes
                & (_FILE_ATTRIBUTE_DIRECTORY | _FILE_ATTRIBUTE_REPARSE_POINT)
                or _win_identity(handle) != expected_identity
            ):
                self.close()
                raise ValueError("atomic evidence publication identity changed")
            return
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        if directory_lease is not None:
            descriptor = os.open(target.name, flags, dir_fd=directory_lease.descriptor)
        elif parent_lease is not None:
            descriptor = os.open(target.name, flags, dir_fd=parent_lease.descriptor)
        else:
            descriptor = os.open(target, flags)
        self._descriptor = descriptor
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (int(opened.st_dev), int(opened.st_ino)) != expected_identity
        ):
            self.close()
            raise ValueError("atomic evidence publication identity changed")

    def _assert_parent(self) -> None:
        if self.parent_lease is not None:
            if self.target.parent != self.parent_lease.parent:
                raise ValueError("owned file is not under its held publication parent")
            self.parent_lease.assert_confined()
        if self.directory_lease is not None:
            if self.target.parent != self.directory_lease.target:
                raise ValueError("owned child is not under its held directory")
            self.directory_lease.assert_identity()

    def assert_visible_binding(self) -> None:
        """Prove the visible basename still names this held regular-file identity."""

        self._assert_parent()
        if self._handle is not None:
            details = _win_info(self._handle)
            if (
                details.dwFileAttributes
                & (_FILE_ATTRIBUTE_DIRECTORY | _FILE_ATTRIBUTE_REPARSE_POINT)
                or _win_identity(self._handle) != self.identity
                or os.path.normcase(os.path.abspath(_win_final_path(self._handle)))
                != os.path.normcase(os.path.abspath(self.target))
            ):
                raise ValueError("visible evidence identity changed during publication")
            return
        if self._descriptor is None:
            raise RuntimeError("evidence lease is closed")
        if self.directory_lease is not None:
            details = os.stat(
                self.target.name,
                dir_fd=self.directory_lease.descriptor,
                follow_symlinks=False,
            )
        elif self.parent_lease is not None:
            details = os.stat(
                self.target.name,
                dir_fd=self.parent_lease.descriptor,
                follow_symlinks=False,
            )
        else:
            details = self.target.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or (int(details.st_dev), int(details.st_ino)) != self.identity
        ):
            raise ValueError("visible evidence identity changed during publication")

    def read(self, *, max_bytes: int) -> bytes:
        if self._handle is not None:
            details = _win_info(self._handle)
            size = (int(details.nFileSizeHigh) << 32) | int(details.nFileSizeLow)
            if size < 1 or size > max_bytes:
                raise ValueError("held evidence size is outside the authorized bound")
            if not _SetFilePointerEx(self._handle, ctypes.c_longlong(0), None, _FILE_BEGIN):
                raise OSError(ctypes.get_last_error(), "failed to rewind held evidence")
            chunks: list[bytes] = []
            remaining = size
            while remaining:
                length = min(_STREAM_CHUNK_BYTES, remaining)
                buffer = ctypes.create_string_buffer(length)
                read = wintypes.DWORD()
                if not _ReadFile(self._handle, buffer, length, ctypes.byref(read), None):
                    raise OSError(ctypes.get_last_error(), "failed to read held evidence")
                if not read.value:
                    raise ValueError("held evidence changed during verification")
                chunks.append(buffer.raw[: read.value])
                remaining -= int(read.value)
            return b"".join(chunks)
        if self._descriptor is None:  # pragma: no cover - internal invariant
            raise RuntimeError("evidence lease is closed")
        self._assert_parent()
        details = os.fstat(self._descriptor)
        if details.st_size < 1 or details.st_size > max_bytes:
            raise ValueError("held evidence size is outside the authorized bound")
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        chunks = []
        remaining = details.st_size
        while remaining:
            chunk = os.read(self._descriptor, min(_STREAM_CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError("held evidence changed during verification")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(self._descriptor, 1):
            raise ValueError("held evidence grew during verification")
        after = os.fstat(self._descriptor)
        if (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
        ) != (
            int(details.st_dev),
            int(details.st_ino),
            int(details.st_size),
        ):
            raise ValueError("held evidence changed during verification")
        return b"".join(chunks)

    def remove_owned_link(self) -> bool:
        if not self.deletable:
            self.close()
            return False
        self._assert_parent()
        if self._handle is not None:
            disposition = _FILE_DISPOSITION_INFO(True)
            if not _SetFileInformationByHandle(
                self._handle,
                _FILE_DISPOSITION_INFO_CLASS,
                ctypes.byref(disposition),
                ctypes.sizeof(disposition),
            ):
                raise OSError(ctypes.get_last_error(), "failed to remove owned evidence link")
            self.close()
            return not self.target.exists() and not self.target.is_symlink()
        if self._descriptor is None:  # pragma: no cover - internal invariant
            return False
        # POSIX does not provide a portable unlink-by-open-handle primitive.
        # A stat-then-unlink sequence could delete a competitor replacement, so
        # cleanup deliberately preserves the orphan for operator quarantine.
        self.close()
        return False

    def close(self) -> None:
        if self._handle is not None:
            handle, self._handle = self._handle, None
            _close_win_handle(handle)
        if self._descriptor is not None:
            descriptor, self._descriptor = self._descriptor, None
            os.close(descriptor)


class _PublisherLock:
    """Hold publication exclusion without ever pathname-unlinking a replacement."""

    def __init__(
        self,
        parent_lease: _PublicationParentLease,
        name: str,
        *,
        active_message: str,
    ) -> None:
        self.parent_lease = parent_lease
        self.path = parent_lease.parent / name
        self._lease: _OwnedFileLease | None = None
        self._descriptor: int | None = None
        if os.name == "nt":
            try:
                descriptor = parent_lease.open_file(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as error:
                raise RuntimeError(active_message) from error
            identity: tuple[int, int] | None = None
            try:
                identity = _path_identity(self.path)
            finally:
                os.close(descriptor)
            try:
                self._lease = _OwnedFileLease(self.path, identity, parent_lease=parent_lease)
            except Exception:
                if identity is not None:
                    _remove_owned_file_path(self.path, identity, parent_lease=parent_lease)
                raise
            return
        fcntl: Any = importlib.import_module("fcntl")

        descriptor = parent_lease.open_file(name, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise RuntimeError(active_message) from error
        self._descriptor = descriptor

    def close(self) -> None:
        if self._lease is not None:
            lease, self._lease = self._lease, None
            lease.remove_owned_link()
            self.parent_lease.flush()
        if self._descriptor is not None:
            fcntl: Any = importlib.import_module("fcntl")

            descriptor, self._descriptor = self._descriptor, None
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


@dataclass(frozen=True)
class _OwnedChildProof:
    name: str
    identity: tuple[int, int]
    sha256: str
    size_bytes: int


class _OwnedDirectoryLease:
    """Hold a newly renamed directory so a replacement can never be removed."""

    def __init__(
        self,
        target: Path,
        expected_identity: tuple[int, int],
        *,
        parent_lease: _PublicationParentLease | None = None,
        mutable: bool = False,
        exclusive: bool = False,
    ) -> None:
        self.target = target
        self.identity = expected_identity
        self.parent_lease = parent_lease
        self.mutable = mutable
        self.exclusive = exclusive
        if exclusive and not mutable:
            raise ValueError("exclusive owned directory must be mutable")
        self._handle: int | None = None
        self._descriptor: int | None = None
        if parent_lease is not None:
            if target.parent != parent_lease.parent:
                raise ValueError("owned directory is not under its held publication parent")
            parent_lease.assert_confined()
        if os.name == "nt":
            handle = _win_open_native(
                target,
                access=_FILE_READ_ATTRIBUTES | ((_GENERIC_WRITE | _DELETE) if mutable else 0),
                share=_FILE_SHARE_READ | (_FILE_SHARE_WRITE if mutable and not exclusive else 0),
                directory=True,
            )
            self._handle = handle
            details = _win_info(handle)
            if (
                not details.dwFileAttributes & _FILE_ATTRIBUTE_DIRECTORY
                or details.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT
                or _win_identity(handle) != expected_identity
            ):
                self.close()
                raise ValueError("published experiment dataset identity changed after rename")
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        if parent_lease is None:
            descriptor = os.open(target, flags)
        else:
            descriptor = os.open(target.name, flags, dir_fd=parent_lease.descriptor)
        self._descriptor = descriptor
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (int(opened.st_dev), int(opened.st_ino)) != expected_identity
        ):
            self.close()
            raise ValueError("published experiment dataset identity changed after rename")

    @property
    def descriptor(self) -> int:
        if self._descriptor is None:
            raise RuntimeError("POSIX owned directory descriptor is unavailable")
        return self._descriptor

    def assert_identity(self) -> None:
        if self.parent_lease is not None:
            self.parent_lease.assert_confined()
        if os.name == "nt":
            if self._handle is None:
                raise RuntimeError("owned directory lease is closed")
            details = _win_info(self._handle)
            if (
                not details.dwFileAttributes & _FILE_ATTRIBUTE_DIRECTORY
                or details.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT
                or _win_identity(self._handle) != self.identity
            ):
                raise ValueError("owned directory identity changed")
            return
        details = os.fstat(self.descriptor)
        if (
            not stat.S_ISDIR(details.st_mode)
            or (int(details.st_dev), int(details.st_ino)) != self.identity
        ):
            raise ValueError("owned directory identity changed")
        if self.parent_lease is not None:
            visible = os.stat(
                self.target.name,
                dir_fd=self.parent_lease.descriptor,
                follow_symlinks=False,
            )
            if (int(visible.st_dev), int(visible.st_ino)) != self.identity or not stat.S_ISDIR(
                visible.st_mode
            ):
                raise ValueError("owned directory is no longer published at its name")

    def flush(self) -> None:
        self.assert_identity()
        if os.name == "nt":
            if not self.mutable:
                raise RuntimeError("sealed owned directory cannot be flushed directly")
            if self._handle is None:  # pragma: no cover - internal invariant
                raise RuntimeError("owned directory lease is closed")
            if not _FlushFileBuffers(self._handle):
                code = ctypes.get_last_error()
                if code not in _UNSUPPORTED_DIRECTORY_FLUSH_ERRORS:
                    raise OSError(code, "failed to flush held owned directory")
            return
        os.fsync(self.descriptor)

    def hold_known_files(
        self, expected_files: tuple[_OwnedChildProof, ...]
    ) -> tuple[_OwnedFileLease, ...]:
        self.assert_identity()
        ordered = tuple(sorted(expected_files, key=lambda item: item.name))
        if len({proof.name for proof in ordered}) != len(ordered):
            raise ValueError("owned child proof inventory contains duplicates")
        if os.name == "nt":
            names = tuple(sorted(entry.name for entry in self.target.iterdir()))
        else:
            names = tuple(sorted(os.listdir(self.descriptor)))
        if names != tuple(proof.name for proof in ordered):
            raise ValueError("owned child inventory changed")
        leases: list[_OwnedFileLease] = []
        try:
            for proof in ordered:
                child = _OwnedFileLease(
                    self.target / proof.name,
                    proof.identity,
                    deletable=False,
                    directory_lease=self,
                )
                leases.append(child)
            self.verify_known_files(tuple(leases), ordered)
            return tuple(leases)
        except Exception:
            for child in leases:
                child.close()
            raise

    def verify_known_files(
        self,
        leases: tuple[_OwnedFileLease, ...],
        expected_files: tuple[_OwnedChildProof, ...],
    ) -> None:
        self.assert_identity()
        ordered = tuple(sorted(expected_files, key=lambda item: item.name))
        if len(leases) != len(ordered):
            raise ValueError("owned child lease inventory is incomplete")
        if os.name == "nt":
            names = tuple(sorted(entry.name for entry in self.target.iterdir()))
        else:
            names = tuple(sorted(os.listdir(self.descriptor)))
        if names != tuple(proof.name for proof in ordered):
            raise ValueError("owned child inventory changed")
        by_name = {lease.target.name: lease for lease in leases}
        if len(by_name) != len(leases):
            raise ValueError("owned child lease inventory contains duplicates")
        for proof in ordered:
            lease = by_name.get(proof.name)
            if lease is None or lease.identity != proof.identity:
                raise ValueError("owned child lease identity changed")
            if os.name == "nt":
                visible_identity = _path_identity(self.target / proof.name)
            else:
                visible = os.stat(proof.name, dir_fd=self.descriptor, follow_symlinks=False)
                if not stat.S_ISREG(visible.st_mode):
                    raise ValueError("owned child is no longer a regular file")
                visible_identity = (int(visible.st_dev), int(visible.st_ino))
            if visible_identity != proof.identity:
                raise ValueError("owned child is no longer published at its name")
            payload = lease.read(max_bytes=max(1, proof.size_bytes))
            if len(payload) != proof.size_bytes or _sha256(payload) != proof.sha256:
                raise ValueError("owned child bytes changed")
        self.assert_identity()

    def rollback_known_files(self, expected_files: tuple[_OwnedChildProof, ...]) -> bool:
        if os.name == "nt" and not self.mutable:
            parent_lease = self.parent_lease
            target = self.target
            identity = self.identity
            self.close()
            try:
                cleanup = _OwnedDirectoryLease(
                    target,
                    identity,
                    parent_lease=parent_lease,
                    mutable=True,
                    exclusive=True,
                )
            except (OSError, ValueError):
                return False
            return cleanup.rollback_known_files(expected_files)
        child_leases: list[_OwnedFileLease] = []
        try:
            if os.name != "nt":
                # See _OwnedFileLease.remove_owned_link: preserving an orphan is
                # safer than a non-atomic identity-check/pathname-delete pair.
                return False
            self.assert_identity()
            if _path_identity(self.target) != self.identity:
                return False
            entries = tuple(sorted(self.target.iterdir(), key=lambda item: item.name))
            ordered_proofs = tuple(sorted(expected_files, key=lambda item: item.name))
            if len({proof.name for proof in ordered_proofs}) != len(ordered_proofs) or tuple(
                entry.name for entry in entries
            ) != tuple(proof.name for proof in ordered_proofs):
                return False
            for entry, proof in zip(entries, ordered_proofs, strict=True):
                details = entry.lstat()
                if (
                    not stat.S_ISREG(details.st_mode)
                    or stat.S_ISLNK(details.st_mode)
                    or bool(
                        getattr(details, "st_file_attributes", 0)
                        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                    )
                ):
                    return False
                try:
                    child = _OwnedFileLease(entry, proof.identity, directory_lease=self)
                    payload = child.read(max_bytes=max(1, proof.size_bytes))
                except (OSError, ValueError):
                    return False
                child_leases.append(child)
                if len(payload) != proof.size_bytes or _sha256(payload) != proof.sha256:
                    return False
            if _path_identity(self.target) != self.identity:
                return False
            for child in child_leases:
                if not child.remove_owned_link():
                    return False
            child_leases.clear()
            if os.name == "nt":
                if self._handle is None:  # pragma: no cover - internal invariant
                    return False
                disposition = _FILE_DISPOSITION_INFO(True)
                if not _SetFileInformationByHandle(
                    self._handle,
                    _FILE_DISPOSITION_INFO_CLASS,
                    ctypes.byref(disposition),
                    ctypes.sizeof(disposition),
                ):
                    raise OSError(
                        ctypes.get_last_error(),
                        "failed to remove owned experiment directory",
                    )
                self.close()
                return not self.target.exists() and not self.target.is_symlink()
            return False  # pragma: no cover - POSIX exits before child cleanup
        finally:
            for child in child_leases:
                child.close()
            self.close()

    def close(self) -> None:
        if self._handle is not None:
            handle, self._handle = self._handle, None
            _close_win_handle(handle)
        if self._descriptor is not None:
            descriptor, self._descriptor = self._descriptor, None
            os.close(descriptor)


def _remove_owned_file_path(
    path: Path,
    identity: tuple[int, int],
    *,
    parent_lease: _PublicationParentLease | None = None,
) -> bool:
    if not path.exists() and not path.is_symlink():
        return True
    try:
        lease = _OwnedFileLease(path, identity, parent_lease=parent_lease)
    except (OSError, ValueError):
        return False
    return lease.remove_owned_link()


def _verify_owned_evidence(lease: _OwnedFileLease, *, max_bytes: int) -> bytes:
    """Exact post-link reopen through the held immutable target handle."""

    return lease.read(max_bytes=max_bytes)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        handle = _win_open_native(
            path,
            access=_FILE_READ_ATTRIBUTES | _GENERIC_WRITE,
            share=_FILE_SHARE_READ | _FILE_SHARE_WRITE,
            directory=True,
        )
        try:
            details = _win_info(handle)
            if (
                not details.dwFileAttributes & _FILE_ATTRIBUTE_DIRECTORY
                or details.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT
            ):
                raise ValueError("directory durability handle is not a real directory")
            if not _FlushFileBuffers(handle):
                code = ctypes.get_last_error()
                if code not in _UNSUPPORTED_DIRECTORY_FLUSH_ERRORS:
                    raise OSError(code, "failed to flush directory metadata")
        finally:
            _close_win_handle(handle)
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    """Atomically publish a directory without ever replacing an existing target."""

    if source.parent != target.parent:
        raise ValueError("dataset publication must remain within one held parent")
    if os.name == "nt":  # Windows rename fails if the destination already exists.
        try:
            os.rename(source, target)
        except FileExistsError as error:
            raise FileExistsError("conflicting experiment dataset already exists") from error
        return
    if sys.platform.startswith("linux"):
        parent_lease = _PublicationParentLease(source.parent, source.parent, create=False)
        try:
            library = ctypes.CDLL(None, use_errno=True)
            renameat2 = getattr(library, "renameat2", None)
            if renameat2 is None:
                raise RuntimeError("atomic no-clobber directory publication is unavailable")
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                parent_lease.descriptor,
                os.fsencode(source.name),
                parent_lease.descriptor,
                os.fsencode(target.name),
                1,
            )
            if result != 0:
                code = ctypes.get_errno()
                if code in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise FileExistsError("conflicting experiment dataset already exists")
                raise OSError(code, os.strerror(code), target)
            return
        finally:
            parent_lease.close()
    raise RuntimeError("atomic no-clobber directory publication is unavailable")


def _requirement_by_id(requirement_id: str) -> FormalCorpusRequirement:
    try:
        return next(
            item for item in FORMAL_CORPUS_REQUIREMENTS if item.requirement_id == requirement_id
        )
    except StopIteration as error:
        raise ValueError("unknown formal corpus requirement") from error


def _publish_experiment_dataset(
    *,
    root: Path,
    requirement: FormalCorpusRequirement,
    bundle: CoreGraphBundle,
    labels: dict[str, dict[str, ScalarLabel]],
    split_manifests: tuple[SplitManifest, ...],
    phase_eligibility: Literal["smoke", "dev", "formal"],
    materializer_id: str,
    materializer_version: str,
    materializer_code_sha256: str,
    materialization_protocol_hash: str,
) -> ExperimentDatasetManifest:
    """Publish already validated experiment artifacts without replacing any target."""

    requirement_id = requirement.requirement_id
    recipe = load_dataset_recipes()[requirement.recipe_id]
    prefix = f"experiment-corpus/{requirement_id}"
    bundle_relative_path = f"{prefix}/bundle.json"
    labels_relative_path = f"{prefix}/labels.json"
    split_inventory_relative_path = f"{prefix}/split-inventory.json"
    label_document = _labels_document(requirement_id, labels)
    split_inventory = _split_inventory_document(requirement, split_manifests)
    if split_inventory.folds[0].split_manifest != bundle.split_manifest:
        raise ValueError("the bundle split must match the first split inventory fold")
    bundle_bytes = _canonical_bytes(bundle)
    label_bytes = _canonical_bytes(label_document)
    split_inventory_bytes = _canonical_bytes(split_inventory)
    split_hash = canonical_sha256(bundle.split_manifest.model_dump(mode="python", by_alias=True))
    manifest_payload: dict[str, Any] = {
        "schemaVersion": _MANIFEST_SCHEMA,
        "requirementId": requirement_id,
        "recipeId": requirement.recipe_id,
        "recipeVersion": recipe.recipe_version,
        "recipeSha256": recipe.recipe_sha256,
        "graphId": requirement.graph_id,
        "phaseEligibility": phase_eligibility,
        "usageScope": requirement.required_usage_scope,
        "splitPolicy": requirement.expected_split_policy,
        "experimentSplitPolicy": requirement.experiment_split_policy,
        "materializerId": materializer_id,
        "materializerVersion": materializer_version,
        "materializerCodeSha256": materializer_code_sha256,
        "materializationProtocolHash": materialization_protocol_hash,
        "manifestRelativePath": requirement.manifest_relative_path,
        "bundleRelativePath": bundle_relative_path,
        "bundleSha256": _sha256(bundle_bytes),
        "labelsRelativePath": labels_relative_path,
        "labelsSha256": _sha256(label_bytes),
        "labelNames": sorted(labels),
        "splitInventoryRelativePath": split_inventory_relative_path,
        "splitInventorySha256": _sha256(split_inventory_bytes),
        "splitCount": len(split_inventory.folds),
        "splitIds": [fold.fold_id for fold in split_inventory.folds],
        "splitManifestHashes": [fold.split_manifest_hash for fold in split_inventory.folds],
        "graphVersionHash": bundle.graph_version_hash,
        "sourceSha256": bundle.source.source_sha256,
        "splitManifestHash": split_hash,
    }
    manifest_payload["manifestHash"] = canonical_sha256(manifest_payload)
    manifest = ExperimentDatasetManifest.model_validate(manifest_payload)
    manifest_bytes = _canonical_bytes(manifest)
    for name, content, maximum in (
        ("bundle", bundle_bytes, _MAX_BUNDLE_BYTES),
        ("labels", label_bytes, _MAX_LABEL_BYTES),
        ("split inventory", split_inventory_bytes, _MAX_SPLIT_INVENTORY_BYTES),
        ("manifest", manifest_bytes, _MAX_MANIFEST_BYTES),
    ):
        if len(content) > maximum:
            raise ValueError(f"{name} artifact exceeds its publication size limit")

    corpus_parent_lease = _PublicationParentLease(root, root / "experiment-corpus", create=True)
    corpus_parent = corpus_parent_lease.parent
    target = corpus_parent / requirement_id
    try:
        lock = _PublisherLock(
            corpus_parent_lease,
            f".{requirement_id}.publisher.lock",
            active_message="experiment dataset already has an active publisher",
        )
    except Exception:
        corpus_parent_lease.close()
        raise
    staging = corpus_parent / f".{requirement_id}.{uuid.uuid4().hex}.staging"
    staging_identity: tuple[int, int] | None = None
    staged_proofs: list[_OwnedChildProof] = []
    published_by_this_call = False
    published_lease: _OwnedDirectoryLease | None = None
    staging_lease: _OwnedDirectoryLease | None = None
    held_children: tuple[_OwnedFileLease, ...] = ()
    try:
        corpus_parent_lease.assert_confined()
        if target.exists() or target.is_symlink():
            expected = {
                "bundle.json": bundle_bytes,
                "labels.json": label_bytes,
                "split-inventory.json": split_inventory_bytes,
                "dataset-manifest.json": manifest_bytes,
            }
            existing_lease: _OwnedDirectoryLease | None = None
            existing_children: tuple[_OwnedFileLease, ...] = ()
            try:
                existing_lease = _OwnedDirectoryLease(
                    target,
                    _path_identity(target),
                    parent_lease=corpus_parent_lease,
                )
                existing_proofs = tuple(
                    _OwnedChildProof(
                        name=name,
                        identity=_path_identity(target / name),
                        sha256=_sha256(content),
                        size_bytes=len(content),
                    )
                    for name, content in sorted(expected.items())
                )
                existing_children = existing_lease.hold_known_files(existing_proofs)
                replay = _observe_requirement(root, requirement)
                expected_status = "ready" if phase_eligibility == "formal" else "usage-ineligible"
                if replay.status != expected_status:
                    raise ValueError("existing dataset failed semantic revalidation")
                existing_lease.verify_known_files(existing_children, existing_proofs)
            except Exception as error:
                raise FileExistsError("conflicting experiment dataset already exists") from error
            finally:
                for child in existing_children:
                    child.close()
                if existing_lease is not None:
                    existing_lease.close()
            return manifest
        corpus_parent_lease.make_directory(staging.name)
        staging_identity = _path_identity(staging)
        staging_lease = _OwnedDirectoryLease(
            staging,
            staging_identity,
            parent_lease=corpus_parent_lease,
            mutable=True,
        )
        for name, content in (
            ("bundle.json", bundle_bytes),
            ("labels.json", label_bytes),
            ("split-inventory.json", split_inventory_bytes),
            ("dataset-manifest.json", manifest_bytes),
        ):
            destination = staging / name
            if os.name == "nt":
                descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            else:
                descriptor = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=staging_lease.descriptor,
                )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            staged_proofs.append(
                _OwnedChildProof(
                    name=name,
                    identity=_path_identity(destination),
                    sha256=_sha256(content),
                    size_bytes=len(content),
                )
            )
        staging_lease.flush()
        # Staging is fully parseable before it becomes visible.  The final rename is
        # no-clobber, and the manifest is exact-reopened after publication.
        if (
            load_core_graph_bundle_json((staging / "bundle.json").read_bytes()) != bundle
            or ExperimentLabels.model_validate_json((staging / "labels.json").read_bytes())
            != label_document
            or ExperimentSplitInventory.model_validate_json(
                (staging / "split-inventory.json").read_bytes()
            )
            != split_inventory
            or ExperimentDatasetManifest.model_validate_json(
                (staging / "dataset-manifest.json").read_bytes()
            )
            != manifest
        ):
            raise ValueError("staged experiment dataset failed exact validation")
        if _path_identity(staging) != staging_identity:
            raise ValueError("staging directory identity changed after validation")
        staging_lease.assert_identity()
        _PUBLICATION_SEAM("dataset", target)
        corpus_parent_lease.assert_confined()
        staging_lease.assert_identity()
        staging_lease.close()
        staging_lease = None
        _rename_directory_no_replace(staging, target)
        published_by_this_call = True
        published_lease = _OwnedDirectoryLease(
            target, staging_identity, parent_lease=corpus_parent_lease
        )
        held_children = published_lease.hold_known_files(tuple(staged_proofs))
        _PUBLICATION_SEAM("dataset-post-rename", target)
        published_lease.verify_known_files(held_children, tuple(staged_proofs))
        corpus_parent_lease.flush()
        observation = _observe_requirement(root, requirement)
        expected_status = "ready" if phase_eligibility == "formal" else "usage-ineligible"
        if observation.status != expected_status:
            raise ValueError("published experiment dataset failed exact reload")
        published_lease.verify_known_files(held_children, tuple(staged_proofs))
        for child in held_children:
            child.close()
        held_children = ()
        published_lease.close()
        return manifest
    except Exception as error:
        for child in held_children:
            child.close()
        held_children = ()
        if published_by_this_call and published_lease is not None:
            try:
                removed = published_lease.rollback_known_files(tuple(staged_proofs))
                if removed:
                    corpus_parent_lease.flush()
                else:
                    error.add_note(
                        "owned experiment directory was preserved because its identity "
                        "or exact file inventory could not be proven"
                    )
            except Exception as cleanup_error:
                error.add_note(f"owned experiment rollback failed closed: {cleanup_error}")
        raise
    finally:
        try:
            for child in held_children:
                child.close()
            if staging_lease is not None:
                staging_lease.close()
            if published_lease is not None:
                published_lease.close()
            if staging_identity is not None and (staging.exists() or staging.is_symlink()):
                try:
                    cleanup_lease = _OwnedDirectoryLease(
                        staging,
                        staging_identity,
                        parent_lease=corpus_parent_lease,
                        mutable=True,
                        exclusive=True,
                    )
                except (OSError, ValueError):
                    pass
                else:
                    if cleanup_lease.rollback_known_files(tuple(staged_proofs)):
                        corpus_parent_lease.flush()
        finally:
            try:
                lock.close()
            finally:
                corpus_parent_lease.close()


def publish_experiment_dataset(
    *,
    runtime_root: Path,
    requirement_id: str,
    bundle: CoreGraphBundle,
    labels: dict[str, dict[str, ScalarLabel]],
    phase_eligibility: Literal["smoke", "dev", "formal"],
) -> ExperimentDatasetManifest:
    """Publish smoke/dev input; formal publication is converter-owned."""

    if phase_eligibility == "formal":
        raise ValueError(
            "formal publication requires the registered dataset-specific formal materializer"
        )
    root = secure_existing_root(runtime_root)
    requirement = _requirement_by_id(requirement_id)
    identity = {
        "materializerId": "socialgraph-fm.unverified-smoke-dev-input",
        "materializerVersion": "1.0",
    }
    code_hash = canonical_sha256(identity)
    return _publish_experiment_dataset(
        root=root,
        requirement=requirement,
        bundle=bundle,
        labels=labels,
        split_manifests=(bundle.split_manifest,),
        phase_eligibility=phase_eligibility,
        materializer_id=identity["materializerId"],
        materializer_version=identity["materializerVersion"],
        materializer_code_sha256=code_hash,
        materialization_protocol_hash=canonical_sha256(
            {**identity, "eligibility": "smoke-dev-only"}
        ),
    )


def materialize_formal_experiment_dataset(
    *, runtime_root: Path, requirement_id: str
) -> ExperimentDatasetManifest:
    """Reparse locked raw bytes and atomically publish the exact formal artifacts."""

    root = secure_existing_root(runtime_root)
    requirement = _requirement_by_id(requirement_id)
    recipe = load_dataset_recipes()[requirement.recipe_id]
    binding = formal_materializer_binding(
        requirement_id=requirement.requirement_id,
        recipe=recipe,
        graph_id=requirement.graph_id,
        bundle_split_policy=requirement.expected_split_policy,
        experiment_split_policy=requirement.experiment_split_policy,
    )
    if binding is None:
        raise ValueError("no dataset-specific formal materializer is registered")
    raw_hashes, raw_snapshots = _formal_source_hashes(root, requirement)
    combined_source_hash = canonical_sha256(raw_hashes)
    derived = derive_registered_formal_dataset(
        requirement_id=requirement.requirement_id,
        recipe=recipe,
        graph_id=requirement.graph_id,
        raw_sources=raw_snapshots,
        combined_source_sha256=combined_source_hash,
        bundle_split_policy=requirement.expected_split_policy,
    )
    labels = _labels_document(requirement.requirement_id, derived.labels)
    split_inventory = _split_inventory_document(requirement, derived.split_manifests)
    _validate_formal_semantics(requirement, derived.bundle, labels, split_inventory)
    return _publish_experiment_dataset(
        root=root,
        requirement=requirement,
        bundle=derived.bundle,
        labels=derived.labels,
        split_manifests=derived.split_manifests,
        phase_eligibility="formal",
        materializer_id=binding.materializer_id,
        materializer_version=binding.materializer_version,
        materializer_code_sha256=binding.materializer_code_sha256,
        materialization_protocol_hash=binding.materialization_protocol_hash,
    )


__all__ = [
    "ExperimentDatasetManifest",
    "ExperimentLabels",
    "FORMAL_CORPUS_REQUIREMENTS",
    "FormalCorpusObservation",
    "FormalCorpusRequirement",
    "FormalPreflightEvidence",
    "LabelTarget",
    "LabelValue",
    "load_formal_preflight",
    "materialize_formal_experiment_dataset",
    "publish_experiment_dataset",
    "run_formal_preflight",
]
