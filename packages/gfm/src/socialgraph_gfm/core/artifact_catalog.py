"""Operator-owned catalog and exact CoreGraphBundle materialization."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from socialgraph_gfm.canonical import canonical_sha256

from .bundle import (
    CategoricalFeature,
    MultiHotFeature,
    NumericFeature,
    CoreGraphBundle,
    load_core_graph_bundle_json,
)
from .inference_contracts import AuthorizedGraphReference
from .safe_paths import read_confined_snapshot, reject_link_components, secure_existing_root


_HASH = r"^[0-9a-f]{64}$"
MAX_BUNDLE_BYTES = 512 * 1024 * 1024


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True, frozen=True)


class FeatureField(_StrictModel):
    kind: Literal["numeric", "categorical", "multiHot"]
    name: str = Field(min_length=1, max_length=200)


class FeatureContract(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-graph-feature-contract/2.0"] = Field(
        alias="schemaVersion"
    )
    node_features: tuple[FeatureField, ...] = Field(alias="nodeFeatures", strict=False)
    structural_feature_names: tuple[str, ...] = Field(
        alias="structuralFeatureNames", strict=False
    )

    @model_validator(mode="after")
    def validate_unique_names(self):
        names = [field.name for field in self.node_features]
        if len(names) != len(set(names)) or len(self.structural_feature_names) != len(
            set(self.structural_feature_names)
        ):
            raise ValueError("feature contract names must be unique")
        return self


class ArtifactEntry(_StrictModel):
    artifact_id: str = Field(alias="artifactId", min_length=1, max_length=300)
    artifact_hash: str = Field(alias="artifactHash", pattern=_HASH)
    bundle_sha256: str = Field(alias="bundleSha256", pattern=_HASH)
    relative_path: str = Field(alias="relativePath", min_length=1, max_length=500)
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    source_graph_fact_hash: str = Field(alias="sourceGraphFactHash", pattern=_HASH)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH)
    graph_schema_version: Literal["socialgraph-fm.core-graph-bundle/2.0"] = Field(
        alias="graphSchemaVersion"
    )
    feature_contract: FeatureContract = Field(alias="featureContract")
    feature_contract_hash: str = Field(alias="featureContractHash", pattern=_HASH)
    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        parsed = PurePosixPath(value.replace("\\", "/"))
        if parsed.is_absolute() or ".." in parsed.parts or ":" in value:
            raise ValueError("artifact path must be a safe relative path")
        return parsed.as_posix()

    @model_validator(mode="after")
    def validate_feature_hash(self):
        observed = canonical_sha256(
            self.feature_contract.model_dump(mode="python", by_alias=True)
        )
        if self.feature_contract_hash != observed:
            raise ValueError("feature contract hash mismatch")
        return self


class ArtifactCatalogDocument(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-serving-graph-catalog/1.0"] = Field(
        alias="schemaVersion"
    )
    generation: int = Field(ge=0)
    artifacts: tuple[ArtifactEntry, ...] = Field(strict=False)

    @model_validator(mode="after")
    def validate_unique_artifacts(self):
        identifiers = [entry.artifact_id for entry in self.artifacts]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("artifactId values must be unique")
        return self


@dataclass(frozen=True)
class CapturedGraphLease:
    catalog_snapshot: bytes
    catalog_sha256: str
    catalog_generation: int
    reference: AuthorizedGraphReference
    bundle_snapshot: bytes

    def materialize(self) -> CoreGraphBundle:
        if hashlib.sha256(self.catalog_snapshot).hexdigest() != self.catalog_sha256:
            raise ValueError("captured artifact catalog bytes changed")
        document = ArtifactCatalogDocument.model_validate_json(self.catalog_snapshot)
        if document.generation != self.catalog_generation:
            raise ValueError("captured artifact catalog generation mismatch")
        entry = next(
            (
                item
                for item in document.artifacts
                if item.artifact_id == self.reference.artifact_id
            ),
            None,
        )
        if entry is None:
            raise LookupError("artifact is not present in the captured catalog")
        return _materialize_entry(entry, self.reference, self.bundle_snapshot)


def feature_contract_for_bundle(bundle: CoreGraphBundle) -> FeatureContract:
    fields: list[dict[str, str]] = []
    for feature in bundle.node_features:
        if isinstance(feature, NumericFeature):
            kind = "numeric"
        elif isinstance(feature, CategoricalFeature):
            kind = "categorical"
        elif isinstance(feature, MultiHotFeature):
            kind = "multiHot"
        else:  # pragma: no cover - the bundle union is closed
            raise TypeError("unsupported feature kind")
        fields.append({"kind": kind, "name": feature.name})
    return FeatureContract.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-graph-feature-contract/2.0",
            "nodeFeatures": fields,
            "structuralFeatureNames": (
                list(bundle.structural_features.names)
                if bundle.structural_features is not None
                else []
            ),
        }
    )


def _materialize_entry(
    entry: ArtifactEntry,
    reference: AuthorizedGraphReference,
    payload: bytes,
) -> CoreGraphBundle:
    expected = (
        entry.artifact_hash,
        entry.bundle_sha256,
        entry.graph_version_id,
        entry.source_graph_fact_hash,
        entry.graph_version_hash,
        entry.graph_schema_version,
        entry.feature_contract_hash,
        entry.node_count,
        entry.edge_count,
    )
    supplied = (
        reference.artifact_hash,
        reference.bundle_sha256,
        reference.graph_version_id,
        reference.source_graph_fact_hash,
        reference.graph_version_hash,
        reference.graph_schema_version,
        reference.feature_contract_hash,
        reference.node_count,
        reference.edge_count,
    )
    if supplied != expected:
        raise ValueError("authorized graph reference does not match artifact catalog")
    observed_hash = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(observed_hash, entry.bundle_sha256):
        raise ValueError("artifact bundle hash does not match catalog")
    bundle = load_core_graph_bundle_json(payload)
    observed_contract = feature_contract_for_bundle(bundle)
    if (
        bundle.schema_version != entry.graph_schema_version
        or bundle.graph_version_hash != entry.graph_version_hash
        or len(bundle.nodes) != entry.node_count
        or len(bundle.edges) != entry.edge_count
        or observed_contract != entry.feature_contract
    ):
        raise ValueError("materialized bundle does not match artifact catalog bindings")
    return bundle


class ArtifactCatalog:
    def __init__(self, path: Path, artifact_root: Path, document: ArtifactCatalogDocument) -> None:
        self.path = path
        self.artifact_root = artifact_root
        self.document = document

    @classmethod
    def load(cls, path: str | Path, *, artifact_root: str | Path) -> ArtifactCatalog:
        root = secure_existing_root(artifact_root)
        catalog_path = reject_link_components(path)
        if not catalog_path.is_file():
            raise ValueError("artifact catalog must be an existing regular file")
        catalog_parent = secure_existing_root(catalog_path.parent)
        snapshot = read_confined_snapshot(
            catalog_parent,
            catalog_path.name,
            max_bytes=16 * 1024 * 1024,
        )
        document = ArtifactCatalogDocument.model_validate_json(snapshot)
        return cls(catalog_parent / catalog_path.name, root, document)

    def entry(self, artifact_id: str) -> ArtifactEntry:
        entry = next(
            (item for item in self.document.artifacts if item.artifact_id == artifact_id), None
        )
        if entry is None:
            raise LookupError("artifact is not present in the authorized catalog")
        return entry

    def _catalog_snapshot(self) -> bytes:
        return read_confined_snapshot(
            self.path.parent, self.path.name, max_bytes=16 * 1024 * 1024
        )

    def acquire_graph_lease(
        self,
        reference: AuthorizedGraphReference,
        *,
        catalog_snapshot: bytes | None = None,
    ) -> CapturedGraphLease:
        for _attempt in range(1 if catalog_snapshot is not None else 3):
            control = (
                catalog_snapshot
                if catalog_snapshot is not None
                else self._catalog_snapshot()
            )
            control_hash = hashlib.sha256(control).hexdigest()
            document = ArtifactCatalogDocument.model_validate_json(control)
            entry = next(
                (item for item in document.artifacts if item.artifact_id == reference.artifact_id),
                None,
            )
            if entry is None:
                raise LookupError("artifact is not present in the authorized catalog")
            payload = read_confined_snapshot(
                self.artifact_root,
                entry.relative_path,
                max_bytes=MAX_BUNDLE_BYTES,
            )
            if (
                catalog_snapshot is None
                and hashlib.sha256(self._catalog_snapshot()).hexdigest() != control_hash
            ):
                continue
            lease = CapturedGraphLease(
                catalog_snapshot=control,
                catalog_sha256=control_hash,
                catalog_generation=document.generation,
                reference=reference,
                bundle_snapshot=payload,
            )
            lease.materialize()
            return lease
        raise ValueError("artifact catalog changed during bounded graph lease acquisition")

    def control_matches(self, lease: CapturedGraphLease) -> bool:
        return hmac.compare_digest(
            hashlib.sha256(self._catalog_snapshot()).hexdigest(), lease.catalog_sha256
        )

    def resolve(self, reference: AuthorizedGraphReference) -> CoreGraphBundle:
        return self.acquire_graph_lease(reference).materialize()


__all__ = [
    "ArtifactCatalog",
    "ArtifactCatalogDocument",
    "ArtifactEntry",
    "CapturedGraphLease",
    "FeatureContract",
    "feature_contract_for_bundle",
]
