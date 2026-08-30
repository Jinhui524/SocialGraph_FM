"""Versioned contracts for the safe SocialGraph-FM Global corpus boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.gfm.corpus.common import atomic_write_json, read_json_object

SHA256_PATTERN = r"^[0-9a-f]{64}$"
CountryId = Literal["china", "cuba", "iran", "russia", "UAE", "venezuela"]
COUNTRY_IDS: tuple[CountryId, ...] = (
    "china",
    "cuba",
    "iran",
    "russia",
    "UAE",
    "venezuela",
)
TRACE_NAMES = ("coRT", "coURL", "hashSeq", "fastRT", "tweetSim")
TRACE_ARRAY_TOKENS = {
    "coRT": "cort",
    "coURL": "courl",
    "hashSeq": "hashseq",
    "fastRT": "fastrt",
    "tweetSim": "tweetsim",
}
GRAPH_STAT_NAMES = (
    "log_nodes",
    "density",
    "component_ratio",
    "isolate_rate",
    "log_degree_p25",
    "log_degree_p50",
    "log_degree_p90",
    "degree_entropy",
    "coRT_edge_proportion",
    "coURL_edge_proportion",
    "hashSeq_edge_proportion",
    "fastRT_edge_proportion",
    "tweetSim_edge_proportion",
)
REQUIRED_ARRAY_NAMES = frozenset(
    {
        "edge_index",
        "text_features",
        "degree_bucket",
        "structure_missing",
        "graph_stats",
        "labels",
        "trace_membership",
        "fused_indptr",
        "fused_indices",
        *(
            f"relation_{token}_{suffix}"
            for token in TRACE_ARRAY_TOKENS.values()
            for suffix in ("indptr", "indices", "weights")
        ),
    }
)


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        allow_inf_nan=False,
    )


def _logical_payload(model: BaseModel, hash_field: str) -> dict[str, Any]:
    return model.model_dump(mode="python", by_alias=True, exclude={hash_field})


def _validate_relative_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value or ":" in value:
        raise ValueError("artifact path must be a portable relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact path must not traverse its root")
    if path.as_posix() != value:
        raise ValueError("artifact path must use canonical POSIX spelling")
    return value


class GlobalArrayDescriptor(_Contract):
    """One immutable, non-pickle NumPy artifact."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    path: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=SHA256_PATTERN)
    dtype: str = Field(min_length=2, max_length=32)
    shape: tuple[int, ...] = Field(min_length=1, max_length=3)
    byte_length: int = Field(alias="byteLength", ge=1)

    _portable_path = field_validator("path")(_validate_relative_path)

    @model_validator(mode="after")
    def validate_shape(self) -> GlobalArrayDescriptor:
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("array dimensions must be non-negative")
        return self


class GlobalSplitDescriptor(_Contract):
    """One deterministic train/validation/test node-mask assignment."""

    split_id: str = Field(alias="splitId", pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
    regime: str = Field(pattern=r"^(full|0\.(5|75|9|95|99|999))$")
    fold: int = Field(ge=0, le=99)
    train_array: str = Field(alias="trainArray", pattern=r"^[a-z][a-z0-9_]{0,127}$")
    validation_array: str = Field(
        alias="validationArray", pattern=r"^[a-z][a-z0-9_]{0,127}$"
    )
    test_array: str = Field(alias="testArray", pattern=r"^[a-z][a-z0-9_]{0,127}$")
    split_hash: str = Field(alias="splitHash", pattern=SHA256_PATTERN)

    @classmethod
    def create(
        cls,
        *,
        split_id: str,
        regime: str,
        fold: int,
        train_array: str,
        validation_array: str,
        test_array: str,
    ) -> GlobalSplitDescriptor:
        payload = {
            "splitId": split_id,
            "regime": regime,
            "fold": fold,
            "trainArray": train_array,
            "validationArray": validation_array,
            "testArray": test_array,
        }
        return cls.model_validate({**payload, "splitHash": canonical_sha256(payload)})

    @model_validator(mode="after")
    def validate_split(self) -> GlobalSplitDescriptor:
        if len({self.train_array, self.validation_array, self.test_array}) != 3:
            raise ValueError("split role arrays must be distinct")
        if self.split_hash != canonical_sha256(_logical_payload(self, "split_hash")):
            raise ValueError("splitHash does not match the split payload")
        return self


def _bound_split_hash(
    split: GlobalSplitDescriptor,
    arrays: Mapping[str, GlobalArrayDescriptor],
) -> str:
    return canonical_sha256(
        {
            "descriptorHash": split.split_hash,
            "train": {
                "name": split.train_array,
                "sha256": arrays[split.train_array].sha256,
            },
            "validation": {
                "name": split.validation_array,
                "sha256": arrays[split.validation_array].sha256,
            },
            "test": {
                "name": split.test_array,
                "sha256": arrays[split.test_array].sha256,
            },
        }
    )


class GlobalCountryManifest(_Contract):
    """Complete content identity for one converted Global country."""

    schema_version: Literal["socialgraph-fm.global-model-country/1.0"] = Field(
        "socialgraph-fm.global-model-country/1.0", alias="schemaVersion"
    )
    corpus_id: Literal["socialgraph-fm"] = Field("socialgraph-fm", alias="corpusId")
    country_id: CountryId = Field(alias="countryId")
    node_count: int = Field(alias="nodeCount", ge=1)
    edge_count: int = Field(alias="edgeCount", ge=0)
    trace_names: tuple[str, ...] = Field(TRACE_NAMES, alias="traceNames")
    text_feature_dim: Literal[768] = Field(768, alias="textFeatureDim")
    structural_buckets: Literal[128] = Field(128, alias="structuralBuckets")
    arrays: tuple[GlobalArrayDescriptor, ...] = Field(min_length=1)
    splits: tuple[GlobalSplitDescriptor, ...] = Field(min_length=1)
    source_hashes: dict[str, str] = Field(alias="sourceHashes", min_length=2)
    split_hashes: dict[str, str] = Field(alias="splitHashes", min_length=1)
    relation_edge_counts: dict[str, int] = Field(alias="relationEdgeCounts")
    preprocessing: dict[str, str] = Field(min_length=1)
    content_hash: str = Field(alias="contentHash", pattern=SHA256_PATTERN)

    @classmethod
    def create(
        cls,
        *,
        country_id: CountryId,
        node_count: int,
        edge_count: int,
        arrays: Sequence[GlobalArrayDescriptor],
        splits: Sequence[GlobalSplitDescriptor],
        source_hashes: Mapping[str, str],
        relation_edge_counts: Mapping[str, int],
        preprocessing: Mapping[str, str],
    ) -> GlobalCountryManifest:
        array_map = {item.name: item for item in arrays}
        split_hashes = {
            split.split_id: _bound_split_hash(split, array_map) for split in splits
        }
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.global-model-country/1.0",
            "corpusId": "socialgraph-fm",
            "countryId": country_id,
            "nodeCount": node_count,
            "edgeCount": edge_count,
            "traceNames": TRACE_NAMES,
            "textFeatureDim": 768,
            "structuralBuckets": 128,
            "arrays": [item.model_dump(mode="python", by_alias=True) for item in arrays],
            "splits": [item.model_dump(mode="python", by_alias=True) for item in splits],
            "sourceHashes": dict(sorted(source_hashes.items())),
            "splitHashes": dict(sorted(split_hashes.items())),
            "relationEdgeCounts": {
                name: int(relation_edge_counts[name]) for name in TRACE_NAMES
            },
            "preprocessing": dict(sorted(preprocessing.items())),
        }
        return cls.model_validate({**payload, "contentHash": canonical_sha256(payload)})

    @model_validator(mode="after")
    def validate_manifest(self) -> GlobalCountryManifest:
        if self.trace_names != TRACE_NAMES:
            raise ValueError("traceNames must use the fixed Global ordering")
        names = [item.name for item in self.arrays]
        paths = [item.path for item in self.arrays]
        if len(names) != len(set(names)) or len(paths) != len(set(paths)):
            raise ValueError("array names and paths must be unique")
        if not REQUIRED_ARRAY_NAMES.issubset(names):
            missing = sorted(REQUIRED_ARRAY_NAMES - set(names))
            raise ValueError(f"country manifest is missing required arrays: {missing}")
        split_ids = [split.split_id for split in self.splits]
        if len(split_ids) != len(set(split_ids)):
            raise ValueError("split IDs must be unique")
        available = set(names)
        for split in self.splits:
            if not {split.train_array, split.validation_array, split.test_array}.issubset(available):
                raise ValueError(f"split {split.split_id!r} references an undeclared mask")
        array_map = {item.name: item for item in self.arrays}
        expected_split_hashes = {
            split.split_id: _bound_split_hash(split, array_map) for split in self.splits
        }
        if self.split_hashes != expected_split_hashes:
            raise ValueError("splitHashes must exactly bind every declared split")
        if set(self.relation_edge_counts) != set(TRACE_NAMES) or any(
            count < 0 for count in self.relation_edge_counts.values()
        ):
            raise ValueError("relationEdgeCounts must bind all five traces to non-negative counts")
        for label, hashes in (("sourceHashes", self.source_hashes), ("splitHashes", self.split_hashes)):
            if any(not key or not re.fullmatch(SHA256_PATTERN, value) for key, value in hashes.items()):
                raise ValueError(f"{label} contains an invalid name or SHA-256 digest")
        expected_shapes = {
            "edge_index": (2, self.edge_count),
            "text_features": (self.node_count, 768),
            "degree_bucket": (self.node_count,),
            "structure_missing": (self.node_count,),
            "graph_stats": (len(GRAPH_STAT_NAMES),),
            "labels": (self.node_count,),
            "trace_membership": (self.node_count, len(TRACE_NAMES)),
            "fused_indptr": (self.node_count + 1,),
            "fused_indices": (self.edge_count,),
        }
        for trace_name, token in TRACE_ARRAY_TOKENS.items():
            relation_count = self.relation_edge_counts[trace_name]
            expected_shapes[f"relation_{token}_indptr"] = (self.node_count + 1,)
            expected_shapes[f"relation_{token}_indices"] = (relation_count,)
            expected_shapes[f"relation_{token}_weights"] = (relation_count,)
        for name, shape in expected_shapes.items():
            if array_map[name].shape != shape:
                raise ValueError(f"array {name!r} has shape {array_map[name].shape}, expected {shape}")
        if self.content_hash != canonical_sha256(_logical_payload(self, "content_hash")):
            raise ValueError("contentHash does not match the country manifest payload")
        return self

    def array(self, name: str) -> GlobalArrayDescriptor:
        try:
            return next(item for item in self.arrays if item.name == name)
        except StopIteration as exc:
            raise KeyError(name) from exc


class GlobalCorpusEntry(_Contract):
    country_id: CountryId = Field(alias="countryId")
    manifest_path: str = Field(alias="manifestPath", max_length=512)
    manifest_hash: str = Field(alias="manifestHash", pattern=SHA256_PATTERN)
    source_hashes: dict[str, str] = Field(alias="sourceHashes", min_length=2)
    split_hashes: dict[str, str] = Field(alias="splitHashes", min_length=1)

    _portable_manifest_path = field_validator("manifest_path")(_validate_relative_path)

    @classmethod
    def from_country_manifest(
        cls,
        manifest: GlobalCountryManifest,
        *,
        manifest_path: str,
    ) -> GlobalCorpusEntry:
        return cls(
            countryId=manifest.country_id,
            manifestPath=manifest_path,
            manifestHash=manifest.content_hash,
            sourceHashes=manifest.source_hashes,
            splitHashes=manifest.split_hashes,
        )


class GlobalCorpusManifest(_Contract):
    schema_version: Literal["socialgraph-fm.global-model-corpus/1.0"] = Field(
        "socialgraph-fm.global-model-corpus/1.0", alias="schemaVersion"
    )
    corpus_id: Literal["socialgraph-fm"] = Field("socialgraph-fm", alias="corpusId")
    license_id: Literal["CC-BY-4.0"] = Field("CC-BY-4.0", alias="licenseId")
    source_uri: str = Field(alias="sourceUri", min_length=1, max_length=2048)
    countries: tuple[GlobalCorpusEntry, ...] = Field(min_length=6, max_length=6)
    content_hash: str = Field(alias="contentHash", pattern=SHA256_PATTERN)

    SOURCE_URI: ClassVar[str] = "https://zenodo.org/records/13357621"

    @classmethod
    def create(
        cls,
        countries: Sequence[GlobalCorpusEntry],
        *,
        source_uri: str = SOURCE_URI,
    ) -> GlobalCorpusManifest:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.global-model-corpus/1.0",
            "corpusId": "socialgraph-fm",
            "licenseId": "CC-BY-4.0",
            "sourceUri": source_uri,
            "countries": [item.model_dump(mode="python", by_alias=True) for item in countries],
        }
        return cls.model_validate({**payload, "contentHash": canonical_sha256(payload)})

    @model_validator(mode="after")
    def validate_corpus(self) -> GlobalCorpusManifest:
        observed = tuple(entry.country_id for entry in self.countries)
        if observed != COUNTRY_IDS:
            raise ValueError(f"countries must use the fixed order {COUNTRY_IDS!r}")
        if len({entry.manifest_path for entry in self.countries}) != len(COUNTRY_IDS):
            raise ValueError("country manifest paths must be unique")
        if self.content_hash != canonical_sha256(_logical_payload(self, "content_hash")):
            raise ValueError("contentHash does not match the corpus manifest payload")
        return self


def atomic_write_contract(path: Path, contract: BaseModel) -> None:
    """Durably replace a JSON contract using canonical serialization."""

    atomic_write_json(path, contract.model_dump(mode="python", by_alias=True))


def read_country_manifest(path: Path) -> GlobalCountryManifest:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    return GlobalCountryManifest.model_validate(read_json_object(manifest_path))


def read_corpus_manifest(path: Path) -> GlobalCorpusManifest:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    return GlobalCorpusManifest.model_validate(read_json_object(manifest_path))


__all__ = [
    "COUNTRY_IDS",
    "GRAPH_STAT_NAMES",
    "TRACE_ARRAY_TOKENS",
    "TRACE_NAMES",
    "CountryId",
    "GlobalArrayDescriptor",
    "GlobalCorpusEntry",
    "GlobalCorpusManifest",
    "GlobalCountryManifest",
    "GlobalSplitDescriptor",
    "atomic_write_contract",
    "read_corpus_manifest",
    "read_country_manifest",
]
