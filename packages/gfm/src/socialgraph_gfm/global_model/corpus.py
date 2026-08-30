"""Memory-mapped, fail-closed readers for converted SocialGraph-FM Global corpora."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np

from socialgraph_gfm.canonical import file_sha256
from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.gfm.corpus.common import resolve_within

from .contracts import (
    GRAPH_STAT_NAMES,
    TRACE_ARRAY_TOKENS,
    TRACE_NAMES,
    CountryId,
    GlobalArrayDescriptor,
    GlobalCorpusEntry,
    GlobalCorpusManifest,
    GlobalCountryManifest,
    GlobalSplitDescriptor,
    read_corpus_manifest,
    read_country_manifest,
)

MmapMode = Literal["r", "c"] | None


def _fail(message: str) -> ContractViolation:
    return ContractViolation(f"SocialGraph-FM Global corpus: {message}")


def _load_array(
    root: Path,
    descriptor: GlobalArrayDescriptor,
    *,
    verify_hash: bool,
    mmap_mode: MmapMode,
) -> np.ndarray:
    path = resolve_within(root, descriptor.path)
    if path.suffix != ".npy":
        raise _fail(f"array {descriptor.name!r} must be a .npy artifact")
    before = path.stat()
    if before.st_size != descriptor.byte_length:
        raise _fail(f"array {descriptor.name!r} byte length does not match its manifest")
    if verify_hash and file_sha256(path) != descriptor.sha256:
        raise _fail(f"array {descriptor.name!r} SHA-256 does not match its manifest")
    try:
        array = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    except (OSError, ValueError) as exc:
        raise _fail(f"array {descriptor.name!r} is not a safe numeric NPY file") from exc
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise _fail(f"array {descriptor.name!r} changed while it was being opened")
    if array.dtype.hasobject or array.dtype.kind in {"O", "S", "U", "V"}:
        raise _fail(f"array {descriptor.name!r} has a forbidden dtype")
    if array.dtype.str != descriptor.dtype or tuple(array.shape) != descriptor.shape:
        raise _fail(f"array {descriptor.name!r} does not match its dtype/shape contract")
    return array


def _require_dtype(name: str, array: np.ndarray, expected: str) -> None:
    if array.dtype.str != expected:
        raise _fail(f"array {name!r} must use dtype {expected}, observed {array.dtype.str}")


@dataclass(frozen=True)
class GlobalSplit:
    descriptor: GlobalSplitDescriptor
    train_mask: np.ndarray
    validation_mask: np.ndarray
    test_mask: np.ndarray


@dataclass(frozen=True)
class GlobalAdjacencyCSR:
    indptr: np.ndarray
    indices: np.ndarray


@dataclass(frozen=True)
class GlobalRelationCSR(GlobalAdjacencyCSR):
    weights: np.ndarray


@dataclass(frozen=True)
class GlobalCountryCorpus:
    root: Path
    manifest: GlobalCountryManifest
    arrays: Mapping[str, np.ndarray]
    splits: Mapping[str, GlobalSplit]

    def close(self) -> None:
        """Release any memory-mapped NPY handles held by this corpus."""

        mappings: dict[int, object] = {}
        for array in self.arrays.values():
            candidate: object | None = array
            visited: set[int] = set()
            while candidate is not None and id(candidate) not in visited:
                visited.add(id(candidate))
                mapping = getattr(candidate, "_mmap", None)
                if mapping is not None:
                    mappings[id(mapping)] = mapping
                candidate = getattr(candidate, "base", None)
        for mapping in mappings.values():
            close = getattr(mapping, "close", None)
            if callable(close):
                close()

    @property
    def edge_index(self) -> np.ndarray:
        return self.arrays["edge_index"]

    @property
    def text_features(self) -> np.ndarray:
        return self.arrays["text_features"]

    @property
    def degree_bucket(self) -> np.ndarray:
        return self.arrays["degree_bucket"]

    @property
    def structure_missing(self) -> np.ndarray:
        return self.arrays["structure_missing"]

    @property
    def graph_stats(self) -> np.ndarray:
        return self.arrays["graph_stats"]

    @property
    def labels(self) -> np.ndarray:
        return self.arrays["labels"]

    @property
    def trace_membership(self) -> np.ndarray:
        return self.arrays["trace_membership"]

    @property
    def fused_csr(self) -> GlobalAdjacencyCSR:
        return GlobalAdjacencyCSR(
            indptr=self.arrays["fused_indptr"],
            indices=self.arrays["fused_indices"],
        )

    def relation(self, trace_name: str) -> GlobalRelationCSR:
        try:
            token = TRACE_ARRAY_TOKENS[trace_name]
        except KeyError as exc:
            raise KeyError(f"unknown Global trace {trace_name!r}") from exc
        return GlobalRelationCSR(
            indptr=self.arrays[f"relation_{token}_indptr"],
            indices=self.arrays[f"relation_{token}_indices"],
            weights=self.arrays[f"relation_{token}_weights"],
        )

    def split(self, split_id: str) -> GlobalSplit:
        try:
            return self.splits[split_id]
        except KeyError as exc:
            raise KeyError(f"unknown Global split {split_id!r}") from exc


@dataclass(frozen=True)
class GlobalCorpusIndex:
    root: Path
    manifest: GlobalCorpusManifest
    entries: Mapping[CountryId, GlobalCorpusEntry]

    def country_root(self, country_id: CountryId) -> Path:
        try:
            entry = self.entries[country_id]
        except KeyError as exc:
            raise KeyError(f"unknown Global country {country_id!r}") from exc
        return resolve_within(self.root, entry.manifest_path).parent

    def load_country(
        self,
        country_id: CountryId,
        *,
        verify_hashes: bool = True,
        verify_values: bool = True,
        mmap_mode: MmapMode = "r",
    ) -> GlobalCountryCorpus:
        return load_country_corpus(
            self.country_root(country_id),
            verify_hashes=verify_hashes,
            verify_values=verify_values,
            mmap_mode=mmap_mode,
        )


def _validate_country_arrays(
    manifest: GlobalCountryManifest,
    arrays: Mapping[str, np.ndarray],
    *,
    verify_values: bool,
) -> Mapping[str, GlobalSplit]:
    _require_dtype("edge_index", arrays["edge_index"], np.dtype(np.int64).str)
    _require_dtype("text_features", arrays["text_features"], np.dtype(np.float32).str)
    _require_dtype("degree_bucket", arrays["degree_bucket"], np.dtype(np.uint8).str)
    _require_dtype("structure_missing", arrays["structure_missing"], np.dtype(np.bool_).str)
    _require_dtype("graph_stats", arrays["graph_stats"], np.dtype(np.float32).str)
    _require_dtype("labels", arrays["labels"], np.dtype(np.uint8).str)
    _require_dtype("trace_membership", arrays["trace_membership"], np.dtype(np.bool_).str)
    _require_dtype("fused_indptr", arrays["fused_indptr"], np.dtype(np.int64).str)
    _require_dtype("fused_indices", arrays["fused_indices"], np.dtype(np.int64).str)

    node_count = manifest.node_count
    edge_index = arrays["edge_index"]
    degree_bucket = arrays["degree_bucket"]
    labels = arrays["labels"]

    def validate_csr(name: str, indptr: np.ndarray, indices: np.ndarray) -> None:
        if (
            indptr.shape != (node_count + 1,)
            or indices.ndim != 1
            or int(indptr[0]) != 0
            or int(indptr[-1]) != indices.shape[0]
            or bool((np.diff(indptr) < 0).any())
        ):
            raise _fail(f"{name} CSR pointer inventory is invalid")
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= node_count):
            raise _fail(f"{name} CSR contains an out-of-range node")
        for row in range(node_count):
            start, stop = int(indptr[row]), int(indptr[row + 1])
            if stop - start > 1 and bool((np.diff(indices[start:stop]) <= 0).any()):
                raise _fail(f"{name} CSR row {row} is not strictly sorted")

    fused_indptr = arrays["fused_indptr"]
    fused_indices = arrays["fused_indices"]
    validate_csr("fused", fused_indptr, fused_indices)
    for row in range(node_count):
        start, stop = int(fused_indptr[row]), int(fused_indptr[row + 1])
        if (
            not bool(np.all(edge_index[0, start:stop] == row))
            or not np.array_equal(edge_index[1, start:stop], fused_indices[start:stop])
        ):
            raise _fail("fused CSR does not exactly match edge_index")
    if not np.array_equal(arrays["structure_missing"], np.diff(fused_indptr) == 0):
        raise _fail("structure_missing does not match factual fused degree zero")

    membership = arrays["trace_membership"]
    for column, trace_name in enumerate(TRACE_NAMES):
        token = TRACE_ARRAY_TOKENS[trace_name]
        indptr = arrays[f"relation_{token}_indptr"]
        indices = arrays[f"relation_{token}_indices"]
        weights = arrays[f"relation_{token}_weights"]
        _require_dtype(f"relation_{token}_indptr", indptr, np.dtype(np.int64).str)
        _require_dtype(f"relation_{token}_indices", indices, np.dtype(np.int64).str)
        _require_dtype(f"relation_{token}_weights", weights, np.dtype(np.float64).str)
        validate_csr(f"relation:{trace_name}", indptr, indices)
        if weights.shape != indices.shape or not bool(np.isfinite(weights).all()):
            raise _fail(f"relation {trace_name!r} weights are invalid")
        participating = np.diff(indptr) > 0
        if bool(np.logical_and(participating, ~membership[:, column]).any()):
            raise _fail(f"relation {trace_name!r} CSR contradicts trace membership")
    if verify_values:
        if not bool(np.isfinite(arrays["text_features"]).all()):
            raise _fail("text_features contains NaN or Infinity")
        if (
            arrays["graph_stats"].shape != (len(GRAPH_STAT_NAMES),)
            or not bool(np.isfinite(arrays["graph_stats"]).all())
            or bool((arrays["graph_stats"] < 0).any())
        ):
            raise _fail("graph_stats is invalid")
        graph_stats = arrays["graph_stats"]
        bounded = graph_stats[[1, 2, 3, 7, 8, 9, 10, 11, 12]]
        relation_proportions = graph_stats[8:13]
        if (
            bool((bounded > 1).any())
            or not bool(graph_stats[4] <= graph_stats[5] <= graph_stats[6])
            or not (
                np.isclose(relation_proportions.sum(), 0.0)
                or np.isclose(relation_proportions.sum(), 1.0)
            )
        ):
            raise _fail("graph_stats normalization or ordering is invalid")
        if edge_index.size and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= node_count
        ):
            raise _fail("edge_index contains a node outside the manifest range")
        if degree_bucket.size and int(degree_bucket.max()) >= manifest.structural_buckets:
            raise _fail("degree_bucket contains an out-of-range bucket")
        if not bool(np.isin(labels, (0, 1)).all()):
            raise _fail("labels must be binary")

    loaded_splits: dict[str, GlobalSplit] = {}
    for descriptor in manifest.splits:
        masks = (
            arrays[descriptor.train_array],
            arrays[descriptor.validation_array],
            arrays[descriptor.test_array],
        )
        for name, mask in zip(("train", "validation", "test"), masks, strict=True):
            _require_dtype(f"{descriptor.split_id}:{name}", mask, np.dtype(np.bool_).str)
            if tuple(mask.shape) != (node_count,):
                raise _fail(f"split {descriptor.split_id!r} {name} mask has the wrong shape")
            if verify_values and not bool(mask.any()):
                raise _fail(f"split {descriptor.split_id!r} {name} mask is empty")
        train, validation, test = masks
        if verify_values and bool(
            np.logical_or(
                np.logical_and(train, validation),
                np.logical_or(np.logical_and(train, test), np.logical_and(validation, test)),
            ).any()
        ):
            raise _fail(f"split {descriptor.split_id!r} role masks overlap")
        loaded_splits[descriptor.split_id] = GlobalSplit(
            descriptor=descriptor,
            train_mask=train,
            validation_mask=validation,
            test_mask=test,
        )
    return MappingProxyType(loaded_splits)


def load_country_corpus(
    root: Path,
    *,
    verify_hashes: bool = True,
    verify_values: bool = True,
    mmap_mode: MmapMode = "r",
) -> GlobalCountryCorpus:
    """Open one country without executing pickle or Torch serialization code."""

    country_root = root.expanduser().resolve()
    manifest = read_country_manifest(country_root / "manifest.json")
    arrays = {
        descriptor.name: _load_array(
            country_root,
            descriptor,
            verify_hash=verify_hashes,
            mmap_mode=mmap_mode,
        )
        for descriptor in manifest.arrays
    }
    split_map = _validate_country_arrays(manifest, arrays, verify_values=verify_values)
    return GlobalCountryCorpus(
        root=country_root,
        manifest=manifest,
        arrays=MappingProxyType(arrays),
        splits=split_map,
    )


def load_corpus_index(root: Path, *, verify_manifests: bool = True) -> GlobalCorpusIndex:
    """Load the six-country index and bind each referenced country manifest."""

    corpus_root = root.expanduser().resolve()
    manifest = read_corpus_manifest(corpus_root / "manifest.json")
    entries: dict[CountryId, GlobalCorpusEntry] = {}
    for entry in manifest.countries:
        manifest_path = resolve_within(corpus_root, entry.manifest_path)
        country_manifest = read_country_manifest(manifest_path)
        if verify_manifests and (
            country_manifest.country_id != entry.country_id
            or country_manifest.content_hash != entry.manifest_hash
            or country_manifest.source_hashes != entry.source_hashes
            or country_manifest.split_hashes != entry.split_hashes
        ):
            raise _fail(f"country index evidence does not match {entry.country_id!r}")
        entries[entry.country_id] = entry
    return GlobalCorpusIndex(
        root=corpus_root,
        manifest=manifest,
        entries=MappingProxyType(entries),
    )


__all__ = [
    "GlobalAdjacencyCSR",
    "GlobalCorpusIndex",
    "GlobalCountryCorpus",
    "GlobalRelationCSR",
    "GlobalSplit",
    "load_corpus_index",
    "load_country_corpus",
]
