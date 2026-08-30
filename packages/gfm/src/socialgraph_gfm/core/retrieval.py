"""Deterministic non-text structural retrieval for local static graphs."""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_sha256


_HASH_PATTERN = r"^[0-9a-f]{64}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        protected_namespaces=("model_dump",),
        strict=True,
    )


class StructuralRecord(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-structural-record/2.0"] = Field(
        alias="schemaVersion"
    )
    record_id: str = Field(alias="recordId", min_length=1, max_length=500)
    kind: Literal["node", "ego", "community"]
    entity_ids: tuple[str, ...] = Field(alias="entityIds", min_length=1)
    vector: tuple[float, ...] = Field(min_length=1)
    representation: Literal["embedding", "motif-signature"]
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH_PATTERN)
    record_hash: str = Field(alias="recordHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_vector_and_hash(self):
        if not all(math.isfinite(value) for value in self.vector):
            raise ValueError("structural vector values must be finite")
        if not any(value != 0.0 for value in self.vector):
            raise ValueError("structural vector must be non-zero")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"record_hash"})
        )
        if self.record_hash != expected:
            raise ValueError("recordHash does not match canonical structural record content")
        return self

    @classmethod
    def create(
        cls,
        *,
        record_id: str,
        kind: Literal["node", "ego", "community"],
        entity_ids: tuple[str, ...],
        vector: tuple[float, ...],
        representation: Literal["embedding", "motif-signature"],
        graph_version_hash: str,
        model_version: str,
        model_version_hash: str,
    ) -> StructuralRecord:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-structural-record/2.0",
            "recordId": record_id,
            "kind": kind,
            "entityIds": entity_ids,
            "vector": vector,
            "representation": representation,
            "graphVersionHash": graph_version_hash,
            "modelVersion": model_version,
            "modelVersionHash": model_version_hash,
        }
        payload["recordHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)


class StructuralSearchResult(_StrictModel):
    record: StructuralRecord
    score: float = Field(ge=-1.0, le=1.0)
    query_provenance_hash: str = Field(alias="queryProvenanceHash", pattern=_HASH_PATTERN)


class StructuralQuery(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-structural-query/2.0"] = Field(
        alias="schemaVersion"
    )
    vector: tuple[float, ...] = Field(min_length=1)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH_PATTERN)
    representation: Literal["embedding", "motif-signature"]
    kinds: tuple[Literal["node", "ego", "community"], ...] = Field(min_length=1)
    limit: int = Field(ge=1, le=100)
    exclude_record_hash: str | None = Field(
        alias="excludeRecordHash", pattern=_HASH_PATTERN
    )
    query_hash: str = Field(alias="queryHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_query(self):
        if not all(math.isfinite(value) for value in self.vector):
            raise ValueError("structural query vector must contain finite values")
        if not any(value != 0.0 for value in self.vector):
            raise ValueError("structural query vector must be non-zero")
        if len(set(self.kinds)) != len(self.kinds):
            raise ValueError("structural query kinds must be unique")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"query_hash"})
        )
        if self.query_hash != expected:
            raise ValueError("queryHash does not match canonical structural query")
        return self

    @classmethod
    def create(
        cls,
        *,
        vector: tuple[float, ...],
        graph_version_hash: str,
        model_version: str,
        model_version_hash: str,
        representation: Literal["embedding", "motif-signature"],
        kinds: tuple[Literal["node", "ego", "community"], ...],
        limit: int,
        exclude_record_hash: str | None,
    ) -> StructuralQuery:
        if not isinstance(vector, tuple):
            raise TypeError("structural query vector must be a tuple")
        if any(isinstance(value, bool) or not isinstance(value, numbers.Real) for value in vector):
            raise TypeError("structural query vector values must be native numeric values")
        if not isinstance(kinds, tuple):
            raise TypeError("structural query kinds must be a tuple")
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("structural query limit must be a native integer")
        if not model_version:
            raise ValueError("structural query model version must be nonempty")
        for name, value in (
            ("graph", graph_version_hash),
            ("model", model_version_hash),
        ):
            if not isinstance(value, str) or not re.fullmatch(_HASH_PATTERN, value):
                raise ValueError(f"structural query {name} hash must be a lowercase SHA-256")
        if exclude_record_hash is not None and (
            not isinstance(exclude_record_hash, str)
            or not re.fullmatch(_HASH_PATTERN, exclude_record_hash)
        ):
            raise ValueError("structural query exclusion must be a lowercase SHA-256")
        if representation not in {"embedding", "motif-signature"}:
            raise ValueError("unsupported structural query representation")
        if not kinds or any(kind not in {"node", "ego", "community"} for kind in kinds):
            raise ValueError("unsupported structural query kind")
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-structural-query/2.0",
            "vector": vector,
            "graphVersionHash": graph_version_hash,
            "modelVersion": model_version,
            "modelVersionHash": model_version_hash,
            "representation": representation,
            "kinds": kinds,
            "limit": limit,
            "excludeRecordHash": exclude_record_hash,
        }
        payload["queryHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)


class StructuralIndex:
    """Small-graph brute-force cosine index with explicit version filters."""

    def __init__(self, records: Iterable[StructuralRecord] = ()) -> None:
        self._records: dict[str, StructuralRecord] = {}
        self._dimensions: dict[tuple[str, str, str], int] = {}
        self._hash_to_version: dict[str, str] = {}
        self._version_to_hash: dict[str, str] = {}
        self._queries: dict[str, StructuralQuery] = {}
        self._query_results: dict[
            tuple[str, str], StructuralSearchResult
        ] = {}
        for record in records:
            self.add(record)

    @property
    def records(self) -> tuple[StructuralRecord, ...]:
        return tuple(sorted(self._records.values(), key=lambda item: item.record_hash))

    def add(self, record: StructuralRecord) -> None:
        validated = StructuralRecord.model_validate(
            record.model_dump(mode="python", by_alias=True)
        )
        key = (
            validated.graph_version_hash,
            validated.model_version_hash,
            validated.representation,
        )
        known_version = self._hash_to_version.get(validated.model_version_hash)
        known_hash = self._version_to_hash.get(validated.model_version)
        if (known_version is not None and known_version != validated.model_version) or (
            known_hash is not None and known_hash != validated.model_version_hash
        ):
            raise ValueError("structural model version/hash mapping is inconsistent")
        dimension = self._dimensions.get(key)
        if dimension is not None and dimension != len(validated.vector):
            raise ValueError("structural record dimension does not match its versioned index")
        existing = self._records.get(validated.record_hash)
        if existing is not None and existing != validated:
            raise ValueError("structural record hash collision")
        self._hash_to_version[validated.model_version_hash] = validated.model_version
        self._version_to_hash[validated.model_version] = validated.model_version_hash
        self._dimensions[key] = len(validated.vector)
        self._records[validated.record_hash] = validated

    def get(self, record_hash: str) -> StructuralRecord:
        try:
            return self._records[record_hash]
        except KeyError as error:
            raise ValueError("unknown structural record hash") from error

    def query(self, query: StructuralQuery) -> tuple[StructuralSearchResult, ...]:
        if not isinstance(query, StructuralQuery):
            raise TypeError("structural retrieval requires a validated StructuralQuery")
        validated = StructuralQuery.model_validate(
            query.model_dump(mode="python", by_alias=True)
        )
        query_vector = validated.vector
        norm = math.sqrt(sum(value * value for value in query_vector))
        key = (
            validated.graph_version_hash,
            validated.model_version_hash,
            validated.representation,
        )
        expected_dimension = self._dimensions.get(key)
        if (
            self._hash_to_version.get(validated.model_version_hash) != validated.model_version
            or self._version_to_hash.get(validated.model_version) != validated.model_version_hash
            or expected_dimension is None
        ):
            raise ValueError("structural query model version/hash is not registered compatibly")
        if len(query_vector) != expected_dimension:
            raise ValueError("structural query dimension does not match versioned index")
        kind_set = set(validated.kinds)
        if validated.exclude_record_hash is not None:
            excluded = self._records.get(validated.exclude_record_hash)
            if excluded is None or (
                excluded.graph_version_hash != validated.graph_version_hash
                or excluded.model_version != validated.model_version
                or excluded.model_version_hash != validated.model_version_hash
                or excluded.representation != validated.representation
            ):
                raise ValueError("structural query exclusion is not compatible with the query")
        scored: list[StructuralSearchResult] = []
        for record in self._records.values():
            if (
                record.graph_version_hash != validated.graph_version_hash
                or record.model_version != validated.model_version
                or record.model_version_hash != validated.model_version_hash
                or record.representation != validated.representation
                or record.kind not in kind_set
                or record.record_hash == validated.exclude_record_hash
            ):
                continue
            if len(record.vector) != len(query_vector):
                raise ValueError("stored structural record dimension is inconsistent")
            record_norm = math.sqrt(sum(value * value for value in record.vector))
            score = sum(a * b for a, b in zip(query_vector, record.vector, strict=True)) / (
                norm * record_norm
            )
            # Collapse insignificant floating error at the cosine boundary.
            score = max(-1.0, min(1.0, score))
            scored.append(
                StructuralSearchResult(
                    record=record,
                    score=score,
                    query_provenance_hash=validated.query_hash,
                )
            )
        results = tuple(
            sorted(scored, key=lambda item: (-item.score, item.record.record_id, item.record.record_hash))[
                : validated.limit
            ]
        )
        existing_query = self._queries.get(validated.query_hash)
        if existing_query is not None and existing_query != validated:
            raise ValueError("structural query hash collision")
        self._queries[validated.query_hash] = validated
        for result in results:
            self._query_results[(validated.query_hash, result.record.record_hash)] = result
        return results

    def resolve_query_result(
        self, *, query_hash: str, record_hash: str
    ) -> tuple[StructuralQuery, StructuralSearchResult]:
        query = self._queries.get(query_hash)
        result = self._query_results.get((query_hash, record_hash))
        if query is None or result is None:
            raise ValueError("unknown registered structural query/result provenance")
        validated_query = StructuralQuery.model_validate(
            query.model_dump(mode="python", by_alias=True)
        )
        validated_result = StructuralSearchResult.model_validate(
            result.model_dump(mode="python", by_alias=True)
        )
        if validated_result.query_provenance_hash != validated_query.query_hash:
            raise ValueError("registered structural query/result provenance mismatch")
        return validated_query, validated_result

    def query_by_record(self, record_hash: str, *, limit: int = 10) -> tuple[StructuralSearchResult, ...]:
        reference = self.get(record_hash)
        query = StructuralQuery.create(
            vector=reference.vector,
            graph_version_hash=reference.graph_version_hash,
            model_version=reference.model_version,
            model_version_hash=reference.model_version_hash,
            representation=reference.representation,
            kinds=("node", "ego", "community"),
            limit=limit,
            exclude_record_hash=reference.record_hash,
        )
        return self.query(query)


__all__ = ["StructuralIndex", "StructuralQuery", "StructuralRecord", "StructuralSearchResult"]
