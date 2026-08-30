"""OpenAlex Graph-AI acquisition with resumable, allowlisted API pages.

The API key is read at call time and is never accepted in configuration,
written to disk, logged, or included in artifact identity.  Topic selectors
must resolve to exactly one official topic; ambiguous human strings stop the
fetch rather than silently changing the corpus definition.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ...canonical import canonical_json, canonical_sha256, file_sha256
from ...errors import ContractViolation
from ...runtime import RuntimeLayout
from ..configuration import load_openalex_spec
from .common import (
    PORTABLE_ID_HASH_ALGORITHM,
    NumericShardWriter,
    ShardRecord,
    append_jsonl_fsync,
    atomic_write_json,
    atomic_write_jsonl,
    build_manifest,
    exclusive_file_lock,
    load_npz_safe,
    portable_id_hash,
    read_json_object,
    read_jsonl,
    resolve_within,
    verify_manifest,
)

API_BASE = "https://api.openalex.org"
LICENSE_ID = "CC0-1.0"
LICENSE_URL = "https://help.openalex.org/hc/en-us/articles/24396686889751-About-us"
CORPUS_ID = "openalex-graph-ai"
DOMAIN_ID = "openalex-graph-ai"
NEWCOMER_OVERLAY_ID = "openalex-newcomer-overlay-v1"
NEWCOMER_OVERLAY_SCHEMA = "gfm.openalex-newcomer-verification/1.0"
MIN_DATE = date(2016, 1, 1)
MAX_DATE = date(2025, 12, 31)
# OpenAlex's current work-type vocabulary has no ``proceedings-article``;
# conference papers are ``article`` works whose source can be a conference.
# The product requirement is therefore represented by the current canonical
# API types below, with the compatibility mapping recorded in every manifest.
ALLOWED_WORK_TYPES = frozenset({"article", "preprint"})
REQUESTED_WORK_CATEGORIES = ("article", "preprint", "proceedings-article")
WORK_TYPE_COMPATIBILITY = {
    "article": "article",
    "preprint": "preprint",
    "proceedings-article": "article (conference source when declared)",
}
ALLOWED_WORK_FIELDS = frozenset(
    {
        "id",
        "doi",
        "display_name",
        "publication_date",
        "publication_year",
        "type",
        "authorships",
        "primary_topic",
        "topics",
        "primary_location",
        "referenced_works",
        "ids",
        "updated_date",
    }
)
FORBIDDEN_FIELDS = frozenset(
    {
        "abstract_inverted_index",
        "works_count",
        "cited_by_count",
        "counts_by_year",
        "summary_stats",
        "last_known_institution",
        "last_known_institutions",
    }
)
CLUSTER_CAPS = (70_000, 65_000, 65_000)
TOTAL_CAP = 200_000
PER_PAGE = 100
MAX_SAMPLE_PER_STRATUM = 10_000
MAX_FETCH_REQUESTS = 2_500
MAX_NEWCOMER_REQUESTS = 20_000
NEWCOMER_RESUME_SCHEMA = "gfm.openalex-newcomer-resume/1.0"
NEWCOMER_GROUP_BY_PAGE_SIZE = 200
NEWCOMER_HISTORY_QUERY_SCHEMA = "gfm.openalex-newcomer-history-query/1.1"
NEWCOMER_HISTORY_QUERY_POLICY = {
    "schemaVersion": NEWCOMER_HISTORY_QUERY_SCHEMA,
    "groupByPageSize": NEWCOMER_GROUP_BY_PAGE_SIZE,
    "groupByCursorStart": "*",
    "completeGroupPageRule": "meta-next-cursor-explicit-null",
    "groupsCountRule": "nonnegative-or-null-telemetry-only",
    "truncatedPageRule": "recurse-only-unresolved-requested-authors",
    "singletonRule": "works-existence-select-id-per-page-one",
    "extraGroupRule": "ignore-valid-author-or-null-and-audit",
    "requestBudgetRule": "reserve-and-commit-before-request",
}
NEWCOMER_HISTORY_AUDIT_KEYS = (
    "history_group_by_requests",
    "history_existence_requests",
    "history_truncated_responses",
    "history_extra_groups_ignored",
    "history_null_groups_ignored",
    "history_returned_groups",
    "history_reported_groups",
)
NEWCOMER_RESUME_PREFIX = f".{CORPUS_ID}-newcomer-"
NEWCOMER_RESUME_SUFFIX = ".resume"
NEWCOMER_VERIFY_LOCK_NAME = f".{CORPUS_ID}-newcomer-verify.lock"
NEWCOMER_INGEST_COMMIT_ROWS = 1_000
MAX_NUMERIC_SHARD_ROWS = 50_000
# Above this size, a paper remains represented by its author--work hyperedge
# (authorship relations), but no O(k^2) coauthor clique is materialized.  This
# is deterministic, avoids inventing an arbitrary author subset, and keeps the
# preparation cost linear in the number of works for a fixed formal bound.
MAX_COAUTHOR_CLIQUE_AUTHORS = 32
PHYSICAL_ACCESS_SCHEMA = "gfm.physical-role-views/1.0"
ACCESS_ROLES = ("train", "validation", "test", "shadow")
ACCESS_ROLE_TAGS = {"train": "tr", "validation": "va", "test": "te", "shadow": "sh"}
RAW_NAME = "works.jsonl"
RESUME_NAME = "resume.json"
RESUME_SCHEMA = "gfm.openalex-resume/1.1"
TOPICS_NAME = "resolved-topics.json"
FETCH_LOCK_NAME = ".fetch.lock"
WORK_ELIGIBILITY_SCHEMA = "gfm.openalex-work-eligibility/1.0"
NO_VALID_AUTHORSHIPS_REASON = "noValidAuthorships"
NULL_AUTHOR_ID = "A9999999999"
WORK_ELIGIBILITY_POLICY = {
    "schemaVersion": WORK_ELIGIBILITY_SCHEMA,
    "emptyAuthorshipPolicy": "exclude-and-refill",
    "malformedAuthorshipPolicy": "fail-closed",
    "nullAuthorPolicy": "remove-authorship-and-exclude-work-if-none-remain",
    "remoteAuthorsCountFilter": False,
}
WORK_ELIGIBILITY_PROTOCOL = WORK_ELIGIBILITY_POLICY
UNRESOLVED_AUTHOR_ID_REASON = "unresolvedAuthorId"
NULL_AUTHOR_REASON = "openAlexNullAuthor"
AUTHORSHIP_DISCARD_REASONS = (UNRESOLVED_AUTHOR_ID_REASON, NULL_AUTHOR_REASON)
WORK_TEXT_ARRAYS = {
    "work_id_hash": (np.dtype(np.uint64).str, 1),
    "publication_timestamp": (np.dtype(np.int64).str, 1),
    "cluster": (np.dtype(np.int16).str, 1),
    "text_available": (np.dtype(np.bool_).str, 1),
}
EVENT_ARRAYS = {
    "src": (np.dtype(np.int64).str, 1),
    "dst": (np.dtype(np.int64).str, 1),
    "timestamp": (np.dtype(np.int64).str, 1),
    "relation": (np.dtype(np.int16).str, 1),
    "work_index": (np.dtype(np.int64).str, 1),
}
TARGET_ARRAYS = {
    "src": (np.dtype(np.int64).str, 1),
    "dst": (np.dtype(np.int64).str, 1),
    "timestamp": (np.dtype(np.int64).str, 1),
    "first_collaboration": (np.dtype(np.bool_).str, 1),
}
NEWCOMER_ARRAYS = {
    "author": (np.dtype(np.int64).str, 1),
    "t0": (np.dtype(np.int64).str, 1),
    "history_verified": (np.dtype(np.bool_).str, 1),
}
Transport = Callable[[str, Mapping[str, str]], Mapping[str, Any]]


def _fail(message: str) -> ContractViolation:
    return ContractViolation(f"OpenAlex: {message}")


def _normalise_id(value: Any, prefix: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"missing {prefix} identifier")
    item = value.rsplit("/", 1)[-1]
    if not item.startswith(prefix) or not item[len(prefix) :].isdigit():
        raise _fail(f"invalid {prefix} identifier {value!r}")
    return item


def _api_key(value: str | None = None) -> str:
    if value is not None:
        raise _fail("API keys may only be supplied through OPENALEX_API_KEY")
    key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if not key:
        raise _fail("OPENALEX_API_KEY is required; no network request was made")
    return key


def _default_transport(url: str, query: Mapping[str, str]) -> Mapping[str, Any]:
    encoded = urllib.parse.urlencode(query, safe=",:|*-_")
    request = urllib.request.Request(
        f"{url}?{encoded}",
        headers={"Accept": "application/json", "User-Agent": "SocialGraph-FM/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            import json

            value = json.load(response)
    except (OSError, ValueError) as exc:
        raise _fail(f"official API request failed: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise _fail("official API response is not a JSON object")
    return value


@dataclass(frozen=True)
class OpenAlexConfig:
    """Validated immutable selection contract for Graph-AI v1."""

    clusters: tuple[tuple[str, tuple[str, ...], int], ...]
    work_fields: tuple[str, ...]
    from_date: date = MIN_DATE
    to_date: date = MAX_DATE
    work_types: tuple[str, ...] = tuple(sorted(ALLOWED_WORK_TYPES))
    maximum_unique_works: int = TOTAL_CAP

    @classmethod
    def pinned(cls) -> OpenAlexConfig:
        spec = load_openalex_spec()
        clusters = tuple(
            (
                str(item["clusterId"]),
                tuple(str(value) for value in item["selectors"]),
                int(item["maximumWorks"]),
            )
            for item in spec["topicClusters"]
        )
        return cls(clusters=clusters, work_fields=tuple(spec["workSelect"]))

    def validate(self) -> None:
        if self.from_date != MIN_DATE or self.to_date != MAX_DATE:
            raise _fail("formal corpus dates must be exactly 2016-01-01 through 2025-12-31")
        if self.work_types != tuple(sorted(ALLOWED_WORK_TYPES)):
            raise _fail("formal corpus work types differ from the pinned allowlist")
        fields = set(self.work_fields)
        if fields != ALLOWED_WORK_FIELDS or fields.intersection(FORBIDDEN_FIELDS):
            raise _fail("work select fields differ from the source allowlist")
        if len(self.clusters) != 3 or tuple(item[2] for item in self.clusters) != CLUSTER_CAPS:
            raise _fail("topic cluster caps must be exactly 70000/65000/65000")
        if (
            self.maximum_unique_works != TOTAL_CAP
            or sum(item[2] for item in self.clusters) != TOTAL_CAP
        ):
            raise _fail("OpenAlex corpus cap must be exactly 200000 unique works")
        if len({item[0] for item in self.clusters}) != len(self.clusters):
            raise _fail("topic cluster IDs must be unique")

    @property
    def identity(self) -> str:
        return canonical_sha256(
            {
                "clusters": self.clusters,
                "workFields": self.work_fields,
                "from": self.from_date,
                "to": self.to_date,
                "workTypes": self.work_types,
                "maximumUniqueWorks": self.maximum_unique_works,
            }
        )


def parse_topic_selector(
    selector: str,
    *,
    transport: Transport | None = None,
    api_key: str | None = None,
) -> dict[str, str | int]:
    """Resolve one selector and reject zero, fuzzy, or ambiguous matches."""

    key = _api_key(api_key)
    selected_transport = transport or _default_transport
    value = selector.strip()
    if not value:
        raise _fail("topic selector cannot be empty")
    direct_id = value.startswith("https://openalex.org/T") or (
        value.startswith("T") and value[1:].isdigit()
    )
    if direct_id:
        topic_id = _normalise_id(value, "T")
        response = selected_transport(
            f"{API_BASE}/topics/{topic_id}",
            {"api_key": key, "select": "id,display_name"},
        )
        candidates = [response]
    else:
        response = selected_transport(
            f"{API_BASE}/topics",
            {"api_key": key, "search": value, "select": "id,display_name", "per_page": "10"},
        )
        results = response.get("results")
        candidates = results if isinstance(results, list) else []
    response_summary = [
        {
            "id": item.get("id"),
            "displayName": item.get("display_name"),
        }
        for item in candidates
        if isinstance(item, dict)
    ]
    exact = [
        item
        for item in candidates
        if isinstance(item, dict)
        and isinstance(item.get("display_name"), str)
        and (direct_id or item["display_name"].casefold() == value.casefold())
    ]
    if len(exact) != 1:
        raise _fail(f"topic selector {selector!r} resolved to {len(exact)} exact matches")
    topic_id = _normalise_id(exact[0].get("id"), "T")
    return {
        "id": topic_id,
        "displayName": str(exact[0]["display_name"]),
        "selector": value,
        "candidateCount": len(response_summary),
        "responseSummaryHash": canonical_sha256(response_summary),
        "apiEndpoint": "topics",
    }


def _validate_work(
    value: Mapping[str, Any], *, allow_empty_authorships: bool = False
) -> dict[str, Any]:
    extra = set(value).difference(ALLOWED_WORK_FIELDS)
    if extra.intersection(FORBIDDEN_FIELDS):
        raise _fail(f"API work contains blocked leakage fields: {sorted(extra & FORBIDDEN_FIELDS)}")
    if extra:
        raise _fail(f"API work contains fields outside the select allowlist: {sorted(extra)}")
    missing = ALLOWED_WORK_FIELDS.difference(value)
    if missing:
        raise _fail(f"API work is missing selected fields: {sorted(missing)}")
    work_id = _normalise_id(value.get("id"), "W")
    try:
        publication = date.fromisoformat(str(value["publication_date"]))
    except ValueError as exc:
        raise _fail(f"work {work_id} has an invalid publication_date") from exc
    if not MIN_DATE <= publication <= MAX_DATE:
        raise _fail(f"work {work_id} falls outside the formal publication interval")
    if value.get("type") not in ALLOWED_WORK_TYPES:
        raise _fail(f"work {work_id} has a blocked type")
    authorships = value.get("authorships")
    if not isinstance(authorships, list):
        raise _fail(f"work {work_id} has invalid authorships")

    # Nested author/institution objects may contain current snapshot counters in
    # unselected API responses.  Enforce the privacy/leakage block recursively.
    def walk(item: Any) -> None:
        if isinstance(item, dict):
            blocked = set(item).intersection(FORBIDDEN_FIELDS)
            if blocked:
                raise _fail(f"work {work_id} contains nested blocked fields: {sorted(blocked)}")
            for nested in item.values():
                walk(nested)
        elif isinstance(item, list):
            for nested in item:
                walk(nested)

    walk(value)
    valid_authorships: list[dict[str, Any]] = []
    for authorship in authorships:
        if not isinstance(authorship, dict):
            raise _fail(f"work {work_id} contains a malformed authorship")
        author = authorship.get("author")
        if not isinstance(author, dict):
            raise _fail(f"work {work_id} contains a malformed authorship author")
        author_id_value = author.get("id")
        if author_id_value is None:
            continue
        author_id = _normalise_id(author_id_value, "A")
        if author_id != NULL_AUTHOR_ID:
            valid_authorships.append(dict(authorship))
    if not valid_authorships and not allow_empty_authorships:
        raise _fail(f"work {work_id} has no valid authorships")
    normalized = dict(value)
    normalized["authorships"] = valid_authorships
    return normalized


def _discarded_authorship_counts(value: Mapping[str, Any]) -> dict[str, int]:
    counts = {reason: 0 for reason in AUTHORSHIP_DISCARD_REASONS}
    authorships = value.get("authorships")
    if not isinstance(authorships, list):
        raise _fail("cannot audit non-array authorships")
    for authorship in authorships:
        if not isinstance(authorship, dict) or not isinstance(authorship.get("author"), dict):
            raise _fail("cannot audit malformed authorship")
        author_id_value = authorship["author"].get("id")
        if author_id_value is None:
            counts[UNRESOLVED_AUTHOR_ID_REASON] += 1
        elif _normalise_id(author_id_value, "A") == NULL_AUTHOR_ID:
            counts[NULL_AUTHOR_REASON] += 1
    return counts


def _empty_exclusion_digest() -> str:
    return canonical_sha256({"schemaVersion": WORK_ELIGIBILITY_SCHEMA, "excludedWorkIds": []})


def _extend_exclusion_digest(previous: str, *, reason: str, work_id: str) -> str:
    return canonical_sha256(
        {
            "schemaVersion": WORK_ELIGIBILITY_SCHEMA,
            "previous": previous,
            "reason": reason,
            "workId": work_id,
        }
    )


def _work_eligibility_audit(state: Mapping[str, Any]) -> dict[str, Any]:
    reasons = state.get("excludedByReason")
    if not isinstance(reasons, dict) or set(reasons) != {NO_VALID_AUTHORSHIPS_REASON}:
        raise _fail("resume work-eligibility reason inventory is invalid")
    count = reasons[NO_VALID_AUTHORSHIPS_REASON]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise _fail("resume work-eligibility exclusion count is invalid")
    fetched_rows = state.get("fetchedRows")
    raw_rows = state.get("rawRows")
    accepted_rows = state.get("acceptedRows")
    if (
        isinstance(fetched_rows, bool)
        or not isinstance(fetched_rows, int)
        or isinstance(raw_rows, bool)
        or not isinstance(raw_rows, int)
        or raw_rows < 0
        or isinstance(accepted_rows, bool)
        or not isinstance(accepted_rows, int)
        or accepted_rows < 0
        or fetched_rows != raw_rows
        or fetched_rows != accepted_rows + count
    ):
        raise _fail("resume work-eligibility row accounting is invalid")
    digest = state.get("excludedWorkIdDigest")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
        or (count == 0 and digest != _empty_exclusion_digest())
    ):
        raise _fail("resume work-eligibility digest is invalid")
    if state.get("workEligibilityProtocol") != WORK_ELIGIBILITY_POLICY:
        raise _fail("resume work-eligibility protocol is invalid")
    discarded = state.get("discardedAuthorshipsByReason")
    if not isinstance(discarded, dict) or set(discarded) != set(AUTHORSHIP_DISCARD_REASONS):
        raise _fail("resume discarded-authorship reason inventory is invalid")
    if any(
        isinstance(discarded[reason], bool)
        or not isinstance(discarded[reason], int)
        or discarded[reason] < 0
        for reason in AUTHORSHIP_DISCARD_REASONS
    ):
        raise _fail("resume discarded-authorship count is invalid")
    return {
        "workEligibilityProtocol": dict(WORK_ELIGIBILITY_POLICY),
        "acceptedRows": accepted_rows,
        "storedRows": raw_rows,
        "inspectedRows": fetched_rows,
        "excludedRows": count,
        "excludedByReason": {NO_VALID_AUTHORSHIPS_REASON: count},
        "excludedWorkIdDigest": digest,
        "discardedAuthorshipsByReason": {
            reason: discarded[reason] for reason in AUTHORSHIP_DISCARD_REASONS
        },
    }


def _scan_work_eligibility_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    stored_rows = 0
    accepted_rows = 0
    excluded_rows = 0
    excluded_digest = _empty_exclusion_digest()
    excluded_strata: set[str] = set()
    discarded = {reason: 0 for reason in AUTHORSHIP_DISCARD_REASONS}
    for row in rows:
        if set(row) != {"clusterId", "stratumId", "work"} or not isinstance(row.get("work"), dict):
            raise _fail("committed raw work row is invalid")
        work = _validate_work(row["work"], allow_empty_authorships=True)
        item_discards = _discarded_authorship_counts(row["work"])
        for reason in AUTHORSHIP_DISCARD_REASONS:
            discarded[reason] += item_discards[reason]
        stored_rows += 1
        if work["authorships"]:
            accepted_rows += 1
            continue
        excluded_rows += 1
        excluded_strata.add(str(row["stratumId"]))
        excluded_digest = _extend_exclusion_digest(
            excluded_digest,
            reason=NO_VALID_AUTHORSHIPS_REASON,
            work_id=_normalise_id(work["id"], "W"),
        )
    return {
        "storedRows": stored_rows,
        "acceptedRows": accepted_rows,
        "excludedRows": excluded_rows,
        "excludedWorkIdDigest": excluded_digest,
        "excludedStrata": excluded_strata,
        "discardedAuthorshipsByReason": discarded,
    }


def _scan_raw_work_eligibility(works_path: Path) -> dict[str, Any]:
    rows: Iterable[Mapping[str, Any]] = ()
    if works_path.exists():
        rows = read_jsonl(works_path)
    return _scan_work_eligibility_rows(rows)


def _raw_work_audit_matches_state(actual: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    reasons = state.get("excludedByReason")
    return bool(
        isinstance(reasons, dict)
        and actual.get("storedRows") == state.get("rawRows")
        and actual.get("acceptedRows") == state.get("acceptedRows")
        and actual.get("excludedRows") == reasons.get(NO_VALID_AUTHORSHIPS_REASON)
        and actual.get("excludedWorkIdDigest") == state.get("excludedWorkIdDigest")
        and actual.get("discardedAuthorshipsByReason") == state.get("discardedAuthorshipsByReason")
    )


def _raw_file_observation(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _rollback_uncommitted_raw_tail(
    state: Mapping[str, Any],
    works_path: Path,
    *,
    strata: Sequence[Mapping[str, Any]],
) -> bool:
    """Roll back one complete page appended after the last durable resume.

    Raw JSONL is fsynced before the resume file is atomically replaced.  If the
    latter replace loses a transient Windows sharing race, the raw file is one
    page ahead.  The durable resume remains the sole commit authority: this
    routine never derives or advances state from the tail.  It truncates only
    after the committed prefix reproduces every eligibility audit field and the
    finite tail is a complete, schema-valid page for the current stratum.
    """

    audit_names = {
        "workEligibilityProtocol",
        "fetchedRows",
        "acceptedRows",
        "excludedByReason",
        "excludedWorkIdDigest",
        "discardedAuthorshipsByReason",
    }
    present = audit_names.intersection(state)
    if present != audit_names:
        return False
    _work_eligibility_audit(state)
    committed = state.get("rawRows")
    if isinstance(committed, bool) or not isinstance(committed, int) or committed < 0:
        raise _fail("resume raw row boundary is invalid")
    if not works_path.exists():
        if committed == 0:
            return False
        raise _fail("resume raw row boundary has no work artifact")

    observation = _raw_file_observation(works_path)
    tail: list[dict[str, Any]] = []

    def committed_rows() -> Iterator[dict[str, Any]]:
        for index, row in enumerate(read_jsonl(works_path)):
            if index < committed:
                yield row
                continue
            if len(tail) >= PER_PAGE:
                raise _fail("uncommitted raw tail exceeds one bounded API page")
            tail.append(row)

    prefix_audit = _scan_work_eligibility_rows(committed_rows())
    if not _raw_work_audit_matches_state(prefix_audit, state):
        raise _fail("committed raw prefix does not match the resume eligibility audit")
    if not tail:
        return False
    if state.get("complete") is True:
        raise _fail("a complete resume has an unexpected raw tail")

    # Validate every tail row before discarding it.  A partial JSON line,
    # malformed work, mixed stratum, or more than one possible page remains a
    # hard failure rather than being treated as a recoverable commit window.
    _scan_work_eligibility_rows(tail)
    stratum_index = state.get("stratumIndex")
    current_received = state.get("currentStratumReceived")
    if (
        isinstance(stratum_index, bool)
        or not isinstance(stratum_index, int)
        or not 0 <= stratum_index < len(strata)
        or isinstance(current_received, bool)
        or not isinstance(current_received, int)
        or current_received < 0
    ):
        raise _fail("resume cannot bind its uncommitted raw tail to a current stratum")
    expected = strata[stratum_index]
    quota = int(str(expected["quota"]))
    remaining = quota - current_received
    if remaining <= 0 or len(tail) > min(PER_PAGE, remaining):
        raise _fail("uncommitted raw tail exceeds the current stratum remainder")
    if any(
        row.get("clusterId") != expected.get("clusterId")
        or row.get("stratumId") != expected.get("stratumId")
        for row in tail
    ):
        raise _fail("uncommitted raw tail does not belong to the current stratum")
    if _raw_file_observation(works_path) != observation:
        raise _fail("raw work artifact changed during tail recovery")

    def durable_prefix() -> Iterator[dict[str, Any]]:
        for index, row in enumerate(read_jsonl(works_path)):
            if index >= committed:
                break
            yield row

    written = atomic_write_jsonl(works_path, durable_prefix())
    if written != committed:
        raise _fail("raw tail rollback did not preserve the committed row boundary")
    repaired = _scan_raw_work_eligibility(works_path)
    if not _raw_work_audit_matches_state(repaired, state):
        raise _fail("raw tail rollback changed the committed eligibility audit")
    return True


def _migrate_or_validate_work_eligibility_state(state: dict[str, Any], works_path: Path) -> bool:
    """Add auditable eligibility state to a compatible pre-fix resume.

    The old adapter admitted non-empty authorship arrays even when every
    dehydrated author ID was null.  During migration, unresolved/NULL
    authorships are stripped and works with no real author ID are excluded from
    the current unfinished stratum.  Completed strata are never rewritten
    semantically because their next deterministic sample draw is not present in
    the legacy state.  Partial audit fields are never guessed.
    """

    audit_names = {
        "workEligibilityProtocol",
        "fetchedRows",
        "acceptedRows",
        "excludedByReason",
        "excludedWorkIdDigest",
        "discardedAuthorshipsByReason",
    }
    present = audit_names.intersection(state)
    if present and present != audit_names:
        raise _fail("resume contains a partial work-eligibility audit")
    if present:
        _work_eligibility_audit(state)
        actual = _scan_raw_work_eligibility(works_path)
        if not _raw_work_audit_matches_state(actual, state):
            raise _fail("resume does not match the committed raw work eligibility audit")
        return False

    raw_rows = state.get("rawRows")
    if isinstance(raw_rows, bool) or not isinstance(raw_rows, int) or raw_rows < 0:
        raise _fail("legacy resume rawRows is invalid")
    actual = _scan_raw_work_eligibility(works_path)
    actual_rows = int(actual["storedRows"])
    accepted_rows = int(actual["acceptedRows"])
    excluded_rows = int(actual["excludedRows"])
    excluded_digest = str(actual["excludedWorkIdDigest"])
    discarded = dict(actual["discardedAuthorshipsByReason"])
    completed = state.get("strata")
    if not isinstance(completed, list) or any(not isinstance(item, dict) for item in completed):
        raise _fail("legacy resume strata inventory is invalid")
    completed_ids = {str(item.get("stratumId")) for item in completed}
    if (set(actual["excludedStrata"]) & completed_ids) or (
        excluded_rows and state.get("complete") is True
    ):
        raise _fail(
            "legacy completed stratum contains a work without a real author ID; "
            "restart acquisition under the audited protocol"
        )
    if actual_rows != raw_rows:
        raise _fail("legacy resume does not match the committed raw work rows")
    if accepted_rows != raw_rows - excluded_rows:
        raise _fail("legacy work-eligibility migration accounting failed")
    current_received = state.get("currentStratumReceived")
    if excluded_rows and (
        state.get("complete") is True
        or isinstance(current_received, bool)
        or not isinstance(current_received, int)
        or current_received < excluded_rows
    ):
        raise _fail("legacy current-stratum count cannot be migrated safely")
    updates: dict[str, Any] = {
        "workEligibilityProtocol": dict(WORK_ELIGIBILITY_POLICY),
        "acceptedRows": accepted_rows,
        "fetchedRows": raw_rows,
        "excludedByReason": {NO_VALID_AUTHORSHIPS_REASON: excluded_rows},
        "excludedWorkIdDigest": excluded_digest,
        "discardedAuthorshipsByReason": discarded,
    }
    if isinstance(current_received, int) and not isinstance(current_received, bool):
        updates["currentStratumReceived"] = current_received - excluded_rows
    state.update(updates)
    _work_eligibility_audit(state)
    return True


def _stable_sample_key(work_id: str, stratum_id: str) -> str:
    return hashlib.sha256(f"20260820\0{stratum_id}\0{work_id}".encode()).hexdigest()


def _strata(clusters: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Allocate exact cluster caps across year/topic strata, deterministically."""

    result: list[dict[str, Any]] = []
    for cluster in clusters:
        topics = cluster["topics"]
        cap = int(cluster["maximumWorks"])
        count = len(topics) * len(range(MIN_DATE.year, MAX_DATE.year + 1))
        quotient, remainder = divmod(cap, count)
        for year in range(MIN_DATE.year, MAX_DATE.year + 1):
            for topic in topics:
                index = len([item for item in result if item["clusterId"] == cluster["clusterId"]])
                quota = quotient + (1 if index < remainder else 0)
                if quota > MAX_SAMPLE_PER_STRATUM:
                    raise _fail("a year/topic stratum exceeds the official sample limit")
                result.append(
                    {
                        "stratumId": f"{cluster['clusterId']}:{year}:{topic['id']}",
                        "clusterId": cluster["clusterId"],
                        "year": year,
                        "topicId": topic["id"],
                        "quota": quota,
                    }
                )
    return result


def fetch_openalex(
    config: OpenAlexConfig | None,
    root: str | Path,
    *,
    transport: Transport | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Fetch the pinned corpus while excluding concurrent recovery/fetch jobs."""

    selected = config or OpenAlexConfig.pinned()
    selected.validate()
    _api_key(api_key)
    layout = RuntimeLayout.from_root(root)
    layout.raw_openalex.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(layout.raw_openalex / FETCH_LOCK_NAME):
        return _fetch_openalex_unlocked(
            selected,
            layout.root,
            transport=transport,
            api_key=api_key,
        )


def _fetch_openalex_unlocked(
    config: OpenAlexConfig | None,
    root: str | Path,
    *,
    transport: Transport | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Fetch the pinned corpus to resumable raw JSONL without exposing a key."""

    selected = config or OpenAlexConfig.pinned()
    selected.validate()
    key = _api_key(api_key)
    request = transport or _default_transport
    formal_source = transport is None
    layout = RuntimeLayout.from_root(root)
    raw = layout.raw_openalex
    raw.mkdir(parents=True, exist_ok=True)
    works_path = raw / RAW_NAME
    resume_path = raw / RESUME_NAME
    topics_path = raw / TOPICS_NAME

    resolved_clusters: list[dict[str, Any]] = []
    for cluster_id, selectors, cap in selected.clusters:
        topics = [parse_topic_selector(item, transport=request) for item in selectors]
        topic_ids = [item["id"] for item in topics]
        if len(topic_ids) != len(set(topic_ids)):
            raise _fail(f"cluster {cluster_id!r} contains duplicate resolved topics")
        resolved_clusters.append({"clusterId": cluster_id, "maximumWorks": cap, "topics": topics})
    topic_payload = {
        "schemaVersion": "gfm.openalex-topics/1.0",
        "apiBase": API_BASE,
        "resolutionProtocol": "exact-display-name-v1",
        "configHash": selected.identity,
        "formalEligible": formal_source,
        "clusters": resolved_clusters,
    }
    if topics_path.exists():
        if read_json_object(topics_path) != topic_payload:
            raise _fail("resolved topic contract changed across resume")
    else:
        atomic_write_json(topics_path, topic_payload)

    strata = _strata(resolved_clusters)
    expected_requests = sum((int(item["quota"]) + PER_PAGE - 1) // PER_PAGE for item in strata)
    if expected_requests > MAX_FETCH_REQUESTS:
        raise _fail(
            f"planned acquisition needs {expected_requests} requests, above the "
            f"{MAX_FETCH_REQUESTS}-request budget"
        )
    state: dict[str, Any] = {
        "schemaVersion": RESUME_SCHEMA,
        "configHash": selected.identity,
        "stratumIndex": 0,
        "sampleDraw": 0,
        "currentStratumReceived": 0,
        "rawRows": 0,
        "acceptedRows": 0,
        "fetchedRows": 0,
        "requests": 0,
        "strata": [],
        "complete": False,
        "formalEligible": formal_source,
        "workEligibilityProtocol": dict(WORK_ELIGIBILITY_POLICY),
        "excludedByReason": {NO_VALID_AUTHORSHIPS_REASON: 0},
        "excludedWorkIdDigest": _empty_exclusion_digest(),
        "discardedAuthorshipsByReason": {reason: 0 for reason in AUTHORSHIP_DISCARD_REASONS},
    }
    resuming = resume_path.exists()
    if resuming:
        state = read_json_object(resume_path)
        if state.get("schemaVersion") != RESUME_SCHEMA:
            raise _fail("resume state uses an obsolete or unknown sampling protocol")
        if state.get("configHash") != selected.identity:
            raise _fail("resume state belongs to a different OpenAlex config")
        if state.get("formalEligible") is not formal_source:
            raise _fail("resume source eligibility differs from the original acquisition")
        _rollback_uncommitted_raw_tail(state, works_path, strata=strata)
    migrated = _migrate_or_validate_work_eligibility_state(state, works_path)
    if migrated:
        atomic_write_json(resume_path, state)
    if state.get("complete") is True:
        audit = _work_eligibility_audit(state)
        return {
            "corpusId": CORPUS_ID,
            "rawPath": str(works_path),
            "rawSha256": file_sha256(works_path),
            "rows": int(str(state["acceptedRows"])),
            "reused": True,
            "formalEligible": formal_source,
            **audit,
        }

    start = int(str(state.get("stratumIndex", 0)))
    for index in range(start, len(strata)):
        stratum = strata[index]
        sample_draw = int(str(state.get("sampleDraw", 0))) if index == start else 0
        received = int(str(state.get("currentStratumReceived", 0))) if index == start else 0
        while received < int(str(stratum["quota"])):
            request_count = int(str(state.get("requests", 0))) + 1
            if request_count > MAX_FETCH_REQUESTS:
                raise _fail("OpenAlex request budget was exhausted")
            remaining = int(str(stratum["quota"])) - received
            draw_size = min(PER_PAGE, remaining)
            query = {
                "api_key": key,
                "filter": (
                    f"publication_year:{stratum['year']},"
                    f"type:{'|'.join(selected.work_types)},"
                    f"topics.id:{stratum['topicId']}"
                ),
                "select": ",".join(selected.work_fields),
                # OpenAlex forbids combining ``sample`` with ``page``.  Draw
                # multiple deterministic samples, as recommended by the
                # official API guide, then deduplicate and stable-hash cap in
                # ``prepare_openalex``.
                "sample": str(draw_size),
                "seed": hashlib.sha256(
                    f"{stratum['stratumId']}\0{sample_draw}".encode()
                ).hexdigest()[:16],
                "per_page": str(draw_size),
            }
            page = request(f"{API_BASE}/works", query)
            rows = page.get("results")
            meta = page.get("meta")
            if not isinstance(rows, list) or not isinstance(meta, dict):
                raise _fail("works response lacks results/meta")
            stored = []
            accepted_count = 0
            excluded_count = 0
            excluded_digest = str(state["excludedWorkIdDigest"])
            discarded = {
                reason: int(state["discardedAuthorshipsByReason"][reason])
                for reason in AUTHORSHIP_DISCARD_REASONS
            }
            for item in rows[:remaining]:
                if not isinstance(item, dict):
                    raise _fail("works response contains a non-object result")
                work = _validate_work(item, allow_empty_authorships=True)
                item_discards = _discarded_authorship_counts(item)
                for reason in AUTHORSHIP_DISCARD_REASONS:
                    discarded[reason] += item_discards[reason]
                stored.append(
                    {
                        "clusterId": stratum["clusterId"],
                        "stratumId": stratum["stratumId"],
                        # Preserve the allowlisted source object so the discarded-
                        # authorship counters can be recomputed from raw bytes.
                        # Preparation applies the normalized ``work`` value.
                        "work": dict(item),
                    }
                )
                if not work["authorships"]:
                    excluded_count += 1
                    excluded_digest = _extend_exclusion_digest(
                        excluded_digest,
                        reason=NO_VALID_AUTHORSHIPS_REASON,
                        work_id=_normalise_id(work["id"], "W"),
                    )
                    continue
                accepted_count += 1
            append_jsonl_fsync(works_path, stored)
            received += accepted_count
            sample_draw += 1
            state.update(
                {
                    "stratumIndex": index,
                    "sampleDraw": sample_draw,
                    "currentStratumReceived": received,
                    "rawRows": int(str(state["rawRows"])) + len(stored),
                    "acceptedRows": int(str(state["acceptedRows"])) + accepted_count,
                    "fetchedRows": int(str(state["fetchedRows"])) + len(stored),
                    "requests": request_count,
                    "complete": False,
                    "excludedByReason": {
                        NO_VALID_AUTHORSHIPS_REASON: int(
                            state["excludedByReason"][NO_VALID_AUTHORSHIPS_REASON]
                        )
                        + excluded_count
                    },
                    "excludedWorkIdDigest": excluded_digest,
                    "discardedAuthorshipsByReason": discarded,
                }
            )
            _work_eligibility_audit(state)
            atomic_write_json(resume_path, state)
            if not rows:
                break
        state_strata = state.get("strata")
        if not isinstance(state_strata, list):
            raise _fail("resume state strata is invalid")
        state_strata.append(
            {
                "stratumId": stratum["stratumId"],
                "requested": stratum["quota"],
                "received": received,
            }
        )
        state.update({"stratumIndex": index + 1, "sampleDraw": 0, "currentStratumReceived": 0})
        atomic_write_json(resume_path, state)
    state.update({"complete": True, "sampleDraw": 0, "stratumIndex": len(strata)})
    atomic_write_json(resume_path, state)
    return {
        "corpusId": CORPUS_ID,
        "rawPath": str(works_path),
        "rawSha256": file_sha256(works_path),
        "rows": int(str(state["acceptedRows"])),
        "reused": False,
        "formalEligible": formal_source,
        **_work_eligibility_audit(state),
    }


def fetch_historical_newcomers(
    author_ids: Sequence[str],
    from_date: date,
    to_date: date,
    *,
    transport: Transport | None = None,
    api_key: str | None = None,
    batch_size: int = 50,
) -> dict[str, date]:
    """Return earliest observed work dates using batched global author queries."""

    if not date(1900, 1, 1) <= from_date <= to_date <= MAX_DATE or not 1 <= batch_size <= 50:
        raise _fail("historical newcomer query has invalid dates or batch size")
    key = _api_key(api_key)
    request = transport or _default_transport
    normalized = tuple(sorted({_normalise_id(item, "A") for item in author_ids}))
    earliest: dict[str, date] = {}
    for offset in range(0, len(normalized), batch_size):
        batch = normalized[offset : offset + batch_size]
        cursor = "*"
        while cursor:
            page = request(
                f"{API_BASE}/works",
                {
                    "api_key": key,
                    "filter": (
                        f"author.id:{'|'.join(batch)},"
                        f"from_publication_date:{from_date.isoformat()},"
                        f"to_publication_date:{to_date.isoformat()}"
                    ),
                    "select": "publication_date,authorships",
                    "per_page": str(PER_PAGE),
                    "cursor": cursor,
                },
            )
            rows = page.get("results")
            meta = page.get("meta")
            if not isinstance(rows, list) or not isinstance(meta, dict):
                raise _fail("historical author response lacks results/meta")
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"publication_date", "authorships"}:
                    raise _fail("historical author response violated its select contract")
                work_date = date.fromisoformat(str(row["publication_date"]))
                authorships = row["authorships"]
                if not isinstance(authorships, list):
                    raise _fail("historical author response has invalid authorships")
                for authorship in authorships:
                    if not isinstance(authorship, dict):
                        continue
                    author = authorship.get("author")
                    if not isinstance(author, dict) or "id" not in author:
                        continue
                    author_id = _normalise_id(author["id"], "A")
                    if author_id in batch and (
                        author_id not in earliest or work_date < earliest[author_id]
                    ):
                        earliest[author_id] = work_date
            next_cursor = meta.get("next_cursor")
            cursor = next_cursor if isinstance(next_cursor, str) else ""
    return earliest


def _newcomer_meta(connection: sqlite3.Connection, key: str) -> str:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    if row is None or not isinstance(row[0], str):
        raise _fail(f"newcomer resume metadata is missing {key}")
    return str(row[0])


def _set_newcomer_metadata(connection: sqlite3.Connection, values: Mapping[str, str]) -> None:
    connection.executemany(
        """
        INSERT INTO metadata(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        tuple((key, value) for key, value in values.items()),
    )


def _set_newcomer_stage(connection: sqlite3.Connection, stage: str) -> None:
    _set_newcomer_metadata(connection, {"stage": stage})
    connection.commit()


def _newcomer_history_protocol_metadata() -> dict[str, str]:
    return {
        "history_query_protocol": canonical_json(NEWCOMER_HISTORY_QUERY_POLICY),
        **{key: "0" for key in NEWCOMER_HISTORY_AUDIT_KEYS},
    }


def _newcomer_history_query_audit(connection: sqlite3.Connection) -> dict[str, Any]:
    if _newcomer_meta(connection, "history_query_protocol") != canonical_json(
        NEWCOMER_HISTORY_QUERY_POLICY
    ):
        raise _fail("newcomer history query protocol is invalid")
    values: dict[str, int] = {}
    for key in NEWCOMER_HISTORY_AUDIT_KEYS:
        raw = _newcomer_meta(connection, key)
        try:
            value = int(raw)
        except ValueError as exc:
            raise _fail("newcomer history query audit is invalid") from exc
        if value < 0 or str(value) != raw:
            raise _fail("newcomer history query audit is invalid")
        values[key] = value
    raw_request_count = _newcomer_meta(connection, "request_count")
    try:
        request_count = int(raw_request_count)
    except ValueError as exc:
        raise _fail("newcomer history request accounting is invalid") from exc
    if request_count < 0 or str(request_count) != raw_request_count:
        raise _fail("newcomer history request accounting is invalid")
    if request_count != (
        values["history_group_by_requests"] + values["history_existence_requests"]
    ):
        raise _fail("newcomer history request accounting is invalid")
    batch_size = int(_newcomer_meta(connection, "batch_size"))
    minimum_root_requests, strict_worst_requests = _newcomer_request_bounds(
        connection, batch_size=batch_size
    )
    return {
        "protocol": dict(NEWCOMER_HISTORY_QUERY_POLICY),
        "requestCount": request_count,
        "minimumRootRequests": minimum_root_requests,
        "strictWorstRequests": strict_worst_requests,
        "configuredRequestBudget": MAX_NEWCOMER_REQUESTS,
        "groupByRequests": values["history_group_by_requests"],
        "existenceRequests": values["history_existence_requests"],
        "truncatedResponses": values["history_truncated_responses"],
        "extraGroupsIgnored": values["history_extra_groups_ignored"],
        "nullGroupsIgnored": values["history_null_groups_ignored"],
        "returnedGroups": values["history_returned_groups"],
        "reportedGroups": values["history_reported_groups"],
    }


def _newcomer_request_bounds(
    connection: sqlite3.Connection, *, batch_size: int
) -> tuple[int, int]:
    """Return root-request minimum and exact binary-fallback worst case."""

    if not 1 <= batch_size <= 50:
        raise _fail("newcomer verification batch_size must be in 1..50")
    minimum_root_requests = 0
    strict_worst_requests = 0
    for (raw_count,) in connection.execute("SELECT COUNT(*) FROM authors GROUP BY t0"):
        count = int(raw_count)
        full_batches, remainder = divmod(count, batch_size)
        minimum_root_requests += full_batches + int(remainder > 0)
        strict_worst_requests += full_batches * (2 * batch_size - 1)
        if remainder:
            strict_worst_requests += 2 * remainder - 1
    return minimum_root_requests, strict_worst_requests


def _migrate_or_validate_newcomer_history_protocol(
    connection: sqlite3.Connection,
) -> bool:
    names = {"history_query_protocol", *NEWCOMER_HISTORY_AUDIT_KEYS}
    present = {
        str(row[0])
        for row in connection.execute(
            f"SELECT key FROM metadata WHERE key IN ({','.join('?' for _ in names)})",
            tuple(sorted(names)),
        )
    }
    if present:
        if present != names:
            raise _fail("newcomer resume contains a partial history query audit")
        _newcomer_history_query_audit(connection)
        return False

    stage = _newcomer_meta(connection, "stage")
    request_count = _newcomer_meta(connection, "request_count")
    last_t0 = _newcomer_meta(connection, "last_t0")
    last_author = _newcomer_meta(connection, "last_author")
    prior_history = int(
        connection.execute("SELECT COALESCE(SUM(prior_history), 0) FROM authors").fetchone()[0]
    )
    if (
        stage not in {"created", "ingested", "selected", "authors_built"}
        or request_count != "0"
        or last_t0 != "-1"
        or last_author
        or prior_history != 0
    ):
        raise _fail(
            "partially executed legacy newcomer history protocol cannot be migrated safely"
        )
    _set_newcomer_metadata(connection, _newcomer_history_protocol_metadata())
    connection.commit()
    _newcomer_history_query_audit(connection)
    return True


def _newcomer_resume_schema() -> str:
    return """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE candidates (
            cluster_id TEXT NOT NULL,
            sample_key TEXT NOT NULL,
            work_id TEXT NOT NULL,
            publication_date TEXT NOT NULL,
            FOREIGN KEY(work_id) REFERENCES candidate_work(work_id)
        );
        CREATE TABLE candidate_work (
            work_id TEXT PRIMARY KEY,
            work_json TEXT NOT NULL
        );
        CREATE TABLE selected_work (
            work_id TEXT PRIMARY KEY,
            FOREIGN KEY(work_id) REFERENCES candidate_work(work_id)
        );
        CREATE TABLE authors (
            author_id TEXT PRIMARY KEY,
            t0 INTEGER NOT NULL,
            prior_history INTEGER NOT NULL DEFAULT 0
        );
        CREATE TRIGGER reject_candidate_work_conflict
        BEFORE INSERT ON candidate_work
        WHEN EXISTS (
            SELECT 1 FROM candidate_work
            WHERE work_id = NEW.work_id AND work_json <> NEW.work_json
        )
        BEGIN
            SELECT RAISE(ABORT, 'conflicting candidate work payload');
        END;
    """


def _open_newcomer_resume_database(
    database: Path,
    *,
    binding: str,
    corpus_source_hash: str,
    raw_sha256: str,
    batch_size: int,
) -> sqlite3.Connection:
    existed = database.exists()
    try:
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA foreign_keys=ON")
        if existed:
            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            if quick_check != ("ok",):
                raise sqlite3.DatabaseError(str(quick_check[0] if quick_check else "incomplete"))
            expected_tables = {
                "metadata",
                "candidates",
                "candidate_work",
                "selected_work",
                "authors",
            }
            actual_tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if actual_tables != expected_tables:
                raise _fail("newcomer resume database schema is invalid")
            actual_triggers = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                )
            }
            if actual_triggers != {"reject_candidate_work_conflict"}:
                raise _fail("newcomer resume database schema is invalid")
            expected_columns = {
                "metadata": ("key", "value"),
                "candidates": (
                    "cluster_id",
                    "sample_key",
                    "work_id",
                    "publication_date",
                ),
                "candidate_work": ("work_id", "work_json"),
                "selected_work": ("work_id",),
                "authors": ("author_id", "t0", "prior_history"),
            }
            for table, columns in expected_columns.items():
                actual_columns = tuple(
                    str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')
                )
                if actual_columns != columns:
                    raise _fail("newcomer resume database schema is invalid")
            expected = {
                "schema_version": NEWCOMER_RESUME_SCHEMA,
                "binding": binding,
                "corpus_source_hash": corpus_source_hash,
                "raw_sha256": raw_sha256,
                "batch_size": str(batch_size),
            }
            if any(_newcomer_meta(connection, key) != value for key, value in expected.items()):
                raise _fail("newcomer resume database belongs to another corpus or configuration")
            if _newcomer_meta(connection, "stage") not in {
                "created",
                "ingested",
                "selected",
                "authors_built",
                "api_complete",
            }:
                raise _fail("newcomer resume database stage is invalid")
            _migrate_or_validate_newcomer_history_protocol(connection)
            return connection
        connection.executescript(_newcomer_resume_schema())
        _set_newcomer_metadata(
            connection,
            {
                "schema_version": NEWCOMER_RESUME_SCHEMA,
                "binding": binding,
                "corpus_source_hash": corpus_source_hash,
                "raw_sha256": raw_sha256,
                "batch_size": str(batch_size),
                "stage": "created",
                "raw_rows": "0",
                "author_work_id": "",
                "request_count": "0",
                "last_t0": "-1",
                "last_author": "",
                **_newcomer_history_protocol_metadata(),
            },
        )
        connection.commit()
        return connection
    except (sqlite3.DatabaseError, ContractViolation) as exc:
        try:
            connection.close()
        except (UnboundLocalError, sqlite3.Error):
            pass
        raise _fail(
            "newcomer resume database is corrupt or incompatible; remove only this exact "
            f"staging directory after confirming no verifier is running: {database.parent}"
        ) from exc


def _ingest_newcomer_candidates(
    connection: sqlite3.Connection,
    raw_path: Path,
    cluster_caps: Mapping[str, Any],
) -> None:
    raw_rows = int(_newcomer_meta(connection, "raw_rows"))
    candidate_rows: list[tuple[str, str, str, str]] = []
    candidate_works: list[tuple[str, str]] = []
    seen_rows = 0
    for row in read_jsonl(raw_path):
        seen_rows += 1
        if seen_rows <= raw_rows:
            continue
        if set(row) != {"clusterId", "stratumId", "work"} or not isinstance(row.get("work"), dict):
            raise _fail("raw work row is invalid during newcomer verification")
        cluster_id = str(row["clusterId"])
        if cluster_id not in cluster_caps:
            raise _fail("raw newcomer candidate has an unknown cluster")
        stratum_id = str(row["stratumId"])
        if not stratum_id.startswith(cluster_id + ":"):
            raise _fail("raw newcomer candidate has a mismatched stratum")
        work = _validate_work(row["work"], allow_empty_authorships=True)
        if work["authorships"]:
            work_id = _normalise_id(work["id"], "W")
            work_json = json.dumps(
                work,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            candidate_works.append((work_id, work_json))
            candidate_rows.append(
                (
                    cluster_id,
                    _stable_sample_key(work_id, stratum_id),
                    work_id,
                    str(work["publication_date"]),
                )
            )
        if seen_rows % NEWCOMER_INGEST_COMMIT_ROWS == 0:
            _insert_newcomer_candidates(connection, candidate_rows, candidate_works)
            candidate_rows.clear()
            candidate_works.clear()
            _set_newcomer_metadata(connection, {"raw_rows": str(seen_rows)})
            connection.commit()
    _insert_newcomer_candidates(connection, candidate_rows, candidate_works)
    _set_newcomer_metadata(connection, {"raw_rows": str(seen_rows)})
    connection.commit()


def _insert_newcomer_candidates(
    connection: sqlite3.Connection,
    candidates: Sequence[tuple[str, str, str, str]],
    works: Sequence[tuple[str, str]],
) -> None:
    if not candidates:
        return
    try:
        connection.executemany("INSERT OR IGNORE INTO candidate_work VALUES (?, ?)", works)
        connection.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?)", candidates)
    except sqlite3.IntegrityError as exc:
        if "conflicting candidate work payload" in str(exc):
            raise _fail("raw candidate repeats a work ID with conflicting content") from exc
        raise _fail("raw newcomer candidate violated the resume database schema") from exc


def _upsert_newcomer_authors(
    connection: sqlite3.Connection, rows: Sequence[tuple[str, int]]
) -> None:
    connection.executemany(
        """
        INSERT INTO authors(author_id, t0) VALUES (?, ?)
        ON CONFLICT(author_id) DO UPDATE SET t0 = MIN(authors.t0, excluded.t0)
        """,
        rows,
    )


def _build_newcomer_authors(connection: sqlite3.Connection) -> None:
    last_work_id = _newcomer_meta(connection, "author_work_id")
    while True:
        work_rows = connection.execute(
            """
            SELECT selected_work.work_id, candidate_work.work_json
            FROM selected_work JOIN candidate_work USING(work_id)
            WHERE selected_work.work_id > ?
            ORDER BY selected_work.work_id LIMIT 1000
            """,
            (last_work_id,),
        ).fetchall()
        if not work_rows:
            break
        author_upserts: list[tuple[str, int]] = []
        for work_id, work_json in work_rows:
            work = _load_work_payload(str(work_json))
            work_timestamp = int(
                datetime.combine(
                    date.fromisoformat(str(work["publication_date"])),
                    datetime.min.time(),
                    tzinfo=UTC,
                ).timestamp()
            )
            author_upserts.extend((author_id, work_timestamp) for author_id in _author_ids(work))
            last_work_id = str(work_id)
        _upsert_newcomer_authors(connection, author_upserts)
        _set_newcomer_metadata(connection, {"author_work_id": last_work_id})
        connection.commit()
    connection.execute("CREATE INDEX IF NOT EXISTS authors_t0_order ON authors(t0, author_id)")
    _set_newcomer_stage(connection, "authors_built")


def _reserve_newcomer_history_request(
    connection: sqlite3.Connection, *, kind: str
) -> int:
    key_by_kind = {
        "group_by": "history_group_by_requests",
        "existence": "history_existence_requests",
    }
    if kind not in key_by_kind:
        raise _fail("newcomer history request kind is invalid")
    audit = _newcomer_history_query_audit(connection)
    request_count = int(audit["requestCount"]) + 1
    if request_count > MAX_NEWCOMER_REQUESTS:
        raise _fail(
            "newcomer history query exhausted its fixed request budget: "
            f"actual={audit['requestCount']}, "
            f"minimumRootRequests={audit['minimumRootRequests']}, "
            f"strictWorstRequests={audit['strictWorstRequests']}, "
            f"configuredRequestBudget={audit['configuredRequestBudget']}"
        )
    key = key_by_kind[kind]
    next_kind_count = int(_newcomer_meta(connection, key)) + 1
    _set_newcomer_metadata(
        connection,
        {"request_count": str(request_count), key: str(next_kind_count)},
    )
    # Reserve before network I/O.  A failed or interrupted request consumes the
    # reservation, which makes the persistent budget a conservative upper bound.
    connection.commit()
    _newcomer_history_query_audit(connection)
    return request_count


def _increment_newcomer_history_audit(
    connection: sqlite3.Connection, *, key: str, amount: int
) -> None:
    if key not in {
        "history_truncated_responses",
        "history_extra_groups_ignored",
        "history_null_groups_ignored",
        "history_returned_groups",
        "history_reported_groups",
    } or isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
        raise _fail("newcomer history audit update is invalid")
    if amount == 0:
        return
    value = int(_newcomer_meta(connection, key)) + amount
    _set_newcomer_metadata(connection, {key: str(value)})
    connection.commit()
    _newcomer_history_query_audit(connection)


def _newcomer_author_has_prior_work(
    connection: sqlite3.Connection,
    *,
    author_id: str,
    cutoff: date,
    request: Transport,
    api_key: str,
) -> bool:
    _reserve_newcomer_history_request(connection, kind="existence")
    response = request(
        f"{API_BASE}/works",
        {
            "api_key": api_key,
            "filter": (
                f"authorships.author.id:{author_id},"
                f"to_publication_date:{cutoff.isoformat()}"
            ),
            "select": "id",
            "per_page": "1",
        },
    )
    rows = response.get("results")
    meta = response.get("meta")
    if not isinstance(rows, list) or not isinstance(meta, dict):
        raise _fail("newcomer existence response lacks results/meta")
    count = meta.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise _fail("newcomer existence response has an invalid count")
    if len(rows) > 1 or (count == 0) != (len(rows) == 0):
        raise _fail("newcomer existence response is inconsistent")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id"}:
            raise _fail("newcomer existence response violated its select contract")
        _normalise_id(row["id"], "W")
    return count > 0


def _resolve_newcomer_prior_history(
    connection: sqlite3.Connection,
    *,
    batch: Sequence[str],
    cutoff: date,
    request: Transport,
    api_key: str,
) -> set[str]:
    requested = tuple(sorted({_normalise_id(value, "A") for value in batch}))
    if not requested or len(requested) != len(batch) or len(requested) > 50:
        raise _fail("newcomer history resolution batch is invalid")
    if len(requested) == 1:
        author_id = requested[0]
        return (
            {author_id}
            if _newcomer_author_has_prior_work(
                connection,
                author_id=author_id,
                cutoff=cutoff,
                request=request,
                api_key=api_key,
            )
            else set()
        )

    _reserve_newcomer_history_request(connection, kind="group_by")
    response = request(
        f"{API_BASE}/works",
        {
            "api_key": api_key,
            "filter": (
                f"authorships.author.id:{'|'.join(requested)},"
                f"to_publication_date:{cutoff.isoformat()}"
            ),
            "group_by": "authorships.author.id",
            "per_page": str(NEWCOMER_GROUP_BY_PAGE_SIZE),
            "cursor": "*",
        },
    )
    groups = response.get("group_by")
    meta = response.get("meta")
    if not isinstance(groups, list) or not isinstance(meta, dict):
        raise _fail("newcomer group_by response is incomplete")
    groups_count = meta.get("groups_count")
    if groups_count is not None and (
        isinstance(groups_count, bool)
        or not isinstance(groups_count, int)
        or groups_count < 0
    ):
        raise _fail("newcomer group_by response has an invalid groups_count")
    if "next_cursor" not in meta:
        raise _fail("newcomer group_by response has no next_cursor")
    next_cursor = meta["next_cursor"]
    if next_cursor is not None and (
        not isinstance(next_cursor, str) or not next_cursor
    ):
        raise _fail("newcomer group_by response has an invalid next_cursor")
    if len(groups) > NEWCOMER_GROUP_BY_PAGE_SIZE:
        raise _fail("newcomer group_by response exceeds its requested page size")
    _increment_newcomer_history_audit(
        connection, key="history_returned_groups", amount=len(groups)
    )
    if isinstance(groups_count, int) and not isinstance(groups_count, bool):
        _increment_newcomer_history_audit(
            connection, key="history_reported_groups", amount=groups_count
        )
    returned: set[str] = set()
    null_groups = 0
    for group in groups:
        if not isinstance(group, dict) or "key" not in group:
            raise _fail("newcomer group_by item is invalid")
        count = group.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise _fail("newcomer group_by item count is invalid")
        key = group["key"]
        if key is None:
            null_groups += 1
            continue
        if not isinstance(key, str):
            raise _fail("newcomer group_by item key is invalid")
        author_id = _normalise_id(key, "A")
        if author_id == NULL_AUTHOR_ID:
            null_groups += 1
            continue
        if author_id in returned:
            raise _fail("newcomer group_by returned a duplicate author")
        returned.add(author_id)

    requested_set = set(requested)
    confirmed = returned.intersection(requested_set)
    extras = returned.difference(requested_set)
    _increment_newcomer_history_audit(
        connection, key="history_extra_groups_ignored", amount=len(extras)
    )
    _increment_newcomer_history_audit(
        connection, key="history_null_groups_ignored", amount=null_groups
    )
    if next_cursor is None:
        # Only an explicit JSON null continuation cursor proves that the page
        # is complete.  Missing/empty cursors fail closed; groups_count is
        # telemetry and may differ from the returned page length legitimately.
        return confirmed

    _increment_newcomer_history_audit(
        connection, key="history_truncated_responses", amount=1
    )
    unresolved = tuple(sorted(requested_set.difference(confirmed)))
    if not unresolved:
        return confirmed
    midpoint = len(unresolved) // 2
    partitions = (unresolved[:midpoint], unresolved[midpoint:]) if midpoint else (unresolved,)
    for partition in partitions:
        if partition:
            confirmed.update(
                _resolve_newcomer_prior_history(
                    connection,
                    batch=partition,
                    cutoff=cutoff,
                    request=request,
                    api_key=api_key,
                )
            )
    return confirmed


def _verify_newcomer_t0_batches(
    connection: sqlite3.Connection,
    *,
    t0_value: int,
    after_author: str,
    batch_size: int,
    request: Transport,
    api_key: str,
) -> None:
    cutoff = datetime.fromtimestamp(t0_value, tz=UTC).date() - timedelta(days=1)
    while True:
        rows = connection.execute(
            "SELECT author_id FROM authors WHERE t0 = ? AND author_id > ? "
            "ORDER BY author_id LIMIT ?",
            (t0_value, after_author, batch_size),
        ).fetchall()
        if not rows:
            return
        batch = [str(row[0]) for row in rows]
        returned = _resolve_newcomer_prior_history(
            connection,
            batch=batch,
            cutoff=cutoff,
            request=request,
            api_key=api_key,
        )
        connection.executemany(
            "UPDATE authors SET prior_history = 1 WHERE author_id = ?",
            ((author_id,) for author_id in sorted(returned)),
        )
        after_author = batch[-1]
        _set_newcomer_metadata(
            connection,
            {
                "last_t0": str(t0_value),
                "last_author": after_author,
            },
        )
        connection.commit()


def _remove_owned_resume_directory(path: Path, parent: Path, *, binding: str) -> None:
    resolved_parent = parent.resolve()
    resolved = path.resolve()
    expected_name = f"{NEWCOMER_RESUME_PREFIX}{binding}{NEWCOMER_RESUME_SUFFIX}"
    if resolved.parent != resolved_parent or path.name != expected_name:
        raise _fail("refused to remove an unowned newcomer resume directory")
    if path.exists():
        shutil.rmtree(path)


def _newcomer_resume_identity(
    output: Path, manifest: Mapping[str, Any], *, batch_size: int
) -> tuple[str, str, Path]:
    source = manifest.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("rawSha256"), str):
        raise _fail("OpenAlex corpus source identity is invalid for newcomer verification")
    source_hash = canonical_sha256(source)
    binding = canonical_sha256(
        {
            "schemaVersion": NEWCOMER_RESUME_SCHEMA,
            "corpusSourceHash": source_hash,
            "rawSha256": source["rawSha256"],
            "batchSize": batch_size,
        }
    )
    directory = output.parent / (f"{NEWCOMER_RESUME_PREFIX}{binding}{NEWCOMER_RESUME_SUFFIX}")
    return binding, source_hash, directory


def _newcomer_overlay_directory(root: str | Path) -> Path:
    """Return the standalone product-label artifact boundary.

    The verified history labels are deliberately a sibling of the immutable
    OpenAlex graph corpus.  Consequently publishing, corrupting, or removing
    this optional overlay cannot change the graph corpus manifest or its
    portable identity.
    """

    return RuntimeLayout.from_root(root).processed_gfm / NEWCOMER_OVERLAY_ID


def _legacy_newcomer_overlay_directory(root: str | Path) -> Path:
    return (
        RuntimeLayout.from_root(root).processed_gfm
        / CORPUS_ID
        / "newcomer-verification"
    )


def _reject_legacy_nested_newcomer_overlay(root: str | Path) -> None:
    legacy = _legacy_newcomer_overlay_directory(root)
    if legacy.exists():
        raise _fail(
            "legacy nested newcomer overlay is unsupported; preserve it for audit, "
            f"then explicitly migrate or rebuild it as sibling {NEWCOMER_OVERLAY_ID!r}"
        )


def _newcomer_base_binding(manifest: Mapping[str, Any]) -> dict[str, str]:
    source = manifest.get("source")
    logical_hash = manifest.get("logicalHash")
    if not isinstance(source, dict) or not isinstance(logical_hash, str):
        raise _fail("OpenAlex base corpus identity is invalid for newcomer verification")
    raw_sha256 = source.get("rawSha256")
    if not isinstance(raw_sha256, str):
        raise _fail("OpenAlex base corpus raw identity is invalid for newcomer verification")
    source_hash = canonical_sha256(source)
    return {
        "baseCorpusId": CORPUS_ID,
        "baseCorpusLogicalHash": logical_hash,
        "baseCorpusSourceHash": source_hash,
        # Retained as a compatibility spelling for operator-facing evidence.
        "corpusSourceHash": source_hash,
        "rawSha256": raw_sha256,
    }


def _validate_newcomer_base_binding(
    overlay: Mapping[str, Any], base: Mapping[str, Any]
) -> None:
    source = overlay.get("source")
    expected = _newcomer_base_binding(base)
    if not isinstance(source, dict) or any(
        source.get(name) != value for name, value in expected.items()
    ):
        raise _fail("newcomer overlay is bound to another OpenAlex base corpus")


def _read_openalex_base_manifest_identity(root: str | Path) -> dict[str, Any]:
    """Authenticate the base identity without opening role-restricted arrays."""

    _reject_legacy_nested_newcomer_overlay(root)
    directory = RuntimeLayout.from_root(root).processed_gfm / CORPUS_ID
    manifest = read_json_object(directory / "manifest.json")
    logical_hash = manifest.get("logicalHash")
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"logicalHash", "createdAt"}
    }
    shards = manifest.get("shards")
    if (
        manifest.get("schemaVersion") != "gfm.openalex-corpus/1.0"
        or manifest.get("corpusId") != CORPUS_ID
        or not isinstance(logical_hash, str)
        or logical_hash != canonical_sha256(payload)
        or not isinstance(shards, list)
        or "newcomerVerification" in manifest
        or any(
            isinstance(record, dict)
            and str(record.get("path", "")).startswith("newcomer-verification/")
            for record in shards
        )
    ):
        raise _fail("OpenAlex base manifest identity is invalid or contains a legacy overlay")
    return manifest


def _remove_completed_newcomer_resume(
    output: Path, manifest: Mapping[str, Any], *, batch_size: int
) -> None:
    binding, source_hash, directory = _newcomer_resume_identity(
        output, manifest, batch_size=batch_size
    )
    if not directory.exists():
        return
    connection = _open_newcomer_resume_database(
        directory / "newcomer-verification.sqlite3",
        binding=binding,
        corpus_source_hash=source_hash,
        raw_sha256=str(manifest["source"]["rawSha256"]),
        batch_size=batch_size,
    )
    try:
        if _newcomer_meta(connection, "stage") != "api_complete":
            raise _fail("published newcomer overlay has an incomplete resume database")
    finally:
        connection.close()
    _remove_owned_resume_directory(directory, output.parent, binding=binding)


def verify_openalex_newcomers(
    root: str | Path,
    *,
    transport: Transport | None = None,
    api_key: str | None = None,
    batch_size: int = 50,
) -> dict[str, Any]:
    """Verify newcomers while excluding concurrent verifier processes."""

    layout = RuntimeLayout.from_root(root)
    layout.processed_gfm.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(layout.processed_gfm / NEWCOMER_VERIFY_LOCK_NAME):
        return _verify_openalex_newcomers_unlocked(
            layout.root,
            transport=transport,
            api_key=api_key,
            batch_size=batch_size,
        )


def _verify_openalex_newcomers_unlocked(
    root: str | Path,
    *,
    transport: Transport | None = None,
    api_key: str | None = None,
    batch_size: int = 50,
) -> dict[str, Any]:
    """Verify candidate first works globally and publish a hash-bound overlay.

    Raw OpenAlex author IDs are re-read from the accepted raw JSONL into an
    ephemeral on-disk SQLite index used only for the API lookup.  Selection is
    reproduced exactly so excluded raw candidates cannot create phantom local
    authors.  The overlay contains local author indices, dates and labels; it
    does not persist source identifiers, raw work objects, or the API key.
    """

    layout = RuntimeLayout.from_root(root)
    if not 1 <= batch_size <= 50:
        raise _fail("newcomer verification batch_size must be in 1..50")
    raw = layout.raw_openalex
    output = layout.processed_gfm / CORPUS_ID
    _reject_legacy_nested_newcomer_overlay(layout.root)
    overlay_directory = _newcomer_overlay_directory(layout.root)
    if overlay_directory.exists():
        base = check_openalex(root)
        checked = check_openalex_newcomers(root)
        _remove_completed_newcomer_resume(output, base, batch_size=batch_size)
        return checked
    manifest = check_openalex(root)
    raw_path = raw / RAW_NAME
    if file_sha256(raw_path) != manifest.get("source", {}).get("rawSha256"):
        raise _fail("accepted raw corpus changed before newcomer verification")
    binding, source_hash, resume_directory = _newcomer_resume_identity(
        output, manifest, batch_size=batch_size
    )
    resume_directory.mkdir(parents=False, exist_ok=True)
    database = resume_directory / "newcomer-verification.sqlite3"
    connection = _open_newcomer_resume_database(
        database,
        binding=binding,
        corpus_source_hash=source_hash,
        raw_sha256=str(manifest["source"]["rawSha256"]),
        batch_size=batch_size,
    )
    published = False
    try:
        cluster_caps = manifest.get("clusterCounts")
        topic_clusters = read_json_object(raw / TOPICS_NAME).get("clusters", [])
        topic_cluster_ids = [
            str(item["clusterId"])
            for item in topic_clusters
            if isinstance(item, dict) and isinstance(item.get("clusterId"), str)
        ]
        if (
            not isinstance(cluster_caps, dict)
            or len(topic_cluster_ids) != len(topic_clusters)
            or len(topic_cluster_ids) != len(set(topic_cluster_ids))
            or set(cluster_caps) != set(topic_cluster_ids)
        ):
            raise _fail("processed cluster order changed before newcomer verification")
        stage = _newcomer_meta(connection, "stage")
        if stage == "created":
            _ingest_newcomer_candidates(connection, raw_path, cluster_caps)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS candidates_order "
                "ON candidates(cluster_id, sample_key, work_id)"
            )
            _set_newcomer_stage(connection, "ingested")

        if _newcomer_meta(connection, "stage") == "ingested":
            for cluster_id in topic_cluster_ids:
                selected_count = int(cluster_caps[cluster_id])
                inserted_count = 0
                for (work_id,) in connection.execute(
                    """
                    SELECT work_id FROM candidates
                    WHERE cluster_id = ?
                    ORDER BY sample_key, work_id
                    """,
                    (str(cluster_id),),
                ):
                    inserted = connection.execute(
                        "INSERT OR IGNORE INTO selected_work VALUES (?)", (work_id,)
                    ).rowcount
                    if inserted:
                        inserted_count += 1
                        if inserted_count == selected_count:
                            break
                if inserted_count != selected_count:
                    raise _fail("raw candidates no longer reproduce processed cluster counts")
            _set_newcomer_stage(connection, "selected")
        selected_work_count = int(
            connection.execute("SELECT COUNT(*) FROM selected_work").fetchone()[0]
        )
        if selected_work_count != int(manifest["nodeCounts"]["work"]):
            raise _fail("selected works no longer reproduce processed work nodes")

        if _newcomer_meta(connection, "stage") == "selected":
            _build_newcomer_authors(connection)
        author_count = int(connection.execute("SELECT COUNT(*) FROM authors").fetchone()[0])
        if author_count <= 0 or author_count != int(manifest["nodeCounts"]["author"]):
            raise _fail("selected authors no longer reproduce processed author nodes")
        minimum_root_requests, strict_worst_requests = _newcomer_request_bounds(
            connection, batch_size=batch_size
        )
        if minimum_root_requests > MAX_NEWCOMER_REQUESTS:
            raise _fail(
                "newcomer history verification root requests exceed the fixed budget: "
                "actual=0, "
                f"minimumRootRequests={minimum_root_requests}, "
                f"strictWorstRequests={strict_worst_requests}, "
                f"configuredRequestBudget={MAX_NEWCOMER_REQUESTS}; no request was made"
            )

        request_count = int(_newcomer_meta(connection, "request_count"))
        if _newcomer_meta(connection, "stage") != "api_complete":
            key = _api_key(api_key)
            request = transport or _default_transport
            last_t0 = int(_newcomer_meta(connection, "last_t0"))
            last_author = _newcomer_meta(connection, "last_author")
            for (t0_value,) in connection.execute(
                "SELECT DISTINCT t0 FROM authors WHERE t0 >= ? ORDER BY t0", (last_t0,)
            ).fetchall():
                after_author = last_author if int(t0_value) == last_t0 else ""
                _verify_newcomer_t0_batches(
                    connection,
                    t0_value=int(t0_value),
                    after_author=after_author,
                    batch_size=batch_size,
                    request=request,
                    api_key=key,
                )
                last_t0 = int(t0_value)
                last_author = _newcomer_meta(connection, "last_author")
            _set_newcomer_stage(connection, "api_complete")
            request_count = int(_newcomer_meta(connection, "request_count"))
        history_query_audit = _newcomer_history_query_audit(connection)

        t0 = np.empty(author_count, dtype=np.int64)
        true_newcomer = np.empty(author_count, dtype=np.bool_)
        author_cursor = connection.execute(
            "SELECT t0, prior_history FROM authors ORDER BY author_id"
        )
        offset = 0
        while rows := author_cursor.fetchmany(10_000):
            end = offset + len(rows)
            t0[offset:end] = np.asarray([row[0] for row in rows], dtype=np.int64)
            true_newcomer[offset:end] = np.asarray(
                [not bool(row[1]) for row in rows], dtype=np.bool_
            )
            offset = end
        if offset != author_count:
            raise _fail("newcomer verification author extraction was incomplete")

        prepared_offset = 0
        for record in _family_records(manifest, "newcomers"):
            loaded = load_npz_safe(
                resolve_within(output, str(record["path"])), expected=NEWCOMER_ARRAYS
            )
            end = prepared_offset + int(record["rows"])
            if (
                not np.array_equal(
                    loaded["author"],
                    np.arange(prepared_offset, end, dtype=np.int64),
                )
                or not np.array_equal(loaded["t0"], t0[prepared_offset:end])
                or bool(loaded["history_verified"].any())
            ):
                raise _fail("newcomer verification no longer aligns with prepared authors")
            prepared_offset = end
        if prepared_offset != author_count:
            raise _fail("prepared newcomer shards have an incomplete author count")

        connection.close()
        publish_directory = (
            output.parent
            / f".{NEWCOMER_OVERLAY_ID}-publish-{uuid.uuid4().hex}.tmp"
        )
        publish_directory.mkdir(parents=False)
        temporary_directory = publish_directory
        overlay = temporary_directory / "artifact.npz"
        from .common import atomic_write_npz, array_inventory

        verified = np.ones(author_count, dtype=np.bool_)
        arrays = {
            "author": np.arange(author_count, dtype=np.int64),
            "t0": t0,
            "history_verified": verified,
            "true_newcomer": true_newcomer,
        }
        atomic_write_npz(overlay, arrays)
        overlay_record = ShardRecord(
            path="artifact.npz",
            sha256=file_sha256(overlay),
            rows=author_count,
            arrays=tuple(array_inventory(arrays)),
        )
        newcomer_role_years = {
            "train": (None, 2020),
            "validation": (2020, 2021),
            "test": (2021, 2022),
            "shadow": (2022, None),
        }
        role_records: dict[str, ShardRecord] = {}
        for role in ACCESS_ROLES:
            lower_year, upper_year = newcomer_role_years[role]
            role_mask = np.ones(author_count, dtype=np.bool_)
            if lower_year is not None:
                role_mask &= t0 > int(
                    datetime(lower_year, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()
                )
            if upper_year is not None:
                role_mask &= t0 <= int(
                    datetime(upper_year, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()
                )
            role_records[role] = NumericShardWriter(
                temporary_directory,
                prefix=f"rv-n-{ACCESS_ROLE_TAGS[role]}",
                rows_per_shard=max(int(role_mask.sum()), 1),
            ).write({name: value[role_mask] for name, value in arrays.items()})
        overlay_manifest = build_manifest(
            schema_version=NEWCOMER_OVERLAY_SCHEMA,
            corpus_id=NEWCOMER_OVERLAY_ID,
            license_id=LICENSE_ID,
            source={
                **_newcomer_base_binding(manifest),
                "query": "adaptive-per-candidate-t0-minus-one-day-author-history",
            },
            shards=(overlay_record, *(role_records[role] for role in ACCESS_ROLES)),
            splits={"labels": "global-pre-2016-history-vs-local-2016-2025-t0"},
            privacy={
                "sourceAuthorIdsPersisted": False,
                "rawWorksPersisted": False,
                "apiKeyPersisted": False,
            },
            extra={
                "authorCount": author_count,
                "verifiedCount": int(verified.sum()),
                "trueNewcomerCount": int(true_newcomer.sum()),
                "requestCount": request_count,
                "minimumRootRequests": minimum_root_requests,
                "strictWorstRequests": strict_worst_requests,
                "configuredRequestBudget": MAX_NEWCOMER_REQUESTS,
                "historyQueryProtocol": dict(NEWCOMER_HISTORY_QUERY_POLICY),
                "historyQueryAudit": history_query_audit,
                "selectionStore": "ephemeral-sqlite-not-published",
                "physicalAccess": {
                    "schemaVersion": PHYSICAL_ACCESS_SCHEMA,
                    "roles": list(ACCESS_ROLES),
                    "roleShards": {role: [role_records[role].path] for role in ACCESS_ROLES},
                    "cohortSemantics": ("train<=2020, validation=2021, test=2022, shadow>=2023"),
                },
            },
        )
        overlay_manifest_path = temporary_directory / "manifest.json"
        atomic_write_json(overlay_manifest_path, overlay_manifest)
        verify_manifest(temporary_directory, overlay_manifest)
        os.replace(temporary_directory, overlay_directory)
        published = True
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass
        if "temporary_directory" in locals() and temporary_directory.exists():
            _remove_owned_staging_directory(
                temporary_directory,
                output.parent,
                prefix=f"{NEWCOMER_OVERLAY_ID}-publish",
            )
    if not published:
        raise _fail("newcomer verification overlay was not published")
    check_openalex_newcomers(root)
    _remove_owned_resume_directory(resume_directory, output.parent, binding=binding)
    return check_openalex_newcomers(root)


def check_openalex_newcomers(root: str | Path) -> dict[str, Any]:
    _reject_legacy_nested_newcomer_overlay(root)
    corpus = check_openalex(root)
    overlay_directory = _newcomer_overlay_directory(root)
    manifest = read_json_object(overlay_directory / "manifest.json")
    _validate_newcomer_base_binding(manifest, corpus)
    node_counts = corpus.get("nodeCounts")
    expected_authors = node_counts.get("author") if isinstance(node_counts, dict) else None
    if (
        manifest.get("schemaVersion") != NEWCOMER_OVERLAY_SCHEMA
        or manifest.get("corpusId") != NEWCOMER_OVERLAY_ID
        or manifest.get("licenseId") != LICENSE_ID
        or manifest.get("verifiedCount") != manifest.get("authorCount")
        or manifest.get("authorCount") != expected_authors
    ):
        raise _fail("newcomer verification is absent, incomplete or bound to another corpus")
    history_audit = manifest.get("historyQueryAudit")
    if (
        manifest.get("historyQueryProtocol") != NEWCOMER_HISTORY_QUERY_POLICY
        or not isinstance(history_audit, dict)
        or history_audit.get("protocol") != NEWCOMER_HISTORY_QUERY_POLICY
        or history_audit.get("requestCount") != manifest.get("requestCount")
        or history_audit.get("minimumRootRequests")
        != manifest.get("minimumRootRequests")
        or history_audit.get("strictWorstRequests")
        != manifest.get("strictWorstRequests")
        or history_audit.get("configuredRequestBudget")
        != manifest.get("configuredRequestBudget")
        or any(
            isinstance(history_audit.get(name), bool)
            or not isinstance(history_audit.get(name), int)
            or int(history_audit[name]) < 0
            for name in (
                "requestCount",
                "minimumRootRequests",
                "strictWorstRequests",
                "configuredRequestBudget",
                "groupByRequests",
                "existenceRequests",
                "truncatedResponses",
                "extraGroupsIgnored",
                "nullGroupsIgnored",
                "returnedGroups",
                "reportedGroups",
            )
        )
        or history_audit.get("requestCount")
        != int(history_audit.get("groupByRequests", -1))
        + int(history_audit.get("existenceRequests", -1))
        or int(history_audit.get("configuredRequestBudget", -1))
        != MAX_NEWCOMER_REQUESTS
        or int(history_audit.get("requestCount", -1))
        > int(history_audit.get("configuredRequestBudget", -1))
        or int(history_audit.get("minimumRootRequests", -1))
        > int(history_audit.get("configuredRequestBudget", -1))
        or int(history_audit.get("minimumRootRequests", -1))
        > int(history_audit.get("strictWorstRequests", -1))
        or int(history_audit.get("configuredRequestBudget", -1)) <= 0
    ):
        raise _fail("newcomer history query audit is invalid")
    verify_manifest(overlay_directory, manifest)
    records = manifest.get("shards")
    access = manifest.get("physicalAccess")
    if (
        not isinstance(records, list)
        or len(records) != 1 + len(ACCESS_ROLES)
        or not isinstance(access, dict)
        or access.get("schemaVersion") != PHYSICAL_ACCESS_SCHEMA
        or access.get("roles") != list(ACCESS_ROLES)
        or access.get("cohortSemantics") != "train<=2020, validation=2021, test=2022, shadow>=2023"
        or not isinstance(access.get("roleShards"), dict)
        or set(access["roleShards"]) != set(ACCESS_ROLES)
    ):
        raise _fail("newcomer physical role-view contract is invalid")
    records_by_path = {
        str(record["path"]): record for record in records if isinstance(record, dict)
    }
    artifact_record = records_by_path.get("artifact.npz")
    if not isinstance(artifact_record, dict):
        raise _fail("newcomer canonical artifact is absent")
    canonical = _load_newcomer_record(overlay_directory, artifact_record)
    seen_authors: set[int] = set()
    role_years = {
        "train": (None, 2020),
        "validation": (2020, 2021),
        "test": (2021, 2022),
        "shadow": (2022, None),
    }
    for role in ACCESS_ROLES:
        paths = access["roleShards"][role]
        if paths != [f"rv-n-{ACCESS_ROLE_TAGS[role]}-00000.npz"]:
            raise _fail("newcomer physical role shard path is invalid")
        record = records_by_path.get(paths[0])
        if not isinstance(record, dict):
            raise _fail("newcomer physical role shard is undeclared")
        actual = _load_newcomer_record(overlay_directory, record)
        lower_year, upper_year = role_years[role]
        mask = np.ones(canonical["t0"].shape, dtype=np.bool_)
        if lower_year is not None:
            mask &= canonical["t0"] > int(
                datetime(lower_year, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()
            )
        if upper_year is not None:
            mask &= canonical["t0"] <= int(
                datetime(upper_year, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()
            )
        if any(not np.array_equal(actual[name], canonical[name][mask]) for name in canonical):
            raise _fail(f"newcomer physical role {role} differs from canonical cohort")
        if seen_authors.intersection(int(value) for value in actual["author"]):
            raise _fail("newcomer author belongs to more than one physical role")
        seen_authors.update(int(value) for value in actual["author"])
    if seen_authors != {int(value) for value in canonical["author"]}:
        raise _fail("newcomer physical roles do not cover canonical authors exactly once")
    return manifest


def _load_newcomer_record(directory: Path, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    raw_arrays = record.get("arrays")
    if not isinstance(raw_arrays, list):
        raise _fail("newcomer overlay array inventory is invalid")
    return load_npz_safe(
        resolve_within(directory, str(record["path"])),
        expected={
            str(item["name"]): (str(item["dtype"]), len(item["shape"]))
            for item in raw_arrays
            if isinstance(item, dict)
        },
    )


def load_openalex_newcomers(root: str | Path) -> dict[str, np.ndarray]:
    """Load the hash-bound, globally verified newcomer overlay."""

    output = _newcomer_overlay_directory(root)
    manifest = check_openalex_newcomers(root)
    records = manifest.get("shards")
    if not isinstance(records, list):
        raise _fail("newcomer overlay shard inventory is invalid")
    record = next(
        (item for item in records if isinstance(item, dict) and item.get("path") == "artifact.npz"),
        None,
    )
    if not isinstance(record, dict):
        raise _fail("newcomer canonical overlay is absent")
    loaded = _load_newcomer_record(output, record)
    expected_names = {"author", "t0", "history_verified", "true_newcomer"}
    if set(loaded) != expected_names:
        raise _fail("newcomer overlay array set is invalid")
    count = int(manifest["authorCount"])
    if any(value.shape != (count,) for value in loaded.values()):
        raise _fail("newcomer overlay arrays are misaligned")
    if not np.array_equal(loaded["author"], np.arange(count, dtype=np.int64)):
        raise _fail("newcomer overlay author indices are not canonical")
    if not bool(loaded["history_verified"].all()):
        raise _fail("newcomer overlay contains unverified history")
    return loaded


def load_openalex_newcomers_view(
    root: str | Path,
    *,
    maximum_role: str,
) -> dict[str, Any]:
    """Load only authorised newcomer cohorts without touching future files."""

    if maximum_role not in ACCESS_ROLES:
        raise _fail("newcomer maximum role is invalid")
    base = _read_openalex_base_manifest_identity(root)
    directory = _newcomer_overlay_directory(root)
    manifest = read_json_object(directory / "manifest.json")
    _validate_newcomer_base_binding(manifest, base)
    logical_hash = manifest.get("logicalHash")
    payload = {
        key: value for key, value in manifest.items() if key not in {"logicalHash", "createdAt"}
    }
    access = manifest.get("physicalAccess")
    node_counts = base.get("nodeCounts")
    expected_authors = node_counts.get("author") if isinstance(node_counts, dict) else None
    if (
        manifest.get("schemaVersion") != NEWCOMER_OVERLAY_SCHEMA
        or manifest.get("corpusId") != NEWCOMER_OVERLAY_ID
        or manifest.get("authorCount") != expected_authors
        or manifest.get("verifiedCount") != expected_authors
        or not isinstance(logical_hash, str)
        or logical_hash != canonical_sha256(payload)
        or not isinstance(access, dict)
        or access.get("schemaVersion") != PHYSICAL_ACCESS_SCHEMA
        or access.get("roles") != list(ACCESS_ROLES)
    ):
        raise _fail("newcomer role-view manifest identity is invalid")
    records = {
        str(record["path"]): record
        for record in manifest.get("shards", [])
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    maximum_index = ACCESS_ROLES.index(maximum_role)
    opened: list[str] = []
    pieces: dict[str, list[np.ndarray]] = defaultdict(list)
    for role in ACCESS_ROLES[: maximum_index + 1]:
        for path in access["roleShards"][role]:
            record = records.get(path)
            if not isinstance(record, dict):
                raise _fail("newcomer selected role shard is undeclared")
            artifact = resolve_within(directory, path)
            if file_sha256(artifact) != record.get("sha256"):
                raise _fail(f"newcomer selected role artifact hash mismatch: {path}")
            loaded = _load_newcomer_record(directory, record)
            for name, value in loaded.items():
                pieces[name].append(value)
            opened.append(path)
    arrays = {
        name: np.concatenate(values) if len(values) > 1 else values[0]
        for name, values in pieces.items()
    }
    future_roles = ACCESS_ROLES[maximum_index + 1 :]
    future_paths = {path for role in future_roles for path in access["roleShards"][role]}
    return {
        "manifest": manifest,
        "arrays": arrays,
        "accessAudit": {
            "schemaVersion": "gfm.newcomer-role-access-audit/1.0",
            "maximumRole": maximum_role,
            "openedPaths": opened,
            "futurePathsOpened": False,
            "testArtifactsOpened": any(path in future_paths for path in opened),
            "manifestLogicalHash": logical_hash,
        },
    }


def newcomer_overlay_status(root: str | Path) -> dict[str, Any]:
    """Return non-mutating task-asset readiness for operator workflows."""

    layout = RuntimeLayout.from_root(root)
    directory = _newcomer_overlay_directory(layout.root)
    legacy = _legacy_newcomer_overlay_directory(layout.root)
    resume_present = any(
        path.is_dir()
        for path in layout.processed_gfm.glob(
            f"{NEWCOMER_RESUME_PREFIX}*{NEWCOMER_RESUME_SUFFIX}"
        )
    )
    status: dict[str, Any] = {
        "schemaVersion": "gfm.openalex-newcomer-overlay-status/1.0",
        "overlayId": NEWCOMER_OVERLAY_ID,
        "ready": False,
        "state": "absent",
        "manifestHash": None,
        "baseCorpusLogicalHash": None,
        "baseCorpusSourceHash": None,
        "resumePresent": resume_present,
        "reason": "globally verified OpenAlex newcomer overlay is absent",
    }
    if legacy.exists():
        status.update(
            {
                "state": "legacy-nested-rejected",
                "reason": (
                    "legacy nested newcomer overlay requires explicit migration or rebuild"
                ),
            }
        )
        return status
    if not directory.exists():
        return status
    try:
        manifest = check_openalex_newcomers(layout.root)
        source = manifest["source"]
        status.update(
            {
                "ready": True,
                "state": "ready",
                "manifestHash": manifest["logicalHash"],
                "baseCorpusLogicalHash": source["baseCorpusLogicalHash"],
                "baseCorpusSourceHash": source["baseCorpusSourceHash"],
                "verifiedCount": int(manifest["verifiedCount"]),
                "reason": None,
            }
        )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        status.update({"state": "invalid", "reason": str(error)})
    return status


def _author_ids(work: Mapping[str, Any]) -> list[str]:
    result = []
    for item in work["authorships"]:
        if not isinstance(item, dict) or not isinstance(item.get("author"), dict):
            raise _fail("work contains malformed authorship")
        result.append(_normalise_id(item["author"].get("id"), "A"))
    return sorted(set(result))


def _historical_institutions(work: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return author/institution IDs recorded on this work's authorship only."""

    result = []
    for item in work["authorships"]:
        author = item.get("author") if isinstance(item, dict) else None
        if not isinstance(author, dict):
            raise _fail("work contains malformed authorship author")
        author_id = _normalise_id(author.get("id"), "A")
        institutions = item.get("institutions", [])
        if not isinstance(institutions, list):
            raise _fail("work authorship institutions are malformed")
        for institution in institutions:
            if isinstance(institution, dict) and institution.get("id"):
                result.append((author_id, _normalise_id(institution["id"], "I")))
    return sorted(set(result))


def _topic_ids(work: Mapping[str, Any]) -> list[str]:
    topics = work.get("topics")
    if not isinstance(topics, list):
        raise _fail("work topics are malformed")
    result = []
    for topic in topics:
        if isinstance(topic, dict) and topic.get("id"):
            result.append(_normalise_id(topic["id"], "T"))
    return sorted(set(result))


def _referenced_work_ids(work: Mapping[str, Any]) -> list[str]:
    references = work.get("referenced_works")
    if not isinstance(references, list):
        raise _fail("work referenced_works are malformed")
    return sorted({_normalise_id(value, "W") for value in references})


def _work_text(work: Mapping[str, Any]) -> str | None:
    """Compose only publication-local public strings; never summary aggregates."""

    candidates: list[Any] = [work.get("display_name")]
    primary_topic = work.get("primary_topic")
    if isinstance(primary_topic, dict):
        candidates.append(primary_topic.get("display_name"))
    topics = work.get("topics")
    if isinstance(topics, list):
        candidates.extend(topic.get("display_name") for topic in topics if isinstance(topic, dict))
    primary_location = work.get("primary_location")
    if isinstance(primary_location, dict):
        source = primary_location.get("source")
        if isinstance(source, dict):
            candidates.append(source.get("display_name"))
    parts: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if not isinstance(value, str) or not value.strip():
            continue
        part = " ".join(value.split())
        key = part.casefold()
        if key not in seen:
            seen.add(key)
            parts.append(part)
    return " [SEP] ".join(parts) if parts else None


def _validate_work_text_mapping(
    arrays: Mapping[str, np.ndarray],
    text_rows: Iterable[Mapping[str, Any]],
    *,
    cluster_count: int,
) -> dict[int, int]:
    """Validate the order-independent work/text join and return hash -> work index."""

    if set(arrays) != set(WORK_TEXT_ARRAYS):
        raise _fail("work/text mapping array set is invalid")
    values = {name: np.asarray(arrays[name]) for name in WORK_TEXT_ARRAYS}
    row_counts = {int(value.shape[0]) for value in values.values() if value.ndim == 1}
    if len(row_counts) != 1 or any(value.ndim != 1 for value in values.values()):
        raise _fail("work/text mapping arrays are not aligned one-dimensional arrays")
    for name, (dtype, _) in WORK_TEXT_ARRAYS.items():
        if values[name].dtype.str != dtype:
            raise _fail(f"work/text mapping array {name!r} has an invalid dtype")
    hashes = values["work_id_hash"]
    hash_to_work = {int(value): index for index, value in enumerate(hashes)}
    if len(hash_to_work) != len(hashes):
        raise _fail("work/text mapping contains a work identifier hash collision")
    if cluster_count < 1:
        raise _fail("work/text mapping cluster inventory is empty")
    clusters = values["cluster"]
    if bool(np.any(clusters < 0)) or bool(np.any(clusters >= cluster_count)):
        raise _fail("work/text mapping contains an out-of-range cluster")
    start = int(datetime.combine(MIN_DATE, datetime.min.time(), tzinfo=UTC).timestamp())
    stop = int(
        datetime.combine(MAX_DATE + timedelta(days=1), datetime.min.time(), tzinfo=UTC).timestamp()
    )
    publication_timestamps = values["publication_timestamp"]
    if bool(np.any(publication_timestamps < start)) or bool(np.any(publication_timestamps >= stop)):
        raise _fail("work/text mapping contains an out-of-range publication timestamp")

    seen: set[int] = set()
    for row in text_rows:
        if set(row) != {"id", "text", "timestamp"}:
            raise _fail("text row schema is invalid")
        identifier = row["id"]
        text = row["text"]
        timestamp = row["timestamp"]
        if not isinstance(identifier, str) or _normalise_id(identifier, "W") != identifier:
            raise _fail("text row work identifier is not canonical")
        if not isinstance(text, str) or not text.strip():
            raise _fail("text row title is empty")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise _fail("text row timestamp is invalid")
        identifier_hash = int(portable_id_hash(identifier))
        if identifier_hash in seen:
            raise _fail("text rows contain a duplicate work identifier hash")
        work = hash_to_work.get(identifier_hash)
        if work is None:
            raise _fail("text row refers to an unknown work identifier hash")
        if int(publication_timestamps[work]) != timestamp:
            raise _fail("text row timestamp differs from its mapped work")
        if not bool(values["text_available"][work]):
            raise _fail("text row is present for a work marked text-unavailable")
        seen.add(identifier_hash)
    expected = {
        int(hashes[index])
        for index, available in enumerate(values["text_available"])
        if bool(available)
    }
    if seen != expected:
        raise _fail("text rows are missing one or more text-available work identifiers")
    return hash_to_work


def _write_sql_numeric_shards(
    connection: sqlite3.Connection,
    output: Path,
    *,
    prefix: str,
    query: str,
    columns: Mapping[str, np.dtype[Any]],
    rows_per_shard: int,
) -> tuple[ShardRecord, ...]:
    """Drain one ordered SQLite query into bounded immutable NPZ shards."""

    writer = NumericShardWriter(output, prefix=prefix, rows_per_shard=rows_per_shard)
    cursor = connection.execute(query)
    records: list[ShardRecord] = []
    while True:
        rows = cursor.fetchmany(rows_per_shard)
        if not rows:
            break
        arrays = {
            name: np.asarray([row[index] for row in rows], dtype=dtype)
            for index, (name, dtype) in enumerate(columns.items())
        }
        records.append(writer.write(arrays))
    if not records:
        records.append(
            writer.write({name: np.empty(0, dtype=dtype) for name, dtype in columns.items()})
        )
    return tuple(records)


def _remove_owned_staging_directory(path: Path, parent: Path, *, prefix: str) -> None:
    """Remove only a UUID-named, unpublished sibling staging directory."""

    resolved_parent = parent.resolve()
    resolved = path.resolve()
    if (
        resolved.parent != resolved_parent
        or not path.name.startswith(f".{prefix}.")
        or not path.name.endswith(".tmp")
    ):
        raise _fail("refused to remove an unowned preparation directory")
    if path.exists():
        shutil.rmtree(path)


def _load_work_payload(value: str) -> dict[str, Any]:
    try:
        loaded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise _fail("temporary work payload is invalid") from exc
    if not isinstance(loaded, dict):
        raise _fail("temporary work payload is not an object")
    return _validate_work(loaded)


def prepare_openalex(
    raw_dir: str | Path,
    root: str | Path,
    *,
    rows_per_shard: int = 50_000,
) -> dict[str, Any]:
    """Select and materialize the fixed corpus with bounded RAM and atomic publish."""

    if not 1 <= rows_per_shard <= MAX_NUMERIC_SHARD_ROWS:
        raise _fail(f"rows_per_shard must be in 1..{MAX_NUMERIC_SHARD_ROWS}")
    raw = Path(raw_dir).expanduser().resolve(strict=True)
    layout = RuntimeLayout.from_root(root)
    if raw != layout.raw_openalex.resolve():
        raise _fail("raw_dir must be the fixed RuntimeLayout OpenAlex directory")
    state = read_json_object(raw / RESUME_NAME)
    if state.get("schemaVersion") != RESUME_SCHEMA:
        raise _fail("raw acquisition uses an obsolete or unknown sampling protocol")
    if state.get("complete") is not True:
        raise _fail("raw acquisition is incomplete")
    if _migrate_or_validate_work_eligibility_state(state, raw / RAW_NAME):
        atomic_write_json(raw / RESUME_NAME, state)
    work_eligibility_audit = _work_eligibility_audit(state)
    topics = read_json_object(raw / TOPICS_NAME)
    cluster_caps = {
        str(item["clusterId"]): int(item["maximumWorks"])
        for item in topics.get("clusters", [])
        if isinstance(item, dict)
    }
    if tuple(cluster_caps.values()) != CLUSTER_CAPS:
        raise _fail("resolved topic cap contract is invalid")

    output = layout.processed_gfm / CORPUS_ID
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if (output / "manifest.json").is_file():
            return check_openalex(root)
        raise _fail("final corpus path exists without a complete manifest")
    staging = output.parent / f".{CORPUS_ID}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(parents=False)
    database = staging / ".prepare.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    published = False
    try:
        connection.executescript(
            """
            CREATE TABLE candidates (
                cluster_id TEXT NOT NULL,
                stratum_id TEXT NOT NULL,
                sample_key TEXT NOT NULL,
                work_id TEXT NOT NULL,
                publication_date TEXT NOT NULL,
                work_json TEXT NOT NULL
            );
            CREATE TABLE selected_work (
                work_id TEXT PRIMARY KEY,
                cluster_id TEXT NOT NULL,
                publication_date TEXT NOT NULL,
                publication_year INTEGER NOT NULL,
                publication_timestamp INTEGER,
                work_index INTEGER,
                work_id_hash TEXT,
                work_cluster INTEGER,
                text_available INTEGER,
                work_json TEXT NOT NULL
            );
            CREATE TABLE authors (
                entity_id TEXT PRIMARY KEY,
                entity_index INTEGER,
                first_timestamp INTEGER
            );
            CREATE TABLE institutions (
                entity_id TEXT PRIMARY KEY,
                entity_index INTEGER
            );
            CREATE TABLE topics (
                entity_id TEXT PRIMARY KEY,
                entity_index INTEGER
            );
            CREATE TABLE events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                src INTEGER NOT NULL,
                dst INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                relation INTEGER NOT NULL,
                work_index INTEGER NOT NULL
            );
            CREATE TABLE targets (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                src INTEGER NOT NULL,
                dst INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                first_collaboration INTEGER NOT NULL
            );
            CREATE TABLE seen_pairs (
                src INTEGER NOT NULL,
                dst INTEGER NOT NULL,
                PRIMARY KEY (src, dst)
            );
            """
        )
        raw_stratum_counts: dict[str, int] = defaultdict(int)
        insert_candidates: list[tuple[str, str, str, str, str, str]] = []
        for row in read_jsonl(raw / RAW_NAME):
            if set(row) != {"clusterId", "stratumId", "work"} or not isinstance(row["work"], dict):
                raise _fail("raw OpenAlex row schema is invalid")
            cluster_id = str(row["clusterId"])
            if cluster_id not in cluster_caps:
                raise _fail("raw OpenAlex row has an unknown cluster")
            work = _validate_work(row["work"], allow_empty_authorships=True)
            if not work["authorships"]:
                continue
            work_id = _normalise_id(work["id"], "W")
            stratum_id = str(row["stratumId"])
            if not stratum_id.startswith(cluster_id + ":"):
                raise _fail("raw OpenAlex row has a mismatched stratum")
            raw_stratum_counts[stratum_id] += 1
            insert_candidates.append(
                (
                    cluster_id,
                    stratum_id,
                    _stable_sample_key(work_id, stratum_id),
                    work_id,
                    str(work["publication_date"]),
                    json.dumps(
                        work,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
            if len(insert_candidates) == 1_000:
                connection.executemany(
                    "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?)",
                    insert_candidates,
                )
                insert_candidates.clear()
        if insert_candidates:
            connection.executemany(
                "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?)",
                insert_candidates,
            )
        connection.execute(
            "CREATE INDEX candidates_order ON candidates(cluster_id, sample_key, work_id, work_json)"
        )

        cluster_counts: dict[str, int] = {}
        for cluster_id, cap in cluster_caps.items():
            count = 0
            rows = connection.execute(
                """
                SELECT work_id, publication_date, work_json
                FROM candidates
                WHERE cluster_id = ?
                ORDER BY sample_key, work_id, work_json
                """,
                (cluster_id,),
            )
            for work_id, publication_date, work_json in rows:
                inserted = connection.execute(
                    """
                    INSERT OR IGNORE INTO selected_work(
                        work_id, cluster_id, publication_date, publication_year, work_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        work_id,
                        cluster_id,
                        publication_date,
                        int(str(publication_date)[:4]),
                        work_json,
                    ),
                ).rowcount
                if inserted:
                    count += 1
                    if count == cap:
                        break
            cluster_counts[cluster_id] = count
        work_count = int(connection.execute("SELECT COUNT(*) FROM selected_work").fetchone()[0])
        if not 0 < work_count <= TOTAL_CAP:
            raise _fail("deduplicated corpus is empty or exceeds the fixed work cap")

        cluster_index = {name: index for index, name in enumerate(cluster_caps)}
        work_hashes_seen: set[int] = set()
        for work_index_value, row in enumerate(
            connection.execute(
                """
                SELECT work_id, cluster_id, work_json
                FROM selected_work
                ORDER BY publication_date, work_id
                """
            )
        ):
            work_id, cluster_id, work_json = row
            work = _load_work_payload(work_json)
            timestamp = int(
                datetime.fromisoformat(str(work["publication_date"]))
                .replace(tzinfo=UTC)
                .timestamp()
            )
            identifier_hash = int(portable_id_hash(work_id))
            if identifier_hash in work_hashes_seen:
                raise _fail("canonical work identifier hashes are not unique")
            work_hashes_seen.add(identifier_hash)
            connection.execute(
                """
                UPDATE selected_work SET
                    publication_timestamp = ?, work_index = ?, work_id_hash = ?,
                    work_cluster = ?, text_available = ?
                WHERE work_id = ?
                """,
                (
                    timestamp,
                    work_index_value,
                    str(identifier_hash),
                    cluster_index[cluster_id],
                    int(_work_text(work) is not None),
                    work_id,
                ),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO authors(entity_id) VALUES (?)",
                ((value,) for value in sorted(set(_author_ids(work)))),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO institutions(entity_id) VALUES (?)",
                ((institution,) for _, institution in sorted(set(_historical_institutions(work)))),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO topics(entity_id) VALUES (?)",
                ((value,) for value in _topic_ids(work)),
            )

        def indexed_entities(table: str) -> dict[str, int]:
            result: dict[str, int] = {}
            updates: list[tuple[int, str]] = []
            for index, (identifier,) in enumerate(
                connection.execute(f"SELECT entity_id FROM {table} ORDER BY entity_id")
            ):
                result[str(identifier)] = index
                updates.append((index, str(identifier)))
                if len(updates) == 10_000:
                    connection.executemany(
                        f"UPDATE {table} SET entity_index = ? WHERE entity_id = ?",
                        updates,
                    )
                    updates.clear()
            if updates:
                connection.executemany(
                    f"UPDATE {table} SET entity_index = ? WHERE entity_id = ?",
                    updates,
                )
            return result

        author_index = indexed_entities("authors")
        institution_index = indexed_entities("institutions")
        topic_index = indexed_entities("topics")
        work_index_map = {
            str(work_id): int(index)
            for work_id, index in connection.execute(
                "SELECT work_id, work_index FROM selected_work"
            )
        }
        author_offset = 0
        work_offset = len(author_index)
        institution_offset = work_offset + work_count
        topic_offset = institution_offset + len(institution_index)
        mega_team_work_count = 0
        suppressed_pair_events = 0
        maximum_authors_per_work = 0

        for work_id, work_index_value, timestamp, work_json in connection.execute(
            """
            SELECT work_id, work_index, publication_timestamp, work_json
            FROM selected_work
            ORDER BY work_index
            """
        ):
            work = _load_work_payload(work_json)
            timestamp = int(timestamp)
            work_index_value = int(work_index_value)
            work_node = work_offset + work_index_value
            work_authors = sorted(set(_author_ids(work)))
            maximum_authors_per_work = max(maximum_authors_per_work, len(work_authors))
            event_rows: list[tuple[int, int, int, int, int]] = []
            for author_id in work_authors:
                author_node = author_index[author_id]
                event_rows.append((author_node, work_node, timestamp, 0, work_index_value))
                connection.execute(
                    """
                    UPDATE authors
                    SET first_timestamp = COALESCE(first_timestamp, ?)
                    WHERE entity_id = ?
                    """,
                    (timestamp, author_id),
                )
            if len(work_authors) <= MAX_COAUTHOR_CLIQUE_AUTHORS:
                for left_position, left in enumerate(work_authors):
                    for right in work_authors[left_position + 1 :]:
                        left_node, right_node = sorted((author_index[left], author_index[right]))
                        pair = (left_node, right_node)
                        first = connection.execute(
                            "INSERT OR IGNORE INTO seen_pairs(src, dst) VALUES (?, ?)",
                            pair,
                        ).rowcount
                        connection.execute(
                            """
                            INSERT INTO targets(src, dst, timestamp, first_collaboration)
                            VALUES (?, ?, ?, ?)
                            """,
                            (*pair, timestamp, int(bool(first))),
                        )
                        event_rows.append((left_node, right_node, timestamp, 1, work_index_value))
            else:
                mega_team_work_count += 1
                suppressed_pair_events += len(work_authors) * (len(work_authors) - 1) // 2
            for author_id, institution_id in sorted(set(_historical_institutions(work))):
                event_rows.append(
                    (
                        author_index[author_id],
                        institution_offset + institution_index[institution_id],
                        timestamp,
                        2,
                        work_index_value,
                    )
                )
            for topic_id in _topic_ids(work):
                event_rows.append(
                    (
                        work_node,
                        topic_offset + topic_index[topic_id],
                        timestamp,
                        3,
                        work_index_value,
                    )
                )
            for referenced_id in _referenced_work_ids(work):
                referenced_index = work_index_map.get(referenced_id)
                if referenced_index is not None:
                    event_rows.append(
                        (
                            work_node,
                            work_offset + referenced_index,
                            timestamp,
                            4,
                            work_index_value,
                        )
                    )
            connection.executemany(
                """
                INSERT INTO events(src, dst, timestamp, relation, work_index)
                VALUES (?, ?, ?, ?, ?)
                """,
                event_rows,
            )
        connection.commit()

        event_shards = _write_sql_numeric_shards(
            connection,
            staging,
            prefix="events",
            query=(
                "SELECT src, dst, timestamp, relation, work_index FROM events ORDER BY sequence"
            ),
            columns={
                "src": np.dtype(np.int64),
                "dst": np.dtype(np.int64),
                "timestamp": np.dtype(np.int64),
                "relation": np.dtype(np.int16),
                "work_index": np.dtype(np.int64),
            },
            rows_per_shard=rows_per_shard,
        )
        target_shards = _write_sql_numeric_shards(
            connection,
            staging,
            prefix="targets",
            query=(
                "SELECT src, dst, timestamp, first_collaboration FROM targets ORDER BY sequence"
            ),
            columns={
                "src": np.dtype(np.int64),
                "dst": np.dtype(np.int64),
                "timestamp": np.dtype(np.int64),
                "first_collaboration": np.dtype(np.bool_),
            },
            rows_per_shard=rows_per_shard,
        )
        newcomer_shards = _write_sql_numeric_shards(
            connection,
            staging,
            prefix="newcomers",
            query=("SELECT entity_index, first_timestamp, 0 FROM authors ORDER BY entity_index"),
            columns={
                "author": np.dtype(np.int64),
                "t0": np.dtype(np.int64),
                "history_verified": np.dtype(np.bool_),
            },
            rows_per_shard=rows_per_shard,
        )
        works_shards = _write_sql_numeric_shards(
            connection,
            staging,
            prefix="works",
            query=(
                "SELECT work_id_hash, publication_timestamp, work_cluster, "
                "text_available FROM selected_work ORDER BY work_index"
            ),
            columns={
                "work_id_hash": np.dtype(np.uint64),
                "publication_timestamp": np.dtype(np.int64),
                "cluster": np.dtype(np.int16),
                "text_available": np.dtype(np.bool_),
            },
            rows_per_shard=rows_per_shard,
        )
        event_role_ranges = {
            "train": (None, int(datetime(2021, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp())),
            "validation": (
                int(datetime(2021, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()),
                int(datetime(2022, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()),
            ),
            "test": (
                int(datetime(2022, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()),
                int(datetime(2023, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()),
            ),
            "shadow": (
                int(datetime(2023, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()),
                None,
            ),
        }
        # Product labels are shifted by one year relative to their context:
        # training cutoffs through 2021 require outcomes through 2022, while
        # 2022/2023/2024 cutoffs map to validation/test/shadow labels.
        target_role_ranges = {
            "train": (None, int(datetime(2022, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp())),
            "validation": (
                int(datetime(2022, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()),
                int(datetime(2023, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()),
            ),
            "test": (
                int(datetime(2023, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()),
                int(datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()),
            ),
            "shadow": (
                int(datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()),
                None,
            ),
        }

        def where_for(column: str, bounds: tuple[int | None, int | None]) -> str:
            lower, upper = bounds
            clauses = []
            if lower is not None:
                clauses.append(f"{column} > {lower}")
            if upper is not None:
                clauses.append(f"{column} <= {upper}")
            return " AND ".join(clauses) if clauses else "1 = 1"

        access_event_shards = {
            role: _write_sql_numeric_shards(
                connection,
                staging,
                prefix=f"access-events-{role}",
                query=(
                    "SELECT src, dst, timestamp, relation, work_index FROM events "
                    f"WHERE {where_for('timestamp', event_role_ranges[role])} "
                    "ORDER BY sequence"
                ),
                columns={
                    "src": np.dtype(np.int64),
                    "dst": np.dtype(np.int64),
                    "timestamp": np.dtype(np.int64),
                    "relation": np.dtype(np.int16),
                    "work_index": np.dtype(np.int64),
                },
                rows_per_shard=rows_per_shard,
            )
            for role in ACCESS_ROLES
        }
        access_target_shards = {
            role: _write_sql_numeric_shards(
                connection,
                staging,
                prefix=f"access-targets-{role}",
                query=(
                    "SELECT src, dst, timestamp, first_collaboration FROM targets "
                    f"WHERE {where_for('timestamp', target_role_ranges[role])} "
                    "ORDER BY sequence"
                ),
                columns={
                    "src": np.dtype(np.int64),
                    "dst": np.dtype(np.int64),
                    "timestamp": np.dtype(np.int64),
                    "first_collaboration": np.dtype(np.bool_),
                },
                rows_per_shard=rows_per_shard,
            )
            for role in ACCESS_ROLES
        }
        access_work_shards = {
            role: _write_sql_numeric_shards(
                connection,
                staging,
                prefix=f"access-works-{role}",
                query=(
                    "SELECT work_id_hash, publication_timestamp, work_cluster, "
                    "text_available FROM selected_work "
                    f"WHERE {where_for('publication_timestamp', event_role_ranges[role])} "
                    "ORDER BY work_index"
                ),
                columns={
                    "work_id_hash": np.dtype(np.uint64),
                    "publication_timestamp": np.dtype(np.int64),
                    "cluster": np.dtype(np.int16),
                    "text_available": np.dtype(np.bool_),
                },
                rows_per_shard=rows_per_shard,
            )
            for role in ACCESS_ROLES
        }

        def text_rows() -> Iterator[dict[str, Any]]:
            # Reverse work order proves that the persisted hash, not JSONL row
            # position, is the supported graph/text join.
            for work_id, timestamp, work_json in connection.execute(
                """
                SELECT work_id, publication_timestamp, work_json
                FROM selected_work
                WHERE text_available = 1
                ORDER BY work_index DESC
                """
            ):
                text = _work_text(_load_work_payload(work_json))
                if text is None:
                    raise _fail("text-available work lost its publication-local text")
                yield {"id": work_id, "text": text, "timestamp": int(timestamp)}

        text_count = atomic_write_jsonl(staging / "text.jsonl", text_rows())
        text_record = ShardRecord(
            path="text.jsonl",
            sha256=file_sha256(staging / "text.jsonl"),
            rows=text_count,
            arrays=(),
        )
        work_year = [
            int(value)
            for (value,) in connection.execute(
                "SELECT publication_year FROM selected_work ORDER BY work_index"
            )
        ]
        work_cluster = [
            int(value)
            for (value,) in connection.execute(
                "SELECT work_cluster FROM selected_work ORDER BY work_index"
            )
        ]
        split_counts = {
            "trainThrough2021": sum(year <= 2021 for year in work_year),
            "validation2022": sum(year == 2022 for year in work_year),
            "test2023": sum(year == 2023 for year in work_year),
            "stress2024To2025": sum(year >= 2024 for year in work_year),
        }
        edge_count = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        shards = (
            *event_shards,
            *target_shards,
            *newcomer_shards,
            *works_shards,
            *(record for role in ACCESS_ROLES for record in access_event_shards[role]),
            *(record for role in ACCESS_ROLES for record in access_target_shards[role]),
            *(record for role in ACCESS_ROLES for record in access_work_shards[role]),
            text_record,
        )
        manifest = build_manifest(
            schema_version="gfm.openalex-corpus/1.0",
            corpus_id=CORPUS_ID,
            license_id=LICENSE_ID,
            source={
                "uri": API_BASE,
                "licenseEvidence": LICENSE_URL,
                "rawSha256": file_sha256(raw / RAW_NAME),
                "topicResolutionSha256": file_sha256(raw / TOPICS_NAME),
                "formalEligible": state.get("formalEligible") is True
                and topics.get("formalEligible") is True,
            },
            shards=shards,
            splits=split_counts,
            privacy={
                "apiKeyPersisted": False,
                "blockedSnapshotAggregates": sorted(FORBIDDEN_FIELDS),
                "pointInTimeResidual": (
                    "current OpenAlex entity resolution is not historically reconstructable"
                ),
                "publicCheckpointEligible": state.get("formalEligible") is True
                and topics.get("formalEligible") is True,
            },
            extra={
                "domainId": DOMAIN_ID,
                "physicalAccess": {
                    "schemaVersion": PHYSICAL_ACCESS_SCHEMA,
                    "roles": list(ACCESS_ROLES),
                    "roleFamilies": {
                        "events": {
                            role: [record.path for record in access_event_shards[role]]
                            for role in ACCESS_ROLES
                        },
                        "targets": {
                            role: [record.path for record in access_target_shards[role]]
                            for role in ACCESS_ROLES
                        },
                        "works": {
                            role: [record.path for record in access_work_shards[role]]
                            for role in ACCESS_ROLES
                        },
                    },
                    "sharedFamilies": {},
                    "mergeOrder": {
                        "events": "timestamp",
                        "targets": "timestamp",
                        "works": "timestamp",
                    },
                    "targetRoleSemantics": (
                        "train<=2022, validation=2023, test=2024, shadow>=2025"
                    ),
                },
                "nodeCounts": {
                    "author": len(author_index),
                    "work": work_count,
                    "institution": len(institution_index),
                    "topic": len(topic_index),
                },
                "nodeOffsets": {
                    "author": author_offset,
                    "work": work_offset,
                    "institution": institution_offset,
                    "topic": topic_offset,
                },
                "relations": {
                    "0": "author-authored-work",
                    "1": "author-coauthored-author",
                    "2": "author-affiliated-at-work-time-institution",
                    "3": "work-has-topic",
                    "4": "work-cites-work",
                },
                "edgeCount": edge_count,
                "clusterCounts": cluster_counts,
                "stratumCounts": dict(sorted(raw_stratum_counts.items())),
                "fetchStrata": state.get("strata", []),
                "workEligibilityAudit": work_eligibility_audit,
                "samplingProtocol": {
                    "remote": "repeated-deterministic-sample-with-distinct-seeds",
                    "remoteSampleMaximum": MAX_SAMPLE_PER_STRATUM,
                    "remotePageSize": PER_PAGE,
                    "pageParameterUsedWithSample": False,
                    "localSelection": "sha256(seed,stratumId,workId)-ascending",
                    "localSeed": 20260820,
                    "deduplicateBy": "canonical-openalex-work-id",
                    "documentation": (
                        "https://developers.openalex.org/api-reference/works/list-works"
                    ),
                },
                "workTypeProtocol": {
                    "requestedCategories": list(REQUESTED_WORK_CATEGORIES),
                    "openAlexApiTypes": sorted(ALLOWED_WORK_TYPES),
                    "compatibility": WORK_TYPE_COMPATIBILITY,
                    "documentation": ("https://developers.openalex.org/api-reference/work-types"),
                },
                "workCap": TOTAL_CAP,
                "workYearSha256": canonical_sha256(work_year),
                "workClusterSha256": canonical_sha256(work_cluster),
                "materialization": {
                    "schemaVersion": "gfm.openalex-materialization/1.0",
                    "selectionStore": "ephemeral-sqlite-not-published",
                    "numericShardRowCap": rows_per_shard,
                    "maximumNumericShardRowCap": MAX_NUMERIC_SHARD_ROWS,
                    "atomicDirectoryPublish": True,
                    "allNumericShardsAtOrBelowCap": all(
                        record.rows <= rows_per_shard for record in shards if record.arrays
                    ),
                },
                "coauthorExpansionPolicy": {
                    "schemaVersion": "gfm.openalex-coauthor-expansion/1.0",
                    "representation": "authorship-hyperedge-via-work-node",
                    "cliqueMaterializedWhenAuthorCountAtMost": (MAX_COAUTHOR_CLIQUE_AUTHORS),
                    "aboveThresholdPolicy": "omit-entire-coauthor-clique",
                    "arbitraryAuthorSubsetUsed": False,
                    "megaTeamWorkCount": mega_team_work_count,
                    "suppressedPotentialPairEventCount": suppressed_pair_events,
                    "maximumAuthorsPerWork": maximum_authors_per_work,
                },
                "workTextMapping": {
                    "schemaVersion": "gfm.openalex-work-text-map/1.1",
                    "worksShards": [record.path for record in works_shards],
                    "textShard": text_record.path,
                    "workIndex": "concatenated-works-shard-row",
                    "workIdHashArray": "work_id_hash",
                    "embeddingIdHashArray": "id_hash",
                    "hashAlgorithm": PORTABLE_ID_HASH_ALGORITHM,
                    "publicationTimestampArray": "publication_timestamp",
                    "clusterArray": "cluster",
                    "textAvailableArray": "text_available",
                    "textFields": [
                        "display_name",
                        "primary_topic.display_name",
                        "topics[].display_name",
                        "primary_location.source.display_name",
                    ],
                    "summaryOrAggregateFieldsIncluded": False,
                    "missingTextPolicy": "allowed-only-when-text_available-is-false",
                },
            },
        )
        connection.close()
        database.unlink()
        atomic_write_json(staging / "manifest.json", manifest)
        verify_manifest(staging, manifest)
        work_arrays = _load_openalex_work_arrays(staging, manifest)
        _validate_work_text_mapping(
            work_arrays,
            read_jsonl(staging / "text.jsonl"),
            cluster_count=len(cluster_counts),
        )
        os.replace(staging, output)
        published = True
        return check_openalex(root)
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass
        if not published and staging.exists():
            _remove_owned_staging_directory(staging, output.parent, prefix=CORPUS_ID)


def _family_records(manifest: Mapping[str, Any], prefix: str) -> list[dict[str, Any]]:
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise _fail("manifest shard inventory is absent")
    records = [
        item
        for item in shards
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and str(item["path"]).startswith(prefix + "-")
        and "/" not in str(item["path"])
    ]
    if not records:
        raise _fail(f"numeric shard family {prefix!r} is absent")
    expected_paths = [f"{prefix}-{index:05d}.npz" for index in range(len(records))]
    if [str(item["path"]) for item in records] != expected_paths:
        raise _fail(f"numeric shard family {prefix!r} is not sequential")
    return records


def _load_openalex_work_arrays(output: Path, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    records = _family_records(manifest, "works")
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in WORK_TEXT_ARRAYS}
    for record in records:
        loaded = load_npz_safe(
            resolve_within(output, str(record["path"])), expected=WORK_TEXT_ARRAYS
        )
        for name in pieces:
            pieces[name].append(loaded[name])
    return {
        name: np.concatenate(values) if len(values) > 1 else values[0]
        for name, values in pieces.items()
    }


def _check_openalex_numeric_families(
    output: Path, manifest: Mapping[str, Any], *, shard_cap: int
) -> None:
    node_counts = manifest["nodeCounts"]
    author_count = int(node_counts["author"])
    work_count = int(node_counts["work"])
    node_count = sum(int(value) for value in node_counts.values())
    previous_event_time: int | None = None
    previous_target_time: int | None = None
    event_rows = 0
    target_rows = 0
    newcomer_rows = 0

    for prefix, expected in (
        ("events", EVENT_ARRAYS),
        ("targets", TARGET_ARRAYS),
        ("newcomers", NEWCOMER_ARRAYS),
        ("works", WORK_TEXT_ARRAYS),
    ):
        records = _family_records(manifest, prefix)
        for record in records:
            if (
                isinstance(record.get("rows"), bool)
                or not isinstance(record.get("rows"), int)
                or not 0 <= int(record["rows"]) <= shard_cap
            ):
                raise _fail(f"numeric shard family {prefix!r} exceeds its row cap")
            inventory = {
                str(item.get("name")): (item.get("dtype"), len(item.get("shape", [])))
                for item in record.get("arrays", [])
                if isinstance(item, dict) and isinstance(item.get("shape"), list)
            }
            if inventory != expected:
                raise _fail(f"numeric shard family {prefix!r} schema is invalid")
            arrays = load_npz_safe(resolve_within(output, str(record["path"])), expected=expected)
            rows = int(record["rows"])
            if prefix == "events":
                event_rows += rows
                if rows:
                    if (
                        int(arrays["src"].min()) < 0
                        or int(arrays["dst"].min()) < 0
                        or int(arrays["src"].max()) >= node_count
                        or int(arrays["dst"].max()) >= node_count
                        or int(arrays["relation"].min()) < 0
                        or int(arrays["relation"].max()) > 4
                        or int(arrays["work_index"].min()) < 0
                        or int(arrays["work_index"].max()) >= work_count
                        or bool(np.any(arrays["timestamp"][1:] < arrays["timestamp"][:-1]))
                        or (
                            previous_event_time is not None
                            and int(arrays["timestamp"][0]) < previous_event_time
                        )
                    ):
                        raise _fail("event shards violate graph or temporal bounds")
                    previous_event_time = int(arrays["timestamp"][-1])
            elif prefix == "targets":
                target_rows += rows
                if rows:
                    if (
                        int(arrays["src"].min()) < 0
                        or int(arrays["dst"].min()) < 0
                        or int(arrays["src"].max()) >= author_count
                        or int(arrays["dst"].max()) >= author_count
                        or bool(np.any(arrays["src"] >= arrays["dst"]))
                        or bool(np.any(arrays["timestamp"][1:] < arrays["timestamp"][:-1]))
                        or (
                            previous_target_time is not None
                            and int(arrays["timestamp"][0]) < previous_target_time
                        )
                    ):
                        raise _fail("target shards violate author or temporal bounds")
                    previous_target_time = int(arrays["timestamp"][-1])
            elif prefix == "newcomers":
                expected_authors = np.arange(newcomer_rows, newcomer_rows + rows, dtype=np.int64)
                if not np.array_equal(arrays["author"], expected_authors) or bool(
                    arrays["history_verified"].any()
                ):
                    raise _fail("unverified newcomer shards are not contiguous or false")
                newcomer_rows += rows
    if event_rows != int(manifest.get("edgeCount", -1)):
        raise _fail("event shard rows differ from edgeCount")
    if target_rows < 0 or newcomer_rows != author_count:
        raise _fail("target/newcomer shard counts are invalid")


def _check_openalex_physical_access(output: Path, manifest: Mapping[str, Any]) -> None:
    contract = manifest.get("physicalAccess")
    expected_merge = {
        "events": "timestamp",
        "targets": "timestamp",
        "works": "timestamp",
    }
    if (
        not isinstance(contract, dict)
        or contract.get("schemaVersion") != PHYSICAL_ACCESS_SCHEMA
        or contract.get("roles") != list(ACCESS_ROLES)
        or contract.get("sharedFamilies") != {}
        or contract.get("mergeOrder") != expected_merge
        or contract.get("targetRoleSemantics")
        != "train<=2022, validation=2023, test=2024, shadow>=2025"
    ):
        raise _fail("physical role-view contract is invalid")
    families = contract.get("roleFamilies")
    if not isinstance(families, dict) or set(families) != set(expected_merge):
        raise _fail("physical role-view family inventory is invalid")
    records = {
        str(item["path"]): item
        for item in manifest.get("shards", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }

    def load_family(prefix: str, expected: Mapping[str, tuple[str, int]]) -> dict[str, np.ndarray]:
        pieces: dict[str, list[np.ndarray]] = {name: [] for name in expected}
        for record in _family_records(manifest, prefix):
            loaded = load_npz_safe(resolve_within(output, str(record["path"])), expected=expected)
            for name in pieces:
                pieces[name].append(loaded[name])
        return {
            name: np.concatenate(values) if len(values) > 1 else values[0]
            for name, values in pieces.items()
        }

    canonical = {
        "events": load_family("events", EVENT_ARRAYS),
        "targets": load_family("targets", TARGET_ARRAYS),
        "works": load_family("works", WORK_TEXT_ARRAYS),
    }
    event_bounds = {
        "train": (None, 2021),
        "validation": (2021, 2022),
        "test": (2022, 2023),
        "shadow": (2023, None),
    }
    target_bounds = {
        "train": (None, 2022),
        "validation": (2022, 2023),
        "test": (2023, 2024),
        "shadow": (2024, None),
    }

    def year_end(year: int) -> int:
        return int(datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp())

    declared_paths: set[str] = set()
    for family, expected in (
        ("events", EVENT_ARRAYS),
        ("targets", TARGET_ARRAYS),
        ("works", WORK_TEXT_ARRAYS),
    ):
        role_map = families.get(family)
        if not isinstance(role_map, dict) or set(role_map) != set(ACCESS_ROLES):
            raise _fail(f"physical role-view {family} roles are invalid")
        timestamp_name = "publication_timestamp" if family == "works" else "timestamp"
        bounds = target_bounds if family == "targets" else event_bounds
        source = canonical[family]
        for role in ACCESS_ROLES:
            paths = role_map[role]
            if (
                not isinstance(paths, list)
                or not paths
                or any(not isinstance(path, str) for path in paths)
                or len(paths) != len(set(paths))
                or declared_paths.intersection(paths)
            ):
                raise _fail(f"physical role-view {family}/{role} paths are invalid")
            declared_paths.update(paths)
            pieces: dict[str, list[np.ndarray]] = {name: [] for name in expected}
            for index, path in enumerate(paths):
                if path != f"access-{family}-{role}-{index:05d}.npz":
                    raise _fail("physical role-view shards are not sequential")
                record = records.get(path)
                if record is None:
                    raise _fail("physical role-view shard is undeclared")
                loaded = load_npz_safe(resolve_within(output, path), expected=expected)
                for name in pieces:
                    pieces[name].append(loaded[name])
            actual = {
                name: np.concatenate(values) if len(values) > 1 else values[0]
                for name, values in pieces.items()
            }
            lower_year, upper_year = bounds[role]
            mask = np.ones(source[timestamp_name].shape, dtype=np.bool_)
            if lower_year is not None:
                mask &= source[timestamp_name] > year_end(lower_year)
            if upper_year is not None:
                mask &= source[timestamp_name] <= year_end(upper_year)
            if any(not np.array_equal(actual[name], source[name][mask]) for name in expected):
                raise _fail(f"physical role-view {family}/{role} differs from canonical data")


def check_openalex(root: str | Path) -> dict[str, Any]:
    output = RuntimeLayout.from_root(root).processed_gfm / CORPUS_ID
    manifest = _read_openalex_base_manifest_identity(root)
    verify_manifest(output, manifest)
    source = manifest.get("source")
    privacy = manifest.get("privacy")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("formalEligible"), bool)
        or not isinstance(privacy, dict)
        or privacy.get("publicCheckpointEligible") is not source.get("formalEligible")
    ):
        raise _fail("processed formal-eligibility evidence is inconsistent")
    eligibility = manifest.get("workEligibilityAudit")
    accepted_rows = eligibility.get("acceptedRows") if isinstance(eligibility, dict) else None
    inspected_rows = eligibility.get("inspectedRows") if isinstance(eligibility, dict) else None
    excluded_rows = eligibility.get("excludedRows") if isinstance(eligibility, dict) else None
    if (
        not isinstance(eligibility, dict)
        or eligibility.get("workEligibilityProtocol") != WORK_ELIGIBILITY_POLICY
        or isinstance(accepted_rows, bool)
        or not isinstance(accepted_rows, int)
        or accepted_rows < int(manifest.get("nodeCounts", {}).get("work", -1))
        or isinstance(inspected_rows, bool)
        or not isinstance(inspected_rows, int)
        or isinstance(excluded_rows, bool)
        or not isinstance(excluded_rows, int)
        or excluded_rows < 0
        or inspected_rows != accepted_rows + excluded_rows
        or eligibility.get("excludedByReason") != {NO_VALID_AUTHORSHIPS_REASON: excluded_rows}
        or not isinstance(eligibility.get("excludedWorkIdDigest"), str)
        or len(str(eligibility.get("excludedWorkIdDigest"))) != 64
    ):
        raise _fail("processed work-eligibility audit is invalid")
    mapping = manifest.get("workTextMapping")
    shards = manifest.get("shards")
    if not isinstance(mapping, dict) or not isinstance(shards, list):
        raise _fail("work/text mapping declaration is absent")
    works_records = _family_records(manifest, "works")
    works_paths = [str(item["path"]) for item in works_records]
    expected_mapping = {
        "schemaVersion": "gfm.openalex-work-text-map/1.1",
        "worksShards": works_paths,
        "textShard": "text.jsonl",
        "workIndex": "concatenated-works-shard-row",
        "workIdHashArray": "work_id_hash",
        "embeddingIdHashArray": "id_hash",
        "hashAlgorithm": PORTABLE_ID_HASH_ALGORITHM,
        "publicationTimestampArray": "publication_timestamp",
        "clusterArray": "cluster",
        "textAvailableArray": "text_available",
        "textFields": [
            "display_name",
            "primary_topic.display_name",
            "topics[].display_name",
            "primary_location.source.display_name",
        ],
        "summaryOrAggregateFieldsIncluded": False,
        "missingTextPolicy": "allowed-only-when-text_available-is-false",
    }
    if mapping != expected_mapping:
        raise _fail("work/text mapping declaration is invalid")
    text_records = [
        item for item in shards if isinstance(item, dict) and item.get("path") == "text.jsonl"
    ]
    if len(text_records) != 1:
        raise _fail("work/text mapping text artifact is not uniquely declared")
    for record in works_records:
        if {
            str(item.get("name")) for item in record.get("arrays", []) if isinstance(item, dict)
        } != set(WORK_TEXT_ARRAYS):
            raise _fail("work/text mapping shard inventory is invalid")
    node_counts = manifest.get("nodeCounts")
    cluster_counts = manifest.get("clusterCounts")
    if not isinstance(node_counts, dict) or not isinstance(cluster_counts, dict):
        raise _fail("work/text mapping counts are absent")
    if sum(int(record.get("rows", -1)) for record in works_records) != int(
        node_counts.get("work", -2)
    ):
        raise _fail("work/text mapping row count differs from canonical work nodes")

    materialization = manifest.get("materialization")
    policy = manifest.get("coauthorExpansionPolicy")
    if (
        not isinstance(materialization, dict)
        or materialization.get("schemaVersion") != "gfm.openalex-materialization/1.0"
        or materialization.get("selectionStore") != "ephemeral-sqlite-not-published"
        or materialization.get("atomicDirectoryPublish") is not True
        or materialization.get("allNumericShardsAtOrBelowCap") is not True
        or isinstance(materialization.get("numericShardRowCap"), bool)
        or not isinstance(materialization.get("numericShardRowCap"), int)
        or not 1 <= int(materialization["numericShardRowCap"]) <= MAX_NUMERIC_SHARD_ROWS
        or materialization.get("maximumNumericShardRowCap") != MAX_NUMERIC_SHARD_ROWS
    ):
        raise _fail("bounded materialization contract is invalid")
    shard_cap = int(materialization["numericShardRowCap"])
    _check_openalex_numeric_families(output, manifest, shard_cap=shard_cap)
    _check_openalex_physical_access(output, manifest)
    if (
        not isinstance(policy, dict)
        or policy.get("schemaVersion") != "gfm.openalex-coauthor-expansion/1.0"
        or policy.get("representation") != "authorship-hyperedge-via-work-node"
        or policy.get("cliqueMaterializedWhenAuthorCountAtMost") != MAX_COAUTHOR_CLIQUE_AUTHORS
        or policy.get("aboveThresholdPolicy") != "omit-entire-coauthor-clique"
        or policy.get("arbitraryAuthorSubsetUsed") is not False
        or any(
            isinstance(policy.get(name), bool)
            or not isinstance(policy.get(name), int)
            or int(policy[name]) < 0
            for name in (
                "megaTeamWorkCount",
                "suppressedPotentialPairEventCount",
                "maximumAuthorsPerWork",
            )
        )
    ):
        raise _fail("coauthor expansion policy is invalid")

    arrays = _load_openalex_work_arrays(output, manifest)
    text_rows = read_jsonl(resolve_within(output, "text.jsonl"))
    _validate_work_text_mapping(arrays, text_rows, cluster_count=len(cluster_counts))
    if int(arrays["text_available"].sum()) != int(text_records[0].get("rows", -1)):
        raise _fail("text row count differs from its manifest")
    return manifest
