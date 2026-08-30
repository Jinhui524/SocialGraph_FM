"""Offline, fail-closed Thailand Governance target-package generation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from ..canonical import canonical_sha256
from .contracts import INPUT_SCHEMA_VERSION, MODALITIES, GovernanceInputManifest

PINNED_ENCODER_MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
PINNED_ENCODER_REVISION = "4328cf26390c98c5e3c738b4460a05b95f4911f5"
SOURCE_SCHEMA_VERSION = "socialgraph-fm.anonymized-posts/1.0"
AUTHORIZATION_SCHEMA_VERSION = "socialgraph-fm.authorized-source/1.0"
RUNTIME_AUTHORIZATION_SCHEMA_VERSION = "socialgraph-fm.defense-runtime-authorization/1.0"
LABEL_SCHEMA_VERSION = "socialgraph-fm.governance-target-label-recipe/1.1"
RECEIPT_SCHEMA_VERSION = "socialgraph-fm.governance-target-package-receipt/1.1"
TARGET_NODE_COUNT = 128
MIN_IO_NODES = 16
MIN_CONTROL_NODES = 64
MIN_MODALITIES = 4
LABELS_PER_CLASS = 8
TWEET_SIM_TOP_K = 5
TWEET_SIM_THRESHOLD = 0.8
TWEET_SIM_PAIR_BUDGET = 10_000
GROUP_RELATION_MAX_ACCOUNTS = 256
GROUP_RELATION_PAIR_BUDGET = 50_000
FAST_RT_WINDOW_SECONDS = 10.0
FAST_RT_PAIR_BUDGET = 50_000
_SOURCE_FILE_LIMIT = 256 * 1024 * 1024
_SOURCE_ROW_LIMIT = 250_000
_SOURCE_ACCOUNT_LIMIT = 5_000
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class SourceValidationError(ValueError):
    """Raised when authorization, source, coverage, or output confinement fails."""


class _Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class _SourcePost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    postid: StrictStr = Field(min_length=1, max_length=256)
    post_text: StrictStr = Field(max_length=20_000)
    post_time: StrictStr = Field(min_length=20, max_length=40)
    accountid: StrictStr = Field(min_length=1, max_length=256)
    is_repost: StrictBool
    reposted_accountid: StrictStr | None = Field(default=None, max_length=256)
    reposted_postid: StrictStr | None = Field(default=None, max_length=256)
    hashtags: list[StrictStr] = Field(max_length=100)
    urls: list[StrictStr] = Field(max_length=20)
    account_mentions: list[StrictStr] = Field(max_length=100)
    in_reply_to_accountid: StrictStr | None = Field(default=None, max_length=256)
    is_control: StrictBool

    @field_validator("postid", "accountid", "reposted_accountid", "reposted_postid", "in_reply_to_accountid")
    @classmethod
    def validate_identifier(cls, value: str | None) -> str | None:
        if value is not None and any(ord(character) < 32 for character in value):
            raise ValueError("identifier contains a control character")
        return value

    @field_validator("post_text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("post_text contains NUL")
        return value

    @field_validator("post_time")
    @classmethod
    def validate_time(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("post_time must be ISO-8601") from error
        if parsed.tzinfo is None:
            raise ValueError("post_time must include a timezone")
        return value

    @field_validator("hashtags")
    @classmethod
    def validate_hashtags(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 100 or "\x00" in value for value in values):
            raise ValueError("hashtags contain an invalid value")
        return values

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 2_048 or "\x00" in value for value in values):
            raise ValueError("urls contain an invalid value")
        return values

    @field_validator("account_mentions")
    @classmethod
    def validate_mentions(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 256 for value in values):
            raise ValueError("account_mentions contain an invalid value")
        return values

    @model_validator(mode="after")
    def validate_repost_binding(self) -> _SourcePost:
        if self.is_repost != bool(self.reposted_postid):
            raise ValueError("is_repost must agree with reposted_postid")
        if self.is_repost and not self.reposted_accountid:
            raise ValueError("reposted_accountid is required for reposts")
        if not self.is_repost and self.reposted_accountid is not None:
            raise ValueError("non-reposts cannot name a reposted account")
        return self


@dataclass(frozen=True)
class ThailandPackage:
    bundle_path: Path
    labels_path: Path
    receipt_path: Path


@dataclass(frozen=True)
class _Relation:
    source: str
    target: str
    modality: str
    weight: float


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    ) + b"\n"


def _stable_hash(*values: str) -> str:
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


def anonymized_node_id(account_id: str) -> str:
    """Return a stable opaque ID without carrying the source identifier into outputs."""

    return f"th:{_stable_hash('socialgraph-thailand-node-v1', account_id)[:24]}"


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _is_reparse(path: Path) -> bool:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(value, "st_file_attributes", 0)
    is_junction = getattr(os.path, "isjunction", lambda _value: False)
    return path.is_symlink() or bool(is_junction(path)) or bool(attributes & 0x400)


def _assert_no_reparse_absolute(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            break
        if _is_reparse(current):
            raise SourceValidationError(f"{label} contains an unsafe reparse point or symlink")


def _assert_no_reparse_segments(root: Path, candidate: Path, *, label: str) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise SourceValidationError(f"{label} must remain inside the authorized runtime root") from error
    current = root
    if _is_reparse(current):
        raise SourceValidationError(f"{label} contains an unsafe reparse point or symlink")
    for part in relative.parts:
        current /= part
        if not current.exists() and not current.is_symlink():
            break
        if _is_reparse(current):
            raise SourceValidationError(f"{label} contains an unsafe reparse point or symlink")


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = _absolute_lexical(path)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise SourceValidationError(f"{label} must remain inside the authorized runtime root") from error
    _assert_no_reparse_segments(root, resolved, label=label)
    return resolved


def _load_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceValidationError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise SourceValidationError(f"invalid {label}")
    return cast(dict[str, Any], value)


def _read_regular_file_snapshot(
    path: Path, *, root: Path, label: str, maximum: int
) -> bytes:
    _assert_no_reparse_segments(root, path, label=label)
    if not path.is_file() or _is_reparse(path):
        raise SourceValidationError(f"{label} is missing or unsafe")
    size = path.stat().st_size
    if size <= 0 or size > maximum:
        raise SourceValidationError(f"{label} exceeds the configured size limit")
    try:
        snapshot = path.read_bytes()
    except OSError as error:
        raise SourceValidationError(f"{label} could not be read") from error
    if len(snapshot) <= 0 or len(snapshot) > maximum:
        raise SourceValidationError(f"{label} exceeds the configured size limit")
    return snapshot


def _validate_runtime(runtime_root: Path) -> Path:
    root = _absolute_lexical(runtime_root)
    if root.name.casefold() != "defense" or root.parent.name.casefold() != "var":
        raise SourceValidationError("runtime root must be the explicit ignored var/defense root")
    _assert_no_reparse_absolute(root, label="runtime root")
    if not root.is_dir() or _is_reparse(root):
        raise SourceValidationError("runtime root is missing or contains an unsafe reparse point")
    marker_path = root / "runtime-authorization.json"
    marker_bytes = _read_regular_file_snapshot(
        marker_path,
        root=root,
        label="runtime authorization metadata",
        maximum=64 * 1024,
    )
    marker = _load_json_object(marker_bytes, label="runtime authorization metadata")
    if marker != {
        "schemaVersion": RUNTIME_AUTHORIZATION_SCHEMA_VERSION,
        "purpose": "authorized-anonymized-defense-data",
    }:
        raise SourceValidationError("runtime authorization metadata is invalid")
    return root


def _validate_authorization(
    source: Path, runtime: Path
) -> tuple[dict[str, Any], Path, bytes]:
    authorization_path = source / "authorization.json"
    authorization_bytes = _read_regular_file_snapshot(
        authorization_path,
        root=runtime,
        label="source authorization metadata",
        maximum=256 * 1024,
    )
    authorization = _load_json_object(
        authorization_bytes, label="source authorization metadata"
    )
    expected_keys = {
        "schemaVersion",
        "datasetId",
        "country",
        "sourceSchemaVersion",
        "sourceFile",
        "sourceSha256",
        "authorizationReference",
        "license",
        "approvedAt",
    }
    if set(authorization) != expected_keys:
        raise SourceValidationError("source authorization metadata has missing or unknown fields")
    if authorization["schemaVersion"] != AUTHORIZATION_SCHEMA_VERSION:
        raise SourceValidationError("source authorization metadata schema is unsupported")
    if authorization["country"] != "TH":
        raise SourceValidationError("source authorization metadata is not bound to Thailand")
    if authorization["sourceSchemaVersion"] != SOURCE_SCHEMA_VERSION:
        raise SourceValidationError("source schema is unsupported")
    if authorization["sourceFile"] != "posts.jsonl":
        raise SourceValidationError("authorized sourceFile must be posts.jsonl")
    for key, maximum in (
        ("datasetId", 100),
        ("authorizationReference", 500),
        ("license", 200),
        ("approvedAt", 40),
    ):
        value = authorization[key]
        if not isinstance(value, str) or not value or len(value) > maximum:
            raise SourceValidationError(f"source authorization metadata field {key} is invalid")
    source_hash = authorization["sourceSha256"]
    if not isinstance(source_hash, str) or not _HASH_RE.fullmatch(source_hash):
        raise SourceValidationError("source authorization metadata hash is invalid")
    try:
        approved_at = datetime.fromisoformat(authorization["approvedAt"])
    except ValueError as error:
        raise SourceValidationError("source authorization approvedAt is invalid") from error
    if approved_at.tzinfo is None:
        raise SourceValidationError("source authorization approvedAt must include a timezone")
    source_file = source / "posts.jsonl"
    source_bytes = _read_regular_file_snapshot(
        source_file,
        root=runtime,
        label="authorized posts.jsonl",
        maximum=_SOURCE_FILE_LIMIT,
    )
    if _sha256_bytes(source_bytes) != source_hash:
        raise SourceValidationError("source hash mismatch")
    return authorization, source_file, source_bytes


def _load_posts(source_bytes: bytes) -> list[_SourcePost]:
    posts: list[_SourcePost] = []
    seen_post_ids: set[str] = set()
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SourceValidationError("authorized posts.jsonl is not UTF-8") from error
    for line_number, line in enumerate(io.StringIO(source_text), start=1):
        if line_number > _SOURCE_ROW_LIMIT:
            raise SourceValidationError("source row count exceeds the configured limit")
        try:
            raw = json.loads(line)
            post = _SourcePost.model_validate(raw)
        except (json.JSONDecodeError, ValueError) as error:
            raise SourceValidationError(f"source row {line_number} is invalid: {error}") from error
        if post.postid in seen_post_ids:
            raise SourceValidationError(f"source row {line_number} duplicates postid")
        seen_post_ids.add(post.postid)
        posts.append(post)
    if not posts:
        raise SourceValidationError("authorized source contains no posts")
    account_ids = {post.accountid for post in posts}
    if len(account_ids) > _SOURCE_ACCOUNT_LIMIT:
        raise SourceValidationError("source account count exceeds the configured limit")
    labels: dict[str, bool] = {}
    for post in posts:
        previous = labels.setdefault(post.accountid, post.is_control)
        if previous != post.is_control:
            raise SourceValidationError("source rows disagree on account is_control")
    return posts


def _top_account_posts(posts: Sequence[_SourcePost | Mapping[str, Any]]) -> dict[str, list[str]]:
    repost_counts = Counter(
        str(_field(post, "reposted_postid"))
        for post in posts
        if bool(_field(post, "is_repost")) and _field(post, "reposted_postid") is not None
    )
    authored: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for post in posts:
        post_id = str(_field(post, "postid"))
        account_id = str(_field(post, "accountid"))
        text = str(_field(post, "post_text"))
        authored[account_id].append((repost_counts[post_id], post_id, text))
    selected: dict[str, list[str]] = {}
    for account_id, candidates in authored.items():
        ordered = sorted(
            candidates,
            key=lambda item: (-item[0], _stable_hash("top-content-v1", item[1])),
        )
        selected[account_id] = [item[2] for item in ordered[:5]]
    return selected


def _field(value: _SourcePost | Mapping[str, Any], name: str) -> Any:
    return getattr(value, name) if isinstance(value, _SourcePost) else value[name]


def aggregate_account_content(
    posts: Sequence[_SourcePost | Mapping[str, Any]],
) -> dict[str, str]:
    """Aggregate up to five highest-observed-repost posts per account."""

    return {account: "\n".join(texts) for account, texts in _top_account_posts(posts).items()}


def _encoder_description(encoder: _Encoder) -> dict[str, Any]:
    model_id = getattr(encoder, "model_id", "injected-unidentified-encoder")
    revision = getattr(encoder, "revision", "unverified")
    cache_sha256 = getattr(encoder, "cache_sha256", "0" * 64)
    if not all(isinstance(value, str) for value in (model_id, revision, cache_sha256)):
        raise SourceValidationError("encoder provenance fields must be strings")
    if not _HASH_RE.fullmatch(cache_sha256):
        raise SourceValidationError("encoder cache_sha256 must be a SHA-256 digest")
    return {
        "modelId": model_id,
        "revision": revision,
        "cacheSha256": cache_sha256,
        "compatibility": "dimension-only-unverified",
        "dimension": 768,
    }


def _directory_hash(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise SourceValidationError("offline encoder cache contains no files")
    for path in files:
        if path.is_symlink():
            raise SourceValidationError("offline encoder cache cannot contain links")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


class _ProductionEncoder:
    model_id = PINNED_ENCODER_MODEL_ID
    revision = PINNED_ENCODER_REVISION

    def __init__(self, model: Any, cache_sha256: str) -> None:
        self._model = model
        self.cache_sha256 = cache_sha256

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )


def _load_production_encoder(cache: Path, runtime: Path) -> _Encoder:
    resolved = _inside(cache, runtime, label="offline encoder cache")
    if not resolved.is_dir():
        raise SourceValidationError("offline encoder cache is missing")
    cache_hash = _directory_hash(resolved)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SourceValidationError(
            "sentence-transformers is not installed; install it explicitly and provide the pinned offline cache"
        ) from error
    try:
        model = SentenceTransformer(
            PINNED_ENCODER_MODEL_ID,
            revision=PINNED_ENCODER_REVISION,
            cache_folder=str(resolved),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise SourceValidationError(
            "pinned encoder could not be loaded from the offline cache; downloads are disabled"
        ) from error
    return _ProductionEncoder(model, cache_hash)


def _encode_account_content(
    posts: Sequence[_SourcePost], encoder: _Encoder
) -> tuple[list[str], np.ndarray]:
    selected = _top_account_posts(posts)
    accounts = sorted(selected)
    flat_texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for account in accounts:
        start = len(flat_texts)
        flat_texts.extend(selected[account])
        spans.append((start, len(flat_texts)))
    raw = np.asarray(encoder.encode(flat_texts))
    if raw.shape != (len(flat_texts), 768):
        raise SourceValidationError("encoder output must have shape [post_count, 768]")
    if raw.dtype != np.float32:
        raw = raw.astype(np.float32)
    if not bool(np.isfinite(raw).all()):
        raise SourceValidationError("encoder output must contain only finite values")
    account_vectors = np.vstack(
        [np.mean(raw[start:stop], axis=0, dtype=np.float32) for start, stop in spans]
    ).astype(np.float32, copy=False)
    norms = np.linalg.norm(account_vectors, axis=1, keepdims=True)
    account_vectors = np.divide(
        account_vectors,
        norms,
        out=np.zeros_like(account_vectors),
        where=norms > 0,
    )
    if not bool(np.isfinite(account_vectors).all()):
        raise SourceValidationError("account feature aggregation produced non-finite values")
    return accounts, account_vectors


def _normalized_hashtags(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        unicodedata.normalize("NFKC", value).strip().lstrip("#").casefold() for value in values
    )


def _add_group_relations(
    weights: dict[tuple[str, str, str], float], modality: str, groups: Mapping[str, set[str]]
) -> None:
    for group_id in sorted(groups):
        accounts = sorted(groups[group_id])
        for left, source in enumerate(accounts):
            for target in accounts[left + 1 :]:
                weights[(source, target, modality)] += 1.0


def _validate_group_relation_density(
    groups_by_modality: Sequence[tuple[str, Mapping[str, set[str]]]],
) -> None:
    potential_pairs = 0
    for modality, groups in groups_by_modality:
        for group_id in sorted(groups):
            size = len(groups[group_id])
            if size > GROUP_RELATION_MAX_ACCOUNTS:
                raise SourceValidationError(
                    f"{modality} group is over-dense ({size} accounts)"
                )
            potential_pairs += size * (size - 1) // 2
    if potential_pairs > GROUP_RELATION_PAIR_BUDGET:
        raise SourceValidationError(
            "group relation pair budget exceeded before relation expansion"
        )


def _derive_relations(
    posts: Sequence[_SourcePost], accounts: Sequence[str], features: np.ndarray
) -> list[_Relation]:
    weights: dict[tuple[str, str, str], float] = defaultdict(float)
    repost_groups: dict[str, set[str]] = defaultdict(set)
    url_groups: dict[str, set[str]] = defaultdict(set)
    hashtag_groups: dict[str, set[str]] = defaultdict(set)
    repost_events: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for post in posts:
        if post.is_repost and post.reposted_postid:
            repost_groups[_stable_hash("repost-v1", post.reposted_postid)].add(post.accountid)
            timestamp = datetime.fromisoformat(post.post_time).timestamp()
            repost_events[_stable_hash("fast-repost-v1", post.reposted_postid)].append(
                (timestamp, post.accountid)
            )
        for url in post.urls:
            normalized = unicodedata.normalize("NFKC", url).strip()
            url_groups[_stable_hash("url-v1", normalized)].add(post.accountid)
        sequence = _normalized_hashtags(post.hashtags)
        if sequence:
            hashtag_groups[_stable_hash("hashtag-sequence-v1", *sequence)].add(post.accountid)
    grouped_relations = (
        ("coRT", repost_groups),
        ("coURL", url_groups),
        ("hashSeq", hashtag_groups),
    )
    _validate_group_relation_density(grouped_relations)
    for modality, groups in grouped_relations:
        _add_group_relations(weights, modality, groups)
    fast_pair_count = 0
    for group_id in sorted(repost_events):
        events = repost_events[group_id]
        ordered_events = sorted(events)
        left = 0
        for right, (timestamp, account) in enumerate(ordered_events):
            while timestamp - ordered_events[left][0] > FAST_RT_WINDOW_SECONDS:
                left += 1
            window_pairs = right - left
            if fast_pair_count + window_pairs > FAST_RT_PAIR_BUDGET:
                raise SourceValidationError(
                    "fastRT pair budget exceeded before relation expansion"
                )
            fast_pair_count += window_pairs
            for previous in range(left, right):
                source, target = sorted(
                    (ordered_events[previous][1], account)
                )
                if source != target:
                    weights[(source, target, "fastRT")] += 1.0

    similarities = np.asarray(features @ features.T, dtype=np.float32)
    np.fill_diagonal(similarities, -np.inf)
    top_neighbors: list[set[int]] = []
    for row in range(len(accounts)):
        eligible = np.flatnonzero(similarities[row] >= TWEET_SIM_THRESHOLD)
        ordered_neighbors = sorted(
            (int(index) for index in eligible),
            key=lambda index: (-float(similarities[row, index]), accounts[index]),
        )
        top_neighbors.append(set(ordered_neighbors[:TWEET_SIM_TOP_K]))
    tweet_pairs: list[tuple[float, str, str]] = []
    for source_index, neighbors in enumerate(top_neighbors):
        for target_index in neighbors:
            if source_index < target_index and source_index in top_neighbors[target_index]:
                tweet_pairs.append(
                    (
                        float(similarities[source_index, target_index]),
                        accounts[source_index],
                        accounts[target_index],
                    )
                )
    for score, source, target in sorted(
        tweet_pairs, key=lambda item: (-item[0], item[1], item[2])
    )[:TWEET_SIM_PAIR_BUDGET]:
        weights[(source, target, "tweetSim")] = score
    return [
        _Relation(source=source, target=target, modality=modality, weight=weight)
        for (source, target, modality), weight in sorted(
            weights.items(), key=lambda item: (MODALITIES.index(item[0][2]), item[0][0], item[0][1])
        )
    ]


def _connected_selection(
    accounts: Sequence[str], relations: Sequence[_Relation], labels: Mapping[str, bool]
) -> tuple[str, ...]:
    adjacency: dict[str, set[str]] = {account: set() for account in accounts}
    for relation in relations:
        adjacency[relation.source].add(relation.target)
        adjacency[relation.target].add(relation.source)
    unseen = set(accounts)
    components: list[set[str]] = []
    while unseen:
        root = min(unseen, key=lambda value: _stable_hash("component-root-v1", value))
        component = {root}
        queue = deque([root])
        unseen.remove(root)
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency[current]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    eligible = [
        component
        for component in components
        if len(component) >= TARGET_NODE_COUNT
        and sum(not labels[node] for node in component) >= MIN_IO_NODES
        and sum(labels[node] for node in component) >= MIN_CONTROL_NODES
    ]
    for component in sorted(
        eligible,
        key=lambda values: (-len(values), min(_stable_hash("component-v1", item) for item in values)),
    ):
        roots = sorted(component, key=lambda item: _stable_hash("selection-root-v1", item))
        for root in roots[: min(64, len(roots))]:
            selected = [root]
            selected_set = {root}
            frontier = set(adjacency[root]) & component
            while frontier and len(selected) < TARGET_NODE_COUNT:
                remaining_after = TARGET_NODE_COUNT - len(selected) - 1
                io_count = sum(not labels[item] for item in selected)
                control_count = len(selected) - io_count

                def priority(
                    candidate: str,
                    *,
                    bound_io_count: int = io_count,
                    bound_control_count: int = control_count,
                    bound_remaining: int = remaining_after,
                    bound_component: set[str] = component,
                ) -> tuple[int, int, str]:
                    io_deficit_after = max(
                        0,
                        MIN_IO_NODES
                        - bound_io_count
                        - int(not labels[candidate]),
                    )
                    control_deficit_after = max(
                        0,
                        MIN_CONTROL_NODES
                        - bound_control_count
                        - int(labels[candidate]),
                    )
                    impossible = int(
                        io_deficit_after + control_deficit_after > bound_remaining
                    )
                    return (
                        impossible,
                        -len(adjacency[candidate] & bound_component),
                        _stable_hash("selection-frontier-v1", candidate),
                    )

                current = min(frontier, key=priority)
                frontier.remove(current)
                selected.append(current)
                selected_set.add(current)
                frontier.update((adjacency[current] & component) - selected_set)
            if len(selected) == TARGET_NODE_COUNT:
                io_count = sum(not labels[item] for item in selected)
                control_count = TARGET_NODE_COUNT - io_count
                if io_count >= MIN_IO_NODES and control_count >= MIN_CONTROL_NODES:
                    return tuple(selected)
    raise SourceValidationError(
        "source cannot produce a connected 128 nodes with at least 16 IO and 64 controls"
    )


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, version=(1, 0), allow_pickle=False)
    return stream.getvalue()


def _fixed_zip_bytes(entries: Sequence[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", allowZip64=False) as archive:
        for name, value in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value, compresslevel=9)
    return stream.getvalue()


def _features_npz(node_ids: Sequence[str], features: np.ndarray) -> bytes:
    return _fixed_zip_bytes(
        (
            ("node_ids.npy", _npy_bytes(np.asarray(node_ids))),
            ("text_features.npy", _npy_bytes(np.asarray(features, dtype=np.float32))),
        )
    )


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _label_recipe(
    selected: Sequence[str], labels: Mapping[str, bool], degree: Mapping[str, int]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    structural_strata = {
        node: min(3, math.floor(rank * 4 / len(selected)))
        for rank, node in enumerate(
            sorted(selected, key=lambda item: (degree[item], anonymized_node_id(item)))
        )
    }
    for control_value, label in ((False, "io"), (True, "control")):
        for stratum in range(4):
            nodes = [
                node
                for node in selected
                if labels[node] is control_value and structural_strata[node] == stratum
            ]
            chosen = sorted(nodes, key=lambda node: _stable_hash("label-choice-v1", node))[:2]
            if len(chosen) != 2:
                raise SourceValidationError("selected graph lacks structural label strata coverage")
            result.extend(
                {
                    "nodeId": anonymized_node_id(node),
                    "label": label,
                    "structuralStratum": structural_strata[node],
                    "fusedDegree": degree[node],
                }
                for node in chosen
            )
    return sorted(result, key=lambda item: item["nodeId"])


def _generate_thailand_package_with_encoder(
    source_directory: str | Path,
    runtime_root: str | Path,
    output: str | Path,
    *,
    encoder: _Encoder,
) -> ThailandPackage:
    """Private deterministic seam used by the pinned public loader and tests."""

    runtime = _validate_runtime(Path(runtime_root))
    source = _inside(Path(source_directory), runtime, label="source directory")
    if not source.is_dir() or _is_reparse(source):
        raise SourceValidationError("source directory is missing or unsafe")
    destination = _inside(Path(output), runtime, label="output path")
    if destination.suffix.lower() != ".zip":
        raise SourceValidationError("output path must end in .zip")
    labels_path = destination.with_suffix(".labels.json")
    receipt_path = destination.with_suffix(".receipt.json")
    if any(path.exists() for path in (destination, labels_path, receipt_path)):
        raise FileExistsError("target package outputs already exist")

    authorization, _source_file, source_bytes = _validate_authorization(source, runtime)
    posts = _load_posts(source_bytes)
    encoder_description = _encoder_description(encoder)
    accounts, features = _encode_account_content(posts, encoder)
    account_labels = {post.accountid: post.is_control for post in posts}
    relations = _derive_relations(posts, accounts, features)
    selected = _connected_selection(accounts, relations, account_labels)
    selected_set = set(selected)
    selected_relations = [
        relation
        for relation in relations
        if relation.source in selected_set and relation.target in selected_set
    ]
    modalities = tuple(
        modality
        for modality in MODALITIES
        if any(relation.modality == modality for relation in selected_relations)
    )
    if len(modalities) < MIN_MODALITIES:
        raise SourceValidationError("selected graph has fewer than four non-empty modalities")
    fused_neighbors: dict[str, set[str]] = defaultdict(set)
    for relation in selected_relations:
        fused_neighbors[relation.source].add(relation.target)
        fused_neighbors[relation.target].add(relation.source)
    degree = Counter({account: len(fused_neighbors[account]) for account in selected})
    if any(degree[account] == 0 for account in selected):
        raise SourceValidationError("selected graph contains an isolate")

    node_order = sorted(selected, key=anonymized_node_id)
    feature_by_account = {account: features[index] for index, account in enumerate(accounts)}
    selected_features = np.vstack([feature_by_account[account] for account in node_order]).astype(
        np.float32, copy=False
    )
    node_id_by_account = {account: anonymized_node_id(account) for account in node_order}
    nodes_bytes = _csv_bytes(
        ("node_id", "display_name"),
        [
            (node_id_by_account[account], f"Anonymous Thailand account {index}")
            for index, account in enumerate(node_order)
        ],
    )
    relation_rows = sorted(
        (
            min(node_id_by_account[item.source], node_id_by_account[item.target]),
            max(node_id_by_account[item.source], node_id_by_account[item.target]),
            item.modality,
            format(item.weight, ".9g"),
        )
        for item in selected_relations
    )
    relations_bytes = _csv_bytes(("source", "target", "modality", "weight"), relation_rows)
    features_bytes = _features_npz(
        [node_id_by_account[account] for account in node_order], selected_features
    )
    files = {
        name: {"sha256": _sha256_bytes(value), "bytes": len(value)}
        for name, value in (
            ("nodes.csv", nodes_bytes),
            ("relations.csv", relations_bytes),
            ("features.npz", features_bytes),
        )
    }
    manifest = GovernanceInputManifest.model_validate(
        {
            "schemaVersion": INPUT_SCHEMA_VERSION,
            "datasetId": f"governance:thailand:{authorization['sourceSha256'][:16]}",
            "displayName": "Authorized Thailand target graph",
            "nodeCount": TARGET_NODE_COUNT,
            "relationRowCount": len(relation_rows),
            "featureDimension": 768,
            "modalities": modalities,
            "files": files,
            "license": authorization["license"],
        }
    )
    manifest_bytes = _canonical_bytes(manifest.model_dump(mode="json"))
    bundle_bytes = _fixed_zip_bytes(
        (
            ("manifest.json", manifest_bytes),
            ("nodes.csv", nodes_bytes),
            ("relations.csv", relations_bytes),
            ("features.npz", features_bytes),
        )
    )
    selection_recipe = {
        "version": "connected-structural-hash-v2",
        "nodeCount": TARGET_NODE_COUNT,
        "requiredIo": MIN_IO_NODES,
        "requiredControls": MIN_CONTROL_NODES,
        "minimumNonemptyModalities": MIN_MODALITIES,
        "scoreInputs": [],
        "groupRelations": {
            "maxGroupAccounts": GROUP_RELATION_MAX_ACCOUNTS,
            "totalPotentialPairBudget": GROUP_RELATION_PAIR_BUDGET,
        },
        "fastRT": {
            "windowSeconds": int(FAST_RT_WINDOW_SECONDS),
            "pairBudget": FAST_RT_PAIR_BUDGET,
            "algorithm": "sorted-sliding-window-v1",
        },
        "tweetSim": {
            "mutualTopK": TWEET_SIM_TOP_K,
            "cosineThreshold": TWEET_SIM_THRESHOLD,
            "pairBudget": TWEET_SIM_PAIR_BUDGET,
        },
    }
    label_rows = _label_recipe(node_order, account_labels, degree)
    label_selection_recipe = {
        "version": "graph-fused-degree-quartile-stable-hash-v2",
        "stratification": "graph-fused-degree-rank-quartile",
        "structuralStrata": 4,
        "labelsPerClass": LABELS_PER_CLASS,
        "labelsPerClassPerStratum": 2,
        "scoreInputs": [],
    }
    labels_document = {
        "schemaVersion": LABEL_SCHEMA_VERSION,
        "datasetId": manifest.datasetId,
        "bundleSha256": _sha256_bytes(bundle_bytes),
        "selectionRecipe": label_selection_recipe,
        "labels": label_rows,
    }
    labels_bytes = _canonical_bytes(labels_document)
    receipt_document = {
        "schemaVersion": RECEIPT_SCHEMA_VERSION,
        "datasetId": manifest.datasetId,
        "sourceSchemaVersion": SOURCE_SCHEMA_VERSION,
        "sourceSha256": authorization["sourceSha256"],
        "authorizationReference": authorization["authorizationReference"],
        "bundleSha256": _sha256_bytes(bundle_bytes),
        "labelsSha256": _sha256_bytes(labels_bytes),
        "encoder": encoder_description,
        "selectionRecipe": selection_recipe,
        "labelSelectionRecipe": label_selection_recipe,
        "coverage": {
            "nodeCount": len(node_order),
            "ioCount": sum(not account_labels[account] for account in node_order),
            "controlCount": sum(account_labels[account] for account in node_order),
            "nonemptyModalities": list(modalities),
            "connected": True,
        },
    }
    receipt_document["receiptHash"] = canonical_sha256(receipt_document)
    receipt_bytes = _canonical_bytes(receipt_document)

    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_segments(runtime, destination.parent, label="output parent")
    with tempfile.TemporaryDirectory(prefix="thailand-package-", dir=destination.parent) as raw:
        staging = Path(raw)
        staged_bundle = staging / destination.name
        staged_labels = staging / labels_path.name
        staged_receipt = staging / receipt_path.name
        staged_bundle.write_bytes(bundle_bytes)
        staged_labels.write_bytes(labels_bytes)
        staged_receipt.write_bytes(receipt_bytes)
        os.replace(staged_bundle, destination)
        os.replace(staged_labels, labels_path)
        os.replace(staged_receipt, receipt_path)
    return ThailandPackage(destination, labels_path, receipt_path)


def generate_thailand_package(
    source_directory: str | Path,
    runtime_root: str | Path,
    output: str | Path,
    *,
    encoder_cache: str | Path,
) -> ThailandPackage:
    """Generate a package with the internally pinned, offline-only production encoder."""

    runtime = _validate_runtime(Path(runtime_root))
    encoder = _load_production_encoder(Path(encoder_cache), runtime)
    return _generate_thailand_package_with_encoder(
        source_directory,
        runtime,
        output,
        encoder=encoder,
    )


__all__ = [
    "AUTHORIZATION_SCHEMA_VERSION",
    "PINNED_ENCODER_MODEL_ID",
    "PINNED_ENCODER_REVISION",
    "RUNTIME_AUTHORIZATION_SCHEMA_VERSION",
    "SOURCE_SCHEMA_VERSION",
    "SourceValidationError",
    "ThailandPackage",
    "aggregate_account_content",
    "anonymized_node_id",
    "generate_thailand_package",
]
