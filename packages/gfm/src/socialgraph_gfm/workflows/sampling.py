"""Domain streams, temporal sampling, product batches, and memory probes.

The implementation is installed into the shared compatibility namespace by
:mod:`socialgraph_gfm.workflows` after all workflow stages are imported.
"""

# ruff: noqa: F403, F405
# mypy: disable-error-code=name-defined
from __future__ import annotations

from ._shared import *


@dataclass
class _DomainStream:
    domain_id: str
    manifest: dict[str, Any]
    src: np.ndarray
    dst: np.ndarray
    timestamp: np.ndarray
    relation: np.ndarray
    node_type: np.ndarray
    node_count: int
    train_end: int
    validation_end: int
    relation_offset: int
    text_embedding: np.ndarray | None
    text_id_hash: np.ndarray | None
    text_timestamp: np.ndarray | None
    text_node_offset: int | None
    work_id_hash: np.ndarray | None
    work_publication_timestamp: np.ndarray | None
    work_cluster: np.ndarray | None
    cursor: int
    epoch: int = 0
    text_store: _BoundedEmbeddingStore | None = None
    text_event_id: np.ndarray | None = None
    incident_offsets: np.ndarray | None = None
    incident_event_indices: np.ndarray | None = None
    work_hash_to_index: dict[int, int] | None = None
    # Most corpora use contiguous temporal cutoffs.  Wikimedia instead assigns
    # a whole page (including its older history) to exactly one split, so its
    # event rows are intentionally non-contiguous by split.
    event_split: np.ndarray | None = None
    split_event_indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    cumulative_split_event_indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    maximum_access_role: Literal["train", "validation", "test", "shadow"] = "validation"
    access_audit: dict[str, Any] | None = None
    negative_sampling_audit: dict[str, Any] = field(default_factory=dict)
    lodo_eligible_pool_cache: dict[tuple[tuple[int, int], int], np.ndarray] = field(
        default_factory=dict
    )
    lodo_eligibility_cache_path: Path | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "domainId": self.domain_id,
            "cursor": self.cursor,
            "epoch": self.epoch,
            "trainEnd": self.train_end,
            "contentHash": self.manifest["logicalHash"],
            "negativeSamplingAudit": deepcopy(self.negative_sampling_audit),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        expected = {
            "domainId": self.domain_id,
            "trainEnd": self.train_end,
            "contentHash": self.manifest["logicalHash"],
        }
        if any(value.get(key) != item for key, item in expected.items()):
            raise ContractViolation("GFM domain stream resume identity differs")
        audit = value.get("negativeSamplingAudit", {})
        if not isinstance(audit, dict):
            raise ContractViolation("GFM negative sampling resume audit is invalid")
        cursor, epoch = int(value["cursor"]), int(value["epoch"])
        lower, train_count = _stream_role_bounds(self, 0)
        if not lower <= cursor <= train_count or epoch < 0:
            raise ContractViolation("GFM domain stream resume cursor is invalid")
        self.cursor, self.epoch = cursor, epoch
        self.negative_sampling_audit = deepcopy(audit)


def _build_incident_index(
    src: np.ndarray, dst: np.ndarray, *, node_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build a compact node -> chronologically sorted event-index CSR."""

    event_indices = np.arange(src.shape[0], dtype=np.int64)
    non_self = src != dst
    endpoint_nodes = np.concatenate((src, dst[non_self])).astype(np.int64, copy=False)
    endpoint_events = np.concatenate((event_indices, event_indices[non_self])).astype(
        np.int64, copy=False
    )
    order = np.lexsort((endpoint_events, endpoint_nodes))
    sorted_nodes = endpoint_nodes[order]
    sorted_events = np.ascontiguousarray(endpoint_events[order])
    counts = np.bincount(sorted_nodes, minlength=node_count)
    offsets = np.empty(node_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    offsets.setflags(write=False)
    sorted_events.setflags(write=False)
    return offsets, sorted_events


def _ensure_stream_indices(stream: _DomainStream) -> None:
    if stream.incident_offsets is None or stream.incident_event_indices is None:
        offsets, event_indices = _build_incident_index(
            stream.src, stream.dst, node_count=stream.node_count
        )
        stream.incident_offsets = offsets
        stream.incident_event_indices = event_indices
    if stream.event_split is not None and stream.split_event_indices is None:
        split = np.asarray(stream.event_split)
        if split.dtype != np.dtype(np.int8):
            raise ContractViolation("Explicit event split dtype must be int8")
        if split.shape != stream.timestamp.shape or bool(np.any((split < 0) | (split > 2))):
            raise ContractViolation("Explicit event split inventory is invalid")
        target_masks = (
            (split == 0) & (stream.timestamp <= stream.train_end),
            (split == 1)
            & (stream.timestamp > stream.train_end)
            & (stream.timestamp <= stream.validation_end),
            (split == 2) & (stream.timestamp > stream.validation_end),
        )
        # A page's older history stays available as causal message context, but
        # only events in the role's calendar window are prediction targets.
        per_role = tuple(
            np.ascontiguousarray(np.flatnonzero(mask), dtype=np.int64) for mask in target_masks
        )
        cumulative = tuple(
            np.ascontiguousarray(np.flatnonzero(split <= role), dtype=np.int64) for role in range(3)
        )
        maximum_role_index = ("train", "validation", "test", "shadow").index(
            stream.maximum_access_role
        )
        required_role_index = min(maximum_role_index, 2)
        if any(per_role[role].size == 0 for role in range(required_role_index + 1)):
            raise ContractViolation("Every authorised explicit event split must be nonempty")
        for indices in (*per_role, *cumulative):
            indices.setflags(write=False)
        stream.split_event_indices = per_role  # type: ignore[assignment]
        stream.cumulative_split_event_indices = cumulative  # type: ignore[assignment]
    if stream.domain_id == DOMAIN_IDS["openalex"]:
        if stream.work_id_hash is None:
            raise ContractViolation("OpenAlex stream lacks work ID hashes")
        if stream.work_hash_to_index is None:
            lookup = {int(value): index for index, value in enumerate(stream.work_id_hash.tolist())}
            if len(lookup) != int(stream.work_id_hash.shape[0]):
                raise ContractViolation("OpenAlex work ID hashes are not unique")
            stream.work_hash_to_index = lookup


def _stream_role_indices(stream: _DomainStream, role: int) -> np.ndarray:
    if role not in (0, 1, 2):
        raise ValueError("GFM split role must be train=0, validation=1, or test=2")
    if stream.event_split is not None:
        _ensure_stream_indices(stream)
        assert stream.split_event_indices is not None
        return stream.split_event_indices[role]
    train_count = int(np.searchsorted(stream.timestamp, stream.train_end, side="right"))
    validation_count = int(np.searchsorted(stream.timestamp, stream.validation_end, side="right"))
    lower, upper = (
        (0, train_count)
        if role == 0
        else (train_count, validation_count)
        if role == 1
        else (validation_count, int(stream.timestamp.size))
    )
    return np.arange(lower, upper, dtype=np.int64)


def _stream_role_bounds(stream: _DomainStream, role: int) -> tuple[int, int]:
    indices = _stream_role_indices(stream, role)
    # A training target needs at least one train-visible historical event.
    lower = 1 if role == 0 else 0
    if indices.size <= lower:
        raise ContractViolation(f"{stream.domain_id} split role {role} is too small")
    return lower, int(indices.size)


def _stream_visible_indices(stream: _DomainStream, role: int, *, end: int) -> np.ndarray:
    """Return event rows visible before a global row boundary for one split role."""

    if stream.event_split is None:
        return np.arange(end, dtype=np.int64)
    _ensure_stream_indices(stream)
    assert stream.cumulative_split_event_indices is not None
    inventory = stream.cumulative_split_event_indices[role]
    return inventory[: int(np.searchsorted(inventory, end, side="left"))]


def _split_bounds(domain_id: str, manifest: Mapping[str, Any]) -> tuple[int, int]:
    if domain_id == DOMAIN_IDS["openalex"]:
        train = int(datetime(2021, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp())
        validation = int(datetime(2022, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp())
        return train, validation
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ContractViolation(f"{domain_id} manifest lacks temporal split bounds")
    return int(splits["trainEndInclusive"]), int(splits["validationEndInclusive"])


def _typed_node_inventory(
    domain_id: str, manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, int, int | None]:
    if domain_id == DOMAIN_IDS["thgl-software"]:
        value = np.asarray(arrays["node_type"], dtype=np.int64)
        return value, int(value.shape[0]), None
    if domain_id == DOMAIN_IDS["openalex"]:
        counts = manifest["nodeCounts"]
        offsets = manifest["nodeOffsets"]
        node_count = sum(int(value) for value in counts.values())
        value = np.empty(node_count, dtype=np.int64)
        for type_index, name in enumerate(("author", "work", "institution", "topic")):
            start = int(offsets[name])
            value[start : start + int(counts[name])] = type_index
        return value, node_count, int(offsets["work"])
    users, pages = int(manifest["userCount"]), int(manifest["pageCount"])
    offsets = manifest.get("nodeOffsets")
    if not isinstance(offsets, dict) or int(offsets.get("page", -1)) != users:
        raise ContractViolation(
            "Wikimedia numeric corpus does not use disjoint user/page node offsets"
        )
    value = np.asarray(arrays.get("node_type"), dtype=np.int64)
    if value.shape != (users + pages,) or not np.array_equal(
        value,
        np.concatenate((np.zeros(users, dtype=np.int64), np.ones(pages, dtype=np.int64))),
    ):
        raise ContractViolation("Wikimedia node_type array differs from typed offsets")
    return value, users + pages, users


def _make_domain_streams(
    layout: RuntimeLayout,
    embeddings: Mapping[str, _BoundedEmbeddingStore],
    *,
    domain_ids: Sequence[str] | None = None,
    maximum_role: Literal["train", "validation", "test", "shadow"] = "validation",
) -> dict[str, _DomainStream]:
    result: dict[str, _DomainStream] = {}
    selected = tuple(DOMAIN_IDS.values()) if domain_ids is None else tuple(domain_ids)
    if len(selected) != len(set(selected)) or not set(selected).issubset(DOMAIN_IDS.values()):
        raise ContractViolation("Domain stream selection is invalid")
    if maximum_role not in ("train", "validation", "test", "shadow"):
        raise ContractViolation("Domain stream maximum role is invalid")
    for domain_id in selected:
        families = (
            ("events", "works") if domain_id == DOMAIN_IDS["openalex"] else ("events", "nodes")
        )
        loaded = load_domain_view(
            layout.root,
            domain_id,
            maximum_role=maximum_role,
            families=families,
        )
        manifest, arrays = loaded["manifest"], loaded["arrays"]
        access_audit = loaded.get("accessAudit")
        if (
            not isinstance(access_audit, dict)
            or access_audit.get("maximumRole") != maximum_role
            or access_audit.get("testArtifactsOpened") is not False
            and maximum_role in {"train", "validation"}
        ):
            raise ContractViolation("Domain role-view access audit is invalid")
        required = {"src", "dst", "timestamp", "relation"}
        if not required.issubset(arrays):
            raise ContractViolation(f"{domain_id} lacks aligned temporal event arrays")
        src = np.asarray(arrays["src"], dtype=np.int64)
        dst = np.asarray(arrays["dst"], dtype=np.int64)
        timestamp = np.asarray(arrays["timestamp"], dtype=np.int64)
        relation = np.asarray(arrays["relation"], dtype=np.int64)
        if not (
            src.shape == dst.shape == timestamp.shape == relation.shape
            and src.ndim == 1
            and src.size > 1
        ):
            raise ContractViolation(f"{domain_id} event arrays are misaligned or empty")
        if np.any(timestamp[1:] < timestamp[:-1]):
            raise ContractViolation(f"{domain_id} events are not sorted by time")
        node_type, node_count, text_offset = _typed_node_inventory(domain_id, manifest, arrays)
        if src.min() < 0 or dst.min() < 0 or src.max() >= node_count or dst.max() >= node_count:
            raise ContractViolation(f"{domain_id} contains an out-of-range endpoint")
        train_end, validation_end = _split_bounds(domain_id, manifest)
        event_split = None
        if domain_id == DOMAIN_IDS["wikimedia-talk"]:
            if "split" not in arrays:
                raise ContractViolation(
                    "Wikimedia corpus lacks its page-disjoint event split array"
                )
            event_split = np.asarray(arrays["split"])
            if event_split.dtype != np.dtype(np.int8):
                raise ContractViolation("Wikimedia event split array must be int8")
            if event_split.shape != timestamp.shape or bool(
                np.any((event_split < 0) | (event_split > 2))
            ):
                raise ContractViolation("Wikimedia event split array is invalid")
            split_contract = manifest.get("splits")
            if (
                not isinstance(split_contract, dict)
                or split_contract.get("strategy") != "page-last-event-time"
                or split_contract.get("eventSplitArray") != "split"
                or split_contract.get("pageDisjoint") is not True
            ):
                raise ContractViolation("Wikimedia manifest does not bind the page-disjoint split")
            minimum_role = np.full(node_count, 3, dtype=np.int8)
            maximum_page_role = np.full(node_count, -1, dtype=np.int8)
            np.minimum.at(minimum_role, dst, event_split)
            np.maximum.at(maximum_page_role, dst, event_split)
            used_pages = maximum_page_role >= 0
            if bool(np.any(minimum_role[used_pages] != maximum_page_role[used_pages])):
                raise ContractViolation("A Wikimedia page appears in more than one workflow split")
            split_counts = (
                int(np.count_nonzero((event_split == 0) & (timestamp <= train_end))),
                int(
                    np.count_nonzero(
                        (event_split == 1) & (timestamp > train_end) & (timestamp <= validation_end)
                    )
                ),
                int(np.count_nonzero((event_split == 2) & (timestamp > validation_end))),
            )
            maximum_index = min(("train", "validation", "test", "shadow").index(maximum_role), 2)
            if (
                any(split_counts[index] < 1 for index in range(maximum_index + 1))
                or split_counts[0] < 2
            ):
                raise ContractViolation(
                    "Wikimedia lacks nonempty page-disjoint train/validation/test splits"
                )
            train_count, validation_count = split_counts[0], split_counts[1]
        else:
            train_count = int(np.searchsorted(timestamp, train_end, side="right"))
            validation_count = int(np.searchsorted(timestamp, validation_end, side="right"))
            if train_count < 2 or validation_count <= train_count:
                raise ContractViolation(f"{domain_id} lacks nonempty train/validation intervals")
        text_embedding = text_timestamp = text_id_hash = None
        work_id_hash = (
            np.asarray(arrays["work_id_hash"], dtype=np.uint64)
            if "work_id_hash" in arrays
            else None
        )
        work_publication_timestamp = (
            np.asarray(arrays["publication_timestamp"], dtype=np.int64)
            if "publication_timestamp" in arrays
            else None
        )
        work_cluster = (
            np.asarray(arrays["cluster"], dtype=np.int64) if "cluster" in arrays else None
        )
        text_store = embeddings.get(domain_id)
        text_event_id = None
        if text_store is not None:
            if (
                text_store.maximum_role is None
                and int(text_store.manifest["rows"]) != text_store.handle.rows
            ) or (
                text_store.maximum_role is not None
                and text_store.handle.rows > int(text_store.manifest["rows"])
            ):
                raise ContractViolation("Text embedding rows differ from their handle")
            if domain_id == DOMAIN_IDS["wikimedia-talk"]:
                text_event_id = np.asarray(arrays.get("revision_pseudonym"), dtype=np.uint64)
                if text_event_id.shape != src.shape:
                    raise ContractViolation("Wikimedia text embeddings are not event-row aligned")
            elif domain_id == DOMAIN_IDS["openalex"]:
                text_store.build_hash_index()
        cursor = max(1, min(train_count - 1, train_count // 20))
        stream = _DomainStream(
            domain_id=domain_id,
            manifest=manifest,
            src=src,
            dst=dst,
            timestamp=timestamp,
            relation=relation,
            node_type=node_type,
            node_count=node_count,
            train_end=train_end,
            validation_end=validation_end,
            relation_offset=DOMAIN_RELATION_OFFSETS[domain_id],
            text_embedding=text_embedding,
            text_id_hash=text_id_hash,
            text_timestamp=text_timestamp,
            text_node_offset=text_offset,
            work_id_hash=work_id_hash,
            work_publication_timestamp=work_publication_timestamp,
            work_cluster=work_cluster,
            cursor=cursor,
            text_store=text_store,
            text_event_id=text_event_id,
            event_split=event_split,
            maximum_access_role=maximum_role,
            access_audit=dict(access_audit),
            lodo_eligibility_cache_path=(layout.cache_torch / "lodo-eligibility" / domain_id),
        )
        _ensure_stream_indices(stream)
        result[domain_id] = stream
    return result


def _recent_causal_edges(
    stream: _DomainStream,
    *,
    end: int,
    seeds: set[int],
    fanout: tuple[int, int],
    maximum_split_role: int | None = None,
) -> np.ndarray:
    """Deterministic last-event causal fanout without consulting future rows."""

    if end < 1:
        return np.empty(0, dtype=np.int64)
    if end > stream.timestamp.shape[0]:
        raise GfmTrainingError("Causal sampler end exceeds the event inventory")
    _ensure_stream_indices(stream)
    assert stream.incident_offsets is not None
    assert stream.incident_event_indices is not None
    frontier = set(seeds)
    selected: set[int] = set()
    # Each node's CSR segment is event-index sorted, which is also temporal
    # order because corpus events are stably sorted.  searchsorted(end) makes
    # it impossible to dereference an event at or after the sample boundary.
    for limit in fanout:
        layer_events: set[int] = set()
        for node in frontier:
            if node < 0 or node >= stream.node_count:
                raise GfmTrainingError("Causal sampler seed is outside the node inventory")
            lower = int(stream.incident_offsets[node])
            upper = int(stream.incident_offsets[node + 1])
            incident = stream.incident_event_indices[lower:upper]
            visible = int(np.searchsorted(incident, end, side="left"))
            eligible = incident[:visible]
            if stream.event_split is not None:
                if maximum_split_role is None:
                    raise GfmTrainingError(
                        "Explicit-split causal sampling requires a maximum split role"
                    )
                eligible = eligible[stream.event_split[eligible] <= maximum_split_role]
            layer_events.update(int(value) for value in eligible[max(0, eligible.size - limit) :])
        if not layer_events:
            break
        ordered = np.fromiter(sorted(layer_events), dtype=np.int64)
        if bool(np.any(ordered >= end)):
            raise GfmTrainingError("Causal incident index exposed a future event")
        selected.update(layer_events)
        layer = set(stream.src[ordered].tolist()) | set(stream.dst[ordered].tolist())
        frontier = layer.difference(frontier)
        if not frontier:
            break
    return np.asarray(sorted(selected), dtype=np.int64)


def _visible_edges_between_local_nodes(
    stream: _DomainStream,
    *,
    end: int,
    local_nodes: np.ndarray,
    maximum_split_role: int | None = None,
) -> np.ndarray:
    """Return every pre-boundary edge whose two endpoints are in ``local_nodes``.

    The encoder still receives only the bounded fanout graph.  Exact negative
    exclusion, however, must consult the complete cutoff-visible graph or a
    historical edge omitted by fanout could be mislabeled as a negative.  The
    incident CSR keeps this proportional to the selected nodes' histories
    instead of rescanning every event for every microbatch.
    """

    if end < 1 or end > stream.timestamp.size:
        raise GfmTrainingError("Visible-edge boundary is outside the event inventory")
    _ensure_stream_indices(stream)
    assert stream.incident_offsets is not None
    assert stream.incident_event_indices is not None
    selected: set[int] = set()
    inventory = {int(node) for node in local_nodes.tolist()}
    for node in inventory:
        lower = int(stream.incident_offsets[node])
        upper = int(stream.incident_offsets[node + 1])
        incident = stream.incident_event_indices[lower:upper]
        visible = int(np.searchsorted(incident, end, side="left"))
        eligible = incident[:visible]
        if stream.event_split is not None:
            if maximum_split_role is None:
                raise GfmTrainingError(
                    "Explicit-split negative inventory requires a maximum split role"
                )
            eligible = eligible[stream.event_split[eligible] <= maximum_split_role]
        selected.update(int(value) for value in eligible)
    if not selected:
        raise GfmTrainingError("Local negative inventory has no cutoff-visible edges")
    indices = np.fromiter(sorted(selected), dtype=np.int64)
    keep = np.fromiter(
        (
            int(stream.src[index]) in inventory and int(stream.dst[index]) in inventory
            for index in indices.tolist()
        ),
        dtype=np.bool_,
        count=indices.size,
    )
    result = indices[keep]
    if (
        result.size == 0
        or bool(np.any(result >= end))
        or (
            stream.event_split is not None
            and maximum_split_role is not None
            and bool(np.any(stream.event_split[result] > maximum_split_role))
        )
    ):
        raise GfmTrainingError("Local negative inventory is empty or exposed a future edge")
    return result


def _local_text(
    stream: _DomainStream,
    *,
    local_nodes: np.ndarray,
    message_indices: np.ndarray,
    cutoff: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if stream.text_store is None:
        # Retain the narrow in-memory path for synthetic unit fixtures only;
        # production streams always use the bounded verified shard store.
        if (
            stream.text_embedding is None
            or stream.text_id_hash is None
            or stream.text_timestamp is None
        ):
            return None
    if stream.text_store is None:
        embedding_store = None
    else:
        embedding_store = stream.text_store
    if embedding_store is None and stream.text_embedding is None:
        return None
    values = np.zeros((local_nodes.shape[0], 1024), dtype=np.float32)
    mask = np.zeros(local_nodes.shape[0], dtype=np.bool_)
    local_index = {int(value): index for index, value in enumerate(local_nodes)}
    if stream.domain_id == DOMAIN_IDS["openalex"]:
        if stream.text_node_offset is None:
            raise ContractViolation("OpenAlex text-to-work offset is absent")
        if stream.work_id_hash is None:
            raise ContractViolation("OpenAlex manifest lacks explicit work ID hash alignment")
        _ensure_stream_indices(stream)
        assert stream.work_hash_to_index is not None
        local_work_nodes = local_nodes[
            (local_nodes >= stream.text_node_offset)
            & (local_nodes < stream.text_node_offset + int(stream.work_id_hash.shape[0]))
        ]
        requested_hashes = stream.work_id_hash[local_work_nodes - stream.text_node_offset]
        if embedding_store is not None:
            found = embedding_store.lookup_hashes(requested_hashes)
            for node, id_hash in zip(
                local_work_nodes.tolist(), requested_hashes.tolist(), strict=True
            ):
                matched_row = found.get(int(id_hash))
                if matched_row is None:
                    continue
                embedding, timestamp = matched_row
                if timestamp <= cutoff:
                    local_position = local_index[int(node)]
                    values[local_position] = embedding
                    mask[local_position] = True
        else:
            assert stream.text_id_hash is not None
            assert stream.text_timestamp is not None
            assert stream.text_embedding is not None
            requested = {int(value) for value in requested_hashes.tolist()}
            for row_index, timestamp in enumerate(stream.text_timestamp):
                id_hash = int(stream.text_id_hash[row_index])
                if id_hash not in requested:
                    continue
                work_index = stream.work_hash_to_index.get(id_hash)
                if work_index is None:
                    raise ContractViolation("BGE text ID does not map to an OpenAlex work node")
                node = stream.text_node_offset + work_index
                legacy_local = local_index.get(node)
                if legacy_local is not None and int(timestamp) <= cutoff:
                    values[legacy_local] = stream.text_embedding[row_index]
                    mask[legacy_local] = True
    else:
        if embedding_store is not None:
            if stream.text_event_id is None:
                raise ContractViolation("Wikimedia text event IDs are absent")
            expected_hashes = np.asarray(
                [
                    portable_id_hash(str(int(stream.text_event_id[index])))
                    for index in message_indices
                ],
                dtype=np.uint64,
            )
            found = embedding_store.lookup_hashes(expected_hashes)
            if len(found) != expected_hashes.size:
                raise ContractViolation(
                    "Wikimedia embedding hash lookup did not cover message events"
                )
            embeddings = np.stack([found[int(id_hash)][0] for id_hash in expected_hashes], axis=0)
            timestamps = np.asarray(
                [found[int(id_hash)][1] for id_hash in expected_hashes],
                dtype=np.int64,
            )
            if not np.array_equal(timestamps, stream.timestamp[message_indices]):
                raise ContractViolation(
                    "Wikimedia embedding rows differ from event ID/time alignment"
                )
        else:
            assert stream.text_embedding is not None
            assert stream.text_timestamp is not None
            if stream.text_embedding.shape[0] != stream.src.shape[0]:
                raise ContractViolation("Wikimedia comment embeddings are not event-aligned")
            embeddings = stream.text_embedding[message_indices]
            timestamps = stream.text_timestamp[message_indices]
        sums: dict[int, np.ndarray] = {}
        counts: defaultdict[int, int] = defaultdict(int)
        for row_index, event_index in enumerate(message_indices):
            node = int(stream.dst[event_index])
            page_local = local_index.get(node)
            if page_local is None or int(timestamps[row_index]) > cutoff:
                continue
            if page_local not in sums:
                sums[page_local] = np.zeros(1024, dtype=np.float32)
            sums[page_local] += embeddings[row_index]
            counts[page_local] += 1
        for local_position, total in sums.items():
            value = total / counts[local_position]
            norm = float(np.linalg.norm(value))
            if norm > 0:
                values[local_position] = value / norm
                mask[local_position] = True
    return values, mask


def _negative_tails(
    *,
    source: np.ndarray,
    positive_target: np.ndarray,
    node_type: np.ndarray,
    message_src: np.ndarray,
    message_dst: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Exact typed negatives with a fixed hard/degree/uniform 50/25/25 cycle."""

    adjacency: defaultdict[int, set[int]] = defaultdict(set)
    degree: defaultdict[int, int] = defaultdict(int)
    forbidden = set()
    for left, right in zip(message_src.tolist(), message_dst.tolist(), strict=True):
        adjacency[int(left)].add(int(right))
        adjacency[int(right)].add(int(left))
        degree[int(left)] += 1
        degree[int(right)] += 1
        forbidden.add((int(left), int(right)))
    positives = set(zip(source.tolist(), positive_target.tolist(), strict=True))
    visible_nodes = sorted(set(message_src.tolist()) | set(message_dst.tolist()))
    rng = np.random.default_rng(seed)
    result: list[int] = []
    used: set[tuple[int, int]] = set()
    cycle = ("hard", "hard", "degree", "uniform")
    for index, (left, right) in enumerate(
        zip(source.tolist(), positive_target.tolist(), strict=True)
    ):
        candidates = [
            node
            for node in visible_nodes
            if node != left
            and node_type[node] == node_type[right]
            and (left, node) not in forbidden
            and (left, node) not in positives
            and (left, node) not in used
        ]
        if not candidates:
            raise GfmTrainingError("No exact typed negative exists at the current cutoff")
        mode = cycle[index % len(cycle)]
        if mode == "hard":
            two_hop = (
                set().union(*(adjacency[node] for node in adjacency[left]))
                if adjacency[left]
                else set()
            )
            hard = sorted(set(adjacency[left]) | two_hop)
            selected_pool = [node for node in candidates if node in hard]
            if not selected_pool:
                selected_pool = candidates
            chosen = selected_pool[index % len(selected_pool)]
        elif mode == "degree":
            chosen = min(candidates, key=lambda node: (abs(degree[node] - degree[right]), node))
        else:
            chosen = int(candidates[int(rng.integers(0, len(candidates)))])
        result.append(chosen)
        used.add((left, chosen))
    return np.asarray(result, dtype=np.int64)


def _record_negative_sampling_audit(
    stream: _DomainStream,
    sample: Any,
    *,
    requested_positive_count: int,
    retained_positive_count: int,
    split_role: int,
    cursor: int,
    cutoff: int,
) -> None:
    """Accumulate the requested and actual mixture without relabelling fallbacks.

    A hard-pool miss is a normal cold-start condition in sparse temporal
    graphs.  It is retained as useful supervision only when the replacement is
    an exact typed uniform non-edge, and is counted under its fallback label.
    This audit therefore makes the effective positive ratio and the actual
    (possibly non-50/25/25) mixture explicit to checkpoints and run reports.
    """

    if requested_positive_count < 1 or not 0 < retained_positive_count <= requested_positive_count:
        raise GfmTrainingError("Negative sampling audit has invalid positive counts")
    roles = ("train", "validation", "test")
    if split_role not in range(len(roles)):
        raise GfmTrainingError("Negative sampling audit has an invalid split role")
    requested_counts = {
        str(name): int(value) for name, value in sample.requested_component_counts.items()
    }
    actual_counts = {
        str(name): int(value) for name, value in sample.actual_component_counts.items()
    }
    negative_count = int(sample.edge_index.shape[1])
    if (
        sum(requested_counts.values()) != negative_count
        or sum(actual_counts.values()) != negative_count
    ):
        raise GfmTrainingError("Negative sampling component audit is inconsistent")
    fallback_positions = [
        position
        for position, label in enumerate(sample.component_labels)
        if label.endswith("_fallback_uniform")
    ]
    fallback_query_count = len(
        {position // int(sample.negatives_per_positive) for position in fallback_positions}
    )
    future_unseen_candidate_count = int(sample.future_unseen_candidate_count)
    if future_unseen_candidate_count != 0:
        raise GfmTrainingError(
            "Negative sampler emitted a tail absent from the cutoff-visible graph"
        )

    audit = deepcopy(stream.negative_sampling_audit)
    if not audit:
        audit = {
            "schemaVersion": "gfm.negative-sampling-audit/1.0",
            "policy": "causal-exact-typed-mixed-with-audited-uniform-fallback",
            "batchCount": 0,
            "requestedPositiveCount": 0,
            "retainedPositiveCount": 0,
            "effectivePositiveRatio": 0.0,
            "negativeCount": 0,
            "requestedComponentCounts": {},
            "actualComponentCounts": {},
            "requestedMix": {},
            "actualMix": {},
            "fallbackDrawCount": 0,
            "fallbackQueryCount": 0,
            "futureUnseenCandidateCount": 0,
            "bySplitRole": {},
            "exactNoFalseNegative": True,
            "typed": True,
            "causal": True,
            "cutoffVisibleCandidatesOnly": True,
            "queryLocalUnique": True,
        }
    if audit.get("schemaVersion") != "gfm.negative-sampling-audit/1.0":
        raise GfmTrainingError("Negative sampling audit schema changed during a run")

    def merge(summary: dict[str, Any]) -> None:
        summary["batchCount"] = int(summary.get("batchCount", 0)) + 1
        summary["requestedPositiveCount"] = (
            int(summary.get("requestedPositiveCount", 0)) + requested_positive_count
        )
        summary["retainedPositiveCount"] = (
            int(summary.get("retainedPositiveCount", 0)) + retained_positive_count
        )
        summary["negativeCount"] = int(summary.get("negativeCount", 0)) + negative_count
        summary["fallbackDrawCount"] = int(summary.get("fallbackDrawCount", 0)) + len(
            fallback_positions
        )
        summary["fallbackQueryCount"] = (
            int(summary.get("fallbackQueryCount", 0)) + fallback_query_count
        )
        summary["futureUnseenCandidateCount"] = (
            int(summary.get("futureUnseenCandidateCount", 0)) + future_unseen_candidate_count
        )
        for key, counts in (
            ("requestedComponentCounts", requested_counts),
            ("actualComponentCounts", actual_counts),
        ):
            merged = {str(name): int(value) for name, value in dict(summary.get(key, {})).items()}
            for name, value in counts.items():
                merged[name] = merged.get(name, 0) + value
            summary[key] = dict(sorted(merged.items()))
        requested_total = int(summary["requestedPositiveCount"])
        retained_total = int(summary["retainedPositiveCount"])
        total_negatives = int(summary["negativeCount"])
        summary["effectivePositiveRatio"] = retained_total / requested_total
        summary["requestedMix"] = {
            name: value / total_negatives
            for name, value in summary["requestedComponentCounts"].items()
        }
        summary["actualMix"] = {
            name: value / total_negatives
            for name, value in summary["actualComponentCounts"].items()
        }

    merge(audit)
    by_role = dict(audit["bySplitRole"])
    role = roles[split_role]
    role_summary = dict(by_role.get(role, {}))
    merge(role_summary)
    by_role[role] = role_summary
    audit["bySplitRole"] = by_role
    audit["lastBatch"] = {
        "splitRole": role,
        "cursor": cursor,
        "cutoff": cutoff,
        "requestedPositiveCount": requested_positive_count,
        "retainedPositiveCount": retained_positive_count,
        "effectivePositiveRatio": retained_positive_count / requested_positive_count,
        "negativeCount": negative_count,
        "requestedComponentCounts": requested_counts,
        "actualComponentCounts": actual_counts,
        "fallbackDrawCount": len(fallback_positions),
        "fallbackQueryCount": fallback_query_count,
        "futureUnseenCandidateCount": future_unseen_candidate_count,
    }
    stream.negative_sampling_audit = audit


def _core_batch(
    stream: _DomainStream,
    *,
    batch_size: int,
    fanout: tuple[int, int],
    seed: int,
    cursor: int | None = None,
    upper_index: int | None = None,
    advance: bool = True,
    allow_negative_fallback: bool = True,
    split_role: int | None = None,
    negatives_per_positive: int = 4,
) -> Any:
    import torch

    if negatives_per_positive < 4:
        raise ValueError("Mixed negative sampling requires at least four negatives")

    role_indexed = stream.event_split is not None or split_role is not None
    active_role = 0 if split_role is None else int(split_role)
    if role_indexed:
        target_inventory = _stream_role_indices(stream, active_role)
        lower, default_upper = _stream_role_bounds(stream, active_role)
        start = stream.cursor if cursor is None else int(cursor)
        upper = default_upper if upper_index is None else int(upper_index)
        if cursor is None and start >= upper:
            if active_role != 0:
                raise GfmTrainingError("Only the training split can advance epochs")
            stream.epoch += 1
            stream.cursor = max(lower, min(upper - 1, upper // 20))
            start = stream.cursor
        if not lower <= start < upper <= default_upper:
            raise GfmTrainingError("Explicit GFM split ordinal bounds are invalid")
        stop = min(start + batch_size, upper)
        target_indices = target_inventory[start:stop]
        message_end = int(target_indices[0])
        visible_before_target = _stream_visible_indices(stream, active_role, end=message_end)
        if visible_before_target.size == 0:
            raise GfmTrainingError(f"{stream.domain_id} target has no split-visible causal history")
        cutoff = int(stream.timestamp[int(visible_before_target[-1])])
    else:
        train_count = int(np.searchsorted(stream.timestamp, stream.train_end, side="right"))
        start = stream.cursor if cursor is None else int(cursor)
        upper = (
            train_count
            if cursor is None
            else int(np.searchsorted(stream.timestamp, stream.validation_end, side="right"))
        )
        if upper_index is not None:
            upper = int(upper_index)
            if not start < upper <= stream.timestamp.shape[0]:
                raise GfmTrainingError("Explicit GFM split upper bound is invalid")
        if start >= upper:
            if cursor is not None:
                raise GfmTrainingError(f"{stream.domain_id} has no events at the requested split")
            stream.epoch += 1
            stream.cursor = max(1, min(train_count - 1, train_count // 20))
            start = stream.cursor
        stop = min(start + batch_size, upper)
        cutoff = int(stream.timestamp[start - 1])
        target_indices = np.arange(start, stop, dtype=np.int64)
        message_end = start
    target_src = stream.src[target_indices]
    target_dst = stream.dst[target_indices]
    seeds = set(target_src.tolist()) | set(target_dst.tolist())
    message_indices = _recent_causal_edges(
        stream,
        end=message_end,
        seeds=seeds,
        fanout=fanout,
        maximum_split_role=active_role if stream.event_split is not None else None,
    )
    if message_indices.size == 0:
        raise GfmTrainingError(f"{stream.domain_id} produced an empty causal message graph")
    original_message_src = stream.src[message_indices]
    original_message_dst = stream.dst[message_indices]
    # The encoder aggregates source -> target.  Add the reverse view only to
    # the message graph so authors/users receive work/page context while the
    # temporal supervision retains the original event direction.
    message_src = np.concatenate((original_message_src, original_message_dst))
    message_dst = np.concatenate((original_message_dst, original_message_src))
    message_timestamps = np.concatenate(
        (stream.timestamp[message_indices], stream.timestamp[message_indices])
    )
    message_relations = np.concatenate(
        (stream.relation[message_indices], stream.relation[message_indices])
    )
    local_nodes = np.asarray(
        sorted(
            set(message_src.tolist())
            | set(message_dst.tolist())
            | set(target_src.tolist())
            | set(target_dst.tolist())
        ),
        dtype=np.int64,
    )
    lookup = {int(value): index for index, value in enumerate(local_nodes)}
    remap = np.vectorize(lookup.__getitem__, otypes=[np.int64])
    local_message_src, local_message_dst = remap(message_src), remap(message_dst)
    local_target_src, local_target_dst = remap(target_src), remap(target_dst)
    in_degree = np.bincount(local_message_dst, minlength=local_nodes.size)
    out_degree = np.bincount(local_message_src, minlength=local_nodes.size)
    activity = in_degree + out_degree
    numeric_target = np.stack(
        (
            np.log1p(in_degree),
            np.log1p(out_degree),
            np.log1p(activity),
        ),
        axis=1,
    ).astype(np.float32)
    attribute_mask = (np.arange(local_nodes.size) + seed + start) % 7 == 0
    if not attribute_mask.any():
        attribute_mask[0] = True
    numeric_input = numeric_target.copy()
    numeric_input[attribute_mask] = 0.0
    local_node_type = stream.node_type[local_nodes]
    if local_node_type.min() < 0 or local_node_type.max() >= 4:
        raise ContractViolation("SocialGraph-FM Core supports four local node types")
    node_type_values = np.eye(4, dtype=np.float32)[local_node_type]
    modalities: dict[str, Any] = {
        "numeric": torch.from_numpy(numeric_input),
        "node_type": torch.from_numpy(node_type_values),
    }
    modality_masks: dict[str, Any] = {
        "numeric": torch.ones(local_nodes.size, dtype=torch.bool),
        "node_type": torch.ones(local_nodes.size, dtype=torch.bool),
    }
    attribute_targets: dict[str, Any] = {"numeric": torch.from_numpy(numeric_target)}
    attribute_masks: dict[str, Any] = {"numeric": torch.from_numpy(attribute_mask)}
    text = _local_text(
        stream,
        local_nodes=local_nodes,
        message_indices=message_indices,
        cutoff=cutoff,
    )
    if text is not None:
        text_values, text_mask = text
        text_attribute_mask = text_mask & (((np.arange(local_nodes.size) + start) % 11) == 0)
        text_input = text_values.copy()
        text_input[text_attribute_mask] = 0.0
        modalities["text"] = torch.from_numpy(text_input)
        modality_masks["text"] = torch.from_numpy(text_mask)
        attribute_targets["text"] = torch.from_numpy(text_values)
        attribute_masks["text"] = torch.from_numpy(text_attribute_mask)
    # All three formal corpus schemas normalize event timestamps to Unix
    # seconds.  A fixed scale prevents validation/test rows from influencing
    # even preprocessing constants for a training batch.
    time_scale = 86_400.0
    relation = message_relations.astype(np.int64) + stream.relation_offset
    positive_relation = stream.relation[target_indices].astype(np.int64) + stream.relation_offset
    relation_mask = ((np.arange(target_indices.size) + start) % 3) == 0
    if not relation_mask.any():
        relation_mask[0] = True
    from ..gfm.sampling import CausalMixedNegativeSampler
    from ..gfm.types import CoreBatch, CoreSampleProvenance

    message_edge_index = torch.from_numpy(np.stack((local_message_src, local_message_dst)))
    positive_edge_index = torch.from_numpy(np.stack((local_target_src, local_target_dst)))
    # Unix epoch is the schema-level origin.  Never derive normalization from
    # row zero: in a page-disjoint corpus that row may belong to a held-out
    # page/split and would leak its timestamp into training preprocessing.
    message_edge_time = torch.from_numpy((message_timestamps / time_scale).astype(np.float32))
    # Bind the scalar cutoff to the same FP32 representation as edge_time so
    # an edge exactly at the cutoff cannot round one ULP into the future.
    cutoff_value = float(np.float32(cutoff / time_scale))
    full_visible_indices = _visible_edges_between_local_nodes(
        stream,
        end=message_end,
        local_nodes=local_nodes,
        maximum_split_role=active_role if stream.event_split is not None else None,
    )
    full_visible_src = remap(stream.src[full_visible_indices])
    full_visible_dst = remap(stream.dst[full_visible_indices])
    exact_visible_edge_index = torch.from_numpy(
        np.stack(
            (
                np.concatenate((full_visible_src, full_visible_dst)),
                np.concatenate((full_visible_dst, full_visible_src)),
            )
        )
    )
    exact_visible_times = np.concatenate(
        (
            stream.timestamp[full_visible_indices],
            stream.timestamp[full_visible_indices],
        )
    )
    exact_visible_edge_time = torch.from_numpy(
        (exact_visible_times / time_scale).astype(np.float32)
    )
    # Frozen validation/test candidates must not drift when training advances a
    # stream epoch.  Only an advancing train batch incorporates the epoch into
    # its sampler seed; explicit non-advancing role views are immutable.
    negative_seed = seed + start
    if advance and active_role == 0:
        negative_seed += stream.epoch * 1_000_003
    negative_sampler = CausalMixedNegativeSampler(
        source_count=local_nodes.size,
        target_count=local_nodes.size,
        visible_edge_index=exact_visible_edge_index,
        visible_edge_time=exact_visible_edge_time,
        cutoff_time=cutoff_value,
        seed=negative_seed,
        directed=True,
        same_node_space=True,
        node_types=torch.from_numpy(local_node_type),
    )
    requested_positive_count = int(positive_edge_index.shape[1])
    try:
        negative_sample = negative_sampler.sample(
            positive_edge_index,
            negatives_per_positive=negatives_per_positive,
            allow_batch_fallback=allow_negative_fallback,
        )
    except ValueError as error:
        # Strict callers fail the whole batch.  They never retain only the
        # easy positives, which would silently bias temporal supervision.
        raise GfmTrainingError(
            f"{stream.domain_id} cannot produce a complete exact typed negative batch: {error}"
        ) from error
    _record_negative_sampling_audit(
        stream,
        negative_sample,
        requested_positive_count=requested_positive_count,
        retained_positive_count=int(positive_edge_index.shape[1]),
        split_role=active_role,
        cursor=start,
        cutoff=cutoff,
    )

    batch = CoreBatch(
        domain_id=stream.domain_id,
        modalities=modalities,
        modality_masks=modality_masks,
        edge_index=message_edge_index,
        edge_type=torch.from_numpy(relation),
        edge_time=message_edge_time,
        cutoff_time=cutoff_value,
        provenance=CoreSampleProvenance(
            domain_id=stream.domain_id,
            graph_version=str(stream.manifest["logicalHash"]),
            cutoff=cutoff_value,
            horizon=max(
                1.0 / time_scale,
                float((int(stream.timestamp[target_indices].max()) - cutoff) / time_scale),
            ),
            task_id="pretrain.temporal_next_event",
            source_corpus_hash=str(stream.manifest["logicalHash"]),
        ),
        attribute_targets=attribute_targets,
        attribute_masks=attribute_masks,
        positive_edge_index=positive_edge_index,
        negative_edge_index=negative_sample.edge_index,
        positive_relation=torch.from_numpy(positive_relation),
        positive_relation_mask=torch.from_numpy(relation_mask),
        time_delta_targets=torch.from_numpy(
            ((stream.timestamp[target_indices] - cutoff) / time_scale).astype(np.float32)
        ),
        time_delta_mask=torch.ones(target_indices.size, dtype=torch.bool),
    )
    if advance and cursor is None:
        stream.cursor = stop
    return batch


def _model_config(
    config: GfmPretrainConfig,
    variant: str,
    *,
    domains: Sequence[str] | None = None,
) -> Any:
    from ..gfm.types import CoreModelConfig

    return CoreModelConfig(
        modality_dims={
            "numeric": 3,
            "node_type": 4,
            "text": config.text_encoder.output_channels,
        },
        domains=tuple(domains or DOMAIN_IDS.values()),
        num_relations=TOTAL_RELATIONS,
        hidden_channels=config.architecture.hidden_channels,
        num_layers=config.architecture.num_layers,
        time_channels=config.architecture.time_channels,
        relation_bases=config.architecture.relation_bases,
        domain_bottleneck=config.architecture.domain_adapter_bottleneck,
        expert_count=config.architecture.moe_experts,
        variant="moe" if variant == "core-moe" else "base",
        dropout=config.architecture.dropout,
        text_modality="text",
    )


@dataclass(frozen=True)
class _FeatureTransform:
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(cls, values: Sequence[np.ndarray] | Iterator[np.ndarray]) -> _FeatureTransform:
        count = 0
        mean = np.zeros(8, dtype=np.float64)
        square = np.zeros(8, dtype=np.float64)
        for source in values:
            matrix = np.asarray(source, dtype=np.float64).copy()
            if matrix.ndim != 2 or matrix.shape[1] != 8 or not np.isfinite(matrix).all():
                raise GfmTrainingError("Product pair features must be finite [E,8]")
            if not matrix.size:
                continue
            matrix[:, :3] = np.log1p(np.maximum(matrix[:, :3], 0.0))
            batch_count = matrix.shape[0]
            batch_mean = matrix.mean(axis=0)
            batch_square = np.sum((matrix - batch_mean) ** 2, axis=0)
            if count:
                delta = batch_mean - mean
                total = count + batch_count
                square += batch_square + delta * delta * count * batch_count / total
                mean += delta * batch_count / total
                count = total
            else:
                count = batch_count
                mean = batch_mean
                square = batch_square
        if count < 1:
            raise GfmTrainingError("Product feature transform requires training rows")
        scale = np.sqrt(square / count)
        scale[scale < 1e-8] = 1.0
        return cls(tuple(mean.tolist()), tuple(scale.tolist()))

    def apply(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64).copy()
        if matrix.ndim != 2 or matrix.shape[1] != 8:
            raise GfmTrainingError("Product feature transform received the wrong shape")
        matrix[:, :3] = np.log1p(np.maximum(matrix[:, :3], 0.0))
        transformed = (matrix - np.asarray(self.mean)) / np.asarray(self.scale)
        if not np.isfinite(transformed).all():
            raise GfmTrainingError("Product feature transformation produced non-finite values")
        return transformed.astype(np.float32)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "gfm.product-feature-transform/1.0",
            "featureNames": [
                "common_neighbors",
                "adamic_adar",
                "resource_allocation",
                "topic_similarity",
                "topic_complementarity",
                "institution_diversity",
                "recency",
                "common_neighbor_change",
            ],
            "log1pIndices": [0, 1, 2],
            "mean": list(self.mean),
            "scale": list(self.scale),
            "fitSplit": "train-only",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> _FeatureTransform:
        if value.get("fitSplit") != "train-only" or value.get("log1pIndices") != [0, 1, 2]:
            raise ContractViolation("Product feature transform provenance is invalid")
        return cls(
            tuple(float(item) for item in value["mean"]),
            tuple(float(item) for item in value["scale"]),
        )


@dataclass(frozen=True)
class _PreparedProductBatch:
    batch: Any
    raw_features: np.ndarray
    baseline_scores: np.ndarray
    participation_baseline: np.ndarray | None = None
    global_edges: np.ndarray | None = None
    institution_group: np.ndarray | None = None
    topic_group: np.ndarray | None = None
    collaboration_kind: Literal["first", "repeat"] | None = None

    def transformed(self, transform: _FeatureTransform) -> Any:
        import torch

        from ..gfm.product_training import ProductAdaptBatch

        return ProductAdaptBatch(
            core_batch=self.batch.core_batch,
            candidate_edge_index=self.batch.candidate_edge_index,
            pair_features=torch.from_numpy(transform.apply(self.raw_features)),
            pair_labels=self.batch.pair_labels,
            query_ids=self.batch.query_ids,
            provenance=self.batch.provenance,
            participation_node_index=self.batch.participation_node_index,
            participation_labels=self.batch.participation_labels,
        )

    def with_query_offset(self, offset: int) -> _PreparedProductBatch:
        from dataclasses import replace

        if not self.batch.query_ids.numel():
            return self
        return replace(  # type: ignore[type-var]
            self,
            batch=replace(self.batch, query_ids=self.batch.query_ids + int(offset)),
        )


def _openalex_array(
    arrays: Mapping[str, np.ndarray], name: str, *, shard: str | None = None
) -> np.ndarray:
    # ``load_domain`` exposes an unqualified key only when it is unique across
    # shards.  A caller that names a shard must always bind to that qualified
    # identity, even when the convenience alias also exists; treating both as
    # two candidates incorrectly made valid corpora look ambiguous.
    if shard is not None:
        qualified = f"{shard}.{name}"
        if qualified in arrays:
            return np.asarray(arrays[qualified])
    if name in arrays:
        return np.asarray(arrays[name])
    raise ContractViolation(f"OpenAlex array {name!r} is absent")


@dataclass
class _OpenAlexContext:
    neighbors: dict[int, set[int]]
    previous_neighbors: dict[int, set[int]]
    topics: dict[int, set[str]]
    topic_authors: dict[str, set[int]]
    institutions: dict[int, str]
    institution_sizes: dict[str, int]
    degree_buckets: dict[int, set[int]]
    inactive_days: Mapping[int, int]
    prior_work_count: dict[int, int]
    dominant_cluster: np.ndarray


class _InactiveDays(Mapping[int, int]):
    def __init__(self, latest: Mapping[int, int], cutoff: int) -> None:
        self._latest = latest
        self._cutoff = cutoff

    def __getitem__(self, key: int) -> int:
        timestamp = self._latest[key]
        return max(0, (self._cutoff - timestamp) // 86_400)

    def __iter__(self) -> Iterator[int]:
        return iter(self._latest)

    def __len__(self) -> int:
        return len(self._latest)


@dataclass
class _OpenAlexContextBuilder:
    """Chronologically advance OpenAlex context without rescanning old events."""

    stream: _DomainStream
    cursor: int = 0
    previous_cursor: int = 0
    cutoff: int = -1
    neighbors: defaultdict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    previous_neighbors: defaultdict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    work_authors: defaultdict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    work_topics: defaultdict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    topics: defaultdict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    topic_authors: defaultdict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    institutions: dict[int, str] = field(default_factory=dict)
    institution_sizes: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    latest: defaultdict[int, int] = field(default_factory=lambda: defaultdict(int))
    prior_work: defaultdict[int, int] = field(default_factory=lambda: defaultdict(int))
    cluster_counts: defaultdict[int, defaultdict[int, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    dominant: np.ndarray | None = None
    degree_buckets: defaultdict[int, set[int]] = field(default_factory=lambda: defaultdict(set))
    degree_bucket_by_author: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        author_count = int(self.stream.manifest["nodeCounts"]["author"])
        self.dominant = np.full(author_count, -1, dtype=np.int64)
        self.degree_buckets[0].update(range(author_count))

    def _update_degree_bucket(self, author: int) -> None:
        old = self.degree_bucket_by_author.get(author, 0)
        new = int(math.floor(math.log2(len(self.neighbors[author]) + 1)))
        if old != new:
            self.degree_buckets[old].discard(author)
            self.degree_buckets[new].add(author)
            self.degree_bucket_by_author[author] = new

    def _add_author_topic(self, author: int, topic: str) -> None:
        self.topics[author].add(topic)
        self.topic_authors[topic].add(author)

    def _advance_current(self, end: int) -> None:
        assert self.dominant is not None
        while self.cursor < end:
            index = self.cursor
            source, target = int(self.stream.src[index]), int(self.stream.dst[index])
            relation = int(self.stream.relation[index])
            timestamp = int(self.stream.timestamp[index])
            if relation == 0:
                self.work_authors[target].add(source)
                for topic in self.work_topics.get(target, set()):
                    self._add_author_topic(source, topic)
                self.prior_work[source] += 1
                self.latest[source] = max(self.latest[source], timestamp)
                if (
                    self.stream.text_node_offset is not None
                    and self.stream.work_cluster is not None
                    and self.stream.text_node_offset
                    <= target
                    < self.stream.text_node_offset + self.stream.work_cluster.shape[0]
                ):
                    cluster = int(self.stream.work_cluster[target - self.stream.text_node_offset])
                    self.cluster_counts[source][cluster] += 1
                    values = self.cluster_counts[source]
                    self.dominant[source] = min(values, key=lambda item: (-values[item], item))
            elif relation == 1:
                self.neighbors[source].add(target)
                self.neighbors[target].add(source)
                self._update_degree_bucket(source)
                self._update_degree_bucket(target)
                self.latest[source] = max(self.latest[source], timestamp)
                self.latest[target] = max(self.latest[target], timestamp)
            elif relation == 2:
                previous = self.institutions.get(source)
                current = str(target)
                if previous != current:
                    if previous is not None:
                        self.institution_sizes[previous] -= 1
                    self.institutions[source] = current
                    self.institution_sizes[current] += 1
            elif relation == 3:
                topic = str(target)
                self.work_topics[source].add(topic)
                for author in self.work_authors.get(source, set()):
                    self._add_author_topic(author, topic)
            self.cursor += 1

    def _advance_previous(self, end: int) -> None:
        while self.previous_cursor < end:
            index = self.previous_cursor
            if int(self.stream.relation[index]) == 1:
                source, target = int(self.stream.src[index]), int(self.stream.dst[index])
                self.previous_neighbors[source].add(target)
                self.previous_neighbors[target].add(source)
            self.previous_cursor += 1

    def advance(self, cutoff: int) -> _OpenAlexContext:
        if cutoff < self.cutoff:
            raise ContractViolation("OpenAlex context cutoffs must advance monotonically")
        end = int(np.searchsorted(self.stream.timestamp, cutoff, side="right"))
        previous_end = int(
            np.searchsorted(self.stream.timestamp, cutoff - 365 * 86_400, side="right")
        )
        self._advance_current(end)
        self._advance_previous(previous_end)
        self.cutoff = cutoff
        assert self.dominant is not None
        return _OpenAlexContext(
            neighbors=self.neighbors,
            previous_neighbors=self.previous_neighbors,
            topics=self.topics,
            topic_authors=self.topic_authors,
            institutions=self.institutions,
            institution_sizes=self.institution_sizes,
            degree_buckets=self.degree_buckets,
            inactive_days=_InactiveDays(self.latest, cutoff),
            prior_work_count=self.prior_work,
            dominant_cluster=self.dominant,
        )


def _openalex_context(stream: _DomainStream, cutoff: int) -> _OpenAlexContext:
    """Compatibility helper for one-off callers; hot paths use the builder."""

    return _OpenAlexContextBuilder(stream).advance(cutoff)


def _raw_pair_features(pairs: np.ndarray, *, context: _OpenAlexContext) -> np.ndarray:
    from ..gfm.product_features import collaboration_pair_features

    rows = []
    for source, target in pairs.T.tolist():
        previous_common = len(
            context.previous_neighbors.get(source, set()).intersection(
                context.previous_neighbors.get(target, set())
            )
        )
        rows.append(
            collaboration_pair_features(
                int(source),
                int(target),
                neighbors=context.neighbors,
                topics=context.topics,
                institutions=context.institutions,
                inactive_days=context.inactive_days,
                previous_common_neighbor_count=previous_common,
            )
        )
    return np.asarray(rows, dtype=np.float32).reshape(-1, 8)


def _product_core_batch(
    stream: _DomainStream,
    *,
    cutoff: int,
    candidate_edges: np.ndarray,
    horizon_days: float,
    task: ProductTask,
    extra_seed_nodes: Sequence[int] = (),
) -> tuple[Any, Any, np.ndarray]:
    import torch

    from ..gfm.types import CoreBatch, CoreSampleProvenance

    end = int(np.searchsorted(stream.timestamp, cutoff, side="right"))
    seeds = {int(value) for value in candidate_edges.reshape(-1).tolist()}
    seeds.update(int(value) for value in extra_seed_nodes)
    if not seeds:
        raise GfmTrainingError("Product graph batch requires at least one target node")
    message_indices = _recent_causal_edges(stream, end=end, seeds=seeds, fanout=(15, 10))
    if not message_indices.size:
        raise GfmTrainingError("Product target has no cutoff-visible graph history")
    original_src, original_dst = stream.src[message_indices], stream.dst[message_indices]
    message_src = np.concatenate((original_src, original_dst))
    message_dst = np.concatenate((original_dst, original_src))
    message_time = np.concatenate(
        (stream.timestamp[message_indices], stream.timestamp[message_indices])
    )
    message_relation = np.concatenate(
        (stream.relation[message_indices], stream.relation[message_indices])
    )
    local_nodes = np.asarray(
        sorted(set(message_src.tolist()) | set(message_dst.tolist()) | seeds),
        dtype=np.int64,
    )
    lookup = {int(value): index for index, value in enumerate(local_nodes)}
    remap = np.vectorize(lookup.__getitem__, otypes=[np.int64])
    local_src, local_dst = remap(message_src), remap(message_dst)
    local_candidates = (
        remap(candidate_edges) if candidate_edges.size else np.empty((2, 0), dtype=np.int64)
    )
    in_degree = np.bincount(local_dst, minlength=local_nodes.size)
    out_degree = np.bincount(local_src, minlength=local_nodes.size)
    numeric = np.stack(
        (np.log1p(in_degree), np.log1p(out_degree), np.log1p(in_degree + out_degree)),
        axis=1,
    ).astype(np.float32)
    node_types = stream.node_type[local_nodes]
    node_type = np.eye(4, dtype=np.float32)[node_types]
    modalities: dict[str, Any] = {
        "numeric": torch.from_numpy(numeric),
        "node_type": torch.from_numpy(node_type),
    }
    masks: dict[str, Any] = {
        "numeric": torch.ones(local_nodes.size, dtype=torch.bool),
        "node_type": torch.ones(local_nodes.size, dtype=torch.bool),
    }
    text = _local_text(
        stream,
        local_nodes=local_nodes,
        message_indices=message_indices,
        cutoff=cutoff,
    )
    if text is not None:
        values, visible = text
        modalities["text"] = torch.from_numpy(values)
        masks["text"] = torch.from_numpy(visible)
    time_scale = 86_400.0
    edge_time = (message_time / time_scale).astype(np.float32)
    cutoff_value = float(np.float32(cutoff / time_scale))
    core = CoreBatch(
        domain_id=stream.domain_id,
        modalities=modalities,
        modality_masks=masks,
        edge_index=torch.from_numpy(np.stack((local_src, local_dst))),
        edge_type=torch.from_numpy(message_relation.astype(np.int64) + stream.relation_offset),
        edge_time=torch.from_numpy(edge_time),
        cutoff_time=cutoff_value,
        provenance=CoreSampleProvenance(
            domain_id=stream.domain_id,
            graph_version=str(stream.manifest["logicalHash"]),
            cutoff=cutoff_value,
            horizon=float(horizon_days),
            task_id=task,
            source_corpus_hash=str(stream.manifest["logicalHash"]),
        ),
    )
    # Return the exact global->local inventory used by this CoreBatch.  Product
    # labels (especially newcomer participation) must never reconstruct a
    # second neighborhood and guess its local row ordering.
    return core, torch.from_numpy(local_candidates), local_nodes


def _stable_target_indices(
    src: np.ndarray,
    dst: np.ndarray,
    timestamp: np.ndarray,
    candidates: np.ndarray,
    *,
    limit: int,
    seed: int,
) -> np.ndarray:
    if candidates.size <= limit:
        return candidates
    # NumPy's pinned PCG64 stream gives deterministic O(limit) memory sampling
    # without materializing and SHA-sorting every candidate in a large cohort.
    rng = np.random.default_rng(seed)
    selected = rng.choice(candidates, size=limit, replace=False).astype(np.int64)
    return selected[np.lexsort((dst[selected], src[selected], timestamp[selected]))]


def _canonical_undirected(source: int, target: int) -> tuple[int, int]:
    return (source, target) if source < target else (target, source)


def _stable_pool(
    values: Sequence[int] | set[int], *, seed: int, source: int, component: str
) -> list[int]:
    return sorted(
        {int(value) for value in values},
        key=lambda target: canonical_sha256(
            {
                "seed": seed,
                "source": source,
                "target": target,
                "component": component,
            }
        ),
    )


def _product_negative_edges(
    positive: np.ndarray,
    *,
    context: _OpenAlexContext,
    author_count: int,
    forbidden_future: set[tuple[int, int]],
    negatives_per_positive: int,
    seed: int,
) -> np.ndarray:
    """Deterministic exact 50/25/25 target corruptions for product ranking.

    The future set is supplied only by the already declared supervision
    horizon.  It is never used to construct message features.  Every returned
    edge is absent both from the cutoff graph and from *all* positive outcomes
    in that query's horizon, preventing false-negative product labels.
    """

    if positive.shape[0] != 2 or positive.shape[1] < 1:
        raise GfmTrainingError("Product negative sampling requires [2,Q] positives")
    counts = {
        "hard": negatives_per_positive // 2,
        "degree": negatives_per_positive // 4,
    }
    counts["uniform"] = negatives_per_positive - sum(counts.values())
    selected_global: set[tuple[int, int]] = set()
    output: list[tuple[int, int]] = []

    def probe_order(*, source: int, query: int, component: str) -> Iterator[int]:
        digest = canonical_sha256(
            {"seed": seed, "source": source, "query": query, "component": component}
        )
        start = int(digest[:16], 16) % author_count
        stride = 1 + int(digest[16:32], 16) % max(1, author_count - 1)
        while math.gcd(stride, author_count) != 1:
            stride = 1 if stride + 1 >= author_count else stride + 1
        for offset in range(author_count):
            yield (start + offset * stride) % author_count

    for query, (raw_source, raw_target) in enumerate(positive.T.tolist()):
        source, target = int(raw_source), int(raw_target)
        visible = {
            _canonical_undirected(source, value)
            for value in context.neighbors.get(source, set())
            if value != source
        }
        forbidden = visible | forbidden_future | {_canonical_undirected(source, target)}
        direct = context.neighbors.get(source, set())
        two_hop = (
            set().union(*(context.neighbors.get(node, set()) for node in direct))
            if direct
            else set()
        )
        three_hop = (
            set().union(*(context.neighbors.get(node, set()) for node in two_hop))
            if two_hop
            else set()
        )
        topic = context.topics.get(source, set())
        topic_candidates = (
            set().union(*(context.topic_authors.get(value, set()) for value in topic))
            if topic
            else set()
        )
        hard = two_hop | three_hop | topic_candidates
        target_bucket = int(math.floor(math.log2(len(context.neighbors.get(target, set())) + 1)))
        chosen_for_query: set[int] = set()
        for component in ("hard", "degree", "uniform"):
            eligible = []
            candidates: Sequence[int] | Iterator[int]
            if component == "hard":
                candidates = (
                    _stable_pool(
                        hard,
                        seed=seed + query,
                        source=source,
                        component=component,
                    )
                    if len(hard) <= 4096
                    else (
                        value
                        for value in probe_order(source=source, query=query, component=component)
                        if value in hard
                    )
                )
            elif component == "degree":
                degree_pool = context.degree_buckets.get(target_bucket, set())
                candidates = (
                    value
                    for value in probe_order(source=source, query=query, component=component)
                    if value in degree_pool
                )
            else:
                candidates = probe_order(source=source, query=query, component=component)
            for candidate in candidates:
                pair = _canonical_undirected(source, candidate)
                if (
                    candidate == source
                    or candidate in chosen_for_query
                    or pair in forbidden
                    or pair in selected_global
                ):
                    continue
                eligible.append(candidate)
                if len(eligible) == counts[component]:
                    break
            if len(eligible) != counts[component]:
                raise GfmTrainingError(
                    f"Product {component} pool cannot supply the fixed exact ratio"
                )
            for candidate in eligible:
                chosen_for_query.add(candidate)
                selected_global.add(_canonical_undirected(source, candidate))
                output.append((source, candidate))
    result = np.asarray(output, dtype=np.int64).T
    expected = positive.shape[1] * negatives_per_positive
    if result.shape != (2, expected):
        raise GfmTrainingError("Product negative sample count is inconsistent")
    return result


def _collaboration_batches(
    stream: _DomainStream,
    arrays: Mapping[str, np.ndarray],
    *,
    cutoff_years: Sequence[int],
    seed: int,
    transform: _FeatureTransform | None,
    query_limit_per_year: int = 512,
    target_kind: Literal["first", "repeat"] = "first",
) -> Iterator[_PreparedProductBatch]:
    import torch

    from ..gfm.product_training import ProductAdaptBatch, SampleProvenance

    target_src = _openalex_array(arrays, "src", shard="targets-00000").astype(np.int64)
    target_dst = _openalex_array(arrays, "dst", shard="targets-00000").astype(np.int64)
    target_time = _openalex_array(arrays, "timestamp", shard="targets-00000").astype(np.int64)
    first = _openalex_array(arrays, "first_collaboration", shard="targets-00000").astype(np.bool_)
    produced = False
    query_base = 0
    author_count = int(stream.manifest["nodeCounts"]["author"])
    context_builder = _OpenAlexContextBuilder(stream)
    for year in cutoff_years:
        cutoff = int(datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp())
        horizon = int(datetime(year + 1, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp())
        kind_mask = first if target_kind == "first" else ~first
        selected = np.flatnonzero(kind_mask & (target_time > cutoff) & (target_time <= horizon))
        selected = _stable_target_indices(
            target_src,
            target_dst,
            target_time,
            selected,
            limit=query_limit_per_year,
            seed=seed + year,
        )
        context = context_builder.advance(cutoff)
        horizon_mask = (target_time > cutoff) & (target_time <= horizon)
        forbidden_future = {
            _canonical_undirected(int(source), int(target))
            for source, target in zip(
                target_src[horizon_mask], target_dst[horizon_mask], strict=True
            )
        }
        for offset in range(0, selected.size, 8):
            indices = selected[offset : offset + 8]
            if not indices.size:
                continue
            positive = np.stack((target_src[indices], target_dst[indices]))
            negatives = _product_negative_edges(
                positive,
                context=context,
                author_count=author_count,
                forbidden_future=forbidden_future,
                negatives_per_positive=99,
                seed=seed + year * 10_000 + offset,
            )
            candidate_global = np.concatenate((positive, negatives), axis=1)
            labels = np.concatenate((np.ones(indices.size), np.zeros(negatives.shape[1]))).astype(
                np.float32
            )
            queries = np.concatenate(
                (
                    np.arange(indices.size) + query_base,
                    np.repeat(np.arange(indices.size) + query_base, 99),
                )
            ).astype(np.int64)
            core, local_candidates, _ = _product_core_batch(
                stream,
                cutoff=cutoff,
                candidate_edges=candidate_global,
                horizon_days=float((horizon - cutoff) / 86_400.0),
                task="collaboration",
            )
            raw = _raw_pair_features(candidate_global, context=context)
            source_institution_sizes = np.asarray(
                [
                    context.institution_sizes.get(context.institutions.get(int(source), ""), 0)
                    for source in candidate_global[0]
                ],
                dtype=np.int64,
            )
            institution_group = np.select(
                (source_institution_sizes < 10, source_institution_sizes < 100),
                (0, 1),
                default=2,
            ).astype(np.int8)
            topic_group = context.dominant_cluster[candidate_global[0]].astype(np.int8, copy=True)
            pair_features = raw if transform is None else transform.apply(raw)
            batch = ProductAdaptBatch(
                core_batch=core,
                candidate_edge_index=local_candidates,
                pair_features=torch.from_numpy(pair_features.astype(np.float32)),
                pair_labels=torch.from_numpy(labels),
                query_ids=torch.from_numpy(queries),
                provenance=SampleProvenance(
                    domain_id=stream.domain_id,
                    graph_version=str(stream.manifest["logicalHash"]),
                    cutoff=float(core.cutoff_time),
                    horizon=float((horizon - cutoff) / 86_400.0),
                    task_id="collaboration",
                    source_corpus_hash=str(stream.manifest["logicalHash"]),
                ),
            )
            query_base += indices.size
            produced = True
            yield _PreparedProductBatch(
                batch=batch,
                raw_features=raw,
                baseline_scores=raw[:, 2].copy(),
                global_edges=candidate_global.copy(),
                institution_group=institution_group,
                topic_group=topic_group,
                collaboration_kind=target_kind,
            )
    if not produced:
        raise GfmTrainingError(
            f"OpenAlex {target_kind}-collaboration split produced no product batches"
        )


def _newcomer_batches(
    stream: _DomainStream,
    arrays: Mapping[str, np.ndarray],
    newcomer_overlay: Mapping[str, np.ndarray],
    *,
    cohort_years: Sequence[int],
    seed: int,
    transform: _FeatureTransform | None,
    cohort_limit: int = 2_000,
    negative_candidates_per_query: int = 99,
) -> Iterator[_PreparedProductBatch]:
    import torch

    from ..gfm.product_features import SupporterCandidate, eligible_supporter
    from ..gfm.product_training import ProductAdaptBatch, SampleProvenance

    authors = np.asarray(newcomer_overlay["author"], dtype=np.int64)
    t0 = np.asarray(newcomer_overlay["t0"], dtype=np.int64)
    true_newcomer = np.asarray(newcomer_overlay["true_newcomer"], dtype=np.bool_)
    target_src = _openalex_array(arrays, "src", shard="targets-00000").astype(np.int64)
    target_dst = _openalex_array(arrays, "dst", shard="targets-00000").astype(np.int64)
    target_time = _openalex_array(arrays, "timestamp", shard="targets-00000").astype(np.int64)
    produced = False
    participation_outcomes: set[float] = set()
    query_base = 0
    context_builder = _OpenAlexContextBuilder(stream)
    for cohort_year in cohort_years:
        year_start = int(datetime(cohort_year, 1, 1, tzinfo=UTC).timestamp())
        year_end = int(datetime(cohort_year, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp())
        cohort = np.flatnonzero(true_newcomer & (t0 >= year_start) & (t0 <= year_end))
        cohort = _stable_target_indices(
            authors,
            authors,
            t0,
            cohort,
            limit=cohort_limit,
            seed=seed + cohort_year,
        )
        cohort = cohort[np.lexsort((authors[cohort], t0[cohort]))]
        for author_row in cohort.tolist():
            newcomer = int(authors[author_row])
            cutoff = int(t0[author_row] + 90 * 86_400)
            horizon = int(t0[author_row] + 365 * 86_400)
            context = context_builder.advance(cutoff)
            future_pair_mask = (
                (target_time > cutoff)
                & (target_time <= horizon)
                & ((target_src == newcomer) | (target_dst == newcomer))
            )
            future_partners = []
            for index in np.flatnonzero(future_pair_mask).tolist():
                partner = (
                    int(target_dst[index])
                    if int(target_src[index]) == newcomer
                    else int(target_src[index])
                )
                future_partners.append(partner)
            future_authorship = (
                (stream.relation == 0)
                & (stream.src == newcomer)
                & (stream.timestamp > cutoff)
                & (stream.timestamp <= horizon)
            )
            participation = bool(future_partners) or bool(future_authorship.any())
            direct = context.neighbors.get(newcomer, set())
            two_hop = (
                set().union(*(context.neighbors.get(node, set()) for node in direct))
                if direct
                else set()
            )
            three_hop = (
                set().union(*(context.neighbors.get(node, set()) for node in two_hop))
                if two_hop
                else set()
            )
            newcomer_topics = context.topics.get(newcomer, set())

            def supporter(
                value: int,
                *,
                fixed_context: _OpenAlexContext = context,
                fixed_direct: set[int] = direct,
                fixed_two_hop: set[int] = two_hop,
                fixed_three_hop: set[int] = three_hop,
                fixed_topics: set[str] = newcomer_topics,
            ) -> SupporterCandidate:
                return SupporterCandidate(
                    candidate_id=value,
                    prior_work_count=fixed_context.prior_work_count.get(value, 0),
                    inactive_days=fixed_context.inactive_days.get(value, 10_000),
                    previously_collaborated=value in fixed_direct,
                    graph_distance=(
                        2 if value in fixed_two_hop else 3 if value in fixed_three_hop else None
                    ),
                    adjacent_topic=bool(
                        fixed_topics.intersection(fixed_context.topics.get(value, set()))
                    ),
                    adjacent_community=False,
                )

            # Every eligible future supporter is a positive for this query.
            # Selecting a single one would silently turn the remaining real
            # supporters into false negatives.
            positive_supporters = sorted(
                {
                    partner
                    for partner in future_partners
                    if partner != newcomer and eligible_supporter(supporter(partner))
                }
            )
            candidate_global = np.empty((2, 0), dtype=np.int64)
            labels = np.empty(0, dtype=np.float32)
            queries = np.empty(0, dtype=np.int64)
            raw = np.empty((0, 8), dtype=np.float32)
            baseline = np.empty(0, dtype=np.float32)
            if positive_supporters:
                future_forbidden = {
                    _canonical_undirected(newcomer, partner)
                    for partner in future_partners
                    if partner != newcomer
                }
                probe_positive = np.asarray([[newcomer], [positive_supporters[0]]], dtype=np.int64)
                negative_edges = _product_negative_edges(
                    probe_positive,
                    context=context,
                    author_count=int(stream.manifest["nodeCounts"]["author"]),
                    forbidden_future=future_forbidden,
                    negatives_per_positive=negative_candidates_per_query,
                    seed=seed + cohort_year * 10_000 + author_row,
                )
                positive_edges = np.asarray(
                    [
                        [newcomer] * len(positive_supporters),
                        positive_supporters,
                    ],
                    dtype=np.int64,
                )
                candidate_global = np.concatenate((positive_edges, negative_edges), axis=1)
                labels = np.concatenate(
                    (
                        np.ones(len(positive_supporters), dtype=np.float32),
                        np.zeros(negative_edges.shape[1], dtype=np.float32),
                    )
                )
                queries = np.full(labels.shape[0], query_base, dtype=np.int64)
                raw = _raw_pair_features(candidate_global, context=context)
                # Fixed transparent newcomer baseline: topic similarity plus
                # cutoff-visible activity.  No future statistic enters it.
                baseline = raw[:, 3] + raw[:, 6]
            core, local_candidates, local_nodes = _product_core_batch(
                stream,
                cutoff=cutoff,
                candidate_edges=candidate_global,
                horizon_days=float((horizon - cutoff) / 86_400.0),
                task="newcomer",
                extra_seed_nodes=(newcomer,),
            )
            position = np.flatnonzero(local_nodes == newcomer)
            if position.shape != (1,):
                raise GfmTrainingError("Newcomer is absent from its combined causal batch")
            local_author = int(position[0])
            pair_features = raw if transform is None else transform.apply(raw)
            batch = ProductAdaptBatch(
                core_batch=core,
                candidate_edge_index=local_candidates,
                pair_features=torch.from_numpy(pair_features.astype(np.float32)),
                pair_labels=torch.from_numpy(labels),
                query_ids=torch.from_numpy(queries),
                provenance=SampleProvenance(
                    domain_id=stream.domain_id,
                    graph_version=str(stream.manifest["logicalHash"]),
                    cutoff=float(core.cutoff_time),
                    horizon=float((horizon - cutoff) / 86_400.0),
                    task_id="newcomer",
                    source_corpus_hash=str(stream.manifest["logicalHash"]),
                ),
                participation_node_index=torch.tensor([local_author], dtype=torch.long),
                participation_labels=torch.tensor([float(participation)], dtype=torch.float32),
            )
            produced = True
            participation_outcomes.add(float(participation))
            yield _PreparedProductBatch(
                batch=batch,
                raw_features=raw,
                baseline_scores=baseline,
                participation_baseline=np.asarray(
                    [
                        min(1.0, context.prior_work_count.get(newcomer, 0) / 3.0)
                        * math.exp(-context.inactive_days.get(newcomer, 10_000) / 365.0)
                    ],
                    dtype=np.float32,
                ),
                global_edges=candidate_global.copy(),
                institution_group=np.asarray(
                    [
                        0
                        if context.institution_sizes.get(context.institutions.get(newcomer, ""), 0)
                        < 10
                        else 1
                        if context.institution_sizes.get(context.institutions.get(newcomer, ""), 0)
                        < 100
                        else 2
                    ],
                    dtype=np.int8,
                ),
                topic_group=np.asarray([context.dominant_cluster[newcomer]], dtype=np.int8),
            )
            if labels.size:
                query_base += 1
    if not produced:
        raise GfmTrainingError("OpenAlex newcomer cohort produced no product batches")
    if len(set(participation_outcomes)) < 2:
        raise GfmTrainingError("Newcomer cohort lacks both participation outcomes")


def _probe_batch_loaders(
    streams: Mapping[str, _DomainStream],
    *,
    batch_size: int,
    fanout: tuple[int, int],
    seed: int,
) -> dict[str, Iterable[Any]]:
    """Create one lazy probe batch per domain without retaining sibling batches.

    A 2,048-target text-domain batch can own several 1024-wide CPU tensors.
    Materialising all domain batches before the round-robin trainer starts is
    not representative of the real generator-backed training path and can
    exhaust host memory before CUDA is probed.  Each iterator below therefore
    constructs exactly one otherwise-identical batch only when its turn is
    requested by ``CoreTrainer``.
    """

    def one(stream: _DomainStream) -> Iterator[Any]:
        yield _core_batch(
            stream,
            batch_size=batch_size,
            fanout=fanout,
            seed=seed,
        )

    return {domain: one(stream) for domain, stream in streams.items()}


def _close_probe_loaders(loaders: Mapping[str, Iterable[Any]]) -> None:
    """Close suspended probe generators in a scope that cannot retain them."""

    for loader in loaders.values():
        close = getattr(loader, "close", None)
        if callable(close):
            close()


def _probe_batch_size(
    *,
    config: GfmPretrainConfig,
    variant: str,
    streams: Mapping[str, _DomainStream],
    device: str,
    seed: int,
) -> tuple[int, float]:
    import torch

    from ..gfm.model import SocialGraphFMCore
    from ..gfm.trainer import CoreTrainer, CoreTrainerConfig

    candidates = tuple(config.optimization.candidate_batch_sizes)
    if device == "cpu":
        return min(candidates[-1], 64), 0.0
    fanout = (
        int(config.architecture.neighbor_fanout[0]),
        int(config.architecture.neighbor_fanout[1]),
    )
    for candidate in candidates:
        # Candidate attempts must not inherit strong references to a failed
        # larger probe.  In particular, an optimizer owns its model parameters
        # and a suspended generator may own its last yielded CPU batch.
        model: Any | None = None
        optimizer: Any | None = None
        trainer: Any | None = None
        loaders: dict[str, Iterable[Any]] | None = None
        snapshots = {key: value.state_dict() for key, value in streams.items()}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = SocialGraphFMCore(_model_config(config, variant))
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.optimization.learning_rate,
                weight_decay=config.optimization.weight_decay,
            )
            accumulation = math.ceil(config.optimization.effective_batch_size / candidate)
            trainer = CoreTrainer(
                model,
                optimizer,
                CoreTrainerConfig(
                    gradient_accumulation_steps=accumulation,
                    gradient_clip=config.optimization.gradient_clip,
                    amp=True,
                ),
                device,
            )
            loaders = _probe_batch_loaders(
                streams,
                batch_size=candidate,
                fanout=fanout,
                seed=seed,
            )
            trainer.train_epoch(loaders)
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            if peak < config.optimization.cuda_memory_limit_mib:
                return candidate, float(peak)
        except (torch.OutOfMemoryError, MemoryError):
            # NumPy/Python allocations for a fixed candidate can fail before
            # its tensors reach CUDA.  That is still a candidate-level memory
            # failure; contract/value errors remain intentionally uncaught.
            pass
        finally:
            for key, value in snapshots.items():
                streams[key].load_state_dict(value)
            if loaders is not None:
                _close_probe_loaders(loaders)
            loaders = None
            trainer = None
            optimizer = None
            model = None
            torch.cuda.empty_cache()
    raise GfmTrainingError("CUDA batch probes 2048/1024/512 all failed or exceeded 7168 MiB")


def _warmup_cosine(step: int, *, maximum: int, warmup_ratio: float) -> float:
    warmup = max(1, int(maximum * warmup_ratio))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, maximum - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _save_run_state(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(payload))


__all__ = []
