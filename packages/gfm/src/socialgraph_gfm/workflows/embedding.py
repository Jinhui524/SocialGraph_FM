"""Embedding creation, bounded loading, and pretraining evidence.

The implementation is installed into the shared compatibility namespace by
:mod:`socialgraph_gfm.workflows` after all workflow stages are imported.
"""

# ruff: noqa: F403, F405
# mypy: disable-error-code=name-defined
from __future__ import annotations

from ._shared import *


def _find_bge_snapshot(layout: RuntimeLayout) -> Path:
    candidates = (
        layout.cache_hf
        / "hub"
        / "models--BAAI--bge-m3"
        / "snapshots"
        / "5617a9f61b028005a4858fdac845db406aefb181",
        layout.cache_hf
        / "models--BAAI--bge-m3"
        / "snapshots"
        / "5617a9f61b028005a4858fdac845db406aefb181",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise ContractViolation(
        "The pinned BAAI/bge-m3 revision is not present below the runtime HF cache; "
        "download it explicitly before offline embedding"
    )


def embed_gfm_text(
    *,
    root: str | Path | None,
    encoder: str,
    domain: Literal["all", "openalex", "wikimedia-talk"] = "all",
) -> dict[str, Any]:
    if encoder != "BAAI/bge-m3":
        raise ContractViolation("SocialGraph-FM Core pins --encoder BAAI/bge-m3")
    require_gfm_optional_runtime(text=True)
    layout = prepare_runtime_layout(root, operation="run")
    selected_domains = (
        TEXT_DOMAINS
        if domain == "all"
        else (DOMAIN_IDS["openalex"],)
        if domain == "openalex"
        else (DOMAIN_IDS["wikimedia-talk"],)
    )
    for corpus_id in selected_domains:
        loaded = load_domain(layout.root, corpus_id)
        contract_path = layout.manifests_gfm / f"{corpus_id}.json"
        if not contract_path.is_file():
            raise ContractViolation(
                f"Formal domain contract is absent for {corpus_id}; run gfm-corpus-prepare"
            )
        contract = GfmDomainCorpusManifest.model_validate_json(contract_path.read_text("utf-8"))
        if contract.content_hash != loaded["manifest"].get("logicalHash"):
            raise ContractViolation(f"Domain contract is stale for {corpus_id}")
    model_dir = _find_bge_snapshot(layout)
    artifacts = []
    for corpus_id in selected_domains:
        text_path = layout.processed_gfm / corpus_id / "text.jsonl"
        role_resolver: Callable[[str, int], Literal["train", "validation", "test", "shadow"]]
        if corpus_id == DOMAIN_IDS["openalex"]:
            train_end = int(datetime(2021, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp())
            validation_end = int(datetime(2022, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp())
            test_end = int(datetime(2023, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp())

            def openalex_role_resolver(
                identifier: str, timestamp: int
            ) -> Literal["train", "validation", "test", "shadow"]:
                del identifier
                return (
                    "train"
                    if timestamp <= train_end
                    else "validation"
                    if timestamp <= validation_end
                    else "test"
                    if timestamp <= test_end
                    else "shadow"
                )

            role_resolver = openalex_role_resolver

        else:
            domain_view = load_domain_view(
                layout.root,
                corpus_id,
                maximum_role="shadow",
                families=("events",),
            )["arrays"]
            revision = np.asarray(domain_view["revision_pseudonym"], dtype=np.uint64)
            event_role = np.asarray(domain_view["split"])
            if (
                event_role.dtype != np.dtype(np.int8)
                or revision.shape != event_role.shape
                or revision.size == 0
                or bool(np.any((event_role < 0) | (event_role > 2)))
            ):
                raise ContractViolation(
                    "Wikimedia embedding roles are not aligned to event pseudonyms"
                )
            role_names: tuple[Literal["train", "validation", "test"], ...] = (
                "train",
                "validation",
                "test",
            )
            order = np.argsort(revision, kind="stable")
            sorted_revision = np.ascontiguousarray(revision[order])
            sorted_event_role = np.ascontiguousarray(event_role[order])
            if bool(np.any(sorted_revision[1:] == sorted_revision[:-1])):
                raise ContractViolation("Wikimedia revision pseudonyms are not unique")

            def wikimedia_role_resolver(
                identifier: str, timestamp: int
            ) -> Literal["train", "validation", "test", "shadow"]:
                del timestamp
                if not identifier.isdigit():
                    raise ContractViolation("Wikimedia text identifier has no physical split role")
                numeric = int(identifier)
                if not 0 <= numeric <= np.iinfo(np.uint64).max:
                    raise ContractViolation("Wikimedia text identifier has no physical split role")
                position = int(np.searchsorted(sorted_revision, np.uint64(numeric), side="left"))
                if position >= sorted_revision.size or int(sorted_revision[position]) != numeric:
                    raise ContractViolation("Wikimedia text identifier has no physical split role")
                return role_names[int(sorted_event_role[position])]

            role_resolver = wikimedia_role_resolver

        source_manifest = read_json_object(text_path.parent / "manifest.json")
        source_manifest_hash = source_manifest.get("logicalHash")
        if not isinstance(source_manifest_hash, str):
            raise ContractViolation("Prepared text source manifest hash is absent")
        manifest = build_bge_m3_embeddings(
            text_path,
            layout.root,
            config=EmbeddingConfig(corpus_id=corpus_id),
            model_dir=model_dir,
            offline=True,
            role_resolver=role_resolver,
            source_manifest_hash=source_manifest_hash,
        )
        artifacts.append(
            {
                "corpusId": corpus_id,
                "logicalHash": manifest["logicalHash"],
                "rows": manifest["rows"],
                "dimension": manifest["dimension"],
            }
        )
    return {
        "schemaVersion": "gfm.workflow-text-embed/1.0",
        "ok": True,
        "encoder": encoder,
        "selectedDomains": list(selected_domains),
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "artifacts": artifacts,
    }


def _load_corpus_contracts(
    layout: RuntimeLayout,
    *,
    verification_domain_ids: Sequence[str] | None = None,
    physical_boundary: bool = False,
) -> tuple[GfmDomainCorpusManifest, ...]:
    selected = (
        set(DOMAIN_IDS.values())
        if verification_domain_ids is None
        else set(verification_domain_ids)
    )
    if not selected.issubset(DOMAIN_IDS.values()):
        raise ContractViolation("Corpus verification requested an unknown domain")
    if verification_domain_ids is None and not physical_boundary:
        check_all_gfm_corpora(layout.root)
    result: list[GfmDomainCorpusManifest] = []
    for domain_id in DOMAIN_IDS.values():
        path = layout.manifests_gfm / f"{domain_id}.json"
        if not path.is_file():
            raise ContractViolation(
                f"Formal domain contract is absent for {domain_id}; run gfm-corpus-prepare"
            )
        contract = GfmDomainCorpusManifest.model_validate_json(path.read_text("utf-8"))
        if domain_id in selected and not physical_boundary:
            loaded = load_domain(layout.root, domain_id)["manifest"]
            if contract.content_hash != loaded.get("logicalHash"):
                raise ContractViolation(f"Domain contract is stale for {domain_id}")
        result.append(contract)
    return tuple(result)


@dataclass
class _BoundedEmbeddingStore:
    """Random access over verified NPZ shards with a strict two-shard LRU."""

    manifest: dict[str, Any]
    prepared_manifest: dict[str, Any]
    handle: Any
    shard_starts: np.ndarray
    maximum_role: Literal["train", "validation", "test", "shadow"] | None = None
    _cache: OrderedDict[int, Any] = field(default_factory=OrderedDict)
    _sorted_id_hash: np.ndarray | None = None
    _sorted_global_row: np.ndarray | None = None

    def _shard(self, index: int) -> Any:
        cached = self._cache.pop(index, None)
        if cached is None:
            cached = self.handle.load_shard(index)
        self._cache[index] = cached
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return cached

    def build_hash_index(self) -> None:
        if self._sorted_id_hash is not None:
            return
        hashes: list[np.ndarray] = []
        rows: list[np.ndarray] = []
        for shard in self.handle.iter_shards():
            hashes.append(np.array(shard.id_hash, copy=True))
            start = int(self.shard_starts[shard.index])
            rows.append(np.arange(start, start + shard.id_hash.shape[0], dtype=np.int64))
        all_hashes = np.concatenate(hashes)
        all_rows = np.concatenate(rows)
        if all_hashes.shape != (self.handle.rows,):
            raise ContractViolation("Embedding hash index row count differs from its handle")
        order = np.argsort(all_hashes, kind="stable")
        sorted_hashes = np.ascontiguousarray(all_hashes[order])
        sorted_rows = np.ascontiguousarray(all_rows[order])
        if sorted_hashes.size > 1 and bool(np.any(sorted_hashes[1:] == sorted_hashes[:-1])):
            raise ContractViolation("Embedding hash index is not globally unique")
        sorted_hashes.setflags(write=False)
        sorted_rows.setflags(write=False)
        self._sorted_id_hash = sorted_hashes
        self._sorted_global_row = sorted_rows
        self._cache.clear()

    def lookup_hashes(self, requested: np.ndarray) -> dict[int, tuple[np.ndarray, int]]:
        self.build_hash_index()
        assert self._sorted_id_hash is not None
        assert self._sorted_global_row is not None
        values = np.asarray(requested, dtype=np.uint64).reshape(-1)
        if not values.size:
            return {}
        unique = np.unique(values)
        positions = np.searchsorted(self._sorted_id_hash, unique, side="left")
        matched = positions < self._sorted_id_hash.size
        safe_positions = np.minimum(positions, self._sorted_id_hash.size - 1)
        matched &= self._sorted_id_hash[safe_positions] == unique
        global_rows = self._sorted_global_row[safe_positions[matched]]
        matched_hashes = unique[matched]
        result: dict[int, tuple[np.ndarray, int]] = {}
        for shard_index in np.unique(
            np.searchsorted(self.shard_starts[1:], global_rows, side="right")
        ):
            selection = np.flatnonzero(
                np.searchsorted(self.shard_starts[1:], global_rows, side="right") == shard_index
            )
            shard = self._shard(int(shard_index))
            local_rows = global_rows[selection] - self.shard_starts[shard_index]
            for hash_value, local_row in zip(matched_hashes[selection], local_rows, strict=True):
                row = int(local_row)
                if int(shard.id_hash[row]) != int(hash_value):
                    raise ContractViolation("Embedding hash index changed after verification")
                result[int(hash_value)] = (
                    np.array(shard.embedding[row], copy=True),
                    int(shard.timestamp[row]),
                )
        return result

    def lookup_rows(self, requested: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.asarray(requested, dtype=np.int64).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= self.handle.rows):
            raise ContractViolation("Embedding row lookup is outside the artifact")
        embeddings = np.empty((rows.size, 1024), dtype=np.float32)
        hashes = np.empty(rows.size, dtype=np.uint64)
        timestamps = np.empty(rows.size, dtype=np.int64)
        if not rows.size:
            return embeddings, hashes, timestamps
        shard_indices = np.searchsorted(self.shard_starts[1:], rows, side="right")
        for shard_index in np.unique(shard_indices):
            selection = np.flatnonzero(shard_indices == shard_index)
            local_rows = rows[selection] - self.shard_starts[shard_index]
            shard = self._shard(int(shard_index))
            embeddings[selection] = shard.embedding[local_rows]
            hashes[selection] = shard.id_hash[local_rows]
            timestamps[selection] = shard.timestamp[local_rows]
        return embeddings, hashes, timestamps


def _open_embedding_store(
    layout: RuntimeLayout,
    corpus_id: str,
    *,
    maximum_role: Literal["train", "validation", "test", "shadow"] | None = None,
) -> _BoundedEmbeddingStore:
    directory = layout.embeddings / f"{corpus_id}-bge-m3-v1"
    handle = (
        open_embedding_artifact(directory)
        if maximum_role is None
        else open_embedding_artifact_view(directory, maximum_role=maximum_role)
    )
    manifest = read_json_object(directory / "manifest.json")
    prepared_manifest = read_json_object(layout.processed_gfm / corpus_id / "manifest.json")
    shard_starts = np.empty(handle.shard_count + 1, dtype=np.int64)
    shard_starts[0] = 0
    np.cumsum(np.asarray(handle.shard_rows, dtype=np.int64), out=shard_starts[1:])
    if int(shard_starts[-1]) != handle.rows:
        raise ContractViolation("Embedding shard offsets differ from verified row count")
    shard_starts.setflags(write=False)
    return _BoundedEmbeddingStore(
        manifest=manifest,
        prepared_manifest=prepared_manifest,
        handle=handle,
        shard_starts=shard_starts,
        maximum_role=maximum_role,
    )


def _load_pretrain_config(
    config: str | Path | None,
    overrides: Mapping[str, Any] | None,
) -> GfmPretrainConfig:
    path: str | Path | None = config
    if config == "socialgraph-core.json":
        path = None
    payload = apply_exploratory_overrides(load_core_config(path), overrides)
    return GfmPretrainConfig.model_validate(payload)


def _task_protocols() -> tuple[GfmTaskProtocolManifest, ...]:
    from ..gfm.task_acceptance import collaboration_protocol

    return (
        collaboration_protocol(),
        GfmTaskProtocolManifest.create(
            protocolId="socialgraph-fm-newcomer-support",
            taskId=NEWCOMER_TASK,
            taskFamily="newcomer_support",
            domainIds=(DOMAIN_IDS["openalex"], DOMAIN_IDS["wikimedia-talk"]),
            splitStrategy="temporal",
            objectives=(
                "cohorts=train-2017,2018,2019,2020;validation=2021;test=2022;later-cohorts=shadow-only",
                "newcomer-t0=globally-verified-first-work;observation-window=90-days;prediction-horizon=365-days",
                "supporter-constraints=cutoff-works>=3,active-within-24-months,no-prior-collaboration,and-2-or-3-hop-or-adjacent-topic-or-community",
                "objectives=pairwise-supporter-or-community-ranking-plus-0.5-sustained-participation-bce;exclude-all-future-supporters-from-negatives",
                "baseline=cutoff-topic-similarity-plus-cutoff-activity",
            ),
            primaryMetrics=("ndcg@20", "auprc"),
        ),
    )


def _register_prerequisites(
    layout: RuntimeLayout,
    corpora: Sequence[GfmDomainCorpusManifest],
) -> tuple[GfmTaskProtocolManifest, ...]:
    registry = _registry(layout)
    for corpus in corpora:
        registry.record_corpus(corpus)
    protocols = _task_protocols()
    for protocol in protocols:
        registry.record_protocol(protocol)
    return protocols


def _experiment_id(
    *, phase: TrainingPhase, config: GfmPretrainConfig, corpora: Sequence[GfmDomainCorpusManifest]
) -> str:
    digest = canonical_sha256(
        {
            "phase": phase,
            "configHash": config.config_hash,
            "runKind": config.run_kind,
            "corpora": sorted(corpus.logical_hash for corpus in corpora),
        }
    )
    return f"socialgraph-core-{phase}-{digest[:16]}"


def _environment_hash(device: str) -> str:
    report = runtime_report(device)
    if not report["runtimeReady"]:
        require_ml_runtime(device)
    return str(report["environmentHash"])


def _ensure_pretrain_evidence(
    layout: RuntimeLayout,
    *,
    formal_required: bool = True,
    embedding_domain_ids: Sequence[str] | None = None,
    corpus_verification_domain_ids: Sequence[str] | None = None,
    maximum_role: Literal["train", "validation", "test", "shadow"] | None = None,
    physical_boundary: bool = False,
) -> tuple[
    tuple[GfmDomainCorpusManifest, ...],
    dict[str, _BoundedEmbeddingStore],
]:
    corpora = _load_corpus_contracts(
        layout,
        verification_domain_ids=corpus_verification_domain_ids,
        physical_boundary=physical_boundary,
    )
    if formal_required and not all(corpus.public_checkpoint_eligible for corpus in corpora):
        ineligible = sorted(
            corpus.domain_id for corpus in corpora if not corpus.public_checkpoint_eligible
        )
        raise ContractViolation(
            "Formal pretraining rejects non-promotable corpus sources: " + ", ".join(ineligible)
        )
    selected_text_domains = (
        set(TEXT_DOMAINS)
        if embedding_domain_ids is None
        else set(embedding_domain_ids).intersection(TEXT_DOMAINS)
    )
    unknown = set(embedding_domain_ids or ()).difference(DOMAIN_IDS.values())
    if unknown:
        raise ContractViolation("Embedding evidence requested an unknown GFM domain")
    embeddings = {
        domain_id: _open_embedding_store(layout, domain_id, maximum_role=maximum_role)
        for domain_id in TEXT_DOMAINS
        if domain_id in selected_text_domains
    }
    for domain_id, store in embeddings.items():
        producer = store.manifest.get("producer")
        if not isinstance(producer, dict):
            raise ContractViolation(f"{domain_id} embedding producer identity is absent")
        if formal_required and producer != {
            "implementation": "FlagEmbedding.BGEM3FlagModel",
            "distribution": "FlagEmbedding",
            "version": "1.4.0",
            "formalEligible": True,
        }:
            raise ContractViolation(
                f"{domain_id} formal embedding artifact was not produced by the pinned "
                "FlagEmbedding 1.4.0 BGEM3 wrapper"
            )
    return corpora, embeddings


def _embedding_artifact_evidence(
    embeddings: Mapping[str, _BoundedEmbeddingStore],
) -> dict[str, dict[str, Any]]:
    """Return the immutable text-feature identities bound into checkpoints."""

    evidence: dict[str, dict[str, Any]] = {}
    for domain_id, store in sorted(embeddings.items()):
        producer = store.manifest.get("producer")
        source_evidence = store.manifest.get("source")
        if not isinstance(producer, dict):
            raise ContractViolation("Embedding producer identity is absent")
        if not isinstance(source_evidence, dict):
            raise ContractViolation("Embedding model source evidence is absent")
        prepared = store.prepared_manifest
        prepared_payload = {
            key: value for key, value in prepared.items() if key not in {"logicalHash", "createdAt"}
        }
        prepared_id = prepared.get("domainId", prepared.get("corpusId"))
        text_path = source_evidence.get("textPathName")
        text_hash = source_evidence.get("textSha256")
        matching_text_shards = [
            item
            for item in prepared.get("shards", ())
            if isinstance(item, dict) and item.get("path") == text_path
        ]
        if (
            store.manifest.get("corpusId") != domain_id
            or prepared_id != domain_id
            or prepared.get("logicalHash") != canonical_sha256(prepared_payload)
            or text_path != "text.jsonl"
            or not isinstance(text_hash, str)
            or len(matching_text_shards) != 1
            or matching_text_shards[0].get("sha256") != text_hash
            or matching_text_shards[0].get("arrays") not in ([], ())
        ):
            raise ContractViolation(
                f"{domain_id} embedding source differs from prepared text inventory"
            )
        evidence[domain_id] = {
            "logicalHash": store.handle.logical_hash,
            # The immutable feature identity always binds the complete artifact;
            # the separately recorded access view proves which role shards this
            # process actually opened.
            "rows": int(store.manifest["rows"]),
            "modelId": source_evidence.get("modelId"),
            "modelRevision": source_evidence.get("modelRevision"),
            "modelInventoryHash": source_evidence.get("modelInventoryHash"),
            "sourceCorpusId": domain_id,
            "sourceTextPathName": text_path,
            "sourceTextSha256": text_hash,
            "preparedCorpusContentHash": prepared["logicalHash"],
            "producer": dict(producer),
            "accessView": {
                "maximumRole": store.maximum_role or "full",
                "selectedRows": store.handle.rows,
                "selectedShardPaths": list(store.handle.shard_paths),
                "selectedShardHashes": list(store.handle.shard_hashes),
            },
        }
    return evidence


__all__ = [
    "embed_gfm_text",
]
