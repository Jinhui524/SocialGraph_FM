"""Corpus acquisition, normalization, contracts, and task-asset checks.

The implementation is installed into the shared compatibility namespace by
:mod:`socialgraph_gfm.workflows` after all workflow stages are imported.
"""

# ruff: noqa: F403, F405
# mypy: disable-error-code=name-defined
from __future__ import annotations

from ._shared import *


def _registry(layout: RuntimeLayout) -> GfmRegistry:
    return GfmRegistry(layout.registry / REGISTRY_NAME)


def _contract_json(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=False)
    if not isinstance(value, dict):
        raise TypeError("workflow result is not JSON serializable")
    return value


def _write_contract(path: Path, value: Any) -> None:
    atomic_write_json(path, _contract_json(value))


def _leakage_audit(
    layout: RuntimeLayout,
    *,
    experiment_id: str,
    audit_id: str,
    evidence: Mapping[str, Any],
    counters: Mapping[str, int | float],
) -> tuple[str, str, dict[str, float]]:
    """Persist immutable, counter-bearing point-in-time audit evidence."""

    required = {
        "future_edge_access_count",
        "cutoff_violation_count",
        "split_overlap_count",
    }
    if not required.issubset(counters):
        raise ContractViolation("Leakage audit is missing required derived counters")
    normalized: dict[str, float] = {}
    for name, raw in counters.items():
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0.0
            or not float(raw).is_integer()
        ):
            raise ContractViolation("Leakage audit counters must be finite nonnegative integers")
        normalized[name] = float(raw)
    # A caller may report observed violations, but it cannot label that audit
    # as passed or register acceptance evidence.
    if any(value != 0.0 for value in normalized.values()):
        raise ContractViolation("Leakage audit observed a policy violation")
    if not evidence:
        raise ContractViolation("Leakage audit requires independently derived evidence")
    payload = {
        "schemaVersion": "gfm.leakage-audit/1.0",
        "experimentId": experiment_id,
        "auditId": audit_id,
        "counters": normalized,
        "evidence": dict(evidence),
    }
    logical_hash = canonical_sha256(payload)
    artifact = {**payload, "logicalHash": logical_hash}
    path = layout.gfm_reports / experiment_id / "audits" / f"{audit_id}.json"
    if path.is_file():
        current = read_json_object(path)
        if current != artifact:
            raise ContractViolation("Immutable leakage audit identity already differs")
    else:
        atomic_write_json(path, artifact)
    return file_sha256(path), str(path.resolve()), normalized


def _evaluation_evidence(
    layout: RuntimeLayout,
    *,
    experiment_id: str,
    evidence_id: str,
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    artifact = {
        "schemaVersion": "gfm.evaluation-evidence/1.0",
        "experimentId": experiment_id,
        "evidenceId": evidence_id,
        "payload": dict(payload),
    }
    artifact["logicalHash"] = canonical_sha256(artifact)
    path = layout.gfm_reports / experiment_id / "evidence" / f"{evidence_id}.json"
    if path.is_file():
        if read_json_object(path) != artifact:
            raise ContractViolation("Immutable evaluation evidence identity already differs")
    else:
        atomic_write_json(path, artifact)
    return file_sha256(path), str(path.resolve())


def _temporal_audit_counters(
    *,
    streams: Sequence[_DomainStream] | Sequence[Any],
    target_pretrain_access_count: int | None = None,
) -> dict[str, int]:
    """Derive violation counts from concrete event arrays and split bounds."""

    future = 0
    overlap = 0
    cutoff = 0
    for stream in streams:
        if stream.event_split is not None:
            split = np.asarray(stream.event_split)
            if split.dtype != np.dtype(np.int8):
                raise ContractViolation("Explicit event split dtype must be int8")
            train_rows = split == 0
            visible = stream.timestamp[train_rows]
            future += int(np.count_nonzero(visible > stream.train_end))
            cutoff += int(bool(visible.size) and int(visible.max()) > stream.train_end)
            # Candidate rows repeat a page by design; overlap means a page was
            # assigned to more than one role, not that complete page histories
            # occupy overlapping calendar years.
            minimum_role = np.full(stream.node_count, 3, dtype=np.int8)
            maximum_role = np.full(stream.node_count, -1, dtype=np.int8)
            last_timestamp = np.full(stream.node_count, -1, dtype=np.int64)
            np.minimum.at(minimum_role, stream.dst, split)
            np.maximum.at(maximum_role, stream.dst, split)
            np.maximum.at(last_timestamp, stream.dst, stream.timestamp)
            used_pages = maximum_role >= 0
            overlap += int(np.count_nonzero(minimum_role[used_pages] != maximum_role[used_pages]))
            expected_role = np.select(
                (
                    last_timestamp[used_pages] <= stream.train_end,
                    last_timestamp[used_pages] <= stream.validation_end,
                ),
                (0, 1),
                default=2,
            ).astype(np.int8)
            cutoff += int(np.count_nonzero(maximum_role[used_pages] != expected_role))
        else:
            train_count = int(np.searchsorted(stream.timestamp, stream.train_end, side="right"))
            visible = stream.timestamp[:train_count]
            future += int(np.count_nonzero(visible > stream.train_end))
            overlap += int(stream.validation_end <= stream.train_end)
            cutoff += int(train_count and int(visible[-1]) > int(stream.train_end))
    result = {
        "future_edge_access_count": future,
        "cutoff_violation_count": cutoff,
        "split_overlap_count": overlap,
    }
    if target_pretrain_access_count is not None:
        result["target_domain_pretrain_access_count"] = int(target_pretrain_access_count)
    return result


def _product_audit_counters(
    prepared: Iterable[_PreparedProductBatch] | Iterable[Any],
) -> dict[str, int]:
    """Derive product cutoff/split counters from the exact materialized batches."""

    future = 0
    cutoff = 0
    seen_queries: set[tuple[str, str, float, float, int]] = set()
    overlap = 0
    for item in prepared:
        batch = item.batch
        edge_time = batch.core_batch.edge_time.detach().cpu().numpy()
        cutoff_value = float(batch.core_batch.cutoff_time)
        future += int(np.count_nonzero(edge_time > cutoff_value))
        cutoff += int(batch.provenance.cutoff != cutoff_value)
        # A ranking query intentionally has many candidate rows with the same
        # query ID.  Split overlap is therefore defined over the unique query
        # identity, not over candidate rows.  Only recurrence in a distinct
        # materialized batch is a violation.
        batch_queries = {
            (
                batch.provenance.task_id,
                batch.provenance.source_corpus_hash,
                float(batch.provenance.cutoff),
                float(batch.provenance.horizon),
                int(query),
            )
            for query in batch.query_ids.detach().cpu().numpy().tolist()
        }
        overlap += len(batch_queries.intersection(seen_queries))
        seen_queries.update(batch_queries)
    return {
        "future_edge_access_count": future,
        "cutoff_violation_count": cutoff,
        "split_overlap_count": overlap,
    }


def _domain_alias(value: str) -> DomainAlias:
    if value in DOMAIN_IDS:
        return value  # type: ignore[return-value]
    if value in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[value]  # type: ignore[return-value]
    raise ContractViolation(f"Unsupported GFM domain: {value}")


def _parse_years(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        if ":" in value:
            start_text, end_text = value.split(":", 1)
            start, end = int(start_text), int(end_text)
            years = tuple(range(start, end + 1))
        else:
            years = tuple(int(item) for item in value.split(",") if item.strip())
    else:
        years = tuple(int(item) for item in value)
    if years != tuple(sorted(set(years))) or not years:
        raise ContractViolation("Wikimedia years must be a nonempty increasing unique range")
    return years


def fetch_gfm_openalex(
    *,
    root: str | Path | None,
    spec: str = "graph-ai",
    api_key_env: str = "OPENALEX_API_KEY",
) -> dict[str, Any]:
    """Fetch only the checked OpenAlex subset and never accept a key value."""

    if spec != "graph-ai":
        raise ContractViolation("The workflow accepts only --spec graph-ai")
    if api_key_env != "OPENALEX_API_KEY":
        raise ContractViolation("The OpenAlex key variable must be OPENALEX_API_KEY")
    if not os.environ.get(api_key_env, "").strip():
        raise ContractViolation("OPENALEX_API_KEY is absent; no API request was made")
    layout = prepare_runtime_layout(root, operation="fetch")
    load_openalex_spec()
    result = fetch_openalex(OpenAlexConfig.pinned(), layout.root)
    return {
        "schemaVersion": "gfm.workflow-fetch/1.0",
        "ok": True,
        "domainId": DOMAIN_IDS["openalex"],
        "result": result,
    }


def fetch_gfm_thgl_software(*, root: str | Path | None, accept_license: str) -> dict[str, Any]:
    layout = prepare_runtime_layout(root, operation="fetch")
    result = fetch_thgl_software(layout.root, accept_license=accept_license)
    return {
        "schemaVersion": "gfm.workflow-fetch/1.0",
        "ok": True,
        "domainId": DOMAIN_IDS["thgl-software"],
        "result": result,
    }


def fetch_gfm_wikimedia_talk(
    *,
    root: str | Path | None,
    years: str | Sequence[int],
    namespace: str,
    accept_license: str,
) -> dict[str, Any]:
    if namespace != "article":
        raise ContractViolation("SocialGraph-FM Core accepts only the article namespace")
    selected_years = _parse_years(years)
    layout = prepare_runtime_layout(root, operation="fetch")
    result = fetch_wikimedia(
        layout.root,
        accept_license=accept_license,
        years=selected_years,
    )
    return {
        "schemaVersion": "gfm.workflow-fetch/1.0",
        "ok": True,
        "domainId": DOMAIN_IDS["wikimedia-talk"],
        "years": list(selected_years),
        "result": result,
    }


def _prepared_counts(alias: DomainAlias, manifest: Mapping[str, Any]) -> tuple[int, int]:
    if alias == "openalex":
        node_counts = manifest.get("nodeCounts")
        if not isinstance(node_counts, dict):
            raise ContractViolation("OpenAlex manifest lacks typed node counts")
        node_count = sum(int(value) for value in node_counts.values())
        edge_count = int(manifest.get("edgeCount", 0))
    elif alias == "thgl-software":
        node_count = int(manifest.get("nodeCount", 0))
        edge_count = int(manifest.get("edgeCount", 0))
    else:
        node_count = int(manifest.get("userCount", 0)) + int(manifest.get("pageCount", 0))
        edge_count = int(manifest.get("eventCount", 0))
    if node_count < 1 or edge_count < 1:
        raise ContractViolation("Prepared GFM corpus has invalid node or edge counts")
    return node_count, edge_count


def _formal_corpus_contract(
    layout: RuntimeLayout,
    alias: DomainAlias,
    manifest: Mapping[str, Any],
) -> GfmDomainCorpusManifest:
    domain_id = DOMAIN_IDS[alias]
    node_count, edge_count = _prepared_counts(alias, manifest)
    source = manifest.get("source")
    splits = manifest.get("splits")
    if not isinstance(source, dict) or not isinstance(splits, dict):
        raise ContractViolation("Prepared GFM corpus lacks source or split evidence")
    evidence = source.get("licenseEvidence")
    if not isinstance(evidence, str) or not evidence:
        raise ContractViolation("Prepared GFM corpus lacks license evidence")
    logical_hash = manifest.get("logicalHash")
    if not isinstance(logical_hash, str) or len(logical_hash) != 64:
        raise ContractViolation("Prepared GFM corpus lacks a logical content hash")
    modalities: tuple[str, ...]
    tasks: tuple[str, ...]
    version: str
    if alias == "openalex":
        modalities = ("categorical", "text", "temporal", "structural")
        # The globally verified newcomer labels are a separately versioned
        # task asset.  Keeping them out of the immutable base-domain contract
        # lets pretraining and collaboration ranking proceed without changing
        # this corpus identity when the optional overlay is completed later.
        tasks = (COLLABORATION_TASK,)
        version = "graph-ai"
    elif alias == "thgl-software":
        modalities = ("categorical", "temporal", "structural")
        tasks = (COLLABORATION_TASK,)
        version = "2.0.0"
    else:
        modalities = ("text", "temporal", "structural")
        tasks = (NEWCOMER_TASK,)
        version = "article-2011-2015"
    artifact = layout.processed_gfm / domain_id
    privacy = manifest.get("privacy")
    if not isinstance(privacy, dict):
        raise ContractViolation("Prepared GFM corpus lacks privacy/promotion evidence")
    public_checkpoint_eligible = (
        source.get("formalEligible") is True and privacy.get("publicCheckpointEligible") is True
    )
    return GfmDomainCorpusManifest.create(
        corpusId=domain_id,
        domainId=domain_id,
        datasetName=domain_id,
        datasetVersion=version,
        datasetRole="pretraining",
        licenseId=str(manifest["licenseId"]),
        licenseEvidenceHash=canonical_sha256(
            {"licenseId": manifest["licenseId"], "evidence": evidence}
        ),
        sourceHash=canonical_sha256(source),
        contentHash=logical_hash,
        splitHash=canonical_sha256(splits),
        nodeCount=node_count,
        edgeCount=edge_count,
        featureModalities=modalities,
        taskIds=tasks,
        pointInTimeSafe=True,
        publicCheckpointEligible=public_checkpoint_eligible,
        temporalCutoff=None,
        sourceUri=source.get("uri") or source.get("articleApi"),
        artifactPath=str(artifact.resolve()),
    )


def _raw_wikimedia_files(layout: RuntimeLayout) -> list[Path]:
    receipt = read_json_object(layout.raw_wikimedia_talk / "fetch-receipt.json")
    records = receipt.get("files")
    if not isinstance(records, list) or len(records) != 5:
        raise ContractViolation(
            "Formal Wikimedia preparation requires accepted files for all 2011--2015 years"
        )
    result: list[Path] = []
    years: list[int] = []
    for record in records:
        if not isinstance(record, dict):
            raise ContractViolation("Wikimedia fetch receipt contains a malformed file")
        name = record.get("name")
        if not isinstance(name, str):
            raise ContractViolation("Wikimedia fetch receipt lacks a file name")
        result.append(layout.raw_wikimedia_talk / name)
        years.append(int(record["year"]))
    if tuple(sorted(years)) != (2011, 2012, 2013, 2014, 2015):
        raise ContractViolation("Formal Wikimedia receipt does not cover 2011--2015")
    return result


def prepare_gfm_corpus(
    *,
    root: str | Path | None,
    domain: str,
    newcomer_overlay: NewcomerOverlayMode = "skip",
) -> dict[str, Any]:
    """Prepare one base corpus and optionally resume the newcomer task asset.

    OpenAlex newcomer verification is intentionally not part of the base
    corpus identity.  ``skip`` is the normal pretraining/collaboration path;
    ``require`` resumes the separately hash-bound overlay and fails closed if
    it cannot finish.  Other domains never accept an overlay mode.
    """

    alias = _domain_alias(domain)
    if newcomer_overlay not in ("skip", "require"):
        raise ContractViolation("newcomer_overlay must be skip or require")
    if alias != "openalex" and newcomer_overlay != "skip":
        raise ContractViolation("The newcomer overlay is valid only for the OpenAlex domain")
    layout = prepare_runtime_layout(root, operation="run")
    if alias == "openalex":
        prepared = prepare_openalex(layout.raw_openalex, layout.root)
    elif alias == "thgl-software":
        require_gfm_optional_runtime(data=True)
        prepared = prepare_thgl_software(
            layout.raw_thgl_software / "thgl-software-2.0.0.zip",
            layout.root,
        )
    else:
        prepared = prepare_wikimedia(_raw_wikimedia_files(layout), layout.root)
    checked = load_domain(layout.root, DOMAIN_IDS[alias])["manifest"]
    if checked.get("logicalHash") != prepared.get("logicalHash"):
        raise ContractViolation("Prepared corpus changed during its verification read")
    contract = _formal_corpus_contract(layout, alias, checked)
    contract_path = layout.manifests_gfm / f"{contract.corpus_id}.json"
    _write_contract(contract_path, contract)
    _registry(layout).record_corpus(contract)
    overlay: dict[str, Any] = {
        "required": False,
        "ready": False,
        "deferred": False,
        "state": "not-applicable",
    }
    if alias == "openalex":
        overlay = {
            **newcomer_overlay_status(layout.root),
            "required": newcomer_overlay == "require",
            "deferred": newcomer_overlay == "skip",
        }
    if alias == "openalex" and newcomer_overlay == "require":
        if not os.environ.get("OPENALEX_API_KEY", "").strip():
            raise ContractViolation(
                "OPENALEX_API_KEY is required only when --newcomer-overlay require is selected"
            )
        verify_openalex_newcomers(layout.root)
        overlay = {
            **newcomer_overlay_status(layout.root),
            "required": True,
            "deferred": False,
        }
        if overlay.get("ready") is not True:
            raise ContractViolation(
                "OpenAlex newcomer verification did not publish a valid standalone overlay"
            )
    return {
        "schemaVersion": "gfm.workflow-prepare/1.0",
        "ok": True,
        "domainId": contract.domain_id,
        "corpusHash": contract.logical_hash,
        "manifest": str(contract_path),
        "preparedManifestHash": checked["logicalHash"],
        "newcomerOverlay": overlay,
    }


def check_gfm_task_assets(
    *, root: str | Path | None, task: ProductTask | None = None
) -> dict[str, Any]:
    """Read and hash task-specific data gates without starting training."""

    if task not in (None, "collaboration", "newcomer"):
        raise ContractViolation("GFM task asset check supports collaboration or newcomer")
    layout = prepare_runtime_layout(root, operation="run")
    selected: tuple[ProductTask, ...]
    if task is None:
        selected = ("collaboration", "newcomer")
    else:
        selected = (task,)
    try:
        corpora = _load_corpus_contracts(layout)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        return {
            "schemaVersion": "gfm.workflow-task-assets/1.0",
            "ok": False,
            "tasks": {
                selected_task: {
                    "ready": False,
                    "evidence": None,
                    "evidenceHash": None,
                    "reason": str(error),
                }
                for selected_task in selected
            },
        }
    results: dict[str, Any] = {}
    for selected_task in selected:
        try:
            evidence = _product_task_asset_evidence(layout, task=selected_task, corpora=corpora)
            results[selected_task] = {
                "ready": True,
                "evidence": evidence,
                "evidenceHash": canonical_sha256(evidence),
                "reason": None,
            }
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
            results[selected_task] = {
                "ready": False,
                "evidence": None,
                "evidenceHash": None,
                "reason": str(error),
            }
    return {
        "schemaVersion": "gfm.workflow-task-assets/1.0",
        "ok": all(item["ready"] is True for item in results.values()),
        "tasks": results,
    }


__all__ = [
    "check_gfm_task_assets",
    "fetch_gfm_openalex",
    "fetch_gfm_thgl_software",
    "fetch_gfm_wikimedia_talk",
    "prepare_gfm_corpus",
]
