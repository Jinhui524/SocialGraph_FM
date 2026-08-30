"""Read-only readiness aggregation for the GFM infrastructure boundary."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from .canonical import canonical_sha256, file_sha256
from .fixtures import get_fixture
from .fixtures import smoke_fit_node_ids
from .identity import DEFAULT_SMOKE_SEED, code_identity_hash, smoke_config
from .locks import verify_lock_manifest
from .profiles import check_compatibility
from .registry import LocalRegistry
from .runtime import (
    StorageOperation,
    artifact_root,
    gfm_optional_runtime_report,
    runtime_report,
    storage_report,
)
from .materialize import homogeneous_tensors, materialize


_GFM_REQUIRED_DOMAINS = frozenset(
    {
        "openalex-graph-ai",
        "thgl-software-2.0.0",
        "wikimedia-talk-article-2011-2015",
    }
)

_COLLABORATION_TASK = "governance.collaboration_recommendation"
_NEWCOMER_TASK = "core.newcomer_support"


def _root_report(root: Path) -> dict[str, Any]:
    # The runtime root is commonly created after preflight.  Probe that exact
    # directory when it exists, otherwise the nearest existing parent where it
    # would be created.  Checking only the filesystem anchor ("/" on POSIX)
    # incorrectly reports ordinary non-root macOS/Linux users as unwritable.
    probe_path = root
    while not probe_path.exists() and probe_path.parent != probe_path:
        probe_path = probe_path.parent
    exists = probe_path.is_dir()
    return {
        "path": str(root),
        "anchor": str(probe_path),
        "anchorExists": exists,
        "anchorWritable": exists and os.access(probe_path, os.W_OK),
    }


def _storage_evidence(root: Path, operation: StorageOperation) -> dict[str, Any]:
    try:
        return storage_report(root, operation=operation)
    except OSError as error:
        return {
            "schemaVersion": "gfm.storage/1.0",
            "operation": operation,
            "root": str(root),
            "ready": False,
            "reason": str(error),
        }


def _smoke_coverage(root: Path, *, device: str, environment_hash: str) -> set[str]:
    database = root / "registry" / "registry.sqlite3"
    if not database.is_file():
        return set()
    try:
        config_hashes: dict[str, str] = {}
        for fixture in ("actor", "hetero"):
            value = materialize(
                get_fixture(fixture),
                purpose="training_smoke",
                fit_node_ids=smoke_fit_node_ids(fixture),
                device=device,
            )
            x, _ = homogeneous_tensors(value)
            config_hashes[fixture] = canonical_sha256(
                smoke_config(
                    fixture=fixture,
                    seed=DEFAULT_SMOKE_SEED,
                    device=device,
                    input_dim=int(x.shape[1]),
                )
            )
        # LocalRegistry does not create anything here because the database already exists.
        return LocalRegistry(database, initialize=False).successful_smoke_coverage(
            code_hash=code_identity_hash(),
            environment_hash=environment_hash,
            config_hashes=config_hashes,
            device=device,
        )
    except (sqlite3.DatabaseError, OSError, RuntimeError):
        return set()


def _formal_corpus_evidence(root: Path) -> tuple[dict[str, Any], str | None]:
    evidence: dict[str, Any] = {
        "ready": False,
        "corpusId": "ogbl-collab",
        "manifestPath": str(root / "datasets" / "manifests" / "ogbl-collab.json"),
        "manifestHash": None,
        "reason": "formal corpus manifest is absent or invalid",
    }
    try:
        from .corpus import check_ogbl_collab_corpus

        manifest = check_ogbl_collab_corpus(root)
        if hasattr(manifest, "model_dump"):
            manifest = manifest.model_dump(mode="json", by_alias=True, exclude_none=False)
        if not isinstance(manifest, dict):
            raise ValueError("corpus checker did not return an object")
        manifest_hash = (
            manifest.get("manifestHash")
            or manifest.get("manifest_hash")
            or manifest.get("logicalHash")
            or manifest.get("logical_hash")
        )
        if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
            raise ValueError("checked corpus manifest has no valid manifestHash")
        evidence.update(
            {
                "ready": True,
                "manifestHash": manifest_hash,
                "reason": None,
            }
        )
        return evidence, manifest_hash
    except (ImportError, KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        evidence["reason"] = str(error)
        return evidence, None


def _baseline_evidence(root: Path, corpus_manifest_hash: str | None) -> dict[str, Any]:
    acceptance_path = root / "reports" / "baseline-acceptance.json"
    database = root / "registry" / "registry.sqlite3"
    evidence: dict[str, Any] = {
        "ready": False,
        "acceptancePath": str(acceptance_path),
        "manifestHash": None,
        "experimentId": None,
        "reasons": [],
    }
    if corpus_manifest_hash is None:
        evidence["reasons"] = ["CorpusReady evidence is required"]
        return evidence
    if not acceptance_path.is_file():
        evidence["reasons"] = ["baseline acceptance manifest is absent"]
        return evidence
    if not database.is_file():
        evidence["reasons"] = ["baseline registry is absent"]
        return evidence
    try:
        payload = json.loads(acceptance_path.read_text(encoding="utf-8"))
        try:
            from .contracts import BaselineAcceptanceReport

            acceptance: Any = BaselineAcceptanceReport.model_validate(payload)
        except ImportError:
            acceptance = payload
        result = LocalRegistry(database, initialize=False).validate_baseline_acceptance(
            acceptance,
            corpus_manifest_hash=corpus_manifest_hash,
        )
        evidence.update(result)
        return evidence
    except (json.JSONDecodeError, sqlite3.DatabaseError, OSError, TypeError, ValueError, RuntimeError) as error:
        evidence["reasons"] = [str(error)]
        return evidence


def _gfm_corpus_evidence(root: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Re-read every formal domain artifact; directory presence is never evidence."""

    evidence: dict[str, Any] = {
        "ready": False,
        "requiredDomains": sorted(_GFM_REQUIRED_DOMAINS),
        "domainManifestHashes": {},
        "reason": "three checked GFM domain corpora are required",
    }
    try:
        from .gfm.corpus import check_all_gfm_corpora

        checked = check_all_gfm_corpora(root)
        domains = checked.get("domains")
        if checked.get("ready") is not True or not isinstance(domains, dict):
            raise ValueError("GFM corpus checker did not attest all required domains")
        hashes: dict[str, str] = {}
        for domain_id in evidence["requiredDomains"]:
            manifest = domains.get(domain_id)
            if not isinstance(manifest, dict):
                raise ValueError(f"checked GFM corpus is absent: {domain_id}")
            privacy = manifest.get("privacy")
            if not isinstance(privacy, dict) or privacy.get(
                "publicCheckpointEligible"
            ) is not True:
                raise ValueError(
                    "checked GFM corpus is not eligible for a public checkpoint: "
                    f"{domain_id}"
                )
            logical_hash = manifest.get("logicalHash")
            if not isinstance(logical_hash, str) or len(logical_hash) != 64:
                raise ValueError(f"checked GFM corpus has no logical hash: {domain_id}")
            hashes[domain_id] = logical_hash
        evidence.update(
            {
                "ready": True,
                "domainManifestHashes": hashes,
                "reason": None,
            }
        )
        return evidence, tuple(sorted(hashes.values()))
    except (ImportError, KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        evidence["reason"] = str(error)
        return evidence, ()


def _gfm_task_asset_evidence(
    root: Path,
    *,
    gfm_corpus_ready: bool,
) -> dict[str, Any]:
    """Derive task-specific corpus readiness without weakening final acceptance.

    The globally verified OpenAlex newcomer cohort is an optional, immutable
    overlay on the base OpenAlex graph.  It is deliberately *not* part of
    ``GfmCorpusReady``: three-domain pretraining and collaboration adaptation
    consume only the base corpora.  The newcomer task remains fail-closed and
    needs both the base three-domain corpus and a deeply re-read overlay.

    This is operator-facing evidence rather than a serving capability promise.
    Model/product validation is still derived exclusively from formal
    acceptance evidence later in :func:`preflight_report`.
    """

    overlay: dict[str, Any] = {
        "ready": False,
        "manifestHash": None,
        "sourceCorpusHash": None,
        "verifiedCount": None,
        "reason": "globally verified OpenAlex newcomer overlay is absent",
    }
    try:
        from .gfm.corpus import newcomer_overlay_status

        status = newcomer_overlay_status(root)
        overlay.update(
            {
                "ready": status.get("ready") is True,
                "state": status.get("state"),
                "manifestHash": status.get("manifestHash"),
                "sourceCorpusHash": status.get("baseCorpusSourceHash"),
                "baseCorpusLogicalHash": status.get("baseCorpusLogicalHash"),
                "verifiedCount": status.get("verifiedCount"),
                "resumePresent": status.get("resumePresent") is True,
                "reason": status.get("reason"),
            }
        )
    except (ImportError, KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        overlay["reason"] = str(error)

    collaboration_ready = bool(gfm_corpus_ready)
    newcomer_ready = collaboration_ready and bool(overlay["ready"])
    return {
        "schemaVersion": "gfm.task-assets/1.0",
        "baseCorporaReady": collaboration_ready,
        "newcomerOverlay": overlay,
        "tasks": {
            _COLLABORATION_TASK: {
                "ready": collaboration_ready,
                "requiredAssets": ["gfm-base-corpora"],
                "missingAssets": (
                    [] if collaboration_ready else ["gfm-base-corpora"]
                ),
            },
            _NEWCOMER_TASK: {
                "ready": newcomer_ready,
                "requiredAssets": [
                    "gfm-base-corpora",
                    "openalex-newcomer-overlay",
                ],
                "missingAssets": [
                    asset
                    for asset, ready in (
                        ("gfm-base-corpora", collaboration_ready),
                        ("openalex-newcomer-overlay", bool(overlay["ready"])),
                    )
                    if not ready
                ],
            },
        },
    }


def _gfm_acceptance_evidence(
    root: Path,
    *,
    checked_domain_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the newest immutable GFM acceptance without mutating the registry."""

    database = root / "registry" / "gfm-registry.sqlite3"
    evidence: dict[str, Any] = {
        "ready": False,
        "registryPath": str(database),
        "experimentId": None,
        "reportHash": None,
        "accepted": False,
        "gates": {},
        "pretrainingValidated": False,
        "productValidated": False,
        "reasons": ["formal GFM acceptance evidence is absent"],
    }
    if not checked_domain_hashes:
        evidence["reasons"] = ["GfmCorpusReady evidence is required"]
        return evidence


    if not database.is_file():
        return evidence
    try:
        from .gfm.acceptance import build_gfm_acceptance
        from .gfm.checkpoint import load_gfm_checkpoint
        from .gfm.contracts import (
            GfmAcceptanceManifest,
            GfmCheckpointManifest,
            GfmDomainCorpusManifest,
            GfmEvaluationReport,
            GfmRunManifest,
        )

        if set(checked_domain_hashes) != _GFM_REQUIRED_DOMAINS:
            raise ValueError("checked raw GFM corpus evidence is incomplete")

        # The portable corpus contracts are immutable artifacts in their own right.
        # Re-reading them prevents an externally edited registry row from asserting a
        # different point-in-time or licence contract for otherwise identical bytes.
        artifact_corpora: dict[str, GfmDomainCorpusManifest] = {}
        manifest_directory = root / "datasets" / "manifests" / "gfm"
        for domain_id in sorted(_GFM_REQUIRED_DOMAINS):
            path = manifest_directory / f"{domain_id}.json"
            corpus = GfmDomainCorpusManifest.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if corpus.domain_id != domain_id or corpus.corpus_id != domain_id:
                raise ValueError(f"GFM corpus contract identity mismatch: {domain_id}")
            if not corpus.public_checkpoint_eligible:
                raise ValueError(
                    "GFM corpus contract is not eligible for a public checkpoint: "
                    f"{domain_id}"
                )
            if corpus.content_hash != checked_domain_hashes[domain_id]:
                raise ValueError(
                    f"checked GFM corpus content differs from its contract: {domain_id}"
                )
            artifact_corpora[domain_id] = corpus

        connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT report_hash, experiment_id, checkpoint_id, accepted, manifest_json
                FROM gfm_acceptances
                ORDER BY created_at DESC, report_hash DESC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return evidence
            acceptance = GfmAcceptanceManifest.model_validate_json(row["manifest_json"])
            if (
                row["report_hash"] != acceptance.report_hash
                or row["experiment_id"] != acceptance.experiment_id
                or row["checkpoint_id"] != acceptance.checkpoint_id
                or bool(row["accepted"]) != acceptance.accepted
            ):
                raise ValueError("GFM acceptance registry columns differ from its contract")

            artifact_hashes = tuple(
                sorted(corpus.logical_hash for corpus in artifact_corpora.values())
            )
            accepted_hashes = tuple(sorted(acceptance.corpus_hashes))
            if accepted_hashes != artifact_hashes:
                raise ValueError(
                    "GFM acceptance corpus hashes differ from checked contract artifacts"
                )

            placeholders = ",".join("?" for _ in accepted_hashes)
            corpus_rows = connection.execute(
                f"""
                SELECT corpus_id, logical_hash, domain_id, manifest_json
                FROM gfm_domain_corpora
                WHERE logical_hash IN ({placeholders})
                """,
                accepted_hashes,
            ).fetchall()
            if len(corpus_rows) != len(_GFM_REQUIRED_DOMAINS):
                raise ValueError("GFM acceptance corpus contracts are not all registered")
            corpora: list[GfmDomainCorpusManifest] = []
            for corpus_row in corpus_rows:
                corpus = GfmDomainCorpusManifest.model_validate_json(
                    corpus_row["manifest_json"]
                )
                if (
                    corpus_row["corpus_id"] != corpus.corpus_id
                    or corpus_row["logical_hash"] != corpus.logical_hash
                    or corpus_row["domain_id"] != corpus.domain_id
                ):
                    raise ValueError(
                        "GFM corpus registry columns differ from their contract"
                    )
                artifact = artifact_corpora.get(corpus.domain_id)
                if (
                    artifact is None
                    or corpus.logical_hash != artifact.logical_hash
                    or corpus.logical_payload() != artifact.logical_payload()
                ):
                    raise ValueError(
                        "registered GFM corpus differs from the checked contract artifact"
                    )
                corpora.append(corpus)
            if {corpus.domain_id for corpus in corpora} != _GFM_REQUIRED_DOMAINS:
                raise ValueError("registered GFM corpus domain coverage is incomplete")

            checkpoint_row = connection.execute(
                """
                SELECT checkpoint_id, logical_hash, run_id, artifact_sha256, manifest_json
                FROM gfm_checkpoints WHERE checkpoint_id=?
                """,
                (acceptance.checkpoint_id,),
            ).fetchone()
            if checkpoint_row is None:
                raise ValueError("GFM acceptance checkpoint is not registered")
            checkpoint = GfmCheckpointManifest.model_validate_json(
                checkpoint_row["manifest_json"]
            )
            if (
                checkpoint_row["checkpoint_id"] != checkpoint.checkpoint_id
                or checkpoint_row["logical_hash"] != checkpoint.logical_hash
                or checkpoint_row["run_id"] != checkpoint.run_id
                or checkpoint_row["artifact_sha256"] != checkpoint.artifact_sha256
            ):
                raise ValueError(
                    "GFM checkpoint registry columns differ from its contract"
                )
            run_row = connection.execute(
                """
                SELECT run_id, manifest_hash, experiment_id, status, config_hash,
                       manifest_json
                FROM gfm_runs WHERE run_id=?
                """,
                (checkpoint.run_id,),
            ).fetchone()
            if run_row is None:
                raise ValueError("GFM acceptance checkpoint run is not registered")
            run = GfmRunManifest.model_validate_json(run_row["manifest_json"])
            if (
                run_row["run_id"] != run.run_id
                or run_row["manifest_hash"] != run.manifest_hash
                or run_row["experiment_id"] != run.experiment_id
                or run_row["status"] != run.status
                or run_row["config_hash"] != run.config_hash
                or run.status != "succeeded"
                or run.experiment_id != acceptance.experiment_id
                or checkpoint.config_hash != run.config_hash
                or tuple(sorted(checkpoint.corpus_hashes)) != accepted_hashes
                or tuple(sorted(run.corpus_hashes)) != accepted_hashes
            ):
                raise ValueError("GFM acceptance checkpoint provenance is inconsistent")
            # Registry columns and a manifest-declared SHA are not physical
            # evidence.  Re-open the selected delivery artifact with the
            # weights-only checkpoint loader so deletion, byte tampering and
            # state/payload drift all make readiness fail closed.
            selected_payload = load_gfm_checkpoint(checkpoint, map_location="cpu")
            del selected_payload

            report_hashes = tuple(sorted(acceptance.evaluation_report_hashes))
            if not report_hashes:
                raise ValueError("GFM acceptance has no evaluation evidence")
            report_placeholders = ",".join("?" for _ in report_hashes)
            evaluation_rows = connection.execute(
                f"""
                SELECT report_id, report_hash, experiment_id, run_id, checkpoint_id,
                       evaluation_kind, manifest_json
                FROM gfm_evaluations
                WHERE report_hash IN ({report_placeholders})
                """,
                report_hashes,
            ).fetchall()
            if len(evaluation_rows) != len(report_hashes):
                raise ValueError("GFM acceptance evaluation evidence is incomplete")
            evaluations: list[GfmEvaluationReport] = []
            report_root = (root / "reports" / "gfm").resolve()
            for evaluation_row in evaluation_rows:
                report = GfmEvaluationReport.model_validate_json(
                    evaluation_row["manifest_json"]
                )
                if (
                    evaluation_row["report_id"] != report.report_id
                    or evaluation_row["report_hash"] != report.report_hash
                    or evaluation_row["experiment_id"] != report.experiment_id
                    or evaluation_row["run_id"] != report.run_id
                    or evaluation_row["checkpoint_id"] != report.checkpoint_id
                    or evaluation_row["evaluation_kind"] != report.evaluation_kind
                    or report.experiment_id != acceptance.experiment_id
                    or "shadow" in report.warnings
                ):
                    raise ValueError(
                        "GFM evaluation registry columns or experiment identity differ"
                    )
                evidence_path = Path(report.evidence_artifact_path).resolve()
                audit_path = Path(report.leakage_audit_path).resolve()
                for artifact_path, expected_hash, label in (
                    (evidence_path, report.evidence_artifact_hash, "evaluation evidence"),
                    (audit_path, report.leakage_audit_hash, "leakage audit"),
                ):
                    if (
                        not artifact_path.is_relative_to(report_root)
                        or not artifact_path.is_file()
                        or file_sha256(artifact_path) != expected_hash
                    ):
                        raise ValueError(
                            f"GFM {label} artifact is absent, outside runtime, or changed"
                        )
                evidence_artifact = json.loads(
                    evidence_path.read_text(encoding="utf-8")
                )
                audit_artifact = json.loads(audit_path.read_text(encoding="utf-8"))
                if not isinstance(evidence_artifact, dict) or not isinstance(
                    audit_artifact, dict
                ):
                    raise ValueError("GFM evaluation evidence artifacts must be objects")
                evidence_logical_hash = evidence_artifact.get("logicalHash")
                evidence_payload_for_hash = {
                    key: value
                    for key, value in evidence_artifact.items()
                    if key != "logicalHash"
                }
                evidence_payload = evidence_artifact.get("payload")
                if (
                    evidence_artifact.get("schemaVersion")
                    != "gfm.evaluation-evidence/1.0"
                    or evidence_artifact.get("experimentId") != report.experiment_id
                    or not isinstance(evidence_payload, dict)
                    or evidence_logical_hash
                    != canonical_sha256(evidence_payload_for_hash)
                    or evidence_payload.get("metrics") != report.metrics
                ):
                    raise ValueError(
                        "GFM evaluation metrics are not bound to verified evidence"
                    )
                audit_logical_hash = audit_artifact.get("logicalHash")
                audit_payload_for_hash = {
                    key: value
                    for key, value in audit_artifact.items()
                    if key != "logicalHash"
                }
                audit_counters = audit_artifact.get("counters")
                required_audit_counters = {
                    "future_edge_access_count",
                    "cutoff_violation_count",
                    "split_overlap_count",
                    *(
                        {"target_domain_pretrain_access_count"}
                        if report.evaluation_kind == "lodo"
                        else set()
                    ),
                }
                if (
                    audit_artifact.get("schemaVersion")
                    != "gfm.leakage-audit/1.0"
                    or audit_artifact.get("experimentId") != report.experiment_id
                    or audit_logical_hash != canonical_sha256(audit_payload_for_hash)
                    or not isinstance(audit_counters, dict)
                    or set(audit_counters) != required_audit_counters
                    or any(
                        report.metrics.get(name) != value
                        for name, value in audit_counters.items()
                    )
                ):
                    raise ValueError(
                        "GFM leakage counters are not bound to verified audit evidence"
                    )
                evaluations.append(report)

            evaluation_run_ids = tuple(sorted({item.run_id for item in evaluations}))
            evaluation_checkpoint_ids = tuple(
                sorted({item.checkpoint_id for item in evaluations})
            )
            run_placeholders = ",".join("?" for _ in evaluation_run_ids)
            checkpoint_placeholders = ",".join(
                "?" for _ in evaluation_checkpoint_ids
            )
            evaluation_run_rows = connection.execute(
                f"""
                SELECT run_id, manifest_hash, experiment_id, status, config_hash,
                       manifest_json
                FROM gfm_runs WHERE run_id IN ({run_placeholders})
                """,
                evaluation_run_ids,
            ).fetchall()
            evaluation_checkpoint_rows = connection.execute(
                f"""
                SELECT checkpoint_id, logical_hash, run_id, artifact_sha256,
                       manifest_json
                FROM gfm_checkpoints
                WHERE checkpoint_id IN ({checkpoint_placeholders})
                """,
                evaluation_checkpoint_ids,
            ).fetchall()
            if len(evaluation_run_rows) != len(evaluation_run_ids) or len(
                evaluation_checkpoint_rows
            ) != len(evaluation_checkpoint_ids):
                raise ValueError(
                    "GFM evaluation run/checkpoint provenance is incomplete"
                )
            evaluation_runs: dict[str, GfmRunManifest] = {}
            for item in evaluation_run_rows:
                evaluation_run = GfmRunManifest.model_validate_json(
                    item["manifest_json"]
                )
                if (
                    item["run_id"] != evaluation_run.run_id
                    or item["manifest_hash"] != evaluation_run.manifest_hash
                    or item["experiment_id"] != evaluation_run.experiment_id
                    or item["status"] != evaluation_run.status
                    or item["config_hash"] != evaluation_run.config_hash
                ):
                    raise ValueError(
                        "GFM evaluation run columns differ from their contract"
                    )
                evaluation_runs[evaluation_run.run_id] = evaluation_run
            evaluation_checkpoints: dict[str, GfmCheckpointManifest] = {}
            for item in evaluation_checkpoint_rows:
                evaluation_checkpoint = GfmCheckpointManifest.model_validate_json(
                    item["manifest_json"]
                )
                if (
                    item["checkpoint_id"] != evaluation_checkpoint.checkpoint_id
                    or item["logical_hash"] != evaluation_checkpoint.logical_hash
                    or item["run_id"] != evaluation_checkpoint.run_id
                    or item["artifact_sha256"]
                    != evaluation_checkpoint.artifact_sha256
                ):
                    raise ValueError(
                        "GFM evaluation checkpoint columns differ from their contract"
                    )
                evaluation_checkpoints[
                    evaluation_checkpoint.checkpoint_id
                ] = evaluation_checkpoint

            # Every checkpoint that contributes a frozen evaluation must still
            # be a loadable, hash-bound physical artifact.  This deliberately
            # checks unique checkpoint identities once rather than trusting the
            # report or registry attestation.
            for evaluation_checkpoint in evaluation_checkpoints.values():
                evaluation_payload = load_gfm_checkpoint(
                    evaluation_checkpoint, map_location="cpu"
                )
                del evaluation_payload

            selected_variant = run.architecture_variant
            accepted_hash_set = set(accepted_hashes)
            for report in evaluations:
                evaluation_run = evaluation_runs[report.run_id]
                evaluation_checkpoint = evaluation_checkpoints[report.checkpoint_id]
                if (
                    evaluation_checkpoint.run_id != evaluation_run.run_id
                    or evaluation_run.status != "succeeded"
                    or evaluation_run.experiment_id != acceptance.experiment_id
                    or evaluation_run.seed != report.seed
                    or evaluation_run.config_hash != run.config_hash
                    or evaluation_run.code_hash != run.code_hash
                    or evaluation_run.environment_hash != run.environment_hash
                    or evaluation_run.architecture_variant != selected_variant
                    or evaluation_checkpoint.config_hash != evaluation_run.config_hash
                    or set(evaluation_checkpoint.corpus_hashes)
                    != set(evaluation_run.corpus_hashes)
                    or not set(evaluation_run.corpus_hashes).issubset(
                        accepted_hash_set
                    )
                ):
                    raise ValueError(
                        "GFM evaluation is not bound to compatible immutable provenance"
                    )
                if report.evaluation_kind == "lodo":
                    if (
                        evaluation_run.phase != "lodo"
                        or evaluation_run.held_out_domain != report.held_out_domain
                        or report.domain_id != report.held_out_domain
                    ):
                        raise ValueError("GFM LODO evaluation provenance is inconsistent")
                elif evaluation_run.phase == "lodo":
                    raise ValueError(
                        "non-LODO GFM evidence cannot be attached to a LODO run"
                    )

            # Recompute every frozen hard gate from only the exact, immutable evidence
            # named by the acceptance.  No stored `accepted` or gate boolean is trusted.
            recomputed = build_gfm_acceptance(
                experiment_id=run.experiment_id,
                checkpoint_id=checkpoint.checkpoint_id,
                corpora=tuple(corpora),
                evaluations=tuple(evaluations),
                config_hash=run.config_hash,
                code_hash=run.code_hash,
                environment_hash=run.environment_hash,
                delivery_evidence_report_hashes=(
                    acceptance.delivery_evidence_report_hashes
                ),
            )
            if (
                recomputed.logical_payload() != acceptance.logical_payload()
                or recomputed.report_hash != acceptance.report_hash
            ):
                raise ValueError(
                    "stored GFM acceptance differs from safely recomputed evidence"
                )
        finally:
            connection.close()

        gates = dict(acceptance.gates)
        pretrain_gate_names = {
            "three_domains",
            "lodo_complete",
            "cuda_memory",
            "fresh_process_verification",
            "temporal_leakage_audit",
        }
        product_gate_names = pretrain_gate_names | {
            "product_metrics",
            "calibration_ece",
        }
        pretraining_validated = all(gates.get(name) is True for name in pretrain_gate_names)
        product_validated = all(gates.get(name) is True for name in product_gate_names)
        evidence.update(
            {
                "ready": bool(acceptance.accepted),
                "experimentId": acceptance.experiment_id,
                "reportHash": acceptance.report_hash,
                "accepted": acceptance.accepted,
                "configHash": acceptance.config_hash,
                "codeHash": acceptance.code_hash,
                "environmentHash": acceptance.environment_hash,
                "gates": gates,
                "pretrainingValidated": pretraining_validated,
                "productValidated": product_validated,
                "reasons": list(acceptance.reasons),
            }
        )
        return evidence
    except (
        ImportError,
        json.JSONDecodeError,
        sqlite3.DatabaseError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        evidence["reasons"] = [str(error)]
        return evidence


def _gfm_collaboration_task_acceptance_evidence(
    root: Path,
    *,
    checked_domain_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Recompute the non-promotable collaboration-only product gate."""

    database = root / "registry" / "gfm-registry.sqlite3"
    evidence: dict[str, Any] = {
        "ready": False,
        "registryPath": str(database),
        "experimentId": None,
        "reportHash": None,
        "accepted": False,
        "taskId": "governance.collaboration_recommendation",
        "gates": {},
        "reasons": ["formal collaboration task acceptance evidence is absent"],
        "promotable": False,
        "exportable": False,
    }
    if set(checked_domain_hashes) != _GFM_REQUIRED_DOMAINS:
        evidence["reasons"] = ["GfmCorpusReady evidence is required"]
        return evidence
    if not database.is_file():
        return evidence
    try:
        from .gfm.contracts import GfmDomainCorpusManifest
        from .gfm.registry import GfmRegistry

        registry = GfmRegistry(database, initialize=False)
        acceptance = registry.latest_task_acceptance()
        if acceptance is None:
            return evidence
        recomputed = registry.verify_task_acceptance(acceptance)
        manifests = []
        manifest_directory = root / "datasets" / "manifests" / "gfm"
        for domain_id in sorted(_GFM_REQUIRED_DOMAINS):
            manifest = GfmDomainCorpusManifest.model_validate_json(
                (manifest_directory / f"{domain_id}.json").read_text(encoding="utf-8")
            )
            if (
                manifest.domain_id != domain_id
                or manifest.content_hash != checked_domain_hashes[domain_id]
            ):
                raise ValueError(
                    "collaboration acceptance corpus differs from checked corpus evidence"
                )
            manifests.append(manifest)
        if set(recomputed.corpus_hashes) != {
            manifest.logical_hash for manifest in manifests
        }:
            raise ValueError(
                "collaboration acceptance is not bound to all checked GFM corpora"
            )
        artifact_path = (
            root
            / "reports"
            / "gfm"
            / recomputed.experiment_id
            / "task-acceptance"
            / f"collaboration-product-{recomputed.report_hash}.json"
        )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact != recomputed.model_dump(mode="json", by_alias=True):
            raise ValueError(
                "collaboration task acceptance artifact differs from its registry contract"
            )
        evidence.update(
            {
                "ready": bool(recomputed.accepted),
                "experimentId": recomputed.experiment_id,
                "reportHash": recomputed.report_hash,
                "accepted": recomputed.accepted,
                "architectureVariant": recomputed.architecture_variant,
                "formalSeeds": list(recomputed.formal_seeds),
                "backboneCheckpointIds": list(
                    recomputed.backbone_checkpoint_ids
                ),
                "backboneStateHashes": list(recomputed.backbone_state_hashes),
                "pretrainingAcceptanceReportHash": (
                    recomputed.pretraining_acceptance_report_hash
                ),
                "configHash": recomputed.config_hash,
                "codeHash": recomputed.code_hash,
                "environmentHash": recomputed.environment_hash,
                "protocolHash": recomputed.protocol_hash,
                "artifactPath": str(artifact_path),
                "gates": dict(recomputed.gates),
                "reasons": list(recomputed.reasons),
            }
        )
        return evidence
    except sqlite3.OperationalError as error:
        # ``gfm_task_acceptances`` was added after the first GFM registry
        # schema was deployed.  Preflight opens registries read-only by design,
        # so an otherwise valid legacy registry cannot be migrated here.  The
        # absent table is therefore equivalent to absent formal evidence; all
        # other operational failures remain visible and fail closed.
        if str(error).casefold() == "no such table: gfm_task_acceptances":
            return evidence
        evidence["reasons"] = [str(error)]
        return evidence
    except (
        ImportError,
        json.JSONDecodeError,
        sqlite3.DatabaseError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        evidence["reasons"] = [str(error)]
        return evidence


def _gfm_pretraining_acceptance_evidence(
    root: Path,
    *,
    checked_domain_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Recompute the newest sibling pretraining acceptance without writes."""

    database = root / "registry" / "gfm-registry.sqlite3"
    evidence: dict[str, Any] = {
        "ready": False,
        "registryPath": str(database),
        "experimentId": None,
        "reportHash": None,
        "accepted": False,
        "selectedVariant": None,
        "selectedCheckpointIds": [],
        "gates": {},
        "reasons": ["formal GFM pretraining acceptance evidence is absent"],
    }
    if set(checked_domain_hashes) != _GFM_REQUIRED_DOMAINS:
        evidence["reasons"] = ["GfmCorpusReady evidence is required"]
        return evidence
    if not database.is_file():
        return evidence
    try:
        from .gfm.contracts import GfmDomainCorpusManifest
        from .gfm.registry import GfmRegistry

        artifact_hashes: set[str] = set()
        manifest_directory = root / "datasets" / "manifests" / "gfm"
        for domain_id in sorted(_GFM_REQUIRED_DOMAINS):
            corpus = GfmDomainCorpusManifest.model_validate_json(
                (manifest_directory / f"{domain_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            if (
                corpus.domain_id != domain_id
                or corpus.corpus_id != domain_id
                or not corpus.public_checkpoint_eligible
                or corpus.content_hash != checked_domain_hashes[domain_id]
            ):
                raise ValueError(
                    f"pretraining corpus artifact differs from checked data: {domain_id}"
                )
            artifact_hashes.add(corpus.logical_hash)
        registry = GfmRegistry(database, initialize=False)
        acceptance = registry.latest_pretraining_acceptance()
        if acceptance is None:
            return evidence
        if set(acceptance.corpus_hashes) != artifact_hashes:
            raise ValueError(
                "pretraining acceptance corpora differ from checked corpus artifacts"
            )
        recomputed = registry.verify_pretraining_acceptance(acceptance)
        evidence.update(
            {
                "ready": bool(recomputed.accepted),
                "experimentId": recomputed.experiment_id,
                "reportHash": recomputed.report_hash,
                "accepted": recomputed.accepted,
                "configHash": recomputed.config_hash,
                "codeHash": recomputed.code_hash,
                "environmentHash": recomputed.environment_hash,
                "selectedVariant": recomputed.selected_variant,
                "selectedCheckpointIds": list(recomputed.selected_checkpoint_ids),
                "gates": dict(recomputed.gates),
                "reasons": list(recomputed.reasons),
            }
        )
        return evidence
    except (
        ImportError,
        json.JSONDecodeError,
        sqlite3.DatabaseError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        evidence["reasons"] = [str(error)]
        return evidence


def preflight_report(device: str = "cpu", root: str | Path | None = None) -> dict[str, Any]:
    selected_root = artifact_root(root)
    runtime = runtime_report(device)
    locks = verify_lock_manifest()
    compatibility = {
        name: check_compatibility(get_fixture(name)).model_dump(mode="json", by_alias=True)
        for name in ("actor", "hetero")
    }
    coverage = (
        _smoke_coverage(
            selected_root,
            device=device,
            environment_hash=runtime["environmentHash"],
        )
        if runtime["runtimeReady"]
        else set()
    )
    root_status = _root_report(selected_root)
    corpus_evidence, corpus_manifest_hash = _formal_corpus_evidence(selected_root)
    baseline_evidence = _baseline_evidence(selected_root, corpus_manifest_hash)
    gfm_corpus_evidence, _ = _gfm_corpus_evidence(selected_root)
    gfm_domain_hashes = dict(gfm_corpus_evidence["domainManifestHashes"])
    gfm_task_asset_evidence = _gfm_task_asset_evidence(
        selected_root,
        gfm_corpus_ready=bool(gfm_corpus_evidence["ready"]),
    )
    gfm_acceptance_evidence = _gfm_acceptance_evidence(
        selected_root, checked_domain_hashes=gfm_domain_hashes
    )
    gfm_pretraining_acceptance_evidence = _gfm_pretraining_acceptance_evidence(
        selected_root, checked_domain_hashes=gfm_domain_hashes
    )
    gfm_collaboration_task_acceptance_evidence = (
        _gfm_collaboration_task_acceptance_evidence(
            selected_root, checked_domain_hashes=gfm_domain_hashes
        )
    )
    acceptance_binding_fields = (
        "experimentId",
        "configHash",
        "codeHash",
        "environmentHash",
    )
    gfm_acceptance_binding_ready = all(
        gfm_acceptance_evidence.get(name) is not None
        and gfm_acceptance_evidence.get(name)
        == gfm_pretraining_acceptance_evidence.get(name)
        for name in acceptance_binding_fields
    )
    collaboration_binding_ready = (
        gfm_collaboration_task_acceptance_evidence["ready"]
        and gfm_pretraining_acceptance_evidence["ready"]
        and all(
            gfm_collaboration_task_acceptance_evidence.get(name)
            == gfm_pretraining_acceptance_evidence.get(name)
            for name in acceptance_binding_fields
        )
        and gfm_collaboration_task_acceptance_evidence.get("architectureVariant")
        == gfm_pretraining_acceptance_evidence.get("selectedVariant")
        and set(
            gfm_collaboration_task_acceptance_evidence.get(
                "backboneCheckpointIds", ()
            )
        )
        == set(
            gfm_pretraining_acceptance_evidence.get("selectedCheckpointIds", ())
        )
        and gfm_collaboration_task_acceptance_evidence.get(
            "pretrainingAcceptanceReportHash"
        )
        == gfm_pretraining_acceptance_evidence.get("reportHash")
    )
    infra_ready = (
        runtime["runtimeReady"]
        and locks["releaseLocksReady"]
        and all(item["compatible"] for item in compatibility.values())
        and coverage == {"actor", "hetero"}
        and root_status["anchorWritable"]
    )
    report = {
        "schemaVersion": "gfm.preflight/1.0",
        "readiness": {
            "WorkbenchInputReady": True,
            "GfmInfrastructureReady": infra_ready,
            "CorpusReady": corpus_evidence["ready"],
            "BaselineValidated": baseline_evidence["ready"],
            "GfmCorpusReady": gfm_corpus_evidence["ready"],
            "NewcomerOverlayReady": gfm_task_asset_evidence["newcomerOverlay"][
                "ready"
            ],
            "GfmPretrainingValidated": gfm_pretraining_acceptance_evidence[
                "ready"
            ],
            "GfmProductValidated": gfm_acceptance_evidence["productValidated"],
            "CollaborationProductValidated": (
                collaboration_binding_ready
            ),
            "ModelValidated": (
                gfm_corpus_evidence["ready"]
                and gfm_acceptance_evidence["ready"]
                and gfm_pretraining_acceptance_evidence["ready"]
                and gfm_acceptance_evidence["productValidated"]
                and gfm_acceptance_binding_ready
            ),
            "GfmServingReady": False,
            "LargeGraphUiReleaseReady": False,
        },
        "runtime": runtime,
        "gfmOptionalRuntime": gfm_optional_runtime_report(),
        "runtimeLocks": locks,
        "artifactRoot": root_status,
        "storage": {
            "fetch": _storage_evidence(selected_root, "fetch"),
            "run": _storage_evidence(selected_root, "run"),
        },
        "formalCorpus": corpus_evidence,
        "baselineAcceptance": baseline_evidence,
        "gfmCorpora": gfm_corpus_evidence,
        "gfmTaskAssets": gfm_task_asset_evidence,
        "gfmAcceptance": gfm_acceptance_evidence,
        "gfmPretrainingAcceptance": gfm_pretraining_acceptance_evidence,
        "gfmCollaborationTaskAcceptance": (
            {
                **gfm_collaboration_task_acceptance_evidence,
                "pretrainingBindingReady": collaboration_binding_ready,
            }
        ),
        "gfmAcceptanceBinding": {
            "ready": gfm_acceptance_binding_ready,
            "fields": list(acceptance_binding_fields),
        },
        "canonicalProfiles": compatibility,
        "smokeCoverage": sorted(coverage),
        "models": [],
    }
    report["manifestHash"] = canonical_sha256(report)
    return report
