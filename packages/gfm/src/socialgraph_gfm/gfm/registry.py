"""Independent WAL registry and fail-closed promotion boundary for formal GFM models."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..canonical import canonical_json, canonical_sha256, file_sha256
from ..errors import RegistrationRejected
from .acceptance import build_gfm_acceptance
from .checkpoint import load_gfm_checkpoint
from .contracts import (
    GfmAcceptanceManifest,
    GfmCheckpointManifest,
    GfmDomainCorpusManifest,
    GfmEvaluationReport,
    GfmPretrainingAcceptanceManifest,
    GfmRunManifest,
    GfmTaskAcceptanceManifest,
    GfmTaskProtocolManifest,
)
from .pretraining_acceptance import build_gfm_pretraining_acceptance
from .task_acceptance import (
    COLLABORATION_PROTOCOL_HASH,
    COLLABORATION_PROTOCOL_ID,
    COLLABORATION_TASK,
    build_collaboration_task_acceptance,
    collaboration_protocol,
)

GFM_SCHEMA = """
PRAGMA foreign_keys=ON;
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS gfm_domain_corpora (
    corpus_id TEXT PRIMARY KEY,
    logical_hash TEXT NOT NULL UNIQUE,
    domain_id TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gfm_task_protocols (
    protocol_id TEXT PRIMARY KEY,
    protocol_hash TEXT NOT NULL UNIQUE,
    task_id TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gfm_runs (
    run_id TEXT PRIMARY KEY,
    manifest_hash TEXT NOT NULL UNIQUE,
    experiment_id TEXT NOT NULL,
    status TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gfm_runs_experiment ON gfm_runs(experiment_id);
CREATE TABLE IF NOT EXISTS gfm_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    logical_hash TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES gfm_runs(run_id),
    artifact_sha256 TEXT NOT NULL,
    registrable INTEGER NOT NULL CHECK(registrable=0),
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gfm_evaluations (
    report_id TEXT PRIMARY KEY,
    report_hash TEXT NOT NULL UNIQUE,
    experiment_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES gfm_runs(run_id),
    checkpoint_id TEXT NOT NULL REFERENCES gfm_checkpoints(checkpoint_id),
    evaluation_kind TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gfm_eval_experiment ON gfm_evaluations(experiment_id);
CREATE TABLE IF NOT EXISTS gfm_acceptances (
    report_hash TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL REFERENCES gfm_checkpoints(checkpoint_id),
    accepted INTEGER NOT NULL CHECK(accepted IN (0, 1)),
    created_at TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gfm_acceptance_experiment
    ON gfm_acceptances(experiment_id, accepted);
CREATE TABLE IF NOT EXISTS gfm_pretraining_acceptances (
    report_hash TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    accepted INTEGER NOT NULL CHECK(accepted IN (0, 1)),
    created_at TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gfm_pretraining_acceptance_experiment
    ON gfm_pretraining_acceptances(experiment_id, accepted);
CREATE TABLE IF NOT EXISTS gfm_task_acceptances (
    report_hash TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    accepted INTEGER NOT NULL CHECK(accepted IN (0, 1)),
    created_at TEXT NOT NULL,
    registrable INTEGER NOT NULL CHECK(registrable=0),
    manifest_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gfm_task_acceptance_experiment
    ON gfm_task_acceptances(experiment_id, task_id, accepted);
CREATE TABLE IF NOT EXISTS gfm_models (
    model_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL REFERENCES gfm_checkpoints(checkpoint_id),
    acceptance_report_hash TEXT NOT NULL REFERENCES gfm_acceptances(report_hash),
    promoted_at TEXT NOT NULL,
    promotion_json TEXT NOT NULL
);
"""


class GfmRegistry:
    """Formal GFM-only registry; baseline and serving registries are never consulted."""

    def __init__(self, path: str | Path, *, initialize: bool = True) -> None:
        self.path = Path(path)
        self._read_only = not initialize
        if initialize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as connection:
                connection.executescript(GFM_SCHEMA)
        elif not self.path.is_file():
            raise FileNotFoundError(self.path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = (
            sqlite3.connect(f"file:{self.path.resolve().as_posix()}?mode=ro", uri=True)
            if self._read_only
            else sqlite3.connect(self.path)
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        if self._read_only:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
        else:
            connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            if not self._read_only:
                connection.commit()
        except BaseException:
            if not self._read_only:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _insert_immutable(
        connection: sqlite3.Connection,
        *,
        table: str,
        identity_column: str,
        identity: str,
        columns: tuple[str, ...],
        values: tuple[Any, ...],
        manifest_json: str,
    ) -> None:
        existing = connection.execute(
            f"SELECT manifest_json FROM {table} WHERE {identity_column}=?", (identity,)
        ).fetchone()
        if existing is not None:
            if existing["manifest_json"] == manifest_json:
                return
            raise RegistrationRejected(
                f"Immutable {table} identity {identity!r} already has different content"
            )
        placeholders = ",".join("?" for _ in values)
        try:
            connection.execute(
                f"INSERT INTO {table} ({','.join(columns)},manifest_json) "
                f"VALUES ({placeholders},?)",
                (*values, manifest_json),
            )
        except sqlite3.IntegrityError as error:
            raise RegistrationRejected(f"Cannot record {table}: {error}") from error

    def record_corpus(self, manifest: GfmDomainCorpusManifest) -> None:
        checked = GfmDomainCorpusManifest.model_validate(manifest)
        raw = canonical_json(checked)
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT manifest_json FROM gfm_domain_corpora WHERE corpus_id=?",
                (checked.corpus_id,),
            ).fetchone()
            if existing is not None:
                registered = GfmDomainCorpusManifest.model_validate_json(existing["manifest_json"])
                # createdAt and artifactPath are physical observations, not the
                # portable corpus identity.  A deterministic re-prepare is an
                # idempotent integrity check when its complete logical payload
                # and hash still match; retain the first immutable registry row.
                if (
                    registered.logical_hash == checked.logical_hash
                    and registered.logical_payload() == checked.logical_payload()
                ):
                    return
                raise RegistrationRejected(
                    f"Immutable gfm_domain_corpora identity {checked.corpus_id!r} "
                    "already has different logical content"
                )
            self._insert_immutable(
                connection,
                table="gfm_domain_corpora",
                identity_column="corpus_id",
                identity=checked.corpus_id,
                columns=("corpus_id", "logical_hash", "domain_id"),
                values=(checked.corpus_id, checked.logical_hash, checked.domain_id),
                manifest_json=raw,
            )

    def record_protocol(self, manifest: GfmTaskProtocolManifest) -> None:
        checked = GfmTaskProtocolManifest.model_validate(manifest)
        if checked.task_id == COLLABORATION_TASK:
            expected = collaboration_protocol()
            if (
                checked.protocol_id != COLLABORATION_PROTOCOL_ID
                or checked.protocol_hash != COLLABORATION_PROTOCOL_HASH
                or checked.logical_payload() != expected.logical_payload()
            ):
                raise RegistrationRejected(
                    "Collaboration task protocol differs from the fixed v1 contract"
                )
        raw = canonical_json(checked)
        with self.connect() as connection:
            if checked.task_id == COLLABORATION_TASK:
                conflicting = connection.execute(
                    """
                    SELECT protocol_id, protocol_hash FROM gfm_task_protocols
                    WHERE task_id=? AND protocol_id<>?
                    """,
                    (COLLABORATION_TASK, COLLABORATION_PROTOCOL_ID),
                ).fetchall()
                if conflicting:
                    raise RegistrationRejected(
                        "Collaboration task protocol registry is ambiguous"
                    )
            self._insert_immutable(
                connection,
                table="gfm_task_protocols",
                identity_column="protocol_id",
                identity=checked.protocol_id,
                columns=("protocol_id", "protocol_hash", "task_id"),
                values=(checked.protocol_id, checked.protocol_hash, checked.task_id),
                manifest_json=raw,
            )

    def _record_run(self, connection: sqlite3.Connection, checked: GfmRunManifest) -> None:
        corpus_hashes = {
            row["logical_hash"]
            for row in connection.execute("SELECT logical_hash FROM gfm_domain_corpora").fetchall()
        }
        protocol_hashes = {
            row["protocol_hash"]
            for row in connection.execute("SELECT protocol_hash FROM gfm_task_protocols").fetchall()
        }
        missing_corpora = set(checked.corpus_hashes) - corpus_hashes
        missing_protocols = set(checked.task_protocol_hashes) - protocol_hashes
        if missing_corpora or missing_protocols:
            raise RegistrationRejected(
                "GFM run references unregistered corpus or task protocol hashes"
            )
        self._insert_immutable(
            connection,
            table="gfm_runs",
            identity_column="run_id",
            identity=checked.run_id,
            columns=("run_id", "manifest_hash", "experiment_id", "status", "config_hash"),
            values=(
                checked.run_id,
                checked.manifest_hash,
                checked.experiment_id,
                checked.status,
                checked.config_hash,
            ),
            manifest_json=canonical_json(checked),
        )

    def _record_checkpoint(
        self, connection: sqlite3.Connection, checked: GfmCheckpointManifest
    ) -> None:
        run_row = connection.execute(
            "SELECT manifest_json FROM gfm_runs WHERE run_id=?", (checked.run_id,)
        ).fetchone()
        if run_row is None:
            raise RegistrationRejected("GFM checkpoint references an unregistered run")
        run = GfmRunManifest.model_validate_json(run_row["manifest_json"])
        if run.config_hash != checked.config_hash or set(run.corpus_hashes) != set(
            checked.corpus_hashes
        ):
            raise RegistrationRejected("GFM checkpoint provenance differs from its run")
        self._insert_immutable(
            connection,
            table="gfm_checkpoints",
            identity_column="checkpoint_id",
            identity=checked.checkpoint_id,
            columns=(
                "checkpoint_id",
                "logical_hash",
                "run_id",
                "artifact_sha256",
                "registrable",
            ),
            values=(
                checked.checkpoint_id,
                checked.logical_hash,
                checked.run_id,
                checked.artifact_sha256,
                0,
            ),
            manifest_json=canonical_json(checked),
        )

    def record_run(self, manifest: GfmRunManifest) -> None:
        checked = GfmRunManifest.model_validate(manifest)
        with self.connect() as connection:
            self._record_run(connection, checked)

    def record_checkpoint(self, manifest: GfmCheckpointManifest) -> None:
        checked = GfmCheckpointManifest.model_validate(manifest)
        with self.connect() as connection:
            self._record_checkpoint(connection, checked)

    def record_completed_run(
        self,
        run_manifest: GfmRunManifest,
        checkpoint_manifest: GfmCheckpointManifest,
    ) -> None:
        """Atomically publish one succeeded run and its selected checkpoint.

        A filesystem run manifest is written before this registry transaction and
        its terminal run-state marker is written afterwards.  Keeping the two
        registry rows in one transaction ensures that a crash can leave either no
        registry authority (and therefore a resumable run) or a complete authority
        pair that can be reconciled safely; it cannot strand only the run row.
        """

        checked_run = GfmRunManifest.model_validate(run_manifest)
        checked_checkpoint = GfmCheckpointManifest.model_validate(checkpoint_manifest)
        if checked_run.status != "succeeded":
            raise RegistrationRejected("Atomic completed-run publication requires a succeeded run")
        if checked_checkpoint.run_id != checked_run.run_id:
            raise RegistrationRejected(
                "Atomic completed-run publication has mismatched run/checkpoint identities"
            )
        with self.connect() as connection:
            existing_checkpoint_ids = {
                str(row["checkpoint_id"])
                for row in connection.execute(
                    "SELECT checkpoint_id FROM gfm_checkpoints WHERE run_id=?",
                    (checked_run.run_id,),
                ).fetchall()
            }
            if existing_checkpoint_ids.difference({checked_checkpoint.checkpoint_id}):
                raise RegistrationRejected(
                    "Completed run already selects a different registered checkpoint"
                )
            self._record_run(connection, checked_run)
            self._record_checkpoint(connection, checked_checkpoint)

    def record_evaluation(self, report: GfmEvaluationReport) -> None:
        checked = GfmEvaluationReport.model_validate(report)
        self._verify_evaluation_artifacts(checked)
        raw = canonical_json(checked)
        with self.connect() as connection:
            run_row = connection.execute(
                "SELECT manifest_json FROM gfm_runs WHERE run_id=?", (checked.run_id,)
            ).fetchone()
            checkpoint_row = connection.execute(
                "SELECT run_id FROM gfm_checkpoints WHERE checkpoint_id=?",
                (checked.checkpoint_id,),
            ).fetchone()
            if run_row is None or checkpoint_row is None:
                raise RegistrationRejected("GFM evaluation references unknown run/checkpoint")
            run = GfmRunManifest.model_validate_json(run_row["manifest_json"])
            if (
                run.experiment_id != checked.experiment_id
                or checkpoint_row["run_id"] != checked.run_id
                or run.seed != checked.seed
                or run.status != "succeeded"
            ):
                raise RegistrationRejected("GFM evaluation identity differs from run/checkpoint")
            protocol_rows = connection.execute(
                "SELECT protocol_hash, task_id FROM gfm_task_protocols"
            ).fetchall()
            protocol_by_task = {
                str(row["task_id"]): str(row["protocol_hash"]) for row in protocol_rows
            }
            if checked.evaluation_kind == "lodo":
                if (
                    run.phase != "lodo"
                    or checked.held_out_domain != run.held_out_domain
                    or checked.domain_id != run.held_out_domain
                ):
                    raise RegistrationRejected("LODO report differs from its isolated run")
            elif checked.evaluation_kind in {"product", "calibration"}:
                expected_protocol = protocol_by_task.get(str(checked.task_id))
                if (
                    run.phase not in {"adapt", "evaluate"}
                    or expected_protocol is None
                    or expected_protocol not in run.task_protocol_hashes
                    or checked.evaluator_code_hash != run.code_hash
                    or checked.evaluator_environment_hash != run.environment_hash
                ):
                    raise RegistrationRejected(
                        "Product report is not bound by its run protocol/evaluator provenance"
                    )
            elif checked.evaluation_kind == "in_domain" and run.phase != "pretrain":
                raise RegistrationRejected("In-domain report requires a pretraining run")
            elif checked.evaluation_kind == "fresh_process" and run.phase not in {
                "pretrain",
                "adapt",
                "evaluate",
            }:
                raise RegistrationRejected(
                    "Fresh-process evidence requires a pretrain, adapt or evaluation run"
                )
            self._insert_immutable(
                connection,
                table="gfm_evaluations",
                identity_column="report_id",
                identity=checked.report_id,
                columns=(
                    "report_id",
                    "report_hash",
                    "experiment_id",
                    "run_id",
                    "checkpoint_id",
                    "evaluation_kind",
                ),
                values=(
                    checked.report_id,
                    checked.report_hash,
                    checked.experiment_id,
                    checked.run_id,
                    checked.checkpoint_id,
                    checked.evaluation_kind,
                ),
                manifest_json=raw,
            )

    def _verify_evaluation_artifacts(self, report: GfmEvaluationReport) -> None:
        allowed = (self.path.parent.parent / "reports" / "gfm").resolve()
        for path_text, expected, name in (
            (
                report.evidence_artifact_path,
                report.evidence_artifact_hash,
                "evaluation evidence",
            ),
            (
                report.leakage_audit_path,
                report.leakage_audit_hash,
                "leakage audit",
            ),
        ):
            path = Path(path_text).resolve()
            if not path.is_relative_to(allowed) or not path.is_file():
                raise RegistrationRejected(f"{name} path is outside the runtime report root")
            if file_sha256(path) != expected:
                raise RegistrationRejected(f"{name} file hash differs from its report")
        try:
            evidence = json.loads(Path(report.evidence_artifact_path).read_text(encoding="utf-8"))
            audit = json.loads(Path(report.leakage_audit_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RegistrationRejected(
                "Evaluation evidence or leakage audit is not canonical JSON"
            ) from error
        for artifact, name in ((evidence, "evaluation evidence"), (audit, "leakage audit")):
            if not isinstance(artifact, dict):
                raise RegistrationRejected(f"{name} is not a JSON object")
            logical_hash = artifact.get("logicalHash")
            logical_payload = {
                key: value for key, value in artifact.items() if key != "logicalHash"
            }
            if logical_hash != canonical_sha256(logical_payload):
                raise RegistrationRejected(f"{name} logical hash is invalid")
        payload = evidence.get("payload")
        if not isinstance(payload, dict):
            raise RegistrationRejected("Evaluation evidence payload is missing")
        if report.evaluation_kind in {"product", "calibration"}:
            audit_evidence = audit.get("evidence")
            if (
                evidence.get("experimentId") != report.experiment_id
                or evidence.get("evidenceId") != report.report_id
                or payload.get("checkpointId") != report.checkpoint_id
                or payload.get("evaluatorCodeHash") != report.evaluator_code_hash
                or payload.get("evaluatorEnvironmentHash")
                != report.evaluator_environment_hash
                or not isinstance(audit_evidence, dict)
                or audit.get("experimentId") != report.experiment_id
                or audit_evidence.get("checkpointId") != report.checkpoint_id
                or audit_evidence.get("evaluatorCodeHash")
                != report.evaluator_code_hash
                or audit_evidence.get("evaluatorEnvironmentHash")
                != report.evaluator_environment_hash
            ):
                raise RegistrationRejected(
                    "Product evaluation artifacts are not bound to their exact "
                    "checkpoint/evaluator provenance"
                )
        evidence_metrics = payload.get("metrics")
        if not isinstance(evidence_metrics, dict) or canonical_sha256(
            evidence_metrics
        ) != canonical_sha256(dict(report.metrics)):
            raise RegistrationRejected(
                "Reported metrics are not exactly hash-bound to evaluation evidence"
            )
        if report.ece is not None and evidence_metrics.get("ece") != report.ece:
            raise RegistrationRejected("Reported ECE differs from evaluation evidence")
        if report.brier is not None:
            evidence_brier = payload.get("brier", evidence_metrics.get("brier"))
            if evidence_brier != report.brier:
                raise RegistrationRejected("Reported Brier score differs from evaluation evidence")
        if report.evaluation_kind == "product":
            definition = payload.get("baselineDefinition")
            if (
                not isinstance(definition, dict)
                or canonical_sha256(definition) != report.baseline_definition_hash
            ):
                raise RegistrationRejected(
                    "Product baseline definition is not hash-bound to its evidence"
                )
        if report.evaluation_kind == "calibration":
            definition = payload.get("strataDefinition")
            if (
                not isinstance(definition, dict)
                or canonical_sha256(definition) != report.strata_definition_hash
            ):
                raise RegistrationRejected(
                    "Calibration strata definition is not hash-bound to its evidence"
                )
        audit_counters = audit.get("counters")
        required_counters = {
            "future_edge_access_count",
            "cutoff_violation_count",
            "split_overlap_count",
        }
        if report.evaluation_kind == "lodo":
            required_counters.add("target_domain_pretrain_access_count")
        if not isinstance(audit_counters, dict) or not required_counters.issubset(audit_counters):
            raise RegistrationRejected("Leakage audit counters are incomplete")
        for name in required_counters:
            value = audit_counters[name]
            if value != 0 and value != 0.0:
                raise RegistrationRejected("Leakage audit contains a nonzero violation")
            if report.metrics.get(name) != float(value):
                raise RegistrationRejected("Leakage audit counters differ from reported metrics")

    def _verify_pretraining_evaluation_provenance(
        self,
        report: GfmEvaluationReport,
        *,
        run: GfmRunManifest,
        corpus_by_domain: dict[str, GfmDomainCorpusManifest],
    ) -> None:
        """Verify evidence details that are specific to the formal matrix."""

        try:
            evidence = json.loads(Path(report.evidence_artifact_path).read_text(encoding="utf-8"))
            audit = json.loads(Path(report.leakage_audit_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RegistrationRejected("Pretraining evaluation artifacts are unreadable") from error
        payload = evidence.get("payload") if isinstance(evidence, dict) else None
        audit_evidence = audit.get("evidence") if isinstance(audit, dict) else None
        if (
            evidence.get("experimentId") != report.experiment_id
            or audit.get("experimentId") != report.experiment_id
            or not isinstance(payload, dict)
            or not isinstance(audit_evidence, dict)
        ):
            raise RegistrationRejected("Pretraining evidence is not bound to its experiment")
        if report.evaluation_kind == "in_domain":
            if (
                run.phase != "pretrain"
                or report.domain_id != "multi-domain"
                or payload.get("testReadCount") != 1
                or "physical-test-view-read-once-after-best" not in report.warnings
            ):
                raise RegistrationRejected(
                    "Formal in-domain evidence lacks one-shot test provenance"
                )
            return
        if report.evaluation_kind == "fresh_process":
            if (
                run.phase != "pretrain"
                or report.domain_id != "multi-domain"
                or payload.get("verificationDigest") != report.verification_digest
                or payload.get("repeatVerificationDigest") != report.verification_digest
            ):
                raise RegistrationRejected(
                    "Formal fresh-process evidence lacks repeated recovery provenance"
                )
            return
        if report.evaluation_kind != "lodo" or run.phase != "lodo":
            raise RegistrationRejected(
                "Pretraining acceptance contains an unsupported evaluation kind"
            )
        held_out = str(report.held_out_domain)
        target = corpus_by_domain.get(held_out)
        expected_sources = {
            corpus.logical_hash for domain, corpus in corpus_by_domain.items() if domain != held_out
        }
        if (
            target is None
            or set(audit_evidence.get("sourceCorpusHashes", ())) != expected_sources
            or audit_evidence.get("targetCorpusHash") != target.logical_hash
            or set(audit_evidence.get("pretrainingLoadedDomainIds", ())) != set(run.domain_ids)
            or audit_evidence.get("targetLoadedAfterSourcePretraining") != held_out
            or not isinstance(payload.get("isolationHash"), str)
            or payload.get("isolationHash") != audit_evidence.get("isolationHash")
        ):
            raise RegistrationRejected(
                "LODO evidence does not prove target-domain pretraining isolation"
            )

    def _recompute_pretraining_acceptance(
        self, *, experiment_id: str
    ) -> GfmPretrainingAcceptanceManifest:
        """Derive pretraining acceptance from registered, physical evidence only."""

        runs = tuple(
            run
            for run in self.list_runs(experiment_id=experiment_id)
            if run.phase in {"pretrain", "lodo"}
        )
        if not runs or not any(run.phase == "pretrain" for run in runs):
            raise RegistrationRejected(
                "Pretraining acceptance requires registered formal pretrain runs"
            )
        relevant_run_ids = {run.run_id for run in runs}
        checkpoints = tuple(
            checkpoint
            for checkpoint in self.list_checkpoints(experiment_id=experiment_id)
            if checkpoint.run_id in relevant_run_ids
        )
        evaluations = tuple(
            report
            for report in self.list_evaluations(experiment_id=experiment_id)
            if report.run_id in relevant_run_ids
            and (
                (
                    report.evaluation_kind in {"in_domain", "fresh_process"}
                    and next(run for run in runs if run.run_id == report.run_id).phase == "pretrain"
                )
                or (
                    report.evaluation_kind == "lodo"
                    and next(run for run in runs if run.run_id == report.run_id).phase == "lodo"
                )
            )
        )
        run_by_id = {run.run_id: run for run in runs}
        checkpoint_by_id = {checkpoint.checkpoint_id: checkpoint for checkpoint in checkpoints}
        corpus_hashes = {corpus_hash for run in runs for corpus_hash in run.corpus_hashes}
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT manifest_json FROM gfm_domain_corpora ORDER BY domain_id, corpus_id"
            ).fetchall()
        corpora = tuple(
            corpus
            for corpus in (
                GfmDomainCorpusManifest.model_validate_json(row["manifest_json"]) for row in rows
            )
            if corpus.logical_hash in corpus_hashes
        )
        corpus_by_domain = {corpus.domain_id: corpus for corpus in corpora}
        for checkpoint in checkpoints:
            run = run_by_id.get(checkpoint.run_id)
            if run is None or checkpoint.config_hash != run.config_hash:
                raise RegistrationRejected("Pretraining checkpoint/run provenance is inconsistent")
            try:
                payload = load_gfm_checkpoint(checkpoint, map_location="cpu")
            except Exception as error:
                raise RegistrationRejected(
                    f"Pretraining checkpoint failed integrity verification: {error}"
                ) from error
            del payload
        for report in evaluations:
            run = run_by_id.get(report.run_id)
            report_checkpoint = checkpoint_by_id.get(report.checkpoint_id)
            if (
                run is None
                or report_checkpoint is None
                or report_checkpoint.run_id != run.run_id
                or report.experiment_id != experiment_id
                or report.seed != run.seed
            ):
                raise RegistrationRejected(
                    "Pretraining evaluation run/checkpoint provenance is inconsistent"
                )
            self._verify_evaluation_artifacts(report)
            self._verify_pretraining_evaluation_provenance(
                report, run=run, corpus_by_domain=corpus_by_domain
            )
        config_hashes = sorted({run.config_hash for run in runs})
        code_hashes = sorted({run.code_hash for run in runs})
        environment_hashes = sorted({run.environment_hash for run in runs})
        return build_gfm_pretraining_acceptance(
            experiment_id=experiment_id,
            corpora=corpora,
            runs=runs,
            checkpoints=checkpoints,
            evaluations=evaluations,
            config_hash=config_hashes[0],
            code_hash=code_hashes[0],
            environment_hash=environment_hashes[0],
        )

    def build_pretraining_acceptance(
        self, *, experiment_id: str
    ) -> GfmPretrainingAcceptanceManifest:
        return self._recompute_pretraining_acceptance(experiment_id=experiment_id)

    def verify_pretraining_acceptance(
        self, manifest: GfmPretrainingAcceptanceManifest
    ) -> GfmPretrainingAcceptanceManifest:
        """Recompute a stored sibling acceptance, including all physical bytes."""

        checked = GfmPretrainingAcceptanceManifest.model_validate(manifest)
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT report_hash, experiment_id, accepted, manifest_json
                FROM gfm_pretraining_acceptances WHERE report_hash=?
                """,
                (checked.report_hash,),
            ).fetchone()
        if row is not None and (
            row["report_hash"] != checked.report_hash
            or row["experiment_id"] != checked.experiment_id
            or bool(row["accepted"]) != checked.accepted
            or row["manifest_json"] != canonical_json(checked)
        ):
            raise RegistrationRejected(
                "Pretraining acceptance registry columns differ from its contract"
            )
        recomputed = self._recompute_pretraining_acceptance(experiment_id=checked.experiment_id)
        if (
            recomputed.logical_payload() != checked.logical_payload()
            or recomputed.report_hash != checked.report_hash
        ):
            raise RegistrationRejected(
                "Stored pretraining acceptance differs from registry-derived evidence"
            )
        return recomputed

    def record_pretraining_acceptance(self, manifest: GfmPretrainingAcceptanceManifest) -> None:
        checked = GfmPretrainingAcceptanceManifest.model_validate(manifest)
        self.verify_pretraining_acceptance(checked)
        raw = canonical_json(checked)
        with self.connect() as connection:
            known_checkpoints = {
                row["checkpoint_id"]
                for row in connection.execute(
                    "SELECT checkpoint_id FROM gfm_checkpoints"
                ).fetchall()
            }
            known_reports = {
                row["report_hash"]
                for row in connection.execute(
                    "SELECT report_hash FROM gfm_evaluations WHERE experiment_id=?",
                    (checked.experiment_id,),
                ).fetchall()
            }
            if not set(checked.evidence_checkpoint_ids).issubset(known_checkpoints) or not set(
                checked.evaluation_report_hashes
            ).issubset(known_reports):
                raise RegistrationRejected(
                    "Pretraining acceptance references unregistered evidence"
                )
            self._insert_immutable(
                connection,
                table="gfm_pretraining_acceptances",
                identity_column="report_hash",
                identity=checked.report_hash,
                columns=(
                    "report_hash",
                    "experiment_id",
                    "accepted",
                    "created_at",
                ),
                values=(
                    checked.report_hash,
                    checked.experiment_id,
                    int(checked.accepted),
                    checked.created_at.isoformat(),
                ),
                manifest_json=raw,
            )

    def collaboration_backbone_bindings(
        self,
        *,
        experiment_id: str,
        product_checkpoint_ids: tuple[str, ...],
    ) -> tuple[GfmPretrainingAcceptanceManifest, dict[str, dict[str, Any]]]:
        """Verify every product checkpoint against an accepted formal backbone."""

        if len(product_checkpoint_ids) != 3 or len(set(product_checkpoint_ids)) != 3:
            raise RegistrationRejected(
                "Collaboration backbone binding requires three unique product checkpoints"
            )
        pretraining = self.latest_pretraining_acceptance(
            experiment_id=experiment_id
        )
        if pretraining is None or not pretraining.accepted:
            raise RegistrationRejected(
                "Collaboration task acceptance requires accepted pretraining evidence"
            )
        pretraining = self.verify_pretraining_acceptance(pretraining)
        selected_backbones = set(pretraining.selected_checkpoint_ids)
        bindings: dict[str, dict[str, Any]] = {}
        for product_checkpoint_id in product_checkpoint_ids:
            product_checkpoint = self.get_checkpoint(product_checkpoint_id)
            if product_checkpoint is None:
                raise RegistrationRejected(
                    "Collaboration product checkpoint is absent"
                )
            product_run = self.get_run(product_checkpoint.run_id)
            if (
                product_run is None
                or product_run.experiment_id != experiment_id
                or product_run.phase != "adapt"
                or product_run.status != "succeeded"
            ):
                raise RegistrationRejected(
                    "Collaboration product checkpoint lacks a succeeded adapt run"
                )
            try:
                product_payload = load_gfm_checkpoint(
                    product_checkpoint, map_location="cpu"
                )
            except Exception as error:
                raise RegistrationRejected(
                    f"Collaboration product checkpoint failed integrity verification: {error}"
                ) from error
            components = product_payload.get("components")
            best_state = product_payload.get("best_state")
            if (
                not isinstance(components, dict)
                or set(components) != {"product", "product_config"}
                or not isinstance(best_state, dict)
                or not isinstance(components.get("product_config"), dict)
            ):
                raise RegistrationRejected(
                    "Collaboration product checkpoint has an invalid component boundary"
                )
            product_config = dict(components["product_config"])
            checked_config = dict(product_config)
            product_config_hash = checked_config.pop("taskConfigHash", None)
            if (
                product_config.get("task") != "collaboration"
                or product_config_hash != canonical_sha256(checked_config)
                or best_state.get("task") != "collaboration"
                or best_state.get("productConfigHash") != product_config_hash
                or product_config.get("seed") != product_run.seed
                or product_config.get("architectureVariant")
                != product_run.architecture_variant
            ):
                raise RegistrationRejected(
                    "Collaboration product checkpoint configuration is invalid"
                )
            backbone_id = product_config.get("backboneCheckpointId")
            backbone_state_hash = product_config.get("backboneStateHash")
            backbone = (
                self.get_checkpoint(backbone_id)
                if isinstance(backbone_id, str)
                else None
            )
            if (
                backbone is None
                or not isinstance(backbone_state_hash, str)
                or backbone.state_hash != backbone_state_hash
                or backbone.checkpoint_id not in selected_backbones
            ):
                raise RegistrationRejected(
                    "Collaboration product checkpoint is not bound to an accepted backbone"
                )
            backbone_run = self.get_run(backbone.run_id)
            if (
                backbone_run is None
                or backbone_run.experiment_id != experiment_id
                or backbone_run.phase != "pretrain"
                or backbone_run.status != "succeeded"
                or backbone_run.seed != product_run.seed
                or backbone_run.architecture_variant
                != pretraining.selected_variant
                or product_run.architecture_variant
                != pretraining.selected_variant
                or backbone_run.config_hash != product_run.config_hash
                or backbone_run.code_hash != product_run.code_hash
                or backbone_run.environment_hash != product_run.environment_hash
                or set(backbone_run.corpus_hashes) != set(product_run.corpus_hashes)
                or backbone.config_hash != product_run.config_hash
                or set(backbone.corpus_hashes) != set(product_run.corpus_hashes)
            ):
                raise RegistrationRejected(
                    "Collaboration backbone and product run provenance differ"
                )
            try:
                backbone_payload = load_gfm_checkpoint(backbone, map_location="cpu")
            except Exception as error:
                raise RegistrationRejected(
                    f"Collaboration backbone failed integrity verification: {error}"
                ) from error
            del backbone_payload
            bindings[product_checkpoint_id] = {
                "checkpointId": backbone.checkpoint_id,
                "stateHash": backbone.state_hash,
                "seed": backbone_run.seed,
                "architectureVariant": backbone_run.architecture_variant,
                "configHash": backbone_run.config_hash,
                "codeHash": backbone_run.code_hash,
                "environmentHash": backbone_run.environment_hash,
                "corpusHashes": tuple(backbone_run.corpus_hashes),
            }
        if {
            value["checkpointId"] for value in bindings.values()
        } != selected_backbones:
            raise RegistrationRejected(
                "Collaboration product matrix does not use every accepted backbone exactly once"
            )
        return pretraining, bindings

    def _recompute_collaboration_task_acceptance(
        self, *, experiment_id: str
    ) -> GfmTaskAcceptanceManifest:
        """Re-read only collaboration evidence and its three one-shot test states."""

        evaluations = self.list_evaluations(experiment_id=experiment_id)
        product_reports = tuple(
            report
            for report in evaluations
            if report.evaluation_kind == "product"
            and report.task_id == COLLABORATION_TASK
            and "shadow" not in report.warnings
        )
        if len(product_reports) != 3:
            raise RegistrationRejected(
                "Collaboration task acceptance requires exactly three formal product reports"
            )
        run_ids = {report.run_id for report in product_reports}
        checkpoint_ids = {report.checkpoint_id for report in product_reports}
        runs = tuple(
            run
            for run in self.list_runs(experiment_id=experiment_id)
            if run.run_id in run_ids
        )
        checkpoints = tuple(
            checkpoint
            for checkpoint in self.list_checkpoints(experiment_id=experiment_id)
            if checkpoint.checkpoint_id in checkpoint_ids
        )
        relevant_reports = tuple(
            report
            for report in evaluations
            if report.run_id in run_ids
            and report.checkpoint_id in checkpoint_ids
            and (
                report.evaluation_kind == "fresh_process"
                or (
                    report.task_id == COLLABORATION_TASK
                    and report.evaluation_kind in {"product", "calibration"}
                    and "shadow" not in report.warnings
                )
            )
        )
        for checkpoint in checkpoints:
            try:
                payload = load_gfm_checkpoint(checkpoint, map_location="cpu")
            except Exception as error:
                raise RegistrationRejected(
                    f"Collaboration checkpoint failed integrity verification: {error}"
                ) from error
            del payload
        for report in relevant_reports:
            self._verify_evaluation_artifacts(report)
        pretraining, backbone_bindings = self.collaboration_backbone_bindings(
            experiment_id=experiment_id,
            product_checkpoint_ids=tuple(sorted(checkpoint_ids)),
        )
        with self.connect() as connection:
            protocol_rows = connection.execute(
                """
                SELECT protocol_id, protocol_hash, task_id, manifest_json
                FROM gfm_task_protocols WHERE task_id=?
                """,
                (COLLABORATION_TASK,),
            ).fetchall()
        if len(protocol_rows) != 1:
            raise RegistrationRejected(
                "Collaboration task protocol is absent or ambiguous"
            )
        protocol_row = protocol_rows[0]
        protocol = GfmTaskProtocolManifest.model_validate_json(
            protocol_row["manifest_json"]
        )
        expected_protocol = collaboration_protocol()
        if (
            protocol_row["protocol_id"] != protocol.protocol_id
            or protocol_row["protocol_hash"] != protocol.protocol_hash
            or protocol_row["task_id"] != protocol.task_id
            or protocol.protocol_id != COLLABORATION_PROTOCOL_ID
            or protocol.protocol_hash != COLLABORATION_PROTOCOL_HASH
            or protocol.logical_payload() != expected_protocol.logical_payload()
        ):
            raise RegistrationRejected(
                "Collaboration task protocol differs from the fixed v1 contract"
            )
        report_root = (self.path.parent.parent / "reports" / "gfm").resolve()
        test_read_directory = (report_root / experiment_id / "test-read").resolve()
        if not test_read_directory.is_relative_to(report_root):
            raise RegistrationRejected("Collaboration test-read directory escaped reports root")
        states: dict[str, dict[str, Any]] = {}
        for checkpoint_id in sorted(checkpoint_ids):
            path = (test_read_directory / f"{checkpoint_id}-test.json").resolve()
            if not path.is_relative_to(test_read_directory) or not path.is_file():
                raise RegistrationRejected(
                    "Collaboration one-shot physical test-read evidence is absent"
                )
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise RegistrationRejected(
                    "Collaboration test-read evidence is unreadable"
                ) from error
            if not isinstance(value, dict):
                raise RegistrationRejected("Collaboration test-read evidence is not an object")
            states[checkpoint_id] = value
        return build_collaboration_task_acceptance(
            experiment_id=experiment_id,
            runs=runs,
            checkpoints=checkpoints,
            evaluations=relevant_reports,
            protocol=protocol,
            test_read_states=states,
            backbone_bindings=backbone_bindings,
            accepted_pretraining_checkpoint_ids=(
                pretraining.selected_checkpoint_ids
            ),
            pretraining_acceptance_report_hash=pretraining.report_hash,
            accepted_pretraining_variant=pretraining.selected_variant,
        )

    def build_collaboration_task_acceptance(
        self, *, experiment_id: str
    ) -> GfmTaskAcceptanceManifest:
        return self._recompute_collaboration_task_acceptance(
            experiment_id=experiment_id
        )

    def verify_task_acceptance(
        self, manifest: GfmTaskAcceptanceManifest
    ) -> GfmTaskAcceptanceManifest:
        checked = GfmTaskAcceptanceManifest.model_validate(manifest)
        recomputed = self._recompute_collaboration_task_acceptance(
            experiment_id=checked.experiment_id
        )
        if (
            recomputed.logical_payload() != checked.logical_payload()
            or recomputed.report_hash != checked.report_hash
        ):
            raise RegistrationRejected(
                "Stored task acceptance differs from registry-derived evidence"
            )
        return recomputed

    def record_task_acceptance(self, manifest: GfmTaskAcceptanceManifest) -> None:
        checked = GfmTaskAcceptanceManifest.model_validate(manifest)
        self.verify_task_acceptance(checked)
        raw = canonical_json(checked)
        with self.connect() as connection:
            self._insert_immutable(
                connection,
                table="gfm_task_acceptances",
                identity_column="report_hash",
                identity=checked.report_hash,
                columns=(
                    "report_hash",
                    "experiment_id",
                    "task_id",
                    "accepted",
                    "created_at",
                    "registrable",
                ),
                values=(
                    checked.report_hash,
                    checked.experiment_id,
                    checked.task_id,
                    int(checked.accepted),
                    checked.created_at.isoformat(),
                    0,
                ),
                manifest_json=raw,
            )

    def record_acceptance(self, manifest: GfmAcceptanceManifest) -> None:
        checked = GfmAcceptanceManifest.model_validate(manifest)
        recomputed = self._recompute_acceptance(
            experiment_id=checked.experiment_id,
            checkpoint_id=checked.checkpoint_id,
        )
        if (
            recomputed.logical_payload() != checked.logical_payload()
            or recomputed.report_hash != checked.report_hash
        ):
            raise RegistrationRejected(
                "GFM acceptance differs from registry-derived immutable evidence"
            )
        raw = canonical_json(checked)
        with self.connect() as connection:
            checkpoint_row = connection.execute(
                "SELECT run_id, manifest_json FROM gfm_checkpoints WHERE checkpoint_id=?",
                (checked.checkpoint_id,),
            ).fetchone()
            if checkpoint_row is None:
                raise RegistrationRejected("GFM acceptance references an unknown checkpoint")
            checkpoint = GfmCheckpointManifest.model_validate_json(checkpoint_row["manifest_json"])
            run_row = connection.execute(
                "SELECT manifest_json FROM gfm_runs WHERE run_id=?",
                (checkpoint_row["run_id"],),
            ).fetchone()
            if run_row is None:
                raise RegistrationRejected("GFM acceptance checkpoint has no registered run")
            run = GfmRunManifest.model_validate_json(run_row["manifest_json"])
            if (
                run.experiment_id != checked.experiment_id
                or run.config_hash != checked.config_hash
                or run.code_hash != checked.code_hash
                or run.environment_hash != checked.environment_hash
                or set(checkpoint.corpus_hashes) != set(checked.corpus_hashes)
            ):
                raise RegistrationRejected("GFM acceptance provenance is inconsistent")
            corpus_rows = connection.execute(
                "SELECT logical_hash FROM gfm_domain_corpora"
            ).fetchall()
            evaluation_rows = connection.execute(
                "SELECT report_hash, experiment_id, checkpoint_id FROM gfm_evaluations"
            ).fetchall()
            known_corpora = {row["logical_hash"] for row in corpus_rows}
            known_reports = {
                row["report_hash"]
                for row in evaluation_rows
                if row["experiment_id"] == checked.experiment_id
            }
            if not set(checked.corpus_hashes).issubset(known_corpora) or not set(
                checked.evaluation_report_hashes
            ).issubset(known_reports):
                raise RegistrationRejected("GFM acceptance references unregistered evidence")
            if checked.accepted and run.status != "succeeded":
                raise RegistrationRejected("Only a succeeded GFM run can be accepted")
            self._insert_immutable(
                connection,
                table="gfm_acceptances",
                identity_column="report_hash",
                identity=checked.report_hash,
                columns=(
                    "report_hash",
                    "experiment_id",
                    "checkpoint_id",
                    "accepted",
                    "created_at",
                ),
                values=(
                    checked.report_hash,
                    checked.experiment_id,
                    checked.checkpoint_id,
                    int(checked.accepted),
                    checked.created_at.isoformat(),
                ),
                manifest_json=raw,
            )

    def build_acceptance(self, *, experiment_id: str, checkpoint_id: str) -> GfmAcceptanceManifest:
        """Build a fail-closed acceptance contract from registry-owned evidence."""

        return self._recompute_acceptance(
            experiment_id=experiment_id,
            checkpoint_id=checkpoint_id,
        )

    def promote_model(self, *, model_id: str, experiment_id: str) -> dict[str, str]:
        """Promote only after accepted evidence and a fresh integrity load."""

        if not model_id:
            raise RegistrationRejected("model_id must be nonempty")
        with self.connect() as connection:
            acceptance_row = connection.execute(
                """
                SELECT report_hash, checkpoint_id, manifest_json
                FROM gfm_acceptances
                WHERE experiment_id=? AND accepted=1
                ORDER BY created_at DESC, report_hash DESC
                LIMIT 1
                """,
                (experiment_id,),
            ).fetchone()
            if acceptance_row is None:
                raise RegistrationRejected("GFM cannot be promoted before acceptance")
            acceptance = GfmAcceptanceManifest.model_validate_json(acceptance_row["manifest_json"])
            recomputed = self._recompute_acceptance(
                experiment_id=experiment_id,
                checkpoint_id=acceptance.checkpoint_id,
            )
            if (
                not recomputed.accepted
                or recomputed.logical_payload() != acceptance.logical_payload()
                or recomputed.report_hash != acceptance.report_hash
            ):
                raise RegistrationRejected(
                    "Stored GFM acceptance no longer matches registry-derived evidence"
                )
            checkpoint_row = connection.execute(
                "SELECT manifest_json FROM gfm_checkpoints WHERE checkpoint_id=?",
                (acceptance.checkpoint_id,),
            ).fetchone()
            if checkpoint_row is None:
                raise RegistrationRejected("Accepted GFM checkpoint is not registered")
            checkpoint = GfmCheckpointManifest.model_validate_json(checkpoint_row["manifest_json"])
            run_row = connection.execute(
                "SELECT status FROM gfm_runs WHERE run_id=?", (checkpoint.run_id,)
            ).fetchone()
            if run_row is None or run_row["status"] != "succeeded":
                raise RegistrationRejected("Accepted GFM run is not succeeded")
            try:
                load_gfm_checkpoint(checkpoint, map_location="cpu")
            except Exception as error:
                raise RegistrationRejected(
                    f"Accepted checkpoint failed integrity verification: {error}"
                ) from error
            promoted_at = datetime.now(UTC).isoformat()
            promotion = {
                "modelId": model_id,
                "experimentId": experiment_id,
                "checkpointId": checkpoint.checkpoint_id,
                "acceptanceReportHash": acceptance.report_hash,
                "promotedAt": promoted_at,
            }
            raw = canonical_json(promotion)
            existing = connection.execute(
                "SELECT promotion_json FROM gfm_models WHERE model_id=?", (model_id,)
            ).fetchone()
            if existing is not None:
                stored = json.loads(existing["promotion_json"])
                comparable = {key: value for key, value in stored.items() if key != "promotedAt"}
                requested = {key: value for key, value in promotion.items() if key != "promotedAt"}
                if comparable != requested:
                    raise RegistrationRejected("model_id already promotes different evidence")
                return stored
            connection.execute(
                """
                INSERT INTO gfm_models(
                    model_id, experiment_id, checkpoint_id, acceptance_report_hash,
                    promoted_at, promotion_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    experiment_id,
                    checkpoint.checkpoint_id,
                    acceptance.report_hash,
                    promoted_at,
                    raw,
                ),
            )
            return promotion

    def _recompute_acceptance(
        self, *, experiment_id: str, checkpoint_id: str
    ) -> GfmAcceptanceManifest:
        """Derive acceptance only from immutable rows, never caller booleans."""

        checkpoint = self.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            raise RegistrationRejected("Acceptance checkpoint is not registered")
        try:
            load_gfm_checkpoint(checkpoint, map_location="cpu")
        except Exception as error:
            raise RegistrationRejected(
                f"Acceptance checkpoint failed integrity verification: {error}"
            ) from error
        run = self.get_run(checkpoint.run_id)
        if run is None or run.experiment_id != experiment_id or run.status != "succeeded":
            raise RegistrationRejected("Acceptance checkpoint belongs to another experiment")
        with self.connect() as connection:
            corpus_rows = connection.execute(
                "SELECT manifest_json FROM gfm_domain_corpora ORDER BY domain_id, corpus_id"
            ).fetchall()
            evaluation_rows = connection.execute(
                """
                SELECT evaluation.manifest_json AS evaluation_json,
                       run.manifest_json AS run_json
                FROM gfm_evaluations AS evaluation
                JOIN gfm_runs AS run ON run.run_id=evaluation.run_id
                WHERE evaluation.experiment_id=? ORDER BY evaluation.report_id
                """,
                (experiment_id,),
            ).fetchall()
        all_corpora = tuple(
            GfmDomainCorpusManifest.model_validate_json(row["manifest_json"]) for row in corpus_rows
        )
        corpora = tuple(
            corpus for corpus in all_corpora if corpus.logical_hash in set(checkpoint.corpus_hashes)
        )
        if {corpus.logical_hash for corpus in corpora} != set(checkpoint.corpus_hashes):
            raise RegistrationRejected(
                "Acceptance checkpoint corpus inventory is not exactly registered"
            )
        evaluations_list: list[GfmEvaluationReport] = []
        semantic_keys: set[tuple[Any, ...]] = set()
        selected_hashes = set(run.corpus_hashes)
        selected_corpus_by_domain = {corpus.domain_id: corpus.logical_hash for corpus in corpora}
        for row in evaluation_rows:
            evidence_run = GfmRunManifest.model_validate_json(row["run_json"])
            report = GfmEvaluationReport.model_validate_json(row["evaluation_json"])
            common_provenance_differs = (
                evidence_run.architecture_variant != run.architecture_variant
                or evidence_run.config_hash != run.config_hash
                or evidence_run.code_hash != run.code_hash
                or evidence_run.environment_hash != run.environment_hash
            )
            if common_provenance_differs:
                continue
            evidence_hashes = set(evidence_run.corpus_hashes)
            if report.evaluation_kind == "lodo":
                held_out = report.held_out_domain
                held_out_hash = selected_corpus_by_domain.get(str(held_out))
                if held_out_hash is None:
                    continue
                expected_source_domains = set(selected_corpus_by_domain) - {str(held_out)}
                expected_source_hashes = selected_hashes - {held_out_hash}
                evidence_payload = json.loads(
                    Path(report.evidence_artifact_path).read_text(encoding="utf-8")
                ).get("payload")
                audit_payload = json.loads(
                    Path(report.leakage_audit_path).read_text(encoding="utf-8")
                )
                isolation_evidence = audit_payload.get("evidence")
                if (
                    evidence_run.phase != "lodo"
                    or evidence_run.held_out_domain != held_out
                    or report.domain_id != held_out
                    or set(evidence_run.domain_ids) != expected_source_domains
                    # The registered checkpoint is target few-shot adapted and
                    # must truthfully bind all three corpora.  The independently
                    # hashed isolation evidence proves that pretraining itself
                    # saw exactly the other two.
                    or evidence_hashes != selected_hashes
                    or not isinstance(evidence_payload, dict)
                    or not isinstance(isolation_evidence, dict)
                    or set(isolation_evidence.get("sourceCorpusHashes", ()))
                    != expected_source_hashes
                    or isolation_evidence.get("targetCorpusHash") != held_out_hash
                    or not isinstance(evidence_payload.get("isolationHash"), str)
                ):
                    continue
            elif evidence_hashes != selected_hashes:
                continue
            evidence_checkpoint = self.get_checkpoint(report.checkpoint_id)
            if evidence_checkpoint is None or evidence_checkpoint.run_id != report.run_id:
                raise RegistrationRejected("Evaluation report checkpoint/run binding is absent")
            try:
                load_gfm_checkpoint(evidence_checkpoint, map_location="cpu")
            except Exception as error:
                raise RegistrationRejected(
                    f"Evaluation checkpoint failed integrity verification: {error}"
                ) from error
            self._verify_evaluation_artifacts(report)
            evidence = json.loads(Path(report.evidence_artifact_path).read_text("utf-8"))
            payload = evidence.get("payload")
            if isinstance(payload, dict) and payload.get("split") == "shadow":
                continue
            if report.evaluation_kind in {"lodo", "product", "calibration"}:
                semantic_key = (
                    report.evaluation_kind,
                    report.domain_id,
                    report.held_out_domain,
                    report.task_id,
                    report.seed,
                )
                if semantic_key in semantic_keys:
                    raise RegistrationRejected(
                        "Acceptance evidence contains duplicate semantic report keys"
                    )
                semantic_keys.add(semantic_key)
            evaluations_list.append(report)
        evaluations = tuple(evaluations_list)
        if not evaluations:
            raise RegistrationRejected("Acceptance requires registered evaluation evidence")
        selected_payload = load_gfm_checkpoint(checkpoint, map_location="cpu")
        delivery_hashes: tuple[str, ...] | None = None
        components = selected_payload.get("components")
        best_state = selected_payload.get("best_state")
        if isinstance(components, dict) and "suite_config" in components:
            if not isinstance(best_state, dict):
                raise RegistrationRejected("Product suite lacks immutable source bindings")
            suite_config = components.get("suite_config")
            if not isinstance(suite_config, dict):
                raise RegistrationRejected("Product suite configuration is absent")
            checked_suite_config = dict(suite_config)
            suite_config_hash = checked_suite_config.pop("taskConfigHash", None)
            if suite_config_hash != canonical_sha256(checked_suite_config):
                raise RegistrationRejected("Product suite configuration hash is invalid")
            source_bindings = best_state.get("sourceBindings")
            source_report_hashes = best_state.get("sourceReportHashes")
            if (
                not isinstance(source_bindings, dict)
                or set(source_bindings) != {"collaboration", "newcomer"}
                or not isinstance(source_report_hashes, (tuple, list))
                or len(source_report_hashes) != 4
            ):
                raise RegistrationRejected("Product suite source binding inventory is invalid")
            report_by_hash = {report.report_hash: report for report in evaluations}
            verified_delivery: list[str] = []
            expected_task_ids = {
                "collaboration": "governance.collaboration_recommendation",
                "newcomer": "core.newcomer_support",
            }
            for task in ("collaboration", "newcomer"):
                binding = source_bindings.get(task)
                if not isinstance(binding, dict):
                    raise RegistrationRejected("Suite source binding is not an object")
                source_checkpoint_id = binding.get("checkpointId")
                source_checkpoint = (
                    self.get_checkpoint(source_checkpoint_id)
                    if isinstance(source_checkpoint_id, str)
                    else None
                )
                if source_checkpoint is None:
                    raise RegistrationRejected("Suite source checkpoint is absent")
                source_run = self.get_run(source_checkpoint.run_id)
                if (
                    source_run is None
                    or source_run.status != "succeeded"
                    or source_run.phase != "adapt"
                    or source_run.experiment_id != experiment_id
                    or source_run.architecture_variant != run.architecture_variant
                    or source_run.seed != run.seed
                    or source_run.config_hash != run.config_hash
                    or source_run.code_hash != run.code_hash
                    or source_run.environment_hash != run.environment_hash
                    or set(source_run.corpus_hashes) != selected_hashes
                ):
                    raise RegistrationRejected(
                        "Suite source run differs from selected model provenance"
                    )
                source_payload = load_gfm_checkpoint(source_checkpoint, map_location="cpu")
                source_components = source_payload.get("components")
                source_product_config = (
                    source_components.get("product_config")
                    if isinstance(source_components, dict)
                    else None
                )
                if (
                    not isinstance(source_components, dict)
                    or not isinstance(source_product_config, dict)
                    or binding.get("stateHash") != source_checkpoint.state_hash
                    or binding.get("componentStateHash")
                    != self._tensor_state_hash(source_components.get("product"))
                    or self._tensor_state_hash(components.get(task))
                    != self._tensor_state_hash(source_components.get("product"))
                    or components.get(f"{task}_config") != source_components.get("product_config")
                    or binding.get("productConfigHash")
                    != source_product_config.get("taskConfigHash")
                ):
                    raise RegistrationRejected(
                        "Suite component/config differs from its source checkpoint"
                    )
                self._validate_formal_embedding_evidence(source_product_config)
                task_hashes = (
                    binding.get("productReportHash"),
                    binding.get("calibrationReportHash"),
                )
                for expected_kind, report_hash in zip(
                    ("product", "calibration"), task_hashes, strict=True
                ):
                    if not isinstance(report_hash, str):
                        raise RegistrationRejected("Suite source report hash is invalid")
                    source_report = report_by_hash.get(report_hash)
                    if (
                        source_report is None
                        or source_report.checkpoint_id != source_checkpoint.checkpoint_id
                        or source_report.evaluation_kind != expected_kind
                        or source_report.seed != run.seed
                        or source_report.task_id != expected_task_ids[task]
                    ):
                        raise RegistrationRejected(
                            "Suite source report differs from its bound checkpoint/kind"
                        )
                    verified_delivery.append(source_report.report_hash)
            fresh_reports = [
                report
                for report in evaluations
                if report.checkpoint_id == checkpoint.checkpoint_id
                and report.evaluation_kind == "fresh_process"
            ]
            if len(fresh_reports) != 1 or set(source_report_hashes) != set(verified_delivery):
                raise RegistrationRejected(
                    "Suite delivery evidence lacks four source reports and one fresh report"
                )
            delivery_hashes = tuple(verified_delivery + [fresh_reports[0].report_hash])
        return build_gfm_acceptance(
            experiment_id=experiment_id,
            checkpoint_id=checkpoint_id,
            corpora=corpora,
            evaluations=evaluations,
            config_hash=run.config_hash,
            code_hash=run.code_hash,
            environment_hash=run.environment_hash,
            delivery_evidence_report_hashes=delivery_hashes,
        )

    @staticmethod
    def _tensor_state_hash(value: Any) -> str:
        from ..tensor_digest import canonical_tensor_digest

        if not isinstance(value, dict):
            raise RegistrationRejected("Checkpoint component state is not a mapping")
        return canonical_sha256(
            {name: canonical_tensor_digest(tensor) for name, tensor in sorted(value.items())}
        )

    @staticmethod
    def _validate_formal_embedding_evidence(config: dict[str, Any]) -> None:
        artifacts = config.get("embeddingArtifacts")
        artifacts_hash = config.get("embeddingArtifactsHash")
        if (
            not isinstance(artifacts, dict)
            or artifacts_hash != canonical_sha256(artifacts)
            or set(artifacts)
            != {
                "openalex-graph-ai",
                "wikimedia-talk-article-2011-2015",
            }
        ):
            raise RegistrationRejected("Formal embedding artifact inventory is invalid")
        expected_producer = {
            "implementation": "FlagEmbedding.BGEM3FlagModel",
            "distribution": "FlagEmbedding",
            "version": "1.4.0",
            "formalEligible": True,
        }
        for evidence in artifacts.values():
            access_view = evidence.get("accessView") if isinstance(evidence, dict) else None
            if (
                not isinstance(evidence, dict)
                or evidence.get("producer") != expected_producer
                or evidence.get("modelId") != "BAAI/bge-m3"
                or evidence.get("modelRevision") != "5617a9f61b028005a4858fdac845db406aefb181"
                or not isinstance(evidence.get("logicalHash"), str)
                or not isinstance(access_view, dict)
                or access_view.get("maximumRole")
                not in {
                    "validation",
                    "test",
                    "full",
                }
                or not isinstance(access_view.get("selectedRows"), int)
                or access_view["selectedRows"] < 1
                or not isinstance(access_view.get("selectedShardPaths"), list)
                or not isinstance(access_view.get("selectedShardHashes"), list)
                or len(access_view["selectedShardPaths"]) != len(access_view["selectedShardHashes"])
            ):
                raise RegistrationRejected("Formal embedding evidence differs from pinned BGE-M3")

    def get_run(self, run_id: str) -> GfmRunManifest | None:
        return self._get_contract("gfm_runs", "run_id", run_id, GfmRunManifest)

    def get_checkpoint(self, checkpoint_id: str) -> GfmCheckpointManifest | None:
        return self._get_contract(
            "gfm_checkpoints", "checkpoint_id", checkpoint_id, GfmCheckpointManifest
        )

    def get_acceptance(self, report_hash: str) -> GfmAcceptanceManifest | None:
        return self._get_contract(
            "gfm_acceptances", "report_hash", report_hash, GfmAcceptanceManifest
        )

    def get_pretraining_acceptance(
        self, report_hash: str
    ) -> GfmPretrainingAcceptanceManifest | None:
        return self._get_contract(
            "gfm_pretraining_acceptances",
            "report_hash",
            report_hash,
            GfmPretrainingAcceptanceManifest,
        )

    def get_task_acceptance(self, report_hash: str) -> GfmTaskAcceptanceManifest | None:
        return self._get_contract(
            "gfm_task_acceptances", "report_hash", report_hash, GfmTaskAcceptanceManifest
        )

    def list_runs(self, *, experiment_id: str) -> tuple[GfmRunManifest, ...]:
        """Return an experiment's immutable runs in stable identity order."""

        with self.connect() as connection:
            rows = connection.execute(
                "SELECT manifest_json FROM gfm_runs WHERE experiment_id=? ORDER BY run_id",
                (experiment_id,),
            ).fetchall()
        return tuple(GfmRunManifest.model_validate_json(row["manifest_json"]) for row in rows)

    def list_checkpoints(self, *, experiment_id: str) -> tuple[GfmCheckpointManifest, ...]:
        """Return checkpoints whose parent runs belong to one experiment."""

        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT checkpoint.manifest_json
                FROM gfm_checkpoints AS checkpoint
                JOIN gfm_runs AS run ON run.run_id=checkpoint.run_id
                WHERE run.experiment_id=?
                ORDER BY checkpoint.checkpoint_id
                """,
                (experiment_id,),
            ).fetchall()
        return tuple(
            GfmCheckpointManifest.model_validate_json(row["manifest_json"]) for row in rows
        )

    def list_evaluations(self, *, experiment_id: str) -> tuple[GfmEvaluationReport, ...]:
        """Return only evidence bound to the requested experiment."""

        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT manifest_json FROM gfm_evaluations
                WHERE experiment_id=? ORDER BY report_id
                """,
                (experiment_id,),
            ).fetchall()
        return tuple(GfmEvaluationReport.model_validate_json(row["manifest_json"]) for row in rows)

    def list_acceptances(self, *, experiment_id: str) -> tuple[GfmAcceptanceManifest, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT manifest_json FROM gfm_acceptances
                WHERE experiment_id=? ORDER BY created_at, report_hash
                """,
                (experiment_id,),
            ).fetchall()
        return tuple(
            GfmAcceptanceManifest.model_validate_json(row["manifest_json"]) for row in rows
        )

    def latest_acceptance(self, *, experiment_id: str) -> GfmAcceptanceManifest | None:
        values = self.list_acceptances(experiment_id=experiment_id)
        return values[-1] if values else None

    def list_pretraining_acceptances(
        self, *, experiment_id: str
    ) -> tuple[GfmPretrainingAcceptanceManifest, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT manifest_json FROM gfm_pretraining_acceptances
                WHERE experiment_id=? ORDER BY created_at, report_hash
                """,
                (experiment_id,),
            ).fetchall()
        return tuple(
            GfmPretrainingAcceptanceManifest.model_validate_json(row["manifest_json"])
            for row in rows
        )

    def latest_pretraining_acceptance(
        self, *, experiment_id: str | None = None
    ) -> GfmPretrainingAcceptanceManifest | None:
        with self.connect() as connection:
            if experiment_id is None:
                row = connection.execute(
                    """
                    SELECT manifest_json FROM gfm_pretraining_acceptances
                    ORDER BY created_at DESC, report_hash DESC LIMIT 1
                    """
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT manifest_json FROM gfm_pretraining_acceptances
                    WHERE experiment_id=?
                    ORDER BY created_at DESC, report_hash DESC LIMIT 1
                    """,
                    (experiment_id,),
                ).fetchone()
        return (
            None
            if row is None
            else GfmPretrainingAcceptanceManifest.model_validate_json(row["manifest_json"])
        )

    def list_task_acceptances(
        self, *, experiment_id: str
    ) -> tuple[GfmTaskAcceptanceManifest, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT manifest_json FROM gfm_task_acceptances
                WHERE experiment_id=? AND task_id=? ORDER BY created_at, report_hash
                """,
                (experiment_id, COLLABORATION_TASK),
            ).fetchall()
        return tuple(
            GfmTaskAcceptanceManifest.model_validate_json(row["manifest_json"])
            for row in rows
        )

    def latest_task_acceptance(
        self, *, experiment_id: str | None = None
    ) -> GfmTaskAcceptanceManifest | None:
        with self.connect() as connection:
            if experiment_id is None:
                row = connection.execute(
                    """
                    SELECT manifest_json FROM gfm_task_acceptances
                    WHERE task_id=? ORDER BY created_at DESC, report_hash DESC LIMIT 1
                    """,
                    (COLLABORATION_TASK,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT manifest_json FROM gfm_task_acceptances
                    WHERE experiment_id=? AND task_id=?
                    ORDER BY created_at DESC, report_hash DESC LIMIT 1
                    """,
                    (experiment_id, COLLABORATION_TASK),
                ).fetchone()
        return (
            None
            if row is None
            else GfmTaskAcceptanceManifest.model_validate_json(row["manifest_json"])
        )

    def _get_contract(self, table: str, column: str, identity: str, model: Any) -> Any:
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT manifest_json FROM {table} WHERE {column}=?", (identity,)
            ).fetchone()
        return None if row is None else model.model_validate_json(row["manifest_json"])

    def counts(self) -> dict[str, int]:
        tables = (
            "gfm_domain_corpora",
            "gfm_task_protocols",
            "gfm_runs",
            "gfm_checkpoints",
            "gfm_evaluations",
            "gfm_acceptances",
            "gfm_pretraining_acceptances",
            "gfm_task_acceptances",
            "gfm_models",
        )
        with self.connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in tables
            }


__all__ = ["GFM_SCHEMA", "GfmRegistry"]
