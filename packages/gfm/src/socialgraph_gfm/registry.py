"""SQLite-backed local run/checkpoint registry with fail-closed promotion rules."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ValidationError

from .canonical import canonical_json, canonical_sha256, file_sha256
from .contracts import (
    BaselineAcceptanceReport,
    BaselineCheckpointManifest,
    BaselineEvaluationReport,
    BaselineRunManifest,
    RunStatus,
    SmokeCheckpointManifest,
    SmokeTrainingRunManifest,
)
from .errors import RegistrationRejected
from .checkpoint import load_checkpoint

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    run_kind TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    smoke_only INTEGER NOT NULL CHECK (smoke_only IN (0, 1)),
    artifact_sha256 TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    checkpoint_id TEXT NOT NULL UNIQUE REFERENCES checkpoints(checkpoint_id),
    validated INTEGER NOT NULL CHECK (validated = 1),
    registered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS baseline_runs (
    manifest_hash TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    experiment_id TEXT NOT NULL,
    status TEXT NOT NULL,
    track TEXT NOT NULL,
    model_kind TEXT NOT NULL,
    registrable INTEGER NOT NULL CHECK (registrable = 0),
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS baseline_checkpoints (
    manifest_hash TEXT PRIMARY KEY,
    checkpoint_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES baseline_runs(run_id),
    registrable INTEGER NOT NULL CHECK (registrable = 0),
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS baseline_evaluations (
    manifest_hash TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES baseline_runs(run_id),
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS baseline_acceptances (
    manifest_hash TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL UNIQUE,
    baseline_validated INTEGER NOT NULL CHECK (baseline_validated IN (0, 1)),
    registrable INTEGER NOT NULL CHECK (registrable = 0),
    manifest_json TEXT NOT NULL
);
"""

_BASELINE_KINDS = {
    "baseline_run": "gfm.baseline-run/1.0",
    "baseline_checkpoint": "gfm.baseline-checkpoint/1.0",
    "baseline_evaluation": "gfm.baseline-evaluation/1.0",
    "baseline_acceptance": "gfm.baseline-acceptance/1.0",
}
_BASELINE_MODELS: dict[str, type[BaseModel]] = {
    "baseline_run": BaselineRunManifest,
    "baseline_checkpoint": BaselineCheckpointManifest,
    "baseline_evaluation": BaselineEvaluationReport,
    "baseline_acceptance": BaselineAcceptanceReport,
}


def _manifest_dict(manifest: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(manifest, BaseModel):
        return manifest.model_dump(mode="json", by_alias=True, exclude_none=False)
    return dict(manifest)


def _value(payload: Mapping[str, Any], camel: str, snake: str | None = None) -> Any:
    if camel in payload:
        return payload[camel]
    return payload.get(snake or camel)


def _require_text(payload: Mapping[str, Any], camel: str, snake: str | None = None) -> str:
    value = _value(payload, camel, snake)
    if not isinstance(value, str) or not value:
        raise RegistrationRejected(f"baseline manifest requires {camel}")
    return value


def _baseline_payload(
    manifest: BaseModel | Mapping[str, Any], expected_kind: str
) -> tuple[dict[str, Any], str]:
    raw_payload = _manifest_dict(manifest)
    if expected_kind in ("baseline_run", "baseline_checkpoint") and _value(
        raw_payload, "registrable"
    ) is not False:
        raise RegistrationRejected("baseline artifacts must set registrable=false")
    if not isinstance(manifest, _BASELINE_MODELS[expected_kind]):
        try:
            manifest = _BASELINE_MODELS[expected_kind].model_validate(manifest)
        except ValidationError as error:
            raise RegistrationRejected(f"invalid {expected_kind} manifest: {error}") from error
    payload = _manifest_dict(manifest)
    kind = _value(payload, "manifestKind", "manifest_kind")
    schema = _value(payload, "schemaVersion", "schema_version")
    if kind is not None and kind != expected_kind:
        raise RegistrationRejected(
            f"expected manifestKind={expected_kind}, found {kind!r}"
        )
    if schema != _BASELINE_KINDS[expected_kind]:
        raise RegistrationRejected(
            f"expected schemaVersion={_BASELINE_KINDS[expected_kind]}, found {schema!r}"
        )
    if expected_kind in ("baseline_run", "baseline_checkpoint") and _value(
        payload, "registrable"
    ) is not False:
        raise RegistrationRejected("baseline artifacts must set registrable=false")
    logical_payload = getattr(manifest, "logical_payload", None)
    if callable(logical_payload):
        expected_hash = canonical_sha256(logical_payload())
    else:
        excluded = {"manifestHash", "manifest_hash"}
        if expected_kind in ("baseline_evaluation", "baseline_acceptance"):
            excluded.update(("reportHash", "report_hash"))
        if expected_kind == "baseline_acceptance":
            excluded.update(("createdAt", "created_at"))
        without_hash = {
            key: value
            for key, value in payload.items()
            if key not in excluded
        }
        expected_hash = canonical_sha256(without_hash)
    claimed_hash = _value(payload, "manifestHash", "manifest_hash")
    if claimed_hash is None and expected_kind in ("baseline_evaluation", "baseline_acceptance"):
        claimed_hash = _value(payload, "reportHash", "report_hash")
    if claimed_hash is None:
        claimed_hash = expected_hash
    if not isinstance(claimed_hash, str) or len(claimed_hash) != 64 or any(
        character not in "0123456789abcdef" for character in claimed_hash
    ):
        raise RegistrationRejected("baseline manifest hash must be a lowercase SHA-256")
    if claimed_hash != expected_hash:
        raise RegistrationRejected("baseline manifestHash does not match its logical payload")
    return payload, claimed_hash


class LocalRegistry:
    def __init__(self, path: str | Path, *, initialize: bool = True) -> None:
        self.path = Path(path)
        if initialize:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as connection:
                connection.executescript(SCHEMA)
        elif not self.path.is_file():
            raise FileNotFoundError(self.path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def record_run(self, manifest: SmokeTrainingRunManifest) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(run_id, status, run_kind, manifest_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    run_kind=excluded.run_kind,
                    manifest_json=excluded.manifest_json
                """,
                (
                    manifest.run_id,
                    manifest.status.value,
                    manifest.run_kind,
                    canonical_json(manifest),
                ),
            )

    def record_checkpoint(self, manifest: SmokeCheckpointManifest) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO checkpoints(
                    checkpoint_id, run_id, smoke_only, artifact_sha256, manifest_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    manifest.checkpoint_id,
                    manifest.run_id,
                    int(manifest.smoke_only),
                    manifest.artifact_sha256,
                    canonical_json(manifest),
                ),
            )

    def register_model(
        self,
        model_id: str,
        run: SmokeTrainingRunManifest,
        checkpoint: SmokeCheckpointManifest,
        *,
        validated: bool,
    ) -> None:
        reasons = []
        if (
            isinstance(run, BaselineRunManifest)
            or isinstance(checkpoint, BaselineCheckpointManifest)
            or getattr(run, "manifest_kind", None) == "baseline_run"
            or getattr(checkpoint, "manifest_kind", None) == "baseline_checkpoint"
        ):
            raise RegistrationRejected("baseline artifacts are never promotable")
        if run.status != RunStatus.SUCCEEDED:
            reasons.append(f"run status is {run.status.value}")
        if run.run_kind == "smoke":
            reasons.append("smoke runs are never promotable")
        if run.corpus.purpose == "synthetic_test_only":
            reasons.append("synthetic_test_only corpora are never promotable")
        if checkpoint.smoke_only:
            reasons.append("smoke checkpoints are never promotable")
        if checkpoint.run_id != run.run_id:
            reasons.append("checkpoint belongs to a different run")
        if not validated:
            reasons.append("validated=true is required")
        if reasons:
            raise RegistrationRejected("; ".join(reasons))
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO models(model_id, checkpoint_id, validated) VALUES (?, ?, 1)",
                (model_id, checkpoint.checkpoint_id),
            )

    def record_baseline_run(self, manifest: BaseModel | Mapping[str, Any]) -> None:
        payload, manifest_hash = _baseline_payload(manifest, "baseline_run")
        run_id = _require_text(payload, "runId", "run_id")
        experiment_id = _require_text(payload, "experimentId", "experiment_id")
        status = _require_text(payload, "status")
        track = _require_text(payload, "track")
        model_kind = _value(payload, "modelKind", "model_kind") or _require_text(
            payload, "model"
        )
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO baseline_runs(
                    manifest_hash, run_id, experiment_id, status, track, model_kind,
                    registrable, manifest_json
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    manifest_hash=excluded.manifest_hash,
                    experiment_id=excluded.experiment_id,
                    status=excluded.status,
                    track=excluded.track,
                    model_kind=excluded.model_kind,
                    registrable=0,
                    manifest_json=excluded.manifest_json
                """,
                (
                    manifest_hash,
                    run_id,
                    experiment_id,
                    status,
                    track,
                    model_kind,
                    canonical_json(payload),
                ),
            )

    def record_baseline_checkpoint(self, manifest: BaseModel | Mapping[str, Any]) -> None:
        payload, manifest_hash = _baseline_payload(manifest, "baseline_checkpoint")
        checkpoint_id = _require_text(payload, "checkpointId", "checkpoint_id")
        run_id = _require_text(payload, "runId", "run_id")
        with self.connect() as connection:
            known_run = connection.execute(
                "SELECT 1 FROM baseline_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if known_run is None:
                raise RegistrationRejected(f"unknown baseline run: {run_id}")
            connection.execute(
                """
                INSERT INTO baseline_checkpoints(
                    manifest_hash, checkpoint_id, run_id, registrable, manifest_json
                ) VALUES (?, ?, ?, 0, ?)
                ON CONFLICT(checkpoint_id) DO UPDATE SET
                    manifest_hash=excluded.manifest_hash,
                    run_id=excluded.run_id,
                    registrable=0,
                    manifest_json=excluded.manifest_json
                """,
                (manifest_hash, checkpoint_id, run_id, canonical_json(payload)),
            )

    def record_baseline_evaluation(self, manifest: BaseModel | Mapping[str, Any]) -> None:
        payload, manifest_hash = _baseline_payload(manifest, "baseline_evaluation")
        evaluation_id = _value(payload, "evaluationId", "evaluation_id") or manifest_hash
        run_id = _require_text(payload, "runId", "run_id")
        with self.connect() as connection:
            known_run = connection.execute(
                "SELECT 1 FROM baseline_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if known_run is None:
                raise RegistrationRejected(f"unknown baseline run: {run_id}")
            connection.execute(
                """
                INSERT INTO baseline_evaluations(
                    manifest_hash, evaluation_id, run_id, manifest_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(evaluation_id) DO UPDATE SET
                    manifest_hash=excluded.manifest_hash,
                    run_id=excluded.run_id,
                    manifest_json=excluded.manifest_json
                """,
                (manifest_hash, evaluation_id, run_id, canonical_json(payload)),
            )

    def record_baseline_acceptance(self, manifest: BaseModel | Mapping[str, Any]) -> None:
        payload, manifest_hash = _baseline_payload(manifest, "baseline_acceptance")
        experiment_id = _require_text(payload, "experimentId", "experiment_id")
        validated = _value(payload, "baselineValidated", "baseline_validated")
        if validated is None:
            validated = _value(payload, "accepted")
        if not isinstance(validated, bool):
            raise RegistrationRejected("baseline acceptance requires baselineValidated boolean")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO baseline_acceptances(
                    manifest_hash, experiment_id, baseline_validated, registrable, manifest_json
                ) VALUES (?, ?, ?, 0, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    manifest_hash=excluded.manifest_hash,
                    baseline_validated=excluded.baseline_validated,
                    registrable=0,
                    manifest_json=excluded.manifest_json
                """,
                (manifest_hash, experiment_id, int(validated), canonical_json(payload)),
            )

    def register_baseline_model(self, *_args: Any, **_kwargs: Any) -> None:
        """Make the non-promotable baseline boundary explicit to callers."""

        raise RegistrationRejected("baseline artifacts are never promotable")

    def baseline_counts(self) -> dict[str, int]:
        with self.connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "baseline_runs",
                    "baseline_checkpoints",
                    "baseline_evaluations",
                    "baseline_acceptances",
                )
            }

    def list_baseline_runs(self, experiment_id: str) -> tuple[BaselineRunManifest, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT manifest_json FROM baseline_runs
                WHERE experiment_id=? ORDER BY run_id
                """,
                (experiment_id,),
            ).fetchall()
        return tuple(BaselineRunManifest.model_validate_json(row[0]) for row in rows)

    def list_baseline_checkpoints(
        self, experiment_id: str
    ) -> tuple[BaselineCheckpointManifest, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT baseline_checkpoints.manifest_json
                FROM baseline_checkpoints
                JOIN baseline_runs ON baseline_runs.run_id=baseline_checkpoints.run_id
                WHERE baseline_runs.experiment_id=?
                ORDER BY baseline_checkpoints.checkpoint_id
                """,
                (experiment_id,),
            ).fetchall()
        return tuple(BaselineCheckpointManifest.model_validate_json(row[0]) for row in rows)

    def list_baseline_evaluations(
        self, experiment_id: str
    ) -> tuple[BaselineEvaluationReport, ...]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT baseline_evaluations.manifest_json
                FROM baseline_evaluations
                JOIN baseline_runs ON baseline_runs.run_id=baseline_evaluations.run_id
                WHERE baseline_runs.experiment_id=?
                ORDER BY baseline_evaluations.evaluation_id
                """,
                (experiment_id,),
            ).fetchall()
        return tuple(BaselineEvaluationReport.model_validate_json(row[0]) for row in rows)

    def get_baseline_acceptance(
        self, experiment_id: str
    ) -> BaselineAcceptanceReport | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT manifest_json FROM baseline_acceptances WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
        return BaselineAcceptanceReport.model_validate_json(row[0]) if row else None

    def validate_baseline_acceptance(
        self,
        manifest: BaseModel | Mapping[str, Any],
        *,
        corpus_manifest_hash: str,
    ) -> dict[str, Any]:
        """Verify acceptance evidence against registry rows, never a hand-set readiness flag."""

        payload, manifest_hash = _baseline_payload(manifest, "baseline_acceptance")
        reasons: list[str] = []
        validated = _value(payload, "baselineValidated", "baseline_validated")
        if validated is None:
            validated = _value(payload, "accepted")
        if validated is not True:
            reasons.append("accepted/baselineValidated is not true")
        acceptance_corpus_hash = _value(
            payload, "corpusManifestHash", "corpus_manifest_hash"
        ) or _value(payload, "corpusHash", "corpus_hash")
        if acceptance_corpus_hash != corpus_manifest_hash:
            reasons.append("acceptance corpus hash does not match the checked corpus")
        if _value(payload, "completedLearningRuns", "completed_learning_runs") != 12:
            reasons.append("completedLearningRuns must equal 12")
        if _value(payload, "completedHeuristicRuns", "completed_heuristic_runs") != 6:
            reasons.append("completedHeuristicRuns must equal 6")
        failures = _value(payload, "failures")
        if failures not in ([], (), None):
            reasons.append("acceptance contains failures")

        gates = _value(payload, "gates")
        required_gates = {
            "corpus_ready",
            "config_frozen",
            "formal_matrix_complete",
            "heuristic_matrix_complete",
            "metrics_complete",
            "cuda_memory_within_limit",
            "official_graphsage_validation_threshold",
            "official_graphsage_test_threshold",
            "official_graphsage_gain_over_mlp",
            "strict_edge_time_audit_passed",
            "test_read_after_selection",
            "checkpoint_recovery_verified",
        }
        if (
            not isinstance(gates, dict)
            or set(gates) != required_gates
            or not all(value is True for value in gates.values())
        ):
            reasons.append("all acceptance gates must be present and true")

        run_hashes = _value(payload, "runManifestHashes", "run_manifest_hashes")
        checkpoint_hashes = _value(
            payload, "checkpointManifestHashes", "checkpoint_manifest_hashes"
        )
        evaluation_hashes = _value(
            payload, "evaluationManifestHashes", "evaluation_manifest_hashes"
        )
        explicit_references = any(
            value is not None for value in (run_hashes, checkpoint_hashes, evaluation_hashes)
        )
        if explicit_references and (
            not isinstance(run_hashes, (list, tuple)) or len(run_hashes) < 12
        ):
            reasons.append("acceptance requires at least 12 run manifest hashes")
            run_hashes = ()
        if explicit_references and (
            not isinstance(checkpoint_hashes, (list, tuple)) or len(checkpoint_hashes) < 12
        ):
            reasons.append("acceptance requires at least 12 checkpoint manifest hashes")
            checkpoint_hashes = ()
        if explicit_references and (
            not isinstance(evaluation_hashes, (list, tuple)) or len(evaluation_hashes) < 12
        ):
            reasons.append("acceptance requires at least 12 evaluation manifest hashes")
            evaluation_hashes = ()

        with self.connect() as connection:
            stored = connection.execute(
                """
                SELECT baseline_validated, registrable, manifest_json
                FROM baseline_acceptances WHERE manifest_hash=?
                """,
                (manifest_hash,),
            ).fetchone()
            if stored is None or stored[:2] != (1, 0):
                reasons.append("acceptance manifest is not registered as validated/non-registrable")
            elif stored[2] != canonical_json(payload):
                reasons.append("registered acceptance payload differs from the checked report")
            experiment_id = _value(payload, "experimentId", "experiment_id")
            if not explicit_references:
                run_rows = connection.execute(
                    "SELECT manifest_hash, status, model_kind, registrable, manifest_json "
                    "FROM baseline_runs WHERE experiment_id=?",
                    (experiment_id,),
                ).fetchall()
                learning = [
                    row for row in run_rows if row[1] == "succeeded" and row[2] in ("mlp", "graphsage")
                ]
                heuristic = [
                    row for row in run_rows if row[1] == "succeeded" and row[2] in ("cn", "aa", "ra")
                ]
                if len(learning) != 12:
                    reasons.append("registry must contain exactly 12 succeeded learning runs")
                if len(heuristic) != 6:
                    reasons.append("registry must contain exactly 6 succeeded heuristic runs")
                if any(row[3] != 0 for row in run_rows):
                    reasons.append("one or more baseline runs are registrable")
                run_manifests: list[dict[str, Any]] = []
                for row in run_rows:
                    try:
                        checked, checked_hash = _baseline_payload(
                            json.loads(row[4]), "baseline_run"
                        )
                        if checked_hash != row[0]:
                            reasons.append("a baseline run registry hash is stale")
                        run_manifests.append(checked)
                    except (json.JSONDecodeError, RegistrationRejected):
                        reasons.append("a baseline run registry payload is invalid")
                expected_learning = {
                    (track, model, seed)
                    for track in ("ogb_official", "strict_edge_time")
                    for model in ("mlp", "graphsage")
                    for seed in (20260812, 20260813, 20260814)
                }
                actual_learning = {
                    (
                        item.get("track"),
                        item.get("model"),
                        item.get("seed"),
                    )
                    for item in run_manifests
                    if item.get("phase") == "formal"
                    and item.get("runKind") == "baseline"
                    and item.get("status") == "succeeded"
                    and item.get("model") in ("mlp", "graphsage")
                }
                expected_heuristic = {
                    (track, model)
                    for track in ("ogb_official", "strict_edge_time")
                    for model in ("cn", "aa", "ra")
                }
                actual_heuristic = {
                    (item.get("track"), item.get("model"))
                    for item in run_manifests
                    if item.get("phase") == "formal"
                    and item.get("runKind") == "baseline"
                    and item.get("status") == "succeeded"
                    and item.get("model") in ("cn", "aa", "ra")
                }
                if actual_learning != expected_learning:
                    reasons.append("formal learning run matrix is incomplete or contaminated")
                if actual_heuristic != expected_heuristic:
                    reasons.append("formal heuristic run matrix is incomplete or contaminated")
                acceptance_config_hash = _value(payload, "configHash", "config_hash")
                if any(
                    item.get("configHash") != acceptance_config_hash
                    or item.get("corpusHash") != corpus_manifest_hash
                    for item in run_manifests
                ):
                    reasons.append("formal runs do not share the accepted config/corpus hashes")
                if len({item.get("codeHash") for item in run_manifests}) != 1:
                    reasons.append("formal runs do not share one code hash")
                if len({item.get("environmentHash") for item in run_manifests}) != 1:
                    reasons.append("formal runs do not share one environment hash")
                if any(
                    float(item.get("peakCudaMemoryMiB", float("inf"))) >= 7168.0
                    for item in run_manifests
                ):
                    reasons.append("one or more formal runs reached the 7168 MiB CUDA limit")
                try:
                    reported_peak = float(_value(payload, "peakCudaMemoryMiB", "peak_cuda_memory_mib"))
                    actual_peak = max(
                        (float(item.get("peakCudaMemoryMiB", 0.0)) for item in run_manifests),
                        default=0.0,
                    )
                    if reported_peak >= 7168.0 or reported_peak + 1e-6 < actual_peak:
                        reasons.append("acceptance CUDA peak is inconsistent or above the limit")
                except (TypeError, ValueError):
                    reasons.append("acceptance CUDA peak is invalid")
                run_ids = [
                    row[0]
                    for row in connection.execute(
                        "SELECT run_id FROM baseline_runs WHERE experiment_id=?",
                        (experiment_id,),
                    ).fetchall()
                ]
                if run_ids:
                    placeholders = ",".join("?" for _ in run_ids)
                    checkpoint_count = int(
                        connection.execute(
                            f"SELECT COUNT(*) FROM baseline_checkpoints "
                            f"WHERE run_id IN ({placeholders})",
                            tuple(run_ids),
                        ).fetchone()[0]
                    )
                    evaluation_count = int(
                        connection.execute(
                            f"SELECT COUNT(*) FROM baseline_evaluations "
                            f"WHERE run_id IN ({placeholders})",
                            tuple(run_ids),
                        ).fetchone()[0]
                    )
                    if checkpoint_count < 12:
                        reasons.append("registry must contain checkpoints for all learning runs")
                    if evaluation_count < 18:
                        reasons.append("registry must contain evaluations for all formal runs")
                    checkpoint_rows = connection.execute(
                        f"SELECT manifest_hash, registrable, manifest_json "
                        f"FROM baseline_checkpoints "
                        f"WHERE run_id IN ({placeholders})",
                        tuple(run_ids),
                    ).fetchall()
                    if any(row[1] != 0 for row in checkpoint_rows):
                        reasons.append("one or more baseline checkpoints are registrable")
                    checkpoint_manifests: list[dict[str, Any]] = []
                    for row in checkpoint_rows:
                        try:
                            checked, checked_hash = _baseline_payload(
                                json.loads(row[2]), "baseline_checkpoint"
                            )
                            if checked_hash != row[0]:
                                reasons.append("a baseline checkpoint registry hash is stale")
                            checkpoint_manifests.append(checked)
                        except (json.JSONDecodeError, RegistrationRejected):
                            reasons.append("a baseline checkpoint registry payload is invalid")
                    if any(
                        item.get("configHash") != acceptance_config_hash
                        or item.get("corpusHash") != corpus_manifest_hash
                        for item in checkpoint_manifests
                    ):
                        reasons.append(
                            "baseline checkpoints do not share the accepted config/corpus hashes"
                        )
                    official_graphsage_ids = {
                        item.get("runId")
                        for item in run_manifests
                        if item.get("track") == "ogb_official"
                        and item.get("model") == "graphsage"
                    }
                    verified_official_graphsage = {
                        item.get("runId")
                        for item in checkpoint_manifests
                        if item.get("runId") in official_graphsage_ids
                        and isinstance(item.get("verificationDigest"), str)
                        and len(item["verificationDigest"]) == 64
                    }
                    if verified_official_graphsage != official_graphsage_ids:
                        reasons.append(
                            "all official GraphSAGE checkpoints require fresh-process verification"
                        )
                    evaluation_rows = connection.execute(
                        f"SELECT manifest_hash, manifest_json FROM baseline_evaluations "
                        f"WHERE run_id IN ({placeholders})",
                        tuple(run_ids),
                    ).fetchall()
                    evaluations: list[dict[str, Any]] = []
                    for row in evaluation_rows:
                        try:
                            checked, checked_hash = _baseline_payload(
                                json.loads(row[1]), "baseline_evaluation"
                            )
                            if checked_hash != row[0]:
                                reasons.append("a baseline evaluation registry hash is stale")
                            evaluations.append(checked)
                        except (json.JSONDecodeError, RegistrationRejected):
                            reasons.append("a baseline evaluation registry payload is invalid")
                    run_by_id = {item.get("runId"): item for item in run_manifests}
                    if any(
                        item.get("experimentId") != experiment_id
                        or item.get("runId") not in run_by_id
                        or item.get("phase") != run_by_id[item.get("runId")].get("phase")
                        or item.get("track") != run_by_id[item.get("runId")].get("track")
                        or item.get("model") != run_by_id[item.get("runId")].get("model")
                        or item.get("seed") != run_by_id[item.get("runId")].get("seed")
                        for item in evaluations
                    ):
                        reasons.append("baseline evaluations are not bound to their registered runs")
                    if any(
                        not isinstance(item.get("testMetrics"), dict)
                        or item.get("testReadAfterSelection") is not True
                        for item in evaluations
                    ):
                        reasons.append(
                            "all formal evaluations require test metrics read after selection"
                        )
                    official_graphsage = [
                        item
                        for item in evaluations
                        if item.get("phase") == "formal"
                        and item.get("track") == "ogb_official"
                        and item.get("model") == "graphsage"
                    ]
                    official_mlp = [
                        item
                        for item in evaluations
                        if item.get("phase") == "formal"
                        and item.get("track") == "ogb_official"
                        and item.get("model") == "mlp"
                    ]
                    if len(official_graphsage) != 3 or len(official_mlp) != 3:
                        reasons.append("official learning evaluations require exactly three seeds")
                    else:
                        try:
                            graph_val = sum(
                                item["validationMetrics"]["hits@50"]
                                for item in official_graphsage
                            ) / 3
                            graph_test = sum(
                                item["testMetrics"]["hits@50"]
                                for item in official_graphsage
                            ) / 3
                            mlp_test = sum(
                                item["testMetrics"]["hits@50"] for item in official_mlp
                            ) / 3
                            if graph_val < 0.40:
                                reasons.append("official GraphSAGE mean validation Hits@50 is below 0.40")
                            if graph_test < 0.35:
                                reasons.append("official GraphSAGE mean test Hits@50 is below 0.35")
                            if graph_test - mlp_test < 0.05:
                                reasons.append(
                                    "official GraphSAGE mean test Hits@50 advantage over MLP is below 0.05"
                                )
                        except (KeyError, TypeError, ValueError, ZeroDivisionError):
                            reasons.append("official learning evaluations have incomplete Hits@50")
                    strict_evaluations = [
                        item
                        for item in evaluations
                        if item.get("phase") == "formal"
                        and item.get("track") == "strict_edge_time"
                    ]
                    if len(strict_evaluations) != 9:
                        reasons.append("strict track requires all nine formal evaluations")
                    elif any(
                        set(item.get("strata", {})) != {"first_time", "repeated"}
                        or "count" not in item["strata"]["first_time"]
                        or "count" not in item["strata"]["repeated"]
                        for item in strict_evaluations
                    ):
                        reasons.append("strict evaluations require first_time/repeated strata")
                else:
                    reasons.append("registry contains no runs for the accepted experiment")
            for table, hashes in (
                ("baseline_runs", run_hashes),
                ("baseline_checkpoints", checkpoint_hashes),
                ("baseline_evaluations", evaluation_hashes),
            ):
                if hashes:
                    placeholders = ",".join("?" for _ in hashes)
                    found = {
                        row[0]
                        for row in connection.execute(
                            f"SELECT manifest_hash FROM {table} "
                            f"WHERE manifest_hash IN ({placeholders})",
                            tuple(hashes),
                        ).fetchall()
                    }
                    missing = sorted(set(hashes) - found)
                    if missing:
                        reasons.append(f"{table} is missing {len(missing)} referenced manifests")
            if checkpoint_hashes:
                placeholders = ",".join("?" for _ in checkpoint_hashes)
                promotable = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM baseline_checkpoints "
                        f"WHERE manifest_hash IN ({placeholders}) AND registrable != 0",
                        tuple(checkpoint_hashes),
                    ).fetchone()[0]
                )
                if promotable:
                    reasons.append("one or more baseline checkpoints are registrable")
        return {
            "ready": not reasons,
            "manifestHash": manifest_hash,
            "experimentId": _value(payload, "experimentId", "experiment_id"),
            "reasons": reasons,
        }

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("runs", "checkpoints", "models")
            }

    def successful_smoke_coverage(
        self,
        *,
        code_hash: str,
        environment_hash: str,
        config_hashes: dict[str, str],
        device: str,
    ) -> set[str]:
        coverage: set[str] = set()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT runs.manifest_json, checkpoints.manifest_json
                FROM runs JOIN checkpoints ON checkpoints.run_id = runs.run_id
                WHERE runs.status='succeeded' AND runs.run_kind='smoke'
                  AND checkpoints.smoke_only=1
                """
            ).fetchall()
        for raw_manifest, raw_checkpoint in rows:
            manifest = json.loads(raw_manifest)
            checkpoint = json.loads(raw_checkpoint)
            if manifest.get("codeHash") != code_hash:
                continue
            if manifest.get("environmentHash") != environment_hash:
                continue
            corpus_id = manifest.get("corpus", {}).get("corpusId", "")
            if corpus_id == "synthetic-actor":
                fixture = "actor"
            elif corpus_id == "synthetic-hetero":
                fixture = "hetero"
            else:
                continue
            expected_config = config_hashes.get(fixture)
            if manifest.get("configHash") != expected_config:
                continue
            if checkpoint.get("configHash") != expected_config:
                continue
            if checkpoint.get("runId") != manifest.get("runId"):
                continue
            metrics = manifest.get("smokeMetrics") or {}
            if metrics.get("device") != device:
                continue
            if metrics.get("freshProcessVerified") is not True:
                continue
            if metrics.get("optimizerRestored") is not True:
                continue
            try:
                if float(metrics.get("elapsedSeconds", float("inf"))) > 120.0:
                    continue
                if float(metrics.get("maxMemoryMb", float("inf"))) > 4096.0:
                    continue
            except (TypeError, ValueError):
                continue
            if metrics.get("checkpointStateHash") != checkpoint.get("stateHash"):
                continue
            if metrics.get("checkpointArtifactSha256") != checkpoint.get("artifactSha256"):
                continue
            try:
                checkpoint_model = SmokeCheckpointManifest.model_validate(checkpoint)
                artifact = Path(checkpoint_model.artifact_path)
                if not artifact.is_file():
                    continue
                if file_sha256(artifact) != checkpoint_model.artifact_sha256:
                    continue
                manifest_paths = {str(Path(path).resolve()) for path in manifest.get("artifacts", ())}
                expected_manifest = str(
                    artifact.with_name(f"{checkpoint_model.checkpoint_id}.manifest.json").resolve()
                )
                if expected_manifest not in manifest_paths or not Path(expected_manifest).is_file():
                    continue
                load_checkpoint(checkpoint_model, map_location="cpu")
            except (KeyError, OSError, TypeError, ValueError, RuntimeError):
                continue
            coverage.add(fixture)
        return coverage
