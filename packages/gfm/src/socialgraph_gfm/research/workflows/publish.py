"""Immutable SocialGraph-FM Research registry publication and stage readiness inspection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from socialgraph_gfm.canonical import canonical_sha256, file_sha256

from ..contracts import (
    ACCOUNT_RISK_TASK,
    COLLABORATION_TASK,
    CONTENT_POLICY_TASK,
    RELEASE_ID,
    RESEARCH_SEED,
    SIGNED_RELATION_TASK,
)
from .common import REGISTRY_SCHEMA, SMOKE_SCHEMA, _atomic_json, _read_hashed_document, _safe_root
from .materialize import _require_publishable_corpus
from .serve import load_export_manifest


def _registry_export_projection(export: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "modelVersionId",
        "modelVersionHash",
        "artifactHash",
        "corpusKind",
        "testOnly",
        "checkpointSha256",
        "modelCardPath",
        "modelCardSha256",
        "modelCardHash",
        "researchConfigSha256",
        "taskIds",
        "graphSchemaVersion",
        "maxNodes",
        "maxEdges",
        "claimStatus",
        "calibrationStatus",
        "taskCalibrationStatus",
        "calibrators",
        "graphVersionHashes",
        "splitHashes",
        "visibleTopologyHashes",
        "adapterSchemaHashes",
        "featureContractHashes",
        "routeContract",
        "routeContractHash",
        "parserContractHash",
        "scenarios",
        "embeddings",
    )
    return {field: export[field] for field in fields}


def _build_registry_payload(
    root: Path, export: Mapping[str, Any], *, smoke_hash: str
) -> dict[str, Any]:
    export_projection = _registry_export_projection(export)
    registry: dict[str, Any] = {
        "schemaVersion": REGISTRY_SCHEMA,
        "releaseId": RELEASE_ID,
        "releaseLabel": "SocialGraph-FM Research",
        "seed": RESEARCH_SEED,
        "preliminary": True,
        "formalReadinessUnaffected": True,
        "modelVersionId": export["modelVersionId"],
        "modelVersionHash": export["modelVersionHash"],
        "artifactHash": export["artifactHash"],
        "corpusKind": export["corpusKind"],
        "testOnly": export["testOnly"],
        "exportManifestPath": "exports/research/export-manifest.json",
        "exportManifestSha256": file_sha256(root / "exports/research/export-manifest.json"),
        "checkpointSha256": export["checkpointSha256"],
        "modelCardPath": export["modelCardPath"],
        "modelCardSha256": export["modelCardSha256"],
        "modelCardHash": export["modelCardHash"],
        "researchConfigSha256": export["researchConfigSha256"],
        "taskIds": export["taskIds"],
        "graphSchemaVersion": export["graphSchemaVersion"],
        "maxNodes": export["maxNodes"],
        "maxEdges": export["maxEdges"],
        "claimStatus": export["claimStatus"],
        "calibrationStatus": export["calibrationStatus"],
        "taskCalibrationStatus": export["taskCalibrationStatus"],
        "calibrators": export["calibrators"],
        "graphVersionHashes": export["graphVersionHashes"],
        "splitHashes": export["splitHashes"],
        "visibleTopologyHashes": export["visibleTopologyHashes"],
        "adapterSchemaHashes": export["adapterSchemaHashes"],
        "featureContractHashes": export["featureContractHashes"],
        "routeContract": export["routeContract"],
        "routeContractHash": export["routeContractHash"],
        "parserContractHash": export["parserContractHash"],
        "scenarios": export["scenarios"],
        "embeddings": export["embeddings"],
        "exportProjection": export_projection,
        "exportProjectionHash": canonical_sha256(export_projection),
        "smokeHash": smoke_hash,
    }
    registry["registryHash"] = canonical_sha256(registry)
    return registry


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _validate_fresh_http_smoke(
    smoke: Mapping[str, Any], export: Mapping[str, Any], root: Path
) -> None:
    if (
        smoke.get("passed") is not True
        or smoke.get("protocol") != "fresh-inference-cli-http/1.0"
        or smoke.get("modelVersionId") != export["modelVersionId"]
        or smoke.get("modelVersionHash") != export["modelVersionHash"]
        or smoke.get("artifactHash") != export["artifactHash"]
        or smoke.get("corpusKind") != export["corpusKind"]
        or smoke.get("testOnly") is not export["testOnly"]
        or not _is_sha256(smoke.get("candidateRegistryHash"))
        or smoke.get("checkpoint")
        != {
            "relativePath": "exports/research/checkpoint.pt",
            "sha256": export["checkpointSha256"],
        }
        or file_sha256(root / "exports/research/checkpoint.pt")
        != export["checkpointSha256"]
    ):
        raise ValueError("research smoke report does not bind the selected export")
    fresh = smoke.get("freshProcess")
    if not isinstance(fresh, dict):
        raise ValueError(  # noqa: TRY004 - persisted artifact validation is a value error
            "research smoke report lacks fresh inference_cli process evidence"
        )
    command = fresh.get("command")
    if (
        not isinstance(command, list)
        or not all(isinstance(item, str) for item in command)
        or len(command) < 10
        or command[1:3] != ["-m", "socialgraph_gfm.core.inference_cli"]
        or fresh.get("commandHash") != canonical_sha256(command)
        or fresh.get("host") != "127.0.0.1"
        or not isinstance(fresh.get("port"), int)
        or not 1 <= fresh["port"] <= 65_535
        or not isinstance(fresh.get("pid"), int)
        or fresh["pid"] < 1
        or fresh.get("terminationMode") != "terminate"
        or not isinstance(fresh.get("exitCode"), int)
        or not _is_sha256(fresh.get("stdoutSha256"))
        or not _is_sha256(fresh.get("stderrSha256"))
        or not _is_sha256(fresh.get("pythonExecutableSha256"))
        or any(flag not in command for flag in ("--runtime-root", "--research-root", "--token-file"))
    ):
        raise ValueError("research smoke report has invalid fresh inference_cli process evidence")
    executable = Path(str(fresh.get("pythonExecutable", "")))
    if (
        not executable.is_file()
        or file_sha256(executable) != fresh["pythonExecutableSha256"]
        or Path(command[0]).resolve() != executable.resolve()
    ):
        raise ValueError("research smoke Python executable identity changed")
    evidence = smoke.get("httpEvidence")
    expected_endpoints = [
        "GET /internal/research/capabilities",
        "GET /internal/research/scenarios",
        "POST /internal/research/runs",
        "GET /internal/research/runs/{runId}",
        "GET /internal/research/runs/{runId}/result",
        "POST /internal/research/similar-nodes",
    ]
    expected_scenarios = {
        "twitch-content-policy": CONTENT_POLICY_TASK,
        "tolokers-account-risk": ACCOUNT_RISK_TASK,
        "wiki-rfa-signed-relation": SIGNED_RELATION_TASK,
        "email-eu-collaboration": COLLABORATION_TASK,
    }
    if (
        not isinstance(evidence, dict)
        or evidence.get("endpointInventory") != expected_endpoints
        or evidence.get("allRequestsAuthenticated") is not True
        or evidence.get("loopbackOnly") is not True
        or not _is_sha256(evidence.get("capabilityHash"))
        or not _is_sha256(evidence.get("scenariosHash"))
    ):
        raise ValueError("research smoke report lacks complete loopback HTTP evidence")
    scenarios = evidence.get("scenarioResults")
    if not isinstance(scenarios, list) or len(scenarios) != len(expected_scenarios):
        raise ValueError("research smoke report lacks four scenario results")
    observed: dict[str, str] = {}
    for item in scenarios:
        if not isinstance(item, dict):
            raise ValueError(  # noqa: TRY004 - persisted artifact validation is a value error
                "research smoke scenario evidence is invalid"
            )
        scenario_id = item.get("scenarioId")
        task_id = item.get("taskId")
        if (
            not isinstance(scenario_id, str)
            or not isinstance(task_id, str)
            or scenario_id in observed
            or expected_scenarios.get(scenario_id) != task_id
            or item.get("repeatDeterministic") is not True
            or not isinstance(item.get("findingCount"), int)
            or item["findingCount"] < 1
            or not isinstance(item.get("runId"), str)
            or not _is_sha256(item.get("requestHash"))
            or not _is_sha256(item.get("stateHash"))
            or not _is_sha256(item.get("resultHash"))
        ):
            raise ValueError("research smoke scenario evidence is incomplete")
        observed[scenario_id] = task_id
    if observed != expected_scenarios:
        raise ValueError("research smoke scenario inventory is not canonical")
    similar = evidence.get("similarNodes")
    if (
        not isinstance(similar, dict)
        or similar.get("repeatDeterministic") is not True
        or not isinstance(similar.get("graphVersionId"), str)
        or not isinstance(similar.get("nodeId"), str)
        or not isinstance(similar.get("matchCount"), int)
        or similar["matchCount"] < 1
        or not _is_sha256(similar.get("resultHash"))
    ):
        raise ValueError("research smoke similar-nodes evidence is incomplete")


def publish_research_model(
    research_root: str | Path, *, allow_test_fixture: bool = False
) -> Path:
    root = _safe_root(research_root)
    export = load_export_manifest(root)
    _require_publishable_corpus(
        export, allow_test_fixture=allow_test_fixture, stage="publish"
    )
    smoke = _read_hashed_document(
        root / "exports/research/smoke-report.json",
        schema=SMOKE_SCHEMA,
        hash_field="smokeHash",
    )
    _validate_fresh_http_smoke(smoke, export, root)
    registry = _build_registry_payload(root, export, smoke_hash=smoke["smokeHash"])
    path = root / "published/registry.json"
    if path.exists():
        existing = _read_hashed_document(path, schema=REGISTRY_SCHEMA, hash_field="registryHash")
        if existing != registry:
            raise FileExistsError("a different SocialGraph-FM Research registry is already published")
        return path
    _atomic_json(path, registry)
    return path


def load_registry(research_root: str | Path) -> dict[str, Any]:
    root = _safe_root(research_root)
    registry = _read_hashed_document(
        root / "published/registry.json",
        schema=REGISTRY_SCHEMA,
        hash_field="registryHash",
    )
    manifest = root / registry["exportManifestPath"]
    if file_sha256(manifest) != registry["exportManifestSha256"]:
        raise ValueError("published SocialGraph-FM Research export identity changed")
    export = load_export_manifest(root)
    projection = _registry_export_projection(export)
    if (
        registry.get("exportProjection") != projection
        or registry.get("exportProjectionHash") != canonical_sha256(projection)
        or _registry_export_projection(registry) != projection
        or export["modelVersionHash"] != registry["modelVersionHash"]
        or export["artifactHash"] != registry["artifactHash"]
    ):
        raise ValueError("published SocialGraph-FM Research registry is stale")
    return registry


def stage_paths(research_root: str | Path) -> dict[str, Path]:
    root = _safe_root(research_root)
    return {
        "corpus": root / "materialized/corpus/corpus-manifest.json",
        "training": root / "runs/shared/training-manifest.json",
        "evaluation": root / "reports/evaluation.json",
        "export": root / "exports/research/export-manifest.json",
        "smoke": root / "exports/research/smoke-report.json",
        "registry": root / "published/registry.json",
    }


def readiness(research_root: str | Path) -> dict[str, bool]:
    paths = stage_paths(research_root)
    stages = ("corpus", "training", "evaluation", "export", "smoke", "registry")
    observed: dict[str, bool] = {}
    prior = True
    for stage in stages:
        present = paths[stage].is_file()
        observed[stage] = prior and present
        prior = observed[stage]
    return observed

COMPAT_EXPORTS = (
    '_registry_export_projection',
    '_build_registry_payload',
    '_is_sha256',
    '_validate_fresh_http_smoke',
    'publish_research_model',
    'load_registry',
    'stage_paths',
    'readiness',
)

__all__ = [
    'load_registry',
    'publish_research_model',
    'readiness',
    'stage_paths',
]
