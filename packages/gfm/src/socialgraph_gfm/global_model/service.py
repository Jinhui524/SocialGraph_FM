"""Hash-bound loopback serving for frozen SocialGraph-FM Global Russia predictions."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
import zipfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from socialgraph_gfm.canonical import canonical_sha256, file_sha256

from .contracts import COUNTRY_IDS, read_corpus_manifest
from .corpus import GlobalCountryCorpus, load_country_corpus

SCHEMA_VERSION = "socialgraph-fm.gfm-global-model/1.0"
HEALTH_SCHEMA_VERSION = "socialgraph-fm.global-model-health/1.0"
MODEL_CARD_SCHEMA_VERSION = "socialgraph-fm.global-model-card/1.0"
REGISTRY_SCHEMA_VERSION = "socialgraph-fm.global-model-registry/1.0"
PROTOCOLS = ("in_domain", "low_label", "cross_domain", "global")
MODALITIES = ("coRT", "coURL", "hashSeq", "fastRT", "tweetSim")
LIMITATIONS = (
    "Anonymous research identifiers only; no real-world identity claim.",
    "Scores are review candidates and never automatic enforcement decisions.",
    "The static snapshot supports transductive ranking, not future-event prediction.",
)
_RUN_ID = re.compile(r"^global-model-[0-9a-f]{32}$")
_NODE_ID = re.compile(r"^(?:russia:)?[0-9]{1,20}$")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_NPZ_BYTES = 512 * 1024 * 1024
_MODEL_CARD_KEYS = {
    "schemaVersion",
    "releaseId",
    "modelVersionId",
    "modelVersionHash",
    "taskId",
    "architecture",
    "protocols",
    "trainingData",
    "intendedUse",
    "outOfScope",
    "limitations",
    "ethics",
    "licenses",
    "sourceAttribution",
    "metrics",
    "artifactHash",
    "modelCardHash",
}
_MODEL_CARD_ARCHITECTURE_KEYS = {
    "name",
    "textFeatures",
    "structuralFeatures",
    "gnnLayers",
    "hiddenDim",
    "router",
}
_MODEL_CARD_PROTOCOL_KEYS = {
    "modelVersionId",
    "modelVersionHash",
    "modelStateHash",
    "state",
}


class GlobalServiceError(Exception):
    status = 409
    code = "GFM_GLOBAL_MODEL_CONFLICT"


class GlobalUnavailable(GlobalServiceError):
    status = 503
    code = "GFM_GLOBAL_MODEL_NOT_INSTALLED"


class GlobalNotFound(GlobalServiceError):
    status = 404
    code = "GFM_GLOBAL_MODEL_NOT_FOUND"


class GlobalInvalid(GlobalServiceError):
    status = 422
    code = "GFM_GLOBAL_MODEL_REQUEST_INVALID"


class GlobalResultNotReady(GlobalServiceError):
    status = 409
    code = "GFM_GLOBAL_MODEL_RESULT_NOT_READY"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bounded_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"invalid Global JSON artifact: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(  # noqa: TRY004 - malformed persisted state is a contract error
            f"Global JSON artifact must be an object: {path.name}"
        )
    return payload


def _safe_file(root: Path, relative: str, expected_hash: str | None = None) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("artifact path must be a non-empty POSIX relative path")
    candidate = (root / relative).resolve(strict=True)
    if not candidate.is_relative_to(root.resolve(strict=True)) or candidate.is_symlink():
        raise ValueError("artifact path escapes Global root")
    if expected_hash is not None and file_sha256(candidate) != expected_hash:
        raise ValueError("artifact hash mismatch")
    return candidate


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if path.stat().st_size > _MAX_NPZ_BYTES:
        raise ValueError("Global result NPZ is oversized")
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > _MAX_NPZ_BYTES:
            raise ValueError("Global result NPZ expands beyond its bound")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("Global result NPZ contains an object array")
    required = {
        "node_ids",
        "scores",
        "logits",
        "structure_missing",
        "router_indices",
        "router_weights",
        "modality_counts",
    }
    if not required.issubset(arrays):
        raise ValueError("Global result NPZ is missing serving arrays")
    count = arrays["node_ids"].shape[0]
    if (
        arrays["node_ids"].ndim != 1
        or arrays["scores"].shape != (count,)
        or arrays["logits"].shape != (count,)
        or arrays["structure_missing"].shape != (count,)
        or arrays["router_indices"].shape != (count, 2)
        or arrays["router_weights"].shape != (count, 2)
        or arrays["modality_counts"].shape != (count, 5)
    ):
        raise ValueError("Global result NPZ array shapes disagree")
    if count != 716 or len(np.unique(arrays["node_ids"])) != count:
        raise ValueError("Russia result must contain exactly 716 unique nodes")
    if not np.all(np.isfinite(arrays["scores"])) or not np.all(
        (arrays["scores"] >= 0.0) & (arrays["scores"] <= 1.0)
    ):
        raise ValueError("Global calibrated scores are invalid")
    return arrays


def _hash_payload(payload: dict[str, Any], field: str) -> dict[str, Any]:
    payload[field] = canonical_sha256(payload)
    return payload


def _verify_document_hash(payload: Mapping[str, Any], field: str) -> None:
    observed = payload.get(field)
    logical = {key: value for key, value in payload.items() if key != field}
    if not isinstance(observed, str) or observed != canonical_sha256(logical):
        raise ValueError(f"Global {field} mismatch")


def _validate_model_card_contract(model_card: Mapping[str, Any]) -> dict[str, Any]:
    if set(model_card) != _MODEL_CARD_KEYS:
        raise ValueError("Global model card has an unexpected shape")
    architecture = model_card.get("architecture")
    if (
        not isinstance(architecture, dict)
        or set(architecture) != _MODEL_CARD_ARCHITECTURE_KEYS
        or architecture.get("gnnLayers") != 2
        or architecture.get("hiddenDim") != 256
        or any(
            not isinstance(architecture.get(field), str) or not architecture[field]
            for field in ("name", "textFeatures", "structuralFeatures", "router")
        )
    ):
        raise ValueError("Global model card architecture is invalid")
    protocols = model_card.get("protocols")
    if not isinstance(protocols, dict) or set(protocols) != set(PROTOCOLS):
        raise ValueError("Global model card protocol inventory is invalid")
    identities: set[tuple[str, str]] = set()
    for protocol in PROTOCOLS:
        item = protocols[protocol]
        expected_state = "servingReady" if protocol == "global" else "frozenDemo"
        if (
            not isinstance(item, dict)
            or set(item) != _MODEL_CARD_PROTOCOL_KEYS
            or item.get("state") != expected_state
            or any(
                not isinstance(item.get(field), str) or not item[field]
                for field in ("modelVersionId", "modelVersionHash", "modelStateHash")
            )
        ):
            raise ValueError(f"Global {protocol} model card identity is invalid")
        identities.add((item["modelVersionId"], item["modelVersionHash"]))
    if len(identities) != len(PROTOCOLS):
        raise ValueError("Global model card protocol identities are not unique")
    training_data = model_card.get("trainingData")
    if (
        not isinstance(training_data, dict)
        or set(training_data) != {
            "countries",
            "nodeCount",
            "nodeCountByCountry",
            "content",
        }
        or training_data.get("countries") != list(COUNTRY_IDS)
        or not isinstance(training_data.get("nodeCountByCountry"), dict)
        or set(training_data["nodeCountByCountry"]) != set(COUNTRY_IDS)
        or not isinstance(training_data.get("nodeCount"), int)
        or isinstance(training_data.get("nodeCount"), bool)
        or not isinstance(training_data.get("content"), str)
        or not training_data["content"]
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in training_data["nodeCountByCountry"].values()
        )
        or sum(training_data["nodeCountByCountry"].values())
        != training_data["nodeCount"]
    ):
        raise ValueError("Global model card training data is invalid")
    for field in ("intendedUse", "outOfScope", "limitations", "ethics"):
        value = model_card.get(field)
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item for item in value
        ):
            raise ValueError(f"Global model card {field} is invalid")
    licenses = model_card.get("licenses")
    if (
        not isinstance(licenses, list)
        or len(licenses) != 2
        or any(
            not isinstance(item, dict)
            or set(item) != {"name", "license", "url"}
            or any(not isinstance(item.get(field), str) or not item[field] for field in item)
            for item in licenses
        )
        or {item["license"] for item in licenses} != {"CC-BY-4.0", "MIT"}
    ):
        raise ValueError("Global model card license inventory is invalid")
    source_attribution = model_card.get("sourceAttribution")
    if (
        not isinstance(source_attribution, dict)
        or set(source_attribution) != {"kind", "paperUrl", "completeReproduction"}
        or source_attribution.get("kind") != "inspired"
        or not isinstance(source_attribution.get("paperUrl"), str)
        or not source_attribution["paperUrl"]
        or source_attribution.get("completeReproduction") is not False
    ):
        raise ValueError("Global model card architecture attribution is invalid")
    metrics = model_card.get("metrics")
    if (
        not isinstance(metrics, dict)
        or set(metrics) != set(PROTOCOLS)
        or any(not isinstance(metrics[protocol], dict) for protocol in PROTOCOLS)
    ):
        raise ValueError("Global model card metric inventory is invalid")
    return protocols


def _risk_band(score: float, threshold: float) -> str:
    review_floor = max(0.25, threshold * 0.6)
    return "high" if score >= threshold else "review" if score >= review_floor else "low"


def _service_identity(
    *,
    model_version_id: str | None,
    model_version_hash: str | None,
    corpus_hash: str | None,
) -> str:
    return canonical_sha256(
        {
            "service": "socialgraph-fm-gfm/global-model",
            "datasetVersionId": "socialgraph-fm:russia",
            "modelVersionId": model_version_id,
            "modelVersionHash": model_version_hash,
            "corpusHash": corpus_hash,
        }
    )


class GlobalServingRuntime:
    """Caches one fully verified immutable Global export per process."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.registry_path = self.root / "registry" / "socialgraph-global.json"
        self.run_root = self.root / "serving-runs"
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="global-model")
        self._registry: dict[str, Any] | None = None
        self._preview: dict[str, Any] | None = None
        self._model_card: dict[str, Any] | None = None
        self._russia: GlobalCountryCorpus | None = None
        self._results: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
        if self.registry_path.is_file():
            self._load_snapshot()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._lock:
            if self._russia is not None:
                self._russia.close()
                self._russia = None
            self._results.clear()

    def _load_snapshot(self) -> None:
        registry = _bounded_json(self.registry_path)
        if registry.get("schemaVersion") != REGISTRY_SCHEMA_VERSION:
            raise ValueError("Global registry schema is unsupported")
        _verify_document_hash(registry, "registryHash")
        if registry.get("state") != "servingReady":
            raise ValueError("Global registry is not servingReady")
        if tuple(registry.get("protocols", ())) != PROTOCOLS:
            raise ValueError("Global registry protocol inventory is invalid")
        for field in (
            "modelVersionId",
            "modelVersionHash",
            "artifactHash",
            "corpusHash",
            "sourceCodeHash",
            "graphVersionHash",
            "checkpointPath",
            "protocolArtifacts",
            "protocolModels",
            "russiaPreviewPath",
            "modelCardPath",
            "modelCardSha256",
        ):
            if field not in registry:
                raise ValueError(f"Global registry is missing {field}")
        _safe_file(
            self.root,
            registry["checkpointPath"],
            registry.get("checkpointSha256"),
        )
        preview_path = _safe_file(
            self.root,
            registry["russiaPreviewPath"],
            registry.get("russiaPreviewSha256"),
        )
        preview = _bounded_json(preview_path)
        if (
            preview.get("schemaVersion") != "socialgraph-fm.global-model-preview/1.0"
            or preview.get("graphVersionHash") != registry["graphVersionHash"]
        ):
            raise ValueError("Global Russia preview identity is invalid")
        _verify_document_hash(preview, "previewHash")
        model_card_path = _safe_file(
            self.root,
            registry["modelCardPath"],
            registry["modelCardSha256"],
        )
        model_card = _bounded_json(model_card_path)
        _verify_document_hash(model_card, "modelCardHash")
        if (
            model_card.get("schemaVersion") != MODEL_CARD_SCHEMA_VERSION
            or model_card.get("releaseId") != "socialgraph-fm"
            or model_card.get("taskId") != "coordination_risk"
            or model_card.get("modelVersionId") != registry["modelVersionId"]
            or model_card.get("modelVersionHash") != registry["modelVersionHash"]
            or model_card.get("artifactHash") != registry["artifactHash"]
        ):
            raise ValueError("Global model card is not bound to the Global model")
        card_protocols = _validate_model_card_contract(model_card)
        registry_protocols = registry["protocolModels"]
        if (
            not isinstance(registry_protocols, dict)
            or set(registry_protocols) != set(PROTOCOLS)
            or any(
                registry_protocols[protocol] != card_protocols[protocol]
                for protocol in PROTOCOLS
            )
        ):
            raise ValueError("Global registry/model card protocol identities disagree")

        corpus_manifest_path = _safe_file(self.root, "corpus/manifest.json")
        corpus_manifest = read_corpus_manifest(corpus_manifest_path)
        if corpus_manifest.content_hash != registry["corpusHash"]:
            raise ValueError("Global serving corpus identity is stale")
        russia_entry = next(
            (entry for entry in corpus_manifest.countries if entry.country_id == "russia"),
            None,
        )
        if russia_entry is None:
            raise ValueError("Global serving corpus has no Russia entry")
        country_manifest_path = _safe_file(
            self.root, f"corpus/{russia_entry.manifest_path}"
        )
        russia = load_country_corpus(
            country_manifest_path.parent,
            verify_hashes=True,
            verify_values=True,
            mmap_mode="r",
        )
        if (
            russia.manifest.content_hash != russia_entry.manifest_hash
            or russia.manifest.source_hashes != russia_entry.source_hashes
            or russia.manifest.split_hashes != russia_entry.split_hashes
            or russia.manifest.content_hash != registry["graphVersionHash"]
            or russia.manifest.node_count != 716
        ):
            raise ValueError("Global Russia corpus binding is invalid")
        raw_edge_count = russia.manifest.edge_count // 2
        if (
            preview.get("nodeCount") != russia.manifest.node_count
            or preview.get("edgeCount") != raw_edge_count
        ):
            raise ValueError("Global preview counts do not match the safe Russia corpus")
        artifacts = registry["protocolArtifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != set(PROTOCOLS):
            raise ValueError("Global protocol artifact map is invalid")
        results: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
        for protocol in PROTOCOLS:
            entry = artifacts[protocol]
            if not isinstance(entry, dict):
                raise ValueError(  # noqa: TRY004 - invalid registry contract
                    "Global protocol artifact entry is invalid"
                )
            for field in (
                "protocolModelVersionId",
                "protocolModelVersionHash",
                "modelStateHash",
                "checkpointPath",
                "checkpointSha256",
            ):
                if field not in entry:
                    raise ValueError(f"Global protocol artifact is missing {field}")
            _safe_file(
                self.root,
                entry["checkpointPath"],
                entry["checkpointSha256"],
            )
            result_paths = entry.get("resultPaths")
            if not isinstance(result_paths, dict) or "russia" not in result_paths:
                raise ValueError("Global Russia result path is missing")
            result_ref = result_paths["russia"]
            if not isinstance(result_ref, dict):
                raise ValueError(  # noqa: TRY004 - invalid registry contract
                    "Global Russia result reference is invalid"
                )
            json_path = _safe_file(
                self.root, result_ref["jsonPath"], result_ref.get("jsonSha256")
            )
            npz_path = _safe_file(
                self.root, result_ref["npzPath"], result_ref.get("npzSha256")
            )
            metadata = _bounded_json(json_path)
            _verify_document_hash(metadata, "resultHash")
            if (
                metadata.get("schemaVersion") != "socialgraph-fm.global-model-result/1.0"
                or metadata.get("protocol") != protocol
                or metadata.get("country") != "russia"
                or metadata.get("graphVersionHash") != registry["graphVersionHash"]
                or metadata.get("corpusHash") != registry["corpusHash"]
                or metadata.get("modelVersionId") != entry["protocolModelVersionId"]
                or metadata.get("modelVersionHash") != entry["protocolModelVersionHash"]
                or metadata.get("modelStateHash") != entry["modelStateHash"]
                or float(metadata.get("threshold", -1.0)) != float(entry["threshold"])
            ):
                raise ValueError(f"Global {protocol} result metadata binding is invalid")
            arrays = _load_npz(npz_path)
            if not np.array_equal(
                np.sort(arrays["node_ids"].astype(np.int64, copy=False)),
                np.arange(russia.manifest.node_count, dtype=np.int64),
            ):
                raise ValueError("Global Russia result IDs do not match the safe corpus")
            card_protocol = card_protocols.get(protocol)
            if not isinstance(card_protocol, dict) or (
                card_protocol.get("modelVersionId") != entry["protocolModelVersionId"]
                or card_protocol.get("modelVersionHash")
                != entry["protocolModelVersionHash"]
                or card_protocol.get("modelStateHash") != entry["modelStateHash"]
            ):
                raise ValueError(f"Global {protocol} model card identity is invalid")
            results[protocol] = (metadata, arrays)
        self._registry = registry
        self._preview = preview
        self._model_card = model_card
        self._russia = russia
        self._results = results

    def capabilities(self) -> dict[str, Any]:
        registry = self._registry
        payload: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "channel": "socialgraph-global",
            "releaseLabel": "SocialGraph-FM Global",
            "seed": 12121995,
            "servingReady": registry is not None,
            "unavailableReason": None if registry is not None else "GFM_GLOBAL_MODEL_NOT_INSTALLED",
            "taskId": "coordination_risk",
            "datasetVersionId": "socialgraph-fm:russia",
            "model": None,
        }
        if registry is not None:
            payload["model"] = {
                key: registry[key]
                for key in (
                    "modelVersionId",
                    "modelVersionHash",
                    "artifactHash",
                    "corpusHash",
                    "sourceCodeHash",
                )
            }
            payload["model"].update(
                {
                    "taskId": "coordination_risk",
                    "protocols": list(PROTOCOLS),
                    "protocolModels": {
                        protocol: {
                            "modelVersionId": registry["protocolArtifacts"][protocol][
                                "protocolModelVersionId"
                            ],
                            "modelVersionHash": registry["protocolArtifacts"][protocol][
                                "protocolModelVersionHash"
                            ],
                            "modelStateHash": registry["protocolArtifacts"][protocol][
                                "modelStateHash"
                            ],
                            "state": "servingReady" if protocol == "global" else "frozenDemo",
                        }
                        for protocol in PROTOCOLS
                    },
                    "state": "servingReady",
                }
            )
        return _hash_payload(payload, "capabilityHash")

    def health(self) -> dict[str, Any]:
        registry = self._registry
        model_version_id = None if registry is None else str(registry["modelVersionId"])
        model_version_hash = None if registry is None else str(registry["modelVersionHash"])
        corpus_hash = None if registry is None else str(registry["corpusHash"])
        payload: dict[str, Any] = {
            "schemaVersion": HEALTH_SCHEMA_VERSION,
            "serviceIdentity": _service_identity(
                model_version_id=model_version_id,
                model_version_hash=model_version_hash,
                corpus_hash=corpus_hash,
            ),
            "servingReady": registry is not None,
            "modelVersionId": model_version_id,
            "modelVersionHash": model_version_hash,
            "corpusHash": corpus_hash,
            "datasetVersionId": "socialgraph-fm:russia",
        }
        return _hash_payload(payload, "healthHash")

    def model_card(self) -> dict[str, Any]:
        if self._registry is None or self._model_card is None:
            raise GlobalUnavailable
        return dict(self._model_card)

    @staticmethod
    def _metric(entry: Mapping[str, Any]) -> dict[str, Any]:
        metrics = entry.get("metrics")
        if not isinstance(metrics, Mapping):
            metrics = {}
        macro_f1 = metrics.get("macroF1", metrics.get("macro_f1", 0.0))
        pr_auc = metrics.get("prAuc", metrics.get("pr_auc", 0.0))
        return {
            "macroF1": float(macro_f1),
            "prAuc": float(pr_auc),
            "threshold": float(entry["threshold"]),
            "labelledTrainNodes": int(entry["labelledTrainNodes"]),
        }

    def scenario(self) -> dict[str, Any]:
        registry = self._registry
        node_count = 716 if self._russia is None else self._russia.manifest.node_count
        edge_count = (
            0 if self._russia is None else self._russia.manifest.edge_count // 2
        )
        payload: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "scenarioId": "russia-coordination-risk",
            "datasetVersionId": "socialgraph-fm:russia",
            "graphVersionHash": None if registry is None else registry["graphVersionHash"],
            "modelVersionId": None if registry is None else registry["modelVersionId"],
            "enabled": registry is not None,
            "unavailableReason": None if registry is not None else "GFM_GLOBAL_MODEL_NOT_INSTALLED",
            "nodeCount": node_count,
            "edgeCount": edge_count,
            "protocols": list(PROTOCOLS),
            "metrics": {protocol: None for protocol in PROTOCOLS},
            "limitations": list(LIMITATIONS),
        }
        if registry is not None:
            payload["metrics"] = {
                protocol: self._metric(registry["protocolArtifacts"][protocol])
                for protocol in PROTOCOLS
            }
        return _hash_payload(payload, "scenarioHash")

    def preview(self) -> dict[str, Any]:
        if self._registry is None or self._preview is None or self._russia is None:
            raise GlobalUnavailable
        raw_nodes = self._preview.get("nodes", ())
        raw_edges = self._preview.get("edges", ())
        nodes = []
        for item in raw_nodes:
            raw_id = str(item.get("id", item.get("nodeId")))
            node_id = raw_id if raw_id.startswith("russia:") else f"russia:{raw_id}"
            nodes.append(
                {
                    "id": node_id,
                    "label": str(item.get("label", f"Account {raw_id}")),
                    "degree": int(item.get("degree", 0)),
                    "structureMissing": bool(item.get("structureMissing", False)),
                }
            )
        edges = []
        for index, item in enumerate(raw_edges):
            source = str(item["source"])
            target = str(item["target"])
            source = source if source.startswith("russia:") else f"russia:{source}"
            target = target if target.startswith("russia:") else f"russia:{target}"
            modality = item.get("modality", "fused")
            if modality not in (*MODALITIES, "fused"):
                modality = "fused"
            edges.append(
                {
                    "id": str(item.get("id", f"preview:{index}")),
                    "source": source,
                    "target": target,
                    "modality": modality,
                }
            )
        payload: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "datasetVersionId": "socialgraph-fm:russia",
            "graphVersionHash": self._registry["graphVersionHash"],
            "nodes": nodes,
            "edges": edges,
            "nodeCount": self._russia.manifest.node_count,
            "edgeCount": self._russia.manifest.edge_count // 2,
            "partialPreview": bool(self._preview["partialPreview"]),
        }
        return _hash_payload(payload, "previewHash")

    def _run_dir(self, run_id: str) -> Path:
        if _RUN_ID.fullmatch(run_id) is None:
            raise GlobalNotFound
        return self.run_root / run_id

    def create_run(self, envelope: Mapping[str, Any]) -> dict[str, Any]:
        registry = self._registry
        if registry is None:
            raise GlobalUnavailable
        if envelope.get("schemaVersion") != SCHEMA_VERSION:
            raise GlobalInvalid
        request = envelope.get("request")
        expected_model = envelope.get("expectedModel")
        binding = envelope.get("datasetBinding")
        if not all(isinstance(value, Mapping) for value in (request, expected_model, binding)):
            raise GlobalInvalid
        assert isinstance(request, Mapping)
        assert isinstance(expected_model, Mapping)
        assert isinstance(binding, Mapping)
        protocol = request.get("protocol")
        artifact = (
            registry["protocolArtifacts"].get(protocol)
            if isinstance(protocol, str)
            else None
        )
        protocol_models = expected_model.get("protocolModels")
        expected_protocol_model = (
            protocol_models.get(protocol)
            if isinstance(protocol_models, Mapping) and isinstance(protocol, str)
            else None
        )
        if (
            request.get("schemaVersion") != SCHEMA_VERSION
            or request.get("taskId") != "coordination_risk"
            or request.get("datasetVersionId") != "socialgraph-fm:russia"
            or protocol not in PROTOCOLS
            or not isinstance(artifact, Mapping)
            or not isinstance(expected_protocol_model, Mapping)
            or request.get("modelVersionId") != artifact.get("protocolModelVersionId")
            or expected_protocol_model.get("modelVersionId")
            != artifact.get("protocolModelVersionId")
            or expected_protocol_model.get("modelVersionHash")
            != artifact.get("protocolModelVersionHash")
            or binding.get("datasetVersionId") != "socialgraph-fm:russia"
            or binding.get("graphVersionHash") != registry["graphVersionHash"]
        ):
            raise GlobalInvalid
        top_k = request.get("topK")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 500:
            raise GlobalInvalid
        request_hash = canonical_sha256(dict(request))
        run_id = f"global-model-{uuid.uuid4().hex}"
        now = _utc_now()
        status = {
            "schemaVersion": SCHEMA_VERSION,
            "runId": run_id,
            "requestHash": request_hash,
            "status": "queued",
            "progress": 0,
            "createdAt": now,
            "updatedAt": now,
            "errorCode": None,
        }
        run_dir = self._run_dir(run_id)
        with self._lock:
            run_dir.mkdir(parents=True, exist_ok=False)
            _atomic_json(run_dir / "request.json", dict(envelope))
            _atomic_json(run_dir / "state.json", status)
        self._executor.submit(self._execute, run_id, dict(request))
        return status

    def _finding(
        self,
        *,
        arrays: Mapping[str, np.ndarray],
        index: int,
        rank: int,
        threshold: float,
        expert_names: tuple[str, ...],
    ) -> dict[str, Any]:
        score = float(arrays["scores"][index])
        raw_id = int(arrays["node_ids"][index])
        routes = [{"expert": "shared", "weight": 1.0}]
        for expert_index, weight in zip(
            arrays["router_indices"][index], arrays["router_weights"][index], strict=True
        ):
            integer_index = int(expert_index)
            if integer_index < 0 or integer_index >= len(expert_names):
                raise ValueError("router expert index is outside the registry catalog")
            routes.append(
                {"expert": expert_names[integer_index], "weight": float(weight)}
            )
        counts = arrays["modality_counts"][index]
        return {
            "nodeId": f"russia:{raw_id}",
            "score": score,
            "rank": rank,
            "riskBand": _risk_band(score, threshold),
            "predictedPositive": score >= threshold,
            "structureMissing": bool(arrays["structure_missing"][index]),
            "routes": routes,
            "modalityEvidence": {
                modality: int(counts[position])
                for position, modality in enumerate(MODALITIES)
            },
        }

    def _execute(self, run_id: str, request: dict[str, Any]) -> None:
        run_dir = self._run_dir(run_id)
        try:
            with self._lock:
                state = _bounded_json(run_dir / "state.json")
                state.update(
                    {"status": "running", "progress": 20, "updatedAt": _utc_now()}
                )
                _atomic_json(run_dir / "state.json", state)
            protocol = str(request["protocol"])
            metadata, arrays = self._results[protocol]
            artifact = self._registry["protocolArtifacts"][protocol]  # type: ignore[index]
            threshold = float(artifact["threshold"])
            expert_names_value = metadata.get("expertNames", self._registry.get("expertNames"))  # type: ignore[union-attr]
            if not isinstance(expert_names_value, list) or not all(
                isinstance(item, str) for item in expert_names_value
            ):
                raise ValueError("Global expert catalog is missing")
            expert_names = tuple(expert_names_value)
            order = np.argsort(-arrays["scores"], kind="stable")[: int(request["topK"])]
            findings = [
                self._finding(
                    arrays=arrays,
                    index=int(index),
                    rank=rank,
                    threshold=threshold,
                    expert_names=expert_names,
                )
                for rank, index in enumerate(order, start=1)
            ]
            completed = _utc_now()
            result: dict[str, Any] = {
                "schemaVersion": SCHEMA_VERSION,
                "runId": run_id,
                "requestHash": canonical_sha256(request),
                "taskId": "coordination_risk",
                "protocol": protocol,
                "datasetVersionId": "socialgraph-fm:russia",
                "graphVersionHash": self._registry["graphVersionHash"],  # type: ignore[index]
                "corpusHash": self._registry["corpusHash"],  # type: ignore[index]
                "splitHash": artifact["splitHash"],
                "modelVersionId": artifact["protocolModelVersionId"],
                "modelVersionHash": artifact["protocolModelVersionHash"],
                "threshold": threshold,
                "metrics": self._metric(artifact),
                "findings": findings,
                "limitations": list(LIMITATIONS),
                "completedAt": completed,
            }
            _hash_payload(result, "resultHash")
            with self._lock:
                _atomic_json(run_dir / "result.json", result)
                state.update(
                    {"status": "succeeded", "progress": 100, "updatedAt": completed}
                )
                _atomic_json(run_dir / "state.json", state)
        except Exception:  # noqa: BLE001 - background failures are persisted as run state
            with self._lock:
                try:
                    state = _bounded_json(run_dir / "state.json")
                    state.update(
                        {
                            "status": "failed",
                            "progress": 100,
                            "updatedAt": _utc_now(),
                            "errorCode": "GFM_GLOBAL_MODEL_EXECUTION_FAILED",
                        }
                    )
                    _atomic_json(run_dir / "state.json", state)
                except Exception:  # noqa: BLE001, S110 - preserve the primary worker failure
                    pass

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            path = self._run_dir(run_id) / "state.json"
            if not path.is_file():
                raise GlobalNotFound
            return _bounded_json(path)

    def get_result(self, run_id: str) -> dict[str, Any]:
        status = self.get_run(run_id)
        if status.get("status") != "succeeded":
            raise GlobalResultNotReady
        result = _bounded_json(self._run_dir(run_id) / "result.json")
        if result.get("resultHash") != canonical_sha256(
            {key: value for key, value in result.items() if key != "resultHash"}
        ):
            raise ValueError("persisted Global result hash mismatch")
        return result

    @staticmethod
    def _result_index(arrays: Mapping[str, np.ndarray], raw_node_id: int) -> int:
        matches = np.flatnonzero(arrays["node_ids"] == raw_node_id)
        if matches.shape != (1,):
            raise ValueError("Global result/corpus node identity is inconsistent")
        return int(matches[0])

    def _relation_evidence(self, source: int, target: int) -> list[dict[str, Any]]:
        if self._russia is None:
            raise GlobalUnavailable
        evidence: list[dict[str, Any]] = []
        for modality in MODALITIES:
            relation = self._russia.relation(modality)
            start = int(relation.indptr[source])
            stop = int(relation.indptr[source + 1])
            row = relation.indices[start:stop]
            position = int(np.searchsorted(row, target))
            if position < row.size and int(row[position]) == target:
                evidence.append(
                    {
                        "modality": modality,
                        "rawWeight": float(relation.weights[start + position]),
                    }
                )
        return evidence

    @staticmethod
    def _scored_node(
        arrays: Mapping[str, np.ndarray],
        raw_node_id: int,
        *,
        hop: int,
        threshold: float,
    ) -> dict[str, Any]:
        index = GlobalServingRuntime._result_index(arrays, raw_node_id)
        score = float(arrays["scores"][index])
        return {
            "nodeId": f"russia:{raw_node_id}",
            "score": score,
            "hop": hop,
            "riskBand": _risk_band(score, threshold),
            "predictedPositive": score >= threshold,
            "structureMissing": bool(arrays["structure_missing"][index]),
        }

    def _two_hop_subgraph(
        self,
        raw_node_id: int,
        *,
        arrays: Mapping[str, np.ndarray],
        threshold: float,
    ) -> tuple[dict[str, Any], dict[int, int]]:
        if self._russia is None:
            raise GlobalUnavailable
        fused = self._russia.fused_csr
        hops = {raw_node_id: 0}
        frontier = {raw_node_id}
        for hop in (1, 2):
            following: set[int] = set()
            for source in sorted(frontier):
                start = int(fused.indptr[source])
                stop = int(fused.indptr[source + 1])
                following.update(int(value) for value in fused.indices[start:stop])
            following.difference_update(hops)
            for node in following:
                hops[node] = hop
            frontier = following
            if not frontier:
                break

        selected = set(hops)
        edges: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for source in sorted(selected):
            start = int(fused.indptr[source])
            stop = int(fused.indptr[source + 1])
            for raw_target in fused.indices[start:stop]:
                target = int(raw_target)
                if target not in selected or target == source:
                    continue
                pair = (min(source, target), max(source, target))
                if pair in seen:
                    continue
                seen.add(pair)
                relations = self._relation_evidence(pair[0], pair[1])
                if not relations:
                    relations = self._relation_evidence(pair[1], pair[0])
                edges.append(
                    {
                        "id": f"russia:{pair[0]}--russia:{pair[1]}",
                        "source": f"russia:{pair[0]}",
                        "target": f"russia:{pair[1]}",
                        "relations": relations,
                        "evidenceRole": "explanationOnly",
                    }
                )
        nodes = [
            self._scored_node(
                arrays,
                node,
                hop=hops[node],
                threshold=threshold,
            )
            for node in sorted(hops, key=lambda item: (hops[item], item))
        ]
        return (
            {
                "depth": 2,
                "nodeCount": len(nodes),
                "edgeCount": len(edges),
                "truncated": False,
                "nodes": nodes,
                "edges": edges,
            },
            hops,
        )

    def evidence(self, run_id: str, node_id: str) -> dict[str, Any]:
        if _NODE_ID.fullmatch(node_id) is None:
            raise GlobalNotFound
        canonical_id = node_id if node_id.startswith("russia:") else f"russia:{node_id}"
        result = self.get_result(run_id)
        finding = next(
            (item for item in result["findings"] if item["nodeId"] == canonical_id), None
        )
        if finding is None:
            raise GlobalNotFound
        if self._russia is None:
            raise GlobalUnavailable
        protocol = str(result["protocol"])
        _metadata, arrays = self._results[protocol]
        raw_node_id = int(canonical_id.removeprefix("russia:"))
        if not 0 <= raw_node_id < self._russia.manifest.node_count:
            raise GlobalNotFound
        threshold = float(result["threshold"])
        fused = self._russia.fused_csr
        start = int(fused.indptr[raw_node_id])
        stop = int(fused.indptr[raw_node_id + 1])
        direct_neighbors = [int(value) for value in fused.indices[start:stop]]
        neighbors = []
        for neighbor in direct_neighbors:
            summary = self._scored_node(
                arrays, neighbor, hop=1, threshold=threshold
            )
            relations = self._relation_evidence(raw_node_id, neighbor)
            if not relations:
                relations = self._relation_evidence(neighbor, raw_node_id)
            neighbors.append(
                {
                    **summary,
                    "modalities": [item["modality"] for item in relations],
                    "relations": relations,
                }
            )
        neighbors.sort(key=lambda item: (-float(item["score"]), str(item["nodeId"])))
        subgraph, hops = self._two_hop_subgraph(
            raw_node_id, arrays=arrays, threshold=threshold
        )
        relation_counts = {
            modality: int(
                self._russia.relation(modality).indptr[raw_node_id + 1]
                - self._russia.relation(modality).indptr[raw_node_id]
            )
            for modality in MODALITIES
        }
        payload: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "runId": run_id,
            "resultHash": result["resultHash"],
            "graphVersionHash": result["graphVersionHash"],
            "modelVersionId": result["modelVersionId"],
            "modelVersionHash": result["modelVersionHash"],
            "threshold": threshold,
            "node": finding,
            "neighbors": neighbors,
            "structuralSignals": {
                "fusedDegree": stop - start,
                "structureMissing": bool(
                    self._russia.structure_missing[raw_node_id]
                ),
                "relationNeighborCounts": relation_counts,
                "twoHopNodeCount": len(hops) - 1,
                "relationEvidenceRole": "explanationOnly",
            },
            "evidenceSubgraph": subgraph,
            "limitation": (
                "Factual CSR relation types and stored raw weights are explanation-only; "
                "they are not labels, proof of coordination, or additional model facts."
            ),
        }
        return _hash_payload(payload, "evidenceHash")

    def dispatch_get(self, path: str) -> dict[str, Any]:
        if path == "/internal/global-model/health":
            return self.health()
        if path == "/internal/global-model/capabilities":
            return self.capabilities()
        if path == "/internal/global-model/model-card":
            return self.model_card()
        if path == "/internal/global-model/scenario":
            return self.scenario()
        if path == "/internal/global-model/scenario/graph-preview":
            return self.preview()
        prefix = "/internal/global-model/runs/"
        if path.startswith(prefix):
            suffix = path[len(prefix) :]
            evidence_match = re.fullmatch(
                r"(global-model-[0-9a-f]{32})/nodes/([^/]+)/evidence", suffix
            )
            if evidence_match:
                return self.evidence(evidence_match.group(1), evidence_match.group(2))
            if suffix.endswith("/result"):
                return self.get_result(suffix[:-7])
            if "/" not in suffix:
                return self.get_run(suffix)
        raise GlobalNotFound

    def dispatch_post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if path == "/internal/global-model/runs":
            return self.create_run(payload)
        raise GlobalNotFound


__all__ = [
    "HEALTH_SCHEMA_VERSION",
    "MODALITIES",
    "MODEL_CARD_SCHEMA_VERSION",
    "PROTOCOLS",
    "SCHEMA_VERSION",
    "GlobalServiceError",
    "GlobalServingRuntime",
]
