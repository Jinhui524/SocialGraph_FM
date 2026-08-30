"""Read-only, checkpoint-bound forward smoke for the published SocialGraph-FM Global model."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from socialgraph_gfm.canonical import canonical_sha256, file_sha256
from socialgraph_gfm.gfm.corpus.common import resolve_within

from .config import (
    COUNTRIES,
    RELEASE_ID,
    SOURCE_COUNTRIES,
    TASK_ID,
    ProtocolId,
    load_global_model_config,
)
from .contracts import COUNTRY_IDS, read_corpus_manifest
from .corpus import GlobalCountryCorpus, load_country_corpus
from .model import GlobalModel, GlobalModelConfig
from .training import tensor_state_hash
from .workflow import EXPORT_SCHEMA, PROTOCOLS

FORWARD_SMOKE_SCHEMA = "socialgraph-fm.global-model-forward-smoke/1.0"
FORWARD_SMOKE_PROTOCOLS: tuple[ProtocolId, ...] = ("global", "in_domain", "low_label", "cross_domain")
SEED_NODE_COUNT = 4
NEIGHBOR_FANOUT = 4
NEIGHBOR_HOPS = 2
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
MAX_EXPORT_MANIFEST_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class _RussiaBatch:
    node_ids: np.ndarray
    seed_node_ids: tuple[int, ...]
    edge_index: np.ndarray
    text_features: np.ndarray
    degree_bucket: np.ndarray
    graph_stats: np.ndarray
    batch_hash: str
    maximum_node_count: int


def _expected_allowed_experts(protocol: ProtocolId) -> tuple[str, ...]:
    if protocol in {"in_domain", "low_label"}:
        return ("domain:russia", "null")
    domains = SOURCE_COUNTRIES if protocol == "cross_domain" else COUNTRIES
    return (*(f"domain:{country}" for country in domains), "null")


def _model_config(raw: Mapping[str, Any]) -> GlobalModelConfig:
    pinned = load_global_model_config()["model"]
    if dict(raw) != pinned:
        raise ValueError("Global checkpoint model configuration is not the pinned Global model")
    return GlobalModelConfig(
        text_dim=int(raw["textDim"]),
        structural_dim=int(raw["structuralDim"]),
        branch_dim=int(raw["branchDim"]),
        hidden_dim=int(raw["hiddenDim"]),
        dropout=float(raw["dropout"]),
        domains=COUNTRY_IDS,
        router_enabled=bool(raw["routerEnabled"]),
        router_bottleneck_dim=int(raw["routerBottleneckDim"]),
        router_top_k=int(raw["routerTopK"]),
    )


def _read_export(root: Path) -> dict[str, Any]:
    path = root / "exports" / "socialgraph-global" / "export-manifest.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_EXPORT_MANIFEST_BYTES:
        raise ValueError("Global export manifest is unavailable or unsafe")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schemaVersion") != EXPORT_SCHEMA:
        raise ValueError("Global export manifest schema is invalid")
    logical = {key: value for key, value in payload.items() if key != "exportHash"}
    if payload.get("exportHash") != canonical_sha256(logical):
        raise ValueError("Global export manifest hash is invalid")
    if (
        payload.get("releaseId") != RELEASE_ID
        or payload.get("taskId") != TASK_ID
        or payload.get("state") != "preliminary"
        or payload.get("fast") is not False
        or tuple(payload.get("protocols", ())) != PROTOCOLS
    ):
        raise ValueError("Global export protocol inventory is invalid")
    artifacts = payload.get("protocolArtifacts")
    models = payload.get("protocolModels")
    inventory = payload.get("artifacts")
    config = payload.get("config")
    expected_names = ["shared", *(f"domain:{country}" for country in COUNTRY_IDS), "null"]
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != set(PROTOCOLS)
        or not isinstance(models, dict)
        or set(models) != set(PROTOCOLS)
        or not isinstance(inventory, dict)
        or not isinstance(config, dict)
        or config.get("model") != load_global_model_config()["model"]
        or payload.get("expertNames") != expected_names
        or len({models[protocol].get("modelVersionId") for protocol in PROTOCOLS}) != 4
        or len({models[protocol].get("modelVersionHash") for protocol in PROTOCOLS}) != 4
        or len({models[protocol].get("modelStateHash") for protocol in PROTOCOLS}) != 4
    ):
        raise ValueError("Global export protocol bindings are invalid")
    return payload


def _load_russia(root: Path, export: Mapping[str, Any]) -> GlobalCountryCorpus:
    corpus_root = root / "corpus"
    manifest = read_corpus_manifest(corpus_root / "manifest.json")
    entry = next(
        (candidate for candidate in manifest.countries if candidate.country_id == "russia"),
        None,
    )
    if entry is None:
        raise ValueError("Global serving corpus has no Russia entry")
    country_root = resolve_within(corpus_root, entry.manifest_path).parent
    russia = load_country_corpus(
        country_root,
        verify_hashes=True,
        verify_values=True,
        mmap_mode="r",
    )
    if (
        manifest.content_hash != export.get("corpusHash")
        or russia.manifest.country_id != "russia"
        or russia.manifest.content_hash != entry.manifest_hash
        or russia.manifest.source_hashes != entry.source_hashes
        or russia.manifest.split_hashes != entry.split_hashes
        or russia.manifest.content_hash != export.get("graphVersionHash")
        or russia.manifest.node_count != 716
    ):
        raise ValueError("Global Russia serving corpus binding is invalid")
    return russia


def _selection_key(graph_hash: str, node_id: int, *, purpose: str) -> bytes:
    return hashlib.sha256(f"{graph_hash}:{purpose}:{node_id}".encode("ascii")).digest()


def _bounded_russia_batch(corpus: GlobalCountryCorpus) -> _RussiaBatch:
    graph_hash = corpus.manifest.content_hash
    node_count = corpus.manifest.node_count
    seed_nodes = tuple(
        sorted(
            sorted(
                range(node_count),
                key=lambda node_id: _selection_key(graph_hash, node_id, purpose="seed"),
            )[:SEED_NODE_COUNT]
        )
    )
    selected = set(seed_nodes)
    frontier = set(seed_nodes)
    fused = corpus.fused_csr
    for hop in range(NEIGHBOR_HOPS):
        next_frontier: set[int] = set()
        for node_id in sorted(frontier):
            start = int(fused.indptr[node_id])
            stop = int(fused.indptr[node_id + 1])
            neighbors = np.asarray(fused.indices[start:stop], dtype=np.int64)
            if neighbors.size <= NEIGHBOR_FANOUT:
                chosen = neighbors
            else:
                offset = int.from_bytes(
                    _selection_key(graph_hash, node_id, purpose=f"hop-{hop}")[:8], "big"
                ) % int(neighbors.size)
                positions = (offset + np.arange(NEIGHBOR_FANOUT)) % neighbors.size
                chosen = neighbors[positions]
            next_frontier.update(int(value) for value in chosen)
        selected.update(next_frontier)
        frontier = next_frontier

    maximum_node_count = SEED_NODE_COUNT * sum(
        NEIGHBOR_FANOUT**depth for depth in range(NEIGHBOR_HOPS + 1)
    )
    node_ids = np.asarray(sorted(selected), dtype=np.int64)
    if not SEED_NODE_COUNT <= node_ids.size <= maximum_node_count:
        raise ValueError("Global forward-smoke sampler exceeded its fixed node bound")
    lookup = np.full(node_count, -1, dtype=np.int64)
    lookup[node_ids] = np.arange(node_ids.size, dtype=np.int64)
    source = np.asarray(corpus.edge_index[0], dtype=np.int64)
    target = np.asarray(corpus.edge_index[1], dtype=np.int64)
    included = np.logical_and(lookup[source] >= 0, lookup[target] >= 0)
    edge_index = np.ascontiguousarray(
        np.stack((lookup[source[included]], lookup[target[included]])), dtype=np.int64
    )
    if edge_index.shape[1] > maximum_node_count**2:
        raise ValueError("Global forward-smoke sampler exceeded its fixed edge bound")
    text_features = np.array(corpus.text_features[node_ids], dtype=np.float32, copy=True, order="C")
    degree_bucket = np.array(corpus.degree_bucket[node_ids], dtype=np.int64, copy=True, order="C")
    graph_stats = np.array(corpus.graph_stats, dtype=np.float32, copy=True, order="C")
    digest = hashlib.sha256()
    for name, values in (
        ("nodeIds", node_ids.astype("<i8", copy=False)),
        ("edgeIndex", edge_index.astype("<i8", copy=False)),
        ("textFeatures", text_features.astype("<f4", copy=False)),
        ("degreeBucket", degree_bucket.astype("<i8", copy=False)),
        ("graphStats", graph_stats.astype("<f4", copy=False)),
    ):
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(values.shape)).encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return _RussiaBatch(
        node_ids=node_ids,
        seed_node_ids=seed_nodes,
        edge_index=edge_index,
        text_features=text_features,
        degree_bucket=degree_bucket,
        graph_stats=graph_stats,
        batch_hash=digest.hexdigest(),
        maximum_node_count=maximum_node_count,
    )


def _safe_checkpoint(root: Path, artifact: Mapping[str, Any]) -> Path:
    relative = artifact.get("checkpointPath")
    expected_hash = artifact.get("checkpointSha256")
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or Path(relative).is_absolute()
        or not isinstance(expected_hash, str)
    ):
        raise ValueError("Global checkpoint reference is invalid")
    path = resolve_within(root, relative)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > MAX_CHECKPOINT_BYTES
        or file_sha256(path) != expected_hash
    ):
        raise ValueError("Global checkpoint bytes do not match the export")
    return path


def _output_hash(outputs: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(outputs.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Global forward output {name!r} is not a Tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _smoke_protocol(
    root: Path,
    export: Mapping[str, Any],
    batch: _RussiaBatch,
    *,
    protocol: ProtocolId,
    device: Any,
) -> dict[str, Any]:
    import torch

    artifacts = cast(Mapping[str, Mapping[str, Any]], export["protocolArtifacts"])
    models = cast(Mapping[str, Mapping[str, Any]], export["protocolModels"])
    inventory = cast(Mapping[str, str], export["artifacts"])
    artifact = artifacts[protocol]
    protocol_model = models[protocol]
    expected_state = "servingReady" if protocol == "global" else "frozenDemo"
    expected_allowed = _expected_allowed_experts(protocol)
    expected_names = ("shared", *(f"domain:{country}" for country in COUNTRY_IDS), "null")
    if (
        artifact.get("state") != expected_state
        or protocol_model.get("state") != expected_state
        or artifact.get("protocolModelVersionId") != protocol_model.get("modelVersionId")
        or artifact.get("protocolModelVersionHash") != protocol_model.get("modelVersionHash")
        or artifact.get("modelStateHash") != protocol_model.get("modelStateHash")
        or tuple(artifact.get("allowedExperts", ())) != expected_allowed
        or inventory.get(f"checkpoints/{protocol}.pt") != artifact.get("checkpointSha256")
    ):
        raise ValueError(f"Global {protocol} export protocol binding is invalid")
    if protocol == "global" and (
        export.get("checkpointPath") != artifact.get("checkpointPath")
        or export.get("checkpointSha256") != artifact.get("checkpointSha256")
        or export.get("modelVersionId") != artifact.get("protocolModelVersionId")
        or export.get("modelVersionHash") != artifact.get("protocolModelVersionHash")
        or export.get("modelStateHash") != artifact.get("modelStateHash")
    ):
        raise ValueError("Global Global top-level identity is not checkpoint-bound")

    checkpoint = _safe_checkpoint(root, artifact)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Global {protocol} checkpoint envelope is invalid")
    allowed_experts = tuple(payload.get("allowedExperts", ()))
    expert_names = tuple(payload.get("expertNames", ()))
    if (
        payload.get("schemaVersion") != "socialgraph-fm.global-model-weights/1.0"
        or payload.get("releaseId") != RELEASE_ID
        or payload.get("taskId") != TASK_ID
        or payload.get("protocol") != protocol
        or payload.get("protocolModelVersionId") != artifact.get("protocolModelVersionId")
        or payload.get("protocolModelVersionHash") != artifact.get("protocolModelVersionHash")
        or payload.get("modelStateHash") != artifact.get("modelStateHash")
        or allowed_experts != expected_allowed
        or expert_names != expected_names
    ):
        raise ValueError(f"Global {protocol} checkpoint protocol binding is invalid")
    raw_config = payload.get("modelConfig")
    raw_state = payload.get("modelState")
    if not isinstance(raw_config, dict) or not isinstance(raw_state, dict):
        raise TypeError(f"Global {protocol} checkpoint payload is incomplete")
    state_hash = tensor_state_hash(raw_state)
    if state_hash != artifact.get("modelStateHash"):
        raise ValueError(f"Global {protocol} checkpoint modelState hash is invalid")

    config = _model_config(raw_config)
    model = GlobalModel(config)
    model.load_state_dict(raw_state, strict=True)
    model.requires_grad_(False)
    model.eval().to(device)
    text = torch.from_numpy(batch.text_features).to(device)
    structural = torch.from_numpy(batch.degree_bucket).to(device)
    edges = torch.from_numpy(batch.edge_index).to(device)
    statistics = torch.from_numpy(batch.graph_stats).to(device)
    domain_id = "russia" if "domain:russia" in allowed_experts else None
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ),
    ):
        output = model(
            text,
            structural,
            edges,
            graph_stats=statistics,
            domain_id=domain_id,
            allowed_experts=allowed_experts,
        )
    if output.router_indices is None or output.router_weights is None:
        raise ValueError(f"Global {protocol} forward did not produce top-2 router evidence")
    tensors = {
        "logits": output.logits,
        "nodeEmbeddings": output.node_embeddings,
        "fusedFeatures": output.fused_features,
        "modalityContributions": output.modality_contributions,
        "routerIndices": output.router_indices,
        "routerWeights": output.router_weights,
    }
    node_count = int(batch.node_ids.size)
    expected_shapes = {
        "logits": (node_count,),
        "nodeEmbeddings": (node_count, config.hidden_dim),
        "fusedFeatures": (node_count, config.hidden_dim),
        "modalityContributions": (node_count, 2),
        "routerIndices": (node_count, config.router_top_k),
        "routerWeights": (node_count, config.router_top_k),
    }
    observed_shapes = {
        name: tuple(int(value) for value in tensor.shape) for name, tensor in tensors.items()
    }
    if observed_shapes != expected_shapes:
        raise ValueError(f"Global {protocol} forward output shape is invalid")
    if any(tensor.device.type != device.type for tensor in tensors.values()):
        raise ValueError(f"Global {protocol} forward output device is invalid")
    floating = tuple(tensor for name, tensor in tensors.items() if name != "routerIndices")
    finite = all(bool(torch.isfinite(tensor).all()) for tensor in floating)
    if not finite:
        raise ValueError(f"Global {protocol} forward output is not finite")
    allowed_indices = tuple(expert_names.index(name) for name in allowed_experts)
    route_indices = output.router_indices
    route_weights = output.router_weights
    tolerance = 2e-3 if device.type == "cuda" else 1e-4
    routes_allowed = bool(
        torch.isin(
            route_indices,
            torch.tensor(allowed_indices, device=device, dtype=route_indices.dtype),
        ).all()
    )
    weights_valid = bool(
        ((route_weights >= 0) & (route_weights <= 1)).all()
        and torch.allclose(
            route_weights.sum(dim=1),
            torch.ones(node_count, device=device, dtype=route_weights.dtype),
            atol=tolerance,
            rtol=tolerance,
        )
    )
    modality_valid = bool(
        ((output.modality_contributions >= 0) & (output.modality_contributions <= 1)).all()
        and torch.allclose(
            output.modality_contributions.sum(dim=1),
            torch.ones(
                node_count,
                device=device,
                dtype=output.modality_contributions.dtype,
            ),
            atol=tolerance,
            rtol=tolerance,
        )
    )
    if not routes_allowed or not weights_valid or not modality_valid:
        raise ValueError(f"Global {protocol} router or modality evidence is invalid")
    if tuple(output.expert_names) != expected_names:
        raise ValueError(f"Global {protocol} forward expert catalog changed")
    if tensor_state_hash(model.state_dict()) != state_hash:
        raise ValueError(f"Global {protocol} forward mutated modelState")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {
        "protocol": protocol,
        "checkpoint": {
            "relativePath": str(artifact["checkpointPath"]),
            "sha256": str(artifact["checkpointSha256"]),
            "schemaVersion": str(payload["schemaVersion"]),
        },
        "model": {
            "modelVersionId": str(artifact["protocolModelVersionId"]),
            "modelVersionHash": str(artifact["protocolModelVersionHash"]),
            "modelStateHash": state_hash,
        },
        "device": device.type,
        "precision": "float16-autocast" if device.type == "cuda" else "float32",
        "batch": {
            "seedNodeCount": len(batch.seed_node_ids),
            "sampledNodeCount": node_count,
            "sampledEdgeCount": int(batch.edge_index.shape[1]),
        },
        "allowedExperts": list(allowed_experts),
        "allowedExpertIndices": list(allowed_indices),
        "allowedExpertMask": [index in allowed_indices for index in range(1, len(expert_names))],
        "router": {
            "routesAllowed": routes_allowed,
            "weightsValid": weights_valid,
            "sumTolerance": tolerance,
            "observedExpertIndices": sorted(
                int(value) for value in torch.unique(route_indices).detach().cpu().tolist()
            ),
        },
        "shape": {name: list(shape) for name, shape in observed_shapes.items()},
        "finite": finite,
        "modalityContributionsValid": modality_valid,
        "modelStateUnchanged": True,
        "outputHash": _output_hash(tensors),
    }
    del model, payload, raw_state, output, tensors, text, structural, edges, statistics
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def run_checkpoint_forward_smoke(
    root: str | Path,
    *,
    device: str,
) -> dict[str, Any]:
    """Run all four published checkpoints on one fixed bounded Russia corpus batch.

    The operation is deliberately read-only: it does not publish a registry, create a
    smoke-report file, or mutate any model/corpus artifact.
    """

    import torch
    import torch_geometric

    if device not in {"cpu", "cuda"}:
        raise ValueError("Global forward-smoke device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    selected = Path(root).expanduser().resolve(strict=True)
    if selected == Path(selected.anchor):
        raise ValueError("Global forward-smoke root must not be a filesystem root")
    selected_device = torch.device(device)
    export = _read_export(selected)
    russia = _load_russia(selected, export)
    batch = _bounded_russia_batch(russia)

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    seed = int.from_bytes(
        hashlib.sha256(
            f"{russia.manifest.content_hash}:checkpoint-forward-smoke-v1".encode("ascii")
        ).digest()[:8],
        "big",
    ) % (2**53)
    try:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        reports = [
            _smoke_protocol(
                selected,
                export,
                batch,
                protocol=protocol,
                device=selected_device,
            )
            for protocol in FORWARD_SMOKE_PROTOCOLS
        ]
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark

    report: dict[str, Any] = {
        "schemaVersion": FORWARD_SMOKE_SCHEMA,
        "passed": True,
        "readOnly": True,
        "device": selected_device.type,
        "deviceName": (
            torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "cpu"
        ),
        "torchVersion": torch.__version__,
        "torchGeometricVersion": torch_geometric.__version__,
        "exportHash": export["exportHash"],
        "corpus": {
            "country": "russia",
            "corpusHash": export["corpusHash"],
            "graphVersionHash": russia.manifest.content_hash,
            "nodeCount": russia.manifest.node_count,
            "edgeCount": russia.manifest.edge_count,
        },
        "batch": {
            "selectionPolicy": "sha256-graph-version-bounded-neighborhood/1.0",
            "seed": seed,
            "seedNodeIds": list(batch.seed_node_ids),
            "seedNodeCount": len(batch.seed_node_ids),
            "neighborFanout": NEIGHBOR_FANOUT,
            "neighborHops": NEIGHBOR_HOPS,
            "maximumNodeCount": batch.maximum_node_count,
            "sampledNodeCount": int(batch.node_ids.size),
            "sampledEdgeCount": int(batch.edge_index.shape[1]),
            "batchHash": batch.batch_hash,
        },
        "protocolCount": len(reports),
        "protocols": reports,
    }
    report["reportHash"] = canonical_sha256(report)
    return report


__all__ = [
    "FORWARD_SMOKE_PROTOCOLS",
    "FORWARD_SMOKE_SCHEMA",
    "NEIGHBOR_FANOUT",
    "NEIGHBOR_HOPS",
    "SEED_NODE_COUNT",
    "run_checkpoint_forward_smoke",
]
