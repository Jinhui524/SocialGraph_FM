"""Checkpoint-bound, unlabeled NeighborLoader inference for Governance Global."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from socialgraph_gfm.canonical import canonical_sha256, file_sha256
from socialgraph_gfm.global_model.contracts import COUNTRY_IDS
from socialgraph_gfm.global_model.model import GlobalModel, GlobalModelConfig
from socialgraph_gfm.global_model.training import tensor_state_hash

from .contracts import MODALITIES
from .materialize import OnlineInferenceData

FANOUT = (20, 10)
BATCH_CANDIDATES = (128, 64, 32)
INFERENCE_SEED_DERIVATION = "sha256-dataset-model-u53-v2"
INFERENCE_SEED_UPPER_EXCLUSIVE = 2**53


class InferenceCancelled(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _read_object(path: Path) -> dict[str, Any]:
    import json

    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"invalid Governance model document: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(  # noqa: TRY004 - malformed persisted contract
            f"Governance model document must be an object: {path.name}"
        )
    return payload


def _safe_model_file(root: Path, relative: str, digest: str) -> Path:
    if not relative or "\\" in relative or Path(relative).is_absolute():
        raise ValueError("model artifact path is not portable")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)) or path.is_symlink():
        raise ValueError("model artifact path escapes the SocialGraph-FM Global root")
    if file_sha256(path) != digest:
        raise ValueError("model artifact SHA-256 mismatch")
    return path


def _model_config(raw: Mapping[str, Any]) -> GlobalModelConfig:
    expected = {
        "textDim": 768,
        "structuralDim": 128,
        "branchDim": 128,
        "hiddenDim": 256,
        "gnnLayers": 2,
        "routerEnabled": True,
        "routerExperts": 8,
        "routerTopK": 2,
    }
    if any(raw.get(name) != value for name, value in expected.items()):
        raise ValueError("Global checkpoint model configuration is incompatible")
    dropout = raw.get("dropout")
    bottleneck = raw.get("routerBottleneckDim")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (float, int))
        or isinstance(bottleneck, bool)
        or not isinstance(bottleneck, int)
    ):
        raise ValueError(  # noqa: TRY004 - malformed persisted contract
            "Global checkpoint model configuration is malformed"
        )
    return GlobalModelConfig(
        dropout=float(dropout),
        router_bottleneck_dim=bottleneck,
        domains=COUNTRY_IDS,
    )


def select_device(requested: str) -> Any:
    import torch

    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("Governance device must be auto, cuda, or cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu")


@dataclass(frozen=True)
class LoadedGlobalModel:
    model: Any
    device: Any
    device_name: str
    dtype_name: str
    model_version_id: str
    model_version_hash: str
    model_state_hash: str
    allowed_experts: tuple[str, ...]
    expert_names: tuple[str, ...]
    temperature: float
    bias: float
    threshold: float
    reference_metrics: Mapping[str, Any]
    loaded_at: str
    execution_environment_hash: str
    runtime_recipe_hash: str


def execution_environment_identity(
    *,
    device_type: str,
    dtype_name: str,
    torch_version: str,
    torch_geometric_version: str,
    pyg_lib_version: str | None,
    cuda_runtime: str | None,
    device_capability: tuple[int, int] | None,
) -> dict[str, object]:
    """Bind runtime numerics without changing the frozen model identity."""

    return {
        "schemaVersion": "socialgraph-fm.governance-execution-environment/1.0",
        "device": device_type,
        "dtype": dtype_name,
        "torchVersion": torch_version,
        "torchGeometricVersion": torch_geometric_version,
        "pygLibVersion": pyg_lib_version,
        "compiledCudaRuntime": cuda_runtime,
        "deviceCapability": list(device_capability) if device_capability else None,
    }


def runtime_recipe_identity(
    *,
    model_state_hash: str,
    allowed_experts: tuple[str, ...],
    use_amp: bool,
    execution_environment_hash: str,
) -> dict[str, object]:
    """Return every deterministic runtime choice that binds a run identity."""

    return {
        "schemaVersion": "socialgraph-fm.governance-runtime-recipe/2.2",
        "modelStateHash": model_state_hash,
        "fanout": list(FANOUT),
        "batchCandidates": list(BATCH_CANDIDATES),
        "amp": use_amp,
        "executionEnvironmentHash": execution_environment_hash,
        "domainId": None,
        "allowedExperts": list(allowed_experts),
        "numWorkers": 0,
        "inferenceSeedDerivation": INFERENCE_SEED_DERIVATION,
        "inferenceSeedUpperExclusive": INFERENCE_SEED_UPPER_EXCLUSIVE,
    }


def load_global_model(global_model_root: str | Path, *, device: str = "auto") -> LoadedGlobalModel:
    """Load and identity-check only the published Global weights (never result snapshots)."""

    import torch
    import torch_geometric

    root = Path(global_model_root).expanduser().resolve(strict=True)
    registry = _read_object(root / "registry" / "socialgraph-global.json")
    logical_registry = {key: value for key, value in registry.items() if key != "registryHash"}
    if (
        registry.get("schemaVersion") != "socialgraph-fm.global-model-registry/1.0"
        or registry.get("registryHash") != canonical_sha256(logical_registry)
        or registry.get("state") != "servingReady"
    ):
        raise ValueError("SocialGraph-FM Global serving registry is not a valid ready snapshot")
    artifact = registry.get("protocolArtifacts", {}).get("global")
    if not isinstance(artifact, dict) or artifact.get("state") != "servingReady":
        raise ValueError("Global protocol artifact is not serving-ready")
    top_level_bindings = {
        "modelVersionId": "protocolModelVersionId",
        "modelVersionHash": "protocolModelVersionHash",
        "modelStateHash": "modelStateHash",
        "checkpointPath": "checkpointPath",
        "checkpointSha256": "checkpointSha256",
    }
    if any(
        registry.get(registry_name) != artifact.get(artifact_name)
        for registry_name, artifact_name in top_level_bindings.items()
    ):
        raise ValueError("Global registry top-level identity is not bound to its artifact")
    checkpoint = _safe_model_file(
        root, str(artifact["checkpointPath"]), str(artifact["checkpointSha256"])
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schemaVersion") != "socialgraph-fm.global-model-weights/1.0":
        raise ValueError("Global checkpoint envelope is invalid")
    identity_fields = (
        ("protocol", "global"),
        ("protocolModelVersionId", artifact.get("protocolModelVersionId")),
        ("protocolModelVersionHash", artifact.get("protocolModelVersionHash")),
        ("modelStateHash", artifact.get("modelStateHash")),
    )
    if any(payload.get(name) != expected for name, expected in identity_fields):
        raise ValueError("Global checkpoint identity disagrees with the registry")
    raw_state = payload.get("modelState")
    if not isinstance(raw_state, dict) or tensor_state_hash(raw_state) != artifact.get("modelStateHash"):
        raise ValueError("Global checkpoint tensor state hash mismatch")
    raw_config = payload.get("modelConfig")
    if not isinstance(raw_config, dict):
        raise ValueError(  # noqa: TRY004 - malformed persisted contract
            "Global checkpoint model config is missing"
        )
    config = _model_config(raw_config)
    allowed = tuple(str(value) for value in payload.get("allowedExperts", ()))
    expected_allowed = tuple(f"domain:{country}" for country in COUNTRY_IDS) + ("null",)
    expert_names = tuple(str(value) for value in payload.get("expertNames", ()))
    expected_names = ("shared", *expected_allowed)
    if allowed != expected_allowed or expert_names != expected_names:
        raise ValueError("Global checkpoint must authorize all six domain experts and null")
    selected_device = select_device(device)
    model = GlobalModel(config)
    model.load_state_dict(raw_state, strict=True)
    model.requires_grad_(False)
    model.eval().to(selected_device)
    raw_temperature = artifact.get("temperature")
    raw_bias = artifact.get("bias")
    raw_threshold = artifact.get("threshold")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (raw_temperature, raw_bias, raw_threshold)
    ):
        raise ValueError("Global calibration parameters are malformed")
    temperature = float(cast(float, raw_temperature))
    bias = float(cast(float, raw_bias))
    threshold = float(cast(float, raw_threshold))
    if (
        not math.isfinite(temperature)
        or temperature <= 0
        or not math.isfinite(bias)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Global calibration parameters are invalid")
    state_hash = str(artifact["modelStateHash"])
    dtype_name = "float16" if selected_device.type == "cuda" else "float32"
    try:
        pyg_lib_version = importlib.metadata.version("pyg-lib")
    except importlib.metadata.PackageNotFoundError:
        pyg_lib_version = None
    environment_identity = execution_environment_identity(
        device_type=selected_device.type,
        dtype_name=dtype_name,
        torch_version=str(torch.__version__),
        torch_geometric_version=str(torch_geometric.__version__),
        pyg_lib_version=pyg_lib_version,
        cuda_runtime=torch.version.cuda,
        device_capability=(
            (
                int(torch.cuda.get_device_capability(selected_device)[0]),
                int(torch.cuda.get_device_capability(selected_device)[1]),
            )
            if selected_device.type == "cuda"
            else None
        ),
    )
    execution_environment_hash = canonical_sha256(environment_identity)
    recipe = runtime_recipe_identity(
        model_state_hash=state_hash,
        allowed_experts=allowed,
        use_amp=selected_device.type == "cuda",
        execution_environment_hash=execution_environment_hash,
    )
    return LoadedGlobalModel(
        model=model,
        device=selected_device,
        device_name=selected_device.type,
        dtype_name=dtype_name,
        model_version_id=str(artifact["protocolModelVersionId"]),
        model_version_hash=str(artifact["protocolModelVersionHash"]),
        model_state_hash=state_hash,
        allowed_experts=allowed,
        expert_names=expert_names,
        temperature=temperature,
        bias=bias,
        threshold=threshold,
        reference_metrics=dict(artifact.get("metrics", {})),
        loaded_at=_utc_now(),
        execution_environment_hash=execution_environment_hash,
        runtime_recipe_hash=canonical_sha256(recipe),
    )


@dataclass(frozen=True)
class OnlineInferenceOutputs:
    logits: np.ndarray
    scores: np.ndarray
    embeddings: np.ndarray
    router_indices: np.ndarray
    router_weights: np.ndarray
    modality_contributions: np.ndarray
    modality_counts: np.ndarray
    batch_size: int
    peak_memory_mib: float | None
    seed: int


def deterministic_inference_seed(dataset_content_hash: str, model_state_hash: str) -> int:
    digest = hashlib.sha256(
        f"{dataset_content_hash}:{model_state_hash}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") % INFERENCE_SEED_UPPER_EXCLUSIVE


def _is_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: memory allocation" in message


def _infer_once(
    data: OnlineInferenceData,
    loaded: LoadedGlobalModel,
    *,
    batch_size: int,
    seed: int,
    progress: Callable[[float], None],
    cancelled: Callable[[], bool],
) -> OnlineInferenceOutputs:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader

    count = len(data.node_ids)
    graph = Data(
        edge_index=torch.from_numpy(np.array(data.edge_index, copy=True)).long().contiguous(),
        text_features=torch.from_numpy(np.array(data.text_features, copy=True)).float().contiguous(),
        structural_features=torch.from_numpy(np.array(data.degree_bucket, copy=True)).long().contiguous(),
        structure_missing=torch.from_numpy(np.array(data.structure_missing, copy=True)).bool().contiguous(),
        num_nodes=count,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = NeighborLoader(
        graph,
        input_nodes=torch.arange(count, dtype=torch.long),
        num_neighbors=list(FANOUT),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        generator=generator,
    )
    logits = np.full(count, np.nan, dtype=np.float32)
    embeddings = np.full((count, 256), np.nan, dtype=np.float32)
    route_indices = np.full((count, 2), -1, dtype=np.int16)
    route_weights = np.full((count, 2), np.nan, dtype=np.float32)
    modality = np.full((count, 2), np.nan, dtype=np.float32)
    seen = np.zeros(count, dtype=np.bool_)
    statistics = torch.from_numpy(np.array(data.graph_stats, copy=True)).float().to(loaded.device)
    if loaded.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(loaded.device)
    completed = 0
    loaded.model.eval()
    with torch.inference_mode():
        for batch in loader:
            if cancelled():
                raise InferenceCancelled
            text = batch.text_features.to(loaded.device, non_blocking=True)
            structural = batch.structural_features.to(loaded.device, non_blocking=True)
            edge_index = batch.edge_index.to(loaded.device, non_blocking=True)
            with torch.autocast(
                device_type=loaded.device.type,
                dtype=torch.float16,
                enabled=loaded.device.type == "cuda",
            ):
                output = loaded.model(
                    text,
                    structural,
                    edge_index,
                    domain_id=None,
                    graph_stats=statistics,
                    allowed_experts=loaded.allowed_experts,
                )
            seed_count = int(batch.batch_size)
            positions = batch.n_id[:seed_count].detach().cpu().numpy().astype(np.int64)
            if bool(seen[positions].any()):
                raise ValueError("NeighborLoader emitted a duplicate seed node")
            if output.router_indices is None or output.router_weights is None:
                raise ValueError("Global online inference requires top-2 router evidence")
            logits[positions] = output.logits[:seed_count].detach().float().cpu().numpy()
            embeddings[positions] = (
                output.node_embeddings[:seed_count].detach().float().cpu().numpy()
            )
            route_indices[positions] = (
                output.router_indices[:seed_count].detach().short().cpu().numpy()
            )
            route_weights[positions] = (
                output.router_weights[:seed_count].detach().float().cpu().numpy()
            )
            modality[positions] = (
                output.modality_contributions[:seed_count].detach().float().cpu().numpy()
            )
            seen[positions] = True
            completed += seed_count
            progress(completed / count)
    if not bool(seen.all()) or not all(
        bool(np.isfinite(array).all())
        for array in (logits, embeddings, route_weights, modality)
    ):
        raise ValueError("Global online inference did not produce one finite row per node")
    calibrated = logits.astype(np.float64) / loaded.temperature + loaded.bias
    scores = (1.0 / (1.0 + np.exp(-np.clip(calibrated, -80, 80)))).astype(np.float32)
    modality_counts = np.empty((count, len(MODALITIES)), dtype=np.int32)
    for column, name in enumerate(MODALITIES):
        token = name.lower()
        indptr = np.asarray(data.arrays[f"relation_{token}_indptr"])
        modality_counts[:, column] = np.diff(indptr).astype(np.int32)
    peak = (
        float(torch.cuda.max_memory_allocated(loaded.device) / (1024 * 1024))
        if loaded.device.type == "cuda"
        else None
    )
    return OnlineInferenceOutputs(
        logits=logits,
        scores=scores,
        embeddings=embeddings,
        router_indices=route_indices,
        router_weights=route_weights,
        modality_contributions=modality,
        modality_counts=modality_counts,
        batch_size=batch_size,
        peak_memory_mib=peak,
        seed=seed,
    )


def run_online_inference(
    data: OnlineInferenceData,
    loaded: LoadedGlobalModel,
    *,
    progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> OnlineInferenceOutputs:
    """Run all-node inference, retrying only genuine CUDA OOM failures."""

    import torch

    notify = progress or (lambda _value: None)
    is_cancelled = cancelled or (lambda: False)
    seed = deterministic_inference_seed(
        data.artifact.dataset_content_hash, loaded.model_state_hash
    )
    torch.manual_seed(seed)
    if loaded.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    last_error: RuntimeError | None = None
    maximum_progress = 0.0

    def monotonic_progress(value: float) -> None:
        nonlocal maximum_progress
        maximum_progress = max(maximum_progress, value)
        notify(maximum_progress)

    for batch_size in BATCH_CANDIDATES:
        try:
            result = _infer_once(
                data,
                loaded,
                batch_size=batch_size,
                seed=seed,
                progress=monotonic_progress,
                cancelled=is_cancelled,
            )
            if result.peak_memory_mib is None or result.peak_memory_mib <= 5632:
                return result
            if batch_size == BATCH_CANDIDATES[-1]:
                raise RuntimeError(
                    "Global inference exceeds the 5632 MiB deployment memory ceiling"
                )
            if loaded.device.type == "cuda":
                torch.cuda.empty_cache()
        except InferenceCancelled:
            raise
        except RuntimeError as error:
            if loaded.device.type != "cuda" or not _is_oom(error):
                raise
            last_error = error
            torch.cuda.empty_cache()
    assert last_error is not None
    raise RuntimeError("Global inference exhausted the 128/64/32 CUDA batch fallback") from last_error


__all__ = [
    "BATCH_CANDIDATES",
    "FANOUT",
    "InferenceCancelled",
    "LoadedGlobalModel",
    "OnlineInferenceOutputs",
    "deterministic_inference_seed",
    "execution_environment_identity",
    "load_global_model",
    "run_online_inference",
    "runtime_recipe_identity",
    "select_device",
]
