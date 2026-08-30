"""Fresh-process, CPU-only recovery for non-promotable local evaluation."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
from pathlib import Path
from typing import Any, Literal

import numpy
import pydantic
import scipy
import torch
import torch_geometric
from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.tensor_digest import canonical_tensor_digest

from .adapters import AdapterSchema, BundleInputAdapter, derive_training_selection
from .bundle import load_core_graph_bundle_json
from .checkpoint import CheckpointBindings, load_checkpoint
from .config import TrainingConfig
from .fold_recovery import _composite_hash, _state_hash
from .model import CoreGFM
from .trainer import CoreTrainer, TrainingGraph
from .training_data import ExecutionPolicy, PreparedGraph


LOCAL_CODE_INVENTORY_RELATIVE_PATHS = (
    "__init__.py",
    "canonical.py",
    "contracts.py",
    "errors.py",
    "public_contracts.py",
    "tensor_digest.py",
    "core/__init__.py",
    "core/adapters.py",
    "core/bundle.py",
    "core/calibration.py",
    "core/checkpoint.py",
    "core/config.py",
    "core/experiment_cli.py",
    "core/experiment_data.py",
    "core/fold_recovery.py",
    "core/formal_materialization.py",
    "core/formal_preflight.py",
    "core/graph_ops.py",
    "core/local_experiments.py",
    "core/local_recovery.py",
    "core/model.py",
    "core/objectives.py",
    "core/safe_paths.py",
    "core/serving_registry.py",
    "core/splits.py",
    "core/structure_features.py",
    "core/supervised.py",
    "core/trainer.py",
    "core/training_data.py",
    "core/datasets/__init__.py",
    "core/datasets/acquire.py",
    "core/datasets/mat_worker.py",
    "core/datasets/materialize.py",
    "core/datasets/parsers.py",
    "core/datasets/penn94_conversion.py",
    "core/datasets/recipes.py",
    "core/datasets/recipes.json",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


class LocalEnvironmentInventory(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-local-environment-inventory/2.0"] = Field(
        alias="schemaVersion"
    )
    python: str
    platform: str
    torch: str
    numpy: str
    scipy: str
    pydantic_version: str = Field(alias="pydantic")
    torch_geometric: str = Field(alias="torchGeometric")
    cuda_runtime: str | None = Field(alias="cudaRuntime")
    device_type: Literal["cpu", "cuda"] = Field(alias="deviceType")
    device_name: str = Field(alias="deviceName")
    inventory_hash: str = Field(alias="inventoryHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hash(self):
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"inventory_hash"})
        )
        if self.inventory_hash != expected:
            raise ValueError("environment inventoryHash does not match exact fields")
        return self


class LocalRecoveryReceipt(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-local-recovery-receipt/4.0"] = Field(
        alias="schemaVersion"
    )
    request_hash: str = Field(alias="requestHash", pattern=r"^[0-9a-f]{64}$")
    recovery_process_id: int = Field(alias="recoveryProcessId", ge=1)
    recovery_parent_process_id: int = Field(alias="recoveryParentProcessId", ge=1)
    recovery_device: Literal["cpu"] = Field(alias="recoveryDevice")
    recovery_interpreter_path: str = Field(alias="recoveryInterpreterPath", min_length=1)
    recovery_interpreter_sha256: str = Field(
        alias="recoveryInterpreterSha256", pattern=r"^[0-9a-f]{64}$"
    )
    checkpoint_sha256: str = Field(alias="checkpointSha256", pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(alias="configHash", pattern=r"^[0-9a-f]{64}$")
    data_hash: str = Field(alias="dataHash", pattern=r"^[0-9a-f]{64}$")
    code_hash: str = Field(alias="codeHash", pattern=r"^[0-9a-f]{64}$")
    environment_hash: str = Field(alias="environmentHash", pattern=r"^[0-9a-f]{64}$")
    recovery_environment_inventory: LocalEnvironmentInventory = Field(
        alias="recoveryEnvironmentInventory"
    )
    recovery_environment_hash: str = Field(
        alias="recoveryEnvironmentHash", pattern=r"^[0-9a-f]{64}$"
    )
    trainer_state_hash: str = Field(alias="trainerStateHash", pattern=r"^[0-9a-f]{64}$")
    composite_state_hash: str = Field(alias="compositeStateHash", pattern=r"^[0-9a-f]{64}$")
    recovery_state_hash: str = Field(alias="recoveryStateHash", pattern=r"^[0-9a-f]{64}$")
    model_state_hash: str = Field(alias="modelStateHash", pattern=r"^[0-9a-f]{64}$")
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=r"^[0-9a-f]{64}$")
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=r"^[0-9a-f]{64}$")
    evaluation_artifact_sha256: str = Field(
        alias="evaluationArtifactSha256", pattern=r"^[0-9a-f]{64}$"
    )
    receipt_hash: str = Field(alias="receiptHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hash(self):
        if (
            self.recovery_environment_inventory.device_type != "cpu"
            or self.recovery_environment_hash != self.recovery_environment_inventory.inventory_hash
        ):
            raise ValueError("recovery environment identity is not exact CPU evidence")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"receipt_hash"})
        )
        if self.receipt_hash != expected:
            raise ValueError("receiptHash does not match CPU recovery evidence")
        return self


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


_ENVIRONMENT_STABLE_FIELDS = (
    "python",
    "platform",
    "torch",
    "numpy",
    "scipy",
    "pydantic",
    "torchGeometric",
    "cudaRuntime",
)


def local_environment_inventory(
    device_type: Literal["cpu", "cuda"],
) -> LocalEnvironmentInventory:
    """Derive package and device identity from imports in this exact process."""

    device = torch.device(device_type)
    if device_type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA environment cannot be rederived on this host")
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-local-environment-inventory/2.0",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": str(numpy.__version__),
        "scipy": str(scipy.__version__),
        "pydantic": str(pydantic.__version__),
        "torchGeometric": str(torch_geometric.__version__),
        "cudaRuntime": torch.version.cuda,
        "deviceType": device_type,
        "deviceName": (
            torch.cuda.get_device_name(device) if device_type == "cuda" else platform.processor()
        ),
    }
    payload["inventoryHash"] = canonical_sha256(payload)
    return LocalEnvironmentInventory.model_validate(payload)


def validate_local_environment_inventory(
    document: Any,
    *,
    expected_device_type: Literal["cpu", "cuda"] | None = None,
    rederive: bool = False,
) -> LocalEnvironmentInventory:
    inventory = LocalEnvironmentInventory.model_validate(document)
    if expected_device_type is not None and inventory.device_type != expected_device_type:
        raise ValueError("local environment device type differs from expected execution")
    if rederive and inventory != local_environment_inventory(inventory.device_type):
        raise ValueError("local environment inventory differs from fresh process imports")
    return inventory


def local_code_inventory(
    root: Path | None = None,
    *,
    relative_paths: tuple[str, ...] = LOCAL_CODE_INVENTORY_RELATIVE_PATHS,
) -> dict[str, Any]:
    """Hash the fixed, versioned inventory that defines local-run behavior."""

    inventory_root = (
        Path(__file__).resolve().parents[1] if root is None else root.resolve(strict=True)
    )
    files = tuple(
        {
            "relativePath": relative,
            "sha256": _hash_file(inventory_root / relative),
        }
        for relative in relative_paths
    )
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-local-code-inventory/2.0",
        "files": files,
    }
    payload["inventoryHash"] = canonical_sha256(payload)
    return payload


def validate_local_code_inventory(
    document: dict[str, Any],
    *,
    root: Path | None = None,
    relative_paths: tuple[str, ...] = LOCAL_CODE_INVENTORY_RELATIVE_PATHS,
) -> dict[str, Any]:
    """Rehash every actual behavior file named by the exact inventory contract."""

    payload = dict(document)
    observed_hash = payload.pop("inventoryHash", None)
    files = payload.get("files")
    if (
        set(document) != {"schemaVersion", "files", "inventoryHash"}
        or payload.get("schemaVersion") != "socialgraph-fm.core-local-code-inventory/2.0"
        or not isinstance(files, (list, tuple))
        or any(
            not isinstance(item, dict) or set(item) != {"relativePath", "sha256"} for item in files
        )
        or tuple(item.get("relativePath") if isinstance(item, dict) else None for item in files)
        != relative_paths
        or observed_hash != canonical_sha256(payload)
    ):
        raise ValueError("local code inventory contract or semantic hash is invalid")
    actual = local_code_inventory(root, relative_paths=relative_paths)
    if tuple(files) != tuple(actual["files"]):
        raise ValueError("local code inventory differs from an actual behavior file")
    payload["inventoryHash"] = observed_hash
    return payload


def _module_state_hash(state: dict[str, torch.Tensor]) -> str:
    return canonical_sha256(
        {name: canonical_tensor_digest(value) for name, value in sorted(state.items())}
    )


def _edge_index(bundle: Any) -> torch.Tensor:
    selection = derive_training_selection(bundle)
    node_index = {node.id: node.index for node in bundle.nodes}
    pairs: list[tuple[int, int]] = []
    for ordinal in selection.visible_edge_indices:
        edge = bundle.edges[ordinal]
        left, right = node_index[edge.source_id], node_index[edge.target_id]
        pairs.append((left, right))
        if not bundle.directed:
            pairs.append((right, left))
    if not pairs:
        raise ValueError("CPU recovery requires at least one train-visible edge")
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _training_config(payload: dict[str, Any]) -> TrainingConfig:
    normalized = dict(payload)
    fanout = normalized.get("fanout")
    if isinstance(fanout, list):
        normalized["fanout"] = tuple(fanout)
    return TrainingConfig(**normalized)


def recover(request_path: Path) -> LocalRecoveryReceipt:
    request = request_path.resolve(strict=True)
    root = request.parent.resolve(strict=True)
    serialized_request = request.read_bytes()
    payload = __import__("json").loads(serialized_request)
    expected_request_keys = {
        "schemaVersion",
        "checkpointName",
        "bundleName",
        "evaluationArtifactName",
        "receiptName",
        "checkpointSha256",
        "bindings",
        "codeInventory",
        "trainingEnvironmentInventory",
        "interpreterPath",
        "interpreterSha256",
        "trainingConfig",
        "trainingSeed",
        "adapterDomain",
        "graphVersionHash",
        "parentProcessId",
        "requestHash",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_request_keys
        or payload.get("schemaVersion") != "socialgraph-fm.core-local-recovery-request/4.0"
        or serialized_request != (canonical_json(payload) + "\n").encode()
    ):
        raise ValueError("unsupported local recovery request")
    request_hash = payload.pop("requestHash", None)
    if request_hash != canonical_sha256(payload):
        raise ValueError("local recovery request hash mismatch")
    payload["requestHash"] = request_hash
    requested_parent_process_id = payload.get("parentProcessId")
    actual_parent_process_id = os.getppid()
    if (
        type(requested_parent_process_id) is not int
        or requested_parent_process_id != actual_parent_process_id
    ):
        raise ValueError("CPU recovery parent process identity differs from request")
    names = {
        "checkpoint": payload.get("checkpointName"),
        "bundle": payload.get("bundleName"),
        "evaluation": payload.get("evaluationArtifactName"),
        "receipt": payload.get("receiptName"),
    }
    if any(
        not isinstance(name, str) or not name or Path(name).name != name or Path(name).is_absolute()
        for name in names.values()
    ):
        raise ValueError("local recovery paths must be simple confined basenames")
    checkpoint_path = root / str(names["checkpoint"])
    bundle_path = root / str(names["bundle"])
    evaluation_path = root / str(names["evaluation"])
    receipt_path = root / str(names["receipt"])
    if _hash_file(checkpoint_path) != payload.get("checkpointSha256"):
        raise ValueError("CPU recovery checkpoint bytes changed")
    bindings_payload = payload.get("bindings")
    if not isinstance(bindings_payload, dict) or set(bindings_payload) != {
        "config_hash",
        "data_hash",
        "code_hash",
        "environment_hash",
    }:
        raise ValueError("CPU recovery bindings are missing")
    bindings = CheckpointBindings(**bindings_payload)
    code_inventory = payload.get("codeInventory")
    if not isinstance(code_inventory, dict):
        raise ValueError("CPU recovery code inventory is missing")
    validated_code_inventory = validate_local_code_inventory(code_inventory)
    if validated_code_inventory["inventoryHash"] != bindings.code_hash:
        raise ValueError("CPU recovery code inventory differs from checkpoint binding")
    training_environment = validate_local_environment_inventory(
        payload.get("trainingEnvironmentInventory")
    )
    if training_environment.inventory_hash != bindings.environment_hash:
        raise ValueError("CPU recovery training environment differs from checkpoint binding")
    recovery_environment = local_environment_inventory("cpu")
    training_environment_document = training_environment.model_dump(mode="python", by_alias=True)
    recovery_environment_document = recovery_environment.model_dump(mode="python", by_alias=True)
    if any(
        training_environment_document[field] != recovery_environment_document[field]
        for field in _ENVIRONMENT_STABLE_FIELDS
    ):
        raise ValueError("CPU recovery environment inventory differs from training imports")
    interpreter = Path(sys.executable).resolve(strict=True)
    interpreter_sha256 = _hash_file(interpreter)
    if (
        payload.get("interpreterPath") != str(interpreter)
        or payload.get("interpreterSha256") != interpreter_sha256
    ):
        raise ValueError("CPU recovery interpreter identity differs from request")
    checkpoint = load_checkpoint(checkpoint_path, expected_bindings=bindings)
    if checkpoint.get("status") != "training" or checkpoint.get("promotable") is not False:
        raise ValueError("CPU recovery requires a non-promotable training checkpoint")
    state = checkpoint["trainer"]
    config_payload = payload.get("trainingConfig")
    if not isinstance(config_payload, dict):
        raise ValueError("CPU recovery training configuration is missing")
    config = _training_config(config_payload)
    if state.get("config") != config.to_dict() or state.get("trainingSeed") != payload.get(
        "trainingSeed"
    ):
        raise ValueError("CPU recovery trainer configuration or seed differs")
    domain = payload.get("adapterDomain")
    if (
        not isinstance(domain, str)
        or set(state.get("adapters", {})) != {domain}
        or set(state.get("adapterSchemas", {})) != {domain}
    ):
        raise ValueError("CPU recovery adapter inventory differs")
    bundle = load_core_graph_bundle_json(bundle_path.read_bytes())
    if bundle.graph_version_hash != payload.get("graphVersionHash"):
        raise ValueError("CPU recovery bundle identity differs")
    schema = AdapterSchema.model_validate(state["adapterSchemas"][domain], strict=False)
    adapter = BundleInputAdapter(bundle, mode="training", schema=schema)
    edge_index = _edge_index(bundle)
    policy = ExecutionPolicy(
        full_batch_edge_threshold=config.full_batch_edge_threshold,
        node_batch_size=config.node_batch_size,
        edge_batch_size=config.edge_batch_size,
        fanout=config.fanout,
    )
    prepared = PreparedGraph.from_edge_index(
        num_nodes=len(bundle.nodes), edge_index=edge_index, directed=bundle.directed
    )
    model = CoreGFM(node_classes=2)
    trainer = CoreTrainer(
        model,
        {
            domain: TrainingGraph.from_bundle(
                adapter=adapter,
                graph=prepared,
                execution_policy=policy,
            )
        },
        config=config,
        seed=int(payload["trainingSeed"]),
    )
    trainer.model.load_state_dict(state["model"], strict=True)
    recovered_adapter = trainer.graphs[domain].adapter
    if recovered_adapter is None:
        raise ValueError("CPU recovery adapter is unavailable")
    recovered_adapter.load_state_dict(state["adapters"][domain], strict=True)
    if next(trainer.model.parameters()).device.type != "cpu" or any(
        value.device.type != "cpu"
        for value in (
            *trainer.model.state_dict().values(),
            *recovered_adapter.state_dict().values(),
        )
    ):
        raise ValueError("fresh local recovery did not remain CPU-only")
    model_hash = _module_state_hash(trainer.model.state_dict())
    adapter_state_hash = _module_state_hash(recovered_adapter.state_dict())
    evaluation_payload = {
        "schemaVersion": "socialgraph-fm.core-local-cpu-evaluation-state/1.0",
        "requestHash": request_hash,
        "model": trainer.model.state_dict(),
        "adapterSchema": schema.model_dump(mode="python", by_alias=True),
        "adapter": recovered_adapter.state_dict(),
    }
    with evaluation_path.open("xb") as stream:
        torch.save(evaluation_payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    trainer_state_hash = _state_hash(state)
    receipt_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-local-recovery-receipt/4.0",
        "requestHash": request_hash,
        "recoveryProcessId": os.getpid(),
        "recoveryParentProcessId": actual_parent_process_id,
        "recoveryDevice": "cpu",
        "recoveryInterpreterPath": str(interpreter),
        "recoveryInterpreterSha256": interpreter_sha256,
        "checkpointSha256": payload["checkpointSha256"],
        "configHash": bindings.config_hash,
        "dataHash": bindings.data_hash,
        "codeHash": bindings.code_hash,
        "environmentHash": bindings.environment_hash,
        "recoveryEnvironmentInventory": recovery_environment_document,
        "recoveryEnvironmentHash": recovery_environment.inventory_hash,
        "trainerStateHash": trainer_state_hash,
        "compositeStateHash": _composite_hash(state),
        "recoveryStateHash": canonical_sha256(
            {
                "trainerStateHash": trainer_state_hash,
                "requestHash": request_hash,
                "recoveryDevice": "cpu",
                "modelStateHash": model_hash,
                "adapterStateHash": adapter_state_hash,
            }
        ),
        "modelStateHash": model_hash,
        "adapterSchemaHash": schema.adapter_schema_hash,
        "adapterStateHash": adapter_state_hash,
        "evaluationArtifactSha256": _hash_file(evaluation_path),
    }
    receipt_payload["receiptHash"] = canonical_sha256(receipt_payload)
    receipt = LocalRecoveryReceipt.model_validate(receipt_payload)
    serialized = (canonical_json(receipt) + "\n").encode()
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args()
    recover(arguments.request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LOCAL_CODE_INVENTORY_RELATIVE_PATHS",
    "LocalEnvironmentInventory",
    "LocalRecoveryReceipt",
    "local_code_inventory",
    "local_environment_inventory",
    "recover",
    "validate_local_code_inventory",
    "validate_local_environment_inventory",
]
