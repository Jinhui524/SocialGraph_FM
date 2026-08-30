"""Strict fresh-resume verification for one fold-scoped GFM checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.tensor_digest import canonical_tensor_digest

from .adapters import BundleInputAdapter, derive_training_selection
from .bundle import CoreGraphBundle
from .config import TrainingConfig
from .model import CoreGFM
from .trainer import CoreTrainer, TrainingGraph
from .training_data import ExecutionPolicy, PreparedGraph


_REQUIRED_TRAINER_FIELDS = frozenset(
    {
        "adapterSchemas",
        "adapters",
        "batchSources",
        "config",
        "domainPrototypes",
        "domainSampler",
        "fitState",
        "gradientAccumulationCursor",
        "model",
        "objectiveGenerator",
        "optimizer",
        "optimizerStep",
        "rng",
        "scaler",
        "scheduler",
        "trainingSeed",
    }
)


@dataclass(frozen=True)
class VerifiedFoldRecovery:
    model: CoreGFM
    adapter: BundleInputAdapter
    optimizer_step: int
    training_seed: int
    composite_state_hash: str
    recovery_state_hash: str


@dataclass(frozen=True)
class VerifiedRunRecoveryInventory:
    """Hash identities observed from one complete cell-level trainer state.

    This verifier deliberately does not claim graph-level resume equivalence.  That
    stronger property is checked per fold by :func:`verify_fold_recovery_state`.
    It does ensure the cell checkpoint contains every recoverable trainer component
    and derives both identities from the reopened object inventory rather than from
    caller-provided digest fields.
    """

    optimizer_step: int
    training_seed: int
    composite_state_hash: str
    recovery_state_hash: str


def _edge_index(bundle: CoreGraphBundle) -> Tensor:
    selection = derive_training_selection(bundle)
    node_index = {item.id: item.index for item in bundle.nodes}
    pairs: list[tuple[int, int]] = []
    for ordinal in selection.visible_edge_indices:
        edge = bundle.edges[ordinal]
        left = node_index[edge.source_id]
        right = node_index[edge.target_id]
        pairs.append((left, right))
        if not bundle.directed:
            pairs.append((right, left))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _normalized(value: Any) -> Any:
    if isinstance(value, Tensor):
        return {"kind": "tensor", "digest": canonical_tensor_digest(value)}
    if isinstance(value, Mapping):
        return {
            f"{type(key).__name__}:{key}": _normalized(item)
            for key, item in sorted(
                value.items(), key=lambda pair: (type(pair[0]).__name__, str(pair[0]))
            )
        }
    if isinstance(value, (tuple, list)):
        return [_normalized(item) for item in value]
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise ValueError(f"unsupported recovery-state value: {type(value).__name__}")


def _state_hash(value: Mapping[str, Any]) -> str:
    return canonical_sha256(_normalized(value))


def _composite_hash(state: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "model": _normalized(state["model"]),
            "adapterSchemas": _normalized(state["adapterSchemas"]),
            "adapters": _normalized(state["adapters"]),
        }
    )


def _recovery_device(objective_state: Tensor) -> torch.device:
    """Select the generator implementation that can consume the persisted RNG bytes."""

    if objective_state.dtype != torch.uint8 or objective_state.ndim != 1:
        raise ValueError("fold recovery objective RNG state must be a rank-one byte tensor")
    candidates: list[torch.device] = []
    cpu = torch.device("cpu")
    if torch.Generator(device=cpu).get_state().numel() == objective_state.numel():
        candidates.append(cpu)
    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        if torch.Generator(device=cuda).get_state().numel() == objective_state.numel():
            candidates.append(cuda)
    if len(candidates) != 1:
        raise ValueError(
            "fold recovery cannot uniquely identify the persisted objective RNG device"
        )
    return candidates[0]


def verify_run_recovery_inventory(
    state: Mapping[str, Any],
    *,
    config: TrainingConfig,
    expected_seed: int,
    expected_cell_id: str,
) -> VerifiedRunRecoveryInventory:
    """Verify and hash an exact cell-level trainer recovery inventory."""

    expected_fields = _REQUIRED_TRAINER_FIELDS | frozenset({"experimentCellId"})
    if not isinstance(state, Mapping) or set(state) != expected_fields:
        raise ValueError("cell checkpoint lacks the complete trainer recovery inventory")
    if state.get("experimentCellId") != expected_cell_id:
        raise ValueError("cell checkpoint recovery identity differs from the experiment cell")
    if state.get("config") != config.to_dict():
        raise ValueError("cell recovery configuration differs from the formal run")
    if state.get("trainingSeed") != expected_seed:
        raise ValueError("cell checkpoint training seed differs from the experiment cell")
    step = state.get("optimizerStep")
    if type(step) is not int or not 1 <= step <= config.max_steps:
        raise ValueError("cell recovery optimizer step is outside the formal configuration")
    if state.get("gradientAccumulationCursor") != 0:
        raise ValueError("cell recovery checkpoint is not at an optimizer boundary")
    scheduler = state.get("scheduler")
    if not isinstance(scheduler, Mapping) or scheduler.get("last_epoch") != step:
        raise ValueError("cell recovery scheduler cursor differs from optimizerStep")
    for name in (
        "model",
        "adapterSchemas",
        "adapters",
        "optimizer",
        "scaler",
        "rng",
        "domainSampler",
        "batchSources",
        "fitState",
        "domainPrototypes",
    ):
        if not isinstance(state.get(name), Mapping):
            raise ValueError(f"cell recovery {name} state is unavailable")
    if not isinstance(state.get("objectiveGenerator"), Tensor):
        raise ValueError("cell recovery objective RNG state is unavailable")
    if not state["adapterSchemas"] or set(state["adapterSchemas"]) != set(state["adapters"]):
        raise ValueError("cell recovery adapter inventory is incomplete")

    trainer_state = {name: value for name, value in state.items() if name != "experimentCellId"}
    return VerifiedRunRecoveryInventory(
        optimizer_step=step,
        training_seed=expected_seed,
        composite_state_hash=_composite_hash(trainer_state),
        recovery_state_hash=canonical_sha256(
            {
                "trainerStateHash": _state_hash(trainer_state),
                "experimentCellId": expected_cell_id,
            }
        ),
    )


def verify_fold_recovery_state(
    state: Mapping[str, Any],
    *,
    bundle: CoreGraphBundle,
    adapter_domain: str,
    config: TrainingConfig,
    expected_seed: int,
    expected_cell_id: str | None = None,
    expected_fold_id: str | None = None,
) -> VerifiedFoldRecovery:
    """Fresh-load every trainer component and return its immutable identities."""

    identity_fields = (
        frozenset({"experimentCellId", "evaluationFoldId"})
        if expected_cell_id is not None and expected_fold_id is not None
        else frozenset()
    )
    expected_fields = _REQUIRED_TRAINER_FIELDS | identity_fields
    if not isinstance(state, Mapping) or not expected_fields <= set(state):
        raise ValueError("fold checkpoint lacks the complete trainer recovery inventory")
    if set(state) != expected_fields:
        raise ValueError("fold checkpoint trainer recovery inventory has unknown fields")
    if bool(expected_cell_id is None) != bool(expected_fold_id is None):
        raise ValueError("fold recovery cell and fold identities must be supplied together")
    if identity_fields and (
        state.get("experimentCellId") != expected_cell_id
        or state.get("evaluationFoldId") != expected_fold_id
    ):
        raise ValueError("fold checkpoint recovery identity differs from the experiment cell")
    if state.get("config") != config.to_dict():
        raise ValueError("fold recovery configuration differs from the formal run")
    if state.get("trainingSeed") != expected_seed:
        raise ValueError("fold checkpoint training seed differs from the experiment cell")
    step = state.get("optimizerStep")
    if type(step) is not int or not 1 <= step <= config.max_steps:
        raise ValueError("fold recovery optimizer step is outside the formal configuration")
    if state.get("gradientAccumulationCursor") != 0:
        raise ValueError("fold recovery checkpoint is not at an optimizer boundary")
    if set(state.get("adapterSchemas", {})) != {adapter_domain} or set(
        state.get("adapters", {})
    ) != {adapter_domain}:
        raise ValueError("fold recovery adapter inventory must contain exactly its target fold")
    scheduler = state.get("scheduler")
    if not isinstance(scheduler, Mapping) or scheduler.get("last_epoch") != step:
        raise ValueError("fold recovery scheduler cursor differs from optimizerStep")
    for name in ("optimizer", "scaler", "rng", "domainSampler", "batchSources"):
        if not isinstance(state.get(name), Mapping):
            raise ValueError(f"fold recovery {name} state is unavailable")
    objective_state = state.get("objectiveGenerator")
    if not isinstance(objective_state, Tensor):
        raise ValueError("fold recovery objective RNG state is unavailable")

    device = _recovery_device(objective_state)

    adapter = BundleInputAdapter(bundle, mode="training")
    edge_index = _edge_index(bundle)
    prepared = PreparedGraph.from_edge_index(
        num_nodes=len(bundle.nodes),
        edge_index=edge_index,
        directed=bundle.directed,
    )
    execution_policy = ExecutionPolicy(
        full_batch_edge_threshold=config.full_batch_edge_threshold,
        node_batch_size=config.node_batch_size,
        edge_batch_size=config.edge_batch_size,
        fanout=config.fanout,
    )
    if execution_policy.mode(edge_count=edge_index.shape[1]) == "full-batch":
        prepared = prepared.to(device)
    model = CoreGFM(node_classes=2).to(device)
    trainer = CoreTrainer(
        model,
        {
            adapter_domain: TrainingGraph.from_bundle(
                adapter=adapter,
                graph=prepared,
                execution_policy=execution_policy,
            )
        },
        config=config,
        seed=expected_seed,
    )
    trainer_state = {name: value for name, value in state.items() if name not in identity_fields}
    try:
        trainer.load_state_dict(trainer_state)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("fold trainer state cannot be recovered exactly") from error
    reopened = trainer.state_dict()
    if _state_hash(reopened) != _state_hash(trainer_state):
        raise ValueError("fresh fold trainer state differs after strict recovery")
    return VerifiedFoldRecovery(
        model=trainer.model,
        adapter=trainer.graphs[adapter_domain].adapter,  # type: ignore[arg-type]
        optimizer_step=step,
        training_seed=expected_seed,
        composite_state_hash=_composite_hash(reopened),
        recovery_state_hash=canonical_sha256(
            {
                "trainerStateHash": _state_hash(reopened),
                "experimentCellId": expected_cell_id,
                "evaluationFoldId": expected_fold_id,
            }
        ),
    )


__all__ = [
    "VerifiedFoldRecovery",
    "VerifiedRunRecoveryInventory",
    "verify_fold_recovery_state",
    "verify_run_recovery_inventory",
]
