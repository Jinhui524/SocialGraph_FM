"""Deterministic validation-only selector for the deployable governance candidate.

The selector deliberately has no test-score or transfer-decision input.  Its
observations are produced by acceptance only after a raw checkpoint has been
strict-loaded and its validation evidence has been reproduced.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Literal

from socialgraph_gfm.canonical import canonical_sha256

from .experiments import ExperimentProtocol, build_experiment_matrix


_HASH = re.compile(r"^[0-9a-f]{64}$")
_TASKS = (
    "github.relation-completion",
    "penn94.community-resilience",
    "tolokers.risk",
    "wiki-rfa.vote-sign",
)
_METHODS = ("multi-graph-shared-gfm", "domain-aligned-gfm")
_SELECTOR_VERSION: Literal["socialgraph-fm.core-governance-winner/1.0"] = (
    "socialgraph-fm.core-governance-winner/1.0"
)


@dataclass(frozen=True)
class GovernanceValidationObservation:
    """One live-reproduced full-budget validation observation."""

    cell_id: str
    task_id: str
    method_id: str
    seed: int
    label_budget: str
    validation_metric: float
    record_hash: str
    best_checkpoint_sha256: str
    head_report_hash: str
    validation_partition_hash: str
    validation_protocol_hash: str
    validation_data_hash: str
    validation_callback_hash: str


@dataclass(frozen=True)
class GovernanceMethodSummary:
    method_id: str
    task_wins: int
    paired_wins: int
    task_mean_metrics: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class GovernanceWinnerSelection:
    selector_version: Literal["socialgraph-fm.core-governance-winner/1.0"]
    protocol_hash: str
    task_ids: tuple[str, ...]
    eligible_methods: tuple[str, ...]
    label_budget: Literal["full"]
    seeds: tuple[int, ...]
    source_runs: tuple[GovernanceValidationObservation, ...]
    method_summaries: tuple[GovernanceMethodSummary, ...]
    selected_method_id: str
    selected_seed: int
    selected_runs: tuple[GovernanceValidationObservation, ...]
    selection_hash: str


def _validate_observation(item: GovernanceValidationObservation) -> None:
    hashes = (
        item.cell_id,
        item.record_hash,
        item.best_checkpoint_sha256,
        item.head_report_hash,
        item.validation_partition_hash,
        item.validation_protocol_hash,
        item.validation_data_hash,
        item.validation_callback_hash,
    )
    if any(_HASH.fullmatch(value) is None for value in hashes):
        raise ValueError("governance validation inventory contains a malformed hash")
    if not math.isfinite(item.validation_metric):
        raise ValueError("governance validation inventory contains a non-finite metric")


def _selection_payload(
    *,
    protocol: ExperimentProtocol,
    source_runs: tuple[GovernanceValidationObservation, ...],
    summaries: tuple[GovernanceMethodSummary, ...],
    selected_method_id: str,
    selected_seed: int,
    selected_runs: tuple[GovernanceValidationObservation, ...],
) -> dict[str, object]:
    return {
        "selectorVersion": _SELECTOR_VERSION,
        "protocolHash": protocol.protocol_hash,
        "taskIds": list(_TASKS),
        "eligibleMethods": list(_METHODS),
        "labelBudget": "full",
        "seeds": list(protocol.seeds),
        "sourceRuns": [asdict(item) for item in source_runs],
        "methodSummaries": [asdict(item) for item in summaries],
        "selectedMethodId": selected_method_id,
        "selectedSeed": selected_seed,
        "selectedRuns": [asdict(item) for item in selected_runs],
    }


def derive_governance_winner(
    protocol: ExperimentProtocol,
    observations: tuple[GovernanceValidationObservation, ...],
) -> GovernanceWinnerSelection:
    """Select one method and seed using only live validation observations."""

    if protocol != ExperimentProtocol.fixed():
        raise ValueError("governance winner requires the fixed formal protocol")
    if any(type(item) is not GovernanceValidationObservation for item in observations):
        raise ValueError("governance validation inventory contains an unsealed observation")
    for item in observations:
        _validate_observation(item)

    expected_cells = {
        (cell.task_id, cell.method_id, cell.seed): cell.cell_id
        for cell in build_experiment_matrix(protocol)
        if cell.task_id in _TASKS and cell.method_id in _METHODS and cell.label_budget == "full"
    }
    by_key = {(item.task_id, item.method_id, item.seed): item for item in observations}
    if (
        len(observations) != len(expected_cells)
        or len(by_key) != len(observations)
        or set(by_key) != set(expected_cells)
        or any(
            item.label_budget != "full" or item.cell_id != expected_cells[key]
            for key, item in by_key.items()
        )
        or any(
            len({getattr(item, field) for item in observations}) != len(observations)
            for field in ("record_hash", "best_checkpoint_sha256", "head_report_hash")
        )
    ):
        raise ValueError("governance validation inventory is not the exact fixed 40-run matrix")

    ordered = tuple(
        by_key[(task_id, method_id, seed)]
        for task_id in _TASKS
        for method_id in _METHODS
        for seed in protocol.seeds
    )
    task_wins = {method_id: 0 for method_id in _METHODS}
    paired_wins = {method_id: 0 for method_id in _METHODS}
    task_means: dict[str, list[tuple[str, float]]] = {method_id: [] for method_id in _METHODS}
    for task_id in _TASKS:
        means = {
            method_id: math.fsum(
                by_key[(task_id, method_id, seed)].validation_metric for seed in protocol.seeds
            )
            / len(protocol.seeds)
            for method_id in _METHODS
        }
        for method_id in _METHODS:
            task_means[method_id].append((task_id, means[method_id]))
        if means[_METHODS[1]] > means[_METHODS[0]]:
            task_wins[_METHODS[1]] += 1
        elif means[_METHODS[0]] > means[_METHODS[1]]:
            task_wins[_METHODS[0]] += 1
        for seed in protocol.seeds:
            shared = by_key[(task_id, _METHODS[0], seed)].validation_metric
            aligned = by_key[(task_id, _METHODS[1], seed)].validation_metric
            if aligned > shared:
                paired_wins[_METHODS[1]] += 1
            elif shared > aligned:
                paired_wins[_METHODS[0]] += 1

    selected_method = _METHODS[0]
    if task_wins[_METHODS[1]] > task_wins[_METHODS[0]] or (
        task_wins[_METHODS[1]] == task_wins[_METHODS[0]]
        and paired_wins[_METHODS[1]] > paired_wins[_METHODS[0]]
    ):
        selected_method = _METHODS[1]

    rank_sum = {seed: 0 for seed in protocol.seeds}
    first_places = {seed: 0 for seed in protocol.seeds}
    seed_order = {seed: index for index, seed in enumerate(protocol.seeds)}
    for task_id in _TASKS:
        ranked = sorted(
            protocol.seeds,
            key=lambda seed: (
                -by_key[(task_id, selected_method, seed)].validation_metric,
                seed_order[seed],
            ),
        )
        for rank, seed in enumerate(ranked, start=1):
            rank_sum[seed] += rank
            if rank == 1:
                first_places[seed] += 1
    selected_seed = min(
        protocol.seeds,
        key=lambda seed: (rank_sum[seed], -first_places[seed], seed_order[seed]),
    )
    selected_runs = tuple(by_key[(task_id, selected_method, selected_seed)] for task_id in _TASKS)
    summaries = tuple(
        GovernanceMethodSummary(
            method_id=method_id,
            task_wins=task_wins[method_id],
            paired_wins=paired_wins[method_id],
            task_mean_metrics=tuple(task_means[method_id]),
        )
        for method_id in _METHODS
    )
    payload = _selection_payload(
        protocol=protocol,
        source_runs=ordered,
        summaries=summaries,
        selected_method_id=selected_method,
        selected_seed=selected_seed,
        selected_runs=selected_runs,
    )
    return GovernanceWinnerSelection(
        selector_version=_SELECTOR_VERSION,
        protocol_hash=protocol.protocol_hash,
        task_ids=_TASKS,
        eligible_methods=_METHODS,
        label_budget="full",
        seeds=protocol.seeds,
        source_runs=ordered,
        method_summaries=summaries,
        selected_method_id=selected_method,
        selected_seed=selected_seed,
        selected_runs=selected_runs,
        selection_hash=canonical_sha256(payload),
    )


__all__ = [
    "GovernanceMethodSummary",
    "GovernanceValidationObservation",
    "GovernanceWinnerSelection",
    "derive_governance_winner",
]
